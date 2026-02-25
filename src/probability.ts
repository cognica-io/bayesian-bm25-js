//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

// Bayesian probability transforms for BM25 scores.
//
// Implements the sigmoid likelihood + composite prior + Bayesian posterior
// framework from "Bayesian BM25" for converting raw BM25 retrieval scores
// into calibrated probabilities.

export const EPSILON = 1e-10;

function clampScalar(p: number): number {
  return Math.max(EPSILON, Math.min(1.0 - EPSILON, p));
}

export function clampProbability(p: number): number;
export function clampProbability(p: number[]): number[];
export function clampProbability(p: number | number[]): number | number[] {
  if (Array.isArray(p)) {
    return p.map(clampScalar);
  }
  return clampScalar(p);
}

function sigmoidScalar(x: number): number {
  if (x >= 0) {
    return 1.0 / (1.0 + Math.exp(-x));
  }
  const expX = Math.exp(x);
  return expX / (1.0 + expX);
}

export function sigmoid(x: number): number;
export function sigmoid(x: number[]): number[];
export function sigmoid(x: number | number[]): number | number[] {
  if (Array.isArray(x)) {
    return x.map(sigmoidScalar);
  }
  return sigmoidScalar(x);
}

function logitScalar(p: number): number {
  const clamped = clampScalar(p);
  return Math.log(clamped / (1.0 - clamped));
}

export function logit(p: number): number;
export function logit(p: number[]): number[];
export function logit(p: number | number[]): number | number[] {
  if (Array.isArray(p)) {
    return p.map(logitScalar);
  }
  return logitScalar(p);
}

export type TrainingMode = "balanced" | "prior_aware" | "prior_free";

const VALID_MODES: ReadonlySet<string> = new Set([
  "balanced",
  "prior_aware",
  "prior_free",
]);

export interface FitOptions {
  learningRate?: number;
  maxIterations?: number;
  tolerance?: number;
  mode?: TrainingMode;
  tfs?: number[];
  docLenRatios?: number[];
}

export interface UpdateOptions {
  learningRate?: number;
  momentum?: number;
  decayTau?: number;
  maxGradNorm?: number;
  avgDecay?: number;
  mode?: TrainingMode;
  tf?: number | number[];
  docLenRatio?: number | number[];
}

// Transforms raw BM25 scores into calibrated probabilities.
//
// Parameters:
//   alpha:     Steepness of the sigmoid likelihood function.
//   beta:      Midpoint (shift) of the sigmoid likelihood function.
//   baseRate:  Corpus-level base rate of relevance, in (0, 1).
//              When set, the posterior is computed in log-odds space as
//              sigmoid(logit(L) + logit(baseRate) + logit(prior)).
//              null (default) disables base rate correction.
//              baseRate=0.5 is neutral (logit(0.5) = 0).
export class BayesianProbabilityTransform {
  public alpha: number;
  public beta: number;
  public readonly baseRate: number | null;

  private _logitBaseRate: number | null;
  private _trainingMode: TrainingMode = "balanced";
  private _nUpdates: number = 0;
  private _gradAlphaEMA: number = 0.0;
  private _gradBetaEMA: number = 0.0;
  private _alphaAvg: number;
  private _betaAvg: number;

  constructor(
    alpha: number = 1.0,
    beta: number = 0.0,
    baseRate: number | null = null,
  ) {
    if (baseRate !== null) {
      if (baseRate <= 0.0 || baseRate >= 1.0) {
        throw new Error(`baseRate must be in (0, 1), got ${baseRate}`);
      }
    }
    this.alpha = alpha;
    this.beta = beta;
    this.baseRate = baseRate;
    this._logitBaseRate =
      baseRate !== null ? (logitScalar(baseRate) as number) : null;
    this._alphaAvg = alpha;
    this._betaAvg = beta;
  }

  get averagedAlpha(): number {
    return this._alphaAvg;
  }

  get averagedBeta(): number {
    return this._betaAvg;
  }

  get nUpdates(): number {
    return this._nUpdates;
  }

  get gradAlphaEMA(): number {
    return this._gradAlphaEMA;
  }

  // Sigmoid likelihood: sigma(alpha * (score - beta))  (Eq. 20)
  likelihood(score: number): number;
  likelihood(score: number[]): number[];
  likelihood(score: number | number[]): number | number[] {
    if (Array.isArray(score)) {
      return score.map((s) => sigmoidScalar(this.alpha * (s - this.beta)));
    }
    return sigmoidScalar(this.alpha * (score - this.beta));
  }

  // Term-frequency prior: 0.2 + 0.7 * min(1, tf / 10)  (Eq. 25)
  static tfPrior(tf: number): number;
  static tfPrior(tf: number[]): number[];
  static tfPrior(tf: number | number[]): number | number[] {
    if (Array.isArray(tf)) {
      return tf.map((t) => 0.2 + 0.7 * Math.min(1.0, t / 10.0));
    }
    return 0.2 + 0.7 * Math.min(1.0, tf / 10.0);
  }

  // Document-length normalisation prior (Eq. 26).
  //
  // P_norm = 0.3 + 0.6 * (1 - min(1, |doc_len_ratio - 0.5| * 2))
  //
  // where doc_len_ratio = doc_len / avgdl (values near 1.0 are average).
  // The prior peaks when doc_len_ratio ~ 0.5 and decreases for extreme lengths.
  static normPrior(docLenRatio: number): number;
  static normPrior(docLenRatio: number[]): number[];
  static normPrior(docLenRatio: number | number[]): number | number[] {
    if (Array.isArray(docLenRatio)) {
      return docLenRatio.map(
        (r) => 0.3 + 0.6 * (1.0 - Math.min(1.0, Math.abs(r - 0.5) * 2.0)),
      );
    }
    return (
      0.3 + 0.6 * (1.0 - Math.min(1.0, Math.abs(docLenRatio - 0.5) * 2.0))
    );
  }

  // Composite prior: clamp(0.7 * P_tf + 0.3 * P_norm, 0.1, 0.9)  (Eq. 27)
  static compositePrior(tf: number, docLenRatio: number): number;
  static compositePrior(tf: number[], docLenRatio: number[]): number[];
  static compositePrior(
    tf: number | number[],
    docLenRatio: number | number[],
  ): number | number[] {
    if (Array.isArray(tf) && Array.isArray(docLenRatio)) {
      const pTf = BayesianProbabilityTransform.tfPrior(tf);
      const pNorm = BayesianProbabilityTransform.normPrior(docLenRatio);
      return pTf.map((pt, i) =>
        Math.max(0.1, Math.min(0.9, 0.7 * pt + 0.3 * pNorm[i]!)),
      );
    }
    const pTf = BayesianProbabilityTransform.tfPrior(tf as number);
    const pNorm = BayesianProbabilityTransform.normPrior(
      docLenRatio as number,
    );
    return Math.max(0.1, Math.min(0.9, 0.7 * pTf + 0.3 * pNorm));
  }

  // Bayesian posterior via two-step Bayes update (Eq. 22, Remark 4.4.5).
  //
  // Without baseRate:
  //   P = L*p / (L*p + (1-L)*(1-p))
  //
  // With baseRate (two-step, avoids expensive logit/sigmoid):
  //   Step 1: p1 = L*p / (L*p + (1-L)*(1-p))
  //   Step 2: P  = p1*br / (p1*br + (1-p1)*(1-br))
  //
  // Equivalent to sigmoid(logit(L) + logit(prior) + logit(baseRate)).
  static posterior(
    likelihoodVal: number,
    prior: number,
    baseRate?: number | null,
  ): number;
  static posterior(
    likelihoodVal: number[],
    prior: number[],
    baseRate?: number | null,
  ): number[];
  static posterior(
    likelihoodVal: number | number[],
    prior: number | number[],
    baseRate: number | null = null,
  ): number | number[] {
    if (Array.isArray(likelihoodVal) && Array.isArray(prior)) {
      return likelihoodVal.map((lv, i) => {
        const p = prior[i]!;
        const numerator = lv * p;
        const denominator = numerator + (1.0 - lv) * (1.0 - p);
        let result = clampScalar(numerator / denominator);
        if (baseRate !== null && baseRate !== undefined) {
          const numeratorBR = result * baseRate;
          const denominatorBR =
            numeratorBR + (1.0 - result) * (1.0 - baseRate);
          result = clampScalar(numeratorBR / denominatorBR);
        }
        return result;
      });
    }
    const lv = likelihoodVal as number;
    const p = prior as number;
    const numerator = lv * p;
    const denominator = numerator + (1.0 - lv) * (1.0 - p);
    let result = clampScalar(numerator / denominator);
    if (baseRate !== null && baseRate !== undefined) {
      const numeratorBR = result * baseRate;
      const denominatorBR = numeratorBR + (1.0 - result) * (1.0 - baseRate);
      result = clampScalar(numeratorBR / denominatorBR);
    }
    return result;
  }

  // Full pipeline: BM25 score -> calibrated probability.
  //
  // Computes likelihood from the score, composite prior from tf and
  // docLenRatio, then applies the Bayesian posterior formula.
  // When baseRate is set, the posterior includes baseRate via a
  // two-step Bayes update (Remark 4.4.5).
  //
  // In prior_free mode (C3), uses prior=0.5 so the posterior equals the
  // likelihood, ignoring the composite prior at inference time.
  scoreToProbability(
    score: number,
    tf: number,
    docLenRatio: number,
  ): number;
  scoreToProbability(
    score: number[],
    tf: number[],
    docLenRatio: number[],
  ): number[];
  scoreToProbability(
    score: number | number[],
    tf: number | number[],
    docLenRatio: number | number[],
  ): number | number[] {
    if (
      Array.isArray(score) &&
      Array.isArray(tf) &&
      Array.isArray(docLenRatio)
    ) {
      const lVal = this.likelihood(score);
      const prior =
        this._trainingMode === "prior_free"
          ? score.map(() => 0.5)
          : BayesianProbabilityTransform.compositePrior(tf, docLenRatio);

      return BayesianProbabilityTransform.posterior(
        lVal,
        prior,
        this.baseRate,
      );
    }

    const lVal = this.likelihood(score as number) as number;
    const prior =
      this._trainingMode === "prior_free"
        ? 0.5
        : (BayesianProbabilityTransform.compositePrior(
            tf as number,
            docLenRatio as number,
          ) as number);

    return BayesianProbabilityTransform.posterior(lVal, prior, this.baseRate);
  }

  // Compute the Bayesian WAND upper bound for safe document pruning (Theorem 6.1.2).
  //
  // Given a standard BM25 upper bound per term, computes the tightest
  // safe Bayesian probability upper bound by assuming the maximum
  // possible prior (pMax from Theorem 4.2.4).
  //
  // Any document's actual Bayesian probability is guaranteed to be
  // at most this value, making it safe for WAND-style pruning.
  wandUpperBound(bm25UpperBound: number): number;
  wandUpperBound(bm25UpperBound: number[]): number[];
  wandUpperBound(bm25UpperBound: number | number[]): number | number[] {
    const pMax = 0.9;
    if (Array.isArray(bm25UpperBound)) {
      const lMax = this.likelihood(bm25UpperBound);
      return BayesianProbabilityTransform.posterior(
        lMax,
        lMax.map(() => pMax),
        this.baseRate,
      );
    }
    const lMax = this.likelihood(bm25UpperBound) as number;
    return BayesianProbabilityTransform.posterior(lMax, pMax, this.baseRate);
  }

  // Learn alpha and beta via gradient descent (Algorithm 8.3.1).
  //
  // Three training modes are supported (C1/C2/C3 conditions):
  //
  // - "balanced" (C1, default): trains on the sigmoid likelihood
  //   pred = sigmoid(alpha*(s-beta)).
  // - "prior_aware" (C2): trains on the full Bayesian posterior
  //   pred = L*p / (L*p + (1-L)*(1-p)) where L is the sigmoid
  //   likelihood and p is the composite prior.  Requires tfs
  //   and docLenRatios.
  // - "prior_free" (C3): same training as balanced, but at
  //   inference time scoreToProbability uses prior=0.5
  //   (posterior = likelihood).
  fit(scores: number[], labels: number[], options: FitOptions = {}): void {
    const {
      learningRate = 0.01,
      maxIterations = 1000,
      tolerance = 1e-6,
      mode = "balanced",
      tfs,
      docLenRatios,
    } = options;

    if (!VALID_MODES.has(mode)) {
      throw new Error(
        `mode must be one of "balanced", "prior_aware", "prior_free", got "${mode}"`,
      );
    }
    if (mode === "prior_aware") {
      if (tfs === undefined || docLenRatios === undefined) {
        throw new Error(
          "tfs and docLenRatios are required when mode='prior_aware'",
        );
      }
    }

    let priors: number[] | null = null;
    if (mode === "prior_aware") {
      priors = BayesianProbabilityTransform.compositePrior(
        tfs!,
        docLenRatios!,
      );
    }

    let alpha = this.alpha;
    let beta = this.beta;

    for (let iter = 0; iter < maxIterations; iter++) {
      const L = scores.map((s) =>
        clampScalar(sigmoidScalar(alpha * (s - beta))),
      );

      let gradAlpha = 0;
      let gradBeta = 0;

      if (mode === "prior_aware") {
        for (let i = 0; i < scores.length; i++) {
          const lv = L[i]!;
          const p = priors![i]!;
          const denom = lv * p + (1.0 - lv) * (1.0 - p);
          const predicted = clampScalar((lv * p) / denom);

          // Chain rule: dBCE/dalpha = (P - y) * dP/dL * dL/dalpha
          const dP_dL = (p * (1.0 - p)) / (denom * denom);
          const dL_dalpha = lv * (1.0 - lv) * (scores[i]! - beta);
          const dL_dbeta = -lv * (1.0 - lv) * alpha;

          const error = predicted - labels[i]!;
          gradAlpha += error * dP_dL * dL_dalpha;
          gradBeta += error * dP_dL * dL_dbeta;
        }
      } else {
        // balanced or prior_free: train on sigmoid likelihood
        for (let i = 0; i < scores.length; i++) {
          const error = L[i]! - labels[i]!;
          gradAlpha += error * (scores[i]! - beta);
          gradBeta += error * -alpha;
        }
      }

      gradAlpha /= scores.length;
      gradBeta /= scores.length;

      const newAlpha = alpha - learningRate * gradAlpha;
      const newBeta = beta - learningRate * gradBeta;

      if (
        Math.abs(newAlpha - alpha) < tolerance &&
        Math.abs(newBeta - beta) < tolerance
      ) {
        alpha = newAlpha;
        beta = newBeta;
        break;
      }

      alpha = newAlpha;
      beta = newBeta;
    }

    this.alpha = alpha;
    this.beta = beta;
    this._trainingMode = mode;
    this._nUpdates = 0;
    this._gradAlphaEMA = 0.0;
    this._gradBetaEMA = 0.0;
    this._alphaAvg = alpha;
    this._betaAvg = beta;
  }

  // Online update of alpha and beta from a single observation or mini-batch.
  //
  // Uses SGD with exponential moving average (EMA) of gradients to
  // smooth out noise from individual feedback signals.  Alpha is
  // constrained to remain positive.
  //
  // After each parameter step, Polyak-style EMA averaging is applied
  // to produce stable averagedAlpha and averagedBeta values.
  update(
    score: number | number[],
    label: number | number[],
    options: UpdateOptions = {},
  ): void {
    const {
      learningRate = 0.01,
      momentum = 0.9,
      decayTau = 1000.0,
      maxGradNorm = 1.0,
      avgDecay = 0.995,
      mode,
      tf,
      docLenRatio,
    } = options;

    const effectiveMode = mode !== undefined ? mode : this._trainingMode;
    if (!VALID_MODES.has(effectiveMode)) {
      throw new Error(
        `mode must be one of "balanced", "prior_aware", "prior_free", got "${effectiveMode}"`,
      );
    }
    if (effectiveMode === "prior_aware") {
      if (tf === undefined || docLenRatio === undefined) {
        throw new Error(
          "tf and docLenRatio are required when mode='prior_aware'",
        );
      }
    }

    const scores = Array.isArray(score) ? score : [score];
    const labels = Array.isArray(label) ? label : [label];

    const L = scores.map((s) =>
      clampScalar(sigmoidScalar(this.alpha * (s - this.beta))),
    );

    let gradAlpha = 0;
    let gradBeta = 0;

    if (effectiveMode === "prior_aware") {
      const tfs = Array.isArray(tf) ? tf : [tf!];
      const dlrs = Array.isArray(docLenRatio) ? docLenRatio : [docLenRatio!];
      const priors = BayesianProbabilityTransform.compositePrior(
        tfs,
        dlrs,
      );

      for (let i = 0; i < scores.length; i++) {
        const lv = L[i]!;
        const p = priors[i]!;
        const denom = lv * p + (1.0 - lv) * (1.0 - p);
        const predicted = clampScalar((lv * p) / denom);

        const dP_dL = (p * (1.0 - p)) / (denom * denom);
        const dL_dalpha = lv * (1.0 - lv) * (scores[i]! - this.beta);
        const dL_dbeta = -lv * (1.0 - lv) * this.alpha;

        const error = predicted - labels[i]!;
        gradAlpha += error * dP_dL * dL_dalpha;
        gradBeta += error * dP_dL * dL_dbeta;
      }
    } else {
      for (let i = 0; i < scores.length; i++) {
        const error = L[i]! - labels[i]!;
        gradAlpha += error * (scores[i]! - this.beta);
        gradBeta += error * -this.alpha;
      }
    }

    gradAlpha /= scores.length;
    gradBeta /= scores.length;

    if (mode !== undefined) {
      this._trainingMode = effectiveMode;
    }

    // EMA smoothing of gradients
    this._gradAlphaEMA =
      momentum * this._gradAlphaEMA + (1 - momentum) * gradAlpha;
    this._gradBetaEMA =
      momentum * this._gradBetaEMA + (1 - momentum) * gradBeta;

    // Bias correction for early updates
    this._nUpdates += 1;
    const correction = 1.0 - Math.pow(momentum, this._nUpdates);
    let correctedGradAlpha = this._gradAlphaEMA / correction;
    let correctedGradBeta = this._gradBetaEMA / correction;

    // Gradient clipping
    const gradNorm = Math.sqrt(
      correctedGradAlpha ** 2 + correctedGradBeta ** 2,
    );
    if (gradNorm > maxGradNorm) {
      const scale = maxGradNorm / gradNorm;
      correctedGradAlpha *= scale;
      correctedGradBeta *= scale;
    }

    // Learning rate decay: lr / (1 + t / tau)
    const effectiveLR = learningRate / (1.0 + this._nUpdates / decayTau);

    this.alpha -= effectiveLR * correctedGradAlpha;
    this.beta -= effectiveLR * correctedGradBeta;

    // Alpha must stay positive
    const ALPHA_MIN = 0.01;
    if (this.alpha < ALPHA_MIN) {
      this.alpha = ALPHA_MIN;
    }

    // Polyak parameter averaging for stable inference
    this._alphaAvg =
      avgDecay * this._alphaAvg + (1.0 - avgDecay) * this.alpha;
    this._betaAvg = avgDecay * this._betaAvg + (1.0 - avgDecay) * this.beta;
  }
}
