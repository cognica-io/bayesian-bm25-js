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

export interface FitOptions {
  learningRate?: number;
  maxIterations?: number;
  tolerance?: number;
}

export interface UpdateOptions {
  learningRate?: number;
  momentum?: number;
  decayTau?: number;
  maxGradNorm?: number;
  avgDecay?: number;
}

// Transforms raw BM25 scores into calibrated probabilities.
export class BayesianProbabilityTransform {
  public alpha: number;
  public beta: number;

  private _nUpdates: number = 0;
  private _gradAlphaEMA: number = 0.0;
  private _gradBetaEMA: number = 0.0;
  private _alphaAvg: number;
  private _betaAvg: number;

  constructor(alpha: number = 1.0, beta: number = 0.0) {
    this.alpha = alpha;
    this.beta = beta;
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

  // Bayesian posterior: L*p / (L*p + (1-L)*(1-p))  (Eq. 22)
  static posterior(likelihoodVal: number, prior: number): number;
  static posterior(likelihoodVal: number[], prior: number[]): number[];
  static posterior(
    likelihoodVal: number | number[],
    prior: number | number[],
  ): number | number[] {
    if (Array.isArray(likelihoodVal) && Array.isArray(prior)) {
      return likelihoodVal.map((lv, i) => {
        const p = prior[i]!;
        const numerator = lv * p;
        const denominator = numerator + (1.0 - lv) * (1.0 - p);
        return clampScalar(numerator / denominator);
      });
    }
    const lv = likelihoodVal as number;
    const p = prior as number;
    const numerator = lv * p;
    const denominator = numerator + (1.0 - lv) * (1.0 - p);
    return clampScalar(numerator / denominator);
  }

  // Full pipeline: BM25 score -> calibrated probability.
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
      const prior = BayesianProbabilityTransform.compositePrior(
        tf,
        docLenRatio,
      );
      return BayesianProbabilityTransform.posterior(lVal, prior);
    }
    const lVal = this.likelihood(score as number);
    const prior = BayesianProbabilityTransform.compositePrior(
      tf as number,
      docLenRatio as number,
    );
    return BayesianProbabilityTransform.posterior(
      lVal as number,
      prior as number,
    );
  }

  // Learn alpha and beta via gradient descent on binary cross-entropy.
  fit(scores: number[], labels: number[], options: FitOptions = {}): void {
    const {
      learningRate = 0.01,
      maxIterations = 1000,
      tolerance = 1e-6,
    } = options;

    let alpha = this.alpha;
    let beta = this.beta;

    for (let iter = 0; iter < maxIterations; iter++) {
      const predicted = scores.map((s) =>
        clampScalar(sigmoidScalar(alpha * (s - beta))),
      );

      let gradAlpha = 0;
      let gradBeta = 0;
      for (let i = 0; i < scores.length; i++) {
        const error = predicted[i]! - labels[i]!;
        gradAlpha += error * (scores[i]! - beta);
        gradBeta += error * -alpha;
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
    this._nUpdates = 0;
    this._gradAlphaEMA = 0.0;
    this._gradBetaEMA = 0.0;
    this._alphaAvg = alpha;
    this._betaAvg = beta;
  }

  // Online update of alpha and beta from a single observation or mini-batch.
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
    } = options;

    const scores = Array.isArray(score) ? score : [score];
    const labels = Array.isArray(label) ? label : [label];

    const predicted = scores.map((s) =>
      clampScalar(sigmoidScalar(this.alpha * (s - this.beta))),
    );

    let gradAlpha = 0;
    let gradBeta = 0;
    for (let i = 0; i < scores.length; i++) {
      const error = predicted[i]! - labels[i]!;
      gradAlpha += error * (scores[i]! - this.beta);
      gradBeta += error * -this.alpha;
    }
    gradAlpha /= scores.length;
    gradBeta /= scores.length;

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
