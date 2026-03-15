//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

// Probabilistic score combination functions.
//
// Implements AND, OR, NOT, and log-odds conjunction for combining multiple
// probability estimates.  The log-odds conjunction (from "From Bayesian
// Inference to Neural Computation") resolves the shrinkage problem of
// naive probabilistic AND by using geometric-mean log-odds with an
// agreement bonus.
//
// All functions operate on arrays and are batch-friendly.

import { clampProbability, logit, sigmoid } from "./probability.js";

// Convert cosine similarity to probability (Definition 7.1.2).
//
// Maps cosine similarity in [-1, 1] to probability in (0, 1) via
// P_vector = (1 + score) / 2, with epsilon clamping for numerical stability.
export function cosineToProbability(score: number): number;
export function cosineToProbability(score: number[]): number[];
export function cosineToProbability(
  score: number | number[],
): number | number[] {
  if (Array.isArray(score)) {
    return clampProbability(score.map((s) => (1.0 + s) / 2.0));
  }
  return clampProbability((1.0 + score) / 2.0);
}

// Probabilistic NOT via complement rule (Eq. 35).
//
// Computes P(NOT R) = 1 - P(R).  In log-odds space this corresponds
// to negation: logit(1 - p) = -logit(p), so NOT simply flips the
// sign of evidence.
export function probNot(prob: number): number;
export function probNot(prob: number[]): number[];
export function probNot(prob: number | number[]): number | number[] {
  if (Array.isArray(prob)) {
    const clamped = clampProbability(prob);
    return clampProbability(clamped.map((p) => 1.0 - p));
  }
  return clampProbability(1.0 - clampProbability(prob));
}

function is2D(probs: number[] | number[][]): probs is number[][] {
  return Array.isArray(probs[0]);
}

function probAndSingle(probs: number[]): number {
  const clamped = clampProbability(probs);
  let logSum = 0;
  for (const p of clamped) {
    logSum += Math.log(p);
  }
  return Math.exp(logSum);
}

// Probabilistic AND via product rule in log-space (Eq. 33-34).
//
// For 1D input: reduces the array to a single probability.
// For 2D input (batched): reduces along the last axis.
export function probAnd(probs: number[]): number;
export function probAnd(probs: number[][]): number[];
export function probAnd(probs: number[] | number[][]): number | number[] {
  if (probs.length === 0) return 0;
  if (is2D(probs)) {
    return probs.map(probAndSingle);
  }
  return probAndSingle(probs as number[]);
}

function probOrSingle(probs: number[]): number {
  const clamped = clampProbability(probs);
  let logSum = 0;
  for (const p of clamped) {
    logSum += Math.log(1.0 - p);
  }
  return 1.0 - Math.exp(logSum);
}

// Probabilistic OR via complement rule in log-space (Eq. 36-37).
//
// For 1D input: reduces the array to a single probability.
// For 2D input (batched): reduces along the last axis.
export function probOr(probs: number[]): number;
export function probOr(probs: number[][]): number[];
export function probOr(probs: number[] | number[][]): number | number[] {
  if (probs.length === 0) return 0;
  if (is2D(probs)) {
    return probs.map(probOrSingle);
  }
  return probOrSingle(probs as number[]);
}

const _SQRT_N_ALPHA = 0.5; // alpha=0.5 implements the sqrt(n) scaling law (Theorem 4.2.1)

// Resolve alpha parameter: "auto" -> sqrt(n) scaling, undefined -> default.
export function resolveAlpha(
  alpha: number | "auto" | undefined,
  defaultValue: number,
): number {
  if (alpha === undefined) {
    return defaultValue;
  }
  if (alpha === "auto") {
    return _SQRT_N_ALPHA;
  }
  if (typeof alpha === "string") {
    throw new Error(
      `alpha must be a number, undefined, or "auto", got "${alpha}"`,
    );
  }
  return alpha;
}

// Apply sparse-signal gating to logit values before aggregation.
//
// - "relu": MAP estimate under sparse prior (Theorem 6.5.3).
//   Zeroes out weak/negative evidence: max(0, logit).
// - "swish": Bayes estimate under sparse prior (Theorem 6.7.4).
//   Soft gating: logit * sigmoid(beta * logit). When beta=1.0
//   this is the standard swish (Theorem 6.7.6).
// - "gelu": Bayesian expected signal under Gaussian noise model
//   (Theorem 6.8.1, Proposition 6.8.2).  Approximated as
//   logit * sigmoid(1.702 * logit), which matches Swish_1.702.
//   The beta parameter is ignored for gelu.
function applyGating(
  logitValues: number[],
  gating: string,
  beta: number = 1.0,
): number[] {
  if (gating === "none") {
    return logitValues;
  }
  if (gating === "relu") {
    return logitValues.map((l) => Math.max(0.0, l));
  }
  if (gating === "swish") {
    return logitValues.map((l) => l * (sigmoid(beta * l) as number));
  }
  if (gating === "gelu") {
    return logitValues.map((l) => l * (sigmoid(1.702 * l) as number));
  }
  throw new Error(
    `gating must be "none", "relu", "swish", or "gelu", got "${gating}"`,
  );
}

function logOddsConjunctionSingle(
  probs: number[],
  alpha: number,
  gating: string,
  gatingBeta: number,
): number {
  const clamped = clampProbability(probs);
  const n = clamped.length;

  // Step 1: mean log-odds (Eq. 20)
  const rawLogits = logit(clamped) as number[];
  const gatedLogits = applyGating(rawLogits, gating, gatingBeta);
  let logitSum = 0;
  for (const l of gatedLogits) {
    logitSum += l;
  }
  const lBar = logitSum / n;

  // Step 2: multiplicative confidence scaling (Eq. 23)
  const lAdjusted = lBar * n ** alpha;

  // Step 3: back to probability (Eq. 26)
  return sigmoid(lAdjusted) as number;
}

function logOddsWeightedSingle(
  probs: number[],
  weights: number[],
  alpha: number,
  gating: string,
  gatingBeta: number,
): number {
  const clamped = clampProbability(probs);
  const n = clamped.length;
  const rawLogits = logit(clamped) as number[];
  const gatedLogits = applyGating(rawLogits, gating, gatingBeta);
  let weightedSum = 0;
  for (let i = 0; i < gatedLogits.length; i++) {
    weightedSum += weights[i]! * gatedLogits[i]!;
  }
  // Log-OP with confidence scaling:
  // sigma(n^alpha * sum(w_i * logit(P_i)))  (Theorem 8.3 + Section 4.2)
  return sigmoid(n ** alpha * weightedSum) as number;
}

// Log-odds conjunction with multiplicative confidence scaling (Paper 2, Section 4).
//
// Resolves the shrinkage problem of naive probabilistic AND by:
//   1. Computing the mean log-odds (Eq. 20)
//   2. Multiplicative confidence scaling by n^alpha (Eq. 23)
//   3. Converting back to probability via sigmoid (Eq. 26)
//
// When weights are provided, uses the Log-OP (Log-linear Opinion
// Pool) formulation from Paper 2, Theorem 8.3 / Remark 8.4 instead:
// sigma(n^alpha * sum(w_i * logit(P_i))) where sum(w_i) = 1 and
// w_i >= 0.  Per-signal weights (Theorem 8.3) and confidence scaling
// by signal count (Section 4.2) are orthogonal and compose
// multiplicatively.
//
// The multiplicative formulation (rather than additive) preserves the
// sign of evidence (Theorem 4.2.2), preventing accidental inversion
// of irrelevance signals (Remark 4.2.4).
export function logOddsConjunction(
  probs: number[],
  alpha?: number | "auto",
  weights?: number[],
  gating?: string,
  gatingBeta?: number,
): number;
export function logOddsConjunction(
  probs: number[][],
  alpha?: number | "auto",
  weights?: number[],
  gating?: string,
  gatingBeta?: number,
): number[];
export function logOddsConjunction(
  probs: number[] | number[][],
  alpha?: number | "auto",
  weights?: number[],
  gating: string = "none",
  gatingBeta: number = 1.0,
): number | number[] {
  if (probs.length === 0) return 0;

  if (weights !== undefined) {
    for (const w of weights) {
      if (w < 0) {
        throw new Error("weights must be non-negative");
      }
    }
    let weightSum = 0;
    for (const w of weights) {
      weightSum += w;
    }
    if (Math.abs(weightSum - 1.0) > 1e-6) {
      throw new Error(`weights must sum to 1, got ${weightSum}`);
    }

    const effectiveAlpha = resolveAlpha(alpha, 0.0);

    if (is2D(probs)) {
      return probs.map((row) =>
        logOddsWeightedSingle(row, weights, effectiveAlpha, gating, gatingBeta),
      );
    }
    return logOddsWeightedSingle(
      probs as number[],
      weights,
      effectiveAlpha,
      gating,
      gatingBeta,
    );
  }

  const effectiveAlpha = resolveAlpha(alpha, 0.5);

  if (is2D(probs)) {
    return probs.map((row) =>
      logOddsConjunctionSingle(row, effectiveAlpha, gating, gatingBeta),
    );
  }
  return logOddsConjunctionSingle(
    probs as number[],
    effectiveAlpha,
    gating,
    gatingBeta,
  );
}

// Min-max normalize to [0, 1].  Returns zeros if range is negligible.
function minMaxNormalize(arr: number[]): number[] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of arr) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (hi - lo < 1e-12) {
    return arr.map(() => 0);
  }
  const range = hi - lo;
  return arr.map((v) => (v - lo) / range);
}

// Balanced log-odds fusion for hybrid sparse-dense retrieval.
//
// Combines Bayesian BM25 probabilities with dense cosine similarities
// by normalizing both signals in logit space.  Min-max normalization
// ensures each signal contributes equally, preventing the heavy-tailed
// sparse logits (from sigmoid unwrapping) from drowning the dense signal.
//
// Pipeline:
//   1. sparse_probs -> logit(p_sparse)
//   2. dense_similarities -> cosine_to_probability -> logit(p_dense)
//   3. Min-max normalize each logit array to [0, 1]
//   4. Return weight * logit_dense_norm + (1 - weight) * logit_sparse_norm
//
// Returns fusion scores (not probabilities).  Higher is more relevant.
export function balancedLogOddsFusion(
  sparseProbs: number[],
  denseSimilarities: number[],
  weight: number = 0.5,
): number[] {
  const logitSparse = logit(clampProbability(sparseProbs)) as number[];
  const logitDense = logit(
    cosineToProbability(denseSimilarities) as number[],
  ) as number[];

  const logitSparseNorm = minMaxNormalize(logitSparse);
  const logitDenseNorm = minMaxNormalize(logitDense);

  return logitSparseNorm.map(
    (ls, i) => weight * logitDenseNorm[i]! + (1.0 - weight) * ls,
  );
}

// Numerically stable softmax: shift by max to prevent overflow.
function softmax(z: number[]): number[] {
  let maxZ = -Infinity;
  for (const v of z) {
    if (v > maxZ) maxZ = v;
  }
  const expZ = z.map((v) => Math.exp(v - maxZ));
  let sum = 0;
  for (const v of expZ) {
    sum += v;
  }
  return expZ.map((v) => v / sum);
}

// Learnable per-signal reliability weights for log-odds conjunction (Remark 5.3.2).
//
// Learns weights that map from the Naive Bayes uniform initialization
// (w_i = 1/n) to per-signal reliability weights, completing the
// correspondence to a fully parameterized single-layer network in
// log-odds space: logit -> weighted sum -> sigmoid.
//
// The gradient dL/dz_j = n^alpha * (p - y) * w_j * (x_j - x_bar_w)
// is Hebbian: the product of pre-synaptic activity (signal deviation
// from weighted mean) and post-synaptic error (prediction minus label).
export class LearnableLogOddsWeights {
  private _nSignals: number;
  private _alpha: number;
  private _baseRate: number | null;
  private _logitBaseRate: number | null;
  private _logits: number[];
  private _nUpdates: number = 0;
  private _gradLogitsEMA: number[];
  private _weightsAvg: number[];

  constructor(
    nSignals: number,
    alpha: number | "auto" = 0.0,
    baseRate: number | null = null,
  ) {
    if (nSignals < 1) {
      throw new Error(`n_signals must be >= 1, got ${nSignals}`);
    }
    if (baseRate !== null && (baseRate <= 0.0 || baseRate >= 1.0)) {
      throw new Error(`base_rate must be in (0, 1), got ${baseRate}`);
    }
    this._nSignals = nSignals;
    this._alpha = resolveAlpha(alpha, 0.0);
    this._baseRate = baseRate;
    this._logitBaseRate =
      baseRate !== null ? (logit(baseRate) as number) : null;

    // Softmax parameterization: zeros -> uniform 1/n (Naive Bayes init)
    this._logits = new Array(nSignals).fill(0);
    this._gradLogitsEMA = new Array(nSignals).fill(0);
    this._weightsAvg = new Array(nSignals).fill(1.0 / nSignals);
  }

  get nSignals(): number {
    return this._nSignals;
  }

  get alpha(): number {
    return this._alpha;
  }

  // Corpus-level base rate of relevance, or null if not set.
  get baseRate(): number | null {
    return this._baseRate;
  }

  // Current weights: softmax of internal logits.
  get weights(): number[] {
    return softmax(this._logits);
  }

  // Polyak-averaged weights for stable inference.
  get averagedWeights(): number[] {
    return [...this._weightsAvg];
  }

  // Combine probability signals via weighted log-odds conjunction.
  combine(probs: number[], useAveraged?: boolean): number;
  combine(probs: number[][], useAveraged?: boolean): number[];
  combine(
    probs: number[] | number[][],
    useAveraged: boolean = false,
  ): number | number[] {
    const w = useAveraged ? this._weightsAvg : this.weights;

    if (this._logitBaseRate === null) {
      return logOddsConjunction(probs as number[], this._alpha, w);
    }

    // With base rate: compute weighted log-odds manually, add logit(baseRate)
    const n = this._nSignals;
    const scale = n ** this._alpha;
    const logitBR = this._logitBaseRate;

    if (is2D(probs as number[] | number[][])) {
      const probsBatch = probs as number[][];
      const results: number[] = [];
      for (const row of probsBatch) {
        const clamped = clampProbability(row);
        const x = logit(clamped) as number[];
        let weightedSum = 0;
        for (let j = 0; j < n; j++) {
          weightedSum += w[j]! * x[j]!;
        }
        let lWeighted = scale * weightedSum;
        lWeighted += logitBR;
        results.push(sigmoid(lWeighted) as number);
      }
      return results;
    }

    const clamped = clampProbability(probs as number[]);
    const x = logit(clamped) as number[];
    let weightedSum = 0;
    for (let j = 0; j < n; j++) {
      weightedSum += w[j]! * x[j]!;
    }
    let lWeighted = scale * weightedSum;
    lWeighted += logitBR;
    return sigmoid(lWeighted) as number;
  }

  // Batch gradient descent on BCE loss to learn weights.
  //
  // The gradient for logit z_j is (averaged over samples):
  //   dL/dz_j = n^alpha * (p - y) * w_j * (x_j - x_bar_w)
  //
  // where x_i = logit(P_i), x_bar_w = sum(w_i * x_i), and
  // p = sigmoid(n^alpha * x_bar_w).
  fit(
    probs: number[][],
    labels: number[],
    options: {
      learningRate?: number;
      maxIterations?: number;
      tolerance?: number;
    } = {},
  ): void {
    const {
      learningRate = 0.01,
      maxIterations = 1000,
      tolerance = 1e-6,
    } = options;

    // Validate dimensions
    for (const row of probs) {
      if (row.length !== this._nSignals) {
        throw new Error(
          `probs last dimension ${row.length} != n_signals ${this._nSignals}`,
        );
      }
    }

    const n = this._nSignals;
    const scale = n ** this._alpha;
    const m = probs.length;

    // Log-odds of input signals: shape (m, n)
    const x = probs.map((row) => logit(clampProbability(row)) as number[]);

    for (let iter = 0; iter < maxIterations; iter++) {
      const w = softmax(this._logits);

      // Compute gradient for each logit z_j, averaged over samples
      const gradLogits = new Array(n).fill(0) as number[];

      for (let s = 0; s < m; s++) {
        // Weighted mean log-odds for this sample
        let xBarW = 0;
        for (let j = 0; j < n; j++) {
          xBarW += w[j]! * x[s]![j]!;
        }

        // Predicted probability
        let lWeighted = scale * xBarW;
        if (this._logitBaseRate !== null) {
          lWeighted += this._logitBaseRate;
        }
        const p = sigmoid(lWeighted) as number;
        const error = p - labels[s]!;

        // Accumulate gradient
        for (let j = 0; j < n; j++) {
          gradLogits[j]! += scale * error * w[j]! * (x[s]![j]! - xBarW);
        }
      }

      // Average over samples
      for (let j = 0; j < n; j++) {
        gradLogits[j]! /= m;
      }

      // Update logits
      let maxChange = 0;
      for (let j = 0; j < n; j++) {
        const change = learningRate * gradLogits[j]!;
        this._logits[j]! -= change;
        if (Math.abs(change) > maxChange) {
          maxChange = Math.abs(change);
        }
      }

      if (maxChange < tolerance) {
        break;
      }
    }

    // Reset online state after batch fit
    this._nUpdates = 0;
    this._gradLogitsEMA = new Array(n).fill(0);
    this._weightsAvg = [...softmax(this._logits)];
  }

  // Online SGD update from a single observation or mini-batch.
  //
  // Follows the same patterns as BayesianProbabilityTransform.update():
  // EMA gradient smoothing with bias correction, L2 gradient clipping,
  // learning rate decay, and Polyak averaging of weights in the simplex.
  update(
    probs: number[] | number[][],
    label: number | number[],
    options: {
      learningRate?: number;
      momentum?: number;
      decayTau?: number;
      maxGradNorm?: number;
      avgDecay?: number;
    } = {},
  ): void {
    const {
      learningRate = 0.01,
      momentum = 0.9,
      decayTau = 1000.0,
      maxGradNorm = 1.0,
      avgDecay = 0.995,
    } = options;

    // Normalize to 2D
    const probsBatch: number[][] = is2D(probs as number[] | number[][])
      ? (probs as number[][])
      : [probs as number[]];
    const labelsBatch: number[] = Array.isArray(label)
      ? label
      : [label];

    // Validate dimensions
    for (const row of probsBatch) {
      if (row.length !== this._nSignals) {
        throw new Error(
          `probs last dimension ${row.length} != n_signals ${this._nSignals}`,
        );
      }
    }

    const n = this._nSignals;
    const scale = n ** this._alpha;
    const w = softmax(this._logits);
    const m = probsBatch.length;

    // Log-odds of input signals
    const x = probsBatch.map(
      (row) => logit(clampProbability(row)) as number[],
    );

    // Gradient for each logit, averaged over mini-batch
    const gradLogits = new Array(n).fill(0) as number[];

    for (let s = 0; s < m; s++) {
      let xBarW = 0;
      for (let j = 0; j < n; j++) {
        xBarW += w[j]! * x[s]![j]!;
      }
      let lWeighted = scale * xBarW;
      if (this._logitBaseRate !== null) {
        lWeighted += this._logitBaseRate;
      }
      const p = sigmoid(lWeighted) as number;
      const error = p - labelsBatch[s]!;

      for (let j = 0; j < n; j++) {
        gradLogits[j]! += scale * error * w[j]! * (x[s]![j]! - xBarW);
      }
    }

    for (let j = 0; j < n; j++) {
      gradLogits[j]! /= m;
    }

    // EMA smoothing of gradients
    for (let j = 0; j < n; j++) {
      this._gradLogitsEMA[j] =
        momentum * this._gradLogitsEMA[j]! + (1.0 - momentum) * gradLogits[j]!;
    }

    // Bias correction for early updates
    this._nUpdates += 1;
    const correction = 1.0 - Math.pow(momentum, this._nUpdates);
    const correctedGrad = this._gradLogitsEMA.map((g) => g / correction);

    // L2 gradient clipping
    let gradNorm = 0;
    for (const g of correctedGrad) {
      gradNorm += g * g;
    }
    gradNorm = Math.sqrt(gradNorm);
    if (gradNorm > maxGradNorm) {
      const clipScale = maxGradNorm / gradNorm;
      for (let j = 0; j < n; j++) {
        correctedGrad[j]! *= clipScale;
      }
    }

    // Learning rate decay: lr / (1 + t / tau)
    const effectiveLR = learningRate / (1.0 + this._nUpdates / decayTau);

    for (let j = 0; j < n; j++) {
      this._logits[j]! -= effectiveLR * correctedGrad[j]!;
    }

    // Polyak averaging of weights in the simplex
    const rawWeights = softmax(this._logits);
    for (let j = 0; j < n; j++) {
      this._weightsAvg[j] =
        avgDecay * this._weightsAvg[j]! + (1.0 - avgDecay) * rawWeights[j]!;
    }
  }
}

// Numerically stable softmax along rows of a 2D array.
function softmax2D(z: number[][]): number[][] {
  return z.map((row) => {
    let maxZ = -Infinity;
    for (const v of row) {
      if (v > maxZ) maxZ = v;
    }
    const expZ = row.map((v) => Math.exp(v - maxZ));
    let sum = 0;
    for (const v of expZ) {
      sum += v;
    }
    return expZ.map((v) => v / sum);
  });
}

// Seeded PRNG (mulberry32) for deterministic initialization.
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller transform: generate N(0, 1) samples from uniform.
function randNormal(rng: () => number): number {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2.0 * Math.log(u1 || 1e-15)) * Math.cos(2.0 * Math.PI * u2);
}

// Query-dependent signal weighting via attention (Paper 2, Section 8).
//
// Computes per-signal softmax attention weights from query features:
// w_i(q) = softmax(W @ features + b)[i], then combines probability
// signals via weighted log-odds conjunction.  This enables the fusion
// weights to adapt per-query rather than being fixed across all queries.
//
// The class is feature-agnostic -- it learns a linear projection from
// arbitrary user-provided query features to softmax attention weights.
export class AttentionLogOddsWeights {
  private _nSignals: number;
  private _nQueryFeatures: number;
  private _alpha: number;
  private _normalize: boolean;
  private _baseRate: number | null;
  private _logitBaseRate: number | null;

  // W: (nSignals x nQueryFeatures), b: (nSignals,)
  private _W: number[][];
  private _b: number[];

  // Online learning state
  private _nUpdates: number = 0;
  private _gradWEMA: number[][];
  private _gradBEMA: number[];

  // Polyak averaging
  private _WAvg: number[][];
  private _bAvg: number[];

  constructor(
    nSignals: number,
    nQueryFeatures: number,
    alpha: number | "auto" = 0.5,
    normalize: boolean = false,
    seed: number = 0,
    baseRate: number | null = null,
  ) {
    if (nSignals < 1) {
      throw new Error(`n_signals must be >= 1, got ${nSignals}`);
    }
    if (nQueryFeatures < 1) {
      throw new Error(
        `n_query_features must be >= 1, got ${nQueryFeatures}`,
      );
    }
    if (baseRate !== null && (baseRate <= 0.0 || baseRate >= 1.0)) {
      throw new Error(`base_rate must be in (0, 1), got ${baseRate}`);
    }
    this._nSignals = nSignals;
    this._nQueryFeatures = nQueryFeatures;
    this._alpha = resolveAlpha(alpha, 0.5);
    this._normalize = normalize;
    this._baseRate = baseRate;
    this._logitBaseRate =
      baseRate !== null ? (logit(baseRate) as number) : null;

    // Xavier-style initialization scaled for softmax input
    const scale = 1.0 / Math.sqrt(nQueryFeatures);
    const rng = mulberry32(seed);
    this._W = [];
    for (let i = 0; i < nSignals; i++) {
      const row: number[] = [];
      for (let j = 0; j < nQueryFeatures; j++) {
        row.push(randNormal(rng) * scale);
      }
      this._W.push(row);
    }
    this._b = new Array(nSignals).fill(0);

    // Online learning state
    this._gradWEMA = [];
    for (let i = 0; i < nSignals; i++) {
      this._gradWEMA.push(new Array(nQueryFeatures).fill(0));
    }
    this._gradBEMA = new Array(nSignals).fill(0);

    // Polyak averaging
    this._WAvg = this._W.map((row) => [...row]);
    this._bAvg = [...this._b];
  }

  get nSignals(): number {
    return this._nSignals;
  }

  get nQueryFeatures(): number {
    return this._nQueryFeatures;
  }

  get alpha(): number {
    return this._alpha;
  }

  // Corpus-level base rate of relevance, or null if not set.
  get baseRate(): number | null {
    return this._baseRate;
  }

  // Whether per-signal logit normalization is enabled.
  get normalize(): boolean {
    return this._normalize;
  }

  // Per-column min-max normalization on logit array.
  // Each column (signal) is independently normalized to [0, 1].
  private static _normalizeLogits(x: number[][]): number[][] {
    if (x.length === 0) return [];
    const nCols = x[0]!.length;
    const result = x.map((row) => [...row]);
    for (let col = 0; col < nCols; col++) {
      const column = result.map((row) => row[col]!);
      const normalized = minMaxNormalize(column);
      for (let row = 0; row < result.length; row++) {
        result[row]![col] = normalized[row]!;
      }
    }
    return result;
  }

  // Weight matrix W of shape (nSignals, nQueryFeatures). Returns a copy.
  get weightsMatrix(): number[][] {
    return this._W.map((row) => [...row]);
  }

  // Compute softmax attention weights from query features.
  // queryFeatures: array of shape (nQueryFeatures,) or array of arrays (m, nQueryFeatures)
  private _computeWeights(
    queryFeatures: number[][],
    useAveraged: boolean = false,
  ): number[][] {
    const W = useAveraged ? this._WAvg : this._W;
    const b = useAveraged ? this._bAvg : this._b;
    const n = this._nSignals;

    // z[i] = queryFeatures[i] @ W.T + b  -> (m, nSignals)
    const z: number[][] = [];
    for (const qf of queryFeatures) {
      const row: number[] = [];
      for (let s = 0; s < n; s++) {
        let dot = b[s]!;
        for (let f = 0; f < this._nQueryFeatures; f++) {
          dot += W[s]![f]! * qf[f]!;
        }
        row.push(dot);
      }
      z.push(row);
    }

    return softmax2D(z);
  }

  // Combine probability signals via query-dependent weighted log-odds.
  combine(
    probs: number[],
    queryFeatures: number[],
    useAveraged?: boolean,
  ): number;
  combine(
    probs: number[][],
    queryFeatures: number[][],
    useAveraged?: boolean,
  ): number[];
  combine(
    probs: number[] | number[][],
    queryFeatures: number[] | number[][],
    useAveraged: boolean = false,
  ): number | number[] {
    // Ensure 2D
    const qf2D =
      typeof queryFeatures[0] === "number"
        ? [queryFeatures as number[]]
        : (queryFeatures as number[][]);

    const w = this._computeWeights(qf2D, useAveraged);

    if (!is2D(probs as number[] | number[][])) {
      // Single sample: normalization cannot apply (no candidates to
      // normalize across), fall through to direct computation.
      const wFlat = w[0]!;
      const xSingle = logit(clampProbability(probs as number[])) as number[];
      const n = this._nSignals;
      const scale = n ** this._alpha;
      let weightedSum = 0;
      for (let j = 0; j < n; j++) {
        weightedSum += wFlat[j]! * xSingle[j]!;
      }
      let lWeighted = scale * weightedSum;
      if (this._logitBaseRate !== null) {
        lWeighted += this._logitBaseRate;
      }
      return sigmoid(lWeighted) as number;
    }

    const probsBatch = probs as number[][];
    let x = probsBatch.map(
      (row) => logit(clampProbability(row)) as number[],
    );

    if (this._normalize) {
      x = AttentionLogOddsWeights._normalizeLogits(x);
    }

    const n = this._nSignals;
    const scale = n ** this._alpha;
    const results: number[] = [];
    for (let i = 0; i < x.length; i++) {
      const wi = w[Math.min(i, w.length - 1)]!;
      let weightedSum = 0;
      for (let j = 0; j < n; j++) {
        weightedSum += wi[j]! * x[i]![j]!;
      }
      let lWeighted = scale * weightedSum;
      if (this._logitBaseRate !== null) {
        lWeighted += this._logitBaseRate;
      }
      results.push(sigmoid(lWeighted) as number);
    }
    return results;
  }

  // Batch gradient descent on BCE loss to learn W and b.
  //
  // queryIds: optional query group identifiers for per-query normalization.
  // When normalize=true and queryIds is provided, logit normalization is
  // applied within each query group. When normalize=true and queryIds is
  // null, the whole batch is normalized as a single group.
  fit(
    probs: number[][],
    labels: number[],
    queryFeatures: number[][],
    options: {
      queryIds?: number[];
      learningRate?: number;
      maxIterations?: number;
      tolerance?: number;
    } = {},
  ): void {
    const {
      queryIds,
      learningRate = 0.01,
      maxIterations = 1000,
      tolerance = 1e-6,
    } = options;

    const m = probs.length;
    const n = this._nSignals;
    const scale = n ** this._alpha;

    // Log-odds of input signals: (m, n)
    let x = probs.map(
      (row) => logit(clampProbability(row)) as number[],
    );

    if (this._normalize) {
      if (queryIds !== undefined) {
        // Per-query normalization
        const uniqueIds = [...new Set(queryIds)];
        for (const qid of uniqueIds) {
          const indices: number[] = [];
          for (let i = 0; i < queryIds.length; i++) {
            if (queryIds[i] === qid) indices.push(i);
          }
          const group = indices.map((i) => x[i]!);
          const normalized = AttentionLogOddsWeights._normalizeLogits(group);
          for (let k = 0; k < indices.length; k++) {
            x[indices[k]!] = normalized[k]!;
          }
        }
      } else {
        x = AttentionLogOddsWeights._normalizeLogits(x);
      }
    }

    for (let iter = 0; iter < maxIterations; iter++) {
      // Compute per-sample attention weights
      // z = queryFeatures @ W.T + b  -> (m, n)
      const z: number[][] = [];
      for (let s = 0; s < m; s++) {
        const row: number[] = [];
        for (let j = 0; j < n; j++) {
          let dot = this._b[j]!;
          for (let f = 0; f < this._nQueryFeatures; f++) {
            dot += this._W[j]![f]! * queryFeatures[s]![f]!;
          }
          row.push(dot);
        }
        z.push(row);
      }
      const w = softmax2D(z);

      // Weighted log-odds per sample
      const xBarW: number[] = [];
      for (let s = 0; s < m; s++) {
        let sum = 0;
        for (let j = 0; j < n; j++) {
          sum += w[s]![j]! * x[s]![j]!;
        }
        xBarW.push(sum);
      }

      // Predicted probability
      const p = xBarW.map((xb) => {
        let lw = scale * xb;
        if (this._logitBaseRate !== null) {
          lw += this._logitBaseRate;
        }
        return sigmoid(lw) as number;
      });
      const error = p.map((pi, s) => pi - labels[s]!);

      // Gradient: dL/dz_j = scale * (p - y) * w_j * (x_j - x_bar_w)
      // gradZ: (m, n)
      const gradZ: number[][] = [];
      for (let s = 0; s < m; s++) {
        const row: number[] = [];
        for (let j = 0; j < n; j++) {
          row.push(
            scale * error[s]! * w[s]![j]! * (x[s]![j]! - xBarW[s]!),
          );
        }
        gradZ.push(row);
      }

      // dL/dW = (1/m) * gradZ.T @ queryFeatures  -> (n, nQueryFeatures)
      const gradW: number[][] = [];
      for (let j = 0; j < n; j++) {
        const row: number[] = [];
        for (let f = 0; f < this._nQueryFeatures; f++) {
          let sum = 0;
          for (let s = 0; s < m; s++) {
            sum += gradZ[s]![j]! * queryFeatures[s]![f]!;
          }
          row.push(sum / m);
        }
        gradW.push(row);
      }

      // gradB = mean(gradZ, axis=0)  -> (n,)
      const gradB: number[] = new Array(n).fill(0);
      for (let s = 0; s < m; s++) {
        for (let j = 0; j < n; j++) {
          gradB[j]! += gradZ[s]![j]!;
        }
      }
      for (let j = 0; j < n; j++) {
        gradB[j]! /= m;
      }

      // Update parameters and check convergence
      let maxChange = 0;
      for (let j = 0; j < n; j++) {
        for (let f = 0; f < this._nQueryFeatures; f++) {
          const change = learningRate * gradW[j]![f]!;
          this._W[j]![f]! -= change;
          if (Math.abs(change) > maxChange) {
            maxChange = Math.abs(change);
          }
        }
        const bChange = learningRate * gradB[j]!;
        this._b[j]! -= bChange;
        if (Math.abs(bChange) > maxChange) {
          maxChange = Math.abs(bChange);
        }
      }

      if (maxChange < tolerance) {
        break;
      }
    }

    // Reset online state after batch fit
    this._nUpdates = 0;
    for (let i = 0; i < n; i++) {
      for (let f = 0; f < this._nQueryFeatures; f++) {
        this._gradWEMA[i]![f] = 0;
      }
    }
    this._gradBEMA = new Array(n).fill(0);
    this._WAvg = this._W.map((row) => [...row]);
    this._bAvg = [...this._b];
  }

  // Online SGD update from a single observation or mini-batch.
  update(
    probs: number[] | number[][],
    label: number | number[],
    queryFeatures: number[] | number[][],
    options: {
      learningRate?: number;
      momentum?: number;
      decayTau?: number;
      maxGradNorm?: number;
      avgDecay?: number;
    } = {},
  ): void {
    const {
      learningRate = 0.01,
      momentum = 0.9,
      decayTau = 1000.0,
      maxGradNorm = 1.0,
      avgDecay = 0.995,
    } = options;

    // Normalize to 2D
    const probsBatch: number[][] = is2D(probs as number[] | number[][])
      ? (probs as number[][])
      : [probs as number[]];
    const labelsBatch: number[] = Array.isArray(label)
      ? label
      : [label];
    const qfBatch: number[][] =
      typeof queryFeatures[0] === "number"
        ? [queryFeatures as number[]]
        : (queryFeatures as number[][]);

    const m = probsBatch.length;
    const n = this._nSignals;
    const scale = n ** this._alpha;

    // Log-odds of input signals: (m, n)
    let x = probsBatch.map(
      (row) => logit(clampProbability(row)) as number[],
    );

    if (this._normalize && x.length > 0 && Array.isArray(x[0])) {
      x = AttentionLogOddsWeights._normalizeLogits(x);
    }

    // z = qfBatch @ W.T + b  -> (m, n)
    const z: number[][] = [];
    for (let s = 0; s < m; s++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) {
        let dot = this._b[j]!;
        for (let f = 0; f < this._nQueryFeatures; f++) {
          dot += this._W[j]![f]! * qfBatch[s]![f]!;
        }
        row.push(dot);
      }
      z.push(row);
    }
    const w = softmax2D(z);

    // Weighted log-odds per sample
    const xBarW: number[] = [];
    for (let s = 0; s < m; s++) {
      let sum = 0;
      for (let j = 0; j < n; j++) {
        sum += w[s]![j]! * x[s]![j]!;
      }
      xBarW.push(sum);
    }

    const p = xBarW.map((xb) => {
      let lw = scale * xb;
      if (this._logitBaseRate !== null) {
        lw += this._logitBaseRate;
      }
      return sigmoid(lw) as number;
    });
    const error = p.map((pi, s) => pi - labelsBatch[s]!);

    // gradZ: (m, n)
    const gradZ: number[][] = [];
    for (let s = 0; s < m; s++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) {
        row.push(
          scale * error[s]! * w[s]![j]! * (x[s]![j]! - xBarW[s]!),
        );
      }
      gradZ.push(row);
    }

    // dL/dW = (1/m) * gradZ.T @ qfBatch  -> (n, nQueryFeatures)
    const gradW: number[][] = [];
    for (let j = 0; j < n; j++) {
      const row: number[] = [];
      for (let f = 0; f < this._nQueryFeatures; f++) {
        let sum = 0;
        for (let s = 0; s < m; s++) {
          sum += gradZ[s]![j]! * qfBatch[s]![f]!;
        }
        row.push(sum / m);
      }
      gradW.push(row);
    }

    const gradB: number[] = new Array(n).fill(0);
    for (let s = 0; s < m; s++) {
      for (let j = 0; j < n; j++) {
        gradB[j]! += gradZ[s]![j]!;
      }
    }
    for (let j = 0; j < n; j++) {
      gradB[j]! /= m;
    }

    // EMA smoothing
    for (let j = 0; j < n; j++) {
      for (let f = 0; f < this._nQueryFeatures; f++) {
        this._gradWEMA[j]![f] =
          momentum * this._gradWEMA[j]![f]! +
          (1.0 - momentum) * gradW[j]![f]!;
      }
      this._gradBEMA[j] =
        momentum * this._gradBEMA[j]! + (1.0 - momentum) * gradB[j]!;
    }

    // Bias correction
    this._nUpdates += 1;
    const correction = 1.0 - Math.pow(momentum, this._nUpdates);

    // Corrected gradients
    const correctedW: number[][] = [];
    for (let j = 0; j < n; j++) {
      correctedW.push(
        this._gradWEMA[j]!.map((g) => g / correction),
      );
    }
    const correctedB = this._gradBEMA.map((g) => g / correction);

    // L2 gradient clipping (joint norm over W and b)
    let gradNorm = 0;
    for (let j = 0; j < n; j++) {
      for (let f = 0; f < this._nQueryFeatures; f++) {
        gradNorm += correctedW[j]![f]! ** 2;
      }
      gradNorm += correctedB[j]! ** 2;
    }
    gradNorm = Math.sqrt(gradNorm);
    if (gradNorm > maxGradNorm) {
      const clipScale = maxGradNorm / gradNorm;
      for (let j = 0; j < n; j++) {
        for (let f = 0; f < this._nQueryFeatures; f++) {
          correctedW[j]![f]! *= clipScale;
        }
        correctedB[j]! *= clipScale;
      }
    }

    // Learning rate decay
    const effectiveLR = learningRate / (1.0 + this._nUpdates / decayTau);

    // Update parameters
    for (let j = 0; j < n; j++) {
      for (let f = 0; f < this._nQueryFeatures; f++) {
        this._W[j]![f]! -= effectiveLR * correctedW[j]![f]!;
      }
      this._b[j]! -= effectiveLR * correctedB[j]!;
    }

    // Polyak averaging
    for (let j = 0; j < n; j++) {
      for (let f = 0; f < this._nQueryFeatures; f++) {
        this._WAvg[j]![f] =
          avgDecay * this._WAvg[j]![f]! +
          (1.0 - avgDecay) * this._W[j]![f]!;
      }
      this._bAvg[j] =
        avgDecay * this._bAvg[j]! + (1.0 - avgDecay) * this._b[j]!;
    }
  }

  // Compute fused probability upper bounds (Theorem 8.7.1).
  //
  // Given per-signal probability upper bounds, compute the maximum
  // possible fused probability for each candidate.
  computeUpperBounds(
    upperBoundProbs: number[][],
    queryFeatures: number[] | number[][],
    useAveraged: boolean = false,
  ): number[] {
    // Ensure 2D
    const qf2D =
      typeof queryFeatures[0] === "number"
        ? [queryFeatures as number[]]
        : (queryFeatures as number[][]);

    const ubClamped = upperBoundProbs.map((row) => clampProbability(row));

    const w = this._computeWeights(qf2D, useAveraged);
    let x = ubClamped.map(
      (row) => logit(row) as number[],
    );
    if (this._normalize) {
      x = AttentionLogOddsWeights._normalizeLogits(x);
    }
    const n = this._nSignals;
    const scale = n ** this._alpha;
    const results: number[] = [];
    for (let i = 0; i < x.length; i++) {
      const wi = w[Math.min(i, w.length - 1)]!;
      let weightedSum = 0;
      for (let j = 0; j < n; j++) {
        weightedSum += wi[j]! * x[i]![j]!;
      }
      let lWeighted = scale * weightedSum;
      if (this._logitBaseRate !== null) {
        lWeighted += this._logitBaseRate;
      }
      results.push(sigmoid(lWeighted) as number);
    }
    return results;
  }

  // Prune candidates whose upper bound is below threshold (Theorem 8.7.1).
  //
  // Returns surviving indices and their fused probabilities.
  prune(
    probs: number[][],
    queryFeatures: number[] | number[][],
    threshold: number,
    upperBoundProbs?: number[][],
    useAveraged: boolean = false,
  ): { survivingIndices: number[]; fusedProbabilities: number[] } {
    const ubProbs = upperBoundProbs !== undefined ? upperBoundProbs : probs;
    const upperBounds = this.computeUpperBounds(
      ubProbs,
      queryFeatures,
      useAveraged,
    );

    const survivingIndices: number[] = [];
    for (let i = 0; i < upperBounds.length; i++) {
      if (upperBounds[i]! >= threshold) {
        survivingIndices.push(i);
      }
    }

    if (survivingIndices.length === 0) {
      return { survivingIndices: [], fusedProbabilities: [] };
    }

    // Ensure 2D query features
    const qf2D =
      typeof queryFeatures[0] === "number"
        ? [queryFeatures as number[]]
        : (queryFeatures as number[][]);

    const survProbs = survivingIndices.map((i) => probs[i]!);
    const survQF =
      qf2D.length > 1
        ? survivingIndices.map((i) => qf2D[i]!)
        : qf2D;

    const fusedProbabilities = this.combine(
      survProbs,
      survQF,
      useAveraged,
    ) as number[];

    return { survivingIndices, fusedProbabilities };
  }
}

// Multi-head attention fusion (Paper 2, Remark 8.6, Corollary 8.7.2).
//
// Creates multiple independent AttentionLogOddsWeights heads, each
// initialized with a different random seed.  At inference time, each
// head produces fused log-odds independently, and the results are
// combined by averaging log-odds across heads before converting back
// to probability via sigmoid.
export class MultiHeadAttentionLogOddsWeights {
  private _nHeads: number;
  private _heads: AttentionLogOddsWeights[];

  constructor(
    nHeads: number,
    nSignals: number,
    nQueryFeatures: number,
    alpha: number | "auto" = 0.5,
    normalize: boolean = false,
  ) {
    if (nHeads < 1) {
      throw new Error(`n_heads must be >= 1, got ${nHeads}`);
    }
    this._nHeads = nHeads;
    this._heads = [];
    for (let h = 0; h < nHeads; h++) {
      this._heads.push(
        new AttentionLogOddsWeights(
          nSignals,
          nQueryFeatures,
          alpha,
          normalize,
          h,
        ),
      );
    }
  }

  get nHeads(): number {
    return this._nHeads;
  }

  get heads(): AttentionLogOddsWeights[] {
    return [...this._heads];
  }

  // Combine probability signals via multi-head attention fusion.
  //
  // Each head produces fused log-odds independently.  The final
  // result averages the log-odds across heads and applies sigmoid.
  combine(
    probs: number[],
    queryFeatures: number[],
    useAveraged?: boolean,
  ): number;
  combine(
    probs: number[][],
    queryFeatures: number[] | number[][],
    useAveraged?: boolean,
  ): number[];
  combine(
    probs: number[] | number[][],
    queryFeatures: number[] | number[][],
    useAveraged: boolean = false,
  ): number | number[] {
    const isSingleSample = !is2D(probs as number[] | number[][]);

    const headResults: number[][] = [];
    for (const head of this._heads) {
      let r: number | number[];
      if (isSingleSample) {
        r = head.combine(
          probs as number[],
          queryFeatures as number[],
          useAveraged,
        );
        headResults.push([r]);
      } else {
        r = head.combine(
          probs as number[][],
          queryFeatures as number[][],
          useAveraged,
        );
        headResults.push(r as number[]);
      }
    }

    // Average in log-odds space, then sigmoid
    const nSamples = headResults[0]!.length;
    const avgLogits: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      let sum = 0;
      for (let h = 0; h < this._nHeads; h++) {
        const clamped = clampProbability(headResults[h]![i]!) as number;
        sum += logit(clamped) as number;
      }
      avgLogits.push(sum / this._nHeads);
    }

    const result = avgLogits.map((l) => sigmoid(l) as number);

    if (isSingleSample) {
      return result[0]!;
    }
    return result;
  }

  // Train all heads on the same data.
  //
  // Different random initializations lead to different learned
  // solutions, providing diversity across heads.
  fit(
    probs: number[][],
    labels: number[],
    queryFeatures: number[][],
    options: {
      queryIds?: number[];
      learningRate?: number;
      maxIterations?: number;
      tolerance?: number;
    } = {},
  ): void {
    for (const head of this._heads) {
      head.fit(probs, labels, queryFeatures, options);
    }
  }

  // Online update for all heads.
  update(
    probs: number[] | number[][],
    label: number | number[],
    queryFeatures: number[] | number[][],
    options: {
      learningRate?: number;
      momentum?: number;
      decayTau?: number;
      maxGradNorm?: number;
      avgDecay?: number;
    } = {},
  ): void {
    for (const head of this._heads) {
      head.update(probs, label, queryFeatures, options);
    }
  }

  // Compute fused upper bounds across heads (Corollary 8.7.2).
  //
  // Each head computes its upper bound independently.  The final
  // upper bound averages the per-head upper bound log-odds and
  // applies sigmoid.
  computeUpperBounds(
    upperBoundProbs: number[][],
    queryFeatures: number[] | number[][],
    useAveraged: boolean = false,
  ): number[] {
    const headBounds: number[][] = [];
    for (const head of this._heads) {
      headBounds.push(
        head.computeUpperBounds(
          upperBoundProbs,
          queryFeatures,
          useAveraged,
        ),
      );
    }

    const nSamples = headBounds[0]!.length;
    const results: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      let sum = 0;
      for (let h = 0; h < this._nHeads; h++) {
        const clamped = clampProbability(headBounds[h]![i]!) as number;
        sum += logit(clamped) as number;
      }
      results.push(sigmoid(sum / this._nHeads) as number);
    }
    return results;
  }

  // Prune candidates using multi-head upper bounds (Corollary 8.7.2).
  prune(
    probs: number[][],
    queryFeatures: number[] | number[][],
    threshold: number,
    upperBoundProbs?: number[][],
    useAveraged: boolean = false,
  ): { survivingIndices: number[]; fusedProbabilities: number[] } {
    const ubProbs = upperBoundProbs !== undefined ? upperBoundProbs : probs;
    const upperBounds = this.computeUpperBounds(
      ubProbs,
      queryFeatures,
      useAveraged,
    );

    const survivingIndices: number[] = [];
    for (let i = 0; i < upperBounds.length; i++) {
      if (upperBounds[i]! >= threshold) {
        survivingIndices.push(i);
      }
    }

    if (survivingIndices.length === 0) {
      return { survivingIndices: [], fusedProbabilities: [] };
    }

    const survProbs = survivingIndices.map((i) => probs[i]!);

    // Ensure 2D query features
    const qf2D =
      typeof queryFeatures[0] === "number"
        ? [queryFeatures as number[]]
        : (queryFeatures as number[][]);
    const survQF =
      qf2D.length > 1
        ? survivingIndices.map((i) => qf2D[i]!)
        : qf2D;

    const fusedProbabilities = this.combine(
      survProbs,
      survQF,
      useAveraged,
    ) as number[];

    return { survivingIndices, fusedProbabilities };
  }
}
