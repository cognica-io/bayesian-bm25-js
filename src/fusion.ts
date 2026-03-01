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

function logOddsConjunctionSingle(probs: number[], alpha: number): number {
  const clamped = clampProbability(probs);
  const n = clamped.length;

  // Step 1: mean log-odds (Eq. 20)
  const logitValues = logit(clamped) as number[];
  let logitSum = 0;
  for (const l of logitValues) {
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
): number {
  const clamped = clampProbability(probs);
  const n = clamped.length;
  const logitValues = logit(clamped) as number[];
  let weightedSum = 0;
  for (let i = 0; i < logitValues.length; i++) {
    weightedSum += weights[i]! * logitValues[i]!;
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
// sigma(sum(w_i * logit(P_i))) where sum(w_i) = 1 and w_i >= 0.
// The alpha parameter is ignored in weighted mode.
//
// The multiplicative formulation (rather than additive) preserves the
// sign of evidence (Theorem 4.2.2), preventing accidental inversion
// of irrelevance signals (Remark 4.2.4).
export function logOddsConjunction(
  probs: number[],
  alpha?: number,
  weights?: number[],
): number;
export function logOddsConjunction(
  probs: number[][],
  alpha?: number,
  weights?: number[],
): number[];
export function logOddsConjunction(
  probs: number[] | number[][],
  alpha?: number,
  weights?: number[],
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

    // Default alpha for weighted mode is 0.0 (no confidence scaling)
    const effectiveAlpha = alpha ?? 0.0;

    if (is2D(probs)) {
      return probs.map((row) =>
        logOddsWeightedSingle(row, weights, effectiveAlpha),
      );
    }
    return logOddsWeightedSingle(
      probs as number[],
      weights,
      effectiveAlpha,
    );
  }

  // Default alpha for unweighted mode is 0.5
  const effectiveAlpha = alpha ?? 0.5;

  if (is2D(probs)) {
    return probs.map((row) =>
      logOddsConjunctionSingle(row, effectiveAlpha),
    );
  }
  return logOddsConjunctionSingle(probs as number[], effectiveAlpha);
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
  private _logits: number[];
  private _nUpdates: number = 0;
  private _gradLogitsEMA: number[];
  private _weightsAvg: number[];

  constructor(nSignals: number, alpha: number = 0.0) {
    if (nSignals < 1) {
      throw new Error(`n_signals must be >= 1, got ${nSignals}`);
    }
    this._nSignals = nSignals;
    this._alpha = alpha;

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

  // Current weights: softmax of internal logits.
  get weights(): number[] {
    return softmax(this._logits);
  }

  // Polyak-averaged weights for stable inference.
  get averagedWeights(): number[] {
    return [...this._weightsAvg];
  }

  // Combine probability signals via weighted log-odds conjunction.
  combine(probs: number[], useAveraged: boolean = false): number;
  combine(probs: number[][], useAveraged: boolean = false): number[];
  combine(
    probs: number[] | number[][],
    useAveraged: boolean = false,
  ): number | number[] {
    const w = useAveraged ? this._weightsAvg : this.weights;
    return logOddsConjunction(probs as number[], this._alpha, w);
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
        const p = sigmoid(scale * xBarW) as number;
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
      const p = sigmoid(scale * xBarW) as number;
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
