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
//   Soft gating: logit * sigmoid(logit).
function applyGating(logitValues: number[], gating: string): number[] {
  if (gating === "none") {
    return logitValues;
  }
  if (gating === "relu") {
    return logitValues.map((l) => Math.max(0.0, l));
  }
  if (gating === "swish") {
    return logitValues.map((l) => l * (sigmoid(l) as number));
  }
  throw new Error(
    `gating must be "none", "relu", or "swish", got "${gating}"`,
  );
}

function logOddsConjunctionSingle(
  probs: number[],
  alpha: number,
  gating: string,
): number {
  const clamped = clampProbability(probs);
  const n = clamped.length;

  // Step 1: mean log-odds (Eq. 20)
  const rawLogits = logit(clamped) as number[];
  const gatedLogits = applyGating(rawLogits, gating);
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
): number {
  const clamped = clampProbability(probs);
  const n = clamped.length;
  const rawLogits = logit(clamped) as number[];
  const gatedLogits = applyGating(rawLogits, gating);
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
): number;
export function logOddsConjunction(
  probs: number[][],
  alpha?: number | "auto",
  weights?: number[],
  gating?: string,
): number[];
export function logOddsConjunction(
  probs: number[] | number[][],
  alpha?: number | "auto",
  weights?: number[],
  gating: string = "none",
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
        logOddsWeightedSingle(row, weights, effectiveAlpha, gating),
      );
    }
    return logOddsWeightedSingle(
      probs as number[],
      weights,
      effectiveAlpha,
      gating,
    );
  }

  const effectiveAlpha = resolveAlpha(alpha, 0.5);

  if (is2D(probs)) {
    return probs.map((row) =>
      logOddsConjunctionSingle(row, effectiveAlpha, gating),
    );
  }
  return logOddsConjunctionSingle(
    probs as number[],
    effectiveAlpha,
    gating,
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
  private _logits: number[];
  private _nUpdates: number = 0;
  private _gradLogitsEMA: number[];
  private _weightsAvg: number[];

  constructor(nSignals: number, alpha: number | "auto" = 0.0) {
    if (nSignals < 1) {
      throw new Error(`n_signals must be >= 1, got ${nSignals}`);
    }
    this._nSignals = nSignals;
    this._alpha = resolveAlpha(alpha, 0.0);

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
  combine(probs: number[], useAveraged?: boolean): number;
  combine(probs: number[][], useAveraged?: boolean): number[];
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
  ) {
    if (nSignals < 1) {
      throw new Error(`n_signals must be >= 1, got ${nSignals}`);
    }
    if (nQueryFeatures < 1) {
      throw new Error(
        `n_query_features must be >= 1, got ${nQueryFeatures}`,
      );
    }
    this._nSignals = nSignals;
    this._nQueryFeatures = nQueryFeatures;
    this._alpha = resolveAlpha(alpha, 0.5);

    // Xavier-style initialization scaled for softmax input
    const scale = 1.0 / Math.sqrt(nQueryFeatures);
    const rng = mulberry32(0);
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
      // Single sample
      const wFlat = w[0]!;
      return logOddsConjunction(probs as number[], this._alpha, wFlat);
    }

    // Batched: each row has its own query-dependent weights
    const probsBatch = probs as number[][];
    const results: number[] = [];
    for (let i = 0; i < probsBatch.length; i++) {
      results.push(
        logOddsConjunction(
          probsBatch[i]!,
          this._alpha,
          w[i]!,
        ) as number,
      );
    }
    return results;
  }

  // Batch gradient descent on BCE loss to learn W and b.
  fit(
    probs: number[][],
    labels: number[],
    queryFeatures: number[][],
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

    const m = probs.length;
    const n = this._nSignals;
    const scale = n ** this._alpha;

    // Log-odds of input signals: (m, n)
    const x = probs.map(
      (row) => logit(clampProbability(row)) as number[],
    );

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
      const p = xBarW.map((xb) => sigmoid(scale * xb) as number);
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
    const x = probsBatch.map(
      (row) => logit(clampProbability(row)) as number[],
    );

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

    const p = xBarW.map((xb) => sigmoid(scale * xb) as number);
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
}
