//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

// Probabilistic score combination functions.
//
// Implements AND, OR, and log-odds conjunction for combining multiple
// probability estimates. The log-odds conjunction (from "From Bayesian
// Inference to Neural Computation") resolves the shrinkage problem of
// naive probabilistic AND by using multiplicative confidence scaling
// in log-odds space.

import { clampProbability, logit, sigmoid } from "./probability.js";

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

// Log-odds conjunction with multiplicative confidence scaling (Paper 2, Section 4).
//
// Resolves the shrinkage problem of naive probabilistic AND by:
//   1. Computing the mean log-odds (Eq. 20)
//   2. Multiplicative confidence scaling by n^alpha (Eq. 23)
//   3. Converting back to probability via sigmoid (Eq. 26)
//
// The multiplicative formulation (rather than additive) preserves the
// sign of evidence (Theorem 4.2.2), preventing accidental inversion
// of irrelevance signals (Remark 4.2.4).
export function logOddsConjunction(
  probs: number[],
  alpha?: number,
): number;
export function logOddsConjunction(
  probs: number[][],
  alpha?: number,
): number[];
export function logOddsConjunction(
  probs: number[] | number[][],
  alpha: number = 0.5,
): number | number[] {
  if (probs.length === 0) return 0;
  if (is2D(probs)) {
    return probs.map((row) => logOddsConjunctionSingle(row, alpha));
  }
  return logOddsConjunctionSingle(probs as number[], alpha);
}
