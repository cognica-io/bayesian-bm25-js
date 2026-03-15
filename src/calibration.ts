//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

// Neural score calibration for integrating external model scores.
//
// Provides sigmoid (Platt) and isotonic (PAVA) calibrators that convert
// raw scores from neural models or other sources into calibrated
// probabilities suitable for Bayesian fusion via logOddsConjunction.
//
// Paper ref: Paper 1, Section 12.2 #5; Paper 2, Section 5.1

import { clampProbability, sigmoid } from "./probability.js";

// Sigmoid calibration: P = sigmoid(a * score + b).
//
// Learns parameters a and b via BCE gradient descent so that
// sigmoid(a * score + b) produces well-calibrated probabilities.
export class PlattCalibrator {
  private _a: number;
  private _b: number;

  constructor(a: number = 1.0, b: number = 0.0) {
    this._a = a;
    this._b = b;
  }

  get a(): number {
    return this._a;
  }

  get b(): number {
    return this._b;
  }

  // Learn a and b via gradient descent on BCE loss.
  fit(
    scores: number[],
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

    const m = scores.length;
    let a = this._a;
    let b = this._b;

    for (let iter = 0; iter < maxIterations; iter++) {
      // predicted = clamp(sigmoid(a * score + b))
      const predicted: number[] = [];
      for (let i = 0; i < m; i++) {
        predicted.push(
          clampProbability(sigmoid(a * scores[i]! + b) as number) as number,
        );
      }

      // error = predicted - labels
      let gradA = 0;
      let gradB = 0;
      for (let i = 0; i < m; i++) {
        const error = predicted[i]! - labels[i]!;
        gradA += error * scores[i]!;
        gradB += error;
      }
      gradA /= m;
      gradB /= m;

      const newA = a - learningRate * gradA;
      const newB = b - learningRate * gradB;

      if (Math.abs(newA - a) < tolerance && Math.abs(newB - b) < tolerance) {
        a = newA;
        b = newB;
        break;
      }

      a = newA;
      b = newB;
    }

    this._a = a;
    this._b = b;
  }

  // Apply sigmoid calibration: sigmoid(a * scores + b).
  calibrate(scores: number): number;
  calibrate(scores: number[]): number[];
  calibrate(scores: number | number[]): number | number[] {
    if (Array.isArray(scores)) {
      return scores.map(
        (s) => sigmoid(this._a * s + this._b) as number,
      );
    }
    return sigmoid(this._a * scores + this._b) as number;
  }
}

// Non-parametric monotone calibration via PAVA (Pool Adjacent Violators).
//
// The Pool Adjacent Violators Algorithm produces a monotonically
// non-decreasing mapping from scores to probabilities.  At inference
// time, binary search with linear interpolation maps new scores to
// calibrated probabilities.
export class IsotonicCalibrator {
  private _x: number[] | null = null; // sorted score breakpoints
  private _y: number[] | null = null; // corresponding calibrated values

  // Fit isotonic regression via PAVA.
  fit(scores: number[], labels: number[]): void {
    // Sort by scores
    const indices = Array.from({ length: scores.length }, (_, i) => i);
    indices.sort((a, b) => scores[a]! - scores[b]!);

    const xSorted = indices.map((i) => scores[i]!);
    const ySorted = indices.map((i) => labels[i]!);

    const n = xSorted.length;

    // PAVA: merge adjacent blocks that violate monotonicity
    const blockSums = [...ySorted];
    const blockCounts = new Array(n).fill(1) as number[];
    const blockXSums = [...xSorted];
    let active = Array.from({ length: n }, (_, i) => i);

    let merged = true;
    while (merged) {
      merged = false;
      const newActive: number[] = [active[0]!];
      for (let j = 1; j < active.length; j++) {
        const prev = newActive[newActive.length - 1]!;
        const curr = active[j]!;
        const valPrev = blockSums[prev]! / blockCounts[prev]!;
        const valCurr = blockSums[curr]! / blockCounts[curr]!;
        if (valPrev > valCurr) {
          // Merge curr into prev
          blockSums[prev] = blockSums[prev]! + blockSums[curr]!;
          blockCounts[prev] = blockCounts[prev]! + blockCounts[curr]!;
          blockXSums[prev] = blockXSums[prev]! + blockXSums[curr]!;
          merged = true;
        } else {
          newActive.push(curr);
        }
      }
      active = newActive;
    }

    // Extract breakpoints: mean score and mean label for each block
    this._x = active.map((i) => blockXSums[i]! / blockCounts[i]!);
    this._y = active.map((i) => blockSums[i]! / blockCounts[i]!);
  }

  // Apply isotonic calibration via binary search + linear interpolation.
  calibrate(scores: number): number;
  calibrate(scores: number[]): number[];
  calibrate(scores: number | number[]): number | number[] {
    if (this._x === null || this._y === null) {
      throw new Error("Call fit() before calibrate().");
    }

    if (Array.isArray(scores)) {
      return scores.map((s) => this._calibrateSingle(s));
    }
    return this._calibrateSingle(scores);
  }

  private _calibrateSingle(score: number): number {
    const x = this._x!;
    const y = this._y!;

    // Binary search: find insertion point
    let lo = 0;
    let hi = x.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (x[mid]! < score) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    const idx = lo;

    let result: number;
    if (idx === 0) {
      result = y[0]!;
    } else if (idx >= x.length) {
      result = y[y.length - 1]!;
    } else {
      // Linear interpolation between adjacent breakpoints
      const x0 = x[idx - 1]!;
      const x1 = x[idx]!;
      const y0 = y[idx - 1]!;
      const y1 = y[idx]!;
      if (x1 - x0 < 1e-12) {
        result = (y0 + y1) / 2.0;
      } else {
        const t = (score - x0) / (x1 - x0);
        result = y0 + t * (y1 - y0);
      }
    }

    return clampProbability(result) as number;
  }
}
