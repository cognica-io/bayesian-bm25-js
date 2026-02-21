//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

// Tests for WAND upper bound (Theorem 6.1.2).

import { describe, expect, it } from "vitest";

import { BayesianProbabilityTransform } from "../src/probability.js";

// Simple LCG for reproducibility
function makeLCG(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

describe("WANDUpperBound", () => {
  it("upper bound exceeds actual probability for all tf/ratio combos", () => {
    const t = new BayesianProbabilityTransform(1.5, 2.0);
    const bm25Score = 3.0;
    const upper = t.wandUpperBound(bm25Score) as number;

    for (const tf of [0, 1, 5, 10, 50]) {
      for (const ratio of [0.1, 0.5, 1.0, 2.0, 5.0]) {
        const actual = t.scoreToProbability(
          bm25Score,
          tf,
          ratio,
        ) as number;
        expect(upper).toBeGreaterThanOrEqual(actual);
      }
    }
  });

  it("is monotonic in BM25 upper bound", () => {
    const t = new BayesianProbabilityTransform(1.0, 1.0);
    const bounds = [1.0, 2.0, 3.0, 5.0, 10.0];
    const upperBounds = t.wandUpperBound(bounds) as number[];
    for (let i = 1; i < upperBounds.length; i++) {
      expect(upperBounds[i]!).toBeGreaterThan(upperBounds[i - 1]!);
    }
  });

  it("output is always in (0, 1)", () => {
    const t = new BayesianProbabilityTransform(2.0, 0.5);
    const bounds = [0.01, 0.5, 1.0, 5.0, 100.0];
    const upperBounds = t.wandUpperBound(bounds) as number[];
    for (const ub of upperBounds) {
      expect(ub).toBeGreaterThan(0);
      expect(ub).toBeLessThan(1);
    }
  });

  it("works without baseRate", () => {
    const t = new BayesianProbabilityTransform(1.0, 1.0, null);
    const result = t.wandUpperBound(5.0) as number;
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThan(1);
    // With high BM25 score and pMax=0.9, upper bound should be high
    expect(result).toBeGreaterThan(0.5);
  });

  it("incorporates baseRate when set", () => {
    const tNone = new BayesianProbabilityTransform(1.0, 1.0, null);
    const tLow = new BayesianProbabilityTransform(1.0, 1.0, 0.01);
    const bound = 5.0;
    const upperNone = tNone.wandUpperBound(bound) as number;
    const upperLow = tLow.wandUpperBound(bound) as number;
    // Low base rate should give lower upper bound
    expect(upperLow).toBeLessThan(upperNone);
  });

  it("scalar input produces scalar output", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    const result = t.wandUpperBound(3.0);
    expect(typeof result).toBe("number");
  });

  it("array input produces array output", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    const result = t.wandUpperBound([1.0, 2.0, 3.0]);
    expect(Array.isArray(result)).toBe(true);
    expect(result).toHaveLength(3);
  });

  it("pruning safety with random documents", () => {
    const rand = makeLCG(42);
    const t = new BayesianProbabilityTransform(1.5, 1.0);

    const bm25UpperBound = 8.0;
    const upper = t.wandUpperBound(bm25UpperBound) as number;

    for (let i = 0; i < 500; i++) {
      const score = rand() * bm25UpperBound;
      const tf = rand() * 20;
      const ratio = 0.1 + rand() * 2.9;

      const actual = t.scoreToProbability(score, tf, ratio) as number;
      expect(upper).toBeGreaterThanOrEqual(actual - 1e-10);
    }
  });

  it("pruning safety with baseRate", () => {
    const rand = makeLCG(123);
    const t = new BayesianProbabilityTransform(2.0, 0.5, 0.05);

    const bm25UpperBound = 5.0;
    const upper = t.wandUpperBound(bm25UpperBound) as number;

    for (let i = 0; i < 500; i++) {
      const score = rand() * bm25UpperBound;
      const tf = rand() * 20;
      const ratio = 0.1 + rand() * 2.9;

      const actual = t.scoreToProbability(score, tf, ratio) as number;
      expect(upper).toBeGreaterThanOrEqual(actual - 1e-10);
    }
  });
});
