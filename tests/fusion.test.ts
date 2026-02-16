//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

import { describe, expect, it } from "vitest";

import { logOddsConjunction, probAnd, probOr } from "../src/fusion.js";

describe("probAnd", () => {
  it("computes product for two values", () => {
    expect(probAnd([0.8, 0.9])).toBeCloseTo(0.72);
  });

  it("returns ~1.0 for all ones", () => {
    expect(probAnd([1.0, 1.0, 1.0])).toBeCloseTo(1.0, 9);
  });

  it("returns ~0 when containing zero", () => {
    expect(probAnd([0.5, 0.0])).toBeCloseTo(0.0, 5);
  });

  it("passes through single value", () => {
    expect(probAnd([0.7])).toBeCloseTo(0.7);
  });

  it("demonstrates shrinkage", () => {
    const result = probAnd([0.9, 0.9]);
    expect(result).toBeCloseTo(0.81);
    expect(result).toBeLessThan(0.9);
  });

  it("handles batched input", () => {
    const result = probAnd([
      [0.8, 0.9],
      [0.5, 0.5],
    ]);
    expect(result[0]).toBeCloseTo(0.72);
    expect(result[1]).toBeCloseTo(0.25);
  });
});

describe("probOr", () => {
  it("computes OR for two values", () => {
    expect(probOr([0.8, 0.9])).toBeCloseTo(0.98);
  });

  it("returns ~0 for all zeros", () => {
    expect(probOr([0.0, 0.0])).toBeCloseTo(0.0, 5);
  });

  it("returns ~1 when containing one", () => {
    expect(probOr([0.5, 1.0])).toBeCloseTo(1.0, 9);
  });

  it("passes through single value", () => {
    expect(probOr([0.7])).toBeCloseTo(0.7);
  });

  it("matches complement rule P(A or B) = 1 - (1-A)(1-B)", () => {
    const a = 0.6;
    const b = 0.7;
    const expected = 1.0 - (1.0 - a) * (1.0 - b);
    expect(probOr([a, b])).toBeCloseTo(expected);
  });

  it("handles batched input", () => {
    const result = probOr([
      [0.8, 0.9],
      [0.5, 0.5],
    ]);
    expect(result[0]).toBeCloseTo(0.98);
    expect(result[1]).toBeCloseTo(0.75);
  });
});

describe("logOddsConjunction", () => {
  it("amplifies agreeing high probabilities", () => {
    const result = logOddsConjunction([0.9, 0.9]);
    expect(result).toBeCloseTo(0.927, 1);
    expect(result).toBeGreaterThan(0.9);
  });

  it("handles moderate agreement", () => {
    const result = logOddsConjunction([0.7, 0.7]);
    expect(result).toBeCloseTo(0.77, 1);
    expect(result).toBeGreaterThan(0.7);
  });

  it("moderates disagreement", () => {
    const result = logOddsConjunction([0.7, 0.3]);
    expect(result).toBeCloseTo(0.54, 1);
    expect(result).toBeGreaterThan(0.45);
    expect(result).toBeLessThan(0.65);
  });

  it("handles agreeing low probabilities", () => {
    const result = logOddsConjunction([0.3, 0.3]);
    expect(result).toBeCloseTo(0.38, 1);
    expect(result).toBeGreaterThan(probAnd([0.3, 0.3]) as number);
  });

  it("preserves near-0.5 for irrelevant signals", () => {
    const result = logOddsConjunction([0.5, 0.5]);
    expect(Math.abs(result - 0.5)).toBeLessThan(0.1);
  });

  it("passes through single signal with alpha=0", () => {
    const result = logOddsConjunction([0.8], 0.0);
    expect(result).toBeCloseTo(0.8, 1);
  });

  it("stays within bounds (0, 1)", () => {
    const testCases = [
      [0.01, 0.01],
      [0.99, 0.99],
      [0.01, 0.99],
      [0.5, 0.5, 0.5, 0.5],
    ];
    for (const probs of testCases) {
      const result = logOddsConjunction(probs);
      expect(result).toBeGreaterThan(0);
      expect(result).toBeLessThan(1);
    }
  });

  it("amplifies with more agreeing signals", () => {
    const two = logOddsConjunction([0.8, 0.8]) as number;
    const three = logOddsConjunction([0.8, 0.8, 0.8]) as number;
    expect(three).toBeGreaterThan(two);
  });

  it("gives stronger bonus with higher alpha", () => {
    const probs = [0.8, 0.8];
    const lowAlpha = logOddsConjunction(probs, 0.1) as number;
    const highAlpha = logOddsConjunction(probs, 1.0) as number;
    expect(highAlpha).toBeGreaterThan(lowAlpha);
  });

  it("handles batched input", () => {
    const result = logOddsConjunction([
      [0.9, 0.9],
      [0.3, 0.3],
    ]);
    expect(result).toHaveLength(2);
    expect(result[0]!).toBeGreaterThan(0.9);
    expect(result[1]!).toBeLessThan(0.5);
  });
});
