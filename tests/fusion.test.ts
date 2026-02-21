//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

import { describe, expect, it } from "vitest";

import {
  cosineToProbability,
  logOddsConjunction,
  probAnd,
  probOr,
} from "../src/fusion.js";

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
    // l_bar=logit(0.9)=2.197, l_adjusted=2.197*sqrt(2)=3.107, sigmoid(3.107)=0.957
    const result = logOddsConjunction([0.9, 0.9]);
    expect(result).toBeCloseTo(0.957, 1);
    expect(result).toBeGreaterThan(0.9);
  });

  it("handles moderate agreement", () => {
    // l_bar=logit(0.7)=0.847, l_adjusted=0.847*sqrt(2)=1.198, sigmoid(1.198)=0.768
    const result = logOddsConjunction([0.7, 0.7]);
    expect(result).toBeCloseTo(0.77, 1);
    expect(result).toBeGreaterThan(0.7);
  });

  it("moderates disagreement", () => {
    // logit(0.7)=0.847, logit(0.3)=-0.847, l_bar=0, sigmoid(0)=0.5
    const result = logOddsConjunction([0.7, 0.3]);
    expect(result).toBeCloseTo(0.5, 1);
    // Symmetric logits cancel to exact uncertainty
    expect(result).toBeGreaterThan(0.49);
    expect(result).toBeLessThan(0.51);
  });

  it("handles agreeing low probabilities", () => {
    // l_bar=logit(0.3)=-0.847, l_adjusted=-0.847*sqrt(2)=-1.198, sigmoid(-1.198)=0.232
    const result = logOddsConjunction([0.3, 0.3]);
    expect(result).toBeCloseTo(0.23, 1);
    expect(result).toBeGreaterThan(probAnd([0.3, 0.3]) as number);
  });

  it("preserves near-0.5 for irrelevant signals", () => {
    // logit(0.5)=0, l_bar=0, l_adjusted=0, sigmoid(0)=0.5
    const result = logOddsConjunction([0.5, 0.5]);
    expect(result).toBeCloseTo(0.5, 1);
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

describe("cosineToProbability", () => {
  it("max similarity (1.0) -> ~1.0", () => {
    expect(cosineToProbability(1.0)).toBeCloseTo(1.0, 5);
  });

  it("min similarity (-1.0) -> ~0.0", () => {
    expect(cosineToProbability(-1.0)).toBeCloseTo(0.0, 5);
  });

  it("zero similarity -> 0.5", () => {
    expect(cosineToProbability(0.0)).toBeCloseTo(0.5);
  });

  it("output is always in (0, 1)", () => {
    for (const score of [-1.0, -0.5, 0.0, 0.5, 1.0]) {
      const result = cosineToProbability(score) as number;
      expect(result).toBeGreaterThan(0);
      expect(result).toBeLessThan(1);
    }
  });

  it("handles array input", () => {
    const scores = [-1.0, -0.5, 0.0, 0.5, 1.0];
    const result = cosineToProbability(scores) as number[];
    expect(result).toHaveLength(5);
    expect(result[0]!).toBeCloseTo(0.0, 5);
    expect(result[1]!).toBeCloseTo(0.25, 5);
    expect(result[2]!).toBeCloseTo(0.5, 5);
    expect(result[3]!).toBeCloseTo(0.75, 5);
    expect(result[4]!).toBeCloseTo(1.0, 5);
  });

  it("is monotonically increasing", () => {
    const scores: number[] = [];
    for (let i = 0; i < 20; i++) {
      scores.push(-1.0 + (2.0 * i) / 19);
    }
    const result = cosineToProbability(scores) as number[];
    for (let i = 1; i < result.length; i++) {
      expect(result[i]!).toBeGreaterThan(result[i - 1]!);
    }
  });
});

describe("weightedLogOddsConjunction", () => {
  it("uniform weights match unweighted alpha=0", () => {
    const probs = [0.7, 0.8];
    const uniformW = [0.5, 0.5];
    const weighted = logOddsConjunction(probs, 0.5, uniformW) as number;
    const unweightedAlpha0 = logOddsConjunction(probs, 0.0) as number;
    expect(weighted).toBeCloseTo(unweightedAlpha0, 10);
  });

  it("higher weight on high probability increases result", () => {
    const probs = [0.9, 0.3];
    const wHighFirst = [0.8, 0.2];
    const wHighSecond = [0.2, 0.8];
    const resultHighFirst = logOddsConjunction(
      probs,
      0.5,
      wHighFirst,
    ) as number;
    const resultHighSecond = logOddsConjunction(
      probs,
      0.5,
      wHighSecond,
    ) as number;
    expect(resultHighFirst).toBeGreaterThan(resultHighSecond);
  });

  it("higher weight on low probability decreases result", () => {
    const probs = [0.9, 0.2];
    const wEqual = [0.5, 0.5];
    const wLowHeavy = [0.2, 0.8];
    const resultEqual = logOddsConjunction(probs, 0.5, wEqual) as number;
    const resultLow = logOddsConjunction(probs, 0.5, wLowHeavy) as number;
    expect(resultLow).toBeLessThan(resultEqual);
  });

  it("weights must sum to 1", () => {
    expect(() => {
      logOddsConjunction([0.5, 0.5], 0.5, [0.3, 0.3]);
    }).toThrow(/weights must sum to 1/);
  });

  it("weights must be non-negative", () => {
    expect(() => {
      logOddsConjunction([0.5, 0.5], 0.5, [-0.5, 1.5]);
    }).toThrow(/weights must be non-negative/);
  });

  it("batched with weights", () => {
    const probs = [
      [0.9, 0.1],
      [0.8, 0.8],
    ];
    const weights = [0.7, 0.3];
    const result = logOddsConjunction(probs, 0.5, weights) as number[];
    expect(result).toHaveLength(2);
    // First batch: high-prob signal weighted heavily -> > 0.5
    expect(result[0]!).toBeGreaterThan(0.5);
    // Second batch: both signals agree at 0.8 -> > 0.5
    expect(result[1]!).toBeGreaterThan(0.5);
  });

  it("single signal with weight=1 passes through", () => {
    const result = logOddsConjunction([0.8], 0.5, [1.0]) as number;
    expect(result).toBeCloseTo(0.8, 6);
  });

  it("three signals with non-uniform weights", () => {
    const probs = [0.9, 0.9, 0.1];
    const w = [0.4, 0.4, 0.2];
    const result = logOddsConjunction(probs, 0.5, w) as number;
    // Two high signals dominate -> result > 0.5
    expect(result).toBeGreaterThan(0.5);
  });
});
