//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

import { describe, expect, it } from "vitest";

import { logOddsConjunction } from "../src/fusion.js";

describe("GELU gating", () => {
  it("GELU ordering: for agreeing signals, result_swish < result_gelu < result_relu", () => {
    const probs = [0.8, 0.9, 0.7]; // all > 0.5, positive logits
    const resultSwish = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "swish",
    ) as number;
    const resultGelu = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "gelu",
    ) as number;
    const resultRelu = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "relu",
    ) as number;
    // For positive logits: swish < gelu < relu
    expect(resultSwish).toBeLessThan(resultGelu);
    expect(resultGelu).toBeLessThan(resultRelu);
  });

  it("GELU matches swish_1.702 within 1e-10", () => {
    const probs = [0.9, 0.3, 0.7, 0.2];
    const resultGelu = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "gelu",
    ) as number;
    const resultSwish1702 = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "swish",
      1.702,
    ) as number;
    expect(resultGelu).toBeCloseTo(resultSwish1702, 10);
  });

  it("GELU ignores gatingBeta: same result with different beta values", () => {
    const probs = [0.9, 0.3, 0.7];
    const resultDefault = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "gelu",
    ) as number;
    const resultBeta1 = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "gelu",
      1.0,
    ) as number;
    const resultBeta5 = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "gelu",
      5.0,
    ) as number;
    expect(resultDefault).toBeCloseTo(resultBeta1, 12);
    expect(resultDefault).toBeCloseTo(resultBeta5, 12);
  });

  it("batched input works", () => {
    const probs = [
      [0.9, 0.3],
      [0.8, 0.8],
    ];
    const result = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "gelu",
    ) as number[];
    expect(result).toHaveLength(2);
    for (const r of result) {
      expect(Number.isFinite(r)).toBe(true);
      expect(r).toBeGreaterThan(0);
      expect(r).toBeLessThan(1);
    }
  });

  it("all results in (0, 1)", () => {
    const testCases = [
      [0.01, 0.01],
      [0.99, 0.99],
      [0.01, 0.99],
      [0.5, 0.5, 0.5, 0.5],
      [0.9, 0.3, 0.7],
    ];
    for (const probs of testCases) {
      const result = logOddsConjunction(
        probs,
        undefined,
        undefined,
        "gelu",
      ) as number;
      expect(result).toBeGreaterThan(0);
      expect(result).toBeLessThan(1);
    }
  });

  it("GELU with weights produces valid output", () => {
    const probs = [0.9, 0.3];
    const w = [0.5, 0.5];
    const result = logOddsConjunction(probs, undefined, w, "gelu") as number;
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThan(1);
  });
});

describe("generalized swish beta", () => {
  it("beta=1.0 matches default swish behavior", () => {
    const probs = [0.9, 0.3, 0.7];
    const resultDefault = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "swish",
    ) as number;
    const resultBeta1 = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "swish",
      1.0,
    ) as number;
    expect(resultBeta1).toBeCloseTo(resultDefault, 12);
  });

  it("different beta values produce different results", () => {
    const probs = [0.9, 0.3, 0.7];
    const resultBeta05 = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "swish",
      0.5,
    ) as number;
    const resultBeta1 = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "swish",
      1.0,
    ) as number;
    const resultBeta2 = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "swish",
      2.0,
    ) as number;
    // All three should be valid probabilities
    for (const r of [resultBeta05, resultBeta1, resultBeta2]) {
      expect(r).toBeGreaterThan(0);
      expect(r).toBeLessThan(1);
    }
    // They should differ from each other
    expect(Math.abs(resultBeta05 - resultBeta1)).toBeGreaterThan(1e-6);
    expect(Math.abs(resultBeta1 - resultBeta2)).toBeGreaterThan(1e-6);
  });

  it("large beta approaches ReLU", () => {
    const probs = [0.9, 0.3];
    const resultRelu = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "relu",
    ) as number;
    const resultLargeBeta = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "swish",
      100.0,
    ) as number;
    expect(resultLargeBeta).toBeCloseTo(resultRelu, 1);
  });

  it("swish_1.702 matches GELU", () => {
    const probs = [0.9, 0.3, 0.7];
    const resultSwish = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "swish",
      1.702,
    ) as number;
    const resultGelu = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "gelu",
    ) as number;
    expect(resultSwish).toBeCloseTo(resultGelu, 10);
  });
});
