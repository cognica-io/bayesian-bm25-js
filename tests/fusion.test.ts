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
  probNot,
  probOr,
  LearnableLogOddsWeights,
  AttentionLogOddsWeights,
} from "../src/fusion.js";
import { logit, sigmoid, clampProbability } from "../src/probability.js";

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
  it("uniform weights match unweighted with same alpha", () => {
    const probs = [0.7, 0.8];
    const uniformW = [0.5, 0.5];
    // Uniform weights w_i=1/n with explicit alpha must match unweighted
    // with the same alpha (Theorem 8.3):
    //   sigma(n^alpha * sum(1/n * logit(P_i))) = sigma(n^alpha * mean(logit(P_i)))
    for (const alpha of [0.0, 0.5, 1.0, 2.0]) {
      const weighted = logOddsConjunction(probs, alpha, uniformW) as number;
      const unweighted = logOddsConjunction(probs, alpha) as number;
      expect(weighted).toBeCloseTo(unweighted, 10);
    }
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

describe("probNot", () => {
  it("NOT 0.8 = 0.2", () => {
    expect(probNot(0.8)).toBeCloseTo(0.2);
  });

  it("NOT 0.5 = 0.5 (uncertainty is self-complementary)", () => {
    expect(probNot(0.5)).toBeCloseTo(0.5);
  });

  it("NOT of near-zero -> near-one", () => {
    expect(probNot(0.01)).toBeCloseTo(0.99);
  });

  it("NOT of near-one -> near-zero", () => {
    expect(probNot(0.99)).toBeCloseTo(0.01);
  });

  it("NOT(NOT(p)) = p (double negation / involution)", () => {
    for (const p of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      expect(probNot(probNot(p) as number)).toBeCloseTo(p, 9);
    }
  });

  it("output is always in (0, 1) with clamping", () => {
    for (const p of [0.0, 0.5, 1.0]) {
      const result = probNot(p) as number;
      expect(result).toBeGreaterThan(0);
      expect(result).toBeLessThan(1);
    }
  });

  it("handles array input", () => {
    const probs = [0.1, 0.3, 0.5, 0.7, 0.9];
    const result = probNot(probs) as number[];
    expect(result).toHaveLength(5);
    expect(result[0]!).toBeCloseTo(0.9);
    expect(result[1]!).toBeCloseTo(0.7);
    expect(result[2]!).toBeCloseTo(0.5);
    expect(result[3]!).toBeCloseTo(0.3);
    expect(result[4]!).toBeCloseTo(0.1);
  });

  it("logit(NOT p) = -logit(p) (log-odds negation)", () => {
    for (const p of [0.2, 0.4, 0.6, 0.8]) {
      expect(logit(probNot(p) as number)).toBeCloseTo(
        -(logit(p) as number),
        9,
      );
    }
  });

  it("De Morgan: NOT(A AND B) = OR(NOT A, NOT B)", () => {
    const a = 0.7;
    const b = 0.8;
    const lhs = probNot(probAnd([a, b]) as number) as number;
    const rhs = probOr([
      probNot(a) as number,
      probNot(b) as number,
    ]) as number;
    expect(lhs).toBeCloseTo(rhs, 9);
  });

  it("De Morgan: NOT(A OR B) = AND(NOT A, NOT B)", () => {
    const a = 0.7;
    const b = 0.8;
    const lhs = probNot(probOr([a, b]) as number) as number;
    const rhs = probAnd([
      probNot(a) as number,
      probNot(b) as number,
    ]) as number;
    expect(lhs).toBeCloseTo(rhs, 9);
  });
});

describe("LearnableLogOddsWeights", () => {
  it("initial weights are uniform 1/n (Naive Bayes init)", () => {
    for (const n of [1, 2, 3, 5, 10]) {
      const learner = new LearnableLogOddsWeights(n);
      const expected = new Array(n).fill(1.0 / n);
      for (let i = 0; i < n; i++) {
        expect(learner.weights[i]).toBeCloseTo(expected[i]!, 15);
      }
    }
  });

  it("n_signals < 1 raises error", () => {
    expect(() => new LearnableLogOddsWeights(0)).toThrow(
      /n_signals must be >= 1/,
    );
    expect(() => new LearnableLogOddsWeights(-1)).toThrow(
      /n_signals must be >= 1/,
    );
  });

  it("combine matches logOddsConjunction at init", () => {
    const learner = new LearnableLogOddsWeights(3, 0.0);
    const probs = [0.7, 0.8, 0.6];
    const result = learner.combine(probs) as number;
    const expected = logOddsConjunction(
      probs,
      0.0,
      [1 / 3, 1 / 3, 1 / 3],
    ) as number;
    expect(result).toBeCloseTo(expected, 10);
  });

  it("combine handles batched inputs", () => {
    const learner = new LearnableLogOddsWeights(2, 0.0);
    const probs = [
      [0.8, 0.9],
      [0.3, 0.7],
    ];
    const result = learner.combine(probs) as number[];
    expect(result).toHaveLength(2);
    // Each row should match independent call
    for (let i = 0; i < 2; i++) {
      const expected = learner.combine(probs[i]!) as number;
      expect(result[i]!).toBeCloseTo(expected, 10);
    }
  });

  it("combine with useAveraged=true uses Polyak-averaged weights", () => {
    const learner = new LearnableLogOddsWeights(2, 0.0);
    const probs = [0.7, 0.8];
    // At init, averaged weights = raw weights = uniform
    const resultRaw = learner.combine(probs, false) as number;
    const resultAvg = learner.combine(probs, true) as number;
    expect(resultRaw).toBeCloseTo(resultAvg, 10);
  });

  it("weights always sum to 1 and are non-negative", () => {
    const learner = new LearnableLogOddsWeights(4, 0.0);
    // Manually perturb logits by accessing private field
    (learner as any)._logits = [1.0, -2.0, 0.5, 3.0];
    const w = learner.weights;
    for (const wi of w) {
      expect(wi).toBeGreaterThanOrEqual(0);
    }
    let sum = 0;
    for (const wi of w) {
      sum += wi;
    }
    expect(sum).toBeCloseTo(1.0, 10);
  });

  it("fit learns reliable signal over noisy one", () => {
    // Simple seeded PRNG for reproducibility
    let seed = 42;
    const rng = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 0x100000000;
    };

    const m = 500;
    const labels: number[] = [];
    const signal0: number[] = [];
    const signal1: number[] = [];

    for (let i = 0; i < m; i++) {
      const label = rng() > 0.5 ? 1 : 0;
      labels.push(label);
      // Signal 0: reliable
      signal0.push(label === 1 ? 0.85 : 0.15);
      // Signal 1: noisy
      signal1.push(0.3 + rng() * 0.4);
    }

    const probs = signal0.map((s0, i) => [s0, signal1[i]!]);

    const learner = new LearnableLogOddsWeights(2, 0.0);
    learner.fit(probs, labels, {
      learningRate: 0.1,
      maxIterations: 2000,
    });

    // Reliable signal should get higher weight
    expect(learner.weights[0]!).toBeGreaterThan(learner.weights[1]!);
    expect(learner.weights[0]!).toBeGreaterThan(0.6);
  });

  it("fit raises error on dimension mismatch", () => {
    const learner = new LearnableLogOddsWeights(3, 0.0);
    const probs = [[0.5, 0.5]]; // 2 signals, not 3
    const labels = [1.0];
    expect(() => learner.fit(probs, labels)).toThrow(/n_signals/);
  });

  it("fit resets online state", () => {
    const learner = new LearnableLogOddsWeights(2, 0.0);
    // Simulate some online updates
    (learner as any)._nUpdates = 10;
    (learner as any)._gradLogitsEMA = [0.5, -0.3];

    const probs = [
      [0.8, 0.2],
      [0.7, 0.3],
    ];
    const labels = [1.0, 0.0];
    learner.fit(probs, labels);

    expect((learner as any)._nUpdates).toBe(0);
    expect((learner as any)._gradLogitsEMA[0]).toBeCloseTo(0.0);
    expect((learner as any)._gradLogitsEMA[1]).toBeCloseTo(0.0);
    for (let i = 0; i < 2; i++) {
      expect(learner.averagedWeights[i]!).toBeCloseTo(
        learner.weights[i]!,
        10,
      );
    }
  });

  it("update moves toward informative signal", () => {
    let seed = 123;
    const rng = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 0x100000000;
    };

    const learner = new LearnableLogOddsWeights(2, 0.0);

    for (let i = 0; i < 200; i++) {
      const label = rng() > 0.5 ? 1.0 : 0.0;
      // Signal 0: informative
      const p0 = label === 1 ? 0.9 : 0.1;
      // Signal 1: noise
      const p1 = 0.3 + rng() * 0.4;
      learner.update([p0, p1], label, { learningRate: 0.05 });
    }

    // Signal 0 should have higher weight
    expect(learner.weights[0]!).toBeGreaterThan(learner.weights[1]!);
  });

  it("update raises error on dimension mismatch", () => {
    const learner = new LearnableLogOddsWeights(2, 0.0);
    expect(() => learner.update([0.5, 0.5, 0.5], 1.0)).toThrow(
      /n_signals/,
    );
  });

  it("update accepts mini-batches", () => {
    const learner = new LearnableLogOddsWeights(2, 0.0);
    const probs = [
      [0.8, 0.2],
      [0.3, 0.7],
    ];
    const labels = [1.0, 0.0];
    // Should not throw
    learner.update(probs, labels);
    expect((learner as any)._nUpdates).toBe(1);
  });

  it("averaged weights are smoother than raw weights", () => {
    let seed = 99;
    const rng = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 0x100000000;
    };

    const learner = new LearnableLogOddsWeights(2, 0.0);
    const rawHistory: number[] = [];
    const avgHistory: number[] = [];

    for (let i = 0; i < 100; i++) {
      const label = rng() > 0.5 ? 1.0 : 0.0;
      const p0 = label === 1 ? 0.8 : 0.2;
      const p1 = 0.2 + rng() * 0.6;
      learner.update([p0, p1], label, { learningRate: 0.1 });
      rawHistory.push(learner.weights[0]!);
      avgHistory.push(learner.averagedWeights[0]!);
    }

    // Variance of averaged should be lower
    const variance = (arr: number[]) => {
      const slice = arr.slice(-50);
      const mean = slice.reduce((a, b) => a + b) / slice.length;
      return slice.reduce((s, v) => s + (v - mean) ** 2, 0) / slice.length;
    };
    expect(variance(avgHistory)).toBeLessThan(variance(rawHistory));
  });

  it("properties return correct values", () => {
    const learner = new LearnableLogOddsWeights(3, 0.5);
    expect(learner.nSignals).toBe(3);
    expect(learner.alpha).toBe(0.5);
    expect(learner.weights).toHaveLength(3);
    expect(learner.averagedWeights).toHaveLength(3);
  });

  it("averagedWeights returns a copy", () => {
    const learner = new LearnableLogOddsWeights(2, 0.0);
    const w1 = learner.averagedWeights;
    const w2 = learner.averagedWeights;
    expect(w1).not.toBe(w2);
    // Mutating returned array should not affect internal state
    w1[0] = 999.0;
    expect(learner.averagedWeights[0]!).not.toBe(999.0);
  });

  it("softmax handles large logit differences without overflow", () => {
    const learner = new LearnableLogOddsWeights(3, 0.0);
    (learner as any)._logits = [1000.0, 0.0, -1000.0];
    const w = learner.weights;
    for (const wi of w) {
      expect(Number.isFinite(wi)).toBe(true);
    }
    let sum = 0;
    for (const wi of w) {
      sum += wi;
    }
    expect(sum).toBeCloseTo(1.0, 10);
    expect(w[0]!).toBeCloseTo(1.0, 10); // Dominant logit
    expect(w[2]!).toBeCloseTo(0.0, 10); // Negligible logit
  });

  it("analytical gradient matches finite difference", () => {
    const learner = new LearnableLogOddsWeights(3, 0.0);
    (learner as any)._logits = [0.5, -0.3, 0.1];

    const probs = [
      [0.8, 0.3, 0.6],
      [0.4, 0.9, 0.5],
    ];
    const labels = [1.0, 0.0];

    const n = learner.nSignals;
    const scale = n ** learner.alpha;

    // Log-odds of input signals
    const x = probs.map(
      (row) => logit(clampProbability(row)) as number[],
    );

    // Analytical gradient
    const logits = [...(learner as any)._logits];
    const softmaxFn = (z: number[]) => {
      let maxZ = -Infinity;
      for (const v of z) if (v > maxZ) maxZ = v;
      const expZ = z.map((v) => Math.exp(v - maxZ));
      const sum = expZ.reduce((a, b) => a + b, 0);
      return expZ.map((v) => v / sum);
    };

    const w = softmaxFn(logits);
    const analyticalGrad = new Array(n).fill(0) as number[];
    for (let s = 0; s < 2; s++) {
      let xBarW = 0;
      for (let j = 0; j < n; j++) {
        xBarW += w[j]! * x[s]![j]!;
      }
      const p = sigmoid(scale * xBarW) as number;
      const error = p - labels[s]!;
      for (let j = 0; j < n; j++) {
        analyticalGrad[j]! +=
          scale * error * w[j]! * (x[s]![j]! - xBarW);
      }
    }
    for (let j = 0; j < n; j++) {
      analyticalGrad[j]! /= 2;
    }

    // Finite-difference gradient
    const eps = 1e-5;
    const fdGrad = new Array(n).fill(0) as number[];
    for (let j = 0; j < n; j++) {
      const logitsPlus = [...logits];
      logitsPlus[j]! += eps;
      const wPlus = softmaxFn(logitsPlus);
      const logitsMinus = [...logits];
      logitsMinus[j]! -= eps;
      const wMinus = softmaxFn(logitsMinus);

      let lossPlus = 0;
      let lossMinus = 0;
      for (let s = 0; s < 2; s++) {
        let xBarPlus = 0;
        let xBarMinus = 0;
        for (let k = 0; k < n; k++) {
          xBarPlus += wPlus[k]! * x[s]![k]!;
          xBarMinus += wMinus[k]! * x[s]![k]!;
        }
        const pPlus = sigmoid(scale * xBarPlus) as number;
        const pMinus = sigmoid(scale * xBarMinus) as number;
        lossPlus +=
          -(
            labels[s]! * Math.log(Math.max(pPlus, 1e-10)) +
            (1 - labels[s]!) * Math.log(Math.max(1 - pPlus, 1e-10))
          );
        lossMinus +=
          -(
            labels[s]! * Math.log(Math.max(pMinus, 1e-10)) +
            (1 - labels[s]!) * Math.log(Math.max(1 - pMinus, 1e-10))
          );
      }
      lossPlus /= 2;
      lossMinus /= 2;
      fdGrad[j] = (lossPlus - lossMinus) / (2 * eps);
    }

    for (let j = 0; j < n; j++) {
      expect(analyticalGrad[j]!).toBeCloseTo(fdGrad[j]!, 4);
    }
  });

  it("accepts alpha='auto'", () => {
    const learner = new LearnableLogOddsWeights(3, "auto");
    expect(learner.alpha).toBe(0.5);
  });

  it("alpha='auto' produces valid output", () => {
    const learner = new LearnableLogOddsWeights(2, "auto");
    const result = learner.combine([0.8, 0.7]);
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThan(1);
  });
});

describe("alpha auto", () => {
  it("matches alpha=0.5 in unweighted mode", () => {
    const probs = [0.8, 0.9];
    const auto = logOddsConjunction(probs, "auto");
    const explicit = logOddsConjunction(probs, 0.5);
    expect(auto).toBeCloseTo(explicit, 12);
  });

  it("matches alpha=0.5 in weighted mode", () => {
    const probs = [0.8, 0.9];
    const w = [0.6, 0.4];
    const auto = logOddsConjunction(probs, "auto", w);
    const explicit = logOddsConjunction(probs, 0.5, w);
    expect(auto).toBeCloseTo(explicit, 12);
  });

  it("amplifies agreement like alpha=0.5", () => {
    const probs = [0.9, 0.9];
    const result = logOddsConjunction(probs, "auto");
    expect(result).toBeGreaterThan(0.9);
  });

  it("works with batched inputs", () => {
    const probs = [
      [0.9, 0.9],
      [0.3, 0.3],
    ];
    const result = logOddsConjunction(probs, "auto") as number[];
    expect(result).toHaveLength(2);
    expect(result[0]!).toBeGreaterThan(0.9);
    expect(result[1]!).toBeLessThan(0.5);
  });

  it("invalid alpha string throws", () => {
    expect(() =>
      logOddsConjunction([0.5, 0.5], "invalid" as any),
    ).toThrow("alpha must be");
  });

  it("alpha=undefined preserves backward compatibility", () => {
    const probs = [0.8, 0.9];
    const noneUnweighted = logOddsConjunction(probs, undefined);
    const halfUnweighted = logOddsConjunction(probs, 0.5);
    expect(noneUnweighted).toBeCloseTo(halfUnweighted, 12);

    const w = [0.6, 0.4];
    const noneWeighted = logOddsConjunction(probs, undefined, w);
    const zeroWeighted = logOddsConjunction(probs, 0.0, w);
    expect(noneWeighted).toBeCloseTo(zeroWeighted, 12);
  });
});

describe("gating", () => {
  it("none gating is identity", () => {
    const probs = [0.8, 0.9];
    const resultNone = logOddsConjunction(probs, undefined, undefined, "none");
    const resultDefault = logOddsConjunction(probs);
    expect(resultNone).toBeCloseTo(resultDefault as number, 12);
  });

  it("relu zeros weak evidence", () => {
    // 0.3 has logit < 0, so ReLU zeroes it out
    const probs = [0.9, 0.3];
    const resultRelu = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "relu",
    );
    const resultNone = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "none",
    );
    // With ReLU, the 0.3 signal is zeroed: result should be higher
    expect(resultRelu).toBeGreaterThan(resultNone as number);
  });

  it("relu on all-above-0.5 is same as no gating", () => {
    const probs = [0.8, 0.9, 0.7];
    const resultRelu = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "relu",
    );
    const resultNone = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "none",
    );
    expect(resultRelu).toBeCloseTo(resultNone as number, 12);
  });

  it("swish is between none and relu for mixed signals", () => {
    const probs = [0.9, 0.3];
    const resultNone = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "none",
    ) as number;
    const resultSwish = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "swish",
    ) as number;
    const resultRelu = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "relu",
    ) as number;
    expect(resultNone).toBeLessThan(resultSwish);
    expect(resultSwish).toBeLessThan(resultRelu);
  });

  it("swish with all above 0.5 is close to no gating", () => {
    const probs = [0.8, 0.9];
    const resultSwish = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "swish",
    ) as number;
    const resultNone = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "none",
    ) as number;
    expect(resultSwish).toBeLessThan(resultNone);
    expect(Math.abs(resultSwish - resultNone)).toBeLessThan(0.06);
  });

  it("gating works with weights", () => {
    const probs = [0.9, 0.3];
    const w = [0.5, 0.5];
    const resultNone = logOddsConjunction(probs, undefined, w, "none") as number;
    const resultRelu = logOddsConjunction(probs, undefined, w, "relu") as number;
    expect(resultRelu).toBeGreaterThan(resultNone);
  });

  it("gating works with alpha auto", () => {
    const probs = [0.9, 0.3, 0.8];
    const result = logOddsConjunction(probs, "auto", undefined, "relu");
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThan(1);
  });

  it("invalid gating throws", () => {
    expect(() =>
      logOddsConjunction([0.5, 0.5], undefined, undefined, "invalid"),
    ).toThrow("gating must be");
  });

  it("relu works with batched inputs", () => {
    const probs = [
      [0.9, 0.3],
      [0.3, 0.9],
    ];
    const result = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "relu",
    ) as number[];
    expect(result).toHaveLength(2);
    // Both batches have one signal zeroed out, symmetric
    expect(result[0]!).toBeCloseTo(result[1]!, 12);
  });

  it("swish works with batched inputs", () => {
    const probs = [
      [0.9, 0.3],
      [0.8, 0.8],
    ];
    const result = logOddsConjunction(
      probs,
      undefined,
      undefined,
      "swish",
    ) as number[];
    expect(result).toHaveLength(2);
    for (const r of result) {
      expect(Number.isFinite(r)).toBe(true);
      expect(r).toBeGreaterThan(0);
      expect(r).toBeLessThan(1);
    }
  });
});

describe("AttentionLogOddsWeights", () => {
  it("initializes with correct shapes", () => {
    const attn = new AttentionLogOddsWeights(3, 5);
    expect(attn.nSignals).toBe(3);
    expect(attn.nQueryFeatures).toBe(5);
    expect(attn.weightsMatrix).toHaveLength(3);
    expect(attn.weightsMatrix[0]!).toHaveLength(5);
    expect(attn.alpha).toBe(0.5);
  });

  it("alpha='auto' resolves to 0.5", () => {
    const attn = new AttentionLogOddsWeights(2, 3, "auto");
    expect(attn.alpha).toBe(0.5);
  });

  it("invalid nSignals throws", () => {
    expect(() => new AttentionLogOddsWeights(0, 3)).toThrow("n_signals");
  });

  it("invalid nQueryFeatures throws", () => {
    expect(() => new AttentionLogOddsWeights(2, 0)).toThrow("n_query_features");
  });

  it("single sample combine returns number in (0, 1)", () => {
    const attn = new AttentionLogOddsWeights(2, 3);
    const result = attn.combine([0.8, 0.7], [1.0, 0.5, -0.3]);
    expect(typeof result).toBe("number");
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThan(1);
  });

  it("batched combine returns array of correct length", () => {
    const attn = new AttentionLogOddsWeights(2, 3);
    const result = attn.combine(
      [
        [0.8, 0.7],
        [0.3, 0.9],
      ],
      [
        [1.0, 0.5, -0.3],
        [0.2, -0.1, 0.8],
      ],
    ) as number[];
    expect(result).toHaveLength(2);
    for (const r of result) {
      expect(r).toBeGreaterThan(0);
      expect(r).toBeLessThan(1);
    }
  });

  it("different queries produce different weights", () => {
    const attn = new AttentionLogOddsWeights(2, 3);
    const probs = [0.9, 0.3];
    const r1 = attn.combine(probs, [1.0, 0.0, 0.0]) as number;
    const r2 = attn.combine(probs, [0.0, 0.0, 1.0]) as number;
    // With random init, different features should produce different results
    expect(Math.abs(r1 - r2)).toBeGreaterThan(1e-6);
  });

  it("fit learns informative features", () => {
    let seed = 42;
    const rng = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
      return (seed >>> 0) / 0x100000000;
    };

    const m = 300;
    const labels: number[] = [];
    const probs: number[][] = [];
    const qf: number[][] = [];

    for (let i = 0; i < m; i++) {
      const label = rng() > 0.5 ? 1.0 : 0.0;
      labels.push(label);
      // Signal 0: reliable, Signal 1: noisy
      const s0 = label === 1 ? 0.85 : 0.15;
      const s1 = 0.3 + rng() * 0.4;
      probs.push([s0, s1]);
      // Constant feature favoring signal 0
      qf.push([1.0, rng() - 0.5, rng() - 0.5]);
    }

    const attn = new AttentionLogOddsWeights(2, 3, 0.0);
    attn.fit(probs, labels, qf, {
      learningRate: 0.1,
      maxIterations: 2000,
    });

    const resultHigh = attn.combine(
      [0.9, 0.5],
      [1.0, 0.0, 0.0],
    ) as number;
    const resultLow = attn.combine(
      [0.1, 0.5],
      [1.0, 0.0, 0.0],
    ) as number;
    expect(resultHigh).toBeGreaterThan(resultLow);
  });

  it("update moves parameters", () => {
    const attn = new AttentionLogOddsWeights(2, 2);
    const wBefore = attn.weightsMatrix;

    for (let i = 0; i < 50; i++) {
      attn.update([0.9, 0.1], 1.0, [1.0, 0.0], { learningRate: 0.05 });
    }

    const wAfter = attn.weightsMatrix;
    // At least one element should have changed
    let changed = false;
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        if (Math.abs(wBefore[i]![j]! - wAfter[i]![j]!) > 1e-6) {
          changed = true;
        }
      }
    }
    expect(changed).toBe(true);
  });

  it("useAveraged uses Polyak-averaged parameters", () => {
    const attn = new AttentionLogOddsWeights(2, 2, 0.0);
    const probs = [0.8, 0.7];
    const qf = [1.0, 0.5];

    // At init, averaged = raw
    const rRaw = attn.combine(probs, qf, false) as number;
    const rAvg = attn.combine(probs, qf, true) as number;
    expect(rRaw).toBeCloseTo(rAvg, 10);
  });

  it("weightsMatrix returns a copy", () => {
    const attn = new AttentionLogOddsWeights(2, 3);
    const w1 = attn.weightsMatrix;
    const w2 = attn.weightsMatrix;
    expect(w1).not.toBe(w2);
    w1[0]![0] = 999.0;
    expect(attn.weightsMatrix[0]![0]).not.toBe(999.0);
  });

  it("fit resets online state", () => {
    const attn = new AttentionLogOddsWeights(2, 2);
    attn.update([0.8, 0.2], 1.0, [1.0, 0.0]);
    expect((attn as any)._nUpdates).toBe(1);

    attn.fit(
      [
        [0.8, 0.2],
        [0.3, 0.7],
      ],
      [1.0, 0.0],
      [
        [1.0, 0.0],
        [0.0, 1.0],
      ],
    );

    expect((attn as any)._nUpdates).toBe(0);
  });
});
