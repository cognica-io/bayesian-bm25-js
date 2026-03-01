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
});
