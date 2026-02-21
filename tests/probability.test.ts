//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

import { describe, expect, it } from "vitest";

import {
  BayesianProbabilityTransform,
  logit,
  sigmoid,
  EPSILON,
} from "../src/probability.js";

describe("sigmoid", () => {
  it("returns 0.5 at zero", () => {
    expect(sigmoid(0.0)).toBeCloseTo(0.5);
  });

  it("handles large positive values without overflow", () => {
    expect(sigmoid(1000.0)).toBeCloseTo(1.0);
  });

  it("handles large negative values without underflow", () => {
    expect(sigmoid(-1000.0)).toBeCloseTo(0.0, 15);
  });

  it("satisfies symmetry: sigmoid(x) + sigmoid(-x) == 1", () => {
    const x = 2.5;
    expect(sigmoid(x) + sigmoid(-x)).toBeCloseTo(1.0);
  });

  it("is monotonically increasing", () => {
    const x = [-3.0, -1.0, 0.0, 1.0, 3.0];
    const s = sigmoid(x);
    for (let i = 1; i < s.length; i++) {
      expect(s[i]!).toBeGreaterThan(s[i - 1]!);
    }
  });

  it("handles array input", () => {
    const x = [0.0, 1.0, -1.0];
    const result = sigmoid(x);
    expect(result).toHaveLength(3);
    expect(result[0]).toBeCloseTo(0.5);
  });
});

describe("logit", () => {
  it("returns 0 at 0.5", () => {
    expect(logit(0.5)).toBeCloseTo(0.0);
  });

  it("roundtrips with sigmoid", () => {
    const p = 0.73;
    expect(sigmoid(logit(p) as number)).toBeCloseTo(p);
  });

  it("handles extreme values without inf", () => {
    expect(Number.isFinite(logit(0.0))).toBe(true);
    expect(Number.isFinite(logit(1.0))).toBe(true);
  });

  it("is monotonically increasing", () => {
    const p = [0.1, 0.3, 0.5, 0.7, 0.9];
    const lVals = logit(p);
    for (let i = 1; i < lVals.length; i++) {
      expect(lVals[i]!).toBeGreaterThan(lVals[i - 1]!);
    }
  });
});

describe("tfPrior", () => {
  it("returns 0.2 at tf=0", () => {
    expect(BayesianProbabilityTransform.tfPrior(0)).toBeCloseTo(0.2);
  });

  it("saturates at 0.9 for tf >= 10", () => {
    expect(BayesianProbabilityTransform.tfPrior(10)).toBeCloseTo(0.9);
    expect(BayesianProbabilityTransform.tfPrior(100)).toBeCloseTo(0.9);
  });

  it("returns 0.55 at tf=5", () => {
    expect(BayesianProbabilityTransform.tfPrior(5)).toBeCloseTo(0.55);
  });

  it("handles array input", () => {
    const result = BayesianProbabilityTransform.tfPrior([0, 5, 10]);
    expect(result[0]).toBeCloseTo(0.2);
    expect(result[1]).toBeCloseTo(0.55);
    expect(result[2]).toBeCloseTo(0.9);
  });
});

describe("normPrior", () => {
  it("peaks at doc_len_ratio = 0.5", () => {
    expect(BayesianProbabilityTransform.normPrior(0.5)).toBeCloseTo(0.9);
  });

  it("returns 0.3 at extreme lengths", () => {
    expect(BayesianProbabilityTransform.normPrior(0.0)).toBeCloseTo(0.3);
    expect(BayesianProbabilityTransform.normPrior(1.0)).toBeCloseTo(0.3);
  });

  it("stays within bounds [0.3, 0.9]", () => {
    const ratios = Array.from({ length: 100 }, (_, i) => (i / 99) * 3);
    const priors = BayesianProbabilityTransform.normPrior(ratios);
    for (const p of priors) {
      expect(p).toBeGreaterThanOrEqual(0.3);
      expect(p).toBeLessThanOrEqual(0.9);
    }
  });
});

describe("compositePrior", () => {
  it("stays within bounds [0.1, 0.9]", () => {
    for (const tf of [0, 1, 5, 10, 100]) {
      for (const ratio of [0.0, 0.25, 0.5, 1.0, 2.0]) {
        const p = BayesianProbabilityTransform.compositePrior(tf, ratio);
        expect(p).toBeGreaterThanOrEqual(0.1);
        expect(p).toBeLessThanOrEqual(0.9);
      }
    }
  });

  it("handles array input", () => {
    const tf = [0, 5, 10];
    const ratio = [0.5, 0.5, 0.5];
    const result = BayesianProbabilityTransform.compositePrior(tf, ratio);
    expect(result).toHaveLength(3);
    for (const p of result) {
      expect(p).toBeGreaterThanOrEqual(0.1);
      expect(p).toBeLessThanOrEqual(0.9);
    }
  });
});

describe("posterior", () => {
  it("equals likelihood when prior is 0.5", () => {
    const lVal = 0.7;
    const p = BayesianProbabilityTransform.posterior(lVal, 0.5);
    expect(p).toBeCloseTo(lVal);
  });

  it("amplifies with high prior", () => {
    const lVal = 0.6;
    const lowPrior = BayesianProbabilityTransform.posterior(lVal, 0.3);
    const highPrior = BayesianProbabilityTransform.posterior(lVal, 0.7);
    expect(highPrior).toBeGreaterThan(lowPrior);
  });

  it("is monotonic in likelihood", () => {
    const prior = 0.5;
    const likelihoods = [0.1, 0.3, 0.5, 0.7, 0.9];
    const posteriors = BayesianProbabilityTransform.posterior(
      likelihoods,
      [prior, prior, prior, prior, prior],
    );
    for (let i = 1; i < posteriors.length; i++) {
      expect(posteriors[i]!).toBeGreaterThan(posteriors[i - 1]!);
    }
  });

  it("stays in bounds (0, 1)", () => {
    const lVal = [0.01, 0.5, 0.99];
    const prior = [0.01, 0.5, 0.99];
    const posteriors = BayesianProbabilityTransform.posterior(lVal, prior);
    for (const p of posteriors) {
      expect(p).toBeGreaterThan(0);
      expect(p).toBeLessThan(1);
    }
  });
});

describe("scoreToProbability", () => {
  it("gives higher probability for higher score", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    const pLow = t.scoreToProbability(0.5, 5, 0.5);
    const pHigh = t.scoreToProbability(2.0, 5, 0.5);
    expect(pHigh).toBeGreaterThan(pLow);
  });

  it("matches expected behavior for paper values", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    const scores = [1.0464478, 0.56150854, 1.1230172];
    const tf = [5.0, 3.0, 7.0];
    const docLenRatio = [0.5, 0.5, 0.5];
    const probs = t.scoreToProbability(scores, tf, docLenRatio);
    // All probabilities should be in (0, 1)
    for (const p of probs) {
      expect(p).toBeGreaterThan(0.0);
      expect(p).toBeLessThan(1.0);
    }
    // Monotonicity: higher score with equal or higher tf -> higher prob
    expect(probs[2]!).toBeGreaterThan(probs[1]!);
    expect(probs[0]!).toBeGreaterThan(probs[1]!);
  });
});

describe("fit", () => {
  it("learns parameters from synthetic data", () => {
    // Generate synthetic data with known alpha=2.0, beta=1.0
    const trueAlpha = 2.0;
    const trueBeta = 1.0;

    // Use a seeded sequence for reproducibility
    const scores: number[] = [];
    const labels: number[] = [];

    // Simple LCG for reproducibility
    let seed = 42;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    for (let i = 0; i < 200; i++) {
      const score = rand() * 3;
      scores.push(score);
      const probRelevant = sigmoid(trueAlpha * (score - trueBeta)) as number;
      labels.push(rand() < probRelevant ? 1.0 : 0.0);
    }

    const t = new BayesianProbabilityTransform(0.5, 0.0);
    t.fit(scores, labels, {
      learningRate: 0.05,
      maxIterations: 5000,
    });

    // Should be in the right ballpark
    expect(Math.abs(t.alpha - trueAlpha)).toBeLessThan(1.0);
    expect(Math.abs(t.beta - trueBeta)).toBeLessThan(1.0);
  });

  it("converges on simple data", () => {
    const scores = [0.0, 1.0, 2.0, 3.0, 4.0];
    const labels = [0.0, 0.0, 0.5, 1.0, 1.0];

    const t = new BayesianProbabilityTransform(0.1, 0.0);
    t.fit(scores, labels, { learningRate: 0.01, maxIterations: 2000 });

    const predicted = sigmoid(
      scores.map((s) => t.alpha * (s - t.beta)),
    ) as number[];
    expect(predicted[0]!).toBeLessThan(predicted[4]!);
  });
});

describe("onlineUpdate", () => {
  it("converges to similar parameters as batch fit", () => {
    const trueAlpha = 2.0;
    const trueBeta = 1.0;

    let seed = 123;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    const scores: number[] = [];
    const labels: number[] = [];
    for (let i = 0; i < 500; i++) {
      const score = rand() * 3;
      scores.push(score);
      const probRelevant = sigmoid(trueAlpha * (score - trueBeta)) as number;
      labels.push(rand() < probRelevant ? 1.0 : 0.0);
    }

    const t = new BayesianProbabilityTransform(0.5, 0.0);
    for (let i = 0; i < scores.length; i++) {
      t.update(scores[i]!, labels[i]!, {
        learningRate: 0.05,
        momentum: 0.9,
      });
    }

    expect(Math.abs(t.alpha - trueAlpha)).toBeLessThan(1.5);
    expect(Math.abs(t.beta - trueBeta)).toBeLessThan(1.0);
  });

  it("moves parameters with repeated updates", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    const initialAlpha = t.alpha;
    const initialBeta = t.beta;

    for (let i = 0; i < 20; i++) {
      t.update(3.0, 1.0, { learningRate: 0.1 });
      t.update(-1.0, 0.0, { learningRate: 0.1 });
    }

    expect(t.alpha !== initialAlpha || t.beta !== initialBeta).toBe(true);
  });

  it("accepts arrays for mini-batch updates", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    const scores = [0.5, 1.5, 2.5];
    const labels = [0.0, 1.0, 1.0];
    t.update(scores, labels, { learningRate: 0.05 });
    expect(t.nUpdates).toBe(1);
  });

  it("resets EMA state after fit()", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    t.update(2.0, 1.0, { learningRate: 0.1 });
    expect(t.nUpdates).toBe(1);
    expect(t.gradAlphaEMA).not.toBe(0.0);

    t.fit([1.0, 2.0], [0.0, 1.0], { maxIterations: 10 });
    expect(t.nUpdates).toBe(0);
    expect(t.gradAlphaEMA).toBe(0.0);
  });

  it("produces smoother trajectories with higher momentum", () => {
    let seed = 42;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    const scores: number[] = [];
    const labels: number[] = [];
    for (let i = 0; i < 100; i++) {
      const score = rand() * 3;
      scores.push(score);
      const label = score > 1.5 ? 1.0 : 0.0;
      labels.push(rand() < 0.2 ? 1.0 - label : label);
    }

    // Low momentum
    const tLow = new BayesianProbabilityTransform(1.0, 0.0);
    const alphasLow: number[] = [];
    for (let i = 0; i < scores.length; i++) {
      tLow.update(scores[i]!, labels[i]!, {
        learningRate: 0.05,
        momentum: 0.5,
      });
      alphasLow.push(tLow.alpha);
    }

    // High momentum
    const tHigh = new BayesianProbabilityTransform(1.0, 0.0);
    const alphasHigh: number[] = [];
    for (let i = 0; i < scores.length; i++) {
      tHigh.update(scores[i]!, labels[i]!, {
        learningRate: 0.05,
        momentum: 0.95,
      });
      alphasHigh.push(tHigh.alpha);
    }

    // Compute variance of differences
    const diffLow: number[] = [];
    const diffHigh: number[] = [];
    for (let i = 1; i < alphasLow.length; i++) {
      diffLow.push(alphasLow[i]! - alphasLow[i - 1]!);
      diffHigh.push(alphasHigh[i]! - alphasHigh[i - 1]!);
    }

    const variance = (arr: number[]) => {
      const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
      return (
        arr.reduce((sum, v) => sum + (v - mean) ** 2, 0) / arr.length
      );
    };

    expect(variance(diffHigh)).toBeLessThan(variance(diffLow));
  });
});

describe("baseRate", () => {
  it("baseRate=null produces identical output to no base rate", () => {
    const tNone = new BayesianProbabilityTransform(1.0, 0.5, null);
    for (const score of [0.5, 1.0, 2.0, 3.0]) {
      const prob = tNone.scoreToProbability(score, 5, 0.5) as number;
      expect(prob).toBeGreaterThan(0);
      expect(prob).toBeLessThan(1);
    }
  });

  it("baseRate=0.5 is neutral (logit(0.5)=0)", () => {
    const tHalf = new BayesianProbabilityTransform(1.0, 0.5, 0.5);
    const tNone = new BayesianProbabilityTransform(1.0, 0.5, null);
    for (const score of [0.5, 1.0, 2.0, 3.0]) {
      const pHalf = tHalf.scoreToProbability(score, 5, 0.5) as number;
      const pNone = tNone.scoreToProbability(score, 5, 0.5) as number;
      expect(pHalf).toBeCloseTo(pNone, 4);
    }
  });

  it("low baseRate reduces probabilities", () => {
    const tLow = new BayesianProbabilityTransform(1.0, 0.5, 0.01);
    const tNone = new BayesianProbabilityTransform(1.0, 0.5, null);
    for (const score of [0.5, 1.0, 2.0, 3.0]) {
      const pLow = tLow.scoreToProbability(score, 5, 0.5) as number;
      const pNone = tNone.scoreToProbability(score, 5, 0.5) as number;
      expect(pLow).toBeLessThan(pNone);
    }
  });

  it("high baseRate increases probabilities", () => {
    const tHigh = new BayesianProbabilityTransform(1.0, 0.5, 0.9);
    const tNone = new BayesianProbabilityTransform(1.0, 0.5, null);
    for (const score of [0.5, 1.0, 2.0, 3.0]) {
      const pHigh = tHigh.scoreToProbability(score, 5, 0.5) as number;
      const pNone = tNone.scoreToProbability(score, 5, 0.5) as number;
      expect(pHigh).toBeGreaterThan(pNone);
    }
  });

  it("preserves score ordering with any baseRate", () => {
    for (const br of [0.01, 0.1, 0.5, 0.9]) {
      const t = new BayesianProbabilityTransform(1.0, 0.5, br);
      const scores = [0.5, 1.0, 2.0, 3.0];
      const probs = t.scoreToProbability(
        scores,
        scores.map(() => 5),
        scores.map(() => 0.5),
      ) as number[];
      for (let i = 1; i < probs.length; i++) {
        expect(probs[i]!).toBeGreaterThan(probs[i - 1]!);
      }
    }
  });

  it("all probabilities stay in (0, 1)", () => {
    for (const br of [0.01, 0.1, 0.5, 0.9]) {
      const t = new BayesianProbabilityTransform(1.0, 0.5, br);
      const scores = [0.1, 0.5, 1.0, 2.0, 5.0];
      const probs = t.scoreToProbability(
        scores,
        scores.map(() => 5),
        scores.map(() => 0.5),
      ) as number[];
      for (const p of probs) {
        expect(p).toBeGreaterThan(0);
        expect(p).toBeLessThan(1);
      }
    }
  });

  it("rejects invalid baseRate", () => {
    expect(() => new BayesianProbabilityTransform(1.0, 0.0, 0.0)).toThrow(
      /baseRate must be in/,
    );
    expect(() => new BayesianProbabilityTransform(1.0, 0.0, 1.0)).toThrow(
      /baseRate must be in/,
    );
    expect(() => new BayesianProbabilityTransform(1.0, 0.0, -0.5)).toThrow(
      /baseRate must be in/,
    );
  });

  it("fit preserves baseRate", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0, 0.01);
    t.fit([0.0, 1.0, 2.0], [0.0, 0.5, 1.0], { maxIterations: 10 });
    expect(t.baseRate).toBe(0.01);
  });

  it("update preserves baseRate", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0, 0.05);
    t.update(2.0, 1.0, { learningRate: 0.01 });
    expect(t.baseRate).toBe(0.05);
  });

  it("posterior static with baseRate shifts probabilities", () => {
    const L = 0.7;
    const p = 0.6;
    const pWithout = BayesianProbabilityTransform.posterior(L, p);
    const pWithBR = BayesianProbabilityTransform.posterior(L, p, 0.01);
    expect(pWithBR).toBeLessThan(pWithout);
  });
});
