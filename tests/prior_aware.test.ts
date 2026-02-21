//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

// Tests for prior-aware training modes (C1/C2/C3 conditions, Algorithm 8.3.1).

import { describe, expect, it } from "vitest";

import {
  BayesianProbabilityTransform,
  sigmoid,
} from "../src/probability.js";

// Simple LCG for reproducibility
function makeLCG(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

describe("BalancedMode", () => {
  it("produces same result as default (no mode specified)", () => {
    const rand = makeLCG(42);
    const scores: number[] = [];
    const labels: number[] = [];
    for (let i = 0; i < 200; i++) {
      const score = rand() * 3;
      scores.push(score);
      const probRelevant = sigmoid(2.0 * (score - 1.0)) as number;
      labels.push(rand() < probRelevant ? 1.0 : 0.0);
    }

    const tDefault = new BayesianProbabilityTransform(0.5, 0.0);
    tDefault.fit(scores, labels, {
      learningRate: 0.05,
      maxIterations: 3000,
    });

    const tBalanced = new BayesianProbabilityTransform(0.5, 0.0);
    tBalanced.fit(scores, labels, {
      learningRate: 0.05,
      maxIterations: 3000,
      mode: "balanced",
    });

    expect(tBalanced.alpha).toBeCloseTo(tDefault.alpha, 10);
    expect(tBalanced.beta).toBeCloseTo(tDefault.beta, 10);
  });
});

describe("PriorAwareMode", () => {
  it("learns parameters with prior", () => {
    const rand = makeLCG(42);
    const n = 500;
    const scores: number[] = [];
    const tfs: number[] = [];
    const docLenRatios: number[] = [];
    for (let i = 0; i < n; i++) {
      scores.push(rand() * 5);
      tfs.push(rand() * 15);
      docLenRatios.push(0.2 + rand() * 1.8);
    }

    const trueAlpha = 1.5;
    const trueBeta = 2.0;
    const likelihoods = scores.map(
      (s) => sigmoid(trueAlpha * (s - trueBeta)) as number,
    );
    const priors = BayesianProbabilityTransform.compositePrior(
      tfs,
      docLenRatios,
    );
    const labels = likelihoods.map((l, i) => {
      const p = priors[i]!;
      const posterior = (l * p) / (l * p + (1.0 - l) * (1.0 - p));
      return rand() < posterior ? 1.0 : 0.0;
    });

    const t = new BayesianProbabilityTransform(0.5, 0.0);
    t.fit(scores, labels, {
      learningRate: 0.01,
      maxIterations: 5000,
      mode: "prior_aware",
      tfs,
      docLenRatios,
    });

    expect(Math.abs(t.alpha - trueAlpha)).toBeLessThan(2.0);
    expect(Math.abs(t.beta - trueBeta)).toBeLessThan(2.0);
  });

  it("requires tfs and docLenRatios", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    const scores = [1.0, 2.0];
    const labels = [0.0, 1.0];

    expect(() => {
      t.fit(scores, labels, { mode: "prior_aware" });
    }).toThrow(/tfs and docLenRatios are required/);

    expect(() => {
      t.fit(scores, labels, { mode: "prior_aware", tfs: [1.0, 2.0] });
    }).toThrow(/tfs and docLenRatios are required/);

    expect(() => {
      t.fit(scores, labels, {
        mode: "prior_aware",
        docLenRatios: [0.5, 0.5],
      });
    }).toThrow(/tfs and docLenRatios are required/);
  });

  it("uses posterior as prediction", () => {
    const t = new BayesianProbabilityTransform(1.0, 1.0);

    const scores = [2.0, 2.0];
    const labels = [1.0, 1.0];
    const tfs = [10.0, 10.0];
    const docLenRatios = [0.5, 0.5];

    const initialAlpha = t.alpha;
    t.fit(scores, labels, {
      learningRate: 0.01,
      maxIterations: 100,
      mode: "prior_aware",
      tfs,
      docLenRatios,
    });
    expect(t.alpha !== initialAlpha || t.beta !== 1.0).toBe(true);
  });
});

describe("PriorFreeMode", () => {
  it("trains identically to balanced (only inference differs)", () => {
    const rand = makeLCG(42);
    const scores: number[] = [];
    const labels: number[] = [];
    for (let i = 0; i < 200; i++) {
      const score = rand() * 3;
      scores.push(score);
      const probRelevant = sigmoid(2.0 * (score - 1.0)) as number;
      labels.push(rand() < probRelevant ? 1.0 : 0.0);
    }

    const tBalanced = new BayesianProbabilityTransform(0.5, 0.0);
    tBalanced.fit(scores, labels, {
      learningRate: 0.05,
      maxIterations: 3000,
      mode: "balanced",
    });

    const tFree = new BayesianProbabilityTransform(0.5, 0.0);
    tFree.fit(scores, labels, {
      learningRate: 0.05,
      maxIterations: 3000,
      mode: "prior_free",
    });

    expect(tFree.alpha).toBeCloseTo(tBalanced.alpha, 10);
    expect(tFree.beta).toBeCloseTo(tBalanced.beta, 10);
  });

  it("inference uses uniform prior (prior=0.5)", () => {
    const t = new BayesianProbabilityTransform(1.0, 1.0);
    t.fit([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 1.0], {
      maxIterations: 100,
      mode: "prior_free",
    });

    // With prior=0.5, posterior = likelihood
    const score = 2.5;
    const prob = t.scoreToProbability(score, 10, 0.5) as number;
    const likelihood = t.likelihood(score) as number;
    expect(prob).toBeCloseTo(likelihood, 8);
  });

  it("produces different inference than balanced", () => {
    const scores = [0.0, 1.0, 2.0, 3.0];
    const labels = [0.0, 0.0, 1.0, 1.0];

    const tBalanced = new BayesianProbabilityTransform(1.0, 1.0);
    tBalanced.fit(scores, labels, { maxIterations: 100, mode: "balanced" });

    const tFree = new BayesianProbabilityTransform(1.0, 1.0);
    tFree.fit(scores, labels, { maxIterations: 100, mode: "prior_free" });

    const testScore = 2.5;
    const pBalanced = tBalanced.scoreToProbability(
      testScore,
      10,
      0.5,
    ) as number;
    const pFree = tFree.scoreToProbability(testScore, 10, 0.5) as number;

    // With tf=10 and ratio=0.5, compositePrior > 0.5, so balanced != free
    expect(Math.abs(pBalanced - pFree)).toBeGreaterThan(1e-4);
  });
});

describe("ModeValidation", () => {
  it("rejects invalid mode in fit()", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    expect(() => {
      t.fit([1.0, 2.0], [0.0, 1.0], {
        mode: "invalid" as "balanced",
      });
    }).toThrow(/mode must be one of/);
  });

  it("rejects invalid mode in update()", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    expect(() => {
      t.update(1.0, 1.0, { mode: "invalid" as "balanced" });
    }).toThrow(/mode must be one of/);
  });
});

describe("OnlineUpdateModes", () => {
  it("requires tf for prior_aware update", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    expect(() => {
      t.update(1.0, 1.0, { mode: "prior_aware" });
    }).toThrow(/tf and docLenRatio are required/);
  });

  it("prior_aware update moves parameters", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    const initialAlpha = t.alpha;
    for (let i = 0; i < 50; i++) {
      t.update(3.0, 1.0, {
        learningRate: 0.1,
        mode: "prior_aware",
        tf: 5.0,
        docLenRatio: 0.5,
      });
      t.update(-1.0, 0.0, {
        learningRate: 0.1,
        mode: "prior_aware",
        tf: 1.0,
        docLenRatio: 2.0,
      });
    }
    expect(t.alpha).not.toBe(initialAlpha);
  });

  it("update without mode uses mode from last fit()", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    t.fit([0.0, 1.0, 2.0], [0.0, 0.5, 1.0], {
      maxIterations: 10,
      mode: "prior_free",
    });

    // prior_free trains like balanced (no tf/docLenRatio required)
    t.update(2.0, 1.0, { learningRate: 0.01 });
    // Should not throw - inherited prior_free mode uses balanced training
  });
});

describe("ConvergenceComparison", () => {
  it("all modes converge", () => {
    const rand = makeLCG(42);
    const n = 300;
    const scores: number[] = [];
    const tfs: number[] = [];
    const docLenRatios: number[] = [];
    const labels: number[] = [];
    for (let i = 0; i < n; i++) {
      const score = rand() * 4;
      scores.push(score);
      tfs.push(rand() * 15);
      docLenRatios.push(0.2 + rand() * 1.8);
      const prob = sigmoid(1.5 * (score - 2.0)) as number;
      labels.push(rand() < prob ? 1.0 : 0.0);
    }

    for (const modeName of [
      "balanced",
      "prior_aware",
      "prior_free",
    ] as const) {
      const t = new BayesianProbabilityTransform(0.5, 0.0);
      const opts: Record<string, unknown> = {
        learningRate: 0.01,
        maxIterations: 5000,
        mode: modeName,
      };
      if (modeName === "prior_aware") {
        opts.tfs = tfs;
        opts.docLenRatios = docLenRatios;
      }

      t.fit(scores, labels, opts as Parameters<typeof t.fit>[2]);

      expect(t.alpha).toBeGreaterThan(0);
      expect(t.alpha !== 0.5 || t.beta !== 0.0).toBe(true);
    }
  });
});
