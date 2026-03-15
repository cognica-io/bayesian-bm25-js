//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

import { describe, expect, it } from "vitest";

import {
  BayesianProbabilityTransform,
  TemporalBayesianTransform,
  sigmoid,
} from "../src/probability.js";

// Seeded PRNG (mulberry32) for deterministic test data.
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("TemporalBayesianTransform", () => {
  it("has correct decayHalfLife property", () => {
    const t = new TemporalBayesianTransform(1.0, 0.0, null, 500.0);
    expect(t.decayHalfLife).toBe(500.0);
  });

  it("timestamp starts at 0", () => {
    const t = new TemporalBayesianTransform();
    expect(t.timestamp).toBe(0);
  });

  it("is an instance of BayesianProbabilityTransform", () => {
    const t = new TemporalBayesianTransform();
    expect(t).toBeInstanceOf(BayesianProbabilityTransform);
  });

  it("throws on negative decayHalfLife", () => {
    expect(() => new TemporalBayesianTransform(1.0, 0.0, null, -1.0)).toThrow(
      /decayHalfLife must be positive/,
    );
  });

  it("throws on zero decayHalfLife", () => {
    expect(() => new TemporalBayesianTransform(1.0, 0.0, null, 0.0)).toThrow(
      /decayHalfLife must be positive/,
    );
  });

  it("fit() without timestamps matches parent class behavior", () => {
    const rng = mulberry32(42);
    const scores: number[] = [];
    const labels: number[] = [];
    for (let i = 0; i < 50; i++) {
      const s = rng() * 3;
      scores.push(s);
      labels.push(s > 1.5 ? 1.0 : 0.0);
    }

    const t = new TemporalBayesianTransform(0.5, 0.0, null, 1000.0);
    t.fit(scores, labels, { learningRate: 0.05, maxIterations: 500 });
    // Parameters should have moved from initial values
    expect(t.alpha !== 0.5 || t.beta !== 0.0).toBe(true);
  });

  it("fit() with timestamps and short half-life favors recent data", () => {
    const rng = mulberry32(42);

    // Early data: generated with alpha=1.0, beta=0.0
    const nEarly = 200;
    const scoresEarly: number[] = [];
    const labelsEarly: number[] = [];
    for (let i = 0; i < nEarly; i++) {
      const s = rng() * 3;
      scoresEarly.push(s);
      const probRelevant = sigmoid(1.0 * s) as number;
      labelsEarly.push(rng() < probRelevant ? 1.0 : 0.0);
    }

    // Recent data: generated with alpha=3.0, beta=2.0
    const nRecent = 200;
    const scoresRecent: number[] = [];
    const labelsRecent: number[] = [];
    for (let i = 0; i < nRecent; i++) {
      const s = rng() * 5;
      scoresRecent.push(s);
      const probRelevant = sigmoid(3.0 * (s - 2.0)) as number;
      labelsRecent.push(rng() < probRelevant ? 1.0 : 0.0);
    }

    const scores = [...scoresEarly, ...scoresRecent];
    const labels = [...labelsEarly, ...labelsRecent];
    const timestamps: number[] = [];
    for (let i = 0; i < scores.length; i++) {
      timestamps.push(i);
    }

    // With short half-life, recent observations dominate
    const tTemporal = new TemporalBayesianTransform(1.0, 0.0, null, 50.0);
    tTemporal.fit(scores, labels, {
      timestamps,
      learningRate: 0.05,
      maxIterations: 3000,
    });

    // Without temporal weighting, use parent class
    const tUniform = new BayesianProbabilityTransform(1.0, 0.0);
    tUniform.fit(scores, labels, {
      learningRate: 0.05,
      maxIterations: 3000,
    });

    // Temporal should have beta closer to 2.0 (recent truth) than uniform
    expect(Math.abs(tTemporal.beta - 2.0)).toBeLessThan(
      Math.abs(tUniform.beta - 2.0),
    );
  });

  it("fit() with very large half-life matches parent class within tolerance", () => {
    const rng = mulberry32(42);
    const scores: number[] = [];
    const labels: number[] = [];
    for (let i = 0; i < 100; i++) {
      const s = rng() * 3;
      scores.push(s);
      const probRelevant = sigmoid(2.0 * (s - 1.0)) as number;
      labels.push(rng() < probRelevant ? 1.0 : 0.0);
    }

    const tTemporal = new TemporalBayesianTransform(0.5, 0.0, null, 1e10);
    const tUniform = new BayesianProbabilityTransform(0.5, 0.0);

    tTemporal.fit(scores, labels, {
      learningRate: 0.05,
      maxIterations: 2000,
    });
    tUniform.fit(scores, labels, {
      learningRate: 0.05,
      maxIterations: 2000,
    });

    // Should be very close
    expect(tTemporal.alpha).toBeCloseTo(tUniform.alpha, 1);
    expect(tTemporal.beta).toBeCloseTo(tUniform.beta, 1);
  });

  it("update() increments timestamp on each call", () => {
    const t = new TemporalBayesianTransform();
    expect(t.timestamp).toBe(0);
    t.update(2.0, 1.0);
    expect(t.timestamp).toBe(1);
    t.update(1.0, 0.0);
    expect(t.timestamp).toBe(2);
  });

  it("scoreToProbability works unchanged from parent", () => {
    const t = new TemporalBayesianTransform(1.0, 0.0);
    const result = t.scoreToProbability(2.0, 5.0, 0.5) as number;
    expect(result).toBeGreaterThan(0.0);
    expect(result).toBeLessThan(1.0);
  });

  it("temporal weighting effect: short half-life favors recent observations", () => {
    const rng = mulberry32(99);

    // All data generated with alpha=2.0, beta=1.0 early,
    // then alpha=2.0, beta=3.0 late
    const nEarly = 150;
    const nRecent = 150;
    const scores: number[] = [];
    const labels: number[] = [];
    const timestamps: number[] = [];

    for (let i = 0; i < nEarly; i++) {
      const s = rng() * 4;
      scores.push(s);
      labels.push(rng() < (sigmoid(2.0 * (s - 1.0)) as number) ? 1.0 : 0.0);
      timestamps.push(i);
    }
    for (let i = 0; i < nRecent; i++) {
      const s = rng() * 6;
      scores.push(s);
      labels.push(rng() < (sigmoid(2.0 * (s - 3.0)) as number) ? 1.0 : 0.0);
      timestamps.push(nEarly + i);
    }

    // Short half-life: should learn beta closer to 3.0
    const tShort = new TemporalBayesianTransform(1.0, 0.0, null, 30.0);
    tShort.fit(scores, labels, {
      timestamps,
      learningRate: 0.05,
      maxIterations: 3000,
    });

    // Long half-life: should learn beta between 1.0 and 3.0
    const tLong = new TemporalBayesianTransform(1.0, 0.0, null, 1e8);
    tLong.fit(scores, labels, {
      timestamps,
      learningRate: 0.05,
      maxIterations: 3000,
    });

    // Short half-life beta should be closer to 3.0 than long half-life
    expect(Math.abs(tShort.beta - 3.0)).toBeLessThan(
      Math.abs(tLong.beta - 3.0),
    );
  });
});
