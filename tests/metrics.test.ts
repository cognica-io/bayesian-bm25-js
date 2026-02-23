//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

import { describe, expect, it } from "vitest";

import {
  brierScore,
  expectedCalibrationError,
  reliabilityDiagram,
} from "../src/metrics.js";

// Seeded PRNG (mulberry32) for deterministic tests.
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randUniform(
  rng: () => number,
  low: number,
  high: number,
  size: number,
): number[] {
  const result: number[] = [];
  for (let i = 0; i < size; i++) {
    result.push(low + rng() * (high - low));
  }
  return result;
}

describe("expectedCalibrationError", () => {
  it("returns 0 for perfect calibration", () => {
    const probs = [0.0, 0.0, 1.0, 1.0];
    const labels = [0.0, 0.0, 1.0, 1.0];
    expect(expectedCalibrationError(probs, labels)).toBeCloseTo(0.0);
  });

  it("returns > 0.5 for worst calibration", () => {
    const probs = [0.9, 0.9, 0.1, 0.1];
    const labels = [0.0, 0.0, 1.0, 1.0];
    const ece = expectedCalibrationError(probs, labels);
    expect(ece).toBeGreaterThan(0.5);
  });

  it("constant prediction equals |p - base_rate|", () => {
    const labels = [0.0, 0.0, 0.0, 1.0, 1.0];
    const probs = [0.5, 0.5, 0.5, 0.5, 0.5];
    // With 1 bin, ECE = |0.5 - 0.4| = 0.1
    const ece = expectedCalibrationError(probs, labels, 1);
    expect(ece).toBeCloseTo(0.1);
  });

  it("is in [0, 1]", () => {
    const rng = mulberry32(42);
    const probs = randUniform(rng, 0, 1, 1000);
    const labels = probs.map(() => (rng() < 0.3 ? 1.0 : 0.0));
    const ece = expectedCalibrationError(probs, labels);
    expect(ece).toBeGreaterThanOrEqual(0.0);
    expect(ece).toBeLessThanOrEqual(1.0);
  });

  it("different nBins produce valid results", () => {
    const rng = mulberry32(42);
    const probs = randUniform(rng, 0, 1, 100);
    const labels = probs.map((p) => (rng() < p ? 1.0 : 0.0));
    for (const nBins of [2, 5, 10, 20, 50]) {
      const ece = expectedCalibrationError(probs, labels, nBins);
      expect(ece).toBeGreaterThanOrEqual(0.0);
      expect(ece).toBeLessThanOrEqual(1.0);
    }
  });
});

describe("brierScore", () => {
  it("returns 0 for perfect predictions", () => {
    const probs = [0.0, 0.0, 1.0, 1.0];
    const labels = [0.0, 0.0, 1.0, 1.0];
    expect(brierScore(probs, labels)).toBeCloseTo(0.0);
  });

  it("returns 1 for completely wrong predictions", () => {
    const probs = [1.0, 1.0, 0.0, 0.0];
    const labels = [0.0, 0.0, 1.0, 1.0];
    expect(brierScore(probs, labels)).toBeCloseTo(1.0);
  });

  it("returns 0.25 for constant 0.5 prediction", () => {
    const probs = new Array(100).fill(0.5);
    const labels = [
      ...new Array(50).fill(0.0),
      ...new Array(50).fill(1.0),
    ];
    expect(brierScore(probs, labels)).toBeCloseTo(0.25);
  });

  it("is in [0, 1] for valid probabilities", () => {
    const rng = mulberry32(42);
    const probs = randUniform(rng, 0, 1, 1000);
    const labels = probs.map(() => (rng() < 0.3 ? 1.0 : 0.0));
    const bs = brierScore(probs, labels);
    expect(bs).toBeGreaterThanOrEqual(0.0);
    expect(bs).toBeLessThanOrEqual(1.0);
  });

  it("better calibration gives lower score", () => {
    const labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    const goodProbs = [0.1, 0.2, 0.1, 0.8, 0.9, 0.8];
    const badProbs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    expect(brierScore(goodProbs, labels)).toBeLessThan(
      brierScore(badProbs, labels),
    );
  });
});

describe("reliabilityDiagram", () => {
  it("returns tuples with valid values", () => {
    const probs = [0.1, 0.2, 0.8, 0.9];
    const labels = [0.0, 0.0, 1.0, 1.0];
    const bins = reliabilityDiagram(probs, labels);
    expect(bins.length).toBeGreaterThan(0);
    for (const [avgPred, avgActual, count] of bins) {
      expect(avgPred).toBeGreaterThanOrEqual(0.0);
      expect(avgPred).toBeLessThanOrEqual(1.0);
      expect(avgActual).toBeGreaterThanOrEqual(0.0);
      expect(avgActual).toBeLessThanOrEqual(1.0);
      expect(count).toBeGreaterThan(0);
    }
  });

  it("total count equals number of samples", () => {
    const rng = mulberry32(42);
    const probs = randUniform(rng, 0, 1, 200);
    const labels = probs.map(() => (rng() < 0.5 ? 1.0 : 0.0));
    const bins = reliabilityDiagram(probs, labels);
    let total = 0;
    for (const [, , count] of bins) {
      total += count;
    }
    expect(total).toBe(200);
  });

  it("perfect calibration produces diagonal", () => {
    const rng = mulberry32(42);
    const n = 10000;
    const probs = randUniform(rng, 0, 1, n);
    const labels = probs.map((p) => (rng() < p ? 1.0 : 0.0));
    const bins = reliabilityDiagram(probs, labels, 5);
    for (const [avgPred, avgActual, count] of bins) {
      if (count >= 100) {
        expect(Math.abs(avgPred - avgActual)).toBeLessThan(0.1);
      }
    }
  });

  it("empty bins are excluded", () => {
    const probs = [0.05, 0.05, 0.95, 0.95];
    const labels = [0.0, 0.0, 1.0, 1.0];
    const bins = reliabilityDiagram(probs, labels, 10);
    // Only the extreme bins should have data
    expect(bins.length).toBeLessThanOrEqual(3);
  });

  it("different nBins produce valid results", () => {
    const rng = mulberry32(42);
    const probs = randUniform(rng, 0, 1, 100);
    const labels = probs.map(() => (rng() < 0.5 ? 1.0 : 0.0));
    for (const nBins of [2, 5, 10, 20]) {
      const bins = reliabilityDiagram(probs, labels, nBins);
      expect(bins.length).toBeGreaterThan(0);
      expect(bins.length).toBeLessThanOrEqual(nBins);
    }
  });
});

describe("mainPackageExport", () => {
  it("metrics are importable from main package", async () => {
    const mod = await import("../src/index.js");
    expect(typeof mod.expectedCalibrationError).toBe("function");
    expect(typeof mod.brierScore).toBe("function");
    expect(typeof mod.reliabilityDiagram).toBe("function");
  });
});
