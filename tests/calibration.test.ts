//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

import { describe, expect, it } from "vitest";

import { PlattCalibrator, IsotonicCalibrator } from "../src/calibration.js";
import { sigmoid } from "../src/probability.js";

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

describe("PlattCalibrator", () => {
  it("has default params a=1, b=0", () => {
    const cal = new PlattCalibrator();
    expect(cal.a).toBeCloseTo(1.0);
    expect(cal.b).toBeCloseTo(0.0);
  });

  it("calibrate(0.5) equals sigmoid(0.5) with default params", () => {
    const cal = new PlattCalibrator();
    const result = cal.calibrate(0.5) as number;
    const expected = sigmoid(0.5) as number;
    expect(result).toBeCloseTo(expected, 10);
  });

  it("preserves array input shape", () => {
    const cal = new PlattCalibrator(1.0, 0.0);
    const scores = [0.0, 1.0, -1.0, 2.5];
    const result = cal.calibrate(scores);
    expect(result).toHaveLength(scores.length);
  });

  it("monotonicity: sorted scores produce sorted probabilities", () => {
    const cal = new PlattCalibrator(1.5, -0.5);
    const scores: number[] = [];
    for (let i = 0; i < 100; i++) {
      scores.push(-5.0 + (10.0 * i) / 99);
    }
    const result = cal.calibrate(scores);
    for (let i = 1; i < result.length; i++) {
      expect(result[i]!).toBeGreaterThan(result[i - 1]!);
    }
  });

  it("parameter recovery: fit on synthetic data with known a=2.0, b=-1.0", () => {
    const rng = mulberry32(42);
    const trueA = 2.0;
    const trueB = -1.0;
    const m = 2000;
    const scores: number[] = [];
    const labels: number[] = [];

    for (let i = 0; i < m; i++) {
      const s = (rng() - 0.5) * 6; // uniform [-3, 3]
      scores.push(s);
      const prob = sigmoid(trueA * s + trueB) as number;
      labels.push(rng() < prob ? 1.0 : 0.0);
    }

    const cal = new PlattCalibrator(0.5, 0.0);
    cal.fit(scores, labels, {
      learningRate: 0.01,
      maxIterations: 5000,
      tolerance: 1e-8,
    });

    expect(Math.abs(cal.a - trueA)).toBeLessThan(0.5);
    expect(Math.abs(cal.b - trueB)).toBeLessThan(0.5);
  });

  it("all outputs in (0, 1)", () => {
    const cal = new PlattCalibrator(2.0, -1.0);
    const rng = mulberry32(42);
    const scores: number[] = [];
    for (let i = 0; i < 200; i++) {
      scores.push((rng() - 0.5) * 20); // uniform [-10, 10]
    }
    const result = cal.calibrate(scores);
    for (const r of result) {
      expect(r).toBeGreaterThan(0.0);
      expect(r).toBeLessThan(1.0);
    }
  });

  it("scalar output is a number", () => {
    const cal = new PlattCalibrator(1.0, 0.0);
    const result = cal.calibrate(0.5);
    expect(typeof result).toBe("number");
  });
});

describe("IsotonicCalibrator", () => {
  it("produces monotone output after fit", () => {
    const rng = mulberry32(42);
    const m = 200;
    const scores: number[] = [];
    const labels: number[] = [];
    for (let i = 0; i < m; i++) {
      const s = rng() * 5;
      scores.push(s);
      const prob = sigmoid(2.0 * s - 3.0) as number;
      labels.push(rng() < prob ? 1.0 : 0.0);
    }

    const cal = new IsotonicCalibrator();
    cal.fit(scores, labels);

    const sortedScores = [...scores].sort((a, b) => a - b);
    const calibrated = cal.calibrate(sortedScores);
    for (let i = 1; i < calibrated.length; i++) {
      expect(calibrated[i]!).toBeGreaterThanOrEqual(calibrated[i - 1]! - 1e-12);
    }
  });

  it("interpolation: calibrate(2.5) between calibrate(2.0) and calibrate(3.0)", () => {
    const scores = [1.0, 2.0, 3.0, 4.0, 5.0];
    const labels = [0.0, 0.0, 0.5, 1.0, 1.0];

    const cal = new IsotonicCalibrator();
    cal.fit(scores, labels);

    const mid = cal.calibrate(2.5) as number;
    const low = cal.calibrate(2.0) as number;
    const high = cal.calibrate(3.0) as number;
    expect(mid).toBeGreaterThanOrEqual(low - 1e-12);
    expect(mid).toBeLessThanOrEqual(high + 1e-12);
  });

  it("extreme scores: below min returns leftmost, above max returns rightmost", () => {
    const scores = [1.0, 2.0, 3.0, 4.0, 5.0];
    const labels = [0.0, 0.2, 0.5, 0.8, 1.0];

    const cal = new IsotonicCalibrator();
    cal.fit(scores, labels);

    const below = cal.calibrate(-100.0) as number;
    const atMin = cal.calibrate(1.0) as number;
    expect(below).toBeCloseTo(atMin, 6);

    const above = cal.calibrate(100.0) as number;
    const atMax = cal.calibrate(5.0) as number;
    expect(above).toBeCloseTo(atMax, 6);
  });

  it("calibrate() before fit() throws", () => {
    const cal = new IsotonicCalibrator();
    expect(() => cal.calibrate(1.0)).toThrow(/fit/);
  });

  it("all outputs in (0, 1)", () => {
    const rng = mulberry32(42);
    const m = 200;
    const scores: number[] = [];
    const labels: number[] = [];
    for (let i = 0; i < m; i++) {
      const s = rng() * 5;
      scores.push(s);
      const prob = sigmoid(2.0 * s - 3.0) as number;
      labels.push(rng() < prob ? 1.0 : 0.0);
    }

    const cal = new IsotonicCalibrator();
    cal.fit(scores, labels);

    const testScores: number[] = [];
    for (let i = 0; i < 100; i++) {
      testScores.push(rng() * 7 - 1); // [-1, 6]
    }
    const calibrated = cal.calibrate(testScores);
    for (const c of calibrated) {
      expect(c).toBeGreaterThan(0.0);
      expect(c).toBeLessThan(1.0);
    }
  });

  it("handles ties in input scores", () => {
    const scores = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    const labels = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0];

    const cal = new IsotonicCalibrator();
    cal.fit(scores, labels);

    const result = cal.calibrate([1.0, 2.0, 3.0]);
    expect(result).toHaveLength(3);
    for (const r of result) {
      expect(r).toBeGreaterThan(0.0);
      expect(r).toBeLessThan(1.0);
    }
  });

  it("scalar output is a number", () => {
    const scores = [1.0, 2.0, 3.0];
    const labels = [0.0, 0.5, 1.0];

    const cal = new IsotonicCalibrator();
    cal.fit(scores, labels);

    const result = cal.calibrate(2.0);
    expect(typeof result).toBe("number");
  });

  it("array output has correct shape", () => {
    const scores = [1.0, 2.0, 3.0, 4.0];
    const labels = [0.0, 0.25, 0.75, 1.0];

    const cal = new IsotonicCalibrator();
    cal.fit(scores, labels);

    const testScores = [1.5, 2.5, 3.5];
    const result = cal.calibrate(testScores);
    expect(result).toHaveLength(testScores.length);
  });
});
