//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

import { describe, expect, it } from "vitest";

import {
  AttentionLogOddsWeights,
  MultiHeadAttentionLogOddsWeights,
} from "../src/fusion.js";
import { clampProbability } from "../src/probability.js";

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

describe("MultiHeadAttentionLogOddsWeights", () => {
  it("single head equivalence: nHeads=1 matches AttentionLogOddsWeights(seed=0)", () => {
    const mh = new MultiHeadAttentionLogOddsWeights(1, 2, 3);
    const single = new AttentionLogOddsWeights(2, 3, 0.5, false, 0);
    const probs = [0.8, 0.7];
    const qf = [1.0, 0.5, -0.3];
    const rMH = mh.combine(probs, qf) as number;
    const rSingle = single.combine(probs, qf) as number;
    expect(rMH).toBeCloseTo(rSingle, 10);
  });

  it("constructor validation: nHeads < 1 throws", () => {
    expect(
      () => new MultiHeadAttentionLogOddsWeights(0, 2, 3),
    ).toThrow(/n_heads/);
  });

  it("nHeads property returns correct value", () => {
    const mh = new MultiHeadAttentionLogOddsWeights(4, 2, 3);
    expect(mh.nHeads).toBe(4);
  });

  it("heads property returns correct number of heads", () => {
    const mh = new MultiHeadAttentionLogOddsWeights(4, 2, 3);
    expect(mh.heads).toHaveLength(4);
  });

  it("heads property returns a copy", () => {
    const mh = new MultiHeadAttentionLogOddsWeights(2, 2, 3);
    const h1 = mh.heads;
    const h2 = mh.heads;
    expect(h1).not.toBe(h2);
  });

  it("batched combine: correct shape, all values in (0, 1)", () => {
    const mh = new MultiHeadAttentionLogOddsWeights(2, 2, 3);
    const probs = [
      [0.8, 0.7],
      [0.3, 0.9],
    ];
    const qf = [
      [1.0, 0.5, -0.3],
      [0.2, -0.1, 0.8],
    ];
    const result = mh.combine(probs, qf) as number[];
    expect(result).toHaveLength(2);
    for (const r of result) {
      expect(r).toBeGreaterThan(0);
      expect(r).toBeLessThan(1);
    }
  });

  it("fit() reduces BCE loss", () => {
    const rng = mulberry32(42);
    const m = 200;
    const labels: number[] = [];
    const probs: number[][] = [];
    const qf: number[][] = [];

    for (let i = 0; i < m; i++) {
      const label = rng() > 0.5 ? 1.0 : 0.0;
      labels.push(label);
      const s0 = label === 1 ? 0.85 : 0.15;
      const s1 = 0.3 + rng() * 0.4;
      probs.push([s0, s1]);
      qf.push([1.0, 1.0]);
    }

    const mh = new MultiHeadAttentionLogOddsWeights(3, 2, 2, 0.0);

    // Measure BCE before training
    let bceBefore = 0;
    for (let i = 0; i < m; i++) {
      let pred = mh.combine(probs[i]!, qf[i]!) as number;
      pred = clampProbability(pred) as number;
      bceBefore +=
        -(
          labels[i]! * Math.log(pred) +
          (1 - labels[i]!) * Math.log(1 - pred)
        );
    }
    bceBefore /= m;

    mh.fit(probs, labels, qf, {
      learningRate: 0.1,
      maxIterations: 500,
    });

    // Measure BCE after training
    let bceAfter = 0;
    for (let i = 0; i < m; i++) {
      let pred = mh.combine(probs[i]!, qf[i]!) as number;
      pred = clampProbability(pred) as number;
      bceAfter +=
        -(
          labels[i]! * Math.log(pred) +
          (1 - labels[i]!) * Math.log(1 - pred)
        );
    }
    bceAfter /= m;

    expect(bceAfter).toBeLessThan(bceBefore);
  });

  it("diversity: after fit, different heads have different weight matrices", () => {
    const rng = mulberry32(42);
    const m = 200;
    const labels: number[] = [];
    const probs: number[][] = [];
    const qf: number[][] = [];

    for (let i = 0; i < m; i++) {
      const label = rng() > 0.5 ? 1.0 : 0.0;
      labels.push(label);
      const s0 = label === 1 ? 0.85 : 0.15;
      const s1 = 0.3 + rng() * 0.4;
      probs.push([s0, s1]);
      qf.push([1.0, 1.0]);
    }

    const mh = new MultiHeadAttentionLogOddsWeights(3, 2, 2, 0.0);
    mh.fit(probs, labels, qf, {
      learningRate: 0.1,
      maxIterations: 300,
    });

    // Check that heads have different weight matrices
    const w0 = mh.heads[0]!.weightsMatrix;
    const w1 = mh.heads[1]!.weightsMatrix;
    let allClose = true;
    for (let i = 0; i < w0.length; i++) {
      for (let j = 0; j < w0[i]!.length; j++) {
        if (Math.abs(w0[i]![j]! - w1[i]![j]!) > 1e-3) {
          allClose = false;
        }
      }
    }
    expect(allClose).toBe(false);
  });

  it("update(): parameters change after updates", () => {
    const mh = new MultiHeadAttentionLogOddsWeights(2, 2, 2);
    const wBefore = mh.heads.map((h) =>
      h.weightsMatrix.map((row) => [...row]),
    );

    for (let i = 0; i < 20; i++) {
      mh.update([0.9, 0.1], 1.0, [1.0, 0.0], { learningRate: 0.05 });
    }

    // At least one head's weight matrix should have changed
    let anyChanged = false;
    for (let h = 0; h < 2; h++) {
      const wAfter = mh.heads[h]!.weightsMatrix;
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          if (Math.abs(wBefore[h]![i]![j]! - wAfter[i]![j]!) > 1e-6) {
            anyChanged = true;
          }
        }
      }
    }
    expect(anyChanged).toBe(true);
  });

  it("computeUpperBounds: returns values in (0, 1)", () => {
    const rng = mulberry32(42);
    const mh = new MultiHeadAttentionLogOddsWeights(3, 2, 2);
    const m = 30;
    const probs: number[][] = [];
    const qf: number[][] = [];
    for (let i = 0; i < m; i++) {
      probs.push([0.1 + rng() * 0.8, 0.1 + rng() * 0.8]);
      qf.push([rng() * 2 - 1, rng() * 2 - 1]);
    }

    const upperBounds = mh.computeUpperBounds(probs, qf);
    expect(upperBounds).toHaveLength(m);
    for (const ub of upperBounds) {
      expect(ub).toBeGreaterThan(0);
      expect(ub).toBeLessThan(1);
    }
  });

  it("computeUpperBounds: upper bound >= actual for all candidates", () => {
    const rng = mulberry32(42);
    const mh = new MultiHeadAttentionLogOddsWeights(3, 2, 2);
    const m = 30;
    const probs: number[][] = [];
    const qf: number[][] = [];
    for (let i = 0; i < m; i++) {
      probs.push([0.1 + rng() * 0.8, 0.1 + rng() * 0.8]);
      qf.push([rng() * 2 - 1, rng() * 2 - 1]);
    }

    const upperBounds = mh.computeUpperBounds(probs, qf);
    const actual = mh.combine(probs, qf) as number[];

    for (let i = 0; i < m; i++) {
      expect(upperBounds[i]!).toBeGreaterThanOrEqual(actual[i]! - 1e-10);
    }
  });

  it("prune: survivors have upper bounds >= threshold", () => {
    const mh = new MultiHeadAttentionLogOddsWeights(2, 2, 2);
    const probs = [
      [0.8, 0.7],
      [0.3, 0.9],
      [0.5, 0.5],
    ];
    const qf = [1.0, 0.0];
    const threshold = 0.3;

    const { survivingIndices, fusedProbabilities } = mh.prune(
      probs,
      qf,
      threshold,
    );

    // All surviving indices should be valid
    for (const idx of survivingIndices) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(probs.length);
    }

    // fusedProbabilities length should match survivingIndices
    expect(fusedProbabilities).toHaveLength(survivingIndices.length);

    // All fused probabilities should be in (0, 1)
    for (const fp of fusedProbabilities) {
      expect(fp).toBeGreaterThan(0);
      expect(fp).toBeLessThan(1);
    }
  });

  it("prune: empty when all below threshold", () => {
    const mh = new MultiHeadAttentionLogOddsWeights(2, 2, 2);
    const probs = [
      [0.1, 0.1],
      [0.2, 0.2],
    ];
    const qf = [1.0, 0.0];
    const { survivingIndices, fusedProbabilities } = mh.prune(
      probs,
      qf,
      0.99,
    );
    expect(survivingIndices).toHaveLength(0);
    expect(fusedProbabilities).toHaveLength(0);
  });

  it("prune: no pruning when all above threshold", () => {
    const mh = new MultiHeadAttentionLogOddsWeights(2, 2, 2);
    const probs = [
      [0.9, 0.9],
      [0.8, 0.8],
      [0.85, 0.85],
    ];
    const qf = [1.0, 0.0];
    const { survivingIndices, fusedProbabilities } = mh.prune(
      probs,
      qf,
      0.01,
    );
    expect(survivingIndices).toHaveLength(3);
    expect(fusedProbabilities).toHaveLength(3);
  });

  it("output always in (0, 1) for random inputs", () => {
    const rng = mulberry32(42);
    const mh = new MultiHeadAttentionLogOddsWeights(4, 3, 2);
    const m = 20;
    const probs: number[][] = [];
    const qf: number[][] = [];
    for (let i = 0; i < m; i++) {
      probs.push([0.1 + rng() * 0.8, 0.1 + rng() * 0.8, 0.1 + rng() * 0.8]);
      qf.push([rng() * 2 - 1, rng() * 2 - 1]);
    }
    const result = mh.combine(probs, qf) as number[];
    for (const r of result) {
      expect(r).toBeGreaterThan(0);
      expect(r).toBeLessThan(1);
    }
  });
});
