//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

import { describe, expect, it } from "vitest";

import { BlockMaxIndex } from "../src/scorer.js";
import { BayesianProbabilityTransform } from "../src/probability.js";

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

// Generate a 2D score matrix (nTerms x nDocs) with uniform random values.
function randomScoreMatrix(
  nTerms: number,
  nDocs: number,
  maxVal: number,
  rng: () => number,
): number[][] {
  const matrix: number[][] = [];
  for (let t = 0; t < nTerms; t++) {
    const row: number[] = [];
    for (let d = 0; d < nDocs; d++) {
      row.push(rng() * maxVal);
    }
    matrix.push(row);
  }
  return matrix;
}

describe("BlockMaxIndex block count", () => {
  const cases: [number, number, number][] = [
    // [nDocs, blockSize, expectedNBlocks]
    [100, 64, 2], // 100 / 64 = 1.5625 -> ceil = 2
    [128, 64, 2], // exact multiple
    [129, 64, 3], // one over
    [1, 64, 1], // single doc
    [64, 64, 1], // exact fit
    [500, 128, 4], // 500 / 128 = 3.906 -> ceil = 4
    [1000, 256, 4], // 1000 / 256 = 3.906 -> ceil = 4
  ];

  for (const [nDocs, blockSize, expected] of cases) {
    it(`ceil(${nDocs} / ${blockSize}) = ${expected}`, () => {
      const scoreMatrix: number[][] = [];
      for (let t = 0; t < 2; t++) {
        scoreMatrix.push(new Array(nDocs).fill(0));
      }
      const bmi = new BlockMaxIndex(blockSize);
      bmi.build(scoreMatrix);
      expect(bmi.nBlocks).toBe(expected);
    });
  }
});

describe("BlockMaxIndex block upper bound", () => {
  it("block upper bound >= all actual scores in that block", () => {
    const rng = mulberry32(42);
    const nTerms = 5;
    const nDocs = 200;
    const blockSize = 64;
    const scoreMatrix = randomScoreMatrix(nTerms, nDocs, 10.0, rng);

    const idx = new BlockMaxIndex(blockSize);
    idx.build(scoreMatrix);

    const nBlocks = idx.nBlocks;
    for (let t = 0; t < nTerms; t++) {
      for (let b = 0; b < nBlocks; b++) {
        const bound = idx.blockUpperBound(t, b);
        const start = b * blockSize;
        const end = Math.min(start + blockSize, nDocs);
        // Check every individual doc
        for (let d = start; d < end; d++) {
          expect(bound).toBeGreaterThanOrEqual(scoreMatrix[t]![d]! - 1e-12);
        }
      }
    }
  });

  it("block upper bound <= global max per term (tighter than WAND)", () => {
    const rng = mulberry32(99);
    const nTerms = 4;
    const nDocs = 300;
    const blockSize = 64;
    const scoreMatrix = randomScoreMatrix(nTerms, nDocs, 8.0, rng);

    const idx = new BlockMaxIndex(blockSize);
    idx.build(scoreMatrix);

    for (let t = 0; t < nTerms; t++) {
      let globalMax = -Infinity;
      for (let d = 0; d < nDocs; d++) {
        if (scoreMatrix[t]![d]! > globalMax) {
          globalMax = scoreMatrix[t]![d]!;
        }
      }
      for (let b = 0; b < idx.nBlocks; b++) {
        const blockBound = idx.blockUpperBound(t, b);
        expect(blockBound).toBeLessThanOrEqual(globalMax + 1e-12);
      }
    }
  });
});

describe("BlockMaxIndex Bayesian block upper bound", () => {
  it("bayesian block upper bound >= actual Bayesian probability for all docs", () => {
    const rng = mulberry32(7);
    const nTerms = 3;
    const nDocs = 200;
    const blockSize = 32;
    const scoreMatrix = randomScoreMatrix(nTerms, nDocs, 6.0, rng);

    const transform = new BayesianProbabilityTransform(1.5, 2.0);

    const bmi = new BlockMaxIndex(blockSize);
    bmi.build(scoreMatrix);

    for (let t = 0; t < nTerms; t++) {
      for (let b = 0; b < bmi.nBlocks; b++) {
        const bayesianBound = bmi.bayesianBlockUpperBound(
          t,
          b,
          transform,
          0.9,
        );
        const start = b * blockSize;
        const end = Math.min(start + blockSize, nDocs);
        for (let d = start; d < end; d++) {
          const rawScore = scoreMatrix[t]![d]!;
          // Check with various tf/docLenRatio combinations
          for (const tf of [0, 1, 5, 10]) {
            for (const ratio of [0.1, 0.5, 1.0, 2.0]) {
              const actualProb = transform.scoreToProbability(
                rawScore,
                tf,
                ratio,
              ) as number;
              expect(bayesianBound).toBeGreaterThanOrEqual(
                actualProb - 1e-10,
              );
            }
          }
        }
      }
    }
  });
});

describe("BlockMaxIndex build() before access check", () => {
  it("nBlocks before build() throws", () => {
    const bmi = new BlockMaxIndex(64);
    expect(() => bmi.nBlocks).toThrow(/build/);
  });

  it("blockUpperBound before build() throws", () => {
    const bmi = new BlockMaxIndex(64);
    expect(() => bmi.blockUpperBound(0, 0)).toThrow(/build/);
  });
});

describe("BlockMaxIndex single block edge case", () => {
  it("nDocs < blockSize produces single block", () => {
    const rng = mulberry32(55);
    const blockSize = 128;
    const nDocs = 50; // fewer docs than blockSize
    const nTerms = 3;
    const scoreMatrix = randomScoreMatrix(nTerms, nDocs, 5.0, rng);

    const bmi = new BlockMaxIndex(blockSize);
    bmi.build(scoreMatrix);

    expect(bmi.nBlocks).toBe(1);

    // The single block bound must equal the global max per term
    for (let t = 0; t < nTerms; t++) {
      const blockBound = bmi.blockUpperBound(t, 0);
      let globalMax = -Infinity;
      for (let d = 0; d < nDocs; d++) {
        if (scoreMatrix[t]![d]! > globalMax) {
          globalMax = scoreMatrix[t]![d]!;
        }
      }
      expect(blockBound).toBeCloseTo(globalMax);
    }
  });
});

describe("BlockMaxIndex blockSize validation", () => {
  it("blockSize = 0 throws", () => {
    expect(() => new BlockMaxIndex(0)).toThrow(/block_size must be >= 1/);
  });

  it("negative blockSize throws", () => {
    expect(() => new BlockMaxIndex(-5)).toThrow(/block_size must be >= 1/);
  });

  it("blockSize property returns correct value", () => {
    for (const bs of [1, 32, 64, 128, 256, 1024]) {
      const bmi = new BlockMaxIndex(bs);
      expect(bmi.blockSize).toBe(bs);
    }
  });
});
