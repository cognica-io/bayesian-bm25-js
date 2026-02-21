//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

import { describe, expect, it } from "vitest";

import { BayesianBM25Scorer } from "../src/scorer.js";
import { BayesianProbabilityTransform } from "../src/probability.js";

const smallCorpus: string[][] = [
  ["the", "cat", "sat", "on", "the", "mat"],
  ["the", "dog", "chased", "the", "cat"],
  [
    "a",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "the",
    "lazy",
    "dog",
  ],
  ["hello", "world"],
  [
    "machine",
    "learning",
    "is",
    "a",
    "subset",
    "of",
    "artificial",
    "intelligence",
  ],
  ["the", "cat", "and", "the", "dog", "are", "friends"],
];

function createScorer(): BayesianBM25Scorer {
  const scorer = new BayesianBM25Scorer({
    k1: 1.2,
    b: 0.75,
    method: "lucene",
  });
  scorer.index(smallCorpus);
  return scorer;
}

describe("indexing", () => {
  it("computes correct document lengths", () => {
    const scorer = createScorer();
    expect(scorer.docLengths).toHaveLength(smallCorpus.length);
    const expectedLengths = smallCorpus.map((doc) => doc.length);
    for (let i = 0; i < expectedLengths.length; i++) {
      expect(scorer.docLengths[i]).toBe(expectedLengths[i]);
    }
  });

  it("computes correct average document length", () => {
    const scorer = createScorer();
    const expected =
      smallCorpus.reduce((sum, doc) => sum + doc.length, 0) /
      smallCorpus.length;
    expect(scorer.avgdl).toBeCloseTo(expected);
  });

  it("reports correct number of documents", () => {
    const scorer = createScorer();
    expect(scorer.numDocs).toBe(smallCorpus.length);
  });
});

describe("retrieve", () => {
  it("returns correct shape", () => {
    const scorer = createScorer();
    const { docIds, probabilities } = scorer.retrieve([["cat"]], 3);
    expect(docIds).toHaveLength(1);
    expect(docIds[0]).toHaveLength(3);
    expect(probabilities).toHaveLength(1);
    expect(probabilities[0]).toHaveLength(3);
  });

  it("returns probabilities in bounds [0, 1]", () => {
    const scorer = createScorer();
    const { probabilities } = scorer.retrieve([["cat"]], 6);
    for (const p of probabilities[0]!) {
      expect(p).toBeGreaterThanOrEqual(0.0);
      expect(p).toBeLessThanOrEqual(1.0);
    }
  });

  it("preserves BM25 ranking order", () => {
    const scorer = createScorer();
    const { probabilities } = scorer.retrieve([["cat"]], 6);
    const nonzeroProbs = probabilities[0]!.filter((p) => p > 0);
    if (nonzeroProbs.length > 1) {
      // Probabilities should be in descending order
      for (let i = 1; i < nonzeroProbs.length; i++) {
        expect(nonzeroProbs[i]!).toBeLessThanOrEqual(nonzeroProbs[i - 1]!);
      }
    }
  });

  it("is monotonic with fixed prior", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.5);
    const scores = [0.2, 0.5, 1.0, 2.0, 3.0];
    const fixedTf = 5.0;
    const fixedRatio = 0.5;
    const probs = t.scoreToProbability(
      scores,
      scores.map(() => fixedTf),
      scores.map(() => fixedRatio),
    );
    for (let i = 1; i < probs.length; i++) {
      expect(probs[i]!).toBeGreaterThan(probs[i - 1]!);
    }
  });

  it("handles multiple queries", () => {
    const scorer = createScorer();
    const queries = [["cat"], ["dog"], ["machine", "learning"]];
    const { docIds, probabilities } = scorer.retrieve(queries, 3);
    expect(docIds).toHaveLength(3);
    expect(probabilities).toHaveLength(3);
    for (let i = 0; i < 3; i++) {
      expect(docIds[i]).toHaveLength(3);
      expect(probabilities[i]).toHaveLength(3);
    }
  });

  it("ranks relevant documents high", () => {
    const scorer = createScorer();
    const { docIds } = scorer.retrieve([["cat"]], 3);
    const topDocs = new Set(docIds[0]);
    const catDocs = new Set([0, 1, 5]);
    let overlap = 0;
    for (const d of topDocs) {
      if (catDocs.has(d)) overlap++;
    }
    expect(overlap).toBeGreaterThanOrEqual(2);
  });
});

describe("getProbabilities", () => {
  it("returns probabilities for all documents", () => {
    const scorer = createScorer();
    const probs = scorer.getProbabilities(["cat"]);
    expect(probs).toHaveLength(smallCorpus.length);
  });

  it("returns probabilities in bounds [0, 1]", () => {
    const scorer = createScorer();
    const probs = scorer.getProbabilities(["cat"]);
    for (const p of probs) {
      expect(p).toBeGreaterThanOrEqual(0.0);
      expect(p).toBeLessThanOrEqual(1.0);
    }
  });

  it("returns nonzero for matching documents", () => {
    const scorer = createScorer();
    const probs = scorer.getProbabilities(["cat"]);
    // Documents 0, 1, 5 contain "cat"
    for (const docId of [0, 1, 5]) {
      expect(probs[docId]).toBeGreaterThan(0);
    }
  });

  it("returns zero for non-matching documents", () => {
    const scorer = createScorer();
    const probs = scorer.getProbabilities(["cat"]);
    // Document 3 ("hello world") should have zero probability
    expect(probs[3]).toBe(0.0);
  });
});

describe("error before index", () => {
  it("throws on retrieve before index", () => {
    const scorer = new BayesianBM25Scorer();
    expect(() => scorer.retrieve([["cat"]])).toThrow(/index/);
  });

  it("throws on getProbabilities before index", () => {
    const scorer = new BayesianBM25Scorer();
    expect(() => scorer.getProbabilities(["cat"])).toThrow(/index/);
  });

  it("throws on docLengths before index", () => {
    const scorer = new BayesianBM25Scorer();
    expect(() => scorer.docLengths).toThrow(/index/);
  });

  it("throws on avgdl before index", () => {
    const scorer = new BayesianBM25Scorer();
    expect(() => scorer.avgdl).toThrow(/index/);
  });
});

describe("baseRate", () => {
  it("default scorer has baseRate=null", () => {
    const scorer = createScorer();
    expect(scorer.baseRate).toBeNull();
  });

  it("explicit baseRate is stored and used", () => {
    const scorer = new BayesianBM25Scorer({
      k1: 1.2,
      b: 0.75,
      method: "lucene",
      baseRate: 0.01,
    });
    scorer.index(smallCorpus);
    expect(scorer.baseRate).toBeCloseTo(0.01);
  });

  it("auto baseRate produces a float in (0, 1)", () => {
    const scorer = new BayesianBM25Scorer({
      k1: 1.2,
      b: 0.75,
      method: "lucene",
      baseRate: "auto",
    });
    scorer.index(smallCorpus);
    expect(scorer.baseRate).not.toBeNull();
    expect(scorer.baseRate!).toBeGreaterThan(0.0);
    expect(scorer.baseRate!).toBeLessThan(1.0);
  });

  it("baseRate reduces probabilities", () => {
    const sNone = createScorer();
    const sLow = new BayesianBM25Scorer({
      k1: 1.2,
      b: 0.75,
      method: "lucene",
      baseRate: 0.01,
    });
    sLow.index(smallCorpus);

    const pNone = sNone.getProbabilities(["cat"]);
    const pLow = sLow.getProbabilities(["cat"]);
    for (let i = 0; i < pNone.length; i++) {
      if (pNone[i]! > 0) {
        expect(pLow[i]!).toBeLessThan(pNone[i]!);
      }
    }
  });

  it("baseRate preserves document ranking", () => {
    const scorer = new BayesianBM25Scorer({
      k1: 1.2,
      b: 0.75,
      method: "lucene",
      baseRate: 0.01,
    });
    scorer.index(smallCorpus);
    const { probabilities } = scorer.retrieve([["cat"]], 6);
    const nonzeroProbs = probabilities[0]!.filter((p) => p > 0);
    for (let i = 1; i < nonzeroProbs.length; i++) {
      expect(nonzeroProbs[i]!).toBeLessThanOrEqual(nonzeroProbs[i - 1]!);
    }
  });
});
