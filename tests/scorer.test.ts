//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

import { describe, expect, it } from "vitest";

import {
  BayesianBM25Scorer,
  type RetrievalResult,
} from "../src/scorer.js";
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

describe("baseRateMethod", () => {
  it("invalid method raises", () => {
    expect(
      () =>
        new BayesianBM25Scorer({
          baseRateMethod: "invalid" as "percentile",
        }),
    ).toThrow(/baseRateMethod/);
  });

  it("percentile is default and produces valid base rate", () => {
    const s = new BayesianBM25Scorer({
      method: "lucene",
      baseRate: "auto",
    });
    s.index(smallCorpus);
    expect(s.baseRate).not.toBeNull();
    expect(s.baseRate!).toBeGreaterThan(0.0);
    expect(s.baseRate!).toBeLessThan(1.0);
  });

  it("mixture method produces valid base rate", () => {
    const s = new BayesianBM25Scorer({
      method: "lucene",
      baseRate: "auto",
      baseRateMethod: "mixture",
    });
    s.index(smallCorpus);
    expect(s.baseRate).not.toBeNull();
    expect(s.baseRate!).toBeGreaterThan(0.0);
    expect(s.baseRate!).toBeLessThanOrEqual(0.5);
  });

  it("elbow method produces valid base rate", () => {
    const s = new BayesianBM25Scorer({
      method: "lucene",
      baseRate: "auto",
      baseRateMethod: "elbow",
    });
    s.index(smallCorpus);
    expect(s.baseRate).not.toBeNull();
    expect(s.baseRate!).toBeGreaterThan(0.0);
    expect(s.baseRate!).toBeLessThanOrEqual(0.5);
  });

  it("mixture EM recovers sensible base rate from bimodal data", () => {
    // Seeded PRNG for deterministic normal variates via Box-Muller
    let seed = 42;
    const rng = () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const normalSample = (mean: number, std: number): number => {
      const u1 = rng();
      const u2 = rng();
      const z = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-12))) * Math.cos(2 * Math.PI * u2);
      return mean + std * z;
    };

    // Low-scoring (non-relevant): mean=1, std=0.5  (900 samples)
    // High-scoring (relevant):    mean=5, std=0.5  (100 samples)
    const scores: number[] = [];
    for (let i = 0; i < 900; i++) {
      const s = normalSample(1.0, 0.5);
      if (s > 0) scores.push(s);
    }
    for (let i = 0; i < 100; i++) {
      const s = normalSample(5.0, 0.5);
      if (s > 0) scores.push(s);
    }

    const result = BayesianBM25Scorer._baseRateMixture([scores]);
    expect(result).toBeGreaterThan(0.01);
    expect(result).toBeLessThan(0.5);
  });

  it("elbow finds knee in clearly kinked distribution", () => {
    const high = new Array(10).fill(10.0) as number[];
    const low: number[] = [];
    for (let i = 0; i < 90; i++) {
      low.push(2.0 - (1.9 * i) / 89);
    }
    const scores = [...high, ...low];
    const result = BayesianBM25Scorer._baseRateElbow([scores]);
    expect(result).toBeGreaterThan(0.01);
    expect(result).toBeLessThan(0.5);
  });

  it("mixture with < 2 scores returns 1e-6", () => {
    const result = BayesianBM25Scorer._baseRateMixture([[1.0]]);
    expect(result).toBeCloseTo(1e-6);
  });

  it("elbow with < 3 scores returns 1e-6", () => {
    const result = BayesianBM25Scorer._baseRateElbow([[1.0, 2.0]]);
    expect(result).toBeCloseTo(1e-6);
  });

  it("baseRateMethod is irrelevant when baseRate is explicit", () => {
    const s = new BayesianBM25Scorer({
      method: "lucene",
      baseRate: 0.01,
      baseRateMethod: "mixture",
    });
    s.index(smallCorpus);
    expect(s.baseRate).toBeCloseTo(0.01);
  });

  it("all methods produce bounded results", () => {
    // Deterministic exponential-like samples
    let seed = 123;
    const rng = () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const scores: number[] = [];
    for (let i = 0; i < 500; i++) {
      const s = -2.0 * Math.log(Math.max(rng(), 1e-12));
      if (s > 0) scores.push(s);
    }
    const perQueryScores = [scores];

    const methods = [
      () => BayesianBM25Scorer._baseRatePercentile(perQueryScores, 1000),
      () => BayesianBM25Scorer._baseRateMixture(perQueryScores),
      () => BayesianBM25Scorer._baseRateElbow(perQueryScores),
    ];

    for (const methodFn of methods) {
      const result = methodFn();
      expect(result).toBeGreaterThanOrEqual(1e-6);
      expect(result).toBeLessThanOrEqual(0.5);
    }
  });
});

describe("addDocuments", () => {
  it("increases document count", () => {
    const scorer = createScorer();
    const originalCount = scorer.numDocs;
    scorer.addDocuments([["new", "document", "here"]]);
    expect(scorer.numDocs).toBe(originalCount + 1);
  });

  it("throws before index", () => {
    const s = new BayesianBM25Scorer();
    expect(() => s.addDocuments([["hello"]])).toThrow(/index/);
  });

  it("preserves search for old documents", () => {
    const s = new BayesianBM25Scorer({
      k1: 1.2,
      b: 0.75,
      method: "lucene",
    });
    s.index(smallCorpus);
    const probsBefore = s.getProbabilities(["cat"]);
    const catDocsBefore = new Set<number>();
    for (let i = 0; i < probsBefore.length; i++) {
      if (probsBefore[i]! > 0) catDocsBefore.add(i);
    }

    s.addDocuments([["completely", "unrelated", "tokens"]]);
    const probsAfter = s.getProbabilities(["cat"]);
    const catDocsAfter = new Set<number>();
    for (let i = 0; i < probsAfter.length; i++) {
      if (probsAfter[i]! > 0) catDocsAfter.add(i);
    }

    // Original cat docs should still appear
    for (const docId of catDocsBefore) {
      expect(catDocsAfter.has(docId)).toBe(true);
    }
  });

  it("finds newly added documents", () => {
    const s = new BayesianBM25Scorer({
      k1: 1.2,
      b: 0.75,
      method: "lucene",
    });
    s.index(smallCorpus);
    const newDocId = smallCorpus.length;

    s.addDocuments([["cat", "cat", "cat", "cat", "cat"]]);
    const probs = s.getProbabilities(["cat"]);
    expect(probs[newDocId]).toBeGreaterThan(0);
  });
});

describe("RetrievalResult", () => {
  it("default returns RetrieveResult (no explanations)", () => {
    const scorer = createScorer();
    const result = scorer.retrieve([["cat"]], 3);
    expect(result.docIds).toHaveLength(1);
    expect(result.docIds[0]).toHaveLength(3);
    expect(result.probabilities).toHaveLength(1);
    expect(result.probabilities[0]).toHaveLength(3);
    expect("explanations" in result).toBe(false);
  });

  it("explain=true returns RetrievalResult", () => {
    const scorer = createScorer();
    const result: RetrievalResult = scorer.retrieve([["cat"]], 3, true);
    expect(result.docIds).toHaveLength(1);
    expect(result.docIds[0]).toHaveLength(3);
    expect(result.probabilities).toHaveLength(1);
    expect(result.probabilities[0]).toHaveLength(3);
    expect(result.explanations).not.toBeNull();
  });

  it("explanations has correct shape", () => {
    const scorer = createScorer();
    const result = scorer.retrieve([["cat"], ["dog"]], 3, true);
    expect(result.explanations).toHaveLength(2);
    for (const qExplanations of result.explanations!) {
      expect(qExplanations).toHaveLength(3);
    }
  });

  it("trace posteriors match probabilities for nonzero docs", () => {
    const scorer = createScorer();
    const result = scorer.retrieve([["cat"]], 6, true);
    for (let rank = 0; rank < 6; rank++) {
      const prob = result.probabilities[0]![rank]!;
      const trace = result.explanations![0]![rank];
      if (prob > 0) {
        expect(trace).not.toBeNull();
        expect(Math.abs(trace!.posterior - prob)).toBeLessThan(1e-6);
      } else {
        expect(trace).toBeNull();
      }
    }
  });
});
