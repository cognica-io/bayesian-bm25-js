//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

import { describe, expect, it } from "vitest";

import { MultiFieldScorer } from "../src/multi_field.js";
import { BayesianBM25Scorer } from "../src/scorer.js";

const twoFieldDocs: Record<string, string[]>[] = [
  {
    title: ["cat", "sat", "mat"],
    body: ["the", "cat", "sat", "on", "the", "mat"],
  },
  {
    title: ["dog", "chased", "cat"],
    body: ["the", "dog", "chased", "the", "cat", "around"],
  },
  {
    title: ["quick", "brown", "fox"],
    body: [
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
  },
  {
    title: ["hello", "world"],
    body: ["hello", "world", "program"],
  },
  {
    title: ["machine", "learning"],
    body: [
      "machine",
      "learning",
      "is",
      "a",
      "subset",
      "of",
      "artificial",
      "intelligence",
    ],
  },
];

function createMultiScorer(): MultiFieldScorer {
  const scorer = new MultiFieldScorer({
    fields: ["title", "body"],
    k1: 1.2,
    b: 0.75,
    method: "lucene",
  });
  scorer.index(twoFieldDocs);
  return scorer;
}

describe("indexAndRetrieve", () => {
  it("basic 2-field index/retrieve returns results", () => {
    const scorer = createMultiScorer();
    const { docIds, probabilities } = scorer.retrieve(["cat"], 3);
    expect(docIds).toHaveLength(3);
    expect(probabilities).toHaveLength(3);
    for (const p of probabilities) {
      expect(p).toBeGreaterThanOrEqual(0.0);
      expect(p).toBeLessThanOrEqual(1.0);
    }
    // Cat documents (0, 1) should be in top results
    expect(docIds.includes(0) || docIds.includes(1)).toBe(true);
  });

  it("getProbabilities returns correct shape", () => {
    const scorer = createMultiScorer();
    const probs = scorer.getProbabilities(["cat"]);
    expect(probs).toHaveLength(scorer.numDocs);
    for (const p of probs) {
      expect(p).toBeGreaterThanOrEqual(0.0);
      expect(p).toBeLessThanOrEqual(1.0);
    }
  });

  it("custom field weights change the distribution", () => {
    const scorerTitle = new MultiFieldScorer({
      fields: ["title", "body"],
      fieldWeights: { title: 0.9, body: 0.1 },
      method: "lucene",
    });
    const scorerBody = new MultiFieldScorer({
      fields: ["title", "body"],
      fieldWeights: { title: 0.1, body: 0.9 },
      method: "lucene",
    });
    scorerTitle.index(twoFieldDocs);
    scorerBody.index(twoFieldDocs);

    const probsTitle = scorerTitle.getProbabilities(["cat"]);
    const probsBody = scorerBody.getProbabilities(["cat"]);

    // Different weights should produce different probability distributions
    let allEqual = true;
    for (let i = 0; i < probsTitle.length; i++) {
      if (Math.abs(probsTitle[i]! - probsBody[i]!) > 1e-8) {
        allEqual = false;
        break;
      }
    }
    expect(allEqual).toBe(false);
  });

  it("missing field raises", () => {
    const scorer = new MultiFieldScorer({
      fields: ["title", "body"],
      method: "lucene",
    });
    const docs = [{ title: ["hello"] }]; // missing "body"
    expect(() =>
      scorer.index(docs as Record<string, string[]>[]),
    ).toThrow(/missing field/);
  });

  it("single field equivalent to BayesianBM25Scorer", () => {
    const multi = new MultiFieldScorer({
      fields: ["body"],
      k1: 1.2,
      b: 0.75,
      method: "lucene",
    });
    multi.index(twoFieldDocs);

    const single = new BayesianBM25Scorer({
      k1: 1.2,
      b: 0.75,
      method: "lucene",
    });
    const bodyCorpus = twoFieldDocs.map((doc) => doc["body"]!);
    single.index(bodyCorpus);

    const query = ["cat"];
    const probsMulti = multi.getProbabilities(query);
    const probsSingle = single.getProbabilities(query);

    // With a single field and weight 1.0, results should be very close
    expect(probsMulti).toHaveLength(probsSingle.length);
    for (let i = 0; i < probsMulti.length; i++) {
      expect(probsMulti[i]).toBeCloseTo(probsSingle[i]!, 5);
    }
  });
});

describe("properties", () => {
  it("numDocs matches corpus size", () => {
    const scorer = createMultiScorer();
    expect(scorer.numDocs).toBe(twoFieldDocs.length);
  });

  it("fields returns correct list", () => {
    const scorer = createMultiScorer();
    expect(scorer.fields).toEqual(["title", "body"]);
  });

  it("default field weights are uniform", () => {
    const scorer = createMultiScorer();
    const weights = scorer.fieldWeights;
    expect(weights["title"]).toBeCloseTo(0.5);
    expect(weights["body"]).toBeCloseTo(0.5);
  });

  it("custom field weights are preserved", () => {
    const scorer = new MultiFieldScorer({
      fields: ["title", "body"],
      fieldWeights: { title: 0.7, body: 0.3 },
      method: "lucene",
    });
    scorer.index(twoFieldDocs);
    const weights = scorer.fieldWeights;
    expect(weights["title"]).toBeCloseTo(0.7);
    expect(weights["body"]).toBeCloseTo(0.3);
  });
});

describe("addDocuments", () => {
  it("increments count and finds new document", () => {
    const scorer = createMultiScorer();
    const originalCount = scorer.numDocs;
    scorer.addDocuments([
      {
        title: ["new", "cat"],
        body: ["brand", "new", "cat", "doc"],
      },
    ]);
    expect(scorer.numDocs).toBe(originalCount + 1);

    const probs = scorer.getProbabilities(["cat"]);
    const newDocId = originalCount;
    expect(probs[newDocId]).toBeGreaterThan(0);
  });

  it("throws before index", () => {
    const scorer = new MultiFieldScorer({
      fields: ["title", "body"],
    });
    expect(() =>
      scorer.addDocuments([{ title: ["a"], body: ["b"] }]),
    ).toThrow(/index/);
  });

  it("missing field in new document raises", () => {
    const scorer = createMultiScorer();
    expect(() =>
      scorer.addDocuments([
        { title: ["only", "title"] } as Record<string, string[]>,
      ]),
    ).toThrow(/missing field/);
  });
});

describe("validation", () => {
  it("empty fields raises", () => {
    expect(() => new MultiFieldScorer({ fields: [] })).toThrow(
      /non-empty/,
    );
  });

  it("duplicate fields raises", () => {
    expect(
      () => new MultiFieldScorer({ fields: ["title", "title"] }),
    ).toThrow(/duplicates/);
  });

  it("weights missing key raises", () => {
    expect(
      () =>
        new MultiFieldScorer({
          fields: ["title", "body"],
          fieldWeights: { title: 1.0 },
        }),
    ).toThrow(/missing key/);
  });

  it("weights bad sum raises", () => {
    expect(
      () =>
        new MultiFieldScorer({
          fields: ["title", "body"],
          fieldWeights: { title: 0.5, body: 0.6 },
        }),
    ).toThrow(/sum to 1/);
  });
});

describe("mainPackageExport", () => {
  it("MultiFieldScorer is importable from main package", async () => {
    const mod = await import("../src/index.js");
    expect(mod.MultiFieldScorer).toBeDefined();
  });
});
