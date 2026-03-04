//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

// Multi-field BM25 scorer with Bayesian probability fusion.
//
// Manages per-field BayesianBM25Scorer instances and fuses field-level
// probabilities via logOddsConjunction.  This enables first-class
// multi-field search (e.g., title + body) with calibrated output.

import { logOddsConjunction, resolveAlpha } from "./fusion.js";
import { BayesianBM25Scorer } from "./scorer.js";

export interface MultiFieldScorerOptions {
  fields: string[];
  fieldWeights?: Record<string, number>;
  alpha?: number | "auto";
  baseRate?: number | "auto" | null;
  k1?: number;
  b?: number;
  method?: "robertson" | "lucene" | "atire";
}

// Multi-field BM25 scorer that fuses per-field Bayesian probabilities.
export class MultiFieldScorer {
  private _fields: string[];
  private _fieldWeights: Record<string, number>;
  private _alpha: number | "auto" | undefined;
  private _baseRate: number | "auto" | null;
  private _k1: number;
  private _b: number;
  private _method: "robertson" | "lucene" | "atire";
  private _scorers: Record<string, BayesianBM25Scorer> = {};
  private _numDocs: number = 0;

  constructor(options: MultiFieldScorerOptions) {
    const { fields } = options;
    if (fields.length === 0) {
      throw new Error("fields must be a non-empty list");
    }
    if (new Set(fields).size !== fields.length) {
      throw new Error("fields must not contain duplicates");
    }

    this._fields = [...fields];
    this._alpha = options.alpha ?? "auto";
    this._baseRate = options.baseRate ?? null;
    this._k1 = options.k1 ?? 1.2;
    this._b = options.b ?? 0.75;
    this._method = options.method ?? "robertson";

    // Resolve field weights
    if (options.fieldWeights === undefined) {
      const n = fields.length;
      this._fieldWeights = {};
      for (const f of fields) {
        this._fieldWeights[f] = 1.0 / n;
      }
    } else {
      for (const f of fields) {
        if (!(f in options.fieldWeights)) {
          throw new Error(`fieldWeights missing key "${f}"`);
        }
      }
      let weightSum = 0;
      for (const f of fields) {
        weightSum += options.fieldWeights[f]!;
      }
      if (Math.abs(weightSum - 1.0) > 1e-6) {
        throw new Error(
          `fieldWeights must sum to 1, got ${weightSum}`,
        );
      }
      this._fieldWeights = {};
      for (const f of fields) {
        this._fieldWeights[f] = options.fieldWeights[f]!;
      }
    }
  }

  get numDocs(): number {
    return this._numDocs;
  }

  get fields(): string[] {
    return [...this._fields];
  }

  get fieldWeights(): Record<string, number> {
    return { ...this._fieldWeights };
  }

  // Build per-field BM25 indexes.
  //
  // Each document is a Record mapping field name to a list of tokens.
  // Every document must contain all fields.
  index(documents: Record<string, string[]>[]): void {
    for (let i = 0; i < documents.length; i++) {
      for (const field of this._fields) {
        if (!(field in documents[i]!)) {
          throw new Error(`Document ${i} missing field "${field}"`);
        }
      }
    }

    this._scorers = {};
    for (const field of this._fields) {
      const scorer = new BayesianBM25Scorer({
        k1: this._k1,
        b: this._b,
        method: this._method,
        baseRate: this._baseRate,
      });
      const fieldCorpus = documents.map((doc) => doc[field]!);
      scorer.index(fieldCorpus);
      this._scorers[field] = scorer;
    }

    this._numDocs = documents.length;
  }

  // Get fused probabilities for all documents (dense array).
  getProbabilities(queryTokens: string[]): number[] {
    if (Object.keys(this._scorers).length === 0) {
      throw new Error("Call index() before getProbabilities().");
    }

    // Collect per-field probabilities: shape (numDocs, numFields)
    const fieldProbArrays: number[][] = [];
    for (const field of this._fields) {
      fieldProbArrays.push(
        this._scorers[field]!.getProbabilities(queryTokens),
      );
    }

    // Transpose to (numDocs, numFields)
    const nDocs = fieldProbArrays[0]!.length;
    const nFields = this._fields.length;
    const fieldProbs: number[][] = [];
    for (let d = 0; d < nDocs; d++) {
      const row: number[] = [];
      for (let f = 0; f < nFields; f++) {
        row.push(fieldProbArrays[f]![d]!);
      }
      fieldProbs.push(row);
    }

    // Build weights array in field order
    const weights = this._fields.map((f) => this._fieldWeights[f]!);

    // Resolve alpha for the conjunction
    const effectiveAlpha = resolveAlpha(this._alpha, 0.5);

    return logOddsConjunction(
      fieldProbs,
      effectiveAlpha,
      weights,
    ) as number[];
  }

  // Retrieve top-k documents by fused probability.
  retrieve(
    queryTokens: string[],
    k: number = 10,
  ): { docIds: number[]; probabilities: number[] } {
    const probs = this.getProbabilities(queryTokens);
    const effectiveK = Math.min(k, probs.length);

    // Get indices sorted by descending probability
    const indices = Array.from({ length: probs.length }, (_, i) => i);
    indices.sort((a, b) => probs[b]! - probs[a]!);
    const topK = indices.slice(0, effectiveK);

    return {
      docIds: topK,
      probabilities: topK.map((i) => probs[i]!),
    };
  }

  // Add documents to the multi-field index.
  //
  // Since the underlying BM25 engine requires IDF recomputation,
  // this method appends the new documents and rebuilds per-field indexes.
  addDocuments(newDocuments: Record<string, string[]>[]): void {
    if (Object.keys(this._scorers).length === 0) {
      throw new Error("Call index() before addDocuments().");
    }

    for (let i = 0; i < newDocuments.length; i++) {
      for (const field of this._fields) {
        if (!(field in newDocuments[i]!)) {
          throw new Error(
            `New document ${i} missing field "${field}"`,
          );
        }
      }
    }

    for (const field of this._fields) {
      const newFieldCorpus = newDocuments.map((doc) => doc[field]!);
      this._scorers[field]!.addDocuments(newFieldCorpus);
    }

    this._numDocs += newDocuments.length;
  }
}
