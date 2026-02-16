//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

// BM25 search engine implementation.
//
// Provides term-level inverted indexing and BM25 scoring with support for
// three IDF variants: Robertson, Lucene, and ATIRE.

export type BM25Method = "robertson" | "lucene" | "atire";

export interface BM25Options {
  k1?: number;
  b?: number;
  method?: BM25Method;
}

interface PostingEntry {
  docId: number;
  tf: number;
}

export interface RetrieveResult {
  documents: number[][];
  scores: number[][];
}

export class BM25 {
  readonly k1: number;
  readonly b: number;
  readonly method: BM25Method;

  private _numDocs: number = 0;
  private _avgdl: number = 0;
  private _docLengths: number[] = [];
  private _invertedIndex: Map<string, PostingEntry[]> = new Map();
  private _idf: Map<string, number> = new Map();
  private _indexed: boolean = false;

  constructor(options: BM25Options = {}) {
    this.k1 = options.k1 ?? 1.2;
    this.b = options.b ?? 0.75;
    this.method = options.method ?? "robertson";
  }

  get numDocs(): number {
    return this._numDocs;
  }

  index(corpusTokens: string[][]): void {
    this._numDocs = corpusTokens.length;
    this._docLengths = corpusTokens.map((doc) => doc.length);

    let totalLength = 0;
    for (const len of this._docLengths) {
      totalLength += len;
    }
    this._avgdl = this._numDocs > 0 ? totalLength / this._numDocs : 0;

    // Build inverted index
    this._invertedIndex.clear();
    for (let docId = 0; docId < corpusTokens.length; docId++) {
      const tokens = corpusTokens[docId]!;
      const termFreqs = new Map<string, number>();
      for (const token of tokens) {
        termFreqs.set(token, (termFreqs.get(token) ?? 0) + 1);
      }
      for (const [term, tf] of termFreqs) {
        let postings = this._invertedIndex.get(term);
        if (!postings) {
          postings = [];
          this._invertedIndex.set(term, postings);
        }
        postings.push({ docId, tf });
      }
    }

    // Compute IDF for each term
    this._idf.clear();
    for (const [term, postings] of this._invertedIndex) {
      const df = postings.length;
      this._idf.set(term, this._computeIDF(df));
    }

    this._indexed = true;
  }

  private _computeIDF(df: number): number {
    const n = this._numDocs;
    switch (this.method) {
      case "robertson":
        return Math.log((n - df + 0.5) / (df + 0.5) + 1.0);
      case "lucene":
        return Math.log(1.0 + (n - df + 0.5) / (df + 0.5));
      case "atire":
        return Math.log(n / df);
      default:
        return Math.log((n - df + 0.5) / (df + 0.5) + 1.0);
    }
  }

  // Compute BM25 scores for all documents given a query.
  getScores(queryTokens: string[]): number[] {
    this._ensureIndexed();

    const scores = new Array<number>(this._numDocs).fill(0);

    for (const token of queryTokens) {
      const idf = this._idf.get(token);
      if (idf === undefined) continue;

      const postings = this._invertedIndex.get(token);
      if (!postings) continue;

      for (const { docId, tf } of postings) {
        const dl = this._docLengths[docId]!;
        const tfNorm =
          (tf * (this.k1 + 1)) /
          (tf + this.k1 * (1.0 - this.b + this.b * (dl / this._avgdl)));
        scores[docId]! += idf * tfNorm;
      }
    }

    return scores;
  }

  // Retrieve top-k documents for each query.
  retrieve(queryTokensBatch: string[][], k: number): RetrieveResult {
    this._ensureIndexed();

    const documents: number[][] = [];
    const scores: number[][] = [];

    for (const queryTokens of queryTokensBatch) {
      const allScores = this.getScores(queryTokens);

      // Create index-score pairs and sort descending by score
      const indexed = allScores.map((score, idx) => ({ idx, score }));
      indexed.sort((a, b) => b.score - a.score);

      const topK = indexed.slice(0, k);
      documents.push(topK.map((e) => e.idx));
      scores.push(topK.map((e) => e.score));
    }

    return { documents, scores };
  }

  private _ensureIndexed(): void {
    if (!this._indexed) {
      throw new Error("Call index() before querying.");
    }
  }
}
