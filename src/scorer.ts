//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

// BM25 scorer with Bayesian probability transforms.
//
// Integrates the built-in BM25 engine with the Bayesian probability
// framework to return calibrated relevance probabilities instead of raw
// BM25 scores.

import { BM25, type BM25Method } from "./bm25.js";
import { BayesianProbabilityTransform } from "./probability.js";

export interface BayesianBM25ScorerOptions {
  k1?: number;
  b?: number;
  method?: BM25Method;
  alpha?: number;
  beta?: number;
}

export interface RetrieveResult {
  docIds: number[][];
  probabilities: number[][];
}

// Seeded PRNG (mulberry32) for deterministic parameter estimation.
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleWithoutReplacement(
  n: number,
  size: number,
  rng: () => number,
): number[] {
  const arr = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j]!, arr[i]!];
  }
  return arr.slice(0, size);
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1]! + sorted[mid]!) / 2;
  }
  return sorted[mid]!;
}

function stddev(values: number[]): number {
  const n = values.length;
  if (n === 0) return 0;
  let sum = 0;
  for (const v of values) {
    sum += v;
  }
  const mean = sum / n;
  let sumSq = 0;
  for (const v of values) {
    sumSq += (v - mean) ** 2;
  }
  return Math.sqrt(sumSq / n);
}

// BM25 scorer that returns Bayesian-calibrated probabilities.
export class BayesianBM25Scorer {
  private _bm25: BM25;
  private _userAlpha: number | undefined;
  private _userBeta: number | undefined;
  private _transform: BayesianProbabilityTransform | null = null;
  private _docLengths: number[] | null = null;
  private _avgdl: number | null = null;
  private _corpusTokens: string[][] | null = null;

  constructor(options: BayesianBM25ScorerOptions = {}) {
    this._bm25 = new BM25({
      k1: options.k1,
      b: options.b,
      method: options.method,
    });
    this._userAlpha = options.alpha;
    this._userBeta = options.beta;
  }

  get numDocs(): number {
    return this._bm25.numDocs;
  }

  get docLengths(): number[] {
    if (this._docLengths === null) {
      throw new Error("Call index() before accessing docLengths.");
    }
    return this._docLengths;
  }

  get avgdl(): number {
    if (this._avgdl === null) {
      throw new Error("Call index() before accessing avgdl.");
    }
    return this._avgdl;
  }

  // Build the BM25 index and compute document statistics.
  index(corpusTokens: string[][]): void {
    this._corpusTokens = corpusTokens;
    this._bm25.index(corpusTokens);

    this._docLengths = corpusTokens.map((doc) => doc.length);

    let totalLength = 0;
    for (const len of this._docLengths) {
      totalLength += len;
    }
    this._avgdl =
      this._docLengths.length > 0
        ? totalLength / this._docLengths.length
        : 0;

    const [alpha, beta] = this._estimateParameters(corpusTokens);
    this._transform = new BayesianProbabilityTransform(alpha, beta);
  }

  private _estimateParameters(
    corpusTokens: string[][],
  ): [number, number] {
    if (this._userAlpha !== undefined && this._userBeta !== undefined) {
      return [this._userAlpha, this._userBeta];
    }

    const n = corpusTokens.length;
    const sampleSize = Math.min(n, 50);
    const rng = mulberry32(42);
    const sampleIndices = sampleWithoutReplacement(n, sampleSize, rng);

    const allScores: number[] = [];
    for (const idx of sampleIndices) {
      const queryTokens = corpusTokens[idx]!;
      if (queryTokens.length === 0) continue;

      const query = queryTokens.slice(0, 5);
      const scores = this._bm25.getScores(query);

      for (const score of scores) {
        if (score > 0) {
          allScores.push(score);
        }
      }
    }

    if (allScores.length === 0) {
      return [this._userAlpha ?? 1.0, this._userBeta ?? 0.0];
    }

    const estimatedBeta = median(allScores);
    const scoreStd = stddev(allScores);
    const estimatedAlpha = scoreStd > 0 ? 1.0 / scoreStd : 1.0;

    const alpha = this._userAlpha ?? estimatedAlpha;
    const beta = this._userBeta ?? estimatedBeta;
    return [alpha, beta];
  }

  // Retrieve top-k documents with Bayesian probabilities.
  retrieve(queryTokens: string[][], k: number = 10): RetrieveResult {
    if (this._transform === null) {
      throw new Error("Call index() before retrieve().");
    }

    const result = this._bm25.retrieve(queryTokens, k);

    const probabilities = this._scoresToProbabilities(
      result.documents,
      result.scores,
      queryTokens,
    );

    return { docIds: result.documents, probabilities };
  }

  // Get probabilities for ALL documents (dense array).
  getProbabilities(queryTokens: string[]): number[] {
    if (this._transform === null) {
      throw new Error("Call index() before getProbabilities().");
    }

    const bm25Scores = this._bm25.getScores(queryTokens);
    const docIds = Array.from({ length: bm25Scores.length }, (_, i) => i);

    const probabilities = this._scoresToProbabilities(
      [docIds],
      [bm25Scores],
      [queryTokens],
    );

    return probabilities[0]!;
  }

  private _computeTF(docTokens: string[], queryTokens: string[]): number {
    const querySet = new Set(queryTokens);
    let count = 0;
    for (const t of docTokens) {
      if (querySet.has(t)) {
        count++;
      }
    }
    return count;
  }

  private _scoresToProbabilities(
    docIds: number[][],
    bm25Scores: number[][],
    queryTokensBatch: string[][],
  ): number[][] {
    const probabilities: number[][] = [];

    for (let qIdx = 0; qIdx < docIds.length; qIdx++) {
      const queryDocIds = docIds[qIdx]!;
      const queryScores = bm25Scores[qIdx]!;
      const query = queryTokensBatch[qIdx]!;
      const queryProbs: number[] = [];

      for (let dIdx = 0; dIdx < queryDocIds.length; dIdx++) {
        const did = queryDocIds[dIdx]!;
        const score = queryScores[dIdx]!;

        if (score <= 0) {
          queryProbs.push(0.0);
          continue;
        }

        const docLen = this._docLengths![did]!;
        const docLenRatio = docLen / this._avgdl!;
        const tf = this._computeTF(this._corpusTokens![did]!, query);

        queryProbs.push(
          this._transform!.scoreToProbability(
            score,
            tf,
            docLenRatio,
          ) as number,
        );
      }

      probabilities.push(queryProbs);
    }

    return probabilities;
  }
}
