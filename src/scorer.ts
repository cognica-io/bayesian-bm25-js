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
import { FusionDebugger, type BM25SignalTrace } from "./debug.js";
import { BayesianProbabilityTransform } from "./probability.js";

type BaseRateMethod = "percentile" | "mixture" | "elbow";
const VALID_BASE_RATE_METHODS: ReadonlySet<string> = new Set([
  "percentile",
  "mixture",
  "elbow",
]);

export interface BayesianBM25ScorerOptions {
  k1?: number;
  b?: number;
  method?: BM25Method;
  alpha?: number;
  beta?: number;
  baseRate?: number | "auto" | null;
  baseRateMethod?: BaseRateMethod;
}

export interface RetrieveResult {
  docIds: number[][];
  probabilities: number[][];
}

// Result object returned by retrieve(explain=true).
export interface RetrievalResult {
  docIds: number[][];
  probabilities: number[][];
  explanations: (BM25SignalTrace | null)[][] | null;
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
//
// Parameters:
//   k1:       BM25 k1 parameter (term frequency saturation).
//   b:        BM25 b parameter (document length normalisation).
//   method:   BM25 variant: "robertson", "lucene", or "atire".
//   alpha:    Sigmoid steepness.  If undefined, auto-estimated from corpus.
//   beta:     Sigmoid midpoint.  If undefined, auto-estimated from corpus.
//   baseRate: Corpus-level base rate of relevance.
//             - null/undefined (default): no base rate correction.
//             - "auto": auto-estimate from corpus score distribution.
//             - number in (0, 1): explicit base rate.
export class BayesianBM25Scorer {
  private _bm25: BM25;
  private _userAlpha: number | undefined;
  private _userBeta: number | undefined;
  private _userBaseRate: number | "auto" | null;
  private _baseRateMethod: BaseRateMethod;
  private _transform: BayesianProbabilityTransform | null = null;
  private _docLengths: number[] | null = null;
  private _avgdl: number | null = null;
  private _corpusTokens: string[][] | null = null;
  private _docTokenSets: Set<string>[] | null = null;

  constructor(options: BayesianBM25ScorerOptions = {}) {
    const baseRateMethod = options.baseRateMethod ?? "percentile";
    if (!VALID_BASE_RATE_METHODS.has(baseRateMethod)) {
      throw new Error(
        `baseRateMethod must be one of "percentile", "mixture", "elbow", ` +
          `got "${baseRateMethod}"`,
      );
    }
    this._bm25 = new BM25({
      k1: options.k1,
      b: options.b,
      method: options.method,
    });
    this._userAlpha = options.alpha;
    this._userBeta = options.beta;
    this._userBaseRate = options.baseRate ?? null;
    this._baseRateMethod = baseRateMethod;
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

  get baseRate(): number | null {
    if (this._transform === null) {
      return null;
    }
    return this._transform.baseRate;
  }

  // Build the BM25 index and compute document statistics.
  index(corpusTokens: string[][]): void {
    this._corpusTokens = corpusTokens;
    this._docTokenSets = corpusTokens.map((tokens) => new Set(tokens));
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

    // Sample pseudo-query scores once, reuse for both estimation steps
    const perQueryScores = this._samplePseudoQueryScores(corpusTokens);

    const [alpha, beta] = this._estimateParameters(perQueryScores);

    // Resolve baseRate
    let baseRate: number | null = null;
    if (this._userBaseRate === "auto") {
      baseRate = this._estimateBaseRate(perQueryScores, corpusTokens.length);
    } else if (typeof this._userBaseRate === "number") {
      baseRate = this._userBaseRate;
    }

    this._transform = new BayesianProbabilityTransform(
      alpha,
      beta,
      baseRate,
    );
  }

  private _samplePseudoQueryScores(
    corpusTokens: string[][],
  ): number[][] {
    const n = corpusTokens.length;
    const sampleSize = Math.min(n, 50);
    const rng = mulberry32(42);
    const sampleIndices = sampleWithoutReplacement(n, sampleSize, rng);

    const perQueryScores: number[][] = [];
    for (const idx of sampleIndices) {
      const queryTokens = corpusTokens[idx]!;
      if (queryTokens.length === 0) continue;

      const query = queryTokens.slice(0, 5);
      const scores = this._bm25.getScores(query);

      const nonzero: number[] = [];
      for (const score of scores) {
        if (score > 0) {
          nonzero.push(score);
        }
      }
      if (nonzero.length > 0) {
        perQueryScores.push(nonzero);
      }
    }
    return perQueryScores;
  }

  private _estimateParameters(
    perQueryScores: number[][],
  ): [number, number] {
    if (this._userAlpha !== undefined && this._userBeta !== undefined) {
      return [this._userAlpha, this._userBeta];
    }

    if (perQueryScores.length === 0) {
      return [this._userAlpha ?? 1.0, this._userBeta ?? 0.0];
    }

    const allScores: number[] = [];
    for (const queryScores of perQueryScores) {
      for (const s of queryScores) {
        allScores.push(s);
      }
    }

    const estimatedBeta = median(allScores);
    const scoreStd = stddev(allScores);
    const estimatedAlpha = scoreStd > 0 ? 1.0 / scoreStd : 1.0;

    const alpha = this._userAlpha ?? estimatedAlpha;
    const beta = this._userBeta ?? estimatedBeta;
    return [alpha, beta];
  }

  private _estimateBaseRate(
    perQueryScores: number[][],
    nDocs: number,
  ): number {
    if (perQueryScores.length === 0) {
      return 1e-6;
    }
    const method = this._baseRateMethod;
    if (method === "percentile") {
      return BayesianBM25Scorer._baseRatePercentile(perQueryScores, nDocs);
    }
    if (method === "mixture") {
      return BayesianBM25Scorer._baseRateMixture(perQueryScores);
    }
    if (method === "elbow") {
      return BayesianBM25Scorer._baseRateElbow(perQueryScores);
    }
    throw new Error(`Unknown baseRateMethod: "${method}"`);
  }

  // 95th-percentile heuristic: fraction of docs above the 95th pct.
  static _baseRatePercentile(
    perQueryScores: number[][],
    nDocs: number,
  ): number {
    const highCountRatios: number[] = [];
    for (const scores of perQueryScores) {
      const sorted = [...scores].sort((a, b) => a - b);
      const pIdx = Math.ceil(sorted.length * 0.95) - 1;
      const threshold = sorted[Math.max(0, pIdx)]!;
      let nAbove = 0;
      for (const s of scores) {
        if (s >= threshold) {
          nAbove++;
        }
      }
      highCountRatios.push(nAbove / nDocs);
    }

    let sum = 0;
    for (const r of highCountRatios) {
      sum += r;
    }
    const baseRate = sum / highCountRatios.length;
    return Math.max(1e-6, Math.min(0.5, baseRate));
  }

  // 2-component Gaussian EM to separate relevant/non-relevant scores.
  static _baseRateMixture(perQueryScores: number[][]): number {
    const allScores: number[] = [];
    for (const scores of perQueryScores) {
      for (const s of scores) {
        allScores.push(s);
      }
    }
    if (allScores.length < 2) {
      return 1e-6;
    }

    // Initialize: split at median
    const sorted = [...allScores].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const medianVal =
      sorted.length % 2 === 0
        ? (sorted[mid - 1]! + sorted[mid]!) / 2
        : sorted[mid]!;

    const lo: number[] = [];
    const hi: number[] = [];
    for (const s of allScores) {
      if (s <= medianVal) lo.push(s);
      else hi.push(s);
    }

    const mean = (arr: number[]): number => {
      let sum = 0;
      for (const v of arr) sum += v;
      return arr.length > 0 ? sum / arr.length : 0;
    };
    const variance = (arr: number[], mu: number): number => {
      let sum = 0;
      for (const v of arr) sum += (v - mu) ** 2;
      return arr.length > 0 ? sum / arr.length : 1.0;
    };

    let mu0 = lo.length > 0 ? mean(lo) : medianVal - 1.0;
    let mu1 = hi.length > 0 ? mean(hi) : medianVal + 1.0;
    let var0 = Math.max(lo.length > 0 ? variance(lo, mu0) : 1.0, 1e-8);
    let var1 = Math.max(hi.length > 0 ? variance(hi, mu1) : 1.0, 1e-8);
    let pi1 = 0.5; // mixing weight of the "relevant" component

    const n = allScores.length;

    for (let iter = 0; iter < 20; iter++) {
      // E-step: responsibilities
      const std0 = Math.sqrt(var0);
      const std1 = Math.sqrt(var1);

      const gamma: number[] = new Array(n);
      for (let i = 0; i < n; i++) {
        const s = allScores[i]!;
        const logP0 =
          -0.5 * ((s - mu0) / std0) ** 2 - Math.log(std0);
        const logP1 =
          -0.5 * ((s - mu1) / std1) ** 2 - Math.log(std1);

        const logW0 = Math.log(Math.max(1.0 - pi1, 1e-10)) + logP0;
        const logW1 = Math.log(Math.max(pi1, 1e-10)) + logP1;

        // Log-sum-exp for numerical stability
        const maxLog = Math.max(logW0, logW1);
        const logTotal =
          maxLog + Math.log(Math.exp(logW0 - maxLog) + Math.exp(logW1 - maxLog));
        gamma[i] = Math.exp(logW1 - logTotal);
      }

      // M-step
      let nEff1 = 0;
      let nEff0 = 0;
      for (let i = 0; i < n; i++) {
        nEff1 += gamma[i]!;
        nEff0 += 1.0 - gamma[i]!;
      }

      if (nEff0 < 1e-8 || nEff1 < 1e-8) {
        break;
      }

      let sumGamma0Score = 0;
      let sumGamma1Score = 0;
      for (let i = 0; i < n; i++) {
        sumGamma0Score += (1.0 - gamma[i]!) * allScores[i]!;
        sumGamma1Score += gamma[i]! * allScores[i]!;
      }
      mu0 = sumGamma0Score / nEff0;
      mu1 = sumGamma1Score / nEff1;

      let sumVar0 = 0;
      let sumVar1 = 0;
      for (let i = 0; i < n; i++) {
        sumVar0 += (1.0 - gamma[i]!) * (allScores[i]! - mu0) ** 2;
        sumVar1 += gamma[i]! * (allScores[i]! - mu1) ** 2;
      }
      var0 = Math.max(sumVar0 / nEff0, 1e-8);
      var1 = Math.max(sumVar1 / nEff1, 1e-8);
      pi1 = nEff1 / n;
    }

    // The higher-mean component is the "relevant" population
    const baseRate = mu1 >= mu0 ? pi1 : 1.0 - pi1;
    return Math.max(1e-6, Math.min(0.5, baseRate));
  }

  // Knee point in sorted score curve.
  static _baseRateElbow(perQueryScores: number[][]): number {
    const allScores: number[] = [];
    for (const scores of perQueryScores) {
      for (const s of scores) {
        allScores.push(s);
      }
    }
    allScores.sort((a, b) => b - a); // descending
    const n = allScores.length;
    if (n < 3) {
      return 1e-6;
    }

    // Line from (0, scores[0]) to (n-1, scores[n-1])
    const dx = n - 1;
    const dy = allScores[n - 1]! - allScores[0]!;
    const lineLen = Math.sqrt(dx * dx + dy * dy);

    if (lineLen < 1e-12) {
      return 1e-6;
    }

    // Perpendicular distance from each point to the line
    let maxDist = -1;
    let kneeIdx = 0;
    for (let i = 0; i < n; i++) {
      const dist =
        Math.abs(dy * i - dx * (allScores[i]! - allScores[0]!)) / lineLen;
      if (dist > maxDist) {
        maxDist = dist;
        kneeIdx = i;
      }
    }

    // Fraction of scores at or above the knee
    const baseRate = Math.max(1, kneeIdx) / n;
    return Math.max(1e-6, Math.min(0.5, baseRate));
  }

  // Add documents to the index.
  //
  // Since the underlying BM25 engine requires IDF recomputation when the
  // corpus changes, this method appends the new documents to the existing
  // corpus and rebuilds the full index.
  addDocuments(newCorpusTokens: string[][]): void {
    if (this._corpusTokens === null) {
      throw new Error("Call index() before addDocuments().");
    }
    const combined = [...this._corpusTokens, ...newCorpusTokens];
    this.index(combined);
  }

  // Retrieve top-k documents with Bayesian probabilities.
  //
  // When explain is true, returns a RetrievalResult with per-document
  // BM25SignalTrace explanations. When false (default), returns the
  // backward-compatible RetrieveResult.
  retrieve(
    queryTokens: string[][],
    k?: number,
    explain?: false,
  ): RetrieveResult;
  retrieve(
    queryTokens: string[][],
    k: number,
    explain: true,
  ): RetrievalResult;
  retrieve(
    queryTokens: string[][],
    k: number = 10,
    explain: boolean = false,
  ): RetrieveResult | RetrievalResult {
    if (this._transform === null) {
      throw new Error("Call index() before retrieve().");
    }

    const result = this._bm25.retrieve(queryTokens, k);

    const probabilities = this._scoresToProbabilities(
      result.documents,
      result.scores,
      queryTokens,
    );

    if (!explain) {
      return { docIds: result.documents, probabilities };
    }

    const debugger_ = new FusionDebugger(this._transform);
    const explanations: (BM25SignalTrace | null)[][] = [];

    for (let qIdx = 0; qIdx < result.documents.length; qIdx++) {
      const query = queryTokens[qIdx]!;
      const queryExplanations: (BM25SignalTrace | null)[] = [];
      for (let rank = 0; rank < result.documents[qIdx]!.length; rank++) {
        const did = result.documents[qIdx]![rank]!;
        const score = result.scores[qIdx]![rank]!;
        if (score > 0 && this._docLengths !== null) {
          const dlRatio = this._docLengths[did]! / this._avgdl!;
          const querySet = new Set(query);
          const docSet = this._docTokenSets![did]!;
          let tf = 0;
          for (const t of querySet) {
            if (docSet.has(t)) tf++;
          }
          queryExplanations.push(
            debugger_.traceBM25(score, tf, dlRatio),
          );
        } else {
          queryExplanations.push(null);
        }
      }
      explanations.push(queryExplanations);
    }

    return {
      docIds: result.documents,
      probabilities,
      explanations,
    };
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

  private _computeTFBatch(
    docIds: number[],
    queryTokens: string[],
  ): number[] {
    const querySet = new Set(queryTokens);
    const docTokenSets = this._docTokenSets!;
    return docIds.map((did) => {
      let count = 0;
      for (const t of querySet) {
        if (docTokenSets[did]!.has(t)) {
          count++;
        }
      }
      return count;
    });
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
      const queryProbs = new Array<number>(queryDocIds.length).fill(0.0);

      // Collect active (nonzero-score) documents
      const activeIndices: number[] = [];
      const activeIds: number[] = [];
      const activeScores: number[] = [];
      for (let dIdx = 0; dIdx < queryDocIds.length; dIdx++) {
        if (queryScores[dIdx]! > 0) {
          activeIndices.push(dIdx);
          activeIds.push(queryDocIds[dIdx]!);
          activeScores.push(queryScores[dIdx]!);
        }
      }

      if (activeIds.length === 0) {
        probabilities.push(queryProbs);
        continue;
      }

      const docLenRatios = activeIds.map(
        (did) => this._docLengths![did]! / this._avgdl!,
      );
      const tfs = this._computeTFBatch(activeIds, query);

      const batchProbs = this._transform!.scoreToProbability(
        activeScores,
        tfs,
        docLenRatios,
      ) as number[];

      for (let i = 0; i < activeIndices.length; i++) {
        queryProbs[activeIndices[i]!] = batchProbs[i]!;
      }

      probabilities.push(queryProbs);
    }

    return probabilities;
  }
}
