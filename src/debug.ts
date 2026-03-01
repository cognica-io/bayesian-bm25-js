//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

// Fusion debugger for tracing intermediate values through the Bayesian BM25 pipeline.
//
// Records every intermediate value -- likelihood, prior, posterior, logits,
// fusion -- so that the final fused probability can be fully explained.
//
// The module is scorer-independent: it works with raw values (scores, tf,
// doc_len_ratio, cosine similarity) and a BayesianProbabilityTransform
// instance.  No dependency on BM25.

import {
  BayesianProbabilityTransform,
  clampProbability,
  logit,
  sigmoid,
} from "./probability.js";
import {
  cosineToProbability,
  probAnd,
  probNot,
  probOr,
} from "./fusion.js";

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

// Trace of a single BM25 signal through the full probability pipeline.
export interface BM25SignalTrace {
  // Input
  rawScore: number;
  tf: number;
  docLenRatio: number;

  // Intermediate
  likelihood: number;
  tfPrior: number;
  normPrior: number;
  compositePrior: number;

  // Logit-space
  logitLikelihood: number;
  logitPrior: number;
  logitBaseRate: number | null;

  // Output
  posterior: number;

  // Transform params snapshot
  alpha: number;
  beta: number;
  baseRate: number | null;
}

// Trace of a cosine similarity through probability conversion.
export interface VectorSignalTrace {
  // Input
  cosineScore: number;

  // Output
  probability: number;

  // Logit-space
  logitProbability: number;
}

// Trace of a probabilistic NOT (complement) operation.
export interface NotTrace {
  // Input
  inputProbability: number;
  inputName: string;

  // Output
  complement: number;

  // Logit-space (sign flip)
  logitInput: number;
  logitComplement: number;
}

// Trace of the combination step for multiple probability signals.
export interface FusionTrace {
  // Input
  signalProbabilities: number[];
  signalNames: string[];

  // Method
  method: string; // "log_odds", "prob_and", "prob_or", "prob_not"

  // Intermediate (for log_odds)
  logits: number[] | null;
  meanLogit: number | null;
  alpha: number | null;
  nAlphaScale: number | null;
  scaledLogit: number | null;
  weights: number[] | null;

  // Output
  fusedProbability: number;

  // Intermediate (for prob_and): log-space product
  logProbs: number[] | null;
  logProbSum: number | null;

  // Intermediate (for prob_or / prob_not): complement log-space product
  complements: number[] | null;
  logComplements: number[] | null;
  logComplementSum: number | null;
}

// Complete trace for one document across all signals and fusion.
export interface DocumentTrace {
  docId: string | number | null;
  signals: Record<string, BM25SignalTrace | VectorSignalTrace>;
  fusion: FusionTrace;
  finalProbability: number;
}

// Comparison of two document traces explaining rank differences.
export interface ComparisonResult {
  docA: DocumentTrace;
  docB: DocumentTrace;
  signalDeltas: Record<string, number>;
  dominantSignal: string;
  crossoverStage: string | null;
}

// ---------------------------------------------------------------------------
// FusionDebugger
// ---------------------------------------------------------------------------

// Traces intermediate values through the Bayesian BM25 fusion pipeline.
export class FusionDebugger {
  private _transform: BayesianProbabilityTransform;

  constructor(transform: BayesianProbabilityTransform) {
    this._transform = transform;
  }

  // Trace a single BM25 score through the full probability pipeline.
  traceBM25(
    score: number,
    tf: number,
    docLenRatio: number,
    options: { docId?: string | number | null } = {},
  ): BM25SignalTrace {
    const t = this._transform;

    const likelihoodVal = t.likelihood(score) as number;
    const tfPriorVal = BayesianProbabilityTransform.tfPrior(tf) as number;
    const normPriorVal = BayesianProbabilityTransform.normPrior(
      docLenRatio,
    ) as number;
    const compositePriorVal = BayesianProbabilityTransform.compositePrior(
      tf,
      docLenRatio,
    ) as number;
    const posteriorVal = BayesianProbabilityTransform.posterior(
      likelihoodVal,
      compositePriorVal,
      t.baseRate,
    ) as number;

    const logitLikelihoodVal = logit(likelihoodVal) as number;
    const logitPriorVal = logit(compositePriorVal) as number;
    const logitBaseRateVal =
      t.baseRate !== null ? (logit(t.baseRate) as number) : null;

    return {
      rawScore: score,
      tf,
      docLenRatio,
      likelihood: likelihoodVal,
      tfPrior: tfPriorVal,
      normPrior: normPriorVal,
      compositePrior: compositePriorVal,
      logitLikelihood: logitLikelihoodVal,
      logitPrior: logitPriorVal,
      logitBaseRate: logitBaseRateVal,
      posterior: posteriorVal,
      alpha: t.alpha,
      beta: t.beta,
      baseRate: t.baseRate,
    };
  }

  // Trace a cosine similarity through probability conversion.
  traceVector(
    cosineScore: number,
    options: { docId?: string | number | null } = {},
  ): VectorSignalTrace {
    const probVal = cosineToProbability(cosineScore) as number;
    const logitVal = logit(probVal) as number;

    return {
      cosineScore,
      probability: probVal,
      logitProbability: logitVal,
    };
  }

  // Trace a probabilistic NOT (complement) operation.
  //
  // In log-odds space, NOT is a sign flip: logit(1-p) = -logit(p).
  traceNot(
    probability: number,
    options: { name?: string } = {},
  ): NotTrace {
    const name = options.name ?? "signal";
    const complement = probNot(probability) as number;
    const logitIn = logit(probability) as number;
    const logitOut = logit(complement) as number;

    return {
      inputProbability: probability,
      inputName: name,
      complement,
      logitInput: logitIn,
      logitComplement: logitOut,
    };
  }

  // Format a NOT trace as human-readable text.
  formatNot(trace: NotTrace): string {
    const lines = [
      `  [NOT ${trace.inputName}]`,
      `    P(${trace.inputName}) = ${trace.inputProbability.toFixed(3)}`,
      `    P(NOT ${trace.inputName}) = 1 - ${trace.inputProbability.toFixed(3)}` +
        ` = ${trace.complement.toFixed(3)}`,
      `    logit(${trace.inputProbability.toFixed(3)}) = ${trace.logitInput >= 0 ? "+" : ""}${trace.logitInput.toFixed(3)}`,
      `    logit(${trace.complement.toFixed(3)}) = ${trace.logitComplement >= 0 ? "+" : ""}${trace.logitComplement.toFixed(3)}` +
        `  (sign flipped)`,
    ];
    return lines.join("\n");
  }

  // Trace the fusion of multiple probability signals.
  traceFusion(
    probabilities: number[],
    options: {
      names?: string[];
      method?: string;
      alpha?: number;
      weights?: number[];
    } = {},
  ): FusionTrace {
    const {
      names,
      method = "log_odds",
      alpha,
      weights,
    } = options;

    const probs = [...probabilities];
    const n = probs.length;
    const signalNames =
      names ?? Array.from({ length: n }, (_, i) => `signal_${i}`);

    if (method === "log_odds") {
      return this._traceLogOdds(probs, signalNames, alpha, weights);
    } else if (method === "prob_and") {
      return this._traceProbAnd(probs, signalNames);
    } else if (method === "prob_or") {
      return this._traceProbOr(probs, signalNames);
    } else if (method === "prob_not") {
      return this._traceProbNot(probs, signalNames);
    } else {
      throw new Error(
        `method must be 'log_odds', 'prob_and', 'prob_or', or 'prob_not', got '${method}'`,
      );
    }
  }

  private _traceLogOdds(
    probs: number[],
    names: string[],
    alpha: number | undefined,
    weights: number[] | undefined,
  ): FusionTrace {
    const n = probs.length;
    const clampedProbs = clampProbability(probs) as number[];
    const logitsArr = clampedProbs.map((p) => logit(p) as number);

    if (weights !== undefined) {
      const effectiveAlpha = alpha ?? 0.0;
      const nAlphaScale = n ** effectiveAlpha;
      // Weighted sum of logits
      let weightedLogit = 0;
      for (let i = 0; i < n; i++) {
        weightedLogit += weights[i]! * logitsArr[i]!;
      }
      const scaled = nAlphaScale * weightedLogit;
      const fused = sigmoid(scaled) as number;
      return {
        signalProbabilities: [...clampedProbs],
        signalNames: names,
        method: "log_odds",
        logits: logitsArr,
        meanLogit: weightedLogit,
        alpha: effectiveAlpha,
        nAlphaScale,
        scaledLogit: scaled,
        weights: [...weights],
        fusedProbability: fused,
        logProbs: null,
        logProbSum: null,
        complements: null,
        logComplements: null,
        logComplementSum: null,
      };
    }

    const effectiveAlpha = alpha ?? 0.5;
    let logitSum = 0;
    for (const l of logitsArr) {
      logitSum += l;
    }
    const meanLogitVal = logitSum / n;
    const nAlphaScale = n ** effectiveAlpha;
    const scaled = meanLogitVal * nAlphaScale;
    const fused = sigmoid(scaled) as number;

    return {
      signalProbabilities: [...clampedProbs],
      signalNames: names,
      method: "log_odds",
      logits: logitsArr,
      meanLogit: meanLogitVal,
      alpha: effectiveAlpha,
      nAlphaScale,
      scaledLogit: scaled,
      weights: null,
      fusedProbability: fused,
      logProbs: null,
      logProbSum: null,
      complements: null,
      logComplements: null,
      logComplementSum: null,
    };
  }

  private _traceProbAnd(
    probs: number[],
    names: string[],
  ): FusionTrace {
    const clampedProbs = clampProbability(probs) as number[];
    const logProbs = clampedProbs.map((p) => Math.log(p));
    let logSum = 0;
    for (const lp of logProbs) {
      logSum += lp;
    }
    const fused = Math.exp(logSum);

    return {
      signalProbabilities: [...clampedProbs],
      signalNames: names,
      method: "prob_and",
      logits: null,
      meanLogit: null,
      alpha: null,
      nAlphaScale: null,
      scaledLogit: null,
      weights: null,
      fusedProbability: fused,
      logProbs,
      logProbSum: logSum,
      complements: null,
      logComplements: null,
      logComplementSum: null,
    };
  }

  private _traceProbOr(
    probs: number[],
    names: string[],
  ): FusionTrace {
    const clampedProbs = clampProbability(probs) as number[];
    const comps = clampedProbs.map((p) => 1.0 - p);
    const logComps = comps.map((c) => Math.log(c));
    let logSum = 0;
    for (const lc of logComps) {
      logSum += lc;
    }
    const fused = 1.0 - Math.exp(logSum);

    return {
      signalProbabilities: [...clampedProbs],
      signalNames: names,
      method: "prob_or",
      logits: null,
      meanLogit: null,
      alpha: null,
      nAlphaScale: null,
      scaledLogit: null,
      weights: null,
      fusedProbability: fused,
      logProbs: null,
      logProbSum: null,
      complements: comps,
      logComplements: logComps,
      logComplementSum: logSum,
    };
  }

  private _traceProbNot(
    probs: number[],
    names: string[],
  ): FusionTrace {
    const clampedProbs = clampProbability(probs) as number[];
    const comps = clampedProbs.map((p) => 1.0 - p);
    const logComps = comps.map((c) => Math.log(c));
    let logSum = 0;
    for (const lc of logComps) {
      logSum += lc;
    }
    const fused = Math.exp(logSum);

    return {
      signalProbabilities: [...clampedProbs],
      signalNames: names,
      method: "prob_not",
      logits: null,
      meanLogit: null,
      alpha: null,
      nAlphaScale: null,
      scaledLogit: null,
      weights: null,
      fusedProbability: fused,
      logProbs: null,
      logProbSum: null,
      complements: comps,
      logComplements: logComps,
      logComplementSum: logSum,
    };
  }

  // Full pipeline trace for one document (convenience method).
  //
  // Traces whichever signals are provided (BM25, vector, or both),
  // then fuses them.
  traceDocument(options: {
    bm25Score?: number;
    tf?: number;
    docLenRatio?: number;
    cosineScore?: number;
    method?: string;
    alpha?: number;
    weights?: number[];
    docId?: string | number | null;
  }): DocumentTrace {
    const {
      bm25Score,
      tf,
      docLenRatio,
      cosineScore,
      method = "log_odds",
      alpha,
      weights,
      docId = null,
    } = options;

    const signals: Record<string, BM25SignalTrace | VectorSignalTrace> = {};
    const probs: number[] = [];
    const names: string[] = [];

    const hasBM25 = bm25Score !== undefined;
    const hasVector = cosineScore !== undefined;

    if (hasBM25) {
      if (tf === undefined || docLenRatio === undefined) {
        throw new Error(
          "tf and doc_len_ratio are required when bm25_score is provided",
        );
      }
      const bm25Trace = this.traceBM25(bm25Score, tf, docLenRatio, {
        docId,
      });
      signals["BM25"] = bm25Trace;
      probs.push(bm25Trace.posterior);
      names.push("BM25");
    }

    if (hasVector) {
      const vecTrace = this.traceVector(cosineScore, { docId });
      signals["Vector"] = vecTrace;
      probs.push(vecTrace.probability);
      names.push("Vector");
    }

    if (probs.length === 0) {
      throw new Error(
        "At least one of bm25_score or cosine_score must be provided",
      );
    }

    const fusionTrace = this.traceFusion(probs, {
      names,
      method,
      alpha,
      weights,
    });

    return {
      docId,
      signals,
      fusion: fusionTrace,
      finalProbability: fusionTrace.fusedProbability,
    };
  }

  // Compare two document traces to explain rank differences.
  //
  // Computes per-signal deltas (a - b) at the final probability
  // stage of each signal, identifies the dominant signal (largest
  // absolute delta), and detects crossover -- when a signal favors
  // the opposite document from the fused result.
  compare(
    traceA: DocumentTrace,
    traceB: DocumentTrace,
  ): ComparisonResult {
    // Collect all signal names (preserving order, no duplicates)
    const allNames: string[] = [];
    const seen = new Set<string>();
    for (const name of Object.keys(traceA.signals)) {
      if (!seen.has(name)) {
        allNames.push(name);
        seen.add(name);
      }
    }
    for (const name of Object.keys(traceB.signals)) {
      if (!seen.has(name)) {
        allNames.push(name);
        seen.add(name);
      }
    }

    const signalDeltas: Record<string, number> = {};
    for (const name of allNames) {
      const probA = FusionDebugger._signalProbability(traceA, name);
      const probB = FusionDebugger._signalProbability(traceB, name);
      signalDeltas[name] = probA - probB;
    }

    // Dominant signal: largest absolute delta
    let dominant = allNames[0]!;
    let maxAbsDelta = 0;
    for (const name of allNames) {
      const absDelta = Math.abs(signalDeltas[name]!);
      if (absDelta > maxAbsDelta) {
        maxAbsDelta = absDelta;
        dominant = name;
      }
    }

    // Crossover detection: does any signal favor the opposite document?
    const fusedDelta =
      traceA.finalProbability - traceB.finalProbability;
    let crossoverStage: string | null = null;
    for (const name of allNames) {
      if (name === dominant) continue;
      const delta = signalDeltas[name]!;
      // A signal "crosses over" when it favors the opposite document
      if (fusedDelta !== 0.0 && delta !== 0.0) {
        if (
          (fusedDelta > 0 && delta < 0) ||
          (fusedDelta < 0 && delta > 0)
        ) {
          crossoverStage = name;
          break;
        }
      }
    }

    return {
      docA: traceA,
      docB: traceB,
      signalDeltas,
      dominantSignal: dominant,
      crossoverStage,
    };
  }

  private static _signalProbability(
    trace: DocumentTrace,
    name: string,
  ): number {
    const sig = trace.signals[name];
    if (sig === undefined) {
      return 0.5; // neutral if signal missing
    }
    if ("posterior" in sig) {
      return (sig as BM25SignalTrace).posterior;
    }
    if ("probability" in sig) {
      return (sig as VectorSignalTrace).probability;
    }
    return 0.5;
  }

  // ------------------------------------------------------------------
  // Formatting
  // ------------------------------------------------------------------

  // Format a document trace as human-readable text.
  formatTrace(trace: DocumentTrace, options: { verbose?: boolean } = {}): string {
    const verbose = options.verbose ?? true;
    const lines: string[] = [];
    const docLabel = trace.docId !== null ? String(trace.docId) : "unknown";
    lines.push(`Document: ${docLabel}`);

    for (const [name, sig] of Object.entries(trace.signals)) {
      if ("posterior" in sig) {
        const s = sig as BM25SignalTrace;
        lines.push(
          `  [${name}] raw=${s.rawScore.toFixed(2)}` +
            ` -> likelihood=${s.likelihood.toFixed(3)}` +
            ` (alpha=${s.alpha.toFixed(2)}, beta=${s.beta.toFixed(2)})`,
        );
        lines.push(
          `         tf=${s.tf.toFixed(0)} -> tf_prior=${s.tfPrior.toFixed(3)}`,
        );
        lines.push(
          `         dl_ratio=${s.docLenRatio.toFixed(2)}` +
            ` -> norm_prior=${s.normPrior.toFixed(3)}`,
        );
        lines.push(
          `         composite_prior=${s.compositePrior.toFixed(3)}`,
        );
        if (s.baseRate !== null) {
          // Show posterior without base_rate first, then with
          const posteriorNoBR = BayesianProbabilityTransform.posterior(
            s.likelihood,
            s.compositePrior,
            null,
          ) as number;
          lines.push(`         posterior=${posteriorNoBR.toFixed(3)}`);
          lines.push(
            `         with base_rate=${s.baseRate.toFixed(3)}:` +
              ` posterior=${s.posterior.toFixed(3)}`,
          );
        } else {
          lines.push(`         posterior=${s.posterior.toFixed(3)}`);
        }
        if (verbose) {
          const logitPost = logit(s.posterior) as number;
          lines.push(
            `         logit(posterior)=${logitPost.toFixed(3)}`,
          );
        }
        lines.push(""); // blank line
      } else if ("probability" in sig) {
        const v = sig as VectorSignalTrace;
        lines.push(
          `  [${name}] cosine=${v.cosineScore.toFixed(3)}` +
            ` -> prob=${v.probability.toFixed(3)}`,
        );
        if (verbose) {
          lines.push(
            `           logit(prob)=${v.logitProbability.toFixed(3)}`,
          );
        }
        lines.push("");
      }
    }

    // Fusion
    const f = trace.fusion;
    const alphaStr = f.alpha !== null ? `, alpha=${f.alpha}` : "";
    const nStr = `, n=${f.signalProbabilities.length}`;
    lines.push(`  [Fusion] method=${f.method}${alphaStr}${nStr}`);
    if (verbose) {
      // log_odds intermediates
      if (f.logits !== null) {
        const logitsStr =
          "[" + f.logits.map((v) => v.toFixed(3)).join(", ") + "]";
        lines.push(`           logits=${logitsStr}`);
      }
      if (f.meanLogit !== null) {
        lines.push(`           mean_logit=${f.meanLogit.toFixed(3)}`);
      }
      if (f.nAlphaScale !== null) {
        lines.push(
          `           n^alpha=${f.nAlphaScale.toFixed(3)},` +
            ` scaled=${f.scaledLogit!.toFixed(3)}`,
        );
      }
      if (f.weights !== null) {
        const weightsStr =
          "[" + f.weights.map((w) => w.toFixed(3)).join(", ") + "]";
        lines.push(`           weights=${weightsStr}`);
      }
      // prob_and intermediates
      if (f.logProbs !== null) {
        const lpStr =
          "[" + f.logProbs.map((v) => v.toFixed(3)).join(", ") + "]";
        lines.push(`           ln(P)=${lpStr}`);
        lines.push(`           sum(ln(P))=${f.logProbSum!.toFixed(3)}`);
      }
      // prob_or / prob_not intermediates
      if (f.complements !== null) {
        const cpStr =
          "[" +
          f.complements.map((v) => v.toFixed(3)).join(", ") +
          "]";
        lines.push(`           1-P=${cpStr}`);
      }
      if (f.logComplements !== null) {
        const lcStr =
          "[" +
          f.logComplements.map((v) => v.toFixed(3)).join(", ") +
          "]";
        lines.push(`           ln(1-P)=${lcStr}`);
        lines.push(
          `           sum(ln(1-P))=${f.logComplementSum!.toFixed(3)}`,
        );
      }
    }
    lines.push(`           -> final=${f.fusedProbability.toFixed(3)}`);

    return lines.join("\n");
  }

  // Compact one-line summary of a document trace.
  formatSummary(trace: DocumentTrace): string {
    const docLabel = trace.docId !== null ? String(trace.docId) : "unknown";
    const parts: string[] = [];
    for (const [, sig] of Object.entries(trace.signals)) {
      if ("posterior" in sig) {
        parts.push(`BM25=${(sig as BM25SignalTrace).posterior.toFixed(3)}`);
      } else if ("probability" in sig) {
        parts.push(
          `Vec=${(sig as VectorSignalTrace).probability.toFixed(3)}`,
        );
      }
    }

    const f = trace.fusion;
    const alphaStr = f.alpha !== null ? `, alpha=${f.alpha}` : "";
    const signalStr = parts.join(" ");
    return (
      `${docLabel}: ${signalStr}` +
      ` -> Fused=${f.fusedProbability.toFixed(3)}` +
      ` (${f.method}${alphaStr})`
    );
  }

  // Format a comparison result as human-readable text.
  formatComparison(comparison: ComparisonResult): string {
    const a = comparison.docA;
    const b = comparison.docB;
    const aLabel = a.docId !== null ? String(a.docId) : "doc_a";
    const bLabel = b.docId !== null ? String(b.docId) : "doc_b";

    const lines: string[] = [];
    lines.push(`Comparison: ${aLabel} vs ${bLabel}`);

    // Header
    lines.push(
      `  ${"Signal".padEnd(12)} ${aLabel.toString().padStart(8)}  ${bLabel.toString().padStart(8)}` +
        `  ${"delta".padStart(8)}   dominant`,
    );

    const allNames = Object.keys(comparison.signalDeltas);
    for (const name of allNames) {
      const probA = FusionDebugger._signalProbability(a, name);
      const probB = FusionDebugger._signalProbability(b, name);
      const delta = comparison.signalDeltas[name]!;
      const dominantMarker =
        name === comparison.dominantSignal ? "   <-- largest" : "";
      const deltaStr = (delta >= 0 ? "+" : "") + delta.toFixed(3);
      lines.push(
        `  ${name.padEnd(12)} ${probA.toFixed(3).padStart(8)}  ${probB.toFixed(3).padStart(8)}` +
          `  ${deltaStr.padStart(8)}${dominantMarker}`,
      );
    }

    // Fused row
    const fusedDelta = a.finalProbability - b.finalProbability;
    const fusedDeltaStr =
      (fusedDelta >= 0 ? "+" : "") + fusedDelta.toFixed(3);
    lines.push(
      `  ${"Fused".padEnd(12)} ${a.finalProbability.toFixed(3).padStart(8)}` +
        `  ${b.finalProbability.toFixed(3).padStart(8)}` +
        `  ${fusedDeltaStr.padStart(8)}`,
    );
    lines.push("");

    // Rank order
    if (fusedDelta > 0) {
      lines.push(
        `  Rank order: ${aLabel} > ${bLabel} (by +${fusedDelta.toFixed(3)})`,
      );
    } else if (fusedDelta < 0) {
      lines.push(
        `  Rank order: ${bLabel} > ${aLabel} (by +${Math.abs(fusedDelta).toFixed(3)})`,
      );
    } else {
      lines.push(`  Rank order: tied`);
    }

    // Dominant signal
    const dom = comparison.dominantSignal;
    const domDelta = comparison.signalDeltas[dom]!;
    const favored = domDelta >= 0 ? aLabel : bLabel;
    const domDeltaStr = (domDelta >= 0 ? "+" : "") + domDelta.toFixed(3);
    lines.push(
      `  Dominant signal: ${dom}` +
        ` (${domDeltaStr} in ${favored}'s favor)`,
    );

    // Crossover note
    if (comparison.crossoverStage !== null) {
      const cross = comparison.crossoverStage;
      const crossDelta = comparison.signalDeltas[cross]!;
      const crossFavored = crossDelta >= 0 ? aLabel : bLabel;
      lines.push(
        `  Note: ${cross} favored ${crossFavored},` +
          ` but ${dom} signal outweighed it`,
      );
    }

    return lines.join("\n");
  }
}
