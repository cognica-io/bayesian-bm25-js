//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

import { describe, expect, it } from "vitest";

import {
  BayesianProbabilityTransform,
  logit,
} from "../src/probability.js";
import {
  cosineToProbability,
  logOddsConjunction,
  probAnd,
  probNot,
  probOr,
} from "../src/fusion.js";
import {
  FusionDebugger,
  type BM25SignalTrace,
  type VectorSignalTrace,
  type FusionTrace,
  type DocumentTrace,
  type ComparisonResult,
  type NotTrace,
} from "../src/debug.js";

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

function createTransform(): BayesianProbabilityTransform {
  return new BayesianProbabilityTransform(0.45, 6.1);
}

function createTransformWithBaseRate(): BayesianProbabilityTransform {
  return new BayesianProbabilityTransform(0.45, 6.1, 0.02);
}

function createDebugger(): FusionDebugger {
  return new FusionDebugger(createTransform());
}

function createDebuggerBR(): FusionDebugger {
  return new FusionDebugger(createTransformWithBaseRate());
}

// ---------------------------------------------------------------------------
// traceBM25
// ---------------------------------------------------------------------------

describe("traceBM25", () => {
  it("intermediate values match direct function calls", () => {
    const transform = createTransform();
    const debugger_ = new FusionDebugger(transform);
    const score = 8.42;
    const tf = 5.0;
    const dl = 0.6;
    const trace = debugger_.traceBM25(score, tf, dl);

    expect(trace.rawScore).toBe(score);
    expect(trace.tf).toBe(tf);
    expect(trace.docLenRatio).toBe(dl);

    expect(trace.likelihood).toBeCloseTo(
      transform.likelihood(score) as number,
    );
    expect(trace.tfPrior).toBeCloseTo(
      BayesianProbabilityTransform.tfPrior(tf) as number,
    );
    expect(trace.normPrior).toBeCloseTo(
      BayesianProbabilityTransform.normPrior(dl) as number,
    );
    expect(trace.compositePrior).toBeCloseTo(
      BayesianProbabilityTransform.compositePrior(tf, dl) as number,
    );
    expect(trace.posterior).toBeCloseTo(
      BayesianProbabilityTransform.posterior(
        trace.likelihood,
        trace.compositePrior,
      ) as number,
    );
  });

  it("captures logit values", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceBM25(8.42, 5.0, 0.6);
    expect(trace.logitLikelihood).toBeCloseTo(
      logit(trace.likelihood) as number,
    );
    expect(trace.logitPrior).toBeCloseTo(
      logit(trace.compositePrior) as number,
    );
    expect(trace.logitBaseRate).toBeNull();
  });

  it("captures param snapshot", () => {
    const transform = createTransform();
    const debugger_ = new FusionDebugger(transform);
    const trace = debugger_.traceBM25(5.0, 3.0, 0.8);
    expect(trace.alpha).toBe(transform.alpha);
    expect(trace.beta).toBe(transform.beta);
    expect(trace.baseRate).toBeNull();
  });

  it("works with base rate", () => {
    const transform = createTransformWithBaseRate();
    const debugger_ = new FusionDebugger(transform);
    const trace = debugger_.traceBM25(8.42, 5.0, 0.6);
    const expectedPosterior = BayesianProbabilityTransform.posterior(
      trace.likelihood,
      trace.compositePrior,
      transform.baseRate,
    ) as number;
    expect(trace.posterior).toBeCloseTo(expectedPosterior);
    expect(trace.baseRate).toBe(0.02);
    expect(trace.logitBaseRate).toBeCloseTo(logit(0.02) as number);
  });

  it("returns BM25SignalTrace", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceBM25(5.0, 3.0, 0.8);
    expect(trace).toHaveProperty("rawScore");
    expect(trace).toHaveProperty("likelihood");
    expect(trace).toHaveProperty("posterior");
  });

  it("extreme score produces likelihood near 1", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceBM25(100.0, 10.0, 0.5);
    expect(trace.likelihood).toBeCloseTo(1.0, 5);
  });

  it("zero score produces valid results", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceBM25(0.0, 1.0, 1.0);
    expect(trace.likelihood).toBeGreaterThan(0.0);
    expect(trace.likelihood).toBeLessThan(1.0);
    expect(trace.posterior).toBeGreaterThan(0.0);
    expect(trace.posterior).toBeLessThan(1.0);
  });
});

// ---------------------------------------------------------------------------
// traceVector
// ---------------------------------------------------------------------------

describe("traceVector", () => {
  it("matches cosineToProbability", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceVector(0.74);
    expect(trace.cosineScore).toBe(0.74);
    expect(trace.probability).toBeCloseTo(
      cosineToProbability(0.74) as number,
    );
    expect(trace.logitProbability).toBeCloseTo(
      logit(trace.probability) as number,
    );
  });

  it("zero cosine -> 0.5", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceVector(0.0);
    expect(trace.probability).toBeCloseTo(0.5);
  });

  it("high cosine -> > 0.9", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceVector(0.99);
    expect(trace.probability).toBeGreaterThan(0.9);
  });

  it("negative cosine -> < 0.5", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceVector(-0.5);
    expect(trace.probability).toBeLessThan(0.5);
  });

  it("returns VectorSignalTrace", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceVector(0.5);
    expect(trace).toHaveProperty("cosineScore");
    expect(trace).toHaveProperty("probability");
    expect(trace).toHaveProperty("logitProbability");
  });
});

// ---------------------------------------------------------------------------
// traceFusion
// ---------------------------------------------------------------------------

describe("traceFusion", () => {
  it("log_odds default alpha is 0.5", () => {
    const debugger_ = createDebugger();
    const probs = [0.8, 0.7];
    const trace = debugger_.traceFusion(probs);
    expect(trace.method).toBe("log_odds");
    expect(trace.alpha).toBe(0.5);
    expect(trace.fusedProbability).toBeCloseTo(
      logOddsConjunction(probs) as number,
      9,
    );
  });

  it("log_odds captures intermediates", () => {
    const debugger_ = createDebugger();
    const probs = [0.8, 0.7];
    const trace = debugger_.traceFusion(probs, { method: "log_odds" });
    expect(trace.logits).not.toBeNull();
    expect(trace.logits!).toHaveLength(2);
    expect(trace.meanLogit).not.toBeNull();
    expect(trace.nAlphaScale).not.toBeNull();
    expect(trace.scaledLogit).not.toBeNull();

    // Verify math: mean of logits
    const expectedMean =
      ((logit(0.8) as number) + (logit(0.7) as number)) / 2.0;
    expect(trace.meanLogit!).toBeCloseTo(expectedMean);

    // n^alpha = 2^0.5 = sqrt(2)
    expect(trace.nAlphaScale!).toBeCloseTo(Math.sqrt(2));

    // scaled = mean * n^alpha
    expect(trace.scaledLogit!).toBeCloseTo(expectedMean * Math.sqrt(2));
  });

  it("log_odds with explicit alpha", () => {
    const debugger_ = createDebugger();
    const probs = [0.8, 0.7];
    const trace = debugger_.traceFusion(probs, { alpha: 0.3 });
    expect(trace.alpha).toBe(0.3);
    const expected = logOddsConjunction(probs, 0.3) as number;
    expect(trace.fusedProbability).toBeCloseTo(expected, 9);
  });

  it("log_odds weighted", () => {
    const debugger_ = createDebugger();
    const probs = [0.8, 0.7];
    const weights = [0.6, 0.4];
    const trace = debugger_.traceFusion(probs, { weights });
    expect(trace.weights).not.toBeNull();
    expect(trace.weights![0]!).toBeCloseTo(0.6);
    expect(trace.weights![1]!).toBeCloseTo(0.4);
    // Default alpha for weighted is 0.0
    expect(trace.alpha).toBe(0.0);
    const expected = logOddsConjunction(
      probs,
      undefined,
      weights,
    ) as number;
    expect(trace.fusedProbability).toBeCloseTo(expected, 9);
  });

  it("log_odds weighted with alpha", () => {
    const debugger_ = createDebugger();
    const probs = [0.8, 0.7];
    const weights = [0.6, 0.4];
    const trace = debugger_.traceFusion(probs, {
      weights,
      alpha: 0.5,
    });
    expect(trace.alpha).toBe(0.5);
    const expected = logOddsConjunction(probs, 0.5, weights) as number;
    expect(trace.fusedProbability).toBeCloseTo(expected, 9);
  });

  it("prob_and method", () => {
    const debugger_ = createDebugger();
    const probs = [0.8, 0.9];
    const trace = debugger_.traceFusion(probs, { method: "prob_and" });
    expect(trace.method).toBe("prob_and");
    expect(trace.fusedProbability).toBeCloseTo(
      probAnd(probs) as number,
    );
    // No log_odds intermediates for prob_and
    expect(trace.logits).toBeNull();
    expect(trace.meanLogit).toBeNull();
  });

  it("prob_or method", () => {
    const debugger_ = createDebugger();
    const probs = [0.8, 0.9];
    const trace = debugger_.traceFusion(probs, { method: "prob_or" });
    expect(trace.method).toBe("prob_or");
    expect(trace.fusedProbability).toBeCloseTo(
      probOr(probs) as number,
    );
  });

  it("default signal names", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceFusion([0.5, 0.6, 0.7]);
    expect(trace.signalNames).toEqual([
      "signal_0",
      "signal_1",
      "signal_2",
    ]);
  });

  it("custom signal names", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceFusion([0.5, 0.6], {
      names: ["BM25", "Vec"],
    });
    expect(trace.signalNames).toEqual(["BM25", "Vec"]);
  });

  it("invalid method raises error", () => {
    const debugger_ = createDebugger();
    expect(() => {
      debugger_.traceFusion([0.5], { method: "invalid" });
    }).toThrow(/method must be/);
  });

  it("single signal with alpha=0", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceFusion([0.8], { alpha: 0.0 });
    expect(trace.fusedProbability).toBeCloseTo(0.8, 5);
  });

  it("returns FusionTrace", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceFusion([0.5, 0.6]);
    expect(trace).toHaveProperty("signalProbabilities");
    expect(trace).toHaveProperty("fusedProbability");
  });

  it("prob_and captures intermediates", () => {
    const debugger_ = createDebugger();
    const probs = [0.8, 0.7, 0.6];
    const trace = debugger_.traceFusion(probs, { method: "prob_and" });
    expect(trace.logProbs).not.toBeNull();
    expect(trace.logProbs!).toHaveLength(3);
    expect(trace.logProbSum).not.toBeNull();

    // Verify math: ln(p_i)
    for (let i = 0; i < probs.length; i++) {
      expect(trace.logProbs![i]!).toBeCloseTo(Math.log(probs[i]!));
    }

    // sum(ln(p_i))
    const logProbSum = trace.logProbs!.reduce((a, b) => a + b, 0);
    expect(trace.logProbSum!).toBeCloseTo(logProbSum);

    // exp(sum) = fused
    expect(trace.fusedProbability).toBeCloseTo(
      Math.exp(trace.logProbSum!),
      9,
    );

    // log_odds fields should be null
    expect(trace.logits).toBeNull();
    expect(trace.complements).toBeNull();
  });

  it("prob_or captures intermediates", () => {
    const debugger_ = createDebugger();
    const probs = [0.8, 0.7, 0.6];
    const trace = debugger_.traceFusion(probs, { method: "prob_or" });
    expect(trace.complements).not.toBeNull();
    expect(trace.logComplements).not.toBeNull();
    expect(trace.logComplementSum).not.toBeNull();
    expect(trace.complements!).toHaveLength(3);
    expect(trace.logComplements!).toHaveLength(3);

    // Verify math: 1-p_i
    for (let i = 0; i < probs.length; i++) {
      expect(trace.complements![i]!).toBeCloseTo(1.0 - probs[i]!, 9);
    }

    // ln(1-p_i)
    for (let i = 0; i < trace.complements!.length; i++) {
      expect(trace.logComplements![i]!).toBeCloseTo(
        Math.log(trace.complements![i]!),
      );
    }

    // sum(ln(1-p_i))
    const logCompSum = trace.logComplements!.reduce((a, b) => a + b, 0);
    expect(trace.logComplementSum!).toBeCloseTo(logCompSum);

    // 1 - exp(sum) = fused
    expect(trace.fusedProbability).toBeCloseTo(
      1.0 - Math.exp(trace.logComplementSum!),
      9,
    );

    // log_odds / prob_and fields should be null
    expect(trace.logits).toBeNull();
    expect(trace.logProbs).toBeNull();
  });

  it("prob_not method", () => {
    const debugger_ = createDebugger();
    const probs = [0.8, 0.9];
    const trace = debugger_.traceFusion(probs, { method: "prob_not" });
    expect(trace.method).toBe("prob_not");
    // prob_not = prod(1-p_i) = complement of prob_or
    const expectedOr = probOr(probs) as number;
    expect(trace.fusedProbability).toBeCloseTo(1.0 - expectedOr, 9);
  });

  it("prob_not captures intermediates", () => {
    const debugger_ = createDebugger();
    const probs = [0.8, 0.7, 0.6];
    const trace = debugger_.traceFusion(probs, { method: "prob_not" });
    expect(trace.complements).not.toBeNull();
    expect(trace.logComplements).not.toBeNull();
    expect(trace.logComplementSum).not.toBeNull();

    // Verify math: 1-p_i
    for (let i = 0; i < probs.length; i++) {
      expect(trace.complements![i]!).toBeCloseTo(1.0 - probs[i]!, 9);
    }

    // fused = exp(sum) (NOT exp, not 1-exp like OR)
    expect(trace.fusedProbability).toBeCloseTo(
      Math.exp(trace.logComplementSum!),
      9,
    );
  });

  it("prob_not is complement of prob_or", () => {
    const debugger_ = createDebugger();
    const probs = [0.8, 0.7, 0.6];
    const notTrace = debugger_.traceFusion(probs, {
      method: "prob_not",
    });
    const orTrace = debugger_.traceFusion(probs, { method: "prob_or" });
    expect(
      notTrace.fusedProbability + orTrace.fusedProbability,
    ).toBeCloseTo(1.0, 9);
  });

  it("prob_not single signal = 1-p", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceFusion([0.8], { method: "prob_not" });
    expect(trace.fusedProbability).toBeCloseTo(0.2, 5);
  });
});

// ---------------------------------------------------------------------------
// traceNot
// ---------------------------------------------------------------------------

describe("traceNot", () => {
  it("complement matches probNot", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceNot(0.7, { name: "snake" });
    expect(trace.inputProbability).toBe(0.7);
    expect(trace.inputName).toBe("snake");
    expect(trace.complement).toBeCloseTo(probNot(0.7) as number);
  });

  it("logit(NOT p) = -logit(p) (sign flip)", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceNot(0.8);
    expect(trace.logitComplement).toBeCloseTo(-trace.logitInput, 9);
  });

  it("logit values are correct", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceNot(0.7);
    expect(trace.logitInput).toBeCloseTo(logit(0.7) as number);
    expect(trace.logitComplement).toBeCloseTo(logit(0.3) as number);
  });

  it("NOT 0.5 = 0.5 (self-complementary)", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceNot(0.5);
    expect(trace.complement).toBeCloseTo(0.5);
    expect(trace.logitInput).toBeCloseTo(0.0, 9);
    expect(trace.logitComplement).toBeCloseTo(0.0, 9);
  });

  it("near zero", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceNot(0.01);
    expect(trace.complement).toBeCloseTo(0.99);
  });

  it("near one", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceNot(0.99);
    expect(trace.complement).toBeCloseTo(0.01);
  });

  it("returns NotTrace", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceNot(0.5);
    expect(trace).toHaveProperty("inputProbability");
    expect(trace).toHaveProperty("complement");
    expect(trace).toHaveProperty("logitInput");
  });

  it("default name is 'signal'", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceNot(0.5);
    expect(trace.inputName).toBe("signal");
  });

  it("NOT(NOT(p)) = p (involution)", () => {
    const debugger_ = createDebugger();
    const trace1 = debugger_.traceNot(0.7);
    const trace2 = debugger_.traceNot(trace1.complement);
    expect(trace2.complement).toBeCloseTo(0.7, 9);
  });
});

// ---------------------------------------------------------------------------
// formatNot
// ---------------------------------------------------------------------------

describe("formatNot", () => {
  it("contains input and complement", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceNot(0.7, { name: "snake" });
    const output = debugger_.formatNot(trace);
    expect(output).toContain("NOT snake");
    expect(output).toContain("0.700");
    expect(output).toContain("0.300");
  });

  it("contains sign flip note", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceNot(0.7, { name: "snake" });
    const output = debugger_.formatNot(trace);
    expect(output).toContain("sign flipped");
  });

  it("contains logit values", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceNot(0.8, { name: "topic" });
    const output = debugger_.formatNot(trace);
    expect(output).toContain("logit");
  });
});

// ---------------------------------------------------------------------------
// traceDocument
// ---------------------------------------------------------------------------

describe("traceDocument", () => {
  it("BM25 only", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      bm25Score: 8.42,
      tf: 5.0,
      docLenRatio: 0.6,
      docId: "doc-1",
    });
    expect(trace.docId).toBe("doc-1");
    expect(trace.signals["BM25"]).toBeDefined();
    expect(trace.signals["Vector"]).toBeUndefined();
    expect(trace.finalProbability).toBe(trace.fusion.fusedProbability);
  });

  it("vector only", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      cosineScore: 0.74,
      docId: "doc-2",
    });
    expect(trace.signals["Vector"]).toBeDefined();
    expect(trace.signals["BM25"]).toBeUndefined();
  });

  it("both signals", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      bm25Score: 8.42,
      tf: 5.0,
      docLenRatio: 0.6,
      cosineScore: 0.74,
      docId: "doc-3",
    });
    expect(trace.signals["BM25"]).toBeDefined();
    expect(trace.signals["Vector"]).toBeDefined();
    expect(trace.fusion.signalProbabilities).toHaveLength(2);
    expect(trace.fusion.signalNames).toEqual(["BM25", "Vector"]);
  });

  it("uses correct fusion method", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      bm25Score: 5.0,
      tf: 3.0,
      docLenRatio: 0.8,
      cosineScore: 0.5,
      method: "prob_and",
    });
    expect(trace.fusion.method).toBe("prob_and");
  });

  it("raises error with no signals", () => {
    const debugger_ = createDebugger();
    expect(() => debugger_.traceDocument({})).toThrow(/At least one/);
  });

  it("raises error when BM25 missing tf", () => {
    const debugger_ = createDebugger();
    expect(() =>
      debugger_.traceDocument({ bm25Score: 5.0 }),
    ).toThrow(/tf and doc_len_ratio/);
  });

  it("raises error when BM25 missing doc_len_ratio", () => {
    const debugger_ = createDebugger();
    expect(() =>
      debugger_.traceDocument({ bm25Score: 5.0, tf: 3.0 }),
    ).toThrow(/tf and doc_len_ratio/);
  });

  it("returns DocumentTrace", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({ cosineScore: 0.5 });
    expect(trace).toHaveProperty("docId");
    expect(trace).toHaveProperty("signals");
    expect(trace).toHaveProperty("fusion");
    expect(trace).toHaveProperty("finalProbability");
  });

  it("finalProbability matches fusion", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      bm25Score: 8.0,
      tf: 4.0,
      docLenRatio: 0.5,
      cosineScore: 0.7,
    });
    expect(trace.finalProbability).toBe(trace.fusion.fusedProbability);
  });

  it("consistency with individual traces", () => {
    const debugger_ = createDebugger();
    const score = 8.42;
    const tf = 5.0;
    const dl = 0.6;
    const cos = 0.74;

    const docTrace = debugger_.traceDocument({
      bm25Score: score,
      tf,
      docLenRatio: dl,
      cosineScore: cos,
      docId: "doc-x",
    });

    const bm25Trace = debugger_.traceBM25(score, tf, dl);
    const vecTrace = debugger_.traceVector(cos);
    const fusionTrace = debugger_.traceFusion(
      [bm25Trace.posterior, vecTrace.probability],
      { names: ["BM25", "Vector"] },
    );

    const docBM25 = docTrace.signals["BM25"] as BM25SignalTrace;
    const docVec = docTrace.signals["Vector"] as VectorSignalTrace;

    expect(docBM25.posterior).toBeCloseTo(bm25Trace.posterior);
    expect(docVec.probability).toBeCloseTo(vecTrace.probability);
    expect(docTrace.finalProbability).toBeCloseTo(
      fusionTrace.fusedProbability,
      9,
    );
  });
});

// ---------------------------------------------------------------------------
// compare
// ---------------------------------------------------------------------------

describe("compare", () => {
  it("computes signal deltas", () => {
    const debugger_ = createDebugger();
    const a = debugger_.traceDocument({
      bm25Score: 8.0,
      tf: 5.0,
      docLenRatio: 0.6,
      cosineScore: 0.74,
      docId: "doc-a",
    });
    const b = debugger_.traceDocument({
      bm25Score: 6.0,
      tf: 3.0,
      docLenRatio: 0.8,
      cosineScore: 0.5,
      docId: "doc-b",
    });
    const result = debugger_.compare(a, b);

    expect(result.signalDeltas["BM25"]).toBeDefined();
    expect(result.signalDeltas["Vector"]).toBeDefined();

    const aBM25 = a.signals["BM25"] as BM25SignalTrace;
    const bBM25 = b.signals["BM25"] as BM25SignalTrace;
    expect(result.signalDeltas["BM25"]!).toBeCloseTo(
      aBM25.posterior - bBM25.posterior,
    );

    const aVec = a.signals["Vector"] as VectorSignalTrace;
    const bVec = b.signals["Vector"] as VectorSignalTrace;
    expect(result.signalDeltas["Vector"]!).toBeCloseTo(
      aVec.probability - bVec.probability,
    );
  });

  it("dominant signal has largest absolute delta", () => {
    const debugger_ = createDebugger();
    const a = debugger_.traceDocument({
      bm25Score: 8.0,
      tf: 5.0,
      docLenRatio: 0.6,
      cosineScore: 0.74,
      docId: "doc-a",
    });
    const b = debugger_.traceDocument({
      bm25Score: 6.0,
      tf: 3.0,
      docLenRatio: 0.8,
      cosineScore: 0.5,
      docId: "doc-b",
    });
    const result = debugger_.compare(a, b);

    let maxDeltaName = "";
    let maxAbsDelta = 0;
    for (const [name, delta] of Object.entries(result.signalDeltas)) {
      if (Math.abs(delta) > maxAbsDelta) {
        maxAbsDelta = Math.abs(delta);
        maxDeltaName = name;
      }
    }
    expect(result.dominantSignal).toBe(maxDeltaName);
  });

  it("detects crossover when signals disagree", () => {
    const debugger_ = createDebugger();
    // doc-a: low BM25 but high vector
    const a = debugger_.traceDocument({
      bm25Score: 3.0,
      tf: 1.0,
      docLenRatio: 1.5,
      cosineScore: 0.95,
      docId: "doc-a",
    });
    // doc-b: high BM25 but low vector
    const b = debugger_.traceDocument({
      bm25Score: 10.0,
      tf: 8.0,
      docLenRatio: 0.5,
      cosineScore: 0.1,
      docId: "doc-b",
    });
    const result = debugger_.compare(a, b);

    // Signals should point in opposite directions
    const bm25Delta = result.signalDeltas["BM25"]!;
    const vecDelta = result.signalDeltas["Vector"]!;
    expect(bm25Delta * vecDelta).toBeLessThan(0); // opposite signs

    // crossover_stage should be set
    expect(result.crossoverStage).not.toBeNull();
  });

  it("no crossover when signals agree", () => {
    const debugger_ = createDebugger();
    const a = debugger_.traceDocument({
      bm25Score: 9.0,
      tf: 7.0,
      docLenRatio: 0.5,
      cosineScore: 0.9,
      docId: "doc-a",
    });
    const b = debugger_.traceDocument({
      bm25Score: 3.0,
      tf: 1.0,
      docLenRatio: 1.5,
      cosineScore: 0.2,
      docId: "doc-b",
    });
    const result = debugger_.compare(a, b);
    expect(result.crossoverStage).toBeNull();
  });

  it("returns ComparisonResult", () => {
    const debugger_ = createDebugger();
    const a = debugger_.traceDocument({ cosineScore: 0.8, docId: "a" });
    const b = debugger_.traceDocument({ cosineScore: 0.3, docId: "b" });
    const result = debugger_.compare(a, b);
    expect(result.docA).toBe(a);
    expect(result.docB).toBe(b);
  });

  it("works with single signal", () => {
    const debugger_ = createDebugger();
    const a = debugger_.traceDocument({ cosineScore: 0.9, docId: "a" });
    const b = debugger_.traceDocument({ cosineScore: 0.3, docId: "b" });
    const result = debugger_.compare(a, b);
    expect(result.dominantSignal).toBe("Vector");
    expect(result.signalDeltas["Vector"]!).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// formatTrace
// ---------------------------------------------------------------------------

describe("formatTrace", () => {
  it("contains document id", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      bm25Score: 8.42,
      tf: 5.0,
      docLenRatio: 0.6,
      docId: "doc-42",
    });
    const output = debugger_.formatTrace(trace);
    expect(output).toContain("Document: doc-42");
  });

  it("contains BM25 fields", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      bm25Score: 8.42,
      tf: 5.0,
      docLenRatio: 0.6,
      docId: "doc-42",
    });
    const output = debugger_.formatTrace(trace);
    expect(output).toContain("[BM25]");
    expect(output).toContain("raw=8.42");
    expect(output).toContain("likelihood=");
    expect(output).toContain("tf_prior=");
    expect(output).toContain("norm_prior=");
    expect(output).toContain("composite_prior=");
    expect(output).toContain("posterior=");
  });

  it("contains vector fields", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      cosineScore: 0.74,
      docId: "doc-42",
    });
    const output = debugger_.formatTrace(trace);
    expect(output).toContain("[Vector]");
    expect(output).toContain("cosine=0.740");
    expect(output).toContain("prob=");
  });

  it("contains fusion fields", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      bm25Score: 8.42,
      tf: 5.0,
      docLenRatio: 0.6,
      cosineScore: 0.74,
      docId: "doc-42",
    });
    const output = debugger_.formatTrace(trace);
    expect(output).toContain("[Fusion]");
    expect(output).toContain("method=log_odds");
    expect(output).toContain("-> final=");
  });

  it("verbose includes logits", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      bm25Score: 8.42,
      tf: 5.0,
      docLenRatio: 0.6,
      cosineScore: 0.74,
      docId: "doc-42",
    });
    const verboseOutput = debugger_.formatTrace(trace, { verbose: true });
    const nonVerbose = debugger_.formatTrace(trace, { verbose: false });
    expect(verboseOutput).toContain("logit(");
    expect(nonVerbose).not.toContain("logit(");
  });

  it("base rate formatting", () => {
    const debugger_ = createDebuggerBR();
    const trace = debugger_.traceDocument({
      bm25Score: 8.42,
      tf: 5.0,
      docLenRatio: 0.6,
      docId: "doc-42",
    });
    const output = debugger_.formatTrace(trace);
    expect(output).toContain("base_rate=");
  });

  it("unknown doc id", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({ cosineScore: 0.5 });
    const output = debugger_.formatTrace(trace);
    expect(output).toContain("Document: unknown");
  });

  it("prob_and shows log probs", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      bm25Score: 8.0,
      tf: 4.0,
      docLenRatio: 0.5,
      cosineScore: 0.7,
      method: "prob_and",
    });
    const output = debugger_.formatTrace(trace);
    expect(output).toContain("method=prob_and");
    expect(output).toContain("ln(P)=");
    expect(output).toContain("sum(ln(P))=");
  });

  it("prob_or shows complements", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      bm25Score: 8.0,
      tf: 4.0,
      docLenRatio: 0.5,
      cosineScore: 0.7,
      method: "prob_or",
    });
    const output = debugger_.formatTrace(trace);
    expect(output).toContain("method=prob_or");
    expect(output).toContain("1-P=");
    expect(output).toContain("ln(1-P)=");
    expect(output).toContain("sum(ln(1-P))=");
  });

  it("prob_and non-verbose hides intermediates", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      cosineScore: 0.7,
      method: "prob_and",
    });
    const output = debugger_.formatTrace(trace, { verbose: false });
    expect(output).not.toContain("ln(P)=");
    expect(output).toContain("-> final=");
  });

  it("prob_or non-verbose hides intermediates", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      cosineScore: 0.7,
      method: "prob_or",
    });
    const output = debugger_.formatTrace(trace, { verbose: false });
    expect(output).not.toContain("1-P=");
    expect(output).toContain("-> final=");
  });

  it("prob_not shows complements", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      bm25Score: 8.0,
      tf: 4.0,
      docLenRatio: 0.5,
      cosineScore: 0.7,
      method: "prob_not",
    });
    const output = debugger_.formatTrace(trace);
    expect(output).toContain("method=prob_not");
    expect(output).toContain("1-P=");
    expect(output).toContain("ln(1-P)=");
    expect(output).toContain("sum(ln(1-P))=");
  });

  it("prob_not non-verbose hides intermediates", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      cosineScore: 0.7,
      method: "prob_not",
    });
    const output = debugger_.formatTrace(trace, { verbose: false });
    expect(output).not.toContain("1-P=");
    expect(output).toContain("-> final=");
  });
});

// ---------------------------------------------------------------------------
// formatSummary
// ---------------------------------------------------------------------------

describe("formatSummary", () => {
  it("returns one line", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      bm25Score: 8.42,
      tf: 5.0,
      docLenRatio: 0.6,
      cosineScore: 0.74,
      docId: "doc-42",
    });
    const summary = debugger_.formatSummary(trace);
    expect(summary).not.toContain("\n");
  });

  it("contains doc id", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      cosineScore: 0.74,
      docId: "doc-42",
    });
    const summary = debugger_.formatSummary(trace);
    expect(summary.startsWith("doc-42:")).toBe(true);
  });

  it("contains signal values", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      bm25Score: 8.42,
      tf: 5.0,
      docLenRatio: 0.6,
      cosineScore: 0.74,
      docId: "doc-42",
    });
    const summary = debugger_.formatSummary(trace);
    expect(summary).toContain("BM25=");
    expect(summary).toContain("Vec=");
    expect(summary).toContain("Fused=");
  });

  it("contains method", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({
      cosineScore: 0.5,
      docId: "doc-1",
    });
    const summary = debugger_.formatSummary(trace);
    expect(summary).toContain("log_odds");
  });

  it("unknown doc id", () => {
    const debugger_ = createDebugger();
    const trace = debugger_.traceDocument({ cosineScore: 0.5 });
    const summary = debugger_.formatSummary(trace);
    expect(summary.startsWith("unknown:")).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// formatComparison
// ---------------------------------------------------------------------------

describe("formatComparison", () => {
  it("contains both doc ids", () => {
    const debugger_ = createDebugger();
    const a = debugger_.traceDocument({
      bm25Score: 8.0,
      tf: 5.0,
      docLenRatio: 0.6,
      cosineScore: 0.74,
      docId: "doc-42",
    });
    const b = debugger_.traceDocument({
      bm25Score: 6.0,
      tf: 3.0,
      docLenRatio: 0.8,
      cosineScore: 0.5,
      docId: "doc-77",
    });
    const result = debugger_.compare(a, b);
    const output = debugger_.formatComparison(result);
    expect(output).toContain("doc-42");
    expect(output).toContain("doc-77");
  });

  it("contains signal table", () => {
    const debugger_ = createDebugger();
    const a = debugger_.traceDocument({
      bm25Score: 8.0,
      tf: 5.0,
      docLenRatio: 0.6,
      cosineScore: 0.74,
      docId: "doc-42",
    });
    const b = debugger_.traceDocument({
      bm25Score: 6.0,
      tf: 3.0,
      docLenRatio: 0.8,
      cosineScore: 0.5,
      docId: "doc-77",
    });
    const result = debugger_.compare(a, b);
    const output = debugger_.formatComparison(result);
    expect(output).toContain("Signal");
    expect(output).toContain("BM25");
    expect(output).toContain("Vector");
    expect(output).toContain("Fused");
    expect(output).toContain("delta");
  });

  it("contains rank order", () => {
    const debugger_ = createDebugger();
    const a = debugger_.traceDocument({ cosineScore: 0.9, docId: "a" });
    const b = debugger_.traceDocument({ cosineScore: 0.3, docId: "b" });
    const result = debugger_.compare(a, b);
    const output = debugger_.formatComparison(result);
    expect(output).toContain("Rank order:");
  });

  it("contains dominant signal", () => {
    const debugger_ = createDebugger();
    const a = debugger_.traceDocument({
      bm25Score: 8.0,
      tf: 5.0,
      docLenRatio: 0.6,
      cosineScore: 0.74,
      docId: "a",
    });
    const b = debugger_.traceDocument({
      bm25Score: 6.0,
      tf: 3.0,
      docLenRatio: 0.8,
      cosineScore: 0.5,
      docId: "b",
    });
    const result = debugger_.compare(a, b);
    const output = debugger_.formatComparison(result);
    expect(output).toContain("Dominant signal:");
  });

  it("contains crossover note when signals disagree", () => {
    const debugger_ = createDebugger();
    const a = debugger_.traceDocument({
      bm25Score: 3.0,
      tf: 1.0,
      docLenRatio: 1.5,
      cosineScore: 0.95,
      docId: "a",
    });
    const b = debugger_.traceDocument({
      bm25Score: 10.0,
      tf: 8.0,
      docLenRatio: 0.5,
      cosineScore: 0.1,
      docId: "b",
    });
    const result = debugger_.compare(a, b);
    if (result.crossoverStage !== null) {
      const output = debugger_.formatComparison(result);
      expect(output).toContain("Note:");
    }
  });

  it("default doc labels when doc_id is null", () => {
    const debugger_ = createDebugger();
    const a = debugger_.traceDocument({ cosineScore: 0.9 });
    const b = debugger_.traceDocument({ cosineScore: 0.3 });
    const result = debugger_.compare(a, b);
    const output = debugger_.formatComparison(result);
    expect(output).toContain("doc_a");
    expect(output).toContain("doc_b");
  });
});

// ---------------------------------------------------------------------------
// End-to-end: full pipeline verification
// ---------------------------------------------------------------------------

describe("endToEnd", () => {
  it("full pipeline without base rate", () => {
    const transform = createTransform();
    const debugger_ = new FusionDebugger(transform);
    const score = 8.42;
    const tf = 5.0;
    const dl = 0.6;
    const cos = 0.74;

    // Manual computation
    const likelihood = transform.likelihood(score) as number;
    const prior = BayesianProbabilityTransform.compositePrior(
      tf,
      dl,
    ) as number;
    const posterior = BayesianProbabilityTransform.posterior(
      likelihood,
      prior,
    ) as number;
    const vecProb = cosineToProbability(cos) as number;
    const fused = logOddsConjunction([posterior, vecProb]) as number;

    // Debugger trace
    const docTrace = debugger_.traceDocument({
      bm25Score: score,
      tf,
      docLenRatio: dl,
      cosineScore: cos,
      docId: "e2e",
    });

    const docBM25 = docTrace.signals["BM25"] as BM25SignalTrace;
    const docVec = docTrace.signals["Vector"] as VectorSignalTrace;

    expect(docBM25.posterior).toBeCloseTo(posterior);
    expect(docVec.probability).toBeCloseTo(vecProb);
    expect(docTrace.finalProbability).toBeCloseTo(fused, 9);
  });

  it("full pipeline with base rate", () => {
    const transform = createTransformWithBaseRate();
    const debugger_ = new FusionDebugger(transform);
    const score = 8.42;
    const tf = 5.0;
    const dl = 0.6;
    const cos = 0.74;

    const posterior = BayesianProbabilityTransform.posterior(
      transform.likelihood(score) as number,
      BayesianProbabilityTransform.compositePrior(tf, dl) as number,
      transform.baseRate,
    ) as number;
    const vecProb = cosineToProbability(cos) as number;
    const fused = logOddsConjunction([posterior, vecProb]) as number;

    const docTrace = debugger_.traceDocument({
      bm25Score: score,
      tf,
      docLenRatio: dl,
      cosineScore: cos,
      docId: "e2e-br",
    });

    const docBM25 = docTrace.signals["BM25"] as BM25SignalTrace;
    expect(docBM25.posterior).toBeCloseTo(posterior);
    expect(docTrace.finalProbability).toBeCloseTo(fused, 9);
  });

  it("prob_and fusion", () => {
    const debugger_ = createDebugger();
    const score = 5.0;
    const tf = 3.0;
    const dl = 0.8;
    const cos = 0.6;
    const docTrace = debugger_.traceDocument({
      bm25Score: score,
      tf,
      docLenRatio: dl,
      cosineScore: cos,
      method: "prob_and",
    });
    const bm25Post = (docTrace.signals["BM25"] as BM25SignalTrace)
      .posterior;
    const vecProb = (docTrace.signals["Vector"] as VectorSignalTrace)
      .probability;
    const expected = probAnd([bm25Post, vecProb]) as number;
    expect(docTrace.finalProbability).toBeCloseTo(expected, 9);
  });

  it("prob_or fusion", () => {
    const debugger_ = createDebugger();
    const score = 5.0;
    const tf = 3.0;
    const dl = 0.8;
    const cos = 0.6;
    const docTrace = debugger_.traceDocument({
      bm25Score: score,
      tf,
      docLenRatio: dl,
      cosineScore: cos,
      method: "prob_or",
    });
    const bm25Post = (docTrace.signals["BM25"] as BM25SignalTrace)
      .posterior;
    const vecProb = (docTrace.signals["Vector"] as VectorSignalTrace)
      .probability;
    const expected = probOr([bm25Post, vecProb]) as number;
    expect(docTrace.finalProbability).toBeCloseTo(expected, 9);
  });

  it("weighted fusion", () => {
    const debugger_ = createDebugger();
    const weights = [0.7, 0.3];
    const docTrace = debugger_.traceDocument({
      bm25Score: 5.0,
      tf: 3.0,
      docLenRatio: 0.8,
      cosineScore: 0.6,
      weights,
    });
    const bm25Post = (docTrace.signals["BM25"] as BM25SignalTrace)
      .posterior;
    const vecProb = (docTrace.signals["Vector"] as VectorSignalTrace)
      .probability;
    const expected = logOddsConjunction(
      [bm25Post, vecProb],
      undefined,
      weights,
    ) as number;
    expect(docTrace.finalProbability).toBeCloseTo(expected, 9);
  });

  it("hierarchical AND/OR/NOT", () => {
    const debugger_ = createDebugger();
    const pTitle = 0.85;
    const pBody = 0.7;
    const pVector = 0.8;
    const pSpam = 0.9;

    // Step 1: OR(title, body)
    const step1 = debugger_.traceFusion([pTitle, pBody], {
      names: ["title", "body"],
      method: "prob_or",
    });
    const expectedOr = probOr([pTitle, pBody]) as number;
    expect(step1.fusedProbability).toBeCloseTo(expectedOr, 9);

    // Step 2: NOT(spam)
    const step2 = debugger_.traceNot(pSpam, { name: "spam" });
    const expectedNot = probNot(pSpam) as number;
    expect(step2.complement).toBeCloseTo(expectedNot, 9);

    // Step 3: AND(step1, vector, NOT(spam))
    const step3 = debugger_.traceFusion(
      [step1.fusedProbability, pVector, step2.complement],
      {
        names: ["OR(title,body)", "vector", "NOT(spam)"],
        method: "prob_and",
      },
    );
    const expectedAnd = probAnd([
      expectedOr,
      pVector,
      expectedNot,
    ]) as number;
    expect(step3.fusedProbability).toBeCloseTo(expectedAnd, 9);
  });

  it("hierarchical nested OR of ANDs", () => {
    const debugger_ = createDebugger();
    const a = 0.9;
    const b = 0.8;
    const c = 0.6;
    const d = 0.7;

    const left = debugger_.traceFusion([a, b], { method: "prob_and" });
    const right = debugger_.traceFusion([c, d], { method: "prob_and" });
    const final = debugger_.traceFusion(
      [left.fusedProbability, right.fusedProbability],
      { names: ["AND(a,b)", "AND(c,d)"], method: "prob_or" },
    );

    const expectedLeft = probAnd([a, b]) as number;
    const expectedRight = probAnd([c, d]) as number;
    const expected = probOr([expectedLeft, expectedRight]) as number;
    expect(final.fusedProbability).toBeCloseTo(expected, 9);
  });

  it("NOT signal fed into log_odds fusion", () => {
    const debugger_ = createDebugger();
    const pMatch = 0.85;
    const pExclude = 0.7;

    const notTrace = debugger_.traceNot(pExclude);
    const fused = debugger_.traceFusion(
      [pMatch, notTrace.complement],
      { names: ["match", "NOT(exclude)"], method: "log_odds" },
    );

    const expected = logOddsConjunction([
      pMatch,
      probNot(pExclude) as number,
    ]) as number;
    expect(fused.fusedProbability).toBeCloseTo(expected, 9);
  });
});
