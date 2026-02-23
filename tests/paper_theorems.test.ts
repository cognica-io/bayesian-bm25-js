//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

// Verification tests for theorems from both papers.
//
// Paper 1: "Bayesian BM25: A Probabilistic Framework for Hybrid Text and
//           Vector Search" (Jeong, 2026)
// Paper 2: "From Bayesian Inference to Neural Computation" (Jeong, 2026)
//
// Each describe block corresponds to a specific theorem, definition, or
// section from the papers.  The tests verify both exact numerical values
// and structural properties that must hold for all valid inputs.

import { describe, expect, it } from "vitest";

import {
  BayesianProbabilityTransform,
  clampProbability,
  logit,
  sigmoid,
} from "../src/probability.js";
import {
  cosineToProbability,
  logOddsConjunction,
  probAnd,
  probOr,
} from "../src/fusion.js";

// -- Seeded PRNG and helpers ------------------------------------------------

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randUniform(
  rng: () => number,
  low: number,
  high: number,
  size: number,
): number[] {
  const result: number[] = [];
  for (let i = 0; i < size; i++) {
    result.push(low + rng() * (high - low));
  }
  return result;
}

function randInt(rng: () => number, low: number, high: number): number {
  return low + Math.floor(rng() * (high - low));
}

function linspace(start: number, end: number, n: number): number[] {
  const result: number[] = [];
  for (let i = 0; i < n; i++) {
    result.push(start + ((end - start) * i) / (n - 1));
  }
  return result;
}

function allClose(
  a: number[],
  b: number[],
  atol: number = 1e-10,
): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i]! - b[i]!) > atol) return false;
  }
  return true;
}

function assertAllClose(
  a: number[],
  b: number[],
  atol: number = 1e-10,
): void {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs(a[i]! - b[i]!)).toBeLessThanOrEqual(atol);
  }
}

// -----------------------------------------------------------------------
// Paper 1: Bayesian BM25
// -----------------------------------------------------------------------

describe("SigmoidProperties (Lemma 2.1.3)", () => {
  it("sigma(x) + sigma(-x) = 1 for all x (symmetry)", () => {
    const rng = mulberry32(42);
    const x = randUniform(rng, -100, 100, 10000);
    for (const xi of x) {
      const sum = (sigmoid(xi) as number) + (sigmoid(-xi) as number);
      expect(Math.abs(sum - 1.0)).toBeLessThanOrEqual(1e-12);
    }
  });

  it("sigma'(x) = sigma(x) * (1 - sigma(x)) (derivative identity)", () => {
    const x = linspace(-10, 10, 1000);
    const h = 1e-7;
    for (const xi of x) {
      const s = sigmoid(xi) as number;
      const analytical = s * (1.0 - s);
      const numerical =
        ((sigmoid(xi + h) as number) - (sigmoid(xi - h) as number)) /
        (2 * h);
      expect(Math.abs(analytical - numerical)).toBeLessThan(1e-6);
    }
  });

  it("0 < sigma(x) < 1 for all finite x (bounds)", () => {
    const rng = mulberry32(42);
    const x = randUniform(rng, -36, 36, 10000);
    for (const xi of x) {
      const s = sigmoid(xi) as number;
      expect(s).toBeGreaterThan(0);
      expect(s).toBeLessThan(1);
    }
  });

  it("sigma is strictly increasing (monotonicity)", () => {
    const x = linspace(-20, 20, 10000);
    const s = sigmoid(x) as number[];
    for (let i = 1; i < s.length; i++) {
      expect(s[i]!).toBeGreaterThan(s[i - 1]!);
    }
  });
});

describe("LogitSigmoidDuality (Lemma 2.1.4)", () => {
  it("sigma(logit(p)) = p for p in (0, 1)", () => {
    const rng = mulberry32(42);
    const p = randUniform(rng, 0.001, 0.999, 10000);
    for (const pi of p) {
      const recovered = sigmoid(logit(pi) as number) as number;
      expect(Math.abs(recovered - pi)).toBeLessThanOrEqual(1e-10);
    }
  });

  it("logit(sigma(x)) = x for finite x", () => {
    const x = linspace(-15, 15, 1000);
    for (const xi of x) {
      const recovered = logit(sigmoid(xi) as number) as number;
      expect(Math.abs(recovered - xi)).toBeLessThanOrEqual(1e-8);
    }
  });
});

describe("PosteriorFormula (Theorem 4.1.3)", () => {
  it("log-odds equivalence: two formulations agree", () => {
    const rng = mulberry32(42);
    const L = randUniform(rng, 0.01, 0.99, 10000);
    const p = randUniform(rng, 0.01, 0.99, 10000);

    for (let i = 0; i < L.length; i++) {
      // Path 1: direct formula (Eq. 22)
      const direct = BayesianProbabilityTransform.posterior(
        L[i]!,
        p[i]!,
      ) as number;

      // Path 2: log-odds addition
      const logOddsPath = sigmoid(
        (logit(L[i]!) as number) + (logit(p[i]!) as number),
      ) as number;

      expect(Math.abs(direct - logOddsPath)).toBeLessThanOrEqual(1e-9);
    }
  });

  it("uniform prior identity: posterior = likelihood when prior = 0.5", () => {
    const rng = mulberry32(42);
    const L = randUniform(rng, 0.01, 0.99, 1000);
    for (const li of L) {
      const posterior = BayesianProbabilityTransform.posterior(
        li,
        0.5,
      ) as number;
      expect(Math.abs(posterior - li)).toBeLessThanOrEqual(1e-9);
    }
  });
});

describe("Monotonicity (Theorem 4.3.1)", () => {
  it("posterior is monotonic in score for fixed prior", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 100; trial++) {
      const alpha = 0.1 + rng() * 4.9;
      const beta = -2.0 + rng() * 7.0;
      const prior = 0.1 + rng() * 0.8;

      const t = new BayesianProbabilityTransform(alpha, beta);
      const scores = randUniform(rng, -5, 10, 50).sort((a, b) => a - b);
      const likelihoods = t.likelihood(scores) as number[];
      const posteriors = BayesianProbabilityTransform.posterior(
        likelihoods,
        likelihoods.map(() => prior),
      ) as number[];

      for (let i = 1; i < posteriors.length; i++) {
        expect(posteriors[i]!).toBeGreaterThanOrEqual(posteriors[i - 1]!);
      }
    }
  });

  it("full pipeline is monotonic when tf and ratio are fixed", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 50; trial++) {
      const alpha = 0.1 + rng() * 4.9;
      const beta = -2.0 + rng() * 7.0;
      const tf = rng() * 20;
      const ratio = 0.1 + rng() * 2.9;

      const t = new BayesianProbabilityTransform(alpha, beta);
      const scores = randUniform(rng, -5, 10, 100).sort((a, b) => a - b);
      const probs = t.scoreToProbability(
        scores,
        scores.map(() => tf),
        scores.map(() => ratio),
      ) as number[];

      for (let i = 1; i < probs.length; i++) {
        expect(probs[i]!).toBeGreaterThanOrEqual(probs[i - 1]!);
      }
    }
  });
});

describe("PriorBounds (Theorem 4.2.4)", () => {
  it("composite prior in [0.1, 0.9] for random inputs", () => {
    const rng = mulberry32(42);
    const tf = randUniform(rng, 0, 100, 10000);
    const ratio = randUniform(rng, 0, 10, 10000);
    const prior = BayesianProbabilityTransform.compositePrior(
      tf,
      ratio,
    ) as number[];
    for (const p of prior) {
      expect(p).toBeGreaterThanOrEqual(0.1);
      expect(p).toBeLessThanOrEqual(0.9);
    }
  });

  it("tf prior in [0.2, 0.9]", () => {
    const rng = mulberry32(42);
    const tf = randUniform(rng, 0, 1000, 10000);
    const p = BayesianProbabilityTransform.tfPrior(tf) as number[];
    for (const pi of p) {
      expect(pi).toBeGreaterThanOrEqual(0.2);
      expect(pi).toBeLessThanOrEqual(0.9);
    }
  });

  it("norm prior in [0.3, 0.9]", () => {
    const rng = mulberry32(42);
    const ratio = randUniform(rng, 0, 100, 10000);
    const p = BayesianProbabilityTransform.normPrior(ratio) as number[];
    for (const pi of p) {
      expect(pi).toBeGreaterThanOrEqual(0.3);
      expect(pi).toBeLessThanOrEqual(0.9);
    }
  });
});

describe("BaseRateLogOdds", () => {
  it("three-term log-odds equivalence", () => {
    const rng = mulberry32(42);
    const L = randUniform(rng, 0.01, 0.99, 10000);
    const p = randUniform(rng, 0.01, 0.99, 10000);

    for (const br of [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999]) {
      for (let i = 0; i < L.length; i++) {
        const direct = BayesianProbabilityTransform.posterior(
          L[i]!,
          p[i]!,
          br,
        ) as number;
        const logOddsPath = sigmoid(
          (logit(L[i]!) as number) +
            (logit(br) as number) +
            (logit(p[i]!) as number),
        ) as number;
        expect(Math.abs(direct - logOddsPath)).toBeLessThanOrEqual(1e-9);
      }
    }
  });

  it("base rate 0.5 reduces to two-term", () => {
    const rng = mulberry32(42);
    const L = randUniform(rng, 0.01, 0.99, 10000);
    const p = randUniform(rng, 0.01, 0.99, 10000);

    for (let i = 0; i < L.length; i++) {
      const twoTerm = BayesianProbabilityTransform.posterior(
        L[i]!,
        p[i]!,
      ) as number;
      const threeTerm = BayesianProbabilityTransform.posterior(
        L[i]!,
        p[i]!,
        0.5,
      ) as number;
      expect(Math.abs(threeTerm - twoTerm)).toBeLessThanOrEqual(1e-9);
    }
  });

  it("monotonicity holds with any base rate", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 100; trial++) {
      const alpha = 0.1 + rng() * 4.9;
      const beta = -2.0 + rng() * 7.0;
      const prior = 0.1 + rng() * 0.8;
      const br = 0.001 + rng() * 0.998;

      const t = new BayesianProbabilityTransform(alpha, beta, br);
      const scores = randUniform(rng, -5, 10, 50).sort((a, b) => a - b);
      const probs = t.scoreToProbability(
        scores,
        scores.map(() => prior),
        scores.map(() => prior),
      ) as number[];

      for (let i = 1; i < probs.length; i++) {
        expect(probs[i]!).toBeGreaterThanOrEqual(probs[i - 1]!);
      }
    }
  });
});

describe("PaperValues (Section 11.1)", () => {
  it("section 11.1 ordering", () => {
    const t = new BayesianProbabilityTransform(1.0, 0.0);
    const scores = [1.0464478, 0.56150854, 1.1230172];
    const tf = [5.0, 3.0, 7.0];
    const ratio = [0.5, 0.5, 0.5];
    const probs = t.scoreToProbability(scores, tf, ratio) as number[];

    for (const p of probs) {
      expect(p).toBeGreaterThan(0);
      expect(p).toBeLessThan(1);
    }
    // Score ordering: s[2] > s[0] > s[1], with comparable priors
    expect(probs[2]!).toBeGreaterThan(probs[1]!);
    expect(probs[0]!).toBeGreaterThan(probs[1]!);
  });
});

// -----------------------------------------------------------------------
// Paper 2: From Bayesian Inference to Neural Computation
// -----------------------------------------------------------------------

describe("ScaleNeutrality (Theorem 4.1.2)", () => {
  it("identical signals with alpha=0 pass through unchanged", () => {
    for (const p of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      for (const n of [1, 2, 3, 5, 10]) {
        const signals = new Array(n).fill(p);
        const result = logOddsConjunction(signals, 0.0) as number;
        expect(Math.abs(result - p)).toBeLessThanOrEqual(1e-8);
      }
    }
  });

  it("identical signals with alpha=0.5 are amplified by n^0.5", () => {
    for (const p of [0.6, 0.7, 0.8, 0.9]) {
      for (const n of [2, 3, 5]) {
        const signals = new Array(n).fill(p);
        const result = logOddsConjunction(signals, 0.5) as number;
        const expected = sigmoid(
          (logit(p) as number) * Math.sqrt(n),
        ) as number;
        expect(Math.abs(result - expected)).toBeLessThanOrEqual(1e-10);
      }
    }
  });
});

describe("SignPreservation (Theorem 4.2.2)", () => {
  it("positive log-odds stays positive", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 1000; trial++) {
      const n = randInt(rng, 2, 6);
      const probs = randUniform(rng, 0.55, 0.99, n);
      let logitSum = 0;
      for (const p of probs) {
        logitSum += logit(p) as number;
      }
      if (logitSum / n <= 0) continue;
      const result = logOddsConjunction(probs) as number;
      expect(result).toBeGreaterThan(0.5);
    }
  });

  it("negative log-odds stays negative", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 1000; trial++) {
      const n = randInt(rng, 2, 6);
      const probs = randUniform(rng, 0.01, 0.45, n);
      let logitSum = 0;
      for (const p of probs) {
        logitSum += logit(p) as number;
      }
      if (logitSum / n >= 0) continue;
      const result = logOddsConjunction(probs) as number;
      expect(result).toBeLessThan(0.5);
    }
  });
});

describe("IrrelevanceNonInversion (Corollary 4.2.3)", () => {
  it("all irrelevant stays irrelevant", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 1000; trial++) {
      const n = randInt(rng, 2, 10);
      const probs = randUniform(rng, 0.01, 0.49, n);
      for (const alpha of [0.0, 0.5, 1.0, 2.0]) {
        const result = logOddsConjunction(probs, alpha) as number;
        expect(result).toBeLessThan(0.5);
      }
    }
  });

  it("all relevant stays relevant", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 1000; trial++) {
      const n = randInt(rng, 2, 10);
      const probs = randUniform(rng, 0.51, 0.99, n);
      for (const alpha of [0.0, 0.5, 1.0, 2.0]) {
        const result = logOddsConjunction(probs, alpha) as number;
        expect(result).toBeGreaterThan(0.5);
      }
    }
  });
});

describe("Paper2NumericalTable (Section 4.5)", () => {
  const tableValues: [number, number, number, number][] = [
    [0.9, 0.9, 0.81, 0.96],
    [0.7, 0.7, 0.49, 0.77],
    [0.7, 0.3, 0.21, 0.50],
    [0.3, 0.3, 0.09, 0.23],
  ];

  for (const [p1, p2, expectedAnd, expectedConj] of tableValues) {
    it(`table values: (${p1}, ${p2})`, () => {
      const probs = [p1!, p2!];
      expect(probAnd(probs) as number).toBeCloseTo(expectedAnd!, 1);
      expect(logOddsConjunction(probs) as number).toBeCloseTo(
        expectedConj!,
        1,
      );
    });
  }

  it("exact computation (0.9, 0.9)", () => {
    const probs = [0.9, 0.9];
    const lBar = logit(0.9) as number;
    const lAdj = lBar * Math.sqrt(2);
    const expected = sigmoid(lAdj) as number;
    const result = logOddsConjunction(probs) as number;
    expect(Math.abs(result - expected)).toBeLessThanOrEqual(1e-10);
  });

  it("exact computation (0.7, 0.3) -> 0.5 exactly", () => {
    const probs = [0.7, 0.3];
    const result = logOddsConjunction(probs) as number;
    // logit(0.7) + logit(0.3) = logit(0.7) + (-logit(0.7)) = 0
    // by sigmoid symmetry (Lemma 2.1.3)
    expect(Math.abs(result - 0.5)).toBeLessThanOrEqual(1e-10);
  });
});

describe("DisagreementModeration (Theorem 4.5.1 (ii))", () => {
  it("symmetric disagreement -> 0.5", () => {
    const testPs = linspace(0.01, 0.99, 50);
    for (const p of testPs) {
      const probs = [p, 1.0 - p];
      const result = logOddsConjunction(probs) as number;
      expect(Math.abs(result - 0.5)).toBeLessThanOrEqual(1e-8);
    }
  });
});

describe("LogisticRegressionEquivalence (Theorem 5.2.1a)", () => {
  it("sigmoid-calibrated signals = logistic regression", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 100; trial++) {
      const n = randInt(rng, 2, 6);
      const alphas = randUniform(rng, 0.5, 3.0, n);
      const betas = randUniform(rng, -1.0, 3.0, n);
      const scores = randUniform(rng, -2.0, 5.0, n);
      const confAlpha = 0.5;

      // Calibrate each signal via sigmoid
      const calibrated: number[] = [];
      for (let i = 0; i < n; i++) {
        calibrated.push(
          sigmoid(alphas[i]! * (scores[i]! - betas[i]!)) as number,
        );
      }

      // Path 1: log-odds conjunction
      const result = logOddsConjunction(calibrated, confAlpha) as number;

      // Path 2: direct logistic regression
      let preActivationSum = 0;
      for (let i = 0; i < n; i++) {
        preActivationSum += alphas[i]! * (scores[i]! - betas[i]!);
      }
      const lBar = preActivationSum / n;
      const lAdj = lBar * n ** confAlpha;
      const expected = sigmoid(lAdj) as number;

      expect(Math.abs(result - expected)).toBeLessThanOrEqual(1e-10);
    }
  });
});

describe("AgreementAmplification (Theorem 4.5.1 (i))", () => {
  it("amplification exceeds input", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 500; trial++) {
      const n = randInt(rng, 2, 6);
      const p = 0.55 + rng() * 0.4;
      const probs = new Array(n).fill(p);
      const result = logOddsConjunction(probs, 0.5) as number;
      expect(result).toBeGreaterThan(p);
    }
  });

  it("more signals means more amplification", () => {
    for (const p of [0.6, 0.7, 0.8, 0.9]) {
      let prev = p;
      for (let n = 2; n < 8; n++) {
        const probs = new Array(n).fill(p);
        const result = logOddsConjunction(probs, 0.5) as number;
        expect(result).toBeGreaterThanOrEqual(prev - 1e-10);
        prev = result;
      }
    }
  });
});

describe("ConjunctionVsProductRule", () => {
  it("conjunction beats product for agreement", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 1000; trial++) {
      const n = randInt(rng, 2, 6);
      const probs = randUniform(rng, 0.55, 0.99, n);
      const conj = logOddsConjunction(probs, 0.5) as number;
      const prod = probAnd(probs) as number;
      expect(conj).toBeGreaterThan(prod);
    }
  });
});

// -----------------------------------------------------------------------
// Output range and numerical stability
// -----------------------------------------------------------------------

describe("ConjunctionStrictBounds (Theorem 5.1.2)", () => {
  it("prob_and(probs) < min(probs) for n >= 2", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 1000; trial++) {
      const n = randInt(rng, 2, 6);
      const probs = randUniform(rng, 0.01, 0.99, n);
      const result = probAnd(probs) as number;
      expect(result).toBeGreaterThan(0);
      expect(result).toBeLessThan(Math.min(...probs));
    }
  });
});

describe("DisjunctionStrictBounds (Theorem 5.2.2)", () => {
  it("prob_or(probs) > max(probs) for n >= 2", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 1000; trial++) {
      const n = randInt(rng, 2, 6);
      const probs = randUniform(rng, 0.01, 0.99, n);
      const result = probOr(probs) as number;
      expect(result).toBeGreaterThan(Math.max(...probs));
      expect(result).toBeLessThan(1);
    }
  });
});

describe("LogOPEquivalence (Theorem 4.1.2a)", () => {
  it("log-odds mean equals normalized PoE", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 1000; trial++) {
      const n = randInt(rng, 2, 7);
      const probs = randUniform(rng, 0.01, 0.99, n);

      // Path 1: log-odds mean (alpha=0)
      let logitSum = 0;
      for (const p of probs) {
        logitSum += logit(p) as number;
      }
      const logOddsResult = sigmoid(logitSum / n) as number;

      // Path 2: normalized PoE
      let prodP = 1.0;
      let prod1mp = 1.0;
      for (const p of probs) {
        prodP *= p ** (1.0 / n);
        prod1mp *= (1.0 - p) ** (1.0 / n);
      }
      const poeResult = prodP / (prodP + prod1mp);

      expect(Math.abs(logOddsResult - poeResult)).toBeLessThanOrEqual(
        1e-10,
      );
    }
  });
});

describe("HeterogeneousSignalCombination (Remark 5.2.3)", () => {
  it("BM25 + cosine pipeline produces valid monotonic probabilities", () => {
    const bm25Scores = [0.5, 1.0, 2.0, 3.0, 5.0];
    const cosineScores = [0.2, 0.4, 0.6, 0.8, 0.95];

    // Calibrate BM25 via sigmoid (alpha=1, beta=1)
    const bm25Probs = bm25Scores.map(
      (s) => sigmoid(1.0 * (s - 1.0)) as number,
    );

    // Calibrate cosine via linear mapping
    const cosineProbs = cosineScores.map(
      (s) => cosineToProbability(s) as number,
    );

    const results: number[] = [];
    for (let i = 0; i < bm25Scores.length; i++) {
      const probs = [bm25Probs[i]!, cosineProbs[i]!];
      const result = logOddsConjunction(probs) as number;
      expect(result).toBeGreaterThan(0);
      expect(result).toBeLessThan(1);
      results.push(result);
    }

    // Monotonicity: both signals increase -> result should increase
    for (let i = 1; i < results.length; i++) {
      expect(results[i]!).toBeGreaterThan(results[i - 1]!);
    }
  });

  it("logit(cosine_to_probability(s)) is nonlinear in s", () => {
    const s = linspace(-0.9, 0.9, 100);
    const transformed = s.map(
      (si) => logit(cosineToProbability(si) as number) as number,
    );

    // Second differences of a linear function are zero
    const secondDiff: number[] = [];
    for (let i = 0; i < transformed.length - 2; i++) {
      secondDiff.push(
        transformed[i + 2]! - 2 * transformed[i + 1]! + transformed[i]!,
      );
    }

    const allZero = secondDiff.every(
      (d) => Math.abs(d) < 1e-8,
    );
    expect(allZero).toBe(false);
  });
});

describe("SingleSignalIdentity (Proposition 4.3.2)", () => {
  it("n=1 returns P_1 for any alpha value", () => {
    const rng = mulberry32(42);
    const probs = randUniform(rng, 0.01, 0.99, 50);
    for (const alpha of [0.0, 0.5, 1.0, 2.0, 5.0]) {
      for (const p of probs) {
        const result = logOddsConjunction([p], alpha) as number;
        expect(Math.abs(result - p)).toBeLessThanOrEqual(1e-8);
      }
    }
  });
});

describe("WeightedAlphaComposition (Theorem 8.3 + Section 4.2)", () => {
  it("weighted alpha composition matches hand-computed values", () => {
    const probs = [0.8, 0.6];
    const w = [0.7, 0.3];
    const alpha = 0.5;
    const n = 2;

    // Hand-compute expected value
    const lWeighted =
      w[0]! * (logit(probs[0]!) as number) +
      w[1]! * (logit(probs[1]!) as number);
    const expected = sigmoid(n ** alpha * lWeighted) as number;

    const result = logOddsConjunction(probs, alpha, w) as number;
    expect(Math.abs(result - expected)).toBeLessThanOrEqual(1e-10);
  });

  it("uniform weights with explicit alpha matches unweighted", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 100; trial++) {
      const n = randInt(rng, 2, 6);
      const probs = randUniform(rng, 0.01, 0.99, n);
      const alpha = rng() * 2.0;
      const uniformW = new Array(n).fill(1.0 / n);

      const unweighted = logOddsConjunction(probs, alpha) as number;
      const weighted = logOddsConjunction(
        probs,
        alpha,
        uniformW,
      ) as number;
      expect(Math.abs(weighted - unweighted)).toBeLessThanOrEqual(1e-10);
    }
  });
});

describe("MonotoneShrinkage (Theorem 3.2.1 + Corollary 3.2.2)", () => {
  it("prob_and decreases with more signals", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 500; trial++) {
      const maxN = randInt(rng, 3, 9);
      const allProbs = randUniform(rng, 0.01, 0.99, maxN);

      let prevResult = probAnd(allProbs.slice(0, 2)) as number;
      for (let n = 3; n <= maxN; n++) {
        const result = probAnd(allProbs.slice(0, n)) as number;
        expect(result).toBeLessThan(prevResult);
        prevResult = result;
      }
    }
  });

  it("many signals push prob_and toward zero", () => {
    const probs = new Array(50).fill(0.9);
    const result = probAnd(probs) as number;
    // 0.9^50 = 0.00515...
    expect(result).toBeLessThan(0.01);
  });
});

describe("InformationLoss (Proposition 3.4.1)", () => {
  it("same product gives same prob_and result", () => {
    // (0.9, 0.1) and (0.3, 0.3) both have product 0.09
    const resultA = probAnd([0.9, 0.1]) as number;
    const resultB = probAnd([0.3, 0.3]) as number;
    expect(Math.abs(resultA - resultB)).toBeLessThanOrEqual(1e-10);
  });

  it("conjunction breaks invariance", () => {
    const disagreement = logOddsConjunction([0.9, 0.1]) as number;
    const agreement = logOddsConjunction([0.3, 0.3]) as number;
    // Disagreement (0.9, 0.1) -> ~0.5
    // Agreement (0.3, 0.3) -> ~0.23
    expect(Math.abs(disagreement - agreement)).toBeGreaterThan(0.01);
  });

  it("randomized same product invariance", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 500; trial++) {
      const target = 0.01 + rng() * 0.98;
      const a = Math.max(target, 0.01) + rng() * (0.99 - Math.max(target, 0.01));
      const b = Math.max(target, 0.01) + rng() * (0.99 - Math.max(target, 0.01));
      const pairA = [a, target / a];
      const pairB = [b, target / b];
      // Both pairs should produce the same prob_and result
      if (pairA[1]! <= 0 || pairA[1]! >= 1) continue;
      if (pairB[1]! <= 0 || pairB[1]! >= 1) continue;
      const resultA = probAnd(pairA) as number;
      const resultB = probAnd(pairB) as number;
      expect(Math.abs(resultA - resultB)).toBeLessThanOrEqual(1e-8);
    }
  });
});

describe("SqrtNScalingLaw (Theorem 4.4.1 + Proposition 4.4.2)", () => {
  it("effective logit scales as sqrt(n) for identical signals", () => {
    for (const p of [0.6, 0.7, 0.8, 0.9]) {
      const baseLogit = logit(p) as number;
      for (const n of [2, 3, 4, 5, 8, 10]) {
        const probs = new Array(n).fill(p);
        const result = logOddsConjunction(probs, 0.5) as number;
        const expected = sigmoid(baseLogit * Math.sqrt(n)) as number;
        expect(Math.abs(result - expected)).toBeLessThanOrEqual(1e-10);
      }
    }
  });

  it("sqrt scaling grows slower than linear scaling", () => {
    const p = 0.8;
    for (const n of [2, 3, 5, 10]) {
      const probs = new Array(n).fill(p);
      const sqrtResult = logOddsConjunction(probs, 0.5) as number;
      const linearResult = logOddsConjunction(probs, 1.0) as number;
      expect(linearResult).toBeGreaterThan(sqrtResult);
    }
  });

  it("sqrt scaling amplifies more than no scaling", () => {
    const p = 0.8;
    for (const n of [2, 3, 5, 10]) {
      const probs = new Array(n).fill(p);
      const sqrtResult = logOddsConjunction(probs, 0.5) as number;
      const noScaleResult = logOddsConjunction(probs, 0.0) as number;
      expect(sqrtResult).toBeGreaterThan(noScaleResult);
    }
  });

  it("confidence growth rate: doubling n gives sqrt(2) ratio", () => {
    const p = 0.75;
    const baseLogit = logit(p) as number;
    for (const n of [2, 4, 8]) {
      const lEffN = baseLogit * Math.sqrt(n);
      const lEff2N = baseLogit * Math.sqrt(2 * n);
      const ratio = lEff2N / lEffN;
      expect(Math.abs(ratio - Math.SQRT2)).toBeLessThanOrEqual(1e-10);
    }
  });
});

describe("SpreadProperty (Theorem 4.5.1 (iii))", () => {
  it("disagreement reduces confidence", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 500; trial++) {
      const pHigh = 0.7 + rng() * 0.25;
      const agreeing = [pHigh, pHigh];
      const resultAgree = logOddsConjunction(agreeing, 0.0) as number;

      const pLow = 0.05 + rng() * 0.25;
      const mixed = [pHigh, pHigh, pLow];
      const resultMixed = logOddsConjunction(mixed, 0.0) as number;

      const distAgree = Math.abs(resultAgree - 0.5);
      const distMixed = Math.abs(resultMixed - 0.5);
      expect(distMixed).toBeLessThan(distAgree);
    }
  });

  it("symmetric spread around 0.5 produces 0.5", () => {
    for (const offset of [0.1, 0.2, 0.3, 0.4]) {
      const probs = [0.5 + offset, 0.5 - offset];
      const result = logOddsConjunction(probs, 0.0) as number;
      expect(Math.abs(result - 0.5)).toBeLessThanOrEqual(1e-8);
    }
  });

  it("alpha=0 is spread-invariant (mean logit preserved)", () => {
    const meanLogit = logit(0.75) as number;
    for (const spread of [0.0, 0.5, 1.0, 1.5]) {
      const p1 = sigmoid(meanLogit + spread) as number;
      const p2 = sigmoid(meanLogit - spread) as number;
      const probs = [p1, p2];
      const result = logOddsConjunction(probs, 0.0) as number;
      const expected = sigmoid(meanLogit) as number;
      expect(Math.abs(result - expected)).toBeLessThanOrEqual(1e-8);
    }
  });
});

describe("GeometricMeanResidual (Remark 4.1.3)", () => {
  it("geometric mean differs from log-odds mean for heterogeneous inputs", () => {
    const rng = mulberry32(42);
    let differCount = 0;
    for (let trial = 0; trial < 1000; trial++) {
      const n = randInt(rng, 2, 6);
      const probs = randUniform(rng, 0.1, 0.9, n);

      // Geometric mean in probability space
      let product = 1.0;
      for (const p of probs) {
        product *= p;
      }
      const geoMean = product ** (1.0 / n);

      // Log-odds mean
      let logitSum = 0;
      for (const p of probs) {
        logitSum += logit(p) as number;
      }
      const logOddsMean = sigmoid(logitSum / n) as number;

      if (Math.abs(geoMean - logOddsMean) > 1e-6) {
        differCount++;
      }
    }
    expect(differCount).toBeGreaterThan(900);
  });

  it("identical signals have no residual", () => {
    for (const p of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      for (const n of [2, 3, 5, 10]) {
        const probs = new Array(n).fill(p);
        let product = 1.0;
        for (const pi of probs) {
          product *= pi;
        }
        const geoMean = product ** (1.0 / n);
        let logitSum = 0;
        for (const pi of probs) {
          logitSum += logit(pi) as number;
        }
        const logOddsMean = sigmoid(logitSum / n) as number;
        expect(Math.abs(geoMean - p)).toBeLessThanOrEqual(1e-10);
        expect(Math.abs(logOddsMean - p)).toBeLessThanOrEqual(1e-10);
      }
    }
  });

  it("geometric mean underestimates for high probabilities", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 500; trial++) {
      const n = randInt(rng, 2, 6);
      const probs = randUniform(rng, 0.6, 0.95, n).sort((a, b) => a - b);

      // Check if all are identical
      const allSame = probs.every(
        (p) => Math.abs(p - probs[0]!) < 1e-10,
      );
      if (allSame) continue;

      let product = 1.0;
      for (const p of probs) {
        product *= p;
      }
      const geoMean = product ** (1.0 / n);

      let logitSum = 0;
      for (const p of probs) {
        logitSum += logit(p) as number;
      }
      const logOddsMean = sigmoid(logitSum / n) as number;

      expect(geoMean).toBeLessThan(logOddsMean);
    }
  });
});

describe("SigmoidUniqueness (Theorem 6.2.1)", () => {
  it("sigmoid satisfies all three properties", () => {
    const x = linspace(-10, 10, 1000);

    for (const xi of x) {
      const s = sigmoid(xi) as number;

      // (a) Output in (0, 1)
      expect(s).toBeGreaterThan(0);
      expect(s).toBeLessThan(1);

      // (b) Symmetry: s(x) + s(-x) = 1
      const sum = s + (sigmoid(-xi) as number);
      expect(Math.abs(sum - 1.0)).toBeLessThanOrEqual(1e-12);
    }

    // (c) Self-derivative
    const h = 1e-7;
    for (const xi of x) {
      const s = sigmoid(xi) as number;
      const analytical = s * (1.0 - s);
      const numerical =
        ((sigmoid(xi + h) as number) - (sigmoid(xi - h) as number)) /
        (2 * h);
      expect(Math.abs(analytical - numerical)).toBeLessThan(1e-6);
    }
  });

  it("ReLU violates range property", () => {
    const x = [2.0, 5.0, 10.0];
    const relu = x.map((xi) => Math.max(0, xi));
    expect(relu.some((v) => v > 1)).toBe(true);
  });

  it("rescaled tanh violates self-derivative property", () => {
    const x = linspace(-5, 5, 1000);

    for (const xi of x) {
      const f = (1.0 + Math.tanh(xi)) / 2.0;

      // (a) Range: satisfied
      expect(f).toBeGreaterThan(0);
      expect(f).toBeLessThan(1);

      // (b) Symmetry: satisfied
      const fNeg = (1.0 + Math.tanh(-xi)) / 2.0;
      expect(Math.abs(f + fNeg - 1.0)).toBeLessThanOrEqual(1e-12);
    }

    // (c) Self-derivative: VIOLATED
    let violated = false;
    for (const xi of x) {
      const f = (1.0 + Math.tanh(xi)) / 2.0;
      const actualDeriv = (1.0 - Math.tanh(xi) ** 2) / 2.0;
      const selfDeriv = f * (1.0 - f);
      if (Math.abs(actualDeriv - selfDeriv) > 1e-4) {
        violated = true;
        break;
      }
    }
    expect(violated).toBe(true);
  });
});

describe("OutputRange", () => {
  it("score_to_probability always in (0, 1)", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 100; trial++) {
      const alpha = 0.01 + rng() * 9.99;
      const beta = -10.0 + rng() * 20.0;
      const t = new BayesianProbabilityTransform(alpha, beta);
      const scores = randUniform(rng, -100, 100, 100);
      const tf = randUniform(rng, 0, 100, 100);
      const ratio = randUniform(rng, 0, 10, 100);
      const probs = t.scoreToProbability(scores, tf, ratio) as number[];
      for (const p of probs) {
        expect(p).toBeGreaterThan(0);
        expect(p).toBeLessThan(1);
      }
    }
  });

  it("log_odds_conjunction always in (0, 1)", () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 1000; trial++) {
      const n = randInt(rng, 2, 10);
      const probs = randUniform(rng, 0.01, 0.99, n);
      const alpha = rng();
      const result = logOddsConjunction(probs, alpha) as number;
      expect(result).toBeGreaterThan(0);
      expect(result).toBeLessThan(1);
    }
  });

  it("extreme inputs produce finite results in (0, 1)", () => {
    const t = new BayesianProbabilityTransform(10.0, 0.0);
    // Very large score
    const p1 = t.scoreToProbability(1000.0, 10, 0.5) as number;
    expect(isFinite(p1)).toBe(true);
    expect(p1).toBeGreaterThan(0);
    expect(p1).toBeLessThan(1);
    // Very negative score
    const p2 = t.scoreToProbability(-1000.0, 0, 5.0) as number;
    expect(isFinite(p2)).toBe(true);
    expect(p2).toBeGreaterThan(0);
    expect(p2).toBeLessThan(1);
  });
});
