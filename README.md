# Bayesian BM25 for JavaScript/TypeScript

A probabilistic framework that converts raw BM25 retrieval scores into calibrated relevance probabilities using Bayesian inference. TypeScript port of [bayesian-bm25](https://github.com/cognica-io/bayesian-bm25).

## Overview

Standard BM25 produces unbounded scores that lack consistent meaning across queries, making threshold-based filtering and multi-signal fusion unreliable. Bayesian BM25 addresses this by applying a sigmoid likelihood model with a composite prior (term frequency + document length normalization) and computing Bayesian posteriors that output well-calibrated probabilities in [0, 1]. A corpus-level base rate prior further improves calibration without requiring relevance labels.

Key capabilities:

- **Score-to-probability transform** -- convert raw BM25 scores into calibrated relevance probabilities via sigmoid likelihood + composite prior + Bayesian posterior
- **Base rate calibration** -- corpus-level base rate prior estimated from score distribution decomposes the posterior into three additive log-odds terms
- **Parameter learning** -- batch gradient descent or online SGD with EMA-smoothed gradients and Polyak averaging, with three training modes: balanced (C1), prior-aware (C2), and prior-free (C3)
- **Probabilistic fusion** -- combine multiple probability signals using log-odds conjunction with multiplicative confidence scaling and optional per-signal reliability weights (Log-OP), which resolves the shrinkage problem of naive probabilistic AND
- **Hybrid search** -- `cosineToProbability()` converts vector similarity scores to probabilities for fusion with BM25 signals via weighted log-odds conjunction
- **Balanced fusion** -- `balancedLogOddsFusion()` min-max normalizes sparse and dense logits to equalize voting power before combining, preventing heavy-tailed BM25 logits from drowning the dense signal
- **Boolean NOT** -- `probNot()` computes `P(NOT R) = 1 - P(R)` for exclusion queries, composing with `probAnd()`, `probOr()`, and `logOddsConjunction()` under De Morgan's laws
- **Learnable weights** -- `LearnableLogOddsWeights` learns per-signal reliability weights via Hebbian gradient descent on BCE loss, with batch `fit()`, online `update()`, and Polyak averaging
- **WAND pruning** -- `wandUpperBound()` computes safe Bayesian probability upper bounds for document pruning in top-k retrieval
- **Calibration metrics** -- `expectedCalibrationError()`, `brierScore()`, and `reliabilityDiagram()` for evaluating probability quality
- **Fusion debugging** -- `FusionDebugger` traces every intermediate value through the probability pipeline (likelihood, prior, posterior, fusion) and explains rank differences between documents
- **Search integration** -- built-in BM25 scorer that returns probabilities instead of raw scores, with support for Robertson, Lucene, and ATIRE variants

## Installation

```bash
npm install bayesian-bm25
```

## Quick Start

### Converting BM25 Scores to Probabilities

```typescript
import { BayesianProbabilityTransform } from "bayesian-bm25";

const transform = new BayesianProbabilityTransform(1.5, 1.0, 0.01);

const scores = [0.5, 1.0, 1.5, 2.0, 3.0];
const tfs = [1, 2, 3, 5, 8];
const docLenRatios = [0.3, 0.5, 0.8, 1.0, 1.5];

const probabilities = transform.scoreToProbability(scores, tfs, docLenRatios);
```

### End-to-End Search with Probabilities

```typescript
import { BayesianBM25Scorer } from "bayesian-bm25";

const corpusTokens = [
  ["python", "machine", "learning"],
  ["deep", "learning", "neural", "networks"],
  ["data", "visualization", "tools"],
];

const scorer = new BayesianBM25Scorer({
  k1: 1.2,
  b: 0.75,
  method: "lucene",
  baseRate: "auto",
});
scorer.index(corpusTokens);

const { docIds, probabilities } = scorer.retrieve(
  [["machine", "learning"]],
  3,
);
```

### Combining Multiple Signals

```typescript
import { logOddsConjunction, probAnd, probOr } from "bayesian-bm25";

const signals = [0.85, 0.70, 0.60];

probAnd(signals);              // 0.357 (shrinkage problem)
logOddsConjunction(signals);   // 0.773 (agreement-aware)
```

### Hybrid Text + Vector Search

```typescript
import { cosineToProbability, logOddsConjunction } from "bayesian-bm25";

// BM25 probabilities (from Bayesian BM25)
const bm25Probs = [0.85, 0.60, 0.40];

// Vector search cosine similarities -> probabilities
const cosineScores = [0.92, 0.35, 0.70];
const vectorProbs = cosineToProbability(cosineScores);  // [0.96, 0.675, 0.85]

// Fuse with reliability weights (BM25 weight=0.6, vector weight=0.4)
const stacked = bm25Probs.map((bp, i) => [bp, vectorProbs[i]!]);
const fused = logOddsConjunction(stacked, undefined, [0.6, 0.4]);

// Fuse with weights and confidence scaling (alpha + weights compose)
const fusedScaled = logOddsConjunction(stacked, 0.5, [0.6, 0.4]);
```

### Boolean NOT (Exclusion Queries)

```typescript
import { probNot, probAnd, logOddsConjunction } from "bayesian-bm25";

// "python AND NOT java": keep python-relevant, exclude java-relevant
const pythonProb = 0.85;
const javaProb = 0.70;

const notJava = probNot(javaProb);                     // 0.30
const exclusion = probAnd([pythonProb, notJava]);       // product rule
const fusedExclusion = logOddsConjunction([pythonProb, notJava]); // log-odds

// Array input
const probs = [0.9, 0.7, 0.3];
const complements = probNot(probs);  // [0.1, 0.3, 0.7]
```

### Balanced Sparse-Dense Fusion

```typescript
import { balancedLogOddsFusion } from "bayesian-bm25";

// BM25 probabilities (from Bayesian BM25)
const bm25Probs = [0.85, 0.60, 0.40];

// Dense cosine similarities (from vector search)
const cosineScores = [0.92, 0.35, 0.70];

// Balanced fusion: normalizes logits before combining
const fused = balancedLogOddsFusion(bm25Probs, cosineScores);

// Asymmetric weighting (0.7 = sparse weight, 0.3 implicit dense weight)
const fusedWeighted = balancedLogOddsFusion(bm25Probs, cosineScores, 0.7);
```

### Learnable Fusion Weights

```typescript
import { LearnableLogOddsWeights, logOddsConjunction } from "bayesian-bm25";

// 3-signal hybrid system: BM25, vector, metadata
const learner = new LearnableLogOddsWeights(3);

// Batch training on labeled data
const signalsBatch = [
  [0.9, 0.8, 0.3],  // doc 1: BM25 and vector agree
  [0.2, 0.7, 0.6],  // doc 2: vector is more reliable
  [0.8, 0.3, 0.9],  // doc 3: BM25 and metadata agree
];
const labels = [1, 1, 1];
learner.fit(signalsBatch, labels);

console.log(learner.weights);          // learned reliability weights
console.log(learner.averagedWeights);  // Polyak-averaged (smoother)

// Combine signals using learned weights
const fused = learner.combine([0.85, 0.70, 0.50]);

// Online update from live feedback
learner.update([0.75, 0.60, 0.40], 1.0, { learningRate: 0.01 });
```

### Debugging Fusion Decisions

```typescript
import { FusionDebugger, BayesianProbabilityTransform } from "bayesian-bm25";

const transform = new BayesianProbabilityTransform(1.5, 1.0, 0.01);
const debugger_ = new FusionDebugger(transform);

// Trace a BM25 score through the full probability pipeline
const bm25Trace = debugger_.traceBM25(2.5, 3, 0.8);
console.log(debugger_.formatTrace(bm25Trace));
// => "BM25  score=2.500  L=0.818  prior=0.650  post=0.790"

// Full document trace (BM25 + vector + fusion)
const docTrace = debugger_.traceDocument(2.5, 3, 0.8, 0.85);
console.log(debugger_.formatSummary(docTrace));

// Compare two documents to explain rank differences
const docA = debugger_.traceDocument(3.0, 5, 0.9, 0.80);
const docB = debugger_.traceDocument(1.5, 2, 1.2, 0.95);
const comparison = debugger_.compare(docA, docB);
console.log(debugger_.formatComparison(comparison));
```

### WAND Pruning with Bayesian Upper Bounds

```typescript
import { BayesianProbabilityTransform } from "bayesian-bm25";

const transform = new BayesianProbabilityTransform(1.5, 2.0, 0.01);

// Standard BM25 upper bound per query term
const bm25UpperBound = 5.0;

// Bayesian upper bound for safe pruning -- any document's actual
// probability is guaranteed to be at most this value
const bayesianBound = transform.wandUpperBound(bm25UpperBound);
```

### Evaluating Calibration Quality

```typescript
import {
  expectedCalibrationError,
  brierScore,
  reliabilityDiagram,
} from "bayesian-bm25";

const probabilities = [0.9, 0.8, 0.3, 0.1, 0.7, 0.2];
const labels = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0];

const ece = expectedCalibrationError(probabilities, labels); // lower is better
const bs = brierScore(probabilities, labels);                // lower is better
const bins = reliabilityDiagram(probabilities, labels, 5);   // [avgPred, avgActual, count]
```

### Online Learning from User Feedback

```typescript
import { BayesianProbabilityTransform } from "bayesian-bm25";

const transform = new BayesianProbabilityTransform(1.0, 0.0);

// Batch warmup on historical data
transform.fit(historicalScores, historicalLabels);

// Online refinement from live feedback
for (const { score, label } of feedbackStream) {
  transform.update(score, label, { learningRate: 0.01, momentum: 0.95 });
}

// Use Polyak-averaged parameters for stable inference
const alpha = transform.averagedAlpha;
const beta = transform.averagedBeta;
```

### Training Modes

```typescript
import { BayesianProbabilityTransform } from "bayesian-bm25";

const transform = new BayesianProbabilityTransform(1.0, 0.0);

// C1 (balanced, default): train on sigmoid likelihood
transform.fit(scores, labels, { mode: "balanced" });

// C2 (prior-aware): train on full Bayesian posterior
transform.fit(scores, labels, { mode: "prior_aware", tfs, docLenRatios });

// C3 (prior-free): train on likelihood, inference uses prior=0.5
transform.fit(scores, labels, { mode: "prior_free" });
```

## API

### BayesianProbabilityTransform

Core class for converting BM25 scores to calibrated probabilities.

```typescript
new BayesianProbabilityTransform(alpha?, beta?, baseRate?)
```

| Method | Description |
|---|---|
| `likelihood(score)` | Sigmoid likelihood (Eq. 20) |
| `scoreToProbability(score, tf, docLenRatio)` | Full pipeline: BM25 score to calibrated probability |
| `wandUpperBound(bm25UpperBound)` | Bayesian WAND upper bound for safe pruning (Theorem 6.1.2) |
| `fit(scores, labels, options?)` | Batch gradient descent with training mode support |
| `update(score, label, options?)` | Online SGD with EMA gradients and Polyak averaging |

Static methods: `tfPrior(tf)`, `normPrior(docLenRatio)`, `compositePrior(tf, docLenRatio)`, `posterior(likelihood, prior, baseRate?)`

All methods accept both scalar (`number`) and array (`number[]`) inputs.

**FitOptions**: `learningRate`, `maxIterations`, `tolerance`, `mode` (`"balanced"` | `"prior_aware"` | `"prior_free"`), `tfs`, `docLenRatios`

**UpdateOptions**: `learningRate`, `momentum`, `decayTau`, `maxGradNorm`, `avgDecay`, `mode`, `tf`, `docLenRatio`

### BayesianBM25Scorer

Integrated BM25 search with Bayesian probability output.

```typescript
new BayesianBM25Scorer({ k1?, b?, method?, alpha?, beta?, baseRate? })
```

| Method | Description |
|---|---|
| `index(corpusTokens)` | Build BM25 index and auto-estimate parameters |
| `retrieve(queryTokens, k?)` | Top-k retrieval with calibrated probabilities |
| `getProbabilities(queryTokens)` | Dense probability array for all documents |

Properties: `numDocs`, `docLengths`, `avgdl`, `baseRate`

The `baseRate` option accepts `null` (default, no correction), `"auto"` (estimated from corpus), or a `number` in (0, 1).

### Fusion Functions

| Function | Description |
|---|---|
| `cosineToProbability(score)` | Convert cosine similarity [-1, 1] to probability (Definition 7.1.2) |
| `probNot(prob)` | Probabilistic NOT via complement rule: `1 - P(R)` (Eq. 35) |
| `probAnd(probs)` | Probabilistic AND via product rule (Eq. 33-34) |
| `probOr(probs)` | Probabilistic OR via complement rule (Eq. 36-37) |
| `logOddsConjunction(probs, alpha?, weights?)` | Log-odds conjunction with optional per-signal weights (Theorem 8.3) |
| `balancedLogOddsFusion(sparse, dense, weight?)` | Min-max normalized logit fusion for hybrid sparse-dense retrieval |

`probNot` accepts scalar (`number`) or array (`number[]`) inputs. Other fusion functions accept 1D (`number[]`) or batched 2D (`number[][]`) inputs.

### LearnableLogOddsWeights

Learns per-signal reliability weights via Hebbian gradient descent (Remark 5.3.2).

```typescript
new LearnableLogOddsWeights(nSignals, alpha?)
```

| Method | Description |
|---|---|
| `combine(probs)` | Fuse signals using current weights via `logOddsConjunction()` |
| `fit(signalsBatch, labels, options?)` | Batch gradient descent on BCE loss |
| `update(signals, label, options?)` | Online SGD with EMA gradients and Polyak averaging |

Properties: `weights`, `averagedWeights`, `nSignals`, `alpha`

**FitOptions**: `learningRate`, `maxIterations`, `tolerance`, `useAveraged`

**UpdateOptions**: `learningRate`, `momentum`, `decayTau`, `maxGradNorm`, `avgDecay`

### FusionDebugger

Traces intermediate values through the probability pipeline for debugging and explanation.

```typescript
new FusionDebugger(transform)
```

| Method | Description |
|---|---|
| `traceBM25(score, tf, docLenRatio)` | Trace BM25 score through likelihood, prior, posterior |
| `traceVector(cosineSimilarity)` | Trace cosine similarity through probability conversion |
| `traceNot(signalTrace)` | Trace probabilistic negation of a signal |
| `traceFusion(signals, method?, weights?, alpha?)` | Trace signal combination with method-specific intermediates |
| `traceDocument(score, tf, dlr, cos, method?, weights?, alpha?)` | Full pipeline trace: BM25 + vector + fusion |
| `compare(traceA, traceB)` | Compare two document traces to explain rank differences |
| `formatTrace(trace)` | Human-readable trace output |
| `formatSummary(docTrace)` | One-line pipeline summary |
| `formatComparison(comparison)` | Side-by-side document comparison |

Fusion methods: `"log_odds"` (default), `"prob_and"`, `"prob_or"`, `"prob_not"`

### Calibration Metrics

| Function | Description |
|---|---|
| `expectedCalibrationError(probabilities, labels, nBins?)` | Expected Calibration Error -- measures predicted vs actual relevance rates |
| `brierScore(probabilities, labels)` | Brier score -- mean squared error between probabilities and labels |
| `reliabilityDiagram(probabilities, labels, nBins?)` | Reliability diagram data: `[avgPredicted, avgActual, count]` per bin |

### BM25

Low-level BM25 search engine supporting Robertson, Lucene, and ATIRE IDF variants.

| Method | Description |
|---|---|
| `index(corpusTokens)` | Build inverted index |
| `getScores(queryTokens)` | Score all documents for a query |
| `retrieve(queryTokensBatch, k)` | Top-k retrieval for multiple queries |

## Citation

If you use this work, please cite the following papers:

```bibtex
@preprint{Jeong2026BayesianBM25,
  author    = {Jeong, Jaepil},
  title     = {Bayesian {BM25}: {A} Probabilistic Framework for Hybrid Text
               and Vector Search},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18414940},
  url       = {https://doi.org/10.5281/zenodo.18414940}
}

@preprint{Jeong2026BayesianNeural,
  author    = {Jeong, Jaepil},
  title     = {From {Bayesian} Inference to Neural Computation: The Analytical
               Emergence of Neural Network Structure from Probabilistic
               Relevance Estimation},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18512411},
  url       = {https://doi.org/10.5281/zenodo.18512411}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

Copyright (c) 2023-2026 Cognica, Inc.
