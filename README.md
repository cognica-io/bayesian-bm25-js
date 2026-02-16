# Bayesian BM25 for JavaScript/TypeScript

A probabilistic framework that converts raw BM25 retrieval scores into calibrated relevance probabilities using Bayesian inference. TypeScript port of [bayesian-bm25](https://github.com/cognica-io/bayesian-bm25).

## Overview

Standard BM25 produces unbounded scores that lack consistent meaning across queries, making threshold-based filtering and multi-signal fusion unreliable. Bayesian BM25 addresses this by applying a sigmoid likelihood model with a composite prior (term frequency + document length normalization) and computing Bayesian posteriors that output well-calibrated probabilities in [0, 1].

Key capabilities:

- **Score-to-probability transform** -- convert raw BM25 scores into calibrated relevance probabilities via sigmoid likelihood + composite prior + Bayesian posterior
- **Parameter learning** -- batch gradient descent or online SGD with EMA-smoothed gradients and Polyak averaging
- **Probabilistic fusion** -- combine multiple probability signals using log-odds conjunction, which resolves the shrinkage problem of naive probabilistic AND
- **Search integration** -- built-in BM25 scorer that returns probabilities instead of raw scores, with support for Robertson, Lucene, and ATIRE variants

## Installation

```bash
npm install bayesian-bm25
```

## Quick Start

### Converting BM25 Scores to Probabilities

```typescript
import { BayesianProbabilityTransform } from "bayesian-bm25";

const transform = new BayesianProbabilityTransform(1.5, 1.0);

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

const scorer = new BayesianBM25Scorer({ k1: 1.2, b: 0.75, method: "lucene" });
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

## API

### BayesianProbabilityTransform

Core class for converting BM25 scores to calibrated probabilities.

| Method | Description |
|---|---|
| `likelihood(score)` | Sigmoid likelihood (Eq. 20) |
| `scoreToProbability(score, tf, docLenRatio)` | Full pipeline: BM25 score to calibrated probability |
| `fit(scores, labels, options?)` | Batch gradient descent on binary cross-entropy |
| `update(score, label, options?)` | Online SGD with EMA gradients and Polyak averaging |

Static methods: `tfPrior(tf)`, `normPrior(docLenRatio)`, `compositePrior(tf, docLenRatio)`, `posterior(likelihood, prior)`

All methods accept both scalar (`number`) and array (`number[]`) inputs.

### BayesianBM25Scorer

Integrated BM25 search with Bayesian probability output.

| Method | Description |
|---|---|
| `index(corpusTokens)` | Build BM25 index and auto-estimate parameters |
| `retrieve(queryTokens, k?)` | Top-k retrieval with calibrated probabilities |
| `getProbabilities(queryTokens)` | Dense probability array for all documents |

### Fusion Functions

| Function | Description |
|---|---|
| `probAnd(probs)` | Probabilistic AND via product rule (Eq. 33-34) |
| `probOr(probs)` | Probabilistic OR via complement rule (Eq. 36-37) |
| `logOddsConjunction(probs, alpha?)` | Log-odds conjunction with agreement bonus |

Fusion functions accept 1D (`number[]`) or batched 2D (`number[][]`) inputs.

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
