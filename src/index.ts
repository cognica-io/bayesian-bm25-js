//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

export {
  BayesianProbabilityTransform,
  sigmoid,
  logit,
  clampProbability,
  EPSILON,
  type FitOptions,
  type UpdateOptions,
  type TrainingMode,
} from "./probability.js";

export {
  cosineToProbability,
  probAnd,
  probOr,
  logOddsConjunction,
} from "./fusion.js";

export {
  expectedCalibrationError,
  brierScore,
  reliabilityDiagram,
} from "./metrics.js";

export {
  BayesianBM25Scorer,
  type BayesianBM25ScorerOptions,
  type RetrieveResult,
} from "./scorer.js";

export { BM25, type BM25Options, type BM25Method } from "./bm25.js";
