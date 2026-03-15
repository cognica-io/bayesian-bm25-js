//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

export {
  BayesianProbabilityTransform,
  TemporalBayesianTransform,
  sigmoid,
  logit,
  clampProbability,
  EPSILON,
  type FitOptions,
  type UpdateOptions,
  type TrainingMode,
  type TemporalFitOptions,
} from "./probability.js";

export {
  cosineToProbability,
  probNot,
  probAnd,
  probOr,
  logOddsConjunction,
  balancedLogOddsFusion,
  LearnableLogOddsWeights,
  AttentionLogOddsWeights,
  MultiHeadAttentionLogOddsWeights,
  resolveAlpha,
} from "./fusion.js";

export {
  PlattCalibrator,
  IsotonicCalibrator,
} from "./calibration.js";

export {
  expectedCalibrationError,
  brierScore,
  reliabilityDiagram,
  CalibrationReport,
  calibrationReport,
} from "./metrics.js";

export {
  BayesianBM25Scorer,
  BlockMaxIndex,
  type BayesianBM25ScorerOptions,
  type RetrieveResult,
  type RetrievalResult,
} from "./scorer.js";

export { BM25, type BM25Options, type BM25Method } from "./bm25.js";

export {
  FusionDebugger,
  type BM25SignalTrace,
  type VectorSignalTrace,
  type NotTrace,
  type FusionTrace,
  type DocumentTrace,
  type ComparisonResult,
} from "./debug.js";

export {
  MultiFieldScorer,
  type MultiFieldScorerOptions,
} from "./multi_field.js";
