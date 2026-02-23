# History

## 0.3.0 (2026-02-23)

- Add calibration metrics module
  - `expectedCalibrationError()`: measures how well predicted probabilities
    match actual relevance rates (lower is better)
  - `brierScore()`: mean squared error between probabilities and labels
  - `reliabilityDiagram()`: returns `[avgPredicted, avgActual, count]` per bin
    for visual calibration assessment
- Support alpha + weights composability in `logOddsConjunction()`
  - Per-signal weights (Theorem 8.3) and confidence scaling by signal count
    (Section 4.2) are orthogonal and compose multiplicatively:
    `sigma(n^alpha * sum(w_i * logit(P_i)))`
  - Change `alpha` default from `0.5` to `undefined` for context-dependent
    resolution: `undefined` resolves to `0.5` in unweighted mode and `0.0`
    in weighted mode
  - Explicit `alpha` applies in both unweighted and weighted modes
- Add comprehensive theorem verification tests (63 tests)
  - Paper 1 (Bayesian BM25): sigmoid properties, logit-sigmoid duality,
    posterior formula equivalence, monotonicity, prior bounds, base rate
    log-odds, Section 11.1 numerical values
  - Paper 2 (From Bayesian Inference to Neural Computation): scale neutrality,
    sign preservation, irrelevance non-inversion, Section 4.5 numerical table,
    disagreement moderation, logistic regression equivalence, agreement
    amplification, conjunction vs product rule, strict bounds, Log-OP
    equivalence, heterogeneous signal combination, single signal identity,
    weighted alpha composition, monotone shrinkage, information loss, sqrt(n)
    scaling law, spread property, geometric mean residual, sigmoid uniqueness,
    output range

## 0.2.0 (2026-02-21)

- Add corpus-level base rate prior for unsupervised probability calibration
  - `BayesianProbabilityTransform` accepts `baseRate` constructor parameter
  - `BayesianBM25Scorer` accepts `baseRate` option (`null`, `"auto"`, or number)
  - Three-term log-odds posterior: `sigmoid(logit(L) + logit(baseRate) + logit(prior))`
  - Auto-estimation via 95th percentile pseudo-query heuristic
- Add cosine similarity to probability conversion for hybrid text + vector search
  - `cosineToProbability()`: maps cosine similarity [-1, 1] to probability (0, 1) via (1 + score) / 2 with epsilon clamping (Definition 7.1.2)
- Add weighted log-odds conjunction for per-signal reliability weighting
  - `logOddsConjunction()` accepts optional `weights` parameter
  - Uses Log-OP formulation: sigma(sum(w_i * logit(P_i))) (Theorem 8.3, Remark 8.4)
  - Weights must be non-negative and sum to 1; unweighted behavior unchanged
- Add WAND upper bound for safe Bayesian document pruning
  - `BayesianProbabilityTransform.wandUpperBound()`: computes tightest safe probability upper bound using pMax=0.9 (Theorem 6.1.2, Theorem 4.2.4)
  - Supports baseRate-aware bounds for tighter pruning
- Add prior-aware training modes (C1/C2/C3 conditions from Algorithm 8.3.1)
  - `fit()` and `update()` accept `mode` option: `"balanced"` (C1, default), `"prior_aware"` (C2), `"prior_free"` (C3)
  - Prior-aware (C2): trains on full Bayesian posterior with chain-rule gradients through dP/dL
  - Prior-free (C3): trains on likelihood, inference uses prior=0.5

## 0.1.2 (2026-02-20)

- Add npm keywords and bump version

## 0.1.1 (2026-02-19)

- Fix `logOddsConjunction` to use multiplicative log-odds mean from Paper 2

## 0.1.0 (2026-02-18)

- Initial release
- Sigmoid likelihood + composite prior (term frequency + document length) + Bayesian posterior
- Batch gradient descent and online SGD with EMA-smoothed gradients and Polyak averaging
- Probabilistic score combination: `probAnd`, `probOr`, `logOddsConjunction`
- Built-in BM25 engine with Robertson, Lucene, and ATIRE IDF variants
- `BayesianBM25Scorer` with auto-estimated sigmoid parameters
