# History

## 0.4.0 (2026-03-01)

- Add `probNot()` for probabilistic negation (complement rule, Eq. 35)
  - Computes `P(NOT R) = 1 - P(R)` with epsilon clamping for numerical stability
  - In log-odds space, NOT corresponds to simple negation: `logit(1 - p) = -logit(p)`
  - Composes naturally with `probAnd()`, `probOr()`, and `logOddsConjunction()`
    for exclusion queries (e.g., "python AND NOT java")
  - Satisfies De Morgan's laws: `NOT(A AND B) = OR(NOT A, NOT B)` and
    `NOT(A OR B) = AND(NOT A, NOT B)`
  - Accepts both scalar (`number`) and array (`number[]`) inputs
- Add `balancedLogOddsFusion()` for hybrid sparse-dense retrieval
  - Converts both Bayesian BM25 probabilities and dense cosine similarities
    to logit space, min-max normalizes each to equalize voting power, and
    combines with configurable weights
  - Prevents heavy-tailed sparse logits (from sigmoid unwrapping) from
    drowning the dense signal while preserving the Bayesian BM25 framework's
    document-length and term-frequency priors
  - Accepts `weight` parameter for asymmetric signal weighting (default 0.5)
- Add `LearnableLogOddsWeights` for per-signal reliability learning (Remark 5.3.2)
  - Learns weights from Naive Bayes uniform initialization (w_i = 1/n) to
    per-signal reliability weights via softmax parameterization over
    unconstrained logits
  - Hebbian gradient: `dL/dz_j = n^alpha * (p - y) * w_j * (x_j - x_bar_w)`
    (pre-synaptic activity x post-synaptic error, backprop-free)
  - Batch `fit()` via gradient descent on BCE loss
  - Online `update()` via SGD with EMA-smoothed gradients, bias correction,
    L2 gradient clipping, learning rate decay, and Polyak averaging of weights
    in the simplex
  - Alpha (confidence scaling) is fixed, only weights are learned; the two are
    orthogonal (Paper 2, Section 4.2)
- Add `FusionDebugger` for transparent pipeline inspection
  - Records every intermediate value through the full probability pipeline
    (likelihood, prior, posterior, fusion) so you can trace why a document
    received a particular fused score
  - `traceBM25()`: trace a single BM25 score through sigmoid likelihood,
    composite prior, and Bayesian posterior, capturing logit-space intermediates
  - `traceVector()`: trace cosine similarity through probability conversion
  - `traceFusion()`: trace the combination of multiple probability signals
    with method-specific intermediates for `log_odds`, `prob_and`, `prob_or`,
    and `prob_not`
  - `traceNot()`: trace probabilistic negation (complement) of a single signal
  - `traceDocument()`: full pipeline trace composing BM25 + vector + fusion
    into a single `DocumentTrace` with all intermediate values
  - `compare()`: compare two `DocumentTrace` objects to explain rank
    differences, identifying the dominant signal and crossover stages where
    signals disagree
  - `formatTrace()`, `formatSummary()`, `formatComparison()`: human-readable
    output for traces, one-line summaries, and side-by-side comparisons
  - Support all four fusion methods as `method` parameter: `"log_odds"`,
    `"prob_and"`, `"prob_or"`, `"prob_not"`
  - Support hierarchical (nested) fusion and weighted log-odds fusion

## 0.3.2 (2026-02-25)

- Optimize posterior computation using two-step Bayes update (Remark 4.4.5)
  - Replaces `sigmoid(logit(L) + logit(br) + logit(p))` with two sequential
    Bayes updates using only multiplication and division
  - `scoreToProbability()` delegates to `posterior()` instead of duplicating
    baseRate logic
- Optimize scorer internals for faster retrieval
  - `_scoresToProbabilities()` processes all k documents per query in one
    batch call instead of a scalar-by-scalar inner loop
  - Add `_computeTFBatch()` for batch term frequency computation
  - Deduplicate pseudo-query sampling: `_samplePseudoQueryScores()` is
    called once during indexing instead of separately by `_estimateParameters()`
    and `_estimateBaseRate()`

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
