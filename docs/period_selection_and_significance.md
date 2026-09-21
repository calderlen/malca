# Production period selection and optional significance

`malca/stv/filter.py::validate_periodicity` now completes observed-data searches,
arbitrates the candidate periods, and saves those selections before starting
optional null simulations. The default bootstrap budget is zero in this function,
`apply_filters`, the filter CLI, the STV pipeline configuration, and
`scripts/compute_stv_local_features.py`.

## Selection

Fresh searches use PDM/Plavchan, conditional entropy, short-period Lomb–Scargle,
and the configured long-period LS/event searches. They run with zero resamples.
The selector compares the PDM, CE, and short-LS best periods, the available
long-LS peaks, and the event-timing proposal with the existing folded-curve
objective. It retains the existing harmonic alternatives and improvement rules.

For each proposal, `_correct_native_period` chooses a harmonic using those
rules. `_select_period_by_fold_fit` then chooses the proposal's corrected
period with the lowest `selection_objective`. This objective uses residual
scatter about a binned phase template, disagreement between band phases, alias
and event-fold penalties, and a harmonic-change penalty. It is a heuristic
fit-quality score, not a probability or a cross-validation error.

Method-specific bootstrap probabilities and PDM/CE support thresholds do not
enter this final selection. The result is written to `periodicity_period`,
`period_consensus_days`, and `period_for_fold_days`; native and harmonic-corrected
periods remain available separately. `periodicity_selection_version` identifies
this selector as `fold_fit_before_significance_v1`.

This production path is separate from the evaluation-only ensemble diagnostic
ranker described in the period-methods manuscript. Changing the selection rule
can change adopted periods. The earlier catalog-agreement percentages have not
been revalidated for this selector.

## Significance and rejection

- `--periodicity-n-bootstrap 0` selects periods without new null simulations.
  Fresh results have `periodicity_significance_status=not_requested`, missing
  unified FAP, and false significance/rejection flags. A false flag here means
  no detection was established; it does not demonstrate nonperiodicity.
- A positive budget, such as `--periodicity-n-bootstrap 1000`, starts a separate
  significance pass after all selections have been saved. It uses the existing
  observing-block permutations and repeats each method's full frequency search.
  Long-period LS receives the requested budget too.
- The source-level statistic is `min(1, M * min(p_method))`, where `M` is four
  searched method/range tests (three with long-period LS disabled). Each
  `p_method` already compares the observed extremum against full-search null
  extrema. The extra factor accounts for choosing among methods; it does not
  require independence between them. Alias-excluded tests contribute 1.
  Missing tests retain their place in `M` and produce a `partial` status.
- `periodicity_significance_scope=source_searches_bonferroni` identifies this
  source-level test. It does not validate the adopted fundamental period, its
  harmonic choice, or the folded-curve score. Its calibration still depends on
  the observing-block null model. Period confidence remains provisional.
- The significance pass preserves all selected-period fields. Error states are
  explicit and clear stale detection flags. `--periodicity-reject` removes
  established detections; obtaining fresh detections requires a positive budget.

A positive budget still requests expensive tests for every eligible input
source. For significance on a chosen subset, pass that subset to
`validate_periodicity` with the same checkpoint directory and a positive budget.
No automatic uncertainty-based trigger is currently applied.

## Resume and existing results

New results use `period_selection_checkpoint.parquet`; the historical
`lsp_checkpoint.parquet` remains untouched. Checkpoints preserve complete
period fields and stage metadata and are written atomically. All selections
are saved before optional significance begins; significance is saved every
10 completions and on normal exit or interruption.

Changing only the bootstrap budget reuses compatible selections. Repeating
the same completed budget reuses significance too. The light-curve fingerprint,
selection version, and other existing settings checks still apply. The null
test routines recompute their observed search as part of testing, but their
period choice never replaces the checkpointed selection.

Explicit historical `periodicity_reuse_from` and catalog-consensus shortcuts
retain their previous provenance and behavior. Their periods are not presented
as new folded-curve selections. Already-running jobs keep their original
seven-element worker contract and bootstrap settings; the new stage ordering
applies when a new validator invocation starts.

## Bounded verification (2026-09-11)

The three-source smoke in `tmp/period_selection_smoke_20260911.py` measured
0.60–0.89 seconds per source for selection. It also exercised multiprocessing,
checkpoint reload, and optional significance with unchanged selected periods.
Timing details are in
`output/diagnostics/period_selection_stages_20260911/smoke.json`.
Three null resamples were used only to check execution and persistence; this
is neither a detection-significance measurement nor a period-recovery benchmark.
