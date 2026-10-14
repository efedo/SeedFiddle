# Benchmark reports

Use the existing `seed_vision.py` entry point for explicit benchmark actions. Ordinary desktop analysis keeps its normal cache ownership and execution path.

```powershell
.venv\Scripts\python.exe seed_vision.py --benchmark-resources workload.json --benchmark-output resources.json
.venv\Scripts\python.exe seed_vision.py --benchmark-comparison comparison.json --benchmark-output comparison-report.json
```

The resource workload selects an existing serialized native working recipe or paper port and a source image. It can specify `repeats`, `warmups`, `modes` (`cold` and/or `warm`), `device`, `maximum_elapsed_seconds`, `maximum_peak_bytes`, `model_asset_paths`, and `correction_records`. Paths resolve beside the configuration file. Warm mode requires an actual warmup. Original-reference recipes are rejected by the native resource action; use the separate public-reference capture route for original PlantCV execution.

Reports retain actual hardware and software versions, synchronized CUDA event and wall durations, per-node timing/cache outcomes, allocator baseline and measured peaks, and failures or cancellation. Dispatcher transfer counts describe logical tensor payloads and explicit scalar reads; physical bus traffic and hidden external transfers remain unknown. Instrumentation overhead is included. Time/memory thresholds stop following runs after measurement; they are not hard kernel or allocation limits. Reports do not claim scientific quality.

Operator correction time is available through `seedvision.benchmarking.performance.measure_operator_correction`. The explicit start/stop context records completed or interrupted work against a supplied operator, acceptance standard, source SHA and recipe SHA. Pass these records to the workload if available. An absent observation remains unavailable; the report does not authenticate the supplied acceptance standard.

Comparison inputs contain exactly `manifest`, `plan`, and `records`. The typed manifest and frozen plan bind samples, truth revisions, methods, recipe/model/training identities, endpoints, directions, absolute limits, noninferiority margins, aggregation, minimum coverage/groups, seed, resampling count and critical strata. See `seedvision/benchmarking/grouped_comparison.py` for the versioned JSON schema and `tests/test_grouped_comparison.py` for complete compact examples.

Resampling uses connected biological, acquisition and duplicated-source groups across the full manifest. Comparative bounds use jointly available pairs with the same group draws for both methods. Absolute candidate bounds include every available candidate, including images with missing reference output. Failed, abstained, unavailable and missing observations remain visible and in coverage denominators; they receive no fabricated score. Both cohorts must meet their prespecified coverage/group gates. Every critical stratum must pass; unknown membership or insufficient evidence makes the result inconclusive. An ineligible leakage audit cannot pass the gates.

For classifier-bearing methods, also supply `--classifier-evidence evidence.json`. The configuration contains `bindings`, each selecting `method_id`, `family`, `model_path`, and an optional `training_path`. Typed model and training artifacts must match the already frozen plan; the action never rewrites it. Known training images must map to declared training access. Unavailable training-image membership remains explicitly unverified and cannot establish eligible classifier evaluation.

These are engineering tools for supplied study declarations and observations. Independent truth, study conduct and scientific acceptance require the planned laboratory studies.
