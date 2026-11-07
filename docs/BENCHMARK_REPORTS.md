# Benchmark reports

Use the existing `seed_vision.py` entry point for the grouped method comparison. Ordinary desktop analysis keeps its normal cache ownership and execution path.

```powershell
.venv\Scripts\python.exe seed_vision.py --benchmark-comparison comparison.json --benchmark-output comparison-report.json
```

Comparison inputs contain exactly `manifest`, `plan`, and `records`. The typed manifest and frozen plan bind samples, truth revisions, methods, endpoints, directions, absolute limits, noninferiority margins, aggregation, minimum coverage/groups, seed, resampling count and critical strata. See `seedvision/benchmarking/grouped_comparison.py` for the versioned JSON schema and `tests/test_grouped_comparison.py` for complete compact examples. Plans whose methods bind a trained classifier are refused, because no adapter ties a model to its training data.

Resampling uses connected biological, acquisition and duplicated-source groups across the full manifest. Comparative bounds use jointly available pairs with the same group draws for both methods. Absolute candidate bounds include every available candidate, including images with missing reference output. Failed, abstained, unavailable and missing observations remain visible and in coverage denominators; they receive no fabricated score. Both cohorts must meet their prespecified coverage/group gates. Every critical stratum must pass; unknown membership or insufficient evidence makes the result inconclusive. An ineligible leakage audit cannot pass the gates.

These are engineering tools for supplied study declarations and observations. Independent truth, study conduct and scientific acceptance require the planned laboratory studies.
