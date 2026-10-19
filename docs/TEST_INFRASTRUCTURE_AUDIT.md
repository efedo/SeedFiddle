# Test infrastructure audit — 15 October 2026

## Current full-suite result — 18 October 2026

The fresh canonical full run passes **639 functional units: 635 passes, four existing skips,
zero failures/errors**, in **762.295 seconds** (762.822 parent seconds). Both the count and
900-second wall-time budgets pass. This is one full invocation with three isolated processes;
all 639 discovered IDs are assigned exactly once, reconciled with unit journals and outcomes,
and included in the result. Worker sizes are 151 desktop, 193 numerical/reference and 295
workflow units. Missing, crashed, incomplete or failed workers cannot produce a passing result.
The controller stops its own process trees when the deadline expires.

The starting inventory contains all 2,823 methods in 248 files. The current inventory has
639 units in 249 files (some files also provide imported fixtures). Every original method has
an explicit disposition. 757 overlapping integrations are retired; related scenarios are
combined. 2,063 original scenario bodies are unchanged; the numerical matrix and full-image
replay preparation are deliberate rewrites, and one runner method adds worker-failure checks. This is source-derived evidence, not a
measured branch/statement coverage percentage.

All 223 recipe specifications still pass real admission and exact frozen specification
comparison. Actual numerical execution now uses 19 operation/port/input-family representatives,
with complete frozen output comparisons, warm-cache checks, verified exports and source-absent
replay. The independent primitive numerical matrices, public eight-profile comparison and
reader comparisons remain. The former 223-way numerical integration cross-product is reduced;
equivalence for every former integration permutation is not claimed.

Immutable catalogue reuse is explicitly enabled for 132 classes. Calculation admission still
runs for selected recipes. Test-only dependency-import reuse executes actual public calls,
reloads PlantCV and verifies requests/source/arrays each time. The actual launcher/transport
witnesses in `test_reference_runner` and `test_reference_watershed_capture` keep one-shot
processes. Immutable compiled-code reuse retains source reads, loaded binding/default checks
and mutation rejection. Scenario setup, teardown, cleanups and mutable state remain private.
Full-image threshold replay prepares only its required dependency prefix instead of running
an unrelated full recovery calculation; its numerical, CUDA and host-memory checks remain.

All 581 pinned inputs, including images and the four frozen baselines, stayed byte-identical
throughout the full run. Subsequent removal of surplus EOF blank lines in 26 test files was
proved to preserve ASTs, locations, recursive executable code and exception/line tables.
Only coverage-hash metadata changed in the feature register. The final metadata gate passes
10 units in 2.481 seconds. The reviewed-source receipt records these formatting-only changes;
the original full-run receipt remains immutable.

```powershell
.venv\Scripts\python.exe -m tests --full -v --report artifacts/my-full-run.json
.venv\Scripts\python.exe -m tests --tier cpu-qt
.venv\Scripts\python.exe tools/audit_test_inventory.py --collect --receipt artifacts/test-functional-current-review-v1.json
```

`--full` defaults to three isolated test processes. `--jobs 1` is a serial diagnostic option;
it retains the same discovered units but is not the runtime benchmark. Ordinary direct
unittest discovery also selects the same units, without the canonical runner's process,
reference-import and compiler-fixture management. CI already invokes the canonical runner.
The four existing skips concern the absent eleven-image pilot batch, the opt-in fill image
diagnostic, case-sensitive paths and Windows directory symlinks. No new skip was added.

Evidence: `artifacts/test-functional-full-v5.json`, its immutable completion receipt
`artifacts/test-functional-completed-v1.json`, and `artifacts/test-functional-current-review-v1.json`.
See [redesign](testing/TEST_SUITE_REDESIGN.md), [current inventory](testing/TEST_INVENTORY.md),
[original inventory](testing/consolidation-2026-10-18/before-TEST_INVENTORY.md) and
[every original disposition](testing/consolidation-2026-10-18/TEST_DISPOSITIONS.md).
Earlier failed, interrupted and over-budget diagnostics remain explicitly nonpassing. The
serial V4 was stopped after 941 recorded seconds and 323 completed units, with no failures;
it is not a full result. Scientific acceptance and original SMPTS status are unchanged.

## Performance follow-up, 18 October 2026

Complete suite coverage is verified across source-checked segments: **2,823 methods, 2,819 passes, four existing skips, zero final failures/errors**. Recorded method and class-fixture durations total **7,587.873 seconds (126.5 minutes)**, versus 10,100.247 seconds (168.3 minutes) in the preceding run: **24.9% less recorded time**. The interrupted process is not a passing full run. This compares recorded work across different process segmentation; no uninterrupted full-suite or parent wall-time speedup is claimed, and machine load can vary.

The largest avoidable CPU cost was repeated catalogue construction: profiling one reader composition test found 1,030 builds of the same five-template catalogue. `tests/catalogue_fixtures.py` explicitly opts 37 test classes into sharing those immutable templates. Source/configuration bytes and factory callable identities key reuse; serialized integrity is checked on entry/exit and imported aliases are restored, including lazy imports. Each selected recipe still passes actual admission. Catalogue/source-mutation witnesses retain the original path. Production validation is unchanged.

`tests/observation.py` replaces whole-interpreter profiling at replay/decoder/dispatch boundaries with local Python 3.12 monitoring events. Actual functions and source identities remain intact. The fallback restores/chains the previous profiler. C-call allocation guards and the profiler-compatibility tests remain.

Four repeated source-absent replay methods are merged into the complete card, recovery, paper and classifier product comparisons: those comparisons now run with their origins hidden. Exact pixels, ports, dispatch/input selection, frames and archive fingerprints are still checked. A native replay fixture is computed/exported once per class, with private archive copies for every test and source-fixture integrity checks at teardown. This optimization shares no writable workspace, window, settings or analysis cache; the replay template is copied before tests can corrupt files.

The before/after inventory retains 2,821 test IDs and all 15,723 direct check expressions in those methods unchanged. Four redundant methods are retired and two fixture/observer regressions added: 2,823 methods in 248 files, with exact import-only reconciliation. This is static assertion evidence, not measured statement/branch coverage. All 223 actual cold/warm profiles, complete frozen JSON comparisons, source-absent replay representatives, public/reader matrices, cancellation/mutation guards and full-resolution memory checks remain. Application source and all four numerical baselines are unchanged.

The long run found a repeatable crop-export test synchronization error: its event loop stopped as soon as the window hid, before the queued worker-completion handler cleared the task pointer. The test now waits for the hidden window, idle pool and cleared pointer; all existing cancellation, completion and no-modal assertions remain. The isolated repair passes, and the entire crop module was rerun with all remaining modules in a fresh canonical process: 288/288 pass in 109.936 test seconds. The other 2,535 outcomes come only from complete modules recorded before the failure. Exact discovered IDs, source hashes and journals reconcile all 2,823 methods. The sole source transition between segments is this Qt test wait condition. No failed or incomplete crop result is reused.

The existing missing pilot batch, opt-in fill diagnostic, case-sensitive-filesystem and Windows directory-symlink skips remain explicit. No new skip or filtering was added. The full command and CI discovery are unchanged.

| Check | Previous full run | Fresh recorded duration |
| --- | ---: | ---: |
| Reader: all 24 composition orders | 120.650 s | 1.400 s |
| Reader: all 24 IDs and versions | 118.763 s | 1.130 s |
| Fill: all 28 contracts / composition | 94.663 s | 2.661 s |
| All 223 cold/warm numerical profiles | 1,443.656 s | 823.760 s |

Evidence: `artifacts/test-suite-speed-completed-segments-v1.json`, its reconciled timing report, `artifacts/test-suite-speed-comparison-v1.json` and `artifacts/test-suite-speed-coverage-v2.json`. The focused helper gate passes nine methods; the final contract/fixture gate passes 27 methods in 74.170 seconds; the register gate passes six methods. A three-method smoke check verifies the incremental driver. The interrupted full V1 has no final result. V2 records the failed crop test and was stopped after stalling; it remains nonpassing. Its incremental journal retains successful complete modules, including all numerical matrices. `artifacts/test-suite-speed-remainder-v1.json` records the fresh canonical remainder. The original process-absence observation was incorrect because sandbox enumeration hid the native PID; the corrected V2 receipt preserves that observation snapshot and records verified native termination. Older checkpoints below retain their original source scope. Scientific acceptance and original SMPTS status are unchanged.

## Previous 15 October implementation follow-up

Implementation follow-up is complete for the dialog/recipe/replay consolidation,
canonical CI discovery, explicit execution tiers, immutable fixture prerequisites,
class-fixture timing and 223-profile coverage mapping. Two workers and an
independent reviewer completed the changes. The fresh full run passes 2,814
methods (2,810 passes, four existing skips, no failures/errors) in 8,902.126
seconds. The nine changed modules use 9.2% less recorded method time, but the
whole run is 4.3% longer than the previous observation; no overall speedup is
claimed. See [implementation and measured results](TESTING.md).

The original findings below describe the audited 2,811-method input snapshot.
The linked inventory now records all 2,814 current methods with fresh outcomes;
the three additions protect the runner. No original test ID, numerical baseline
or frozen profile was retired, and no application implementation changed.
Historical transition checks remain explicitly selectable and also retained in
full/acceptance. Broader public-capture fixture extraction and numerical-basis
reductions remain candidates requiring the new timing/ownership evidence.

At the original audit checkpoint, the suite had **2,811 test methods in 239 files**.
The strongest further simplification opportunities are repeated recipe composition
and replay executions, especially exhaustive combinations exercised through both
contracts and legacy dialogs. Removing inexpensive numerical checks merely to
reduce the method count would miss the largest costs.

This audit makes recommendations. It does not remove tests, change application
behavior, update frozen expectations, or claim that the current suite is minimal.

- [Complete inventory, grouped by module and class](testing/TEST_INVENTORY.md)
- [Structured inventory with all direct checks and static call evidence](testing/test_inventory.json)
- [Reproducible inventory generator](../tools/audit_test_inventory.py)
- [Existing test commands and preceding cleanup results](TESTING.md)

## Pass 1: inventory and reconciliation

Every method has a canonical unittest ID, source location, functionality scenario,
assertion evidence, call/helper evidence, fixture/decorator information, loop and
subtest sites, default-core membership, historical outcome and method duration.
Module purposes are taken from explicit source scope descriptions or a reviewed
purpose mapping for older files. Source hashes bind the inventory to the files
inspected. Shared helper checks are recorded once in JSON and referenced by ID.

Import-only discovery with the repository as the top-level directory produced
exactly the same **2,811 IDs** as the source inventory, with zero discovery errors,
missing methods, additional methods or duplicate IDs. No test method or setup
fixture was executed during that reconciliation. Normal module import checks still
occur. There are no module-level `load_tests`, `setUpModule` or `tearDownModule`
functions in the current test package.

| Inventory measure | Count |
| --- | ---: |
| Test modules | 239 |
| Classes declaring discovered test methods | 370 |
| Discovered methods | 2,811 |
| Lines in the 239 test modules, including comments and embedded reference scripts | 90,061 |
| Other test infrastructure files | 4 / 330 lines |
| Methods selected by the default core command | 132 |
| Methods containing an explicit `for` loop or comprehension | 1,531 |
| Methods containing `subTest` calls | 482 |
| Static `subTest` call sites | 542 |
| Class setup definitions in test modules | 121 |
| Per-test setup definitions in test modules | 71 |

The four support files are `tests/__init__.py`, `tests/__main__.py`,
`tests/acceptance_diff.py` and `tests/review_fixtures.py`. Their scopes and hashes
are included in the structured inventory. Eleven other top-level classes in test
modules declare no discovered test methods; they are not additional test classes.
The setup counts are definitions, not measured invocation counts.

These exclusive, module-level categories provide navigation. Mixed suites contain
several types of checks; the per-method evidence is more specific.

| Primary area | Files | Methods | Historical method seconds |
| --- | ---: | ---: | ---: |
| Recipe, replay and reference infrastructure | 88 | 1,081 | 4,593.069 |
| Numerical primitives and public-source parity | 43 | 413 | 644.456 |
| Persistence, export, identity and outcomes | 26 | 390 | 599.921 |
| Image analysis, calibration and rendering | 16 | 292 | 282.963 |
| Pipeline workspace and controls | 11 | 196 | 137.936 |
| Annotation and interactive image review | 24 | 179 | 65.520 |
| Measurements and shape models | 15 | 139 | 5.579 |
| Learning and trait analysis | 7 | 79 | 19.871 |
| Startup, resources and diagnostics | 6 | 28 | 8.543 |
| Frozen acceptance matrices | 3 | 14 | 1,670.973 |
| **Total** | **239** | **2,811** | **8,028.832** |

The matrix count illustrates why a method count is insufficient. One method runs
223 recipe profiles through cold execution, warm-cache execution, verified archive
round trips and source-absent replay. Other methods contain public pixel matrices,
24 action orders or multiple full-resolution memory probes. The inventory records
these loops without presenting their cases as additional unittest methods.

The feature register is a feature-to-test map, not the suite inventory: its 200
referenced test files do not cover all 239 discovered files. Repeated references
to the same method in that register are not repeated test executions.

### Evidence limits

Functionality descriptions are inferred from source scenarios, docstrings and
assertions. They are not a manual proof of every assertion's correctness. Static
production-call candidates resolve imports and test helpers, but do not resolve
assigned object types, dynamic callbacks or inherited behavior. Interface calls
and source links remain available where resolution is incomplete. Markdown shows
short assertion excerpts; JSON contains all direct assertion expressions.

No measured line or branch coverage was collected. Thus this inventory cannot
establish that every production path is covered, that every recorded test would
detect a regression, or that two tests with different names cover different risks.

## Pass 2: overlap and regression value

Normalized AST bodies, excluding method names and leading docstrings, contain
**no exact duplicate method bodies**. That does not establish semantic uniqueness.
The following overlap candidates come from inspecting execution and assertion
patterns, rather than matching names alone.

| Candidate | Concrete evidence | Recommended treatment | Distinct checks to retain |
| --- | --- | --- | --- |
| Factorial reader/ROI action orders in both contracts and dialogs | [Contract matrix](../tests/test_native_reader_recipe.py#L80): 24 permutations, 87.211 s. [Dialog matrix](../tests/test_native_reader_recipe.py#L261): the same 24 orders, 80.444 s; it programmatically clicks buttons that it also asserts are hidden. | Keep composition independence at the contract layer. Reduce the dialog to representative worker/Keep/reopen behavior, and exercise exposed controls through the existing Pipeline workspace tests. First map any different parameter values or layout assertions so those are transferred. | Every independent control's effect, graph identity/order independence, retained layout, worker locks, project persistence and actual Pipeline controls. |
| Earlier three-operation permutations | [Median ROI contract matrix](../tests/test_median_roi_recipe.py) takes 18.868 s; its six-order desktop matrix takes 18.157 s. Reader tests also combine median, fill and morphology, adding reader as a fourth operation. | Check whether the six older permutations add a version boundary or value combination absent from the reader contract. Consolidate only the overlap; keep older-version compatibility as a small direct witness. | Historical serialization/version behavior and parameter preservation. |
| Shared card prefix executed in nine full graph variants | [Card stage replay matrix](../tests/test_native_card_stage_replay.py#L115) takes 133.600 s. Nine full recipes are run and exported before four common prefix stages are replayed. Other methods already compare all full/ROI card-stage products. | Prove graph wiring for all nine variants with structural checks. Reuse one real prefix archive per distinct input/frame/control contract, plus representative composed executions. Audit the 223-profile acceptance matrix for overlap before dropping a numerical execution. | Card outputs, source/ROI frames, scale units, selected payloads, composition wiring and a real downstream consumer. |
| Repeated replay framework guard matrices | 226 methods across 16 replay-named modules take 1,659.226 s. Within those, 32 methods named for cancellation/callback/mutation take 227.286 s. Card and recovery suites both rebuild archive fixtures and probe dispatch/payload selection through similar `attempt`/`changed` helpers. | Centralize shared test scaffolding and test common replay admission/publication guards at their owner. Add a compact per-adapter wiring matrix. Do not assume differently shaped model, contour, compact or CUDA payloads have identical guards. | Adapter-specific payload selection, every unique buffer type, final cancellation and mutation priority, archive immutability and no-origin replay. |
| Repeated public reference capture fixtures | [Reference capture](../tests/test_reference_capture.py), [project capture](../tests/test_project_reference_captures.py) and [public attempt QC](../tests/test_public_reference_attempt_qc.py) build closely related source images/capture evidence in class setup. | Extract immutable request/result fixtures with exact environment/source/parameter keys. Keep a live public producer witness; copy files and writable results for each orchestration/failure case. | Actual public-source agreement, decode count, source binding, publication ownership and independent failure cleanup. |
| Historical implementation-transition checks in ordinary acceptance | [Recipe acceptance](../tests/test_recipe_acceptance.py#L585) embeds exact prior exporter/code/driver transitions and requires several ignored review receipts before current acceptance can run. | Separate immutable historical-evidence verification from current numerical regression execution. Keep historical records intact and explicitly runnable; avoid making every ordinary numerical check depend on reconstructing development history. | Current provenance admission, exact numerical expectations, schema compatibility and explicit historical evidence integrity. |
| Full desktop construction for small control/state assertions | Direct `MainWindow` construction remains in 19 test modules, alongside the newer private tiny-image fixtures. Some are real startup/image witnesses; others assert control application or status. | Inspect remaining constructors individually. Use the existing isolated-window fixture where image content/startup is irrelevant; prefer a component fixture for a component-only assertion. | A real startup path, image/calibration behavior, navigation, worker shutdown and private settings/graphs/caches. |
| Heavy fixtures embedded in discoverable test modules | 89 modules import helpers from other `tests.test_*` modules, creating 172 distinct importer/dependency pairs. Imports can pull in unrelated environment checks and fixture definitions. | Move shared fixtures to support modules by responsibility, starting with the expensive recipe/reference foundations. Keep mutable state local. | Independent writable fixtures, deterministic source identities and explicit live-oracle witnesses. |

These are **consolidation candidates**, not a list of methods safe to delete
unchanged. A shared helper alone reduces maintenance duplication; it reduces
execution time only if it also eliminates repeated setup or equivalent runs.
Similarly, putting the same loops into one test method does not reduce work.

Keep checks with independent numerical or behavioral oracles: exact public/native
pixels, analytically known geometry, high-precision pooling/overflow cases,
held input/output immutability, changed-node cache invalidation and meaningful
save/reopen/batch/worker boundaries. Detailed k-means suites should be considered
by distinct algorithmic decision, not removed because their filenames look alike:
the 154 methods in 19 `test_plantcv_kmeans*` modules account for only 51.619 s of
recorded method time, although their class oracle setup costs are additional.

The audit has not produced an assertion-to-production-branch proof sufficient to
authorize retiring every repeated-looking guard. That map is the next prerequisite
for numerical coverage reductions.

## Pass 3: runtime and fixture costs

Timing evidence is the completed, source-pinned
[`test-suite-cleanup-full-v2` run](../artifacts/test-suite-cleanup-full-v2.json),
not a new run performed for this audit. All **564 recorded file hashes** still
match; all 239 test module hashes match. The detailed timing report's SHA256 was
verified against its run receipt. Historical results remain **2,807 passed,
4 skipped, no failures/errors**.

| Timing boundary | Seconds | Interpretation |
| --- | ---: | --- |
| Recorded method durations, 2,809 entries | 8,028.832 | Includes each timed method's per-test setup/teardown; excludes class setup/teardown. Two decorator skips have no duration entry. |
| unittest overall interval | 8,532.209 | Includes shared fixture work. |
| Runner report interval | 8,535.149 | Includes its discovery/loading interval. |
| Parent process interval | 8,585.360 | Includes process overhead and shutdown. |

The **503.377 s** difference between method durations and the unittest interval
is unallocated shared-fixture/runner work. It is not valid to attribute all of it
to GPU computation or to a particular class. Fixture reuse can help, but most
elapsed time is already inside recorded methods.

| Slowest methods retained | Recorded seconds | Share of all method time |
| --- | ---: | ---: |
| 1 | 1,327.016 | 16.53% |
| 10 | 2,297.216 | 28.61% |
| 25 | 3,114.049 | 38.79% |
| 50 | 3,860.548 | 48.08% |
| 100 | 4,817.896 | 60.01% |
| 250 | 6,268.132 | 78.07% |

The recipe lifecycle matrix alone takes **22.1 minutes**. Public seed acceptance
takes 278.218 s in its main method; card variant replay takes 133.600 s. Together,
the recipe/reference infrastructure and frozen-matrix categories account for
**78.02% of recorded method time**. Those are the first optimization targets.
These figures are observations, not promised recoverable savings.

The preceding cleanup improved real execution substantially, but retired only
three overlapping methods while adding two contract regressions: the net method
count fell by one. It did not perform a full semantic reduction of the suite.
That distinction explains why the count still looks excessive after a 41.0%
runtime reduction.

Existing reuse is useful and should survive: immutable workflow templates are
shared with source/config/callable keys and integrity checks; trait model JSON
is reused while fresh model instances are constructed; calibrated capture session
fault tests reuse actual public results instead of rerunning the producer for
every injected error. Private windows, graph state, caches, settings and files
must not become shared mutable fixtures.

## Pass 4: infrastructure, portability and maintenance

| Finding | Evidence and consequence | Recommendation |
| --- | --- | --- |
| CI bypasses the current runner and canonical namespace | Both jobs in [the workflow](../.github/workflows/tests.yml) use `unittest discover -s tests -v`, without `-t .`. Import-only reproduction discovered 2,811 tests with no errors, but **63 files were loaded under both `test_*` and `tests.test_*` names**. This is duplicate module state, not proof of duplicate test executions. | Route jobs through `-m tests` with explicit scope. For direct unittest use `-t .`. Use the runner's immediate discovery-error check and private temp setup. |
| Clean checkout cannot reproduce the advertised full gate | `artifacts/` is ignored. [Public acceptance inventory](../tests/test_public_seed_acceptance.py#L252) reads a frozen baseline without a reference-environment skip. Reader and recipe inventory checks also require local baselines/review receipts. The CI workflow only checks out code and installs runtime dependencies; it does not provision these files. | Declare and provision a hash-verified acceptance fixture bundle separately from generated outputs. Provide a quick preflight listing missing baseline/reference/runtime prerequisites. Do not regenerate expectations during tests or silently skip required acceptance. |
| Test tiers are incomplete | The current runner exposes a manually selected 132-method core or unfiltered full discovery. The hosted CPU job still requests the full suite, and the optional CUDA job uses the same command. | Define explicit core, CPU/Qt, CUDA numerical/integration and exhaustive acceptance selections from shared metadata. Preserve an unfiltered full command. Report selected tests, capabilities and omissions so green subset results remain honest. |
| Shared setup is not timed by class | The method report leaves 503.377 s unallocated and gives no class fixture breakdown. | Add setup/teardown attribution before deciding which shared public-oracle fixtures to extract next. Measure the whole selected run as well as methods. |
| Fixture dependencies are coupled to test discovery | The 89 cross-importing modules and the 63 duplicate legacy namespace loads show why fixture state can diverge between commands. | Extract support fixtures and use one canonical loader. Avoid importing whole test classes just to obtain a fixture function. |
| Core membership can drift | The core list is handwritten in `tests/__main__.py`; every new test outside those names is full-only. The inventory now records core membership for all methods. | Review selection by affected feature and cost. Do not use method count as a coverage percentage or imply that the 132 core methods cover the rest. |
| Runner error-path verification is historical/manual | The preceding cleanup saved a deliberate discovery-failure smoke result under ignored artifacts; current discovered tests do not target `tests.__main__`'s loading/reporting logic. | Retain a small subprocess-level check for invalid discovery, nonzero exit and honest reporting when changing the runner. Avoid creating another large matrix for this small interface. |

The clean-checkout finding is based on source and workflow inspection, not an
observed hosted CI failure. The legacy discovery reproduction is saved locally
as `artifacts/test-audit-legacy-discovery.json`.

The four historical skips remain explicit: the unavailable eleven-image pilot
batch, the opt-in full-image fill diagnostic, a case-sensitive-filesystem case,
and Windows directory-symlink privilege. These cannot be counted as passing
coverage. Missing scientific study inputs remain separate from software tests.

## Recommended implementation sequence

1. Align CI and local discovery; add prerequisite preflight and explicit execution
   tiers. This fixes reproducibility and selection without altering numerical tests.
2. Remove factorial repetition from legacy dialog tests after transferring unique
   assertions to contract/Pipeline workspace witnesses. Audit older six-order
   coverage against the newer 24-order contract matrix.
3. Map each expensive recipe/profile to its distinct algorithm, control, frame,
   cache and replay behavior. Preserve every unique numerical expectation, while
   separating all-variant structural checks from representative live integration.
   Keep the complete frozen acceptance gate explicit until that map proves a
   smaller numerical basis sufficient.
4. Attribute class fixture cost, then reuse immutable public-oracle/capture data
   and shared archive scaffolding. Keep mutation tests on private copies. Reduce
   common guard execution only after checking the owning production entry point
   and retaining adapter-specific wiring/buffer cases.
5. Separate historical transition integrity from ordinary current-version
   regression checks, and record every retired method's surviving assertions.

Success should be measured by lower whole-run time, a smaller set of independently
useful scenarios, clear prerequisite handling and preserved numerical/regression
oracles. Renaming methods, bundling unchanged loops, skipping unavailable fixtures
or loosening comparisons would not satisfy that objective.

## Validation of this audit

The generator reconciles all current methods through import-only discovery.
Inventory row IDs, source line links, source hashes, historical status counts and
timing IDs were checked against source and the completed full-run receipt. The
legacy CI discovery behavior was reproduced separately without executing tests.
No application or test implementation changed, so the unchanged numerical suite
was not rerun for this documentation/source audit. Previous acceptance artifacts
and frozen baselines were preserved.
