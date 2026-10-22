# Running and maintaining the tests

Runner policy (21 October 2026): wall time is advisory. `python -m tests --full` records whether the run stayed under 1,000 units and 900 seconds in `performance_budget` and prints a warning when it did not, but the verdict depends only on test outcomes and on every discovered unit being accounted for. A worker is stopped only when its journal shows no started or finished unit for `--stall-seconds` (default 900); the report then names the hung unit in `stalled_workers` and `errors`. Stale run directories are removed with `python scripts/prune_artifacts.py --older-than 7 --delete`.

Shared project-input checkpoint (20 October 2026): **Project → Image selector → Species and metadata**, plus **Species reference library**, are mandatory in Default and all 18 PlantCV/SMPTS factory templates. The selector and left Images list choose the same current raw image, including while calculations are frozen. Shared context positions and selection survive pipeline/layout save and load; version 23 migrates older Default settings without changing calculation parameters. The extra workspace nodes do not alter factory recipe definitions or imply species-library inputs to recipes that do not consume them. PlantCV colour-card calibration remains optional and distinct from built-in geometry correction and neutral balance.

The fresh unfiltered full suite passes **641 tests: 637 passes, 4 existing skips, zero failures/errors in 635.404 seconds** (635.468 parent seconds). Both the under-1,000-test and 15-minute budgets pass. Source and fixtures stayed unchanged during the run. The four frozen numerical baselines are unchanged; the provenance successor records only the workspace adapter source identity. Actual Qt controls were visually checked in Default, PlantCV and SMPTS workspaces with the 6,240×4,160 source loaded; this particular UI check did not execute image-processing operators. Evidence: artifacts/project-context-full-v2.json, artifacts/project-context-full-v2.receipt.json, artifacts/project-context-final-focus-v2.json and artifacts/project-context-ui/result.json. Full V1 completed in 662.245 seconds with four stale canvas expectations and a subsequent hover failure. The counts/Project-port expectations were updated; the unchanged hover assertions passed in the repaired clean module. The interrupted earlier CPU/Qt diagnostic remains incomplete, and the misspelled focused-test target ran no tests; none is counted as passing evidence. Original SMPTS parity and L01–L04 scientific acceptance are unchanged. Earlier dated checkpoints retain their original scope.

Workflow repair checkpoint (19 October 2026): the five reported issues are addressed. The fresh unfiltered suite passes **640 tests: 636 passes, 4 existing skips, zero failures/errors in 806.689 seconds** (806.821 parent seconds). Both the under-1,000-test and 15-minute budgets pass. Two reusable implementation workers and one independent reviewer were used. The native Qt harness triggered the real Run pipeline action and CUDA worker on the full 6,240×4,160 photograph for three cases: PlantCV working analysis with a foreground region, SMPTS factory defaults, and SMPTS working analysis with a foreground region. All completed; screenshots were inspected at 1440×900 and 1050×650. External desktop automation timed out. No calculation mocks or user project files were used. This verifies execution, not biological seed counts or original SMPTS conformance.

Foreground regions preserve full-frame coordinates through save, variants, export and replay. Peak filtering uses bounded tiles and global ordering. SMPTS work is charged as executed; its metadata-derived 7,936-component and 144 MiB packed-crop limits retain the composed per-operation 1 GiB CUDA bound. The factory case produced 4,918 source components and 6,255 crops; those counts are not validated seed counts. Per-operation admission does not bound total app memory. Size/colour count controls admit 1,024 resolved labels; precision, byte, traversal and work budgets independently limit calculation, archive and replay. Diagnostic geometry preparation and painting each retain a separate 16-million-visit bound. Distance previews use logarithmic colour without changing distances. Shared controls explain prerequisites and annotation editing uses the verified calibrated frame. Local/mixed optimization works without a library pin; configured invalid pins and Species library only remain errors. Per-image contexts and eligibility exclusions are explicit. Imported drafts, applied in-memory labels (which may remain unsaved after autosave failure), lossless 16-bit PNG export and training bundles have distinct readiness states; exporting a bundle does not train a model.

Evidence: artifacts/workflow-repairs-full-v2.json, artifacts/workflow-repairs-full-v2.receipt.json and artifacts/workflow-repairs-app-session-v3/completed.json. The four frozen numerical baselines remain byte-identical; the source-provenance successor changes code identities only. The first full V1 attempt timed out after 900.430 seconds: completed workers reported 344 units and the third worker remained incomplete, with its partial journal retained. Its two known assertion failures (stale paper-work planning and shared diagnostic work-counter drift) were repaired without changing frozen numerical baselines. Earlier dated checkpoints retain their original scope.

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

## Previous Pipeline file/backend verification

Latest Pipeline file/backend verification (18 October 2026): the unfiltered suite
passes **2,825 methods: 2,821 passes, four existing skips, no failures/errors**,
in **10,100.478 test seconds** (10,163.027 parent seconds). All **597 pinned
inputs** stayed unchanged. The current inventory records all outcomes and timings
in **248 test files**. This update retires **33 obsolete Reference-window methods**
and adds **29 file/backend methods**; mixed service, model, QC, persistence and
numerical assertions remain. All **223 frozen profiles** and four immutable
numerical baselines are preserved. No overall speedup is claimed.

Evidence: `artifacts/pipeline-backend-full-v3.json` and its timing report/log,
source/visual review V4 and final evidence review V5. The interrupted full V1,
full V2 with four editable-fixture failures and CPU/Qt V1 toolbar failure remain
retained. The procedural-fit fixture now explicitly unfreezes for its editable
completion scenarios; its assertions and production guards remain. The responsive toolbar repair
passed its focused gate and the fresh full suite. Earlier audit timing tables
below retain their original, dated source scope.

The [complete test inventory](testing/TEST_INVENTORY.md) records the suite's
methods and their functionality, with detailed assertion, helper, fixture and
timing evidence in [JSON](testing/test_inventory.json). The separate
[multi-pass infrastructure audit](TEST_INFRASTRUCTURE_AUDIT.md) examines overlap,
expensive combinations, fixture reuse, CI discovery and clean-checkout prerequisites.
Regenerate the inventory without executing tests or setup fixtures:

```powershell
.venv\Scripts\python.exe tools/audit_test_inventory.py --collect --receipt artifacts/test-suite-speed-completed-segments-v1.json
```

Timing enrichment requires the selected completed source-pinned receipt and its
ignored timing report. The command above reproduces the current inventory;
omitting `--receipt` uses the original cleanup full-v2 audit input. Add
`--without-history` for a source-only inventory without local receipts;
historical outcomes and timings are then unrecorded.

From the project directory in PowerShell:

```powershell
.venv\Scripts\python.exe -m tests
.venv\Scripts\python.exe -m tests --full
```

Explicit subsets use the same canonical discovery and runner:

```powershell
.venv\Scripts\python.exe -m tests --tier cpu-qt
.venv\Scripts\python.exe -m tests --tier cuda
.venv\Scripts\python.exe -m tests --tier acceptance
.venv\Scripts\python.exe -m tests --tier history
.venv\Scripts\python.exe -m tests --tier cpu-qt --list
```

`tests/selection.py` declares the CPU/Qt allowlist and the three acceptance
modules. Mixed-module methods that perform actual CUDA calculations have explicit
CUDA overrides. Every remaining discovered ID belongs to the CUDA group. These
three groups partition the entire suite exactly once; new or misspelled selection
entries cannot silently remove methods. The core and historical selections are
overlapping convenience subsets. Reports record selected/omitted IDs and group
counts. `--list` imports modules but runs no methods or setup fixtures; it cannot
be combined with preflight or fixture provisioning. `--full` always runs every
discovered test, irrespective of tier membership.

The default command runs the core regression selection: pipeline behavior,
settings and project persistence, recipe contracts, exact threshold/median
calculations, image decoding, and the main workspace's execution/cache/batch
paths. It reports its scope explicitly. It is a development check, not a claim
that the complete suite passed.

`--full` uses ordinary unfiltered `unittest` discovery with the repository as
the top-level directory. This keeps every test in the `tests` package namespace
so imported fixtures and discovered tests share the same modules. It is also
available directly:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -t . -v --durations 20
```

Run a particular module, class, or method by passing its unittest name. To retain
per-test durations and the overall result, select a new report filename:

```powershell
.venv\Scripts\python.exe -m tests tests.test_pipeline_workflow_ui -v
.venv\Scripts\python.exe -m tests --full --report artifacts/my-full-test-run.json
```

Reports cannot overwrite an existing file and stay under ignored `artifacts/`.
Each invocation uses a private temporary directory under `artifacts/test-temp/`;
it does not depend on writable Windows user-temp permissions. Add `-f` to stop a
focused diagnostic at its first failure.
Discovery/import errors stop the command before any test methods run. The report
records zero executed tests and the discovery tracebacks, so an invalid test
module cannot trigger hours of unrelated numerical work.
The complete suite still needs the existing reviewed numerical baselines and
pinned reference environment. Tests do not regenerate expectations or fetch
reference code. Missing fixtures and platform-dependent skips remain explicit.
Run affected tests during implementation and the complete suite before hand-off.

## Prerequisites and CI

`config/test_fixtures.json` declares 30 immutable reference/acceptance input files
and their SHA256 values, including the unchanged four frozen numerical baselines.
They remain under ignored `artifacts/`, separate from generated outputs. A full,
CUDA or acceptance tier fails before execution if its required fixtures/runtime
are missing or changed. CUDA tiers require an available CUDA device and the pinned
PlantCV interpreter. Existing platform/fixture/opt-in test skips remain explicit.
Check or restore prerequisites without running the suite:

```powershell
.venv\Scripts\python.exe -m tests --full --preflight
.venv\Scripts\python.exe -m tests --full --preflight --fixtures-from D:\SeedFiddle-test-fixtures
```

The local bundle uses repository-relative paths such as
`D:\SeedFiddle-test-fixtures\artifacts\recipe-acceptance-baseline-v2.json`.
Only missing files whose bytes match the committed manifest are copied; existing
files are never overwritten. Install the pinned reference environment separately
using the existing [reference setup workflow](BENCHMARK_REPORTS.md). Tests never
download source or produce a replacement expectation. The CPU/Qt tier does not
require ignored acceptance or public-reference fixtures.

Hosted CI runs the explicit CPU/Qt tier. The optional self-hosted CUDA job runs
unfiltered full acceptance. It preserves its configured `.venv/` and `artifacts/`
while cleaning source files; required fixture hashes are still checked. Set the
repository variable `SEEDFIDDLE_TEST_FIXTURES` to a local bundle directory if the
runner should provision missing reviewed inputs. Full reports use unique run IDs
and are uploaded alongside CPU/Qt reports. A subset pass is never labeled full
acceptance.

Reports also contain `fixture_durations` for actual class setup and class
teardown/cleanup boundaries. The runner delegates those operations to unittest;
it does not replace fixture functions or share mutable test state. Method timing
includes per-test setup/teardown. Overall test, discovery/preflight and parent
process timing remain separate measures.

## Audit recommendation implementation, 15 October 2026

Two workers consolidated repeated execution in nine recipe/dialog/replay suites;
an independent reviewer checked their assertion-survival maps and the runner/CI
changes. Reader/median/fill legacy dialog construction fell from 24/6/2 instances
to one per adapter. All 24 reader and six historical median contract orders remain,
including their distinct selected values and layout/version assertions.

The nine card variants retain exact structural prefix/input-edge checks with one
live prefix execution and four target replays. Eight crop variants retain exact
wiring checks, real base/diagnostic consumers and one additional live replay.
Five distinct ROI crops still compare their exact pixels, frames and aliases;
only unchanged upstream calculations are reused through a private method cache.
Shared archive/profiling helpers live in `tests/replay_fixtures.py` and hold no
global mutable numerical result. Every adapter-specific mutation/cancellation,
admission, numerical oracle and source-absent replay check remains.

At that audit checkpoint no original test ID was removed. Actual repeated executions were consolidated;
three small runner regressions were added for error reporting, exhaustive tier
selection/fixture lifecycle, and immutable fixture provisioning. All 223 frozen
cold/warm numerical profiles and their complete comparisons remain unchanged.
The [profile coverage map](testing/RECIPE_PROFILE_COVERAGE.md) records their exact
control/frame distinctions and the existing 70 source-absent replay representatives.
The fresh unfiltered run passed **2,814 methods: 2,810 passes, four existing
skips, no failures/errors**. Source pins remained unchanged for all **573 files**;
the four frozen numerical baselines were preserved byte for byte. The inventory
reconciles all 2,814 IDs in 240 files and records the fresh outcomes and timings.

| Timing boundary | Previous full run | Fresh full run |
| --- | ---: | ---: |
| unittest interval | 8,532.209 s | 8,902.126 s |
| Runner interval including discovery/preflight | 8,535.149 s | 8,905.395 s |
| Parent process including shutdown | 8,585.360 s | 8,955.872 s |
| Method time in the nine changed test modules | 1,588.402 s | 1,442.039 s |
| Method time in unchanged test modules | 6,440.430 s | 6,921.818 s |

The changed modules used **9.2% less recorded method time**, while the whole run
took **4.3% longer**. These are two observed runs, not a controlled benchmark;
the evidence does not establish a cause for the increases or an overall speedup.
The three added runner methods together took 0.244 seconds in the full run.

| Consolidated method | Previous | Fresh |
| --- | ---: | ---: |
| Reader dialog lifecycle | 80.444 s | 5.974 s |
| Median dialog lifecycle | 18.157 s | 6.048 s |
| Fill dialog lifecycle | 6.141 s | 5.789 s |
| Nine card variants / shared prefix replay | 133.600 s | 12.751 s |
| Eight crop variants / representative replay | 57.659 s | 12.093 s |

Fixture attribution records 129 class setup and 29 teardown/cleanup boundaries,
with no duplicate class/fixture keys. Recorded methods total 8,364.102 seconds;
class fixtures total 537.818 seconds. About 0.207 seconds remains within the
unittest interval. The largest setup is the actual calibrated composition
fixture at 136.861 seconds; it is not an extra method execution. This information
supports later fixture work without guessing from method names alone.

The separate clean-checkout CPU/Qt check passed **616 methods, 615 passes and
one existing case-sensitive-filesystem skip**, in **221.599 test seconds**
(251.244 parent seconds), with CUDA hidden and ignored reference/acceptance
inputs absent. The initial copy check caught three real CUDA methods in mixed
modules; explicit CUDA overrides fixed the partition without adding skips.
The final partition is 616 CPU/Qt + 2,184 CUDA + 14 exhaustive acceptance methods.
Hosted GitHub jobs were not dispatched during this local verification.

The four full-run skips remain the unavailable eleven-image pilot batch, opt-in
full-image fill diagnostic, case-sensitive-filesystem case, and Windows directory
symlink privilege. They are unchanged limits, not passing numerical coverage.

Evidence for that audit checkpoint is `artifacts/audit-recommendations-full-v1.json`, its log and
timing report, `artifacts/audit-timing-comparison-v1.json`, both worker assertion
survival maps, clean-checkout CPU/Qt V2, and independent review V5. The original
audit and preceding cleanup results below retain their historical scope.

## Cleanup audit, 15 October 2026

The preceding complete run took 14,470.840 seconds: 2,812 methods, 2,808 passes,
four skips. Its log contains no repeated test IDs; reducing the number of names
alone would not address the repeated work.

Three overlapping executions were retired with their distinct checks retained:

| Retired execution | Retained coverage |
| --- | --- |
| Standalone 28-image format/depth/alpha/orientation matrix in `test_native_image_inputs` | CPU source/snapshot/decode checks in reader acceptance's frozen inventory; the existing complete reader matrix still runs the actual public oracle and CUDA preparations. Exact pixels, dtype/shape, failures, 23 successful inputs, filenames and Unicode remain checked. |
| Standalone 28-image gray-reader matrix in `test_native_reader_recipe` | The complete reader matrix already checks both gray modes, differing pixels, failures, Unicode and cache behavior. Node editing, mutation and cancellation tests remain independent. |
| Separate partial-mean boundary run in recipe acceptance | The full lifecycle capture already executes and reopens that identical partial boundary; its outer assertions now use that result. Distinct complete-mean comparisons against PlantCV remain. |

The structural recipe inventory no longer repeats admission validation of the
same profiles. Every one of the 223 actual cold/warm profile executions still
passes through the real admission and runtime checks. All frozen numerical
comparisons, verified archive checks and source-absent replay checks remain.

Selected UI fixtures reuse only immutable workflow templates and serialized
trait-model data. Windows, widgets, graphs, settings, images, writable files and
analysis caches remain private. Template reuse has a source/config/callable key
and integrity checks; real catalogue, source-validation, startup and classifier
training witnesses continue to use their real paths.

Profiling also found repeated whole-recipe serialization inside graph projection
and validation. A local identity is now reused within one graph operation, and
each initial stage record is constructed once. Public calls still compute a
fresh identity; file/factory/runtime validation is not memoized. Iterator-order
and late mapping-mutation regressions protect those boundaries.

The complete suite now takes **8,532.209 seconds**, down from 14,470.840
seconds: **41.0% less test time, 1.70 times as fast**. It runs
2,811 methods (2,807 passes, 4 existing skips, zero failures/errors).
The core selection takes **35.303 seconds** for 132 methods (131 passes, one
case-sensitive-filesystem skip); it is not complete acceptance. Overall core
wall time including discovery and shutdown is 39.457 seconds; full-run
wall time including discovery and shutdown is 8585.360 seconds.

| Setup benchmark | Before | After |
| --- | ---: | ---: |
| Three real window constructors, profiled total | 33.766 s | 12.295 s |
| Workflow template catalogue, unprofiled | 4.004 s | 1.362 s |
| Warm shared-template window constructor, profiled | — | 0.308–0.309 s |

Setup measurements are separate microbenchmarks, not full-run timings. Individual
method durations do not include class setup; the overall timing includes it.
The full-run ledger/log and detailed timings are in ignored artifacts under
`test-suite-cleanup-full-v2`; core results are `test-suite-cleanup-core-v2.json`.
Historical baselines and earlier run receipts remain unchanged. This cleanup
does not change scientific or original-SMPTS acceptance.
