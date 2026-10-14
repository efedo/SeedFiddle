# SeedFiddle hand-off

Ready for user evaluation: the current software is ready, with no engineering prerequisite identified before evaluation. Full V5 verification records 2,786 passes, four explicit skips, zero failures and zero errors; the final metadata V3 gate records six passes. L01-L04 remain pending final scientific validation rather than coding blocks. Further SMPTS source recovery is deferred by the user; F09.2 remains unresolved, no distance node is invented and original SMPTS parity is unverified. This evaluation handoff does not claim scientific or full-objective acceptance.

## Current software scope and completed V5 verification (13 October)

39 of 40 operations implement their declared guarded software domains. F09.2 remains an unresolved original SMPTS source question, with no invented stage. F08.2 retains its single select_contacts mapping and blocked original-source conformance. Paper-port conformance remains provisional; scientific statuses and external L01-L04 requirements are unchanged.

The distinct unfiltered V5 full suite completed **PASS_WITH_SKIPS**: **2,790 discovered/tests, 2,786 passes, four skips, zero failures and zero errors in 13,406.906 seconds**, with zero expected failures or unexpected successes. Full-suite completion/pass, completion verification and source verification before/after are true. All-discovered-methods-executed is false because four methods were skipped; no individual pass is claimed for them.
- tests.test_image_calibration.ImageCalibrationTests.test_pilot_batch_rims_match_the_calibrated_96_mm_outer_edges: The complete eleven-image pilot batch is not present
- tests.test_plantcv_fill.PlantCVFillTests.test_committed_real_image_mask_performance_and_memory_diagnostic: run explicitly with SEEDFIDDLE_RUN_FILL_IMAGE_DIAGNOSTIC=1
- tests.test_project_analysis_persistence.ProjectAnalysisPersistenceTests.test_posix_project_can_store_source_locators_that_differ_only_by_case: requires a case-sensitive source filesystem
- tests.test_project_correction_inputs.ProjectCorrectionInputTests.test_bound_path_cannot_resolve_outside_workspace: Directory symlinks unavailable: [WinError 1314] A required privilege is not held by the client: 'D:\\Programming\\SeedFiddle\\artifacts\\test-temp-f16\\tmpj_psqvyz\\outside' -> 'D:\\Programming\\SeedFiddle\\artifacts\\test-temp-f16\\tmpj_psqvyz\\workspace\\redirect'

The full 223-profile recipe comparison, public seed acceptance, reader acceptance, profile inventory and all three formerly failing methods passed in V5. Root independently verified their recorded results, driver/log hashes and all **471 source/test/fixed-baseline files** unchanged. Source V7 remains frozen at SHA256 e42953f53a45460cf3574fb9a71ab16c65422d1f272534f15f0a6826ae975889: 33 candidate and 441 prior entries plus four fixed baselines, 471 unique files. The register still references **197 test files** and **78 new method references**; full discovery includes 236 files.

The completed failed V4 run, interrupted V2/V3 attempts, V1 preflight failure and initial five-profile exporter-pin failure remain separately retained with their original scopes. The six-file post-V4 repair passed seven focused methods in 10.219 seconds; bounded independent static review did not execute tests/GPU or inspect visuals. Earlier six documentation methods passed before V5. Final documentation verification is recorded separately in artifacts/complete-software-documentation-focus-run-v3.json and artifacts/complete-software-documentation-update-v8.json; these references do not claim a pass or establish scientific acceptance. Earlier scoped QC, resource and visual evidence retains its limits.

Current evidence: artifacts/complete-software-full-suite-run-v5.json (SHA256 5f3fe8a4bea362bd3f4f5e38dbeb742010b0f2ef00fff7a209e67cafad5acc06) and artifacts/complete-software-completed-full-v5-checkpoint-v1.json (SHA256 13b738fdae98606b19f0b0715bfdc6c062ae399790ae316b2e1d99ce0e7ee6be). Exact pre-reconciliation documents remain under artifacts/complete-software-final-audit-input-v1/ with the same relative paths.

The goal remains active. No overall completion or scientific acceptance is claimed. No further concrete coding gap is established within the implemented declared software domains by this gate; F09.2 requires original SMPTS source to resolve, original parity requires source/license/dependencies, and independent L01-L04 studies remain external. The four skipped checks retain their explicit fixture/opt-in/platform limits. Final documentation verification is recorded separately in artifacts/complete-software-documentation-focus-run-v3.json and artifacts/complete-software-documentation-update-v8.json; these references do not claim a pass or establish scientific acceptance. The single seed_vision.py/native PySide6 architecture, CUDA residency and node-local cache remain intact.

The 13 October primary-capsule access recheck found 403 through both ordinary public web fetch and the normal browser. The paper reports an MIT license; actual archive/license bytes and executable dependencies were not obtained. Evidence: artifacts/complete-software-smpts-access-recheck-v1.json (SHA256 416f1172424b1215bbdc8a0d54744553d6969d69ef16b2f43ed753c7996eebdd). The historical 3 October pin remains retained. The focused source search does not prove that no mirror exists.

## Prior documentation V6 checkpoint before V5 (historical)

39 of 40 operations implement their declared guarded software domains. F09.2 remains an unresolved original SMPTS source question, with no invented stage. F08.2 retains its single select_contacts mapping and blocked original-source conformance. Paper-port evidence, scientific statuses and external L01-L04 requirements remain separate and unchanged.

The distinct unfiltered V4 full suite completed with **FAIL**: 2,788 tests, 2,781 passes, four skips, one failure and two errors in **13,320.016 seconds**. Source, documentation and the complete 236-file test inventory verified before and after the run. The full 223-profile recipe comparison, public seed acceptance and reader acceptance passed within this failed run; those passing subsets do not make the full suite pass. The earlier V1 preflight failure and interrupted V2/V3 attempts remain retained separately.

The six-file post-V4 repair preserves typed tensor-callback mutation propagation across K-means sampling/training, preserves unchanged callback exception identities, and supplies valid explicit metadata defaults in two worker fixtures. The focused repair gate passed **seven of seven methods in 10.219 seconds**, with zero failures, errors or skips. GPT-6.1 Sol medium independently inspected the bounded repair files; review used static source inspection without tests, GPU execution or visual inspection. This evidence makes no scientific claim.

Source V7 freezes 33 candidate and 441 prior entries plus four fixed baselines, **471 unique verified files**. Its SHA256 is e42953f53a45460cf3574fb9a71ab16c65422d1f272534f15f0a6826ae975889. The feature register references **197 test files** and 78 new method references; full discovery contains **236 files**, including 39 other pre-existing test files. Fixed numerical baselines remain unchanged.

Earlier affected gates retain their own scopes: 123/123 in 380.25 seconds, the separate 11/11 limited repair/documentation gate, and the five-profile/Qt V2 capture in 98.688 seconds with zero complete JSON changes. Earlier three QC views and two cold/warm CUDA resource measurements remain scoped evidence. The initial five-profile exporter-pin failure remains immutable history. No all-DPI or independent scientific result follows.

The fresh distinct V5 full-suite pass remains **pending**; it has not run at this documentation freeze. The goal remains active. No completed-goal, full-suite-pass or scientific acceptance is claimed.

Current evidence: artifacts/complete-software-full-suite-run-v4.json, artifacts/complete-software-focus-run-v4.json, artifacts/complete-software-bounded-repair-review-v1.json, artifacts/complete-software-frozen-hashes-v7.json and artifacts/complete-software-full-discovery-test-inventory-v4.json. The review SHA256 is a264c35c1508ed49c868fadc1d54a7eba75a6cb62e25fb38b9710eb824ef70f5; the inventory SHA256 is f6274fafb323f9e81263a59b2d740c94094059c430d3a3ae2861023fdae55b17. The earlier supplied-only V4 projection review retains its historical scope. The single seed_vision.py/native PySide6 architecture, CUDA residency and node-local cache remain intact.

## Prior documentation V4 checkpoint (historical)

39 of 40 operations implement their declared guarded software domains. F09.2 remains an unresolved original SMPTS source question, with no invented stage. F08.2 retains its single select_contacts mapping and blocked original-source conformance. Paper-port evidence, scientific statuses and external L01-L04 requirements remain separate and unchanged.

The affected repair gate passed **123 of 123 tests in 380.25 seconds**, with zero failures, errors or skips. The fresh limited repair capture then passed **11 of 11 checks**, also with zero failures, errors or skips: four complete JSON difference tests, one historical baseline migration with 67 exact dependency-pin changes and six peer-feature-register methods. These are separate gates; the 123-test gate did not include the documentation suite. Earlier 183-test focused evidence and the one-test compatibility repair retain their recorded scopes.

The fresh five-profile/Qt V2 capture passed in **98.688 seconds**. All five actual native profiles matched the complete expected JSON, with **zero changes**, using the reviewed 495 exact code-provenance transitions. Fixed numerical baselines remain unchanged. Root viewed the new actual trait-inspector PNG at 800 by 900 with Segoe UI 9; both manual controls and the explicit optimization exemption were readable. No model or image analysis was loaded; the displayed bypass status is the recorded historical default. Earlier three QC views and two cold/warm CUDA CLI resource measurements retain their scope. No all-DPI or scientific claim follows.

Source V6 remains unchanged: 29 candidate and 441 prior entries plus fixed baselines, **471 unique verified files**. The feature register references **197 test files** and 76 new method references. Full discovery contains **236 files**, including 39 other pre-existing test files. The same reviewer formally approved supplied repair and all 495 code-pin projections without independent I/O, tests or visual inspection.

The first five-profile QA failed on one exporter-source code pin and remains immutable history. V1 full-suite inventory preflight ran zero tests; V2 and V3 full suites were interrupted and incomplete. V3 recorded 2,074 completed log entries: 2,061 passes, six failures, three errors and four skips; its final matrix comparison did not complete. None is a full-suite pass. The current full 223-profile complete comparison and a planned fresh distinct unfiltered V4 full suite remain pending. No completed-goal or scientific acceptance is claimed.

Current evidence: artifacts/complete-software-focus-run-v3.json, artifacts/complete-software-repair-qc-v2.json, artifacts/complete-software-repair-root-visual-qa-v1.json, artifacts/complete-software-projection-review-v4.json, artifacts/complete-software-frozen-hashes-v6.json and artifacts/complete-software-documentation-update-v4.json. See [Benchmark reports](docs/BENCHMARK_REPORTS.md) for existing launch-point actions and limits. The single seed_vision.py/native PySide6 architecture, CUDA residency, node-local cache and user-authorized full goal remain intact and active.
## Historical verification and planning

Prior records below keep their dated scope and nonpassing history; current predicates above supersede their former next-work statements.

## Prior V2 finite software reconciliation (historical)

39 of the 40 operation rows implement their declared guarded software domains. F09.2 remains an unresolved original SMPTS source question: no distance-transform stage is invented. PlantCV conformance is scoped to pinned witnessed domains; paper-port engineering evidence stays distinct from blocked original parity. Separately named native/enhanced methods do not acquire PlantCV identity. Scientific statuses are unchanged.

The final concrete gaps are implemented: desktop public-reference exception/cancellation and withheld-delivery QC; actual resource/version/cache/synchronization/peak/logical-transfer and correction-time reports; prespecified absolute and paired connected-group comparisons with complete failure/abstention/stratum accounting; and typed frozen Bayes/K-means model/training-evidence binding to audited access and known consumed image identities. Unknown source membership stays ineligible. See [Benchmark reports](docs/BENCHMARK_REPORTS.md) for existing launch-point actions and exact limits.

Root focused verification:183/183 PASS319.328s, zero skips, including68 new methods. V1 remains retained (129 passes,16 failures,38 errors), largely Windows fixture-read denials plus the repaired non-session guard ordering. V2 freeze SHA c204c6e336fbeb443699de14ce1da35f12040553435147ad70503af9bca038da covers15 candidate plus450 prior entries; the unique source/test/fixed union has466 verified files, including195 registered test files. The same reviewer approved supplied15-file projections without independent I/O/tests/visual inspection. One automatic focused fixture execution was accepted; zero human project-edit permission requests. No scientific or completed-goal claim follows.

The distinct full unittest suite and final root QA remain pending. Original SMPTS source/license/dependencies and independent L01-L04 inputs remain external. Preserve seed_vision.py/native PySide6, CUDA residency, node-local recomputation and honest controls. The user-authorized full goal remains active. Generic historical broader-domain text below is not a new software gate; declared unsupported/numerically withheld cases remain withheld.

Current evidence: artifacts/complete-software-focus-run-v2.json, artifacts/complete-software-frozen-hashes-v2.json, artifacts/complete-software-projection-review-v1.json and artifacts/complete-software-documentation-update-v1.json.

## Historical checkpoints and prior planning (dated scope retained)

Everything below retains its original checkpoint scope. Former partial statuses and next-work statements are superseded by the current finite reconciliation above; historical evidence and nonpassing executions remain retained.

## Original PlantCV saved-project capture and QC (13 October)

Saved projects now capture selected supported original PlantCV seed recipes through the pinned public runner and independently reopen complete manifests before QC publication. Reference reports remain distinct from CUDA native archives. Complete, partial, empty, failed, unavailable and cancelled facts retain their own scope; unsupported originals do not request a runner. The shared desktop QC action, exact acknowledgments and private project/snapshot transport preserve recorded evidence without rereading original sources. Actual independent desktop and batch producers agree on deterministic analytical products, recipe/source/frame, pins, parameters, observations and calibration facts; each actual capture keeps its own QC revision.

Root verification:91 current methods PASS270.266s, zero skips, including19 new methods. Earlier fixture/Windows compatibility failures and the passing90-method V3 are retained. An actual original PlantCV grayscale color-statistics failure is retained as partial; fresh chromatic QA passes without production changes. Root inspected three actual Qt views with registered Segoe UI9; the same reviewer approved supplied projections without independent I/O/tests/visual inspection. 447 prior files and four fixed baselines remain unchanged, four production supersessions are explicit,454 files are frozen and190 test files registered. Two narrow fixture requests used automatic review; zero renewed human project-edit approvals. All test/capture processes are terminal.

Evidence: `artifacts/f20-public-reference-batch-gate-v1.json`, `artifacts/f20-public-reference-batch-frozen-hashes-v4.json`, `artifacts/f20-public-reference-batch-projection-review-v1.json` and `artifacts/f20-public-reference-batch-root-visual-qa-v1.json`. Coarse statuses are retained pending the finite all-feature reconciliation; scientific/full-goal acceptance is not claimed.

Complete the concrete remaining software gaps found by the bounded F01-F22 audit: retain shared attempt QC for actual desktop public-reference exceptions/cancellation; produce a consolidated actual hardware/version/cache/synchronization/peak/transfer/correction-time-method resource report; implement prespecified absolute and comparative gates with paired resampling of connected biological/acquisition groups and complete failure/abstention/stratum accounting; bind classifier evaluation to audited manifest, actual training evidence and frozen model identity. Existing graph/primitive/cache/persistence witnesses count against the literal acceptance predicates; historical partial labels and generic broader-domain text do not create new implementation requirements. Original SMPTS source/license/dependencies and independent laboratory L01-L04 studies remain external. Reconcile all operation statuses against the finite audit and run a distinct full unittest suite before completed handoff. Goal active; reuse the same two workers and reviewer.

## F22 fixed catalogue source fidelity (12 October)

Catalogue deserialization and explicitly injected reference-graph catalogues now validate against independently reconstructed authored recipes before audited-function claims appear. Coherent operation, source-symbol, stage, wiring, port, parameter, provenance and paper-condition changes reject even when IDs and pins remain valid. Only graph layout is excluded from strict canonical byte comparison; selected working and historical metadata remains preserved through separate APIs.

Root verification: 68 distinct targeted methods have passing executions across three terminal runs (553.876 seconds), zero skips and eight new methods. The initial authoring fixture error and the compatibility run with one method containing two Windows path-read subcase errors remain retained; the repaired fixture and narrow automatic path rerun pass without production guard changes. The same reviewer approves the supplied three-file V2 freeze without independent I/O, tests or visual inspection. Root inspected actual PlantCV and unresolved SMPTS selected graph viewports at 1440x900 with registered Segoe UI9; complete offscreen graph and all-DPI QA remain open. 448 prior files and four fixed baselines match, with two explicit production supersessions. 189 test files are registered. All 40 coarse statuses remain unchanged. One automatic fixture-path review was accepted; zero renewed human project-edit requests.

Evidence: `artifacts/f22-catalogue-source-gate-v1.json`, `artifacts/f22-catalogue-source-frozen-hashes-v2.json`, `artifacts/f22-catalogue-source-projection-review-v1.json` and `artifacts/f22-catalogue-source-root-visual-qa-v1.json`. The reproduced fixed-catalogue defect is closed; whole F22 and scientific acceptance are not claimed. All tests/captures are terminal.

Implement the supported original PlantCV saved-project capture route using the existing pinned public runner and independently reopened manifests. Keep reference reports distinct from CUDA native archives; unsupported, unselected and unavailable original SMPTS chains remain explicitly unavailable. Reconcile the finite F01-F22/WP0-WP6 acceptance predicates against actual producer witnesses, including exact desktop/batch recipe-source-backend facts, failed/unavailable outcomes, crop/calibration mapping and revision-bound acknowledgment transport. F19 does not require new terminal records for unrelated library, training, import or profile management actions unless they prevent or alter the evaluated recipe/prediction request. Complete remaining F22 source/graph/checkpoint/cache/persistence acceptance and run a distinct full unittest suite before completed handoff. Original SMPTS source/license/dependencies and independent laboratory L01-L04 evidence remain external. The goal remains active. Reuse the same worker/reviewer pool.

## F19 recipe row and batch QC (12 October)

Saved graph rows now retain exact verified archive QC or distinct failed, unavailable and cancelled attempt evidence. Separate batch records preserve the validated selection and completed prefix when admission, a row, cancellation between rows or final publication fails. Stop after a completed row preserves its successful QC; staging cleanup, historical linkage and current unverified archive availability remain distinct. Original/reference executors and missing correction bindings remain unavailable without fallback. Explicit project ownership, exact acknowledgment and dependent QC bytes survive private snapshot/restore/resnapshot after originals are removed. The Qt viewer loads an explicitly selected ledger row and reviews separate batch terminal records.

Root verification: 61 distinct methods pass across two disjoint terminal runs (196.625 seconds), zero skips and 15 new methods. The same reviewer approves the supplied seven-file freeze without independent I/O, tests, whole-file or visual inspection. Root inspected actual unavailable-row and between-row cancellation dialogs at 860x650 with registered Segoe UI9; no all-DPI claim. 443 prior files and four fixed baselines match; five supersessions are explicit. 188 test files are registered. All 40 coarse statuses remain unchanged. The initial Windows source-read capture failure is retained; a fresh automatic-review capture passes without runtime guard edits or renewed human project-edit approval.

Evidence: `artifacts/f19-recipe-batch-gate-v1.json`, `artifacts/f19-recipe-batch-frozen-hashes-v1.json`, `artifacts/f19-recipe-batch-projection-review-v1.json` and `artifacts/f19-recipe-batch-root-visual-qa-v1.json`. All tests/captures are terminal.

Reconcile the finite remaining F01-F22/WP0-WP6 software/conformance predicates against actual producer coverage. Preserve request-capture failures before a captured request exists and audit remaining sidecar/library/training/import and post-profile-success boundaries. Continue current F22 and all remaining requested functionality without expanding historical QC into scientific approval. Original SMPTS source/license/dependencies and independent laboratory studies remain external. Run a distinct full unittest suite before completed handoff. The goal remains active. Reuse the same worker/reviewer pool.

## F19 correction attempt QC (12 October)

Four-file and saved-project correction failures, cancellation and rejected results now retain distinct QC-v2 attempt records with the actual invoked wrapper and four selected-file observations. Returned source and target hashes/frames remain separate from unverified outputs. Stop or stale delivery withholds result/Keep/export installation; replacement workers and unrelated cache survive. Qt delivery records no fresh file reads. Exact cancellation QC and acknowledgment survive private snapshot/restore/resnapshot after originals are removed.

Root verification: 92 distinct methods have passing current-production executions across two terminal runs (414.313 seconds), zero skips and 11 new methods. The first 92-method run had 90 passes and two new fixture failures; both repaired fixtures pass in the narrow 11-method rerun with all runtime guards and production hashes unchanged. The same reviewer approves the supplied four-file V2 freeze without independent I/O, tests, whole-file or visual inspection. Root inspected actual failed/cancelled QC dialogs at 860x650 with registered Segoe UI9; no all-DPI claim. 444 prior files and four fixed baselines match; three supersessions are explicit. 187 test files are registered. All 40 coarse statuses remain unchanged. Three automatic fixture/capture reviews were accepted; zero renewed human project-edit approvals.

Evidence: `artifacts/f19-correction-attempt-gate-v1.json`, `artifacts/f19-correction-attempt-frozen-hashes-v2.json`, `artifacts/f19-correction-attempt-projection-review-v1.json` and `artifacts/f19-correction-attempt-root-visual-qa-v1.json`. All tests/captures are terminal.

Integrate attempt QC into unavailable/reference batch graph rows and successful verified row linkage. Preserve separate batch-level cancellation during staged publication or between rows, with exact acknowledgment/history association and no relabeling of successful recipes. Preserve request-capture failures before a captured request exists. Audit wider sidecar/library/training/import and post-profile-success boundaries. Continue current F22 and all remaining F01-F22/WP0-WP6 software/conformance requirements. Original SMPTS and independent laboratory studies remain external. Run a distinct full unittest suite before completed handoff. The goal remains active. Reuse the same worker/reviewer pool.

## F19 generic recipe attempt QC (12 October)

Generic native recipe failures and cancellation now retain distinct QC-v2 attempt records. Bounded selected-file byte observations, actual invoked wrapper, failure phase and returned source/frame facts are captured only where available. Archive, node-output, measurement and science facts remain unknown. Stop after actual CUDA return or queued delivery withholds outputs; a stale signal preserves history without changing a replacement worker, chart, selection or cache. The shared Qt viewer exports/reopens/acknowledges and explicitly retains attempt QC. Mixed v1/v2 exact bytes and acknowledgment events survive private snapshot/restore/resnapshot after originals are removed; historical loading never probes sources.

Root verification: 67 distinct current frozen-body methods pass across two disjoint terminal runs (112.068 seconds), zero skips and 13 new methods. Earlier six Windows strict fixture-path failures, a superseded executor-attribution candidate and an unusable-font capture are retained. Four automatic reviews accepted narrow fixture/capture runs with runtime guards unchanged and zero renewed human project-edit approvals. The same reviewer approves the supplied five-file delta/freeze without independent I/O, tests, whole-file or visual inspection. Root inspected two QC dialogs at 860x650 with registered Segoe UI9; default-font/all-DPI guarantees are not claimed. 442 prior files and four fixed baselines match; three supersessions are explicit. 186 test files are registered; all 40 coarse operation statuses remain unchanged.

Evidence: `artifacts/f19-recipe-attempt-gate-v1.json`, `artifacts/f19-recipe-attempt-frozen-hashes-v2.json`, `artifacts/f19-recipe-attempt-projection-review-v1.json` and `artifacts/f19-recipe-attempt-root-visual-qa-v1.json`. All tests/captures are terminal. Four-file correction and batch producer integration remain open; byte observations do not verify selected formats or approve outputs.

Integrate attempt QC into the four-file correction worker and unavailable/reference batch graph rows, including cancellation during staged graph publication and exact acknowledgment/history linkage. Preserve request-capture failures before a captured request exists. Audit wider sidecar/library/training/import boundaries and post-profile-success failures. Continue current F22 and all remaining F01-F22/WP0-WP6 software/conformance requirements. Original SMPTS and independent laboratory studies remain external. Run a distinct full unittest suite before completed handoff. The goal remains active. Reuse the same worker/reviewer pool and preserve protected project data and immutable receipts.

## F19 QC project and snapshot history (12 October)

Explicitly retained recipe QC and its exact analytical-revision acknowledgments now survive private project snapshot restoration. Ownership uses the selected master filename, isolating projects sharing a directory; unassigned workspace QC is not swept into arbitrary projects. Distinct layout views, exact bytes and multiple recorded account/time events survive original archive/checkpoint store removal and a second snapshot/restore. Strict bounded inventory and exact object linkage reject malformed, orphan, incomplete or omitted evidence before complete publication. Actual restored MainWindow and native/reference Qt QC actions preserve analysis, annotation drafts and graph records. Historical loading never rechecks original artifacts or approves seeds.

Root verification: 107 distinct methods pass across two disjoint terminal runs (109.014 seconds), zero skips and 17 new methods. An earlier interrupted unrelated classifier setup and test cleanup error are retained; the project-only desktop fixture and corrected cleanup pass. One narrow automatic Windows test-path rerun completed with runtime guards unchanged. The same reviewer approves the supplied seven-file delta/freeze without independent I/O, tests, whole-file or visual inspection. Root inspected two actual QC dialogs at 860x650 with Segoe UI9; no all-DPI claim. 438 prior files and four fixed baselines match; five supersessions are explicit. 185 test files are registered; all 40 coarse operation statuses remain unchanged.

Evidence: `artifacts/f19-qc-transport-gate-v1.json`, `artifacts/f19-qc-transport-frozen-hashes-v1.json`, `artifacts/f19-qc-transport-projection-review-v1.json` and `artifacts/f19-qc-transport-root-visual-qa-v1.json`. Existing project-edit authorization was used with zero renewed human approval requests. All test and capture processes are terminal. QC and acknowledgment retention are separate publications; filesystem races and cross-file atomicity are outside this guarantee.

Preserve whole-recipe prewrapper failures and cancellation with explicit unknown facts, then unavailable/reference batch graph QC and acknowledgment records. Audit wider sidecar/library/training/import boundaries and post-profile-success failures. Continue current F22 and all remaining F01-F22/WP0-WP6 software/conformance requirements. Original SMPTS and independent laboratory studies remain external. Run a distinct full unittest suite before completed handoff. The goal remains active. Reuse the same worker/reviewer pool and preserve protected project data and immutable receipts.

## F19 shared recipe QC (12 October)

Native and paper recipe exports now include a verified recorded QC sidecar. It preserves actual executor, recipe/source/frame identity and ordered node outcomes; whole archive verification also checks unrequested arrays. Selected reference-stage QC keeps declared execution separate from captured checkpoint inspection, including missing or unavailable evidence. The shared Qt viewer exports/reopens QC and records acknowledgment against its exact revision. Historical loading does not recheck original artifacts. Measurement eligibility and scientific validity remain unknown; recorded acknowledgment does not approve seeds.

Root verification: 79 distinct methods pass across two disjoint terminal runs (189.919 seconds), zero skips and 14 new methods. Earlier failed candidates and the reproduced Windows strict fixture-path failures are retained. One narrow automatic test-path rerun completed; runtime guards remain unchanged. This is not a single green first-run or full-suite claim. The same reviewer approves the supplied six-file delta/freeze without independent I/O, tests, whole-file or visual inspection. 437 prior files and four fixed baselines match; three supersessions are explicit. 184 test files are registered; all 40 coarse operation statuses remain unchanged.

Evidence: `artifacts/f19-method-qc-gate-v1.json`, `artifacts/f19-method-qc-frozen-hashes-v2.json` and `artifacts/f19-method-qc-projection-review-v1.json`. Existing project-edit authorization was used with zero renewed human approval requests. All test and capture processes are terminal. Project/snapshot QC transport and whole-recipe terminal records remain open.

Integrate shared recipe QC and exact acknowledgments into project/snapshot evidence and batch records for unavailable/reference graphs. Preserve whole-recipe prewrapper failures and cancellation with explicit unknown facts. Audit wider sidecar/library/training/import boundaries and post-profile-success failures. Continue current F22 and all remaining F01-F22/WP0-WP6 software/conformance requirements. Original SMPTS and independent laboratory studies remain external. Run a distinct full unittest suite before completed handoff. The goal remains active. Reuse the same worker/reviewer pool and preserve protected project data and immutable receipts.

## F19 terminal snapshot transport (12 October)

Project snapshots now transport saved native execution/preflight terminal records and their recorded acknowledgment events. Exact bytes and historical paths survive original workspace removal and a second snapshot/restore. Evidence restores privately beside the selected working project; Qt reopening displays recorded account/time without installing historical predictions or approving seeds. Bounded strict validation rejects malformed, conflicting, orphan and incomplete evidence and omitted object mappings before complete publication. Legacy snapshot-v1 remains supported.

Root verification: all 135 distinct targeted methods have successful frozen-body executions across two runs (103.022 seconds), zero skips and 18 new methods. Initial 39 Windows strict fixture-path errors and an interrupted narrow rerun are retained; this is not a single green first-run or full-suite claim. The same reviewer approves the supplied four-file delta/freeze without independent I/O, tests, whole-file or visual inspection. 436 prior files and four fixed baselines match; two supersessions are explicit. 183 test files are registered, and all 40 coarse operation statuses remain unchanged.

Evidence: `artifacts/f19-terminal-snapshot-gate-v1.json`, `artifacts/f19-terminal-snapshot-frozen-hashes-v1.json` and `artifacts/f19-terminal-snapshot-projection-review-v1.json`. Existing project-edit authorization was used with zero renewed approval requests. Two narrow Windows test-path reruns used automatic approval review; the first was interrupted, and the second completed. Runtime path guards remain unchanged; all tests are terminal. Filesystem races, multi-file atomicity and completion-marker write failures remain outside the tested guarantee.

Integrate exact native/reference/paper recipe outcomes and QC/export/load/acknowledgment round trips. Audit wider sidecar/library/training/import boundaries and failures after successful profile application. Continue current F22 and all remaining F01-F22/WP0-WP6 software/conformance requirements. Original SMPTS and independent laboratory studies remain external. Run a distinct full unittest suite before completed handoff. The goal remains active. Reuse the same worker/reviewer pool and preserve protected project data and immutable receipts.

## F19 desktop terminal boundaries and history reopening (12 October)

Desktop source guards, project/profile loading and installation, and mask import rejections now preserve terminal evidence at the actual selected-file boundary. Rejected profile mutation rolls back before project session clearing; existing drafts and replacement tasks survive the tested failures. Saved native execution/preflight history reopens through bounded metadata reads or an explicit JSON selection. Malformed and conflicting records produce diagnostics; incomplete inventories are withheld. Historical acknowledgment/export stays separate from current analysis and seed approval, including after original sources are deleted.

Root verification: 109 focused tests pass in 59.537 seconds, with zero skips and 16 new methods. The earlier 31-method focus is retained separately. The same reviewer approves the supplied six-file delta/freeze without independent I/O, tests, whole-file or visual inspection. 432 prior files and four fixed baselines match; four supersessions are explicit. 182 test files are registered, and all 40 coarse operation statuses remain unchanged.

Evidence: `artifacts/f19-desktop-history-gate-v1.json`, `artifacts/f19-desktop-history-frozen-hashes-v1.json` and `artifacts/f19-desktop-history-projection-review-v1.json`. Existing project-edit authorization was used, with zero renewed approval requests or sandbox escalations. All tests are terminal. Filesystem race resistance and failures after successful profile application remain outside the tested rollback guarantee; no visual/all-DPI or full-suite claim is made.

Implement terminal project/snapshot transport, followed by exact native/reference/paper recipe outcomes and QC/export/load/acknowledgment round trips. Audit wider sidecar/library/training/import boundaries and failures after successful profile application. Continue current F22 and all remaining F01-F22/WP0-WP6 software/conformance requirements. Original SMPTS and independent laboratory studies remain external. Run a distinct full unittest suite before completed handoff. The goal remains active. Reuse the same worker/reviewer pool and preserve protected project data and immutable receipts.

## F19 native preflight terminal records (12 October)

Malformed requests and scoped batch/desktop preparation failures now preserve immutable terminal evidence before analysis starts. Records identify the actual selected project or image, preparation phase, exception and observed file changes. Uncaptured request, image, calibration, prediction and backend facts remain unknown. Each incomplete attempt owns its acknowledgment; project records cannot install image state or approve seeds. Batch context errors preserve original exception/traceback, and recording failure reports both causes. Qt preparation failures retain dirty work and in-memory evidence even when disk persistence fails.

Root verification: all105 distinct targeted methods have successful frozen-body executions across two runs (312.741seconds), zero skips and18 new methods. The initial15 Windows strict fixture-path errors and earlier two test-authoring errors are retained; this is not a single green first-run or full-suite claim. The same reviewer approves the supplied six-file delta/freeze with no independent I/O, tests, whole-file or visual inspection.430 prior files and four fixed baselines match; four supersessions are explicit.181 test files are registered; all40 coarse operation statuses remain unchanged.

Evidence: `artifacts/f19-preflight-gate-v1.json`, `artifacts/f19-preflight-frozen-hashes-v1.json` and `artifacts/f19-preflight-projection-review-v1.json`. Ordinary project edits used existing authorization with zero renewed approval requests. One narrow Windows test-path exception was automatically accepted. Runtime path guards remain intact; all tests are terminal.

Preserve earlier desktop source-guard, project-open/import and settings boundaries with honest terminal evidence. Then implement terminal history reopening and project/snapshot transport, followed by exact native/reference/paper recipe outcome and QC/export/load/acknowledgment round trips. Continue current F22 and all remaining F01-F22/WP0-WP6 software/conformance requirements. Original SMPTS and independent laboratory studies remain external. Run a distinct full unittest suite before completed handoff. The goal remains active. Reuse the same worker/reviewer pool and preserve protected project data and immutable receipts.

## F19 native execution terminal records (12 October)

Shared desktop/batch execution now preserves failed, cancelled, rejected and unavailable runs as immutable terminal JSON. Supplied request/source/artifact facts and observed completion changes bind outcome identity; uncompleted calibration and prediction stay unknown. Stop after calculation still preserves a terminal record. Stale worker signals preserve history without interfering with a replacement run, including a replacement at the same pipeline revision. Native Qt acknowledgment and export are separate from seed approval. Batch integrity/cancellation/device failures block companions; an ordinary computation failure attempts independently guarded companions.

Root verification: all 160 distinct targeted methods have successful executions across three documented runs (357.899 seconds), zero skips and 16 new methods. The initial 31 sandbox fixture errors and two stale schema fixture failures are retained; this is not a single green first-run or full-suite claim. The schema fixtures now match current schema5 and an actual schema3 field inventory; runtime validators remain unchanged. The same reviewer approves the supplied seven-file delta/freeze with no independent I/O, tests, whole-file or visual inspection. 427 prior files and four fixed baselines match; five supersessions are explicit. 180 test files are registered, and all 40 coarse operation statuses remain unchanged.

Evidence: `artifacts/f19-run-outcomes-gate-v1.json`, `artifacts/f19-run-outcomes-frozen-hashes-v2.json` and `artifacts/f19-run-outcomes-projection-review-v1.json`. Ordinary project edits used existing authorization with zero renewed approval requests. Two narrow Windows test-path exceptions were automatically accepted. All tests are terminal.

Preserve pre-wrapper desktop request-capture and batch project/reference/library rejections with only facts successfully captured. Then implement terminal history reopening and project/snapshot transport, followed by exact native/reference/paper recipe outcome and QC/export/load/acknowledgment round trips. Continue current F22 and all remaining F01-F22/WP0-WP6 software/conformance requirements. Original SMPTS and independent laboratory studies remain external. Run a distinct full unittest suite before completed handoff. The goal remains active. Reuse the existing worker/reviewer pool and preserve protected project data and immutable receipts.

## F19 native backend outcome records (12 October)

Revision-v7 preserves producer-captured procedural, U-Net and StarDist outcomes. Reports distinguish available empty/nonempty predictions, missing checkpoint/foreground, attempted inference/decoder failure and disabled nodes. Each available branch is checked against its actual stored prediction facts before export, including branches that were not selected. Diagnostics remain visible while stable outcome facts bind review identity. Qt review/comparison and pipeline status expose a failed branch alongside any selected fallback. Legacy missing execution evidence remains unknown.

Root verification: 160 distinct focused tests pass in 121.014 seconds, with zero skips and 11 new methods. The same reviewer approves the supplied eight-file delta/freeze without independent I/O, tests, whole-file or visual inspection. 424 prior files and four fixed baselines remain unchanged; six supersessions are explicit. 179 test files are registered, and all 40 coarse operation statuses remain unchanged. Cache success/failure/recovery/missing/disabled transitions, real Qt/batch delivery, actual invalid checkpoint loading, prior model relocation/crops/reviews and cancellation/fatal-error propagation pass.

Evidence: `artifacts/f19-backend-outcomes-gate-v1.json`, `artifacts/f19-backend-outcomes-frozen-hashes-v1.json` and `artifacts/f19-backend-outcomes-projection-review-v1.json`. Ordinary project edits used existing authorization. One narrow Windows native-test temporary-path exception was automatically accepted; runtime path guards remain unchanged. All tests are terminal.

Preserve whole-analysis failure/cancellation outcomes through shared desktop/batch run records, then integrate exact native/reference/paper recipe outcome and QC/export/load round trips and failure acknowledgment. Retain explicit unknown facts when no calibration or prediction completed. Continue current F22 and every remaining F01-F22/WP0-WP6 software/conformance requirement. Original SMPTS and independent laboratory studies remain external. Run a distinct full unittest suite before completed handoff. The goal remains active. Reuse the same worker/reviewer pool and preserve protected project data and immutable receipts.

## F19 portable result and review identity (12 October)

Revision-v6 binds recognized native artifact selections to verified content while retaining actual paths and raw executable settings in the report. Current files, availability and producer correspondence are rechecked before export. Native procedural/trait, U-Net and StarDist desktop/batch results retain their complete identity after verified snapshot/restore and deletion of originals. Disabled selected learned bytes remain format-unproven; unhashable or legacy selections remain opaque. Existing approvals never migrate between revision contracts.

Review history restores into the new project’s private directory. Shared workspace evidence takes precedence; malformed, incompatible or unreadable primary evidence cannot revive an archived approval. Snapshot inventory/path/hash guards and bounded review reads remain in force. Actual crop records, geometry and probabilities round-trip unchanged.

Root verification:192 distinct focused tests PASS122.192s,zero skips,11 new methods. The same reviewer approves the supplied ten-file delta/freeze without independent I/O/tests or whole-file/visual inspection.420 prior files and four fixed baselines remain unchanged; eight supersessions are explicit.178 test files are registered; all40 coarse statuses are unchanged. Evidence: `artifacts/f19-portable-result-identity-gate-v1.json`, `artifacts/f19-portable-result-identity-frozen-hashes-v1.json`, `artifacts/f19-portable-result-identity-projection-review-v1.json`. One narrow Windows native-test path exception was automatically accepted; routine edits needed no renewed approval. All tests are terminal.

Continue finite F19 backend/QC failed, unavailable, valid-empty and not-run coverage, then current F22 and every remaining F01-F22/WP0-WP6 software/conformance requirement. Original SMPTS and independent laboratory studies remain external. The goal stays active; distinct full unittest remains required before completed handoff. Reuse the same worker/reviewer pool, preserve protected project data and immutable receipts.


## F19 shared native desktop/batch results (12 October)

Native analysis now captures aligned annotation/input facts with the result. Desktop and saved-project batch share revision-v5 and exact saved-review lookup; raw project/archive and execution evidence remain in a separate report field. Missing, malformed and incompatible reviews remain unresolved with distinct diagnostics. Per-image biological contexts survive desktop load/save, and a missing species stays unknown. Existing reports and historical decisions are preserved; old approvals do not migrate into the new revision contract.

Root verification:151 distinct focused tests pass in73.439s,zero skips,seven new methods. Actual real decoding/calibration/CUDA trait features, Qt worker delivery and report producers agree for nonzero crop and nonidentity transforms. Independent geometry/scale expectations, bidirectional review lookup, layout-only revision stability, annotation/context/model-byte invalidation, missing-model abstention and historical bytes pass. The same reviewer approves the supplied ten-file delta/freeze without independent I/O/tests or whole-file/visual inspection.418 prior frozen files and four fixed baselines are unchanged; seven supersessions are explicit.177 test files are registered; all40 coarse statuses remain unchanged.

Evidence: `artifacts/f19-authoritative-parity-gate-v1.json`, `artifacts/f19-authoritative-parity-frozen-hashes-v1.json` and `artifacts/f19-authoritative-parity-projection-review-v1.json`. Same-location native parity is scoped acceptance; relocated artifact-path identity and full backend/QC failed/unavailable coverage remain open. Continue remaining F19, then F22 and all remaining F01-F22/WP0-WP6 requirements. Original SMPTS and independent laboratory studies remain external; distinct full unittest is required before completed handoff. Goal active; all tests terminal. Routine project edits remain authorized with no renewed permission request. One narrow native-test Windows strict-path escalation was automatically accepted; runtime path guards remain unchanged.

## F18/F19 actual trait predictions in review (12 October)

The native review table now displays stored supplied trait labels or their actual abstention reasons instead of a fixed “Abstained” value. Tooltips retain definition/evidence, model identity and predictive-probability meaning without refitting or changing annotations. Summary counts match the stored rows. A learned-only result no longer requires unrelated procedural foreground metadata; missing foreground still cannot create procedural evidence.

Root verification:79 focused tests PASS15.182s,zero skips,three new real Qt/CUDA methods. Assigned/abstained and missing/invalid/support states, exact threshold ties, saved-review staleness, historical bytes and unchanged probabilities/export pass. Two actual960x640 Qt renders are root-inspected with a project-runtime QA font; the unusable sandbox default-font capture is retained, and default-font/all-DPI guarantees are not claimed. The same reviewer approves the supplied three-file delta/freeze without independent I/O/tests/visual or whole-file inspection.422 prior frozen files and four fixed baselines are unchanged; two supersessions are explicit.176 test files are registered; all40 coarse statuses remain unchanged.

Evidence: `artifacts/f18-trait-review-ui-gate-v1.json`, `artifacts/f18-trait-review-ui-frozen-hashes-v1.json`, `artifacts/f18-trait-review-ui-projection-review-v1.json` and `artifacts/f18-trait-review-ui-visual-review-v1.json`. Next audit finite F19 backend/QC and failed/unavailable states, authoritative desktop/batch crop/calibration/dependency/review records, then current F22 clauses and remaining F01-F22/WP0-WP6 requirements. The goal stays active; original SMPTS and independent laboratory studies remain external. Distinct full unittest remains required before completed handoff. Tests/captures are terminal. Project edits remain authorized; this checkpoint used the workspace sandbox with no escalation or renewed approval.

## F18/F19 portable trait models and finite acceptance (12 October)

Project snapshots now carry every explicit frozen trait model selection, including disabled nodes, as bounded validated bytes with raw and canonical SHA identities. Restore verifies the selection and inventory before publishing a working project and relocates the path. `None` stays absent; incomplete legacy selected-model snapshots reject. Original project/model bytes remain unchanged.

Two final gates run116 distinct tests in82.714s:115 pass and one existing case-sensitive-filesystem test is skipped on Windows. Actual relocated real-decoder/CUDA batch and Qt worker inference preserves predictions, supplied vocabulary/audit, threshold behavior and reviewed known/unknown/hilum traits after originals are removed. The same reviewer approves the supplied four-file delta/freeze, without independent I/O/tests or whole-file inspection.420 prior frozen files and four fixed baselines remain unchanged; two supersessions and nine new test methods are recorded.175 test files are registered.

Evidence: `artifacts/f18-trait-snapshot-gate-v1.json`, `artifacts/f18-trait-snapshot-frozen-hashes-v1.json`, `artifacts/f18-trait-snapshot-projection-review-v1.json` and `artifacts/f18-trait-acceptance-clause-audit-v1.json`. F18 software predicates are reconciled for the declared bounded, guarded contract; independent expert definitions/truth and locked biological agreement/reliability/abstention remain L04. F19 whole-route backend/QC/export acceptance still needs its finite audit. All40 coarse statuses remain unchanged.

Next audit F19 authoritative desktop/batch records, crop/calibration/dependency/review lineage and failed/unavailable states, then current F22 graph/source/Qt clauses and remaining F01-F22/WP0-WP6 requirements. The goal stays active; distinct full unittest remains required before completed handoff. Project edits remain authorized. Two narrow native-test Windows strict-path access escalations were automatically accepted; ordinary project edits required no renewed approval. All tests are terminal. Reuse the same worker/reviewer pool and preserve protected project data and immutable historical receipts.

## F16 guarded pooling and acceptance reconciliation (12 October)

Matching Gaussian updates now rescue tested intermediate precision, mean-product and predictive-variance overflow using the existing conditioned formula, with no added ridge or clipping. Successful legacy arithmetic stays exact. Positive subnormal variances remain available; lost normalized/restored information, singular covariance and unrepresentable outputs or support increments retain the exact prior or mixed physical shell. Mixed pooling keeps its existing ridge and physical block. Native export records the actual result without refitting.

The final numerical/compatibility gate passes114 tests in10.573s; a separate acceptance-clause gate passes36 tests in9.968s,150 distinct checks and zero skips. The same reviewer approves the supplied three-file delta/freeze without independent I/O, tests or whole-file inspection.419 prior frozen files and four fixed baselines are unchanged; two supersessions and seven new test methods are explicit.174 test files are registered. Evidence: `artifacts/f16-pooling-overflow-gate-v1.json`, `artifacts/f16-pooling-overflow-frozen-hashes-v1.json`, `artifacts/f16-pooling-overflow-projection-review-v1.json` and `artifacts/f16-acceptance-clause-audit-v1.json`.

F16 software acceptance now has named witnesses for definitions, honest measurement frames, explicit visibility, units/uncertainty, eligibility and withholding physical-accuracy claims. Source association, scoped native generation binding and diagnostic persistence are implemented; stale planned predicates are reconciled. The guarded numerical contract does not promise every mathematically representable IEEE posterior. The remaining F16 gate is the independent measurement/coverage study (L03). Unknown historical generation and exact historical display transforms remain unproven; no scientific validity is inferred.

Continue a finite clause-by-clause F18 acceptance audit, then all remaining F01-F22/WP0-WP6 software/conformance requirements. All40 coarse statuses remain unchanged and the goal stays active. Original SMPTS and laboratory truth remain external. Focused tests run during implementation; a distinct full unittest suite is required before completed handoff. All tests are terminal. Project edits remain authorized; all checks and edits used the workspace sandbox with zero escalations or renewed approval. Reuse the same worker/reviewer pool and preserve protected project data and immutable historical receipts.

## F18 native trait training and activation (12 October)

The desktop Learning menu now trains a supplied-definition object-level classifier from explicit permitted train/calibration evidence. Compact supplied features retain unverified extractor lineage; reviewed reference seeds use the same twenty-column CUDA extractor with recorded transforms and immutable hash-matching image/archive snapshots. Known generation mismatches, actual source/mask split aliases, unsupported pixels and incomplete/border outlines reject; missing generation history stays unknown. Controls affect fitting, calibration or abstention.

Cancellation runs between evidence rows, fitting iterations and calibration candidates. Validated bounded frozen JSON is saved atomically; a failure or stop before commit preserves previous bytes. A late stop preserves valid saved output and skips activation. Explicit activation validates the artifact and graph transaction first, then invalidates only classifier/dependents. Serial Qt tasks guard progress/completion by identity; Stop, close and graph changes retain honest outcomes. Existing U-Net/StarDist training remains compatible. Usage and exact manifest fields: [Native seed-trait training](docs/TRAIT_CLASSIFIER_TRAINING.md).

Root verification:164 focused checks PASS71.519s,zero skips,21 new methods. The same reviewer approves the supplied six-file substantive delta/repair projection without independent I/O/tests or whole-file inspection.415 prior frozen files and four fixed baselines are unchanged; three supersessions are explicit.173 test files are registered; all40 coarse statuses remain unchanged. Evidence: `artifacts/f18-trait-training-gate-v1.json`, `artifacts/f18-trait-training-frozen-hashes-v1.json`, `artifacts/f18-trait-training-projection-review-v1.json`. Tests and authorized edits ran in the workspace sandbox without escalation or renewed approval.

Continue remaining F16 numeric and F01-F22/WP0-WP6 software/conformance domains. Expert definitions/truth, independent locked biological acceptance, broader resource domains, original SMPTS and L01-L04 remain open. The goal stays active; focused tests run during implementation and a distinct full unittest suite remains required before completed handoff. All test processes are terminal. Reuse the same worker/reviewer pool; never access the protected project master, rerun historical one-shot drivers or overwrite fixed baselines.

## F18 native trait inference (12 October)

The opt-in Object-level seed traits node now applies a frozen supplied-definition model to the same enabled instance branch used by result export. Features remain on CUDA. Model-content changes reuse features; threshold changes reuse features and scores. Missing/invalid models, incompatible frames/support and crop-boundary instances have explicit abstention. Typed graph dependencies permit alternative disabled instance calculations while preserving required wiring and invalidation. Settings version22 supports narrow pre-classifier migration with the new node disabled.

Stored predictions, probabilities, definitions, model/audit identity and frame association now bind report, review and crop revisions. Desktop workers and saved-project batch analysis share settings. Export uses the stored snapshot without fitting or downloading new analysis rasters. Bounded reads remain bounded if a model grows during loading; runtime cache signatures include tensor headers/storage. General unsafe version-bypassing content writes are outside this guarantee.

Root verification:206 focused checks pass in66.236s,zero skips,18 new methods. The same reviewer approves the supplied twelve-file V2 delta projection without independent I/O/tests.406 prior frozen files and four fixed baselines are unchanged; nine original and two review-repair supersessions are explicit.172 test files are registered; all40 coarse operation statuses are unchanged. Evidence: `artifacts/f18-trait-native-gate-v2.json`, `artifacts/f18-trait-native-frozen-hashes-v2.json`, `artifacts/f18-trait-native-projection-review-v1.json`. All checks ran in the workspace sandbox without escalation; project edit authorization remains in force.

Next implement native evidence/training, atomic frozen-model saving, activation and cancellation, then remaining F16 numeric and F01-F22/WP0-WP6 software/conformance domains. Expert definitions/adjudicated truth and independent biological acceptance remain required; original SMPTS and L01-L04 remain external. The goal stays active. Relevant focused tests run during implementation; a distinct full unittest suite remains required before completed handoff. All test processes are terminal. Reuse the same worker/reviewer pool; never access the protected project master, rerun historical one-shot drivers or overwrite fixed baselines.

## F18 CUDA trait-classifier core (11 October)

A standalone object-level classifier core now extracts twenty colour/visible-pixel-geometry features on CUDA, trains from permitted supplied definitions/truth, calibrates on isolated groups, freezes finite non-executable JSON and predicts with actual confidence-threshold abstention. Training statistics and weights use training rows alone; denied rows never enter fitting/calibration. Example/group/physical identities reject split leakage, contradictory physical views reject, and schema guards stop incompatible feature meanings before computation. Models bind definitions, feature order, normalization, weights, evidence hashes, audits and development reliability/coverage diagnostics. Held-out evaluation never refits; nonrepresentable calibrated diagnostics reject. Pixel geometry remains resolution-dependent and does not infer physical scale.

Root verification:18 focused checks PASS3.578s,zero skips,17 new methods. Synthetic classification, independent NumPy feature/softmax oracles, actual CUDA retention, permission/split isolation, calibration/threshold effects, frozen JSON/hash/detachment, invalid inputs, schema and numeric boundaries pass. V3 supplied-projection review approves three files without independent I/O/tests.412 preceding files and four fixed baselines remain unchanged. V1/V2 receipts are preserved; schema and diagnostic guard supersessions are explicit.171 test files are registered. F18.2 functionality advances from gap to partial_native for this core; the other39 operation statuses are unchanged. Evidence: `artifacts/f18-trait-classifier-gate-v3.json`, `artifacts/f18-trait-classifier-frozen-hashes-v3.json`, `artifacts/f18-trait-classifier-projection-review-v1.json`. All classifier checks ran in the workspace sandbox without an approval escalation; project edits remain authorized.

Next integrate explicit native training/model activation, per-instance feature/cache nodes and authoritative report/crop consumers. Current report rows still default to classifier abstention until that integration exists. Expert biological definitions/adjudicated truth, independent scientific acceptance, remaining F16 numeric domains and other F01-F22/WP0-WP6 software/conformance remain open. Original SMPTS and L01-L04 remain external. The goal stays active; relevant focused tests run during implementation and distinct full unittest is required before completed handoff. All tests are terminal. Reuse the same worker/reviewer pool; never access the protected project master, rerun historical one-shot drivers or overwrite fixed baselines.

## F16 runtime shape-model export (11 October)

Per-analysis results now persist the actual runtime shape model and its component statistics/support, source mode, in-sample state and update state/reason/method/conditioning scales. The snapshot binds its own recorded summary to the exact exported reviewed measurements before revision hashing. Known stale summaries reject; absent legacy history stays unknown. Update outcomes remain distinct from scoring and physical-fit availability; retained covariance does not imply successful pooling. Export neither refits nor publishes library evidence and adds no CUDA raster transfer.

Root verification:129 focused tests PASS19.365s,zero skips,14 new methods. Actual file analysis/cache and mask-derived local/ordinary/mixed/conditioned/unavailable outcomes, JSON detachment, stale review/crop rejection and finite diagnostic contracts are exercised. Supplied-projection review approves three files without independent I/O/tests.409 preceding files and four fixed baselines remain unchanged;one results.py supersession is explicit.170 test files are registered;40 coarse operation statuses remain unchanged. Evidence: `artifacts/f16-shape-model-export-gate-v1.json`, `artifacts/f16-shape-model-export-frozen-hashes-v1.json`, `artifacts/f16-shape-model-export-projection-review-v1.json`. The initial new crop fixture's frozen-field assignment was repaired before the final gate. Project edits remain authorized; automatic review accepted narrow native-test Windows temporary-path access.

Next implement actual F18.2 classifier software and native consumers, then remaining F16 numeric and F01-F22/WP0-WP6 software/conformance work. Original SMPTS and L01-L04 remain external. Scientific validity and full-goal completion are not claimed. The goal stays active; focused tests run during implementation, with distinct full unittest required before completed handoff. All tests are terminal. Reuse the same worker/reviewer pool. Never access the protected project master, rerun historical one-shot drivers or overwrite fixed baselines.

## F16 native annotation-generation canvas binding (11 October)

Native seed-label events now bind to the corrected Qt canvas snapshot displayed when editing began. Compact source/image digests, shape, affine and runtime epoch reject stale events across image/result/display/history transitions, including unknown-source edits and erasure. First drawing on an empty verified canvas records its source; partial edits to nonempty legacy labels never upgrade history. Trait-only edits preserve outline history.

Undo/redo/revert, supported recorded imports, Apply/autosave and two-image bulk drafts preserve labels and optional generation source together through existing archive v7. External origin text does not create proof. The runtime snapshot digest/transform is not a persisted generation audit, and current-buffer immutability, material/trait provenance and physical/scientific validity are not claimed.

Root verification:113 tests PASS44.909s,zero skips,17 new methods. Supplied-projection review approves four files without independent I/O/tests.406 preceding files and four fixed baselines remain unchanged;two supersessions are explicit.169 test files are registered;40 coarse operation statuses remain unchanged. Evidence: `artifacts/f16-annotation-generation-gate-v1.json`, `artifacts/f16-annotation-generation-frozen-hashes-v1.json`, `artifacts/f16-annotation-generation-projection-review-v1.json`. Initial test fixture-option and no-op-transition assertions were repaired before the final gate. Automatic review accepted narrow native-test temporary-path access; project edits remain authorized without renewed human approval.

Next implement persistent shape update diagnostics and remaining numeric domains, then remaining F01-F22/WP0-WP6 software/conformance work, including F18.2 classifier functionality. The goal stays active; original SMPTS and L01-L04 remain external. Relevant tests run during implementation; distinct full unittest remains required before completed handoff. All tests are terminal. Reuse the same worker/reviewer pool. Never access the protected project master, rerun historical one-shot drivers or overwrite fixed baselines.

## F16 reference-source identity preservation (11 October)

Reference archives v7 now retain verified source-file association separately from optional recorded annotation-snapshot source identity. Versions1–6 retain archive association while intentionally ignoring the new generation field; missing generation history stays unknown. Known mismatches stop save/load before raster use, file analysis before decode, and direct array analysis before CUDA setup.

The desktop controller, worker and saved-project batch path preserve both identities. Reviewed context/cache/export carries independent source states; metadata changes invalidate measurement dependents and export review. Aggregate verified means every recorded fact matches, not that missing annotation generation is proven. Direct analyze_image validates supplied source facts without proving file decode or annotation generation. Export preserves stored unverified history.

Root verification:185 methods run70.831s;184 passed,one POSIX-only case-sensitive-filesystem test skipped on Windows;15 new methods. Supplied-projection review approves nine files without independent I/O/tests.399 preceding frozen files and four fixed baselines remain unchanged;seven supersessions are explicit.168 test files are registered;40 coarse operation statuses remain unchanged. Evidence: `artifacts/f16-reference-source-gate-v1.json`, `artifacts/f16-reference-source-frozen-hashes-v1.json`, `artifacts/f16-reference-source-projection-review-v1.json`. Initial focused fixture/import errors were repaired before the final gate. Automatic review accepted native-test temporary-path access; project edits remain authorized without renewed human approval.

Next implement annotation-generation producer-to-canvas binding, persistent shape update diagnostics and remaining numeric domains, then remaining F01-F22/WP0-WP6. Independent physical/scientific validity is not claimed. The goal stays active; original SMPTS and L01-L04 remain external. Relevant tests run during implementation; distinct full unittest remains required before completed handoff. All tests are terminal. Reuse the same worker/reviewer pool. Never access `analysis.seedfiddle-project.json`, rerun historical one-shot drivers or overwrite fixed baselines.

## F16 conditioned shape pooling (11 October)

Ordinary anisotropic mask fits now pool successfully. When raw pseudo-inversion would discard covariance directions, a dimensionless coordinate solve preserves the same Gaussian formula with no added ridge or noise. Each inverse is diagonally equilibrated; means are centered at the prior before correction. Successful legacy arithmetic and mixed shape pooling remain unchanged. Actual assembly retains the pooling method and detached coordinate scales, and native Qt shows the actual applied state without refitting.

Tests cover scalar Gaussian and correlated-block oracles, length/area unit congruence, actual masks at uncalibrated/mm/extreme-fit-fallback scales and three shrinkage weights, differing cohorts, identical large offsets, signed cancellation and explicit singular/solver failures. These supersede the previous ordinary-mask withholding in the tested domains. Truly singular/nearly singular correlation and unrepresentable arithmetic still retain exact baseline/support. Metadata remains ephemeral; persistence/export and independent physical/scientific validity are not claimed.

Root verification:216 compatibility tests PASS23.170s plus a separate24-test native-export gate PASS26.301s,240 distinct checks,zero skips,11 new methods. Supplied-projection review approves six files without independent I/O/tests.400 preceding frozen files and four fixed baselines remain unchanged;five supersessions are explicit.166 test files are registered;40 coarse operation statuses remain unchanged. Evidence: `artifacts/f16-conditioned-pooling-gate-v1.json`, `artifacts/f16-conditioned-pooling-frozen-hashes-v1.json`, `artifacts/f16-conditioned-pooling-projection-review-v1.json`. The initial57-test gate had one exact-zero assertion, repaired with an explicit four-epsilon rounding bound. Automatic review accepted native-test temporary-path access; project edits remain authorized without renewed human approval.

Next implement physical/reference generation source traceability and persistent update diagnostics, then remaining numeric and F01-F22/WP0-WP6 domains. The goal stays active; original SMPTS and L01-L04 remain external. Relevant tests run during implementation; distinct full unittest remains required before completed handoff. All tests are terminal. Reuse the same worker/reviewer pool. Never access `analysis.seedfiddle-project.json`, rerun historical one-shot drivers or overwrite fixed baselines.

## F16 precision-update availability (11 October)

Precision pooling now reports its availability separately from fitting and scoring. Failed matching updates retain the exact prior; failed mixed updates retain the exact physical shell and its support. Unused local evidence is not counted. Native model tooltips explain the reason and retained source. Successful supported arithmetic, the existing mixed shape ridge, processing scalar and library contracts remain compatible.

Ill-conditioned ordinary-mask fixtures currently retain their prior rather than successfully pool. Their covariance directions fall below the existing pseudo-inverse cutoff. This verifies honest failure handling, not complete update support; a later conditioned solve is needed. Update diagnostics are ephemeral assembly/Qt facts, with no persistence or export claim.

Root verification:189 compatibility tests PASS18.219s plus a separate24-test native-export gate PASS29.111s,213 distinct checks,zero skips,14 new methods. Supplied-projection review approves six files within this restricted domain, without independent I/O/tests.399 preceding frozen files and four fixed baselines remain unchanged;five supersessions are explicit.165 test files are registered;40 coarse operation statuses remain unchanged. Evidence: `artifacts/f16-precision-update-gate-v1.json`, `artifacts/f16-precision-update-frozen-hashes-v1.json`, `artifacts/f16-precision-update-projection-review-v1.json`. The initial46-test gate exposed two old assumptions, now replaced with explicit fallback/support checks. Automatic review accepted native-test temporary-path access; project edits remain authorized without renewed human approval.

Next complete conditioned pooling and broader update domains, diagnostic persistence where required, and physical/reference source traceability, then remaining F01-F22/WP0-WP6. The goal stays active; original SMPTS and L01-L04 remain external. Relevant tests run during implementation; distinct full unittest remains required before completed handoff. All tests are terminal. Reuse the same worker/reviewer pool. Never access `analysis.seedfiddle-project.json`, rerun historical one-shot drivers or overwrite fixed baselines.

## F16 stored reviewed measurements and source frame (11 October)

Reviewed geometry now snapshots the actual CPU labels and consumed annotations, with strict seed IDs, image/calibration facts and declared transforms. Applied reprojection establishes the corrected frame; original reference-source identity remains explicitly unrecorded. Changes to labels, annotations or context invalidate the measurement cache and its dependents while preserving unrelated upstream products. Native tooltips use the verified frame from the stored value bundle.

Optional reviewed export namespace1 carries separate pixel and calibration-derived scaled bundles, their definitions, sensitivity, eligibility and reasons. Every exported fact binds result revision-v4; outer report3 and prediction physical/scientific withholding remain. Known stored source/image/transform/calibration mismatches require reanalysis; absent legacy facts stay unknown. This does not validate seed-plane applicability or independent physical accuracy.

Final relevant gate:199 tests PASS40.537s,zero skips,14 new methods, including actual producer/cache, Qt, saved-project and native export consumers. Supplied-projection review approves eight-file v2 without independent I/O/tests.396 other frozen files and four fixed baselines remain unchanged;five original supersessions and two v1 projection repairs are explicit.164 test files are registered;40 coarse operation statuses remain unchanged. Evidence: `artifacts/f16-reviewed-lineage-gate-v2.json`, `artifacts/f16-reviewed-lineage-frozen-hashes-v2.json`, `artifacts/f16-reviewed-lineage-projection-review-v2.json`. Earlier gates overlap and retain their execution-time scope. Automatic review accepted the native-test temporary-path escalation; project edits remain authorized without renewed human approval.

Next audit broader precision-update and physical/reference traceability domains, then remaining F01-F22/WP0-WP6. The full goal remains active; original SMPTS and L01-L04 remain external. Relevant tests run during implementation; distinct full unittest remains required before completed handoff. All tests are terminal. Reuse the same worker/reviewer pool. Never access `analysis.seedfiddle-project.json`, rerun historical one-shot drivers or overwrite fixed baselines.

## F16 candidate scoring availability (11 October)

Candidate comparisons now expose nullable score/distance/verdict fields and explicit invalid-input, invalid-covariance or numeric-range reasons. Failed arithmetic cannot claim a perfect fit or an out-of-family verdict. Selected covariance validation is bounded; excluded coordinates do not disable dimensionless scoring. Ordinary scoring arithmetic and legacy assessment construction are preserved. Pose hypotheses reuse one snapshot of a supplied iterable.

The relevant gate passes 101 tests in 11.600s, zero skips, including 12 new methods and related fit/library/annotation/Qt checks. Supplied-projection review approves two files without independent execution. 399 other frozen files and four baselines remain unchanged; one source supersession is recorded. The register pins 163 test files. This covers the scoring API/model-hypothesis path; a new native ranking workflow is not claimed. Evidence: `artifacts/f16-scoring-gate-v1.json`, `artifacts/f16-scoring-frozen-hashes-v1.json`, `artifacts/f16-scoring-projection-review-v1.json`.

Next implement stored source-image-frame/reviewed-export lineage and audit broader precision domains, then remaining F01-F22/WP0-WP6. The full goal remains active; original SMPTS and L01-L04 remain external. Run relevant tests during implementation and distinct full unittest before completed handoff. Tests are terminal. Reuse the same worker/reviewer pool; project edits remain authorized. Never access `analysis.seedfiddle-project.json`, rerun historical one-shot drivers or overwrite fixed baselines.

## F16 physical fit numeric availability (11 October)

Finite calibrated population and local fits now report `numeric_range` when their physical statistics cannot be represented. Original observations, biological identities and source/calibrated evidence remain retained; dimensionless shape is fitted when representable. Stable grouping cannot silently replace positive RMS evidence with zero. Fits accept compatible parents; successful physical shells clear the computational reason. Retained calibrated counts describe evidence/effective support, not successful physical-fit counts.

The optional component reason survives strict library serialization; legacy digests verify before absent metadata defaults to unknown. The 19-value vector and bank-v3 identity stay compatible. Predictive intervals must be finite. Native Qt model tooltips explain the numeric reason. Broader precision/scoring and actual source-image-frame/reviewed-export lineage remain open.

Final relevant gate: 108 tests PASS in 15.481s, zero skips, including 18 new methods and library/annotation/Qt/shape-guided/pilot consumers. The reviewer approves the supplied six-file v2 projection without independent I/O/tests. 394 other frozen files and four fixed baselines are unchanged; five supersessions are explicit. The register pins 162 test files; all 40 coarse operation statuses remain unchanged. Evidence: `artifacts/f16-fit-availability-gate-v2.json`, `artifacts/f16-fit-availability-frozen-hashes-v2.json`, `artifacts/f16-fit-availability-projection-review-v2.json`. Earlier gates overlap and retain their execution-time scope.

Next: actual source-image-frame and reviewed measurement export lineage, remaining numeric scoring/precision domains, then remaining F01-F22/WP0-WP6. Original SMPTS and L01-L04 remain external inputs. The goal stays active; whole feature/scientific completion is not claimed. Relevant tests run during implementation; a distinct full unittest is required before completed handoff. All test processes are terminal. Reuse the same GPT-6.1 Sol medium worker/reviewer pool. Project edits remain authorized; no renewed human approval. Never rerun historical one-shot drivers, overwrite fixed baselines or access `analysis.seedfiddle-project.json`.

## F16 local model units and descriptor overlays (11 October)

Current pixel fits are dimensionless and have no calibrated support. Mixed library/current updates preserve the physical dimensional block and pool only dimensionless shape. Shrinkage affects shape pooling. Cross-block covariance is zero as an explicit modeling assumption; retained source-count maximum is not a verified union. The processing-diameter scalar and library schemas remain unchanged.

Reviewed body/span/plot tooltips now expose the descriptions from their actual value bundle. Missing/legacy sensitivity omits bars; valid zero and scale-only estimates remain available. Sigma wording is replaced with sensitivity estimates. Source image lineage stays explicitly unknown. Native export schema3 and physical/scientific withholding are preserved.

Final relevant gate:114 tests PASS32.472s,zero skips,13 new methods,including native desktop/saved export and library/annotation/shape-guided/pilot consumers. Supplied-projection review approves5 files without independent I/O/tests.394 preceding files and4 fixed baselines remain unchanged;3 source/test supersessions are recorded.161 test files are registered;40 coarse statuses remain unchanged. Evidence: `artifacts/f16-local-overlay-gate-v1.json`, `artifacts/f16-local-overlay-frozen-hashes-v1.json`, `artifacts/f16-local-overlay-projection-review-v1.json`. Earlier85 local and83 metadata gates retain their execution scopes and overlap.

Next implement explicit extreme population-fit availability and actual source-frame/export lineage, then remaining F01-F22/WP0-WP6. Original SMPTS and L01-L04 remain external inputs. Whole feature/scientific completion is not claimed. Focused tests run during implementation; distinct full unittest remains required before completed handoff. All tests are terminal. Reuse the same GPT-6.1 Sol medium pool. Project edits remain authorized; automatic review accepted the narrow export test temp-file escalation without a human request. Never rerun historical one-shot drivers, overwrite fixed baselines or access `analysis.seedfiddle-project.json`.

## F16 per-view normalization and measurement descriptions (11 October)

Repeated views are normalized before biological-seed averaging, including mixed pixel/mm views. Calibrated physical grouping stays separate. Guarded means/RMS and positive actual spans retain representable values without a physical-unit floor.

Current observed-mask bundles expose immutable value records with calculation definitions, units, input-mask coordinates, visibility and actual sensitivity provenance. Raw/corrected image lineage remains unknown. Missing boundary sensitivity is unavailable; scale-only sensitivity is explicit. Legacy definitions remain unknown. The definition revision is fixed metadata, not a user control. Library vector/schema contracts are unchanged.

The final relevant gate passes83 tests in11.279s,zero skips,including18 new methods and library/annotation/Qt/pilot compatibility. Supplied-projection review approves7 files without independent I/O/tests.390 other prior frozen files and4 fixed baselines are unchanged.159 test files are registered;40 coarse statuses remain unchanged. Evidence: `artifacts/f16-view-description-gate-v1.json`, `artifacts/f16-view-description-frozen-hashes-v1.json`, `artifacts/f16-view-description-projection-review-v1.json`. The earlier80-test gate predates metadata refinements; scopes overlap.

Next: correct local model physical/dimensionless semantics, integrate actual export/image-frame facts and complete numeric population-fit domains, then remaining F01-F22/WP0-WP6. Whole feature and scientific completion are not claimed. Relevant tests run during implementation; distinct full discovery is required before completed handoff. The old interrupted discovery remains preserved. All focused processes are terminal. Reuse the same GPT-6.1 Sol medium pool; ordinary project edits remain authorized.

## Current execution and remaining work (11 October)

The 10 October full discovery was interrupted after 166 completed `ok` lines, with no unittest terminal summary. Its exec handle and both recorded processes are absent. All 403 frozen files and four fixed baselines still match. This is not a full-suite pass; the original ledger/log are retained. Evidence: `artifacts/feature-continuation-full-suite-interruption-v1.json`.

Continue all F01-F22/WP0-WP6 work. Run relevant focused tests during implementation and a distinct full unittest before completed handoff. Next F16 work is per-view unit normalization before repeated-seed aggregation, then measurement definitions/frame/units/uncertainty metadata and numeric population-fit domains. Existing scoped gates retain their actual tested boundaries; original SMPTS source/license/dependencies and L01-L04 remain external inputs.

The reusable pool is two GPT-6.1 Sol medium implementation workers plus the existing GPT-6.1 Sol medium reviewer. Root owns I/O, edits and execution. Routine project edits are already authorized; do not request renewed approval. Never access `analysis.seedfiddle-project.json`, rerun historical one-shot drivers or overwrite fixed baselines. No tests/GPU/captures are live at this reconciliation.

## F16 numeric scale and homogeneous measurement summaries (10 October)

Direct mask measurements now reject invalid, boolean or nonfinite scales and withhold dimensions/uncertainties that overflow or turn a positive value into zero. Ordinary dimensional arithmetic is retained; sequential area division handles an invalid squared-scale intermediate. Pixel-derived ovality remains dimensionless even below the old calibrated-width floor. Complete uncertainty bundles and finite aspect overrides are checked; legacy missing aspect fields retain pixel behavior.

Reviewed summaries keep pixel evidence and successful per-instance mm bundles separately. A mixed conversion result uses pixels for every primary summary value, so statistics/vectors cannot mix units; actual complete eligible mm bundles determine calibrated_count. Invalid and empty calibration have reasons, and numerically unavailable aggregate statistics remain unavailable. Typed model observations preserve their actual calibration group. These numeric facts do not establish physical accuracy or seed-plane applicability.

Final focused/compatibility gate:59 tests PASS in7.164s,zero skips,12 new methods. It includes normal/mixed grouping and JSON, reference libraries, annotations, shape-guided Qt and actual pilot-scale consumers. The earlier59-test gate had one legacy fixture error (nested feature reconstructed as a dictionary); it was repaired. Supplied-projection review approves3 current files without independent I/O/tests.391 prior frozen files and4 fixed baselines are unchanged;2 source supersessions are recorded.157 test files are registered;40 coarse statuses remain unchanged.

Evidence: `artifacts/measurement-scale-gate-v1.json`, `artifacts/measurement-scale-frozen-hashes-v1.json`, `artifacts/measurement-scale-projection-review-v1.json`. This supersedes the pending F16 numeric-scale slice only. Complete definition/source-frame/visibility/uncertainty metadata and extreme population-fit domains remain open, along with L01–L04, original SMPTS and scientific acceptance.

Next run fresh full unittest, then continue F16 metadata/fit-domain and remaining F01-F22/WP0-WP6 implementation. Full goal remains active. Routine project edits stay authorized under Approve for me/auto_review; no renewed human approval. Root owns I/O/edits/tests and reuses existing text workers/reviewer. All focused processes are terminal. Never rerun historical one-shot scripts or overwrite fixed baselines; never inspect/edit/stage `analysis.seedfiddle-project.json`. No manual source/test/config/runtime/artifact-script edits while tests/GPU/captures are LIVE.


All6 register checks PASS1.231s,zero skips; scoped whitespace checks and raw-byte register/157test links pass. Scoped closure is `artifacts/measurement-scale-closure-v1.json`. The one-shot full-suite driver is `artifacts/run-feature-continuation-full-suite-v1.py`; its live/terminal state belongs to `artifacts/feature-continuation-full-suite-status-v1.json` and output to `artifacts/feature-continuation-full-suite-v1.log`. Never restart it or edit project source/tests/docs/config/scripts while it is LIVE; poll its existing exec handle until terminal. Preparation is not test completion.

## Selected neutral controls, calibration cache and report facts (10 October)

The native deskew/colour node now accepts automatic or unique local grid IDs1–24 and source/corrected sampling. Automatic/source preserves historical pixels and gains. Corrected mode samples transformed unbalanced CUDA pixels; invalid/missing selections stay explicitly unavailable with identity gains. Both controls affect calculations, have honest inspector enablement and labelled raw/corrected overlays. Strict settings20→21 migration preserves existing flags, values and topology.

Ruler-only changes reuse guarded upstream geometry; actual source bytes, concrete CUDA device, settings, retained rasters/tensor and compact reference contents govern reuse. Report schema3/revision-v3 bind actual calibration settings, reference/scale facts and selected sampling witnesses. Changed facts stale prior review even when pixels are unchanged. Legacy absent facts remain unknown; physical measurements and seed-plane applicability remain withheld.

Final focused gate:80 tests in97.720s,79passes and one existing skip for the absent complete eleven-image pilot batch. It covers29 new methods, actual card/ruler serialization, pilot cache, current prediction/crop and saved-project/async desktop export. Earlier122-test compatibility and60-test export gates retain their execution-time scopes; they overlap. Supplied-projection review approves15 frozen files without independent I/O/tests.378 other prior frozen files and all four immutable baselines remain unchanged;11 historical supersessions are recorded.156 test files are registered;40 coarse statuses are unchanged.

Evidence: `artifacts/neutral-calibration-gate-v1.json`, `artifacts/neutral-calibration-frozen-hashes-v1.json`, `artifacts/neutral-calibration-projection-review-v1.json`. This supersedes earlier pending selected-neutral/cache/report work; current native acceptance v2 and public baselines remain separate. No fresh full suite, whole-domain or scientific completion is claimed.

Next implement F16 finite positive scale and representable dimensional bundles with homogeneous summary units, then continue remaining F01-F22/WP0-WP6 domains. Original SMPTS source/license/dependencies and L01–L04 remain external gates. Full goal remains active and full unittest is required before completed handoff.

Routine project edits remain authorized under Approve for me/auto_review; no renewed human confirmation. Root owns I/O, edits and tests; reuse existing text workers/reviewer.80985 is terminalexit0. Never rerun old one-shot drivers/updaters, overwrite fixed baselines or inspect/edit/stage `analysis.seedfiddle-project.json`. Never manually edit source/test/config/runtime/artifact scripts while a test/GPU/capture process is LIVE.


All6 register checks PASS in1.304s,zero skips; scoped whitespace checks pass. Raw-byte links and156 registered test pins match. Scoped closure is `artifacts/neutral-calibration-closure-v1.json`; the updater/finalizer executed once and must not be rerun.

## Current native runtime acceptance revision (10 October)

The current native lifecycle capture completed all 223 profiles in 2238.021s: 3747 nodes and 70 distinct source-absent replay contracts. Strict comparison found exactly 67 runtime-closure pin changes and no numerical output, recipe, input, frame, preparation, fixture, environment or helper-pin changes. The 156 unaffected profiles are identical. Fresh closure calculations match every changed hash. The raw comparison retained an additional in-memory TorchVersion type flag; both stored JSON version strings are identical.

A supplied-projection review supports separate immutable `artifacts/recipe-acceptance-baseline-v2.json` (SHA256 `82861ec3480b7ac9bcaa65be1d27bbc9dc15d928d3e813210e0fad29c5bf6eb1`). Historical v1 and the other fixed baselines remain unchanged. Three focused migration/inventory/partial-boundary tests pass in 365.660s, zero skips. This capture and focused gate are separate from a fresh full baseline unittest or full suite. Engineering scope only; whole-domain, original-SMPTS and scientific gates remain open.

This supersedes the earlier pending current-runtime revision marker. Next implement selected neutral-patch controls with source/corrected CUDA sampling, preserve upstream geometry for ruler-only changes, bind report revisions to actual calibration facts, and complete F16 numerical scale safety before continuing remaining F01-F22/WP0-WP6 domains. Evidence: `artifacts/polynomial-runtime-candidate-exact-review-v1.json`, `artifacts/polynomial-runtime-revision-frozen-hashes-v1.json`, `artifacts/neutral-geometry-calibration-domain-audit-v1.json`, `artifacts/calibration-node-local-cache-audit-v1.json`.

Routine project edits remain authorized under Approve for me/auto_review; no renewed human confirmation. Root owns I/O, edits and tests; existing workers provide text advice/review only. Captures31068 and focused82176 are terminal exit0. Never rerun old one-shot drivers/updaters or overwrite expected evidence. Never inspect/edit/stage `analysis.seedfiddle-project.json`. Full F01-F22/WP0-WP6 goal remains active; final full unittest is still required.

## Polynomial correction integrity (10 October)

The direct CUDA polynomial correction now retains all four image/mask input checks across extraction, fitting and application. Extracted/copied chip matrices and private fit arrays remain guarded through the final output check; mutation takes precedence over cancellation and caller exceptions. Numerical formulas, summation order, controls and CUDA residency are unchanged.

The final focused gate passes38 tests in96.072s,zero skips, including7 new integrity/validation checks, actual pinned PlantCV comparisons, node-local cache behavior and four-file version5 archive/source-removal contracts. A stale test expecting version4 was corrected. This is focused engineering evidence, not full unittest, whole-domain or scientific acceptance. The supplied-projection review approves the three-file freeze without independent I/O/tests.

The register references154 test files with40 coarse statuses unchanged. All39 preceding frozen files and three fixed baseline files remain intact. The historical223-profile native baseline pins the earlier correction implementation, so current aggregate runtime acceptance is pending a separate candidate capture and strict provenance-only diff review. Never overwrite/refresh v1 or normalize away unexpected differences. Evidence: artifacts/polynomial-integrity-gate-v1.json and artifacts/polynomial-integrity-frozen-hashes-v1.json.

Next: capture/review that runtime revision, then continue F01-F03 and remaining requirement/domain implementation. Original SMPTS and L01-L04 remain external gates; full F01-F22/WP0-WP6 stays active.

No test/GPU/capture process is LIVE at this reconciliation.81033 is terminalexit0. Root owns I/O/edits/tests and reuses the existing text workers/reviewer. Routine project edits remain authorized under Approve for me/auto_review; no renewed human permission. Never inspect/edit/stage analysis.seedfiddle-project.json. Never rerun this updater or old capture/updater/finalizer scripts. The forthcoming runtime candidate must have its own one-shot driver/status ledger and retain v1.

All6 register checks PASS1.209s,zero skips; scoped implementation closure is artifacts/polynomial-integrity-closure-v1.json. The separate one-shot candidate driver is artifacts/capture-polynomial-runtime-candidate-v1.py. Consult artifacts/polynomial-runtime-candidate-status-v1.json for execution state; candidate/diff remain unreviewed until explicit reconciliation. Do not restart a LIVE/terminal driver or edit source/tests/config/capture scripts while it is LIVE.

## F05 acceptance reconciliation (10 October, historical)

The current extended-threshold gate passes 49 tests in 218.422s, zero skips: 22 primitive, 8 recipe, 6 control/Qt and 13 saved-replay checks. Runtime/test files, 39 preceding freezes and fixed baselines remain unchanged. This is a focused gate, not full unittest or science.

The F05 contracts now reflect mask-only extended outputs, even Gaussian requests, source-coordinate Bayes masks of differing dimensions, guarded positive constant-channel roundoff covariance, and implemented directory/project/Qt/SVG paths. Historical hash-matched classifier gates retain their exact scope and execution freshness; they are not promoted to a fresh whole-pipeline gate.

The acceptance audit is docs/PLANTCV_SMPTS_ACCEPTANCE_AUDIT.json: 40 requirements are listed; F05 has 12 criteria: 9 scoped, 2 pending training reconciliations and 1 external scientific gate. The other 38 operation rows await detailed reconciliation. All 40 coarse statuses, latest runtime implementation and immutable acceptance baseline are preserved. Each scoped criterion states current/historical evidence, exact selected profile, domain, unmet predicate and dependency. No whole F05/domain or scientific completion is claimed.

Next: reconcile F01.1/F01.2/F02.1/F03.1/F03.2, then remaining registered requirements and explicit F05 domains. L01-L04 and original SMPTS source/license/dependencies remain external gates. Full F01-F22/WP0-WP6 stays active. Focused register results and closure are recorded in artifacts/f05-domain-reconciliation-register-final-v1.log and artifacts/f05-domain-reconciliation-closure-v1.json.

No test/GPU/capture process is LIVE;32756 is terminalexit0. Root owns all I/O/edits/tests; reuse the existing text workers/reviewer. No renewed project-edit approval. Never inspect/edit/stage analysis.seedfiddle-project.json. Never rerun this updater or the focused one-shot driver.

## Calibrated mean/Otsu desktop roundtrips (10 October)

Two real asynchronous public method sessions now cover mean and Otsu capture/save/reopen alongside the closed binary desktop milestone. Both execute10 public calls and retain15 verified checkpoints. The mean control requests8/offset2/light; the saved spec/UI preserve8 while actual effective run facts preserve9. Original child selections stay unchanged. Original source files are deleted before reopening under guards prohibiting source identity reads and public reruns; exact composition, addressed manifest and every checkpoint revision restore.

Focused gate:2 tests PASS in108.983s,zero skips,session9822 terminalexit0. Arrays and observations are checked against actual returned results for retention/serialization. This is no new independent numerical oracle, visual claim or native/runtime change. Earlier direct binary/mean/Otsu coordinator comparisons, binary27-test UI gate and four readable complete/partial views retain their own scopes. Bounded supplied-projection review approves the single test-file freeze without independent I/O/tests/visual inspection.

The register now references153 test files,40 operation statuses unchanged,48native replay/13public runner operations. All38 preceding frozen files,152prior registered test hashes and fixed baselines remain unchanged. Latest runtime implementation and immutable public acceptance checkpoint are preserved; this evidence is a separate latest verification checkpoint. All6 calibrated methods register checks PASS in1.173s,zero skips; final verification and closure are recorded in artifacts/calibrated-public-methods-register-final-v1.log and artifacts/calibrated-public-methods-closure-v1.json.

Stale method-matrix wording that whole-source-chain/Qt integration remained open is replaced with exact tested selected scopes and remaining predicates. Next: requirement-by-requirement software coverage reconciliation across all F01-F22/WP0-WP6, starting extended thresholds and Bayes/KMeans. Enumerate canonical recipe/version/parameter profiles, declared supported/withheld inputs, actual oracle/native evidence, graph/cache/resource/persistence/UI coverage and precise unmet cases.223 native cold/warm/archive profiles,48replay operations and70source-absent contracts are existing evidence to reconcile, not whole-goal completion. Original SMPTS source/license/dependencies and L01-L04 laboratory/scientific inputs remain explicit external gates.

No test/GPU/capture process is LIVE; session9822 is terminal and must not be restarted. Root owns I/O/edits/tests and reuses the same text workers/reviewer. Routine project edits remain authorized under Approve for me/auto_review without renewed confirmation. Never inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. No manual source/test/artifact-script/config/runtime edits while tests are LIVE. The new updater and finalizer each executed once; methods closure is complete. Never rerun either or historical updaters. Full goal remains active; completed-goal full unittest and scientific validation remain open.

## Calibrated public desktop capture and saved results (10 October, historical)

The source-graph view now opens a separate calibrated public capture for exact original binary, mean and Otsu selections. Honest card controls and distinct card/seed bounds freeze a selected composition. Public correction, corrected crop, child seed calls and independent source-scaled size run asynchronously. Native CUDA calculations/caches, the ordinary capture dialog and historical attachments remain unchanged.

Final focused gate:27 tests PASS in101.981s,zero skips (15new,9ordinary desktop,3ordinary layout). One actual selected binary ten-call session binds original bytes, prepared RGB, corrected crop and source-returned mm scale. Async saved reopen restores exact child/card/seed ROI without original-image reads; editing invalidates it. Frozen source/generation guards and Stop/close drain discard stale replies. The initial27-test gate had25passes/two failures103.758s; color's rgb input key and long button labels were repaired before the green gate.

Declared replay faults retain successful sibling phases and independent scaled size after child size/color failure, with separate pixel/mm measurement facts and no complete partial manifest. Before displaying downstream observations, shared-budget loads verify exact refs/public pins/source/frame/spec/attempt, corrected crop/final mask, actual watershed labels and returned scale lineage. Preview copies share4Mi aggregate pixels; coordinate/palette facts declare original/displayed counts and at most4096 display elements. Checkpoint numeric budgets exclude metadata, temporary/crop-validation/preview copies, codec/loader/public scratch and RSS.

Root inspected four readable complete-correction/partial-watershed views at1440x900/1050x650. Separate renderer12268 exits0 after another actual ten-call capture and declared color-fault replay. Windows font registration is confined to the ignored driver;3387QTimer ticks establish responsiveness in that run only. Fontlessv2 views prove bounds only; v3 failed before capture with0calls. Evidence: artifacts/calibrated-public-desktop-visual-v4/evidence.json. Bounded supplied-projection reviewer approves the three-file freeze without independent I/O/tests/visual inspection.

All35 other preceding frozen files,151prior registered test files and immutable expected baselines remain unchanged. The graph dialog's previous SHA is explicitly superseded in artifacts/calibrated-public-desktop-frozen-hashes-v1.json. Forty operation statuses remain unchanged,48native replay/13public runner operations,152registered test files. All6 calibrated desktop register checks PASS in1.201s,zero skips; final verification and closure are recorded in artifacts/calibrated-public-desktop-register-final-v1.log and artifacts/calibrated-public-desktop-closure-v1.json. The latest acceptance checkpoint stays the earlier immutable public seed baseline.

Full F01-F22/WP0-WP6 remains active. Next: bounded real mean/Otsu calibrated dialog capture/save/reopen and exact requested/effective selections. Broader method/input/resource domains, separate polynomial correction, original SMPTS, L01-L04laboratory inputs, science and completed-goal full unittest remain open.

No test/GPU/capture process is LIVE; failed58966,passing87150 andrenderer12268 are terminal and must not be restarted. Root owns I/O/edits/tests and reuses the same two text workers/reviewer. Routine project edits remain authorized under Approve for me/auto_review without renewed human confirmation. Never inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. No source/test/artifact-script/config/runtime edits while tests are LIVE. The successful updater and finalizer have each executed once; closure is complete. The earlier updater failed before document edits on mixed-encoding historical log bytes and remains history. Never rerun these or historical updaters.

## Calibrated public immutable snapshot and saved session (10 October, historical)

A separate session layer now captures one immutable byte-identified source with one local OpenCV color decode and an RGB-only prepared parent. Original source byte identity, selected composition, decoder/build and local preparation code/frame remain explicit. Source stamps/spec/code are checked during execution and original bytes are rehashed before saving. The frozen public correction/crop/child/scaled-size coordinator and complete reader remain unchanged.

All18 focused session checks PASS in290.868s,zero skips. One actual selected binary capture makes10 public requests and retains15 verified checkpoints with exact prepared RGB pixels and source-returned mm scale. Reopen from a moved store uses its exact saved child/card/seed ROI spec and checkpoints without original source I/O. Declared faults cover changed source bytes, sidecar/spec/decoder/code facts, partial/corrupt records, cancellation and publication failures. Earlier direct pinned binary/mean/Otsu comparisons retain their previous scope; replayed results test contracts only.

Admission separates full-parent RGB transmission from crop-only watershed. Owned preparation is bounded at6P before the contiguous RGB copy. The conservative retained envelope is6P+76Q+3; actual complete reader allocations use a remaining budget, including duplicate loads, and pass exact-total/one-byte-below checks. Encoded bytes, metadata, temporary copies, codec/loader/validation/public scratch and RSS remain outside numeric bounds. Initial decode precedes output-shape admission, so no codec peak bound is claimed. Local preparation/crop retains actual_public_reader=False/actual_public_crop=False.

Canonical spec sidecars use unique same-folder temporary files, flush/fsync and atomic replacement. The last arbitrary caller callback precedes a callback-free full graph/preparation audit. Precommit failure removes only this attempt's exact hash-verified parent manifest; checkpoints, complete children and changed replacement files survive. Cancellation after commit retains the revision. An exception reported after an exact committed manifest/sidecar pair is present preserves that pair while propagating the error. No transaction-wide exclusion of asynchronous external file writers is claimed.

Bounded supplied-projection review approves the exact two-file freeze in artifacts/calibrated-capture-session-frozen-hashes-v1.json; reviewer performed no independent I/O/tests/visual inspection. All34 preceding frozen files and fixed expected baselines are unchanged. Forty operation statuses remain unchanged,48 native replay operations,13 public runner operations and151 referenced test files. All six calibrated-session register checks PASS in0.871s,zero skips; final verification and closure are recorded in artifacts/calibrated-capture-session-register-final-v1.log and artifacts/calibrated-capture-session-closure-v1.json. The latest acceptance checkpoint remains the earlier immutable public seed baseline; this checkpoint adds session implementation evidence.

Full F01-F22/WP0-WP6 remains active. Next: separate calibrated public Qt dialog/worker, honest card/seed ROI controls, saved reopen/partial checkpoint display, source/generation guards and stop/close drain. Broader domains, separate polynomial correction, original SMPTS, L01-L04 laboratory inputs, science and completed-goal full unittest remain open.

No test/GPU/capture process is LIVE; session37184 is terminal/exit0 and must not be restarted. Root owns I/O/edits/tests and reuses the existing two text workers/reviewer. Routine project edits remain authorized under Approve for me/auto_review without renewed human confirmation. Never inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. No source/test/artifact-script/config/runtime edits while tests are LIVE. The new updater and finalizer have executed once; closure is complete. Never rerun either or historical updaters.

## Public corrected seed coordinator and typed reader (10 October, historical)

The public benchmark now composes actual pinned full-image color-card correction, an explicit half-open seed crop, the unchanged binary/mean/Otsu eight-call child, and a separate actual scaled-size call. This is a SeedFiddle composition of public calls. Original PNG byte identity, corrected pixel identity, crop translation, exact selected child, correction checkpoint and actual watershed labels are bound separately. Size uses the scale returned by correction; positions remain ROI-local pixels and biological validation remains false.

All26 final focused/adapter tests PASS in362.060s,zero skips:17 new checks and9 preceding adapters. Three complete selected chains match direct pinned corrected/cropped/preprocessing/watershed arrays, color observations and pixel/scaled-size observations/diagnostic pixels exactly. Default empty makes7 requests and skips analyses. Explicit7 empty RGB makes10 requests and completes with unchanged bounds; the earlier ALL-space empty failure remains separate immutable history. Declared faults verify independent scaled size after child size/color failures, cancellation, changed inputs, mutation precedence and withheld publication.

The complete reader checks15 retained checkpoint entries under the remaining aggregate numeric allocation budget, including independent duplicate loads. Metadata, temporary copies, codec/loader/validation/public scratch and process RSS are outside this ledger. The last caller callback precedes a callback-free final graph audit and atomic parent commit. Successful individual/child checkpoints survive partial attempts; no transaction-wide exclusion of asynchronous external writers is claimed. Local OpenCV preparation/crop explicitly records actual_public_reader=False and actual_public_crop=False.

Bounded supplied-projection review approves the exact three-file freeze in artifacts/public-calibrated-seed-frozen-hashes-v1.json; reviewer performed no independent I/O/tests/visual inspection. All31 preceding frozen files and fixed expected baselines are unchanged. The intermediate12-test pass predates final guards. The24-test run had22pass/1failure/1error420.948s: a wrong expected RGB-empty failure and a read-only mutation fixture. Both test setups were corrected; the final26-test gate is green.

Forty operation statuses remain unchanged,48 native replay operations,13 public runner operations and150 referenced test files. All six corrected-seed feature-register checks PASS in0.880s,zero skips; the final verification log and closure are recorded in artifacts/public-calibrated-seed-register-final-v1.log and artifacts/public-calibrated-seed-closure-v1.json. This implementation checkpoint does not replace the immutable earlier acceptance baseline. Full F01-F22/WP0-WP6 stays active. Next: RGB-only immutable path snapshot and saved composition session, then asynchronous Qt controls/reopen/partial display and stale-generation guards. Broader domains, separate polynomial correction, original SMPTS, L01-L04, science and completed-goal full unittest remain open.

No test/GPU/capture process is LIVE. Sessions20297/20660/57664 are terminal; do not restart them. Root owns I/O/edits/tests and reuses the same two text workers/reviewer. Routine project edits remain authorized under Approve for me/auto_review; no renewed human permission. Never inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. No source/test/artifact-script/config/runtime edits while tests are LIVE. The new updater and finalizer have executed once; closure is complete. Never rerun either or historical updaters.

## Public card ROI and size calibration adapters (10 October, historical)

The isolated public runner now passes selected half-open card bounds through an explicit pcv.Objects adapter into the actual pinned auto_correct_color. The correction still returns the full image. Bounds must be integer, nonempty and inside the decoded frame; no clamping occurs. Absent ROI retains the old call and metadata. Explicit null records full-frame selection. The pinned _rect_filter convention and adapter origin are declared.

Public analyze.size accepts an explicit finite positive px_width/px_height context with a pixels or mm unit. It runs the real public calculation under those temporary pcv.params values and restores/checks the actual previous values in finally. Nonunit factors labelled pixels retain public source multiplication; they establish no physical calibration. Caller scale origin and corrected-parent/crop/checkpoint lineage remain to be implemented. The isolated worker processes one request, so another request cannot observe its temporary globals.

All 29 focused/runner checks PASS in 74.328s, zero skips: nine new adapter checks and twenty existing runner checks. Four contexts compare direct pinned correction and size calls: no ROI, shifted edge-touching ROI, and two competing-card ROIs of the same size. Corrected pixels, size observations and diagnostic pixels match exactly; selecting the other card changes the corrected image. Explicit null/full-frame legacy checks agree. Mock tests cover pre-call rejection, exact Objects containers, calibration calculation/default isolation and prior nondefault restoration on a public exception. The initial default sandbox run reported five ERROR headers and was interrupted during timeout/cleanup; no final test count or traceback was emitted. The automatic-review rerun passes.

Bounded supplied-projection review approves the two-file freeze in artifacts/public-calibration-adapter-frozen-hashes-v1.json; reviewer performed no independent reads/tests/visual inspection. Twenty-nine prior frozen files and fixed expected baselines are unchanged. The historical worker SHA is explicitly superseded. Native CUDA calculations/caches, original authored recipes and the existing capture UI are unchanged.

Forty operation statuses remain unchanged, with 48 native replay operations, 13 public runner operations and 149 referenced test files. All six adapter feature-register checks PASS in 1.268s, zero skips; closure is recorded in artifacts/public-calibration-adapter-closure-v1.json. The latest implementation checkpoint is this bounded adapter milestone; the latest acceptance checkpoint remains the earlier immutable eight-case public seed record. F01-F22/WP0-WP6 stays active. Next: actual corrected full RGB, explicit seed crop, original preprocessing/watershed and actual scaled size/color, with typed parent lineage, cancellation, verified saved records and desktop integration. Broader domains, separate polynomial correction, original SMPTS, L01-L04 laboratory inputs, science and completed-goal full unittest remain open.

No test/GPU/capture process is LIVE. Failed sandbox17179 is terminal/exit-1 after stopping only the verified owned test process33552; no final summary. Automatic-review65395 is terminal/exit0,29 PASS74.328s. Root owns I/O/edits/tests; reuse the same text-only two workers/reviewer. Routine project edit authorization persists under auto_review; do not ask again. Never inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. No source/test/artifact-script/config/runtime edits while tests are LIVE. The new updater and finalizer both execute once; closure is complete. Never rerun either or historical updaters.

## Closed finite public seed acceptance and desktop layout checkpoint (historical)

Eight fixed actual public chains now retain exact deterministic inputs, masks, watershed locals/labels, size/color observations, histogram data, selected/effective controls and source/package/build identities. Binary light/dark, mean requested8/effective9 and Otsu complete. Explicit7 on nonempty/empty labels executes size but preserves the public color UnboundLocalError as partial. Default empty invokes/verifies watershed once and skips size/color. Gray RGB retains the float16 JSON color failure. Four cases complete,one is empty,three are partial; partial/empty attempts publish no complete manifest.

The immutable expected record SHA is5f84f7f8e8d9b71e6f0704a262c6002e4ed382a16cc13511566596b0cf87eb6a. Fresh tests reconstruct actual calls and never refresh expected evidence. Transient durations/UUIDs/checkpoint addresses and shuffled palette/display ordering are outside exact repeatability; full per-run evidence retains them. Actual BGR/RGB relation and sorted palette multiset remain checked/bound. Controlled-copy projection checks are normalization tests only. Prepared RGB/gray use one local OpenCV snapshot with actual_public_reader=False; no public-reader parity follows.

The preview now fits its actual widget on resize, keeps aspect ratio and clears retained state on image replacement/invalidation. Long paths and hashes yield horizontal space. Root inspected eight readable offscreen views at1440x900/1050x650: saved complete reopen, BGR/RGB products, running controls and a fresh genuine partial outcome matching the baseline. All requested dimensions and preview bounds hold. Installed-font override exists only in the ignored evidence driver; native application font/calculations/CUDA caches are unchanged.2605 QTimer ticks establish measured responsiveness for one run,not a latency guarantee.

Final compatibility29 tests PASS661.965s,zero skips, plus separate same-size replacement1 PASS0.497s:30 distinct passing IDs across two executions,not a single combined30-test gate. The earlier13-test run had12 passes/one width failure17.879s; the repaired13-test gate passes15.499s and overlaps final coverage. Runtime is unchanged after29; only the extra focused test was added. Bounded supplied-projection review approves the six-file freeze; no independent reviewer I/O/tests/visual inspection. Evidence: artifacts/public-seed-acceptance-frozen-hashes-v1.json, public-seed-acceptance-review-v1.json and public-capture-layout-evidence-v1/visual-evidence.json. All24 other preceding frozen files remain unchanged; the prior UI freeze stays historical and is explicitly superseded.

Forty operation statuses remain unchanged,48 native replay operations,13 public runner operations and148 referenced test files. All6 final feature-register checks PASS0.848s,zero skips; closure: artifacts/public-seed-acceptance-closure-v1.json. F01-F22/WP0-WP6 remains active. Next: actual public color-card corrected capture/acceptance, then broader method/input/default/resource domains. Original SMPTS, L01-L04 laboratory inputs, scientific validation and completed-goal full unittest remain open.

No test/GPU/capture process remains LIVE. Candidate99105, focused56647/77976, renderer93368 and compatibility87154 are terminal; do not restart them. Root owns I/O/edits/tests; reuse the same text-only two workers/reviewer. Routine project-edit approval persists under auto_review,with no renewed human permission. Never inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. No source/test/artifact-script/config/runtime edits while tests are LIVE. The new register updater and finalizer both executed once; closure is complete. Never rerun either or any earlier updater.

## Closed public reference capture reader and desktop checkpoint (historical)

The source-graph view now launches a separate public seed capture for exact selected historical binary, mean and Otsu recipes. Honest calculation controls create an immutable request. Public preprocessing, one watershed capture and independent size/color calls run off the UI thread. Captured locals remain artifacts of one call; native calculations, CUDA caches and historical graph records remain unchanged.

The complete-manifest reader verifies all ten typed checkpoints against trusted caller roots/digest, exact source/frame/recipe/measurement/stage/role identities, actual source pins and runtimes, label facts and unchanged size/color label contexts. It rederives the authored maximum-label or explicit positive loop bound. Numeric-array descriptors use the remaining aggregate budget before each load; metadata/codec/loader/public/process scratch and RSS are outside that bound. The local OpenCV decoder prepares RGB and color-derived gray explicitly with actual_public_reader=False.

Saved captures restore the exact selected recipe after verification. Worker generations and source stamps reject stale completion; Stop/close drain without stale previews. Actual observations/charts and compact artifacts appear as source facts, with original-pixel hashes and at most4Mi aggregate sampled preview pixels. Raw BGR is adapted only for Qt display. Cancellation/error before the recipe sidecar commit retracts only this attempt's exact hash-verified manifest; changed replacement files and individual checkpoints survive. After commit, the completed stored revision persists even if the UI discards a stale result.

Final compatibility58 tests PASS269.004s,zero skips:24 new reader/session/desktop checks and34 existing graph/persistence/entrypoint checks. Actual selected chromatic binary capture executes eight public calls; stored capture loading also exercises asynchronous Qt delivery. The initial8-test/setup-error run76.258s retains a genuine public color float16-observation JSON failure on gray RGB. The24-test run138.879s had19 passes/five bound-method fault-fixture failures; a test-only repair precedes the final green gate. These failed runs remain history. Bounded review approves the supplied five-file freeze/root evidence, with no independent reviewer I/O/tests/visual inspection and no root visual-inspection claim. Evidence: artifacts/public-reference-capture-frozen-hashes-v1.json, public-reference-capture-review-v1.json and public-reference-capture-compatibility-v3.log. All20 prior coordinator/capture/recipe/reader frozen files remain unchanged.

Forty operation statuses remain unchanged,48 native replay operations,13 public runner operations and146 referenced test files. All6 final feature-register checks PASS1.920s,zero skips; closure: artifacts/public-reference-capture-closure-v1.json. F01-F22/WP0-WP6 stays active; next is a finite actual binary/mean/Otsu whole-chain outcome matrix and representative complete/partial public Qt rendering. Broader input/default/resource domains, original SMPTS, L01-L04 laboratory inputs, scientific validation and completed-goal full unittest remain open.

No test/GPU process remains LIVE; sessions3444,17360 and2364 are terminal and must not be restarted. Root owns I/O/edits/tests; reuse the same text-only two workers/reviewer. Routine edit authorization persists under auto_review; no renewed human approval. Never inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. No source/test/artifact-script/config/runtime edits while tests are LIVE. The new register updater and finalizer both executed; closure is complete. Do not rerun either or any earlier updater.

## Closed public seed-chain coordinator checkpoint (historical)

The benchmark coordinator now binds prepared gray and RGB checkpoints to the exact source, frame and selected authored recipe before any public call. It reuses the five existing preprocessing bridges, calls watershed.capture once, then runs actual public size and color independently. Captured locals remain artifacts of that one watershed call. Size/color consume the original prepared RGB and content-verified published labels; their labels_context raster is explicitly an input context. Actual source observations, charts, raw BGR diagnostic and separate RGB adapter retain provenance.

Unselected n_labels uses int(public_labels.max()), exactly the authored rule. Explicit positive selections pass unchanged, even on an empty raster; actual public errors remain failures. Marker count, distinct positive final count and maximum label ID are separate facts. Default zero labels skip both analyses honestly. Successful individual checkpoints survive a partial result; only a completed required call/capture set publishes the final atomic manifest. Cancellation and changed inputs withhold publication; each independent analysis reloads the immutable original labels.

Validation has35 distinct passing check IDs across two executions: the35-test compatibility run had34 passes/one fault-fixture error in771.932s,zero skips, and the corrected fixture passes its focused1-test repair in84.939s. There is no single green35-test gate. The initial9-test PASS204.395s predates final guards/counters/new actual-empty checks. Actual public coverage is a selected binary eight-call chain plus empty default/explicit7 requests; existing preprocessing checks cover binary/mean/Otsu. Fault injection checks orchestration contracts and is not a public numerical oracle. Bounded review approves the supplied final two-file freeze and root evidence; no independent reviewer I/O/tests/visual inspection. Evidence: artifacts/reference-seed-chain-frozen-hashes-v1.json, reference-seed-chain-review-v1.json, reference-seed-chain-compatibility-v1.log and reference-seed-chain-repair-v1.log. All18 prior capture/recipe/reader frozen files and canonical peer manifest remain unchanged.

Forty operation statuses remain unchanged,48 native replay operations,13 public runner operations and145 referenced test files. All6 final feature-register checks PASS1.826s,zero skips; closure: artifacts/reference-seed-chain-closure-v1.json. Native application/Qt/cache behavior is unchanged. The typed manifest reader and asynchronous Qt capture view are next; mean/Otsu whole-chain and broader public method/input/default acceptance remain open. Full F01-F22/WP0-WP6, completed-goal full unittest, original SMPTS and L01-L04 laboratory inputs remain active/open; scientific validation is false.

No test/GPU process remains LIVE; sessions90861,82219 and17588 are terminal and must not be restarted. Root owns I/O/edits/tests; reuse the same text-only two workers/reviewer. Routine project authorization persists under auto_review; never inspect/touch/stage analysis.seedfiddle-project.json, no staging/commits. No source/test/artifact-script/config/runtime edits while tests are LIVE. Generated evidence stays ignored in artifacts/. The new coordinator register reconciliation and finalizer executed once; register checks/closure are complete. Do not rerun either. Do not rerun any prior updater. Next implement the typed manifest reader and asynchronous Qt capture path without treating captured locals as independent public stage executions.

## Closed public watershed capture checkpoint (historical)

A separate segmentation.watershed.capture reference operation executes the exact pinned PlantCV function once and captures its actual distance, coordinates, peaks, markers, negative distance, labels, BGR diagnostic and shuffled palette. RGB is a declared adapter of the same diagnostic. No public helper is replaced or intermediate recalculated. The ordinary labels-only operation and historical authored recipe definitions stay unchanged. Empty coordinates use an exact typed zero-length transport record; general NPZ raster rules remain strict.

Public source/runtime/helper/adapter identities are verified before and after the call. A malformed capture retains only independently copied, validated public labels and an unavailable reason; public-call errors propagate. Existing profiler/trace behavior and actual public RNG advancement are preserved. Cancellation withholds publication and cleans transport. Retained numeric arrays are admitted at42P+3 bytes and16P+3 elements with actual descriptor checks; metadata, validation scratch and public/process peak memory are outside that bound.

Final68 focused/compatibility tests PASS313.624s,zero skips, including15 new tests and53 existing runner/preprocessing/mask/stage checks. The prior13-test gate PASS21.376s predates final runtime/fallback-label fixes and overlaps these68. The exact four-file freeze has bounded reviewer approval from supplied projections/root execution; no independent reviewer I/O, tests or visual inspection. Canonical peer manifest and14 prior recipe/reader frozen files remain unchanged. Evidence: artifacts/reference-watershed-capture-frozen-hashes-v1.json, reference-watershed-capture-review-v1.json and reference-watershed-capture-compatibility-v1.log.

All eight register/planning mirrors now record this API checkpoint:40 operation statuses remain unchanged,48 native replay operations,13 public runner operations and144 referenced test files. All6 capture register checks PASS0.833s,zero skips; closure: artifacts/reference-watershed-capture-closure-v1.json. Full F01-F22/WP0-WP6 stays active. Next: source whole-chain coordinator, typed checkpoints and captured-artifact links to one public call, followed by asynchronous Qt integration. Broader public method/input/default domains, completed-goal full unittest, original SMPTS and L01-L04 laboratory inputs remain open; no scientific validation or whole-chain/desktop completion is claimed.

No test/GPU process remains LIVE; sessions4085,51691 and18350 are terminal and must not be polled/restarted. Root owns I/O, edits and tests; reuse the same text-only two-worker/reviewer pool. Routine project actions remain authorized under auto_review; never inspect/touch/stage analysis.seedfiddle-project.json, no staging/commits. No source/test/artifact-script/config/runtime edits while tests are LIVE. Generated evidence stays ignored under artifacts/. The recipe closure updater/finalizer and this capture register updater already executed; do not repeat them. Capture register checks and closure are complete. Next implement whole-source-chain checkpoints with explicit RGB/gray frame/source/recipe identities and source int(labels.max()) or explicit positive caller n_labels, distinct from marker count.

## Closed recipe acceptance checkpoint (historical)

Fresh session4085 is terminal/exit0: all4 tests PASS in2736.266s,zero skips. The immutable baseline verifies223 selected profiles (69 canonical,56 median ROI,48 reader,48 reader-plus-median and2 correction drivers),3747 ordered nodes, declared outputs/statuses, cold/warm cache/key equality and fully verified v5 archives. Seventy first-representative operation/serialized-port replay contracts cover all48 supported replay operations with source/model origins hidden. External image/model providers and four-input correction remain outside generic replay.

The known partial mean ellipse boundary is excluded from223 and verified separately. Selected plain9/-2 and calibrated21/-1 mean chains match actual pinned public raster/color products and size observations under the existing1e-6 tolerance. Baseline SHA b8fefc1b5d3bf0d595bfa243303bcc194f181d49f36cf05502d69654ace4c8f2; test SHA4804505a6dcea146bb316b2c495e08e57e0ae6487e308da222e3d3538f8c222d. Tests never regenerate expected evidence. The exact two-file freeze has bounded review approval from supplied projections/root evidence, with no independent reviewer I/O, tests or visual inspection. Fourteen final/prior-reader frozen files,13 fixture/exporter source files and capture helper identities are verified unchanged; runtime application code is unchanged.

All eight register/planning mirrors now record this checkpoint. Forty operation statuses and48 replay operations remain unchanged;143 referenced test files include causal recipe acceptance references. All6 structural register checks PASS2.094s,zero skips. Evidence: artifacts/recipe-acceptance-frozen-hashes-v1.json, recipe-acceptance-final-root-evidence-v1.json, recipe-acceptance-review-v1.json and recipe-acceptance-focused-v1.log; closure is recorded in recipe-acceptance-closure-v1.json. Failed v1/v2/initial-mean history remains in the candidate audit and preceding handoff history. No defaults, broader-domain, all223 public-source parity, maximum-resource or scientific claim follows from selected native lifecycle agreement.

Next: extend actual pinned public whole-chain reference capture and acceptance. Full F01-F22/WP0-WP6 stays active; remaining method/input domains, completed-goal full unittest, original SMPTS source and L01-L04 laboratory inputs remain open.

No test/GPU process remains LIVE. Root owns I/O, edits and tests; reuse the same text-only two workers/reviewer. Routine project actions remain authorized under auto_review. Never inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. Generated records stay in ignored artifacts/. No source/test/artifact-script/config/runtime edits while tests are LIVE. The recipe closure updater executed once; do not rerun it or earlier updaters.

Next implementation design: preserve ordinary labels-only watershed. A separate reference capture operation calls the exact pinned public function once and snapshots actual bounded locals on normal return through sys.setprofile, chaining/restoring any prior profiler and preserving trace hooks. Retain distance, coordinates (including typed zero-length transport), peaks, markers, negative distance, labels, actual BGR diagnostic and shuffled palette; an explicit adapter adds RGB. Conservative admission for all retained raw/adapted products is42*H*W+3 bytes and16*H*W+3 elements in the pinned intp64 environment, with actual summed descriptor checks before copying; this is not a public-call peak/RSS bound. Source/runtime and helper pins must hold before/after. Preserve RNG advancement, not independent shuffled-call pixel parity. Capture products are observations of one public call; source drift aborts, while malformed capture after valid public labels is separately unavailable. Historical public size/color use int(labels.max()), distinct from marker count; preserve explicit positive caller selections and honest unavailable zero-count calls. Science=false. No implementation approval is claimed yet.

## Closed reader acceptance checkpoint (historical)

No test/GPU process remains live. Reader acceptance session94782 is terminal with2 passing tests; both capture sessions are terminal. Root owns I/O, edits and tests; do not edit source/test/artifact scripts while a test/GPU process is LIVE. Reuse the same text-only two workers/reviewer. Authorization remains in force under auto_review; never inspect/touch/stage analysis.seedfiddle-project.json and do not stage/commit. Generated evidence stays in ignored artifacts/. Previous document updaters have executed; do not rerun them. The acceptance baseline/test is frozen; the final register gate and closure are recorded; do not rerun prior document updaters.

Reader acceptance checkpoint (10 October). A frozen28-fixture record pins the ordered source bytes/SHA, public rgb/gray outputs and native color-derived/codec-gray outputs, frames, uint8 dtypes and pixel digests. Oracle, fixture builder, decoder builds, input helper and preparation code identities are bound. Torch2.13.0+cu126/CUDA12.6/RTX3070 is the measured acceptance environment, not a universal device guarantee.

Public46 successes/10 failures and native48 successes/8 rejections remain distinct outcomes. The two Unicode native successes match the identical ASCII input while both public path reads fail; no path-outcome parity is claimed. All48 successful native mode cases reuse identical warm preparations and pixels.24 actual color-to-codec switches preserve RGB tensor identity and contents; first-mode switch checks are not applicable.19 fixtures produce different grayscale pixels.

Fresh2 tests PASS21.645s,zero skips, independently reconstruct the full fixed record and never regenerate expected evidence. Missing or changed baseline bytes reject. Bounded review approves the supplied test/evidence freeze (artifacts/reader-acceptance-frozen-hashes-v1.json); all generated records stay in ignored artifacts/. The preceding ten-file reader implementation is unchanged. No independent reviewer I/O/tests/visual inspection, maximum-domain, complete-family or scientific claim. All six final feature-register checks PASS0.791s,zero skips. Forty operation mirrors/current table rows,142 referenced test-file hashes,two frozen test/evidence files and the preceding unchanged ten-file reader implementation are verified. Closure: artifacts/reader-acceptance-closure-v1.json.

F01-F22/WP0-WP6 remains active:40 registered operations,48 replay contracts and142 referenced test files. F01.1 stays partial_native/pending_recipe_freeze/not_applicable. Next: finite complete-recipe cold/warm/archive/replay output acceptance. Remaining method/input domains and all-recipe output freeze, completed-goal full unittest, original SMPTS and L01-L04 stay open.

## Reviewed reader implementation checkpoint

Compatibility session28700 is terminal with202 passing tests. Root owns I/O, edits and tests; no source/test/artifact-script edits while a test/GPU process is LIVE. Reuse the same text-only two-worker/reviewer pool. Project actions remain authorized under auto_review without renewed permission. Never inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. Historical fill/median document updaters have already executed. The reader register/planning reconciliation has executed once; its final gates and closure are recorded; do not rerun historical updaters.

Codec grayscale reader checkpoint (10 October). Versions3.0/3.1 add an explicit color-derived/codec-gray choice to24 existing uncalibrated working identities. The reader composes with median, fill and morphology ROI controls in all24 action orders, retaining selected parameters and dragged layout. Historical recipes remain supported; calibrated workflows derive grayscale from corrected RGB. No recipe IDs or replay operations were added.

Both decodes use one immutable source snapshot and must agree on uint8 dimensions/orientation. Graph and owned-array bounds precede gray decode/upload. A mode switch reuses guarded RGB, retains the old preparation through final audits and atomically publishes new gray. Direct inputs and replay verify mode/flag facts. Final9 integration tests PASS291.543s,zero skips, covering public28-case fixture/outcome checks, actual pixel differences, cache/ROI reuse, CPU/CUDA mutation and cancellation, desktop Keep/reopen, actual project batch pixels and source-independent median replay.

Ten implementation/test/config files are frozen in artifacts/reader-frozen-hashes-v1.json. Compatibility202 tests PASS2149.069s,zero skips, including the final3 reader contract checks.211 distinct code tests have passing executions across the separate9-test integration and202-test compatibility gates; no combined211-test gate is claimed. Bounded reviewer approval covers the supplied ten-file freeze and root evidence (artifacts/reader-review-v1.json). All six final feature-register checks PASS0.925s,zero skips. Forty row mirrors,141 referenced test-file hashes,ten unchanged frozen files and the canonical peer manifest are verified. Closure: artifacts/reader-closure-v1.json. Initial12-test gate:9 passes,2 failures,1 error; repair3-test gate:2 passes,1 error. These are nonpassing history. Mocked project functions/constants correctly triggered source audits; final tests observe unchanged functions. No independent reviewer reads/tests/visual inspection, maximum-resource or scientific claim. Prior shared-source freezes retain historical scope after this change.

F01-F22/WP0-WP6 stays active with40 registered operations,48 replay operations and141 referenced test files. F01.1 remains partial_native/pending_recipe_freeze/not_applicable. Next: record finite decoded-input and recipe-output acceptance. Remaining reader/codec and method/domain acceptance, all-recipe output freeze, completed-goal full unittest, original SMPTS source and L01-L04 laboratory inputs stay open.

## Previous median ROI checkpoint (historical)
Median ROI checkpoint (10 October). One version2.0.0 working contract adds independent null or half-open median bounds on56 existing recipe IDs, with no new IDs or replay operations. Historical factories and the CUDA median primitive/boundary rejection rules stay unchanged. All six median/fill/morphology ROI action orders preserve selected kernels, separate bounds and dragged layout. Same-ID historical/new versions coexist in project selection, save/load and batch. Diagnostic addition after median stays unavailable.

116 exact pinned public gray/binary outputs pass, including dtype, exterior pixels and strided CUDA input immutability. Changed median bounds/kernel recompute median and descendants while upstream nodes reuse; identical selections reuse all. Current archives for historical and new contracts replay without the source, including calibrated range-mask median in its96x128 local frame at translation600,260. Desktop async preview, controls/locks and Keep/reopen pass. Replay validates crop geometry and plans crop work before selected loading/upload while retaining the full-canvas memory floor.

Focused9 tests PASS210.302s and affected169 PASS1383.752s,zero skips.178 distinct code tests have passing executions across these two separate gates; no combined178-test gate is claimed. The separate56-ID construction probe is not a unittest count. Five unchanged source/test files are frozen in artifacts/median-roi-frozen-hashes-v1.json with bounded review approval from supplied projections/root evidence. No independent reviewer reads/tests/visual inspection or new maximum-resource/scientific claim. All six final register checks pass in0.820s,zero skips. Forty row mirrors,140 referenced test-file hashes,five unchanged frozen files and the canonical peer manifest are verified; scoped13-file whitespace passes. Closure: artifacts/median-roi-closure-v1.json.

F01-F22/WP0-WP6 remains active:40 registered operations,48 replay operations,140 referenced test files. F07.1 remains partial_native/pending_recipe_freeze/not_validated. Next bounded software gap: an explicit codec-direct grayscale reader path alongside the current RGB-derived default; preserve calibrated grayscale from corrected RGB. Remaining method/domain and all-recipe reference-output freeze, completed-goal full unittest, original SMPTS source and L01-L04 laboratory inputs stay open.

## Previous small-object ROI checkpoint (historical)
No test/GPU process remains live. Root owns I/O, edits and tests; reuse the same two workers and reviewer. Routine project edits remain authorized under auto_review; do not ask again. Direct workspace patches and narrow automatic shell review work. Do not inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. No source/test/artifact-script edits while test/GPU processes are LIVE. The fill ROI document updater and its finalizer already executed; do not repeat either.

Small-object ROI checkpoint (10 October). CUDA fill now filters independent nonempty half-open rectangles, counts4-connected objects locally and preserves exterior pixels. Twenty-eight additive seed/calibrated/debug variants keep historical definitions and independent erosion/dilation controls. Both desktop action orders, dragged layout, Keep/reopen, project batch, node-local caching and source-independent composed replay are verified.

125 pinned public mask-value comparisons pass. Public explicit-ROI replacement returns NumPy platform integer dtype; native BinaryMask remains uint8 with identical0/255 values. Typed-output parity for that source quirk is not claimed. Tiny ROI retains full-canvas admission. Strict callback/storage guards cover original/private buffers; the historical bridge now verifies the actual imported callback module aliases as well as helper code.

The20-test focused gate PASS400.342s and146-test affected gate PASS949.067s precede the final nonnumerical alias correction. The final19-test gate PASS425.616s,zero skips, retests all12 new ROI checks and the affected bridge/helper checks.166 distinct code-test IDs have passing executions across these gates; no single combined green166-test gate is claimed. Initial11 tests had10 passes/one layout-hash assertion failure and remain nonpassing history.

Eight final files are frozen in artifacts/fill-roi-frozen-hashes-v2.json; six stayed unchanged during the last two-file correction. Bounded review approves supplied projections/root execution. No independent reviewer reads/tests/visual inspection or new maximum-resource/scientific claim is made. All six final feature-register checks pass in0.853s,zero skips. Forty row mirrors,139 referenced test-file hashes,eight final frozen files and the unchanged canonical peer manifest are verified; scoped whitespace passes. Closure: artifacts/fill-roi-closure-v1.json.

F01-F22/WP0-WP6 remains active with40 registered operations,48 saved-stage replay operations and139 referenced test files. F07.2 remains partial_native/pending_recipe_freeze/not_validated. Next software gap: expose existing median ROI in working graph/replay controls while preserving boundary rejection rules. Remaining method/domain acceptance, all-recipe reference-output freeze and completed-goal full unittest stay open. Original SMPTS source and L01-L04 laboratory inputs remain external gates.

## Previous rectangular morphology checkpoint (historical)
No test/GPU process remains live. Root owns I/O, edits and tests; reuse the same two workers and reviewer. Routine project edits remain authorized under auto_review; do not ask again. Direct workspace patches succeeded; narrow shell automatic review handles the known Windows launcher error. Do not inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. No source/test/artifact-script edits while test/GPU processes are LIVE.

Rectangular morphology checkpoint (10 October). Erosion and dilation now expose null or explicit nonempty half-open ROI bounds in fourteen additive seed/calibrated/diagnostic working variants. CUDA filters the crop independently and replaces it in a full-canvas copy; exterior pixels stay unchanged. Kernel2..255, iterations1..64, node-local invalidation, dragged layout, Keep/reopen, project batch and source-independent composed replay are tested.

All104 public outputs match the verified PlantCV4.11.3/OpenCV4.11.0.86 wrapper/helper environment. Controls use helper slice endpoints; the public rectangle constructor stores x+w-1/y+h-1 endpoints, which its helper then treats as exclusive. Arbitrary contours and empty/out-of-frame rectangles are outside the native domain. Tiny ROI retains full-canvas admission. Shared callback/storage guards and caller-independent source recompilation are tested; no new maximum-resource or visual-inspection claim is made.

Focused26 tests PASS159.366s and repaired export24 tests PASS20.988s,zero skips. The broader171-test run had170 passes and one stale schema1 test assertion; it remains nonpassing history. The repaired assertion requires schema2 and the exact revision contract.197 distinct code-test IDs have passing executions across these gates; this is not a single combined green197-test run. The ten-file freeze is artifacts/morphology-roi-frozen-hashes-v1.json. Bounded review approved supplied projections/root evidence; no independent reads/tests/visual inspection.

All six feature-register checks pass in0.770s,zero skips. Scoped tracked/new-file whitespace checks pass. Closure: artifacts/morphology-roi-closure-v1.json. The document updater and its recovery already executed; never rerun either. Forty row mirrors,138 referenced test-file hashes, the ten-file tested freeze and unchanged canonical peer manifest are verified.

F01-F22/WP0-WP6 remains active:40 registered operations,48 saved-stage replay operations and138 referenced test files. F07.3 remains partial_native/pending_recipe_freeze/not_validated. Remaining operation/domain acceptance, all-recipe/reference-output freeze and completed-goal full unittest are open. Original SMPTS source and L01-L04 laboratory inputs remain external gates. Prior source freezes retain their historical scope; shared graph/helper/UI sources changed in this checkpoint.

## Previous prediction review revision checkpoint (historical)
No test/GPU process remains live. Root owns I/O, edits and tests; reuse the same two workers and reviewer. Routine project edits remain authorized under auto_review; do not ask again. Direct workspace patches succeeded without escalation this turn; shell reads/tests used narrow automatic review for the known Windows launcher failure. Do not inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. No source/test/artifact-script edits while test/GPU processes are LIVE.

Prediction review revision checkpoint (10 October). Review/report schema2 now binds instance confidences, candidate scores/selections and rejected manual-centre coordinates/reasons alongside the source, labels, settings and checkpoint. The same detached numeric snapshots supply identity and exported values. Changes produce a new revision; old decisions remain attached to their prior revision. Malformed compact facts reject before review attachment.

The final affected gate passes all33 tests in6.436s,zero skips, including10 new checks. Coverage includes actual desktop review loading, saved-project batch serialization and CUDA crop publication. The two-file change has bounded review approval based on supplied projections/root results; no independent reviewer reads/tests or visual inspection is claimed. The earlier10/23 passing gates overlap these33 tests.

All six feature-register checks pass in0.853s. Forty matrix rows,137 referenced test-file hashes, the two-file tested freeze and the unchanged ten-file Otsu freeze are verified; scoped whitespace passes.

F01-F22/WP0-WP6 remains active with40 registered operations,48 saved-stage replay operations and137 referenced test-file hashes. The omitted-prediction-fact software gap is closed; broader eligibility/method integration, remaining explicit operation/domain acceptance, all-recipe/reference-output freeze and fresh completed-goal full unittest verification remain open. Original SMPTS source and L01-L04 laboratory inputs remain external gates.

## Previous binary and Otsu checkpoint (historical)
No test/GPU process remains live. Root owns I/O, edits and tests; reuse the same two workers and reviewer. Routine project edits remain authorized under auto_review; do not ask again. Narrow retries handle the known Windows apply deny-read ACLs launcher failure. Do not inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. No source/test/artifact-script edits while test/GPU processes are LIVE.

All ten implementation hashes, forty inventory rows, 136 referenced test hashes and unchanged canonical peer identity are verified. Six feature-register checks pass in0.830s. Final review approves the ten-file freeze from supplied projections/root evidence; no independent reads/tests/visual inspection. Review record: artifacts/otsu-cutoff-review-checkpoint-v1.json. The one-run Otsu document updater has already executed; never rerun it or the historical updaters.

Binary and Otsu cutoff checkpoint (10 October). Fixed threshold now has512 exact public mask checks spanning every uint8 value, integer cutoff and polarity (131072 decisions), including a noncontiguous CUDA input. Otsu exposes native-selected cutoff facts from the same single histogram as its mask, preserving historical mask-only ports. Cache reuse, fact integrity, required v5 archive facts, strict v1-4 legacy loading and saved-input mask-plus-fact replay are implemented; older cutoffs are explicitly unrecorded.

The pinned reference adapter retains the actual public PlantCV mask and separately captures the cv2.threshold scalar after exact mask verification. Combined conformance requires both mask and cutoff equality; scalar differences fail even when masks agree. Sixteen Otsu mask cases and two exact scalar pairs pass. Constants, thin/singleton, imbalanced and empty-bin cases are covered; occupied ties/near-ties remain withheld. Normal async desktop facts and replay after deleting the source pass. No visual-inspection or new full-resolution/resource-domain claim is made.

The final15 focused checks pass in57.416s and128 affected checks pass in548.421s,zero skips. Freeze: artifacts/otsu-cutoff-frozen-hashes-v2.json (10 files). The initial13-test run had12 passes and one incorrect test result attribute; it remains nonpassing history. Bounded review uses supplied projections/root evidence only; no independent reads/tests/visual inspection.

F01-F22/WP0-WP6 remains active and partial_native, with40 registered operations,48 saved-stage replay operations and136 referenced test-file hashes. Next: remaining operation/domain acceptance and all-recipe/reference-output freeze. Original SMPTS source/license/dependencies, L01-L04 laboratory inputs and a fresh full unittest suite at completed-goal handoff remain open. Exact mask/cutoff evidence is not scientific validation.

## Previous adaptive threshold checkpoint (historical)
No test/GPU process remains live. Root owns I/O, edits and tests; reuse the same two workers and reviewer. Routine project edits remain authorized under auto_review; do not ask again. Narrow retries handle the known Windows apply deny-read ACLs launcher failure. Do not inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. All thirteen final hashes, forty inventory rows, 135 referenced test-file hashes and unchanged canonical peer identity are verified. All six feature-register checks pass in0.760s; scoped tracked/new-file whitespace passes. The supplied-projection/root-evidence review approves the thirteen-file freeze; see artifacts/adaptive-even-kernels-review-checkpoint-v1.json. Earlier adaptive freezes/gates retain their exact historical scope. No source/test/artifact-script edits while test/GPU processes are LIVE.

Adaptive threshold kernel checkpoint (9 October). CUDA mean accepts integer requests3..255 and Gaussian3..15, rounding even sizes upward to odd exactly as PlantCV does. Requested values remain in recipe/settings/cache identity; effective values drive actual computation and resource admission and appear in reference/native checkpoints, replay work details and desktop facts. Live tooltips explain the rule without changing the historical catalogue. Existing precision certificates and card-specific controls retain their domains.

The pinned public checks cover756 structured mean masks,54 thin/singleton masks,4 kernel-effect cases and72 Gaussian masks. Requested254 still withholds the scalar/SIMD rounding counterexample with no published checkpoint. Requested14 uses effective15 for Gaussian work/scratch limits. Normal async mean4 and Gaussian4-to14, distinct requested4/5 cache identity, exact reference/native selected/effective metadata, even4 conformance, and both Gaussian archived stages after deleting their origin are verified. A below-minimum typed edit rejects before execution and preserves valid cache; inconsistent recorded controls reject at export before an archive exists. No additional full-resolution even-kernel domain is claimed.

All39 final affected checks pass in157.063s with zero skips under artifacts/adaptive-even-kernels-frozen-hashes-v4.json (13 files). The earlier93-,39- and23-test runs remain nonpassing history; repaired fixtures and the strict conformance domain string did not weaken mask comparisons or precision guards. Review uses supplied projections/root evidence only; no independent reviewer reads/tests/visual inspection. Final log: artifacts/adaptive-even-kernels-final-green-v7.log.

F01-F22/WP0-WP6 remains active and partial_native. The register/matrix still covers40 operations and48 saved-stage replay operations, now with135 referenced test-file hashes. Next software work: remaining binary/Otsu method-domain acceptance and selected-cutoff adapter metadata, then complete operation/domain and all-recipe/reference-output freeze. Original SMPTS source/license/dependencies, L01-L04 laboratory inputs and fresh full unittest verification at completed-goal handoff remain open. No scientific or original-code parity is claimed.

## Previous image input checkpoint (historical)
No test/GPU process remains live. Final eleven focused, eighteen shared and six feature-register checks pass (zero skips); register checks take approximately 0.75s. Scoped whitespace, four reviewed hashes, all forty inventory rows, 134 referenced test-file hashes and unchanged canonical peer identity are verified. Python process count is zero. Root owns I/O, edits and tests; reuse the same two workers and reviewer. Routine project edits remain authorized under auto_review; do not ask again. The Windows sandbox launcher previously failed with apply deny-read ACLs, including reads. Narrow already-authorized retries use automatic review; authorization has not expired. Do not inspect/touch/stage analysis.seedfiddle-project.json; no staging/commits. New decoder probes and updater are one-run artifacts already executed; never rerun them or historical scripts. Full-goal verification and remaining domain work are open.

Image input and decoder checkpoint (9 October). Native recipes now decode the exact bounded immutable byte snapshot identified by source size/SHA-256. The separately pinned PlantCV 4.11.3 readimage(mode=rgb) contract supplies BGR uint8; CUDA derives RGB and pinned grayscale from the same color-decoded raster. Alpha dropping, 16-bit quantization, EXIF orientation and first-image behavior remain explicit. Cache identity binds canonical path/bytes/device, helper/preparation/converter/guard code and actual decoder versions/build/flags. Initial path-entry, fresh preparation and cached-return callbacks share strict bool, source, metadata and tensor checks. Detached source/frame/preparation facts prevent stale reuse; final source/code checks precede cache publication. Node-local recomputation remains intact.

The final eleven focused checks pass in 33.189s and eighteen compatibility checks pass in 107.482s, with zero skips. They cover source/cache replacement, cancellation phases, caller-exception identity, mutation priority, private CPU headers/pixels and CUDA buffers, all nine preparation arms, early model behavior, saved projects/batch, node-local reuse and actual async Qt paths. Four reviewed files remain frozen in artifacts/native-image-inputs-frozen-hashes-v1.json. Review approval uses supplied projections and root evidence; no independent reviewer reads, tests or visual inspection.

The saved 32-row public reader matrix has 27 successful exact full BGR codec comparisons and four matching invalid-input outcomes. CUDA RGB/gray are exact on the 23 small public-success cases plus the four additional committed/long-row/singleton cases. PNG/JPEG/BMP/TIFF, gray/palette/alpha/depth variants, JPEG EXIF 2–8, CMYK JPEG, first TIFF page and misleading extensions are covered. The remaining Unicode filename is an explicit native path extension: pinned OpenCV4.11 Windows imread fails that path, while Python byte loading succeeds and matches the identical ASCII-path codec input. Path-outcome parity and other readimage modes are not claimed.

Six fresh preparation children start at zero allocated CUDA bytes. The unchanged 2080x3120 image peaks at 47,000,064 bytes within a 48,572,928-byte preparation envelope and 330,820,096-byte graph admission, both with no callback and a strict False callback. The unchanged 4160x6240 image peaks at 183,281,664 within 184,854,528 and 973,290,496 respectively. A 1x65553 row and one pixel also match. The generated 7000x8000 image decodes but exceeds the actual graph byte budget, with zero CUDA uploads and zero peak allocation. Traced host peaks are 11,449,487–179,440,754 bytes; encoded caps, traced peaks and conservative preparation envelopes are not decoder memory/process RSS guarantees or maximum-domain proof.

Root inspected normal async source-stage views at 1440x900 and 1050x650 using explicitly registered Windows Segoe UI 9. Provided status, source hash/frame and the image preview are readable; repeated runs reuse prepared tensors and all completed nodes. The selected CJK filename glyph is a square in the offscreen font label, so this evidence does not claim all Unicode font rendering. Numeric/cold probe v1 stopped afterward on a wrong provided-versus-executed source-node assertion; functional UI v2 captures lacked registered fonts. These remain nonpassing/superseded history; isolated UI v3 and final evidence are artifacts/native-image-inputs-final-evidence-v3.json. Application code was unchanged by the probe repairs.

There are still 48 declared saved-stage replay operations and 40 registered operation rows. This checkpoint closes the selected decoder matrix, not complete method/input-domain acceptance. Next: execute the remaining operation/domain inventory and final all-recipe/reference-output freeze. Original SMPTS source/license/dependencies, L01–L04 laboratory inputs and a fresh full unittest suite at completed-goal handoff remain open. F01–F22/WP0–WP6 stays active and partial_native; no original-code or scientific parity is claimed.

## Previous crop checkpoint (historical)

No test/GPU process remains live. The crop checkpoint below has bounded review approval. All six feature-register checks pass in 0.742s; scoped tracked/untracked whitespace, five reviewed hashes, 48-operation replay coverage, all 40 method inventory rows and canonical peer identity are verified. Python process count is zero. Root owns I/O, edits and tests; reuse the same two workers and reviewer for bounded text work. The crop updater, one-run probe and method-inventory generator have already run; never rerun them or historical one-run scripts. Do not inspect/touch/stage analysis.seedfiddle-project.json. No staging or commits. Routine project edits remain authorized under auto_review; do not ask again. The Windows sandbox launcher previously failed with apply deny-read ACLs, including reads. Narrow already-authorized retries use automatic review; edit authorization has not expired.

Calibrated seed crop replay checkpoint (9 October). calibrated_seed.crop now replays only verified saved corrected-RGB, source-frame, card-geometry and card-ROI inputs. There are 48 declared replay operations. Exact half-open seed bounds, controls, direct and card-ancestor producer keys, source/parent/local frames, raster headers and fixed metadata bind before upload. Original source images and upstream correction/detection are not opened or recomputed. Both local CUDA rasters and every metadata field are compared; a bounded but wrong overlap count is a numerical mismatch.

The full 13-test crop gate passes in 259.160s with zero skips. The broader 68-test compatibility gate has 67 passes and one public cancellation-exception error in 745.081s and remains nonpassing history. A narrow exception-only adapter restores AutomaticCardCancelled; all six affected checks pass in 164.153s under the final v4 freeze. They include initial/mid/final cancellation, caller-exception identity, mutation priority, saved-input guards, all eight calibrated/diagnostic graphs and actual async Qt replay. Numerical calculations are unchanged by the exception adapter; a fresh full unittest suite remains required at the completed-goal handoff.

Six independent pinned OpenCV 4.11.0/NumPy 2.2.6 oracles match saved-input RGB slices, grayscale bytes, inclusive overlap counts and all frame/translation metadata. Six fresh-process replays start with zero allocated CUDA bytes and fit their plans, including the first runtime audit. The 2080x3120 full crop peaks at 59,008,000 bytes within 88,244,736 planned; its 96x128 crop peaks at 19,912,704 within 20,435,456. Full-width rows, partial/shifted equal-size crops and one pixel also match. Contiguous row/full/singleton slices may alias admitted parent storage; admission conservatively charges a fresh RGB copy. Traced host peaks are 24,602,703–212,857,363 bytes, not process RSS limits or maximum-domain proof.

The large parent is a generated 2080x3120 fixture made from the committed image plus a controlled card/seed patch; committed images are unchanged. Median is intentionally unselected in the full/alias/singleton source runs, so the crop executes while those whole graphs remain partial. The ordinary partial/shifted seed graphs execute completely. Arbitrary parent IDs/transforms are covered by the focused tests, rather than the cold probes.

Two root-inspected normal async initial-run/stage-selection/replay Qt captures at 1440x900 and 1050x650 use Segoe UI 9. Bounds, local-frame translation, overlap and RGB previews are readable; active results and archive hashes are preserved. Evidence: artifacts/native-crop-replay-probe-v1/evidence.json. Five-file freeze: artifacts/native-crop-replay-frozen-hashes-v4.json. Bounded review: artifacts/native-crop-replay-review-checkpoint.json; approval uses supplied projections and root evidence, with no independent reviewer reads, tests or visual inspection.

All 40 registered F01–F22 operation rows are mapped in docs/PLANTCV_SMPTS_METHOD_DOMAIN_MATRIX.md and its JSON inventory, with contracts, source functions, native mappings, domain axes, test candidates and required acceptance work. This is an audit map, not completed method/domain acceptance. Next engineering work is the decoding/channel/orientation/format freeze, explicit remaining method-domain acceptance and final recipe/reference-output freeze. Original SMPTS source/license/dependencies, L01–L04 laboratory inputs and the completed-goal full suite remain open. F01–F22/WP0–WP6 stays active and partial_native; reconstruction agreement does not establish original-code or scientific parity.

Earlier checkpoints follow; their operation counts, freezes and evidence retain historical scope.

## Previous paper checkpoint (historical)

No test/GPU process remains live. The paper checkpoint below has final bounded review approval. All six feature-register checks pass in 0.745s; scoped tracked/untracked whitespace, four reviewed hashes,47-operation coverage and canonical peer identity are verified. Python process count is zero. Root owns I/O, edits and tests; reuse the same two workers and reviewer for bounded text work. The paper checkpoint updater has already run; never rerun it or historical one-run scripts. Do not inspect/touch/stage analysis.seedfiddle-project.json. No staging or commits. Routine project edits remain authorized under auto_review; do not ask again. The Windows sandbox launcher has failed with apply deny-read ACLs, including reads. Narrow already-authorized retries use automatic review; this is an execution failure, not expired edit authorization.

Paper saved-stage replay checkpoint (9 October). Eight provisional smpts_paper.* operations now replay all nine calculated paper stages, including both component stages, bringing the declared replay count to 47. Exact controls, producer keys, typed ports, raster headers, coordinate frames, inventories, contact IDs, crop bounds/offsets/parents and provisional metadata are bound before upload. Only the selected stage executes; original image paths and ancestor calculations are not opened or recomputed. Packed crops may exceed source pixel count because component boxes can overlap. Contacts uploads its direct label dependency, even though the calculation uses its shape.

The final calculation gate passes all 13 paper tests in 198.543s with zero skips. The shared 53-test gate has 52 passes and one tuple/list fixture assertion failure; the repaired assertion passes in 15.240s. Those logs retain their respective scope. Ten affected contract/desktop checks pass in 46.684s after the selector repair. The final contrasting-active/archive Qt regression passes in 7.660s after failure/cancellation coverage and visible-label cleanup. Replay crop selection reads replay buffers; node reselection restores active crops, and failed/cancelled replay hides the selector. Calculation code is unchanged from the final 13-test and cold gates.

Fourteen fresh-process replays match from zero allocated CUDA bytes and fit their allocation plans, including the first runtime audit. The committed 2080x3120 fixture has 503 source components, 31 selected parents and 1118 final crops. An isolated pinned OpenCV 4.11.0.86/NumPy 2.2.6 oracle matches all nine rasters, all packed RGB/mask crops and their inventories/lineage. Full mean replay peaks at 242,178,048 bytes within 913,182,208 planned; packed crops peak at 142,010,368 within 209,440,768. Controlled, singleton, empty and overlapping-box cases also pass. Traced host peaks are 21,383,714-292,383,624 bytes; these are not process RSS limits or maximum-domain evidence.

Six final actual async Qt views at 1440x900 and 1050x650 use the normal initial run and stage-selection handlers with Segoe UI 9. Root inspected contact/crop summaries, coloured labels, individual replay crops and corrected button text; active runs and archives are preserved. Final freeze: artifacts/native-paper-replay-frozen-hashes-v8.json. Oracle/cold evidence: artifacts/native-paper-replay-final-probe-v2/evidence.json. Final visuals: artifacts/native-paper-replay-final-ui-v5/evidence.json. Bounded review: artifacts/native-paper-replay-review-checkpoint.json; the reviewer used supplied projections/root evidence and performed no independent reads, tests or visual inspection. Earlier failed schema adapters and superseded captures remain historical evidence.

F01-F22/WP0-WP6 remains active and partial_native. Next: calibrated_seed.crop saved-input replay, the complete registered method/input/domain matrix and final recipe freeze. Original SMPTS source/license/dependencies, L01-L04 laboratory inputs and fresh completed-goal full unittest discovery remain open. Reconstruction agreement does not establish original-code conformance or scientific validation.

Next calibrated crop contract design is supplied by the same two workers; it is not implemented yet. Bind canonical corrected-RGB/source-frame/card-geometry/card-ROI producers and exact nonempty half-open seed ROI controls. Preserve valid parent IDs/transforms and derive the distinct local frame plus translation; equal-size shifted ROIs must remain distinguishable. Account for unique archived full-parent RGB, expected local RGB/gray, possible full-width aliases, conservative fresh crop storage, source-derived tiled grayscale scratch, comparison scratch and first-audit reserve. Read the actual conversion helper before filling scratch coefficients. Focused cases: full frame, full-width rows, partial width, one pixel, shifted same-size ROI, malformed producer/frame/geometry/control identity, valid numeric mismatch, deleted origins, cancellation/mutation, cold admission and normal Qt active-result preservation. Replay must not recompute upstream correction or card geometry.


No test/GPU process remains live. The 60-test shared gate passes in 607.405s; two final added checks pass in 39.583s. Four independent source cases, twelve cold replays and six root-inspected Qt renders pass. All six feature-register checks pass in 0.726s; scoped tracked/untracked whitespace passes. Seven final hashes and canonical peer identity are verified. Root owns I/O/edits/tests; reuse the same two workers and reviewer for bounded text work. The final-stages updater has already run; do not rerun it or historical scripts. Do not inspect/touch/stage analysis.seedfiddle-project.json. No staging or commits. Routine project edits remain authorized under auto_review; do not ask again. The Windows sandbox launcher has failed with apply deny-read ACLs; narrow authorized retries use automatic review, not renewed user consent.

Recovery partition, annotation and measurements replay checkpoint (9 October). All six experimental `recovery.*` operations now support verified saved-input replay, bringing the declared operation count to 39. The four generic upstream calculations also replay, covering all ten calculated nodes in the recovery graph. Partition preserves nine typed CUDA products and input/output aliases; annotation preserves the shared 0/1 uncertainty band; measurements selects zero arrays and consumes archived annotation metadata. Original image paths and ancestor calculations are not executed. Measurement replay explicitly reports that upstream pixels and statistics are not recomputed.

The affected 60-test gate passes in 607.405s with zero skips. Two added final checks pass in 39.583s: mutations of every nested partition tensor followed by a caller exception withhold publication, and all four upstream stages match. Calculation source is unchanged after the shared gate. Coverage includes exact raster/compact comparisons, malformed aliases/headers/encodings, finite elevation, all uncertainty radii, both policies, disappeared IDs, empty/singleton/unseeded inputs, source/cache behavior, coordinated frozen-record/snapshot replacement and actual recovery/generic/card/model Qt replay. The initial 15-test gate had 13 passes and two fixture defects; it remains nonpassing history. The repaired four-test gate passes in 44.453s.

Four independent controlled reference cases pass: 259x513 full/ROI touching cores, 1x1 singleton and 2x3 empty input. Pinned public PlantCV threshold/erosion, SciPy four-connected IDs, OpenCV chamfer and skimage watershed match all partition arrays. NumPy boundary records/traits and SciPy four-neighbor dilation match annotation and measurement records. Full support retains three active IDs; ROI retains two plus a disappeared inventory row. Both have two interface pixels. The withhold/flag policies permit one/two cores respectively. This is computational engineering evidence, not original SMPTS or scientific validation.

Twelve fresh-process replays match from zero allocated CUDA bytes, selecting 9/10/0 arrays for partition/annotation/measurements. The first audit's 13,312 retained bytes are included. Full/ROI partition peaks at 4,405,248 bytes within 17,869,312 planned; annotation at 3,739,648 within 12,313,600. Measurements peaks at 29,184 within its 65,536-byte audit reserve. Singleton/empty partition peaks are 286,720/284,672 and annotation 34,816/34,304, all admitted. Traced host peaks are 19,663,770–22,520,649 bytes; encoded caps are not RSS guarantees. Partition admission uses conservative D=P before upload; active IDs L and original inventory I remain distinct. The downstream source limit is 8,064 objects, while the existing whole-report 8 MiB/one-million-traversal archive limits remain separate from the 64 MiB per-product replay decoder. Maximum whole-archive capacity is not established.

Six actual async Qt renders at 1440x900 and 1050x650 use Windows fonts and the normal source setter. Root inspected distinct partition colours, the nonzero uncertainty preview, eligibility/unavailable observations and source captions; active runs and archives are preserved. Evidence: `artifacts/native-recovery-final-stages-probe-v2/evidence.json`; final seven-file freeze: `artifacts/native-recovery-final-stages-frozen-hashes-v2.json`; bounded review: `artifacts/native-recovery-final-stages-review-checkpoint.json`. The reviewer approved supplied projections and root execution/visual evidence, with no independent reads, tests or visual inspection. Earlier freezes and the merged-core first probe retain historical scope.

F01–F22/WP0–WP6 remains active and partial_native. Next: the eight provisional paper-helper operation contracts (including both component stages), the complete registered method/input/domain matrix and final recipe freeze. Original SMPTS source/license/dependencies, L01–L04 laboratory inputs and fresh completed-goal full unittest discovery remain open.

Earlier recovery clipping checkpoint follows.

No test/GPU process remains live. The focused clipping gate passes 11 tests in 27.121s. The shared gate has 115 passes and one UI failure in 908.560s (nonpassing history); the repaired desktop gate passes all four tests in 29.839s. Five cold probes and two root-inspected Qt renders pass. All six feature-register checks pass in 0.730s after reconciliation; scoped tracked/untracked whitespace checks pass. Seven frozen source/test hashes and canonical peer identity are unchanged. Root owns I/O/edits/tests; reuse the same two workers and reviewer for bounded text work. The clipping checkpoint updater has already run; do not rerun it or historical scripts. Do not inspect/touch/stage analysis.seedfiddle-project.json. No staging or commits. Routine project edits remain authorized under auto_review; do not ask again.

Recovery clipping saved-stage replay checkpoint (9 October). `recovery.clip_cores` now replays verified archived markers and validity in the full source frame. There are 36 verified declared replay operations, including three experimental recovery stages. Every clipped marker pixel is compared; surviving sparse IDs remain unchanged. Canonical producer keys, controls, headers, core inventory and source/ROI metadata are bound before upload. Ancestor numerical calculations and original image paths are not executed or opened. Clipping retains the upstream 32,768-core domain and does not impose the downstream 8,064-object metadata limit.

Recovery object records, resource plans and final annotation statuses now use detached primitive-field snapshots. Private captured baselines also detect replacement of a retained snapshot during a callback. In-place frozen-field changes, including changes followed by caller exceptions, withhold publication. The first focused gate passes 11 tests in 27.121s. The seven-module shared gate has 115 passes and one desktop-summary failure in 908.560s; it remains nonpassing history. The summary was hidden for a raster-only output and was repaired by rendering available summaries. All four affected actual async Qt replay tests then pass in 29.839s. The broader source/calculation checks include full-resolution recovery, cache reuse, generic/card/model replay and all nine card variants. Fresh full unittest discovery remains required at completed-goal handoff.

Five controlled input/reference cases pass: 259x513 full/ROI, 1x1 singleton, 32,768 sparse cores in a 1x98,304 image and all-invalid saved support. Pinned public PlantCV threshold/erosion and vectorized SciPy four-connected first-pixel IDs establish the core inputs; NumPy conditional clipping supplies the independent expected result. The all-invalid case explicitly consumes saved support without recomputing upstream ROI pixels. Five fresh processes start with zero allocated CUDA bytes and select three unique direct arrays. Full/ROI/all-invalid peaks are 2,940,416 bytes within 2,991,616 planned; singleton is 31,232 within 69,120; maximum inventory is 2,176,512 within 2,228,736. The first runtime audit's 13,312 bytes are included. Traced host peaks are 17,531,643–60,219,099 bytes; encoded caps are not RSS guarantees. This is controlled engineering evidence, not complete raster-domain, original SMPTS or scientific validation.

Two actual async Qt renders at 1440x900 and 1050x650 use Windows fonts and the normal source setter. Root inspected the readable clipping summary, full-frame preview and source caption. Active runs and recursive archive hashes are preserved. Evidence: artifacts/native-recovery-clip-probe-v1/evidence.json; seven-file freeze: artifacts/native-recovery-clip-frozen-hashes.json; bounded review: artifacts/native-recovery-clip-review-checkpoint.json. Review approves supplied projections and root execution/visual evidence only; no independent reads, tests or visual inspection are claimed. The earlier core/validity 93-test gate and six cold probes remain historical in artifacts/native-recovery-stage-replay-probe-v1/evidence.json after the shared recovery source changed.

F01–F22/WP0–WP6 remains active and partial_native. Next: partition, annotation and measurement replay, paper helper contracts, the complete registered method/input/domain matrix and recipe freeze. Original SMPTS source/license/dependencies, L01–L04 laboratory inputs and fresh completed-goal full unittest discovery remain open. Routine project edits remain authorized under auto_review; do not ask again.

Next partition admission note: keep domain pixels D, active IDs L and original inventory count I distinct. Archived D/L plan fields are claims, not permission to exceed a lower replay cap. Use a conservative D=P budget before upload unless a separate guarded replay-specific actual-count admission runs before large allocations. Charge unique archived array aliases, preserve inner/separate label aliases, validate 0/1 domain/interface/unresolved masks separately from 0/255 foreground/validity, and reconstruct only known transient recovery records with fresh snapshots. The remaining three recovery stages have not been implemented by this checkpoint.

Earlier colour-card checkpoint follows.

Colour-card saved-stage replay checkpoint (9 October). All eight `color_card.*` stages now replay verified archived direct inputs: BGR, preprocessing, geometry, masks, scale, means, standard targets and affine correction. There are 33 verified declared replay operations. Canonical producer keys, controls and source/ROI frames bind nine card graphs: standalone, four calibrated and four calibrated diagnostic variants. The three plain seed diagnostic variants contain no card stages; the earlier foundation planning statement of seven card diagnostic compositions was incorrect.

The final affected ten-module gate passes 129 tests in 725.227s with zero skips, including22 card integration tests. It verifies every raster/compact output, deleted origins, all nine variants, explicit full-bounds ROI identity, label validation before uint8 narrowing, strict/final cancellation and private CUDA/CPU/header/field mutation guards. The earlier181-test shared gate had179 passes and two public mutation-error-family failures; it is nonpassing history. Those failures were repaired at the public detector/correction boundary and passed the focused3-test gate in5.090s before the final129-test gate. Calibrated/range/diagnostic/cache/UI regressions passed in the earlier shared gate. The repair preserves public AutomaticCardError compatibility, unrelated exceptions and numerical calculations.

Controlled620x460 full-frame and380x540 ROI workflows agree with pinned public PlantCV preprocessing, geometry, masks, scales and standard targets. Chip means use2e-15 absolute tolerance; the declared affine pixel tolerance is one byte, with zero difference in both probes. All sixteen fresh-process stage replays match from zero allocated CUDA bytes and fit their per-stage plans, including the first audit. Scale selects no arrays and peaks at29,184 bytes within65,536; standard selects one855,600-byte BGR array and peaks at885,248 within929,792. The largest observed tensor-allocation peak is19,490,304 bytes for affine within20,529,664 planned. Traced host peaks range15,952,201–20,284,769 bytes; encoded caps are not RSS guarantees. These are controlled engineering domains, not maximum-domain or scientific evidence.

Sixteen actual async Qt renders at1440x900 and1050x650 use Windows fonts and the normal source setter, preserve active runs/archives and retain correct requested sizes. Root inspected all eight operations represented across both sizes: compact summaries and raster previews. Initial renders with a directly assigned source path are superseded caption evidence. Source/recipe/runtime/oracle/probe/output/frame and recursive archive hashes are recorded in artifacts/native-card-stage-replay-probe-v1/evidence.json. Ten-file freeze: artifacts/native-card-stage-replay-frozen-hashes.json. Bounded review approves supplied projections and root execution/visual evidence only; no independent reads, tests or visual inspection are claimed. Earlier source freezes retain historical scope when shared modules change.

F01–F22/WP0–WP6 stays active and partial_native. Next: recovery core-label/validity replay, remaining recovery and paper helper contracts, then the complete registered method/input/domain matrix and recipe freeze. Original SMPTS source/license/dependencies, L01–L04 laboratory inputs and fresh completed-goal full unittest discovery remain open. Routine project edits remain authorized under auto_review; do not ask again.

No test/GPU process remains live. All six feature-register checks pass in 0.702s after reconciliation. The fifteen final frozen source/test hashes and canonical peer manifest remain unchanged; scoped tracked/untracked whitespace checks pass. Root owns I/O/edits/tests; the reusable two-worker/reviewer pool supplied bounded text design/review. The model-stage checkpoint update script has already run and must not be rerun. Do not inspect/touch/stage analysis.seedfiddle-project.json. No staging or commits.

Archived-model saved-stage replay checkpoint (9 October). `analysis.naive_bayes_classifier`, `analysis.kmeans_predict` and `analysis.kmeans_masks` now replay verified archived inputs. There are 25 declared verified replay operations. Bayes TSV and legacy/provenance K-means model bytes remain host-only; original model paths are not resolved or opened. Prediction binds canonical image/model ancestry and both image headers while selecting only the active raster. Masks replay verifies label/model lineage and identities without recomputing upstream numerical predictions or parsing raw model bytes. Every returned mask pixel, ordered class key/name and model-identity leaf is compared on its typed contract.

The final 11-module focused gate passes 110 tests in 160.816s with zero skips. It covers public primitive comparisons, source/model/private CUDA buffer and nested-output mutation guards including callback exceptions, strict/final cancellation, deleted origins, both model formats, absent/duplicate/large categories, malformed lineage/headers, numerical mismatches, node-local caches and actual desktop replay. The actual 56-class small-mask regression admits allocator rounding. The 76-test earlier gate is historical; an initial fixture/module gate and the first three probe attempts are nonpassing history. The third probe exposed and led to correction of the comparison reduction's int64 temporary budget. Saved archive integers retain their existing 256-bit bound; direct category calculation permits larger absent IDs.

Controlled source/public/reference probes: Bayes259x513 matches all three pinned public masks and398,616 replay values, selecting834,788 bytes. Replay adds3,303,936 CUDA bytes within19,654,144 planned;944,128 preexisting bytes are separate. Gray131x259/p2/legacy and RGB131x259/p3/envelope K-means labels and all masks match pinned public PlantCV. Predictions add1,175,552/3,570,688 CUDA bytes within68,226,560/68,817,920 planned. Masks select135,716 bytes, compare101,803 values and add546,816 within680,960 planned. A separate fresh-process masks replay starts at zero allocated and adds560,128 bytes, including its first runtime audit. All classes in these probes are nonempty. Traced host peaks are13-14MiB; encoded/raw metadata caps do not guarantee RSS. These are controlled engineering domains, not corpus, maximum-domain or scientific evidence.

Source, recipe, model, archive, output, runtime and oracle-script hashes plus six actual async Qt views at1440x900/1050x650 with explicitly loaded Windows fonts are recorded in artifacts/native-model-stage-replay-probe-v4/evidence.json. Root inspected the class selectors, masks and coloured label previews; active runs and archives were preserved. The initial font-missing renders remain superseded probe evidence. Final source/test freeze: artifacts/native-model-stage-replay-frozen-hashes.json; gate: artifacts/native-model-stage-replay-final-shared-gate.log. Bounded review approves the supplied fifteen-file freeze using supplied projections/root execution and visual evidence; the reviewer performed no independent reads/tests/visual inspection. Earlier freezes retain historical scope when shared modules change.

F01-F22/WP0-WP6 remains active and partial_native. Next software work: remaining card, recovery and paper helper replay, then the complete method/input/domain matrix and recipe freeze. Original SMPTS source/license/dependencies and L01-L04 laboratory gates remain open. Fresh full unittest discovery is required at completed-goal handoff. Routine project edits remain authorized under auto_review; do not ask again. The Windows sandbox launcher may fail with `apply deny-read ACLs`; narrow already-authorized tool retries use automatic review. This is an execution setup failure, not renewed project-edit authorization.

Earlier verified checkpoints follow.

Archived-model replay foundation (9 October). Bounded primitive contracts now bind Bayes TSV and both legacy/provenance K-means JSON encodings to archived byte length/SHA, parsed numerical identity, source selection and ordered class-mask headers. Original model path text is validated without resolving or opening it. Native K-means prediction and replay preparation share allocation-free shape/work/storage planning; replay charges active image, RGB-to-BGR conversion, expected/actual products, primitive buffers, comparison tile and audit reserve separately. Raw model/metadata caps do not guarantee process RSS.

No test/GPU process remains live. The final six feature-register checks pass in 0.690s after document reconciliation; all four reviewed hashes and the canonical peer manifest remain unchanged, and scoped whitespace checks pass. Root owns I/O/edits/tests; the reusable two-worker/reviewer pool supplied bounded text designs/review. Do not inspect/touch/stage analysis.seedfiddle-project.json. No staging or commits. The foundation-update script has already run and must not be rerun.

The final focused five-module gate passes 53 tests in 27.355s with zero skips, including 19 new foundation tests and existing pinned public prediction/preprocessing, cache/API and actual desktop classifier tests. Four supplied frozen source/test hashes have bounded review approval using root execution evidence; no independent reads/tests are claimed. Evidence: artifacts/native-model-replay-foundation-reviewed-gate.log, native-model-replay-foundation-frozen-hashes.json and native-model-replay-foundation-review-checkpoint.json.

This is a helper foundation. Saved-stage classifier dispatch, producer/frame binding, replay callback/private-buffer guards, every-mask comparison and replay UI integration remain unfinished. The verified replay count remains 22. Next: connect these contracts for analysis.naive_bayes_classifier, analysis.kmeans_predict and analysis.kmeans_masks, then complete remaining card/recovery/paper replay and the method/input/domain matrix. F01-F22/WP0-WP6 remains active and partial_native; original SMPTS and laboratory gates plus fresh completed-goal full discovery remain open. Routine project edits remain authorized under auto_review; do not ask again.

Earlier verified checkpoints follow.

Size and units saved-stage replay checkpoint (9 October). `analysis.size` now replays original RGB/labels and all seventeen observation records in source or verified seed-ROI frames, comparing the optional CUDA diagnostic separately. Exact source descriptors, nominal datatypes and finite local coordinates are preserved. `calibrated_seed.size_units` replays the compact physical conversion with zero selected arrays and binds the saved size/frame and card-scale producers. Its conversion allocates no CUDA tensors; the shared runtime audit has a separately admitted 64 KiB reserve. Upstream label maxima and card measurements are not recomputed by the units stage. There are now 22 declared replay operations.

The shared 15-module gate passes 185 tests in 1155.051s before the final audit-budget refinement. The final affected size/unit and existing planner/contract gate passes 28 tests in 86.761s after that refinement (zero skips; overlapping gates). Nine final source/test files have bounded supplied-projection/root-evidence review approval. Earlier freezes remain historical when shared modules change.

The controlled direct-input 1027x1025 graph uses 128 actually computed labels with independently prepared, hashed gray/RGB arrays; it makes no single-image decoding claim and changes no upstream outputs. All seventeen size variables agree with pinned public PlantCV under the declared 1e-6 numerical tolerances and ten-decimal longest-path check; the diagnostic agrees byte-for-byte. The public explicit-one empty-mask behavior agrees too. Source replay selects three arrays (10,526,750 bytes), compares 3,171,593 values and adds 21,663,232 CUDA bytes within 268,960,128 planned; 13,312 preexisting bytes are separate. Whole source replay traces 63,354,720 additional peak host bytes. Physical/pixel units select no arrays and add 15,872/15,872 CUDA bytes within the separate 65,536-byte shared-audit reserve; an isolated cold audit measured 29,184 bytes. Encoded metadata caps do not guarantee Python RSS. These are controlled engineering domains, not maximum-domain, corpus or scientific evidence.

Freeze: artifacts/native-size-replay-frozen-hashes.json; final gates: artifacts/native-size-replay-final-shared-gate.log and native-size-replay-final-gate.log; source/oracle/recipe/archive/output/runtime hashes and actual Qt views: artifacts/native-size-replay-probe-final-v2/evidence.json. Actual async size/unit summaries and the size image were inspected at 1440x900 and 1050x650 with Windows fonts; the active run and every archive were preserved. Review uses supplied projections and root execution/visual evidence, with no independent file reads/tests/visual inspection claimed.

No test/GPU process remains live. The six final feature-register checks pass in 0.681s after replacing bare test-file references with concrete test symbols. Scoped tracked/untracked whitespace checks pass; all nine frozen hashes and the canonical peer source manifest remain unchanged. Routine project edits remain authorized under auto_review; do not ask again. Do not inspect/touch/stage analysis.seedfiddle-project.json. No staging/commits. Reusable two-worker/reviewer pool has supplied the next model replay design. Workers remain text-only; root handles I/O, edits and tests. The first two test gates, initial probe failures and the initial register-reference failure remain nonpassing historical evidence; see their logs. Final units preflight rejects a below-reserve cap before runtime construction, selected loading or dispatch.

The F01-F22/WP0-WP6 goal remains active and partial_native. Next: archived-model classifier replay, then remaining card, recovery and paper helper contracts and the complete method/input/domain audit. Original SMPTS source/license/dependencies and L01-L04 laboratory gates remain open. Fresh full unittest discovery is required before completed-goal handoff.

Earlier verified checkpoints follow.

Colour saved-stage replay checkpoint (9 October). `analysis.color` now supports typed saved-input replay in source and verified calibrated seed-ROI frames. All RGB/LAB/HSV/ALL histograms, bins, statistics, label records and returned hue-chart fields are compared; explicit nonfinite tags preserve NaN/inf outcomes. The native Qt view shows the returned chart and concise observations while preserving the active run. There are now 20 declared replay operations.

The shared 15-module gate passes 179 tests in 979.683s before the final character-length precheck; the final affected colour gate passes 14 tests in 125.179s after that refinement (zero skips; overlapping gates). Six final source/test files have bounded supplied-projection/root-evidence review approval. Earlier freezes remain historical when shared modules change.

The controlled 257x513 selected-port fixture uses 128 labels and ALL, crossing two full tiles plus 769 pixels. All 634,380 compact leaves match saved-stage replay. Actual pinned public histograms/bins agree exactly; finite statistics use the declared 1e-10/1e-5 circular-std tolerances. Replay adds 12,357,120 CUDA bytes within 151,093,376 planned, with 13,312 preexisting separate; selected arrays total 922,887 bytes. Host compact parsing/decoding/snapshots/comparison trace 44,363,468 additional peak bytes; whole replay traces 75,029,055. Process working-set/lifetime-peak readings include imports and the CUDA runtime. Encoded metadata caps do not promise Python RSS. This fixture verifies colour on controlled selected inputs, not its modified upstream watershed history, maximum raster domain, corpus or scientific accuracy.

Freeze: artifacts/native-color-replay-frozen-hashes.json; gates: artifacts/native-color-replay-shared-gate.log and native-color-replay-final-gate.log; source/oracle/recipe/output hashes and actual Qt views: artifacts/native-color-replay-probe-final/evidence.json. No test/GPU process remains live. Do not inspect/touch/stage analysis.seedfiddle-project.json. No staging/commits. Routine project edits remain authorized under auto_review; do not request the same approval again.

F01-F22/WP0-WP6 remains active and partial_native. Next: typed size/unit observation replay, then remaining model, card, recovery and paper helper contracts and the full method/input/domain audit. Original SMPTS source/license/dependencies and laboratory inputs remain separate open gates. Fresh full unittest discovery is required before completed-goal handoff.

Earlier checkpoint history follows.

Calibrated range workflow checkpoint (9 October). The additive `plantcv_calibrated_seed_range` graph corrects the colour card, crops an explicit seed ROI, applies channel-range selection, smooths its binary mask, and executes fill/morphology/watershed plus size/colour observations. Traits use the original corrected cropped RGB; the range masked RGB is a pre-median diagnostic. This is a user-selected composition of pinned operations, not a recovered historical fixed-range workflow or scientific validation.

The final 11-module gate passes 143 tests in 647.331s, including 11 new integration tests, with zero skips. All nine prior calibrated/diagnostic recipe serializations remain exact. Tests cover five spaces plus partial ranges, complete pinned corrected-ROI outputs/physical observations, range/median/ROI/units cache boundaries, empty-result recovery, fresh/cached/debug admission, excess input backing, final callback/runtime mutation, actual Qt replay/Keep/reopen and saved-project/batch/archive round trips.

Read [all requested feature statuses](docs/FEATURE_IMPLEMENTATION_STATUS.md). Six-file freeze: artifacts/calibrated-range-frozen-hashes.json; gate: artifacts/calibrated-range-shared-gate.log; controlled source/reference/recipe/output hashes and actual Qt views: artifacts/calibrated-range-probe/evidence.json. The 620x900 synthetic source uses a 96x128 ROI at (600,260), selecting1571 pixels. Range adds20,045,824 CUDA bytes within322,003,456 planned;10,765,312 preexisting bytes are separate. Range/debug and binary probes independently fit their bounds; differing sequential cold/warm baselines do not establish comparative performance. Saved range/median replay selects86,016/24,576 bytes, compares49,152/12,288 values and preserves archive hashes. Bounded review approves supplied projections and root execution/visual evidence only; no independent reads/tests/visual inspection. No tests/GPU processes remain live. Do not inspect/touch/stage analysis.seedfiddle-project.json. No staging/commits. Next: bounded typed colour-observation replay, then remaining operation/helper domains.

F01-F22/WP0-WP6 remains active and partial_native. Remaining model/helper/paper/observation replay and the full method/input/domain freeze remain open, alongside original SMPTS source and laboratory gates. Fresh full unittest discovery is required before completed-goal handoff. Earlier source freezes retain historical scope when shared modules change.

After reconciliation, all six feature-register checks pass in 0.687s; scoped whitespace checks pass. The six reviewed source/test hashes and canonical peer manifest remain unchanged. No task test/GPU process remains live. Routine project edits remain authorized under auto_review; do not request the same approval again.

Earlier checkpoint history follows.

Custom-range graph and saved-stage replay checkpoint (9 October). The separate `plantcv_threshold_custom_range` recipe now executes the existing pinned CUDA primitive with real RGB/HSV/LAB/CMYK/GRAY and inclusive byte-bound controls. Both the binary mask and masked original RGB are retained, exported and replayed. RGB bounds use pinned B,G,R storage-slot semantics; CMYK retains the source C/Y swap. The four existing extended recipe identities and historical reference catalogue remain unchanged.

The final shared gate passes 133 tests in 220.734s (11 new range tests; zero skipped). Actual 887x887 LAB/CMYK probes select 353,079/214,138 pixels and match both pinned public outputs and isolated saved-input replay. Replay adds 29,599,232/52,102,656 CUDA bytes within 234,924,160 planned; 13,312 preexisting bytes are separate. These are controlled engineering domains, not maximum-domain, corpus or scientific evidence. There are now 19 declared saved-stage replay operations.

Read [the concise F01-F22 implementation status](docs/FEATURE_IMPLEMENTATION_STATUS.md). Source/test freeze: artifacts/native-range-frozen-hashes.json; final gate: artifacts/native-range-shared-gate.log; probe and inspected actual Qt views: artifacts/native-range-probe-final/evidence.json. Bounded eight-file review approves supplied projections/root execution evidence only; no independent reads/tests. No test/GPU process remains live. Do not inspect/touch/stage analysis.seedfiddle-project.json. No staging/commits.

Next: implement the composed range seed-workflow arm and remaining model/helper/paper/observation replay, then complete the requirement-by-requirement method/domain audit. Original SMPTS source/license/dependencies and laboratory gates remain explicit. F01-F22/WP0-WP6 remains active; fresh full unittest discovery is required before completed-goal handoff.

Earlier checkpoint history follows.

SMPTS executable paper reconstruction checkpoint (8 October). The additive `smpts_paper_fixed_mean` graph now executes the paper-supported branch/filter/intersection/contact/erosion/extraction sequence. Defaults25/5/13 come from the paper; offset/polarity, fixed cutoff, connectivity, axis-aligned boxes, box-area cutoff, border, erosion shape and crop rules remain declared assumptions. The historical unresolved graph is unchanged.

The 111-test relevant gate passes in 117.973s (19 new tests;0 skipped). The 2080x3120 committed fixture matches nine pinned OpenCV substep rasters and all 1118 packed crops. This verifies the declared reconstruction, not original SMPTS conformance or scientific accuracy. The full F01-F22/WP0-WP6 goal remains active; full discovery is still required before completed handoff.

Native/CUDA: source 503 components,31 selected parents,1118 crops. Packed crops use original RGB masked by each final component, with source-frame half-open boxes, contiguous offsets and parent/child/disappearance records. No recovered measurement outline is published. CCL8 uses monotone atomic hooks, double-buffer pointer jumps and verified final adjacency/self-roots;64 union rounds,512 jump passes,32B modeled visits,4096 components,128MiB aggregate crop bytes,32MiB compact host cap and 64P+192MiB<=1GiB graph admission. All dense products stay CUDA.

Controlled custom-stream fixture:2.111s,229,198,848 additional CUDA bytes /616,660,992 planned; 25,958,400 preexisting separate. The 4096-component probe uses359,936 additional CUDA bytes and traces 1,804,850 host bytes /17,825,792 modeled. These are controlled engineering probes, not maximum domain, corpus or laboratory evidence. Runtime/source/storage guards and cancellation withhold incomplete products; changed controls evict only descendants. Ordinary PyTorch mutation is guarded; external raw-pointer writes are outside that contract.

Evidence: artifacts/smpts-paper-shared-runner-gate.log, artifacts/smpts-paper-full-fixture/evidence.json, artifacts/smpts-paper-ui/evidence.json and artifacts/smpts-paper-frozen-hashes.json. Actual async crop views inspected at 1440x900 and 1050x650 with Windows fonts; reference entry point inspected. Reusable workers supplied text-only designs; reviewer uses supplied projections/root evidence, no independent reads/tests. Bounded review approves the eleven frozen files: artifacts/smpts-paper-review-checkpoint.json. No independent reads/tests are claimed. No test/GPU process remains live. Do not inspect/touch/stage analysis.seedfiddle-project.json. No commits/staging.

Next: finish remaining operation/helper saved-stage replay and requirement-by-requirement method coverage, including separate paper-port replay. Keep original source/license/dependency and laboratory blockers explicit. Fresh full unittest discovery is reserved for completed-goal handoff.

Earlier checkpoint history follows.

Extended threshold saved-stage replay checkpoint (8 October): eighteen declared
operations now support saved-input replay. Gaussian, triangle, texture and
dual-channel each replay their existing standalone source-frame graph. The
reader selects exactly the input raster and expected mask; calculation and
every binary-mask comparison remain on CUDA. Producer-port/frame/runtime
binding and active-run preservation use the existing guarded replay path.
Malformed controls/headers are invalid archives; valid wrong masks report
mismatch. Source precision/preamble/resource limits report unavailable without
products. Cancellation and mutation withhold all incomplete outputs.

Primitive execution and replay share allocation-free tile/scratch admission.
Gaussian 11/13/15 retains its 1B window-sample cap; texture retains 4M pixels
and bounded pair/sort scratch; triangle retains the 2**24 histogram source
ceiling beneath the stricter replay CUDA ceiling. Dual counts its RGB-to-BGR
copy and at most two distinct tiled conversions. Selected arrays remain capped
at 48MiB and CUDA at 1GiB. The fourteen earlier plans are unchanged. Baseline
replay admission units and primitive visits are recorded separately; neither
is an instruction count or elapsed-time guarantee. Texture coverage concerns
binary decisions, not byte parity for its internal texture map.

Verification: 141 relevant tests pass in 619.534s before the final CMYK repair;
then 58 affected tests pass in 212.202s after the repair (0 skipped;
gates overlap). A new CUDA regression compares all four CMYK channel bytes
against the pinned helper on a 196,625-pixel pattern including all 256 neutral values.
The shared conversion now uses a CUDA float64 tensor divisor, preserving true
double division before byte truncation; the earlier scalar reciprocal path
caused 30 public dual-mask disagreements in the larger probe. Seven actual archives match
pinned public PlantCV masks, including guarded Gaussian and LAB/CMYK dual
conversion. Actual asynchronous Qt replay works for all four operations.
Three controlled probes cross three tiles plus a tail and match public/native
masks and isolated replay: gaussian 125x241: 59,357,696 additional / 138,073,728 planned CUDA bytes, 13,312 preexisting; texture 113x101: 24,600,064 additional / 138,481,343 planned CUDA bytes, 13,312 preexisting; dual_channels 275x715: 12,585,472 additional / 159,385,728 planned CUDA bytes, 13,312 preexisting.
Primitive scratch stays within the shared allowance; archive hashes remain
unchanged. These probes do not establish maximum-domain or corpus coverage.

Bounded review approves the five files recorded in
artifacts/extended-threshold-replay-frozen-hashes.json using supplied projections
and root execution evidence only; no independent reads/tests are claimed.
Earlier freezes and fixture evidence retain historical scope. F01-F22/WP0-WP6
remains active, partial_native and scientifically unvalidated. Complete method
matrices, remaining model/helper replay, broader domains, original/SMPTS
execution and laboratory gates remain open. Fresh full unittest discovery is
required before the completed-goal hand-off.

Earlier watershed debug saved-stage replay checkpoint (8 October): fourteen declared
operations now support saved-input replay. The new debug contract works in all
six source/calibrated binary, mean and Otsu graphs. It binds both incoming
branches and each raster frame, selects exactly RGB input, int32 labels and
expected RGB, then compares every fresh image value on CUDA. Its frameless
compact output has an exact 31-field schema and immutable palette/provenance.
Primitive decoding has a conservative 256 KiB encoded budget, 30,000 visits and
depth 4. Every leaf is compared, including each missing palette channel.
Structurally valid wrong counters, palettes, populations and pixels report
mismatch. Malformed contracts, incompatible runtimes, cancellation and native
resource unavailability remain distinct; incomplete products are withheld.

Selected arrays admit up to 128 MiB. This operation uses CUDA admission
max(128P+128MiB,13P+256MiB+64min(P,1M)+3MiB)<=1GiB, where P is frame pixels.
The previous thirteen plans are unchanged. Native scan/topology/geometry/host
and painter limits remain effective; bounds do not guarantee runtime or RSS.

Verification: 113 tests pass in 659.099s (0 skipped); then 15 tests pass in 156.141s (0 skipped); gates overlap.
Fresh 2080x3120 verification matches pinned public PlantCV labels, palette and
debug pixels (1294 labels). Final isolated replay adds 118,130,688 CUDA bytes
within 964,886,528 planned bytes and takes 178.878s; upstream tensors
are released and 0 preexisting bytes are separate. Only 64,896,000
array bytes are selected. All 19,468,800 RGB values and 5,208 metadata leaves
match, and archive hashes remain unchanged. The controlled grayscale/gray-repeat
RGB, cutoff 100, fill 1 and peak-distance 8 fixture remains diagnostic coverage,
outside the separate 128-label trait domain. Geometry remains near its 64M cap.
Actual replay-specific Qt views were inspected at 1440x900 and 1050x650.

Bounded review approves the four final source/test files recorded in
artifacts/native-watershed-diagnostic-replay-final-hashes.json, using supplied
projections and root execution/visual evidence only. Earlier freezes retain
historical scope. PlantCV returns labels only; the image remains a debug side
effect, and count is separate. F01-F22/WP0-WP6 stays active, partial_native and
scientifically unvalidated. Remaining operation/helper replay, broader domains,
original/SMPTS execution, laboratory gates and fresh full unittest before
completed hand-off stay open. Next: saved-stage Gaussian, triangle, texture and
dual-channel thresholds with source-derived precision/resource admission.

Earlier watershed debug overlay checkpoint (8 October): six additive binary/mean/Otsu
graphs support the source frame or calibrated seed ROI. PlantCV returns labels
only; the new contour image implements its separate debug side effect. The
count remains a separate observation. Colormap, explicit reproducible uint32
shuffle seed and line thickness affect the image and reuse segmentation.

The private CUDA RGB renderer preserves pinned contour order, holes, islands,
overdraw, closure and LINE_8 pixels. Adjacent original edges are drawn in bounded
batches, without contour approximation. Strict callback/storage checks withhold
partial products. Shared scan/topology/stroke/paint limits and a 256MiB modeled
host geometry allowance remain explicit. Composed CUDA admission is
max(base,64P+16min(P,1M)+256MiB)+3P <=1GiB, including the private canvas;
larger input backing stores are counted. Bounds do not guarantee process RSS or runtime.

Verification: 30 tests pass in 80.641s (0 skipped); then 183 tests pass in 645.147s (0 skipped); gates overlap. The 2080x3120 committed fixture matches
actual public labels, palette and diagnostic pixels (1294 labels).
Additional CUDA peak is 517,786,624 bytes against 719,238,656
planned bytes; caller source backing is excluded from the measured additional peak.
Native execution took 276.814s. Geometry used 63,164,962/64M
work units; peak modeled host geometry was 12,369,424 bytes.
This geometry result is near the cap and does not establish broader input coverage.
The full-fixture oracle uses decoded grayscale and explicit gray-repeated RGB,
binary cutoff 100, fill size 1 and peak distance 8. Its 1,294 labels exceed the
separate 128-label downstream measurement domain. This is diagnostic coverage.
Actual async calibrated Qt views were inspected at 1440x900 and 1050x650.
Bounded review approves eleven frozen files from supplied projections and root
execution/visual evidence only: artifacts/native-watershed-diagnostic-frozen-hashes.json.
This supersedes current hashes of files changed since earlier checkpoints;
their recorded freezes retain historical scope. Earlier wording about a
public diagnostic return is corrected here to a debug-only side effect.

F01-F22/WP0-WP6 remains active; affected features stay partial_native and science
is unvalidated. The new debug stage itself still lacks saved-input replay.
Remaining replay/domain/original/SMPTS/laboratory gates and fresh full unittest
discovery before completed hand-off remain open.

After reconciliation, all six feature-register checks pass in 0.637s; every
existing test reference resolves. The canonical peer manifest and all eleven
frozen source/test hashes are unchanged. No task test process remains live.

Earlier typed watershed saved-stage replay checkpoint (8 October): saved-input replay now covers all five
remaining watershed stages: local maxima (coordinates and boolean mask), connected
markers, negative distance, masked flood (labels and completion token), and the
compact estimated-object-count observation. Together with the earlier eight
operations, there are thirteen declared replay contracts in full-source or
canonical calibrated seed-ROI frames. Every incoming branch, including RGB and
the flood sequencing token, binds its declared producer port and frame ancestry.

The selected reader verifies the whole archive and retains only exact selected
arrays. All selected inputs, expected tensors and fresh tensors stay on CUDA;
every output is compared. Coordinates require in-frame unique row/column values,
their exact painted mask and stable descending-distance order. Marker/instance
IDs have a bounded nonnegative domain; flood labels preserve in-mask seeds, stay
inside foreground and use only IDs present in those seeds. Completion is exactly
True; count records have exact fields, label, type and bounded integer value.
Immutable compact outputs remain metadata. Numeric float comparison treats signed
zeros as equal. Failures and cancellation withhold all products.

The five new contracts admit up to 128 MiB selected bytes; earlier operations keep
48 MiB. Replay-owned CUDA remains128P + 128 MiB <=1GiB (7,340,032 pixels). Peaks now
use mathematically equivalent separable axis maxima for greedy suppression and
a ceiling of 64 billion estimated NMS sample visits, separate from legacy 8 billion raster-work-unit
admission. For NMS, P_roi is ROI pixels, K is candidate count and k=2*distance-1.
The estimate is P_roi*(24+4k)+64K per round, capped at 4,096 rounds before
NMS allocation; incomplete results are withheld. Native candidate/workspace,
marker, queue and cancellation bounds remain effective. These bounds do not
guarantee elapsed time or whole-process memory.

Verification: 93 tests pass in 563.810s (0 skipped), followed by 5 tests pass in 19.608s (0 skipped) after the final compact-count presentation
repair. These gates overlap. The committed 2080x3120 fixture matches all five
typed stages under the default cap: peaks use 63 NMS rounds and406,576,128 additional
CUDA bytes; connected markers 457,726,976; negation 81,260,032; flood 192,462,848;
count 79,220,224. Each is within 964,886,528 planned bytes. This evidence covers
available watershed products; the fixture's 1,328 markers exceed the separate
128-label downstream measurement domain. The existing pinned full-fixture peak
and watershed chains also pass. Actual asynchronous calibrated maxima/count
views were checked at 1440x900 and 1050x650, with count 2 visible above replay details.

Bounded review approves seven frozen source/test files from supplied projections,
root execution and root visual evidence only. Independent filesystem reads or
test execution are not claimed. The freeze is
artifacts/native-typed-stage-replay-frozen-hashes.json.

F01-F22/WP0-WP6 implementation remains active; affected features stay
partial_native and scientific validation is false. Remaining threshold/model/
color/size/geometry/helper replay contracts, public watershed diagnostic return,
broader computational domains, original/SMPTS execution and laboratory gates
remain open. Fresh full unittest discovery is required before completed hand-off.
Earlier checkpoints retain their historical scope and evidence.

After reconciliation, all six feature-register checks pass in 0.622s; every
existing test reference resolves and the canonical peer manifest is unchanged.
The seven reviewed source/test hashes still match. No task test process remains
live. The full feature goal is active.

Earlier Calibrated local-stage replay checkpoint (8 October): the eight supported saved-input raster
operations now also replay in the explicit seed ROI of the canonical calibrated
binary/mean/Otsu graphs. Frame resolution uses declared ports, saved integer
half-open bounds and verified crop ancestry. Parent source/frame, local frame,
translation and transform remain distinct, including an explicit full-bounds crop
whose dimensions equal its parent. Every traversed local raster and the full-parent
corrected RGB descriptor must match their shape/dtype/layout contracts. All four
crop input keys, compact source/frame records, card bounds and bounding rectangle
are checked; card overlap is recomputed with the shared crop calculation.

Replay uploads only the exact saved local incoming raster and expected output,
dispatches only the selected operation and compares all elements on CUDA. The
whole archive remains verified; unrelated parent pixels are not retained or
uploaded by the selected reader. Admission uses local pixels and selected bytes.
The existing 128P+128MiB CUDA envelope, 48MiB selected-byte limit, archive limits
and operation-specific work limits remain unchanged. For the actual 96x128 ROI,
selected bytes are 24,576 for byte-output stages and 61,440 for distance output.

Verification: 35 focused tests pass in 281.940s (0 skipped), covering local/full-source replay, calibrated pinned
reference chains, crop/callback/cache/persistence behavior, rehashed frame and
lineage faults, cancellation and actual asynchronous Qt preview/current-run
preservation. Actual calibrated replay views were checked at1440x900 and1050x650.
Bounded review approves four frozen source/test files from supplied projections,
root execution and root visual evidence only; independent filesystem reads or
test execution are not claimed. The freeze is
artifacts/native-local-stage-replay-frozen-hashes.json.

F01-F22/WP0-WP6 implementation remains active and affected functionality stays
partial_native. Replay agreement is computational, with scientific validation
false. Arbitrary cropped graphs, multi-input/model/marker/label/geometry/
observation and remaining helper-stage replay, broader domains, original/SMPTS
execution and laboratory gates remain open. Fresh full unittest discovery is
required before completed hand-off. Earlier checkpoints retain their historical
scope and evidence.

After reconciliation, six feature-register checks pass in0.609s; every existing
test reference resolves and the canonical peer manifest is unchanged. No test
process remains live. The full feature goal is active.

Earlier Saved-stage replay checkpoint (8 October): actual archive-backed replay is now implemented
for median, binary/mean/Otsu threshold, small-object filtering, erosion,
dilation and the pinned watershed distance map. It loads the exact saved
incoming raster and expected output, dispatches only the selected operation,
and compares all elements on CUDA. Source/frame, recipe settings, producer-port
lineage, payload hashes and current loaded calculation/runtime identities bind
the attempt. Match/mismatch, unavailable, unsupported, invalid, failed and
cancelled outcomes stay distinct; no output is returned after failure/cancel.

The selected reader verifies every archive member, including unselected data,
while retaining only requested arrays and bounded auxiliary correction fits.
Incremental compressed reads avoid reserving the full archive allowance for a
small file. Header/hash/value/CRC/tail faults, rehashed nonfinite fit matrices,
cancellation and loaded-helper changes withhold results. Ordinary all-array
loading and archive versions1/2/3/4 remain supported.

Replay-owned CUDA admission is128P+128MiB <=1GiB, at most7,340,032 pixels;
selected payloads are capped at48MiB. Whole-archive admission is256MiB expanded
and compressed,128MiB per array. Operation-specific work is capped at8B tensor
work units, not device instructions or elapsed time. General fill includes
all edge batches,64 union passes and26 full-parent jumps; it rejects the full
committed fixture under this conservative work bound, except size1/>P shortcuts.
Distance has two row sweeps with an explicit65,536 scan-chunk ceiling. UHD and
local/cropped replay are explicitly unsupported by this first replay scope.
This does not narrow the separate executable graph/primitive domains.

The committed2080x3120 saved threshold stage matches in7.003s, with32,448,000
additional CUDA bytes within964,886,528 planned and223,651,427 traced host bytes.
A32MiB unselected member streams with37,520,970 traced host bytes including its
33,554,942-byte snapshot. These observed peaks are not whole-process guarantees.

Qt's Replay saved stage action uses an asynchronous worker, exact selected
recipe settings, Stop/close-and-drain, separate attempt facts and a fresh preview.
The active run, node cache, historical records and archive bytes are preserved.
Actual views were visually checked at1440x900 and1050x650.
Integration gate: 171 tests pass in481.283s (0 skipped); archive-origin/watershed gate: 12 tests pass in35.600s (0 skipped); final API/Qt gate after the cropped-frame outcome repair: 17 tests pass in122.279s (0 skipped). These gates overlap and are not a count of distinct tests.
Bounded review approves nine source/test files in
artifacts/native-stage-replay-frozen-hashes.json from supplied projections,
root execution and root visual evidence only; no independent filesystem read
or test execution is claimed.

F01-F22/WP0-WP6 implementation remains active. F21/F22 and all affected families
remain partial_native; this establishes saved-input computational agreement,
not PlantCV conformance or scientific validation. Multi-input, cropped/local,
model, label/geometry/observation and remaining helper-stage replay, broader
computational domains, original/SMPTS execution and laboratory gates remain
open. Fresh full unittest discovery is required before completed hand-off.
Earlier checkpoints below retain their historical scope and evidence.

After reconciliation, six feature-register checks pass in0.605s; all existing
test references resolve. No test process remains live. The full goal is active.

Earlier F13 native recovery graph checkpoint (8 October): the experimental CUDA
partition/annotation APIs now have an additive 11-stage native working recipe.
Original grayscale threshold and explicit erosion produce four-connected cores;
each core ID is its first raster index plus one. IDs survive validity edits,
while changing erosion can define different cores. The named elevation is the
negative pinned 5x5 distance of the original foreground. A dimension-bound
optional half-open ROI marks valid support without cropping/rescaling the frame.

Pre-validity core inventory distinguishes wholly clipped known IDs from
foreground with no surviving erosion core. Known clipped IDs remain explicit
unavailable records; unknown unseeded components remain zero/unresolved. The
original foreground union and compact support/contact facts are retained.
Radius0..8 and withhold_uncertain/flag_uncertain annotations reuse labels;
only permitted assigned area and bounding width/height are calculated in pixels.
Shared boundaries remain inferred and scientific_validation stays false.

Qt provides an Experimental core recovery launcher, asynchronous execution,
previews, actual editable controls, Save/Keep/reopen and existing project/batch/
archive paths. Nested immutable tensor/fact mappings expose every CUDA raster
to cache/export traversal without persisting session pointers. Source/frame and
prepared-pixel handoffs, final loaded calculations, callback failures, capacity
errors and dirty descendants are checked before publication. The inspector
scrolls at minimum size and sparse IDs receive distinct preview colours within
the graph's supported range. Historical catalogue graphs remain unchanged.

CUDA BFS admission includes actual input backing/output/queue and compact
tables. The graph CUDA envelope is64P+40MiB, capped at1GiB. Graph-owned CPU
metadata/cache is separately capped at64MiB using8192 bytes per ID plus1MiB,
limiting this graph to8064 IDs; the producer and direct API admit32768 IDs.
Oversized graphs retain upstream inventory but publish no partial partition.
Typed streaming fingerprints retain fixed-size signatures while checking all
metadata and tensor versions. At8064 IDs, retained metadata measured8,188,709
bytes and traced peak33,994,936 bytes. Explicit archive encoding has separate
existing limits and can reject an executable graph; the metadata cap is not a
whole-process/export memory guarantee. The graph ceiling is16,121,856 pixels
before excess caller backing, with bounded work. Dense UHD
core labeling measured9.471s and75,289,088 combined input/additional CUDA bytes
within83,300,384 planned bytes. The committed2080x3120 complete graph measured
53.288s and258,381,824 bytes within457,277,440 planned bytes, with51 cores;
full markers/labels match the actual pinned PlantCV/SciPy/OpenCV/skimage chain.
These are observed experimental timings, without a real-time guarantee.

Constructed isolated and touching disk controls recover visible masks with
IoU1.0 versus erosion-only IoU0.610..0.611 and match the fixed native flood
control. Touching shared pixels remain uncertain and measurements are withheld
by default. These fixture-specific engineering results do not establish
biological identity, hidden overlap, corpus improvement or laboratory accuracy.
Real count/queue/batch/byte limits, custom streams, mutation/cancellation,
unseeded/excluded/canvas cases and moved-project/archive/batch checks pass.

Final recovery/preview gate: 20 tests in 141.423s (20 pass; 0 skipped). Shared executor/Qt/project/export/classifier compatibility gate on the final frozen source: 198 tests in 370.864s (198 pass; 0 skipped).
Bounded review approves the seven files in
artifacts/native-core-recovery-frozen-hashes.json from supplied source projections
and root execution only; no independent filesystem reads or test run is claimed.
The prior three recovery primitive files and canonical peer manifest are unchanged.

F13 remains partial_native. WP4 is now in progress; its exit gate is not met.
Corpus contact/debris ablations, complete per-stage replay and laboratory gates
remain open. This does not recover the original SMPTS erosion/contact rules.
Full F01-F22/WP0-WP6, broader input/computational domains, remaining helper graphs,
original/SMPTS execution and fresh full unittest discovery before completed
handoff remain active.

After reconciliation, all six feature-register checks pass in0.598s;
existing test references resolve. No test process remains live.

Earlier F13 recovery primitive checkpoint (8 October): a distinct experimental CUDA
API now partitions valid original foreground from supplied positive eroded-core
IDs and explicit finite float32 elevation. It preserves the pinned four-connected
watershed heap/tie behavior and every core pixel/ID, including sparse int32 IDs.
The original foreground union is retained separately. Unseeded components stay
zero and are reported as unresolved; foreground excluded by validity has an
explicit count and all-excluded status. No missing pre-erosion identity is invented.

CUDA bookkeeping returns symmetric shared interfaces, per-ID seed/assigned/
interface counts, half-open bounds and canvas/invalid-support/unresolved-neighbor
flags. A separate annotation API reuses labels: radius0 includes the interface,
and radius1..8 adds four-neighbor dilation clipped to the valid foreground.
withhold_uncertain and flag_uncertain control computational measurements only.
An uncontested visible partition is an input-domain status, not biological
identity, a hidden individual outline, measured improvement or scientific validation.

Dense products, ID lookup, compact sorting, queue and statistics remain CUDA;
only compact facts cross to CPU. Distinct IDs are capped at32768, lookup probes
at128 and full input backing, retained workspace, CPU metadata and work are
admitted together. The bounded flood driver reuses the unchanged noticed
watershed kernel. Cancellation, ordinary PyTorch tensor/storage mutation,
capacity failures and invalid labels withhold incomplete results.

Sixteen recovery tests and five existing watershed tests pass in49.845s:
artifacts/core-recovery-reviewed-compatibility-gate.log. Thirty small actual
pinned-source cases and the committed2080x3120 fixture match full labels and
compact facts. The new full-fixture API allocates103,025,152 additional CUDA
bytes within its204,885,976-byte combined plan. Maximum-ID annotations, custom
streams, compact-only transfer, radius/policy reuse and injected failures pass.
The isolated engineering fixture recovers its pre-erosion contour exactly.
Bounded review approves the three unchanged files in
artifacts/core-recovery-frozen-hashes.json using supplied projections/root
execution, without independent reads/run. The canonical peer manifest is unchanged.

F13 is partial_native: the primitive and annotation API are implemented; native
working-recipe/node-cache/Qt/project/batch/archive integration remains next.
Contact comparisons against erosion-only/reference controls and laboratory
evidence remain open. This does not recover original SMPTS erosion/contact
rules. Full F01-F22/WP0-WP6, complete replay, original/SMPTS execution, fresh full
unittest discovery before completed handoff and scientific gates remain active.

After documentation reconciliation, all six feature-register checks pass in0.579s; every existing-test reference resolves. No test process remains live.

Earlier Bayes mask-coordinate checkpoint (8 October): CUDA directory training now
supports different decoded image and mask dimensions using the pinned source's
row/column indexing. Mask[y,x] addresses image[y,x]; masks are not resized, and flat mask
offsets are not reused as flat image offsets. Smaller masks select the corresponding
top-left image coordinates. Larger masks are supported when every categorical
0/255 position is in bounds; all other mask values remain ignored, including
outside the image. Foreground then background bounds failures occur after HSV
conversion and before this pair's random draws, even for zero sampled rows.
Earlier pairs' consumed draws remain consumed and no completion is published.

Image and mask pixel caps, mask-derived count tiles and image conversion work
are separate. Compact four-column tile counts audit categories/bounds before
position/output allocation. Mapped positions, HSV, gathers and retained samples
stay CUDA. Cumulative admission includes unique input backing, retained prior
samples/work, both count/offset tables and actual conversion/gather sizes.
Actual custom-stream peak and tile-repartition checks pass, as does a maximum
16-million-pixel ignored mask with a one-pixel image. Only compact tile counts
cross to CPU during direct sampling.

New directory records use parser-v2 and an exact fourteen-field sampling plan
with both image and mask dimensions. The codec checks successful categorical
counts against the shape intersection and derives the new resource facts.
Parser-v1 retains its exact twelve-field matching-shape contract, old formulas
and canonical bytes. Legacy record annotation changes reuse classifier products;
project move, source removal, companion batch and current archives preserve
the exact old record. The interoperable model TSV and classifier semantics are
unchanged; training records remain descriptive rather than authentication or
scientific certificates.

Seven actual public dimension cases verify sampled columns and complete RNG
states; five categorical bounds cases verify failure priority and no draws.
Four actual decoded directory comparisons verify PDFs and typed facts. First/
late source failures, strict record mutations, mixed-dimension asynchronous
training and atomic save/reload, moved source-independent projects/archives and
compact-only SVG plotting are covered. The four-module focused gate passes59
tests in55.279s. The final20-module compatibility gate passes258 tests in358.626s,
including ten new regressions, with no source/test edits during execution:
artifacts/naive-bayes-mask-coordinates-reviewed-compatibility-gate.log.
Bounded review approves the seven unchanged source/test files in
artifacts/naive-bayes-mask-coordinates-frozen-hashes.json using supplied
projections/root execution, without independent reads/run. These sampler,
directory, codec and related test identities supersede their prior freezes.
The canonical peer source manifest is unchanged. No test process remains live.

PNG/JPEG contents remain the native directory header domain; other source-
decodable formats and broader numerical/input domains remain open. Next major
engineering work is F13's distinct experimental recovery from supplied eroded
core markers, preserving the known foreground union and reporting inferred
shared boundaries. This does not recover the unavailable SMPTS implementation.
Full F01-F22/WP0-WP6, complete per-stage replay, original/SMPTS execution, fresh
full unittest discovery before completed handoff and laboratory/scientific gates
remain open. Feature and work-package statuses are not promoted by this gate.

After the eight-document/register reconciliation, all six register checks pass
in0.573s and every existing-test reference resolves. No test process remains live.

Earlier Bayes constant-channel checkpoint (8 October): the CUDA fitter now accepts
constant HSV channels when the pinned source's weighted mean produces positive
roundoff covariance. An exact mean equal to the byte still fails as singular;
no epsilon, variance floor or observed dispersion is invented. The specialized
positive-sum covariance interval and byte-bin underflow guards preserve the
existing ordered CUDA reductions and generic varying-channel uncertainty gates.
Peak normalization allowances remain empirical and tied to the pinned whitening
path and tested runtime; this is not an arbitrary-vendor or scientific certificate.
Samples, reusable intermediates and PDF evaluation remain CUDA.

Constant records use the separate numpy-2.2.6-buffer8192-constant-v1 profile and
one strict constant_value byte. The compact codec reproduces the uniform-term
moments and validates covariance, bandwidth and budget relationships; a mean
slightly above255 is allowed only when that exact audit reproduces it. Legacy
and buffered varied records retain their exact ten-field shapes; constant
records have eleven fields. Buffered varied/constant channels may mix, while
legacy/new-family mixtures fail. Existing record bytes remain readable. A
standalone frozen record keeps the existing loaded-source identity checker
intact, including its post-init hook. TSV format and classifier identity do not
change; provenance remains descriptive rather than proof of authenticity.

Actual pinned SciPy success/failure and all256 PDF bins are checked on778
constant cases, including all bytes at n3/6/7 and selected large/boundary sizes
through1,000,000. Another768 all-byte cases around8192 compare initial CUDA and
compact-codec moment bits against the pinned source. Mixed channels, a late
singular class and exact source failures withhold incomplete results. Actual
public sampled-RGB and directory training, typed-record rejection, asynchronous
atomic saving/reload, project move with source removal, companion batch/archive
and compact-only SVG plotting exercise successful constant fits end to end.

The final20-module compatibility gate passes248 tests in330.530s, including
eight net new regressions. The log is
artifacts/naive-bayes-constant-channel-reviewed-compatibility-gate.log.
The six new integration checks pass in11.108s. Earlier
dataclass source-identity setup errors were corrected before this final gate;
they are not counted as passes. Bounded review approves the nine unchanged
source/test/notice files in artifacts/naive-bayes-constant-channel-frozen-hashes.json
using supplied projections and root execution, without independent reads/run.
These files supersede their prior freezes; the canonical peer source manifest
is unchanged. No test process remains live.

Matching decoded PNG/JPEG image/mask shapes remain the current directory domain.
Next is source-success coordinate indexing for different mask dimensions,
followed by other input/numerical domains. Full F01-F22/WP0-WP6, complete per-stage
replay, original/SMPTS execution, fresh full unittest discovery before completed
handoff and independent laboratory/scientific gates remain open. Statuses stay
partial_native / pending_recipe_freeze / blocked_laboratory_inputs as applicable.

After reconciling the eight documents/register, all six register checks pass
in0.582s and every existing-test reference resolves. No test process remains live.

Earlier Bayes buffered-reduction checkpoint (8 October): CUDA weighted moments now
follow the pinned NumPy 2.2.6 default outer iteration: pairwise reductions within
8,192-element buffers, then ordered accumulation of buffer results. The original
full-array pairwise tree omitted that outer step. Actual pinned NumPy sums and
initial statistics match every binary64 bit across 231 fixtures / 693 channels:
29 lengths through 1,000,000, seven constant bytes and two varied patterns around
the outer boundaries. Full actual SciPy KDE PDFs at 8,193 / 65,537 / 1,000,000 match
all 256 bins within both declared empirical budgets and a separate tighter check.
The mean-error budget includes chunk and outer accumulation depth; covariance
and PDF uncertainty gates remain. Work admission retains its input-visit metric,
not an exact CUDA instruction count. Samples and reusable reductions remain CUDA.
New fit records identify numpy-2.2.6-buffer8192-v1. The fixed reference profile is
independent of installed NumPy buffer settings; customized reference buffer
settings are outside it. Old record profiles/bytes remain readable, mixed old/new
profiles in one fit are rejected, and annotation-only profile changes reuse
prediction products while refreshing current metadata. The interoperable model
TSV and classifier identity are unchanged. Additional pinned NumPy source hashes
are calculation constants with attribution in docs/BAYES_KDE_NOTICES.md.
The 20-module compatibility gate passes 240 tests in 308.330 s, including four
new reduction/profile regressions and shared directory/UI/project/cache/export/
plotting paths (artifacts/naive-bayes-buffered-reduction-reviewed-compatibility-gate.log).
The initial three-module focused gate passes 34 tests in 56.530 s. Bounded review
approves the five unchanged files frozen in
artifacts/naive-bayes-buffered-reduction-frozen-hashes.json using supplied
projections/root execution, with no independent reads/run. These CUDA/codec/
core-test/sample-UI-test identities supersede their earlier freezes; notice
updates are included. The canonical peer source manifest is unchanged.
Constant-channel fitting is still withheld. Correcting the mean path alone does
not establish the specialized positive-covariance/PDF contract. Remaining source-
success domains, full F01-F22/WP0-WP6, complete per-stage replay, original/SMPTS
execution, fresh full discovery before completed handoff and laboratory/scientific
gates remain active.
After reconciling the eight documents/register, all six register checks pass in
0.563 s and all four new test references resolve. No test process remains live.

Earlier Bayes PDF plotting checkpoint (8 October): the native two-mode training dialog
now offers optional **Save probability density plots (SVG) beside the model**.
A background worker renders hue_pdf.svg, saturation_pdf.svg and value_pdf.svg
from compact parsed probability tables, preserving class order, implicit
x=0..255 and legend loc=best. Matplotlib3.11.2 is a preferred runtime dependency
through the existing launcher manifest. Independent noninteractive SVG Figures,
default-style context and a renderer lock coordinate this module's own jobs.
Streaming8MiB/per-SVG and24MiB/bundle bounds, compact label limits, model guards,
cancellation and finally cleanup withhold incomplete rendered bundles. The UI
rechecks model/revision, withholds stale/cancelled output and drains rendering
and publication/reporting before parent/child close. All plots render before
publication. Exact TSV commit and actual reload precede individually atomic
SVG writes; each failed atomic write preserves its prior file. Later SVG failure
reports actual written/failed names while retaining the verified TSV model.
These four files are not a multi-file transaction. Rendered artifact metadata
records actual hashes and renderer/default identities separately from model
training provenance and successful publication results.
The20-module compatibility gate passes236 tests in278.746s, including15 new
plot/source/UI cases and existing training/project/batch/cache/export/bootstrap
checks (artifacts/naive-bayes-plots-reviewed-compatibility-gate.log).
Actual pinned public plotting calls on three fixtures match every curve/order/
legend and SVG structure after only timestamp/generated-ID normalization.
Actual760x370 Segoe UI sample/directory views and a trained-density SVG were
inspected. Bounded review approves seven unchanged frozen files in
artifacts/naive-bayes-plots-frozen-hashes.json using supplied projections/root
execution, with no independent reads/run. The training dialog supersedes its
prior freeze; the other directory UI3, directory API/sampler4, core/API4 and
canonical peer manifest remain unchanged. There is no arbitrary-style/platform
or byte-identical SVG claim. Numerical conformance remains empirical.
After the eight-document/register reconciliation, six register checks pass in
0.558s and all15 new plotting test references resolve.
Remaining source-success training/sampling domains, full F01-F22/WP0-WP6,
complete per-stage replay, original/SMPTS execution, fresh full discovery before
completed handoff and laboratory/scientific gates remain active.

Earlier binary Bayes directory desktop/project checkpoint (8 October): **Train model…**
now offers Sampled RGB table and Images and masks modes. Directory selection
freezes both roots and the mode; asynchronous CUDA training retains source
ordering and actual global MT19937 advancement. Directory GUI jobs serialize
their use of that stream. Stop, changed selections and parent/child closing
withhold late results and drain workers; queued cancellation consumes no draws.
Only compact TSV/model/record facts are inspected by the UI. Exact TSV saving
uses QSaveFile and actual postcommit reload before selecting the model.
The standard-library provenance codec now validates a distinct canonical UTF-8
directory record, ordered traversal/pairing, unique source identities, full RNG
state digests/chains, derived resource facts and model/table/class bindings.
Origin paths are descriptive and never read by the codec. Keep, project Save/
Open/move, actual companion batch and whole-run archives retain the bound model
and record after both training roots and the original TSV are removed.
Annotation-only changes reuse numerical products and refresh current history.
The 17-module compatibility gate passes 207 tests in 264.992s, including 15 new
directory codec/UI/project cases; the existing 15 sample-table UI cases remain
green (artifacts/naive-bayes-directory-ui-reviewed-compatibility-gate.log).
Actual Segoe UI 760x370 sample/directory views were inspected without clipping.
After reconciling eight planning/operator/register files, all six feature-register
checks pass in 0.546s; all 15 new existing-test references resolve. No test or
preview process remains live. Do not rerun the one-shot documentation helper.
Bounded review approves the four unchanged files recorded in
artifacts/naive-bayes-directory-ui-frozen-hashes.json using supplied projections
and root execution evidence; no independent reads/run are claimed. The codec,
training dialog and parent action supersede their earlier UI freezes. The other
five earlier UI files, four training core/API files, four directory API/sampler
files and canonical peer source manifest are unchanged.
PNG/JPEG contents, matching decoded shapes and varying HSV channels remain the
native directory domain. Numerical conformance remains empirical; record
validation does not establish authenticity, numerical or scientific validity.
Plotting, broader domains, full F01-F22/WP0-WP6, complete per-stage replay,
original/SMPTS execution, fresh full discovery before completed handoff and
laboratory/scientific gates remain active.

Earlier binary Bayes directory API checkpoint (8 October): source-ordered recursive
pairing now connects bounded PNG/JPEG colour/gray decoding, the reviewed CUDA
sampler, cumulative retained-sample admission, shared KDE fitting and exact
interoperable TSV output. Unsorted walk order, flat masks, duplicate basenames,
case-sensitive last-three-character suffix selection and missing-mask skips
are retained. Each pair is decoded in order; late failures/cancellation retain
actual consumed MT19937 draws and publish no completion. No seed is invented.
Retained HSV samples, full analysis intermediates and fitting remain CUDA.
Per-image admission charges previous sample backing/rows/work before current
draws; final concatenation and fit coexistence is preflighted. Actual custom-
stream peak is tested. Final callback-free traversal/source-byte/code audits
withhold changed sources, newly available masks and reordered occurrences.
Distinct immutable directory provenance records ordered pair bytes, plans,
draw identities, initial/final full RNG states, model/tables and actual loaded
calculation/decoder/runtime identities. It is descriptive API metadata;
directory Qt controls and project-held typed codec integration remain next.
Real-file pinned public-trainer tests cover PNG/JPEG,16-bit source conversion,
selection/RNG/sample observations, actual empirical KDE PDFs and standalone
inference after source removal. Matching decoded image/mask dimensions remain
the native domain; some smaller masks succeed in the source and are explicitly
unsupported here. Codec-internal/whole-process CPU memory is not certified.
The six-module compatibility/register gate passes52 tests in49.609s, including
13 new directory/cumulative-budget cases
(artifacts/naive-bayes-directory-training-reviewed-gate.log). Bounded approval
covers supplied projections and root execution for the four frozen files;
no independent reads/run or full-suite completion is claimed.
The previous12 core/API/UI/project/export freezes and canonical source manifest
are unchanged. The cumulative-budget sampler/test changes supersede their
earlier two-file freeze; four current files are recorded in
artifacts/naive-bayes-directory-training-frozen-hashes.json.
Full F01-F22/WP0-WP6, broader domains, plotting, complete per-stage replay,
original/SMPTS execution, fresh full discovery before completed handoff and
laboratory/scientific gates remain active.

Earlier binary Bayes CUDA sampling checkpoint (8 October): the new direct image/mask
primitive preserves exact0/255 row-major populations, independent HSV-channel
sampling, the source's excluded final population pixel, six ordered legacy
RandomState calls and their actual state advancement. None uses the existing
global stream; explicit captured legacy RandomState is supported without an
invented seed. Zero-sized calls remain calls; later source failures and
cancellation preserve already consumed draws and publish no samples.
Full BGR/mask/HSV/population positions and selected values stay CUDA. Explicit
count/compaction kernels transfer only bounded per-tile counts to CPU; source
random vectors/state are compact metadata. Unique full backing storage,
combined samples, repeated tile setup/work, CPU metadata and actual CUDA peak
are bounded. Callback/final tensor and RNG guards include raised callbacks;
a shadow stream detects changes during a draw. This is a sampling component,
not a directory-training desktop action or a numerical/scientific certificate.
The final five-module gate passes39 tests in34.008s, including10 new sampling
tests (artifacts/naive-bayes-image-sampling-reviewed-gate.log). Actual pinned
public trainer observations match selected columns and final/intermediate RNG
states; a separate actual KDE fit matches within the existing empirical budget.
The KDE spy for zero-row sampling is distinct from that real-fit evidence.
Bounded review covers supplied projections/root execution and the two files in
artifacts/naive-bayes-image-sampling-frozen-hashes.json, with no independent
reads/run. Previous12 source/test freezes and the canonical manifest are unchanged.
Source-ordered directory pairing/materialization, combined directory fitting/TSV,
directory provenance/Qt integration and plotting remain next. Full F01-F22/WP0-WP6,
broader computational/input domains, complete per-stage replay, original/SMPTS
execution, fresh full-suite before completed handoff and laboratory gates stay active.

Earlier Bayes training desktop/project checkpoint (8 October): the native Bayes
dialog now offers Train from samples with asynchronous CUDA fitting, Stop,
worker draining and atomic saving of the exact interoperable PlantCV TSV.
Only a committed, reread and byte-verified TSV becomes the selected model.
The optional canonical training record stays in recipe/project metadata;
the standalone TSV remains unchanged and legacy five-field selections work.
Typed bounded records bind model/tables/classes and training facts before
asset reads. Keep in project, Save/Open/move, actual companion batch and
whole-run archives retain the record and verified model copy after original
samples/model files are removed. Origin paths are descriptive metadata.
Annotation-only changes reuse numerical products while refreshing current
node parameters and archive metadata. Verified runner closure includes the
actual provenance codec body. The parent releases completed CUDA fits after
the modal worker drains. Failed/stale/cancelled training publishes no selection.
The final12-module compatibility gate passes141 tests in228.543s
(artifacts/naive-bayes-training-ui-reviewed-compatibility-gate.log). The separate
strengthened integration/model-assets/register gate passes43 tests in18.178s,
including15 new integration cases. Actual Windows-font training and1440x900
classifier views were inspected. Bounded review covers supplied projections and
root execution evidence for the eight frozen files in
artifacts/naive-bayes-training-ui-frozen-hashes.json; no independent reads/run
occurred. The first141-test gate had15 common setup errors from a too-strict
range-module record schema; the narrow computed-constant fix and rejection cases
are included in both final passing gates. Previous core/API freezes and the
canonical source manifest are unchanged. No tests/preview process remains live.
Numerical conformance remains empirical; constant/roundoff-only channels,
image/mask-directory sampling, plotting and broader numerical domains remain
open. Full F01-F22/WP0-WP6, complete per-stage replay, original/SMPTS execution,
fresh full discovery before completed handoff and laboratory gates remain active.

Earlier sample-table Naive Bayes training checkpoint (8 October): the native API now
parses bounded sampled-RGB tables, preserves duplicate-header/sample order,
converts reusable samples to CUDA HSV and fits per-class/channel Gaussian PDFs.
The ordered CUDA fitter uses the pinned weighted covariance/Scott estimator and
evaluates all256 byte values without an Nx256 temporary or SciPy dependency.
Actual TSV output loads through the existing frozen-model parser/classifier;
compact provenance binds selected bytes, parsed samples, model/tables, loaded
calculation/runtime and the source pins. Numerical conformance is empirical:
reduction-derived budgets include a separately identified empirical allowance
for host/device transcendental differences. No byte-exact training or certified
classification claim is made. Constant/roundoff-only channels are explicitly
unsupported, including cases where the pinned SciPy source returns a tiny
positive covariance. One-class training follows the public implementation.
The strengthened five-module compatibility/register gate passes40 tests in41.535s
(artifacts/naive-bayes-training-final-gate.log), including15 new training tests.
Bounded approval covers the four frozen source/test files listed in
artifacts/naive-bayes-training-frozen-hashes.json. The reviewer used supplied
projections and root execution evidence, with no independent reads or run.
Image/mask-directory sampling, training Qt controls/atomic save, project-held
training provenance and broader numerical domains remain open. Full F01-F22 /
WP0-WP6, per-stage replay, original/SMPTS execution, fresh full-suite verification
before completed hand-off and laboratory gates remain active.

Next: extend saved-input replay beyond the thirteen source and canonical
calibrated seed-ROI contracts, beginning with remaining extended-threshold,
model, color/size and geometry/helper stages. Preserve exact declared product
types, every input branch, current loaded identity, archive verification,
resource admission and separate historical/active results.
Continue remaining helper/input/computational domains across F01-F22/WP0-WP6.
F13 corpus/contact/debris ablations, original/SMPTS execution, fresh full unittest
discovery before completed hand-off and laboratory evidence remain open.

Earlier replay increment (implemented within the declared scope):
implement the next bounded native per-stage replay increment using
the existing archive/checkpoint contracts. Keep explicit selected recipe/stage,
source/frame/input lineage, effective parameters, loaded implementation identity
and unavailable products; current whole-run archives are not complete stage replay.
Continue remaining helper/input/computational domains across F01-F22/WP0-WP6.
F13 corpus/contact/debris ablations and laboratory evidence remain open.

Earlier native integration task (implemented by the current checkpoint):
integrate the reviewed F13 recovery/annotation API as distinct native
working-recipe nodes with explicit supplied cores/elevation, frame/source
binding, node-local cache invalidation, effective controls, asynchronous
Qt execution and project/batch/archive output contracts. Radius/policy
edits must reuse the partition. Preserve unresolved coverage and flags
through measurements/exports; no scientific or hidden-outline claims.

Earlier primitive design (implemented by the current checkpoint):
Implement F13's experimental contour recovery from supplied eroded
core labels. The reusable workers returned text-only designs. This feature is
a proposed enhancement, not a recovered SMPTS erosion/contact algorithm. Do
not infer the unresolved F06/F08/F12 original-source rules or claim measured gain.

Start with a bounded CUDA primitive: original foreground, coordinate-validity
mask, positive labeled core markers and explicit float32 elevation. Define the
partition domain as foreground & validity; require markers inside it. Reuse
plantcv_watershed.masked_watershed's supported four-connected, compactness0,
watershed_lineFalse behavior. Keep explicit elevation semantics; an optional
named negative-distance producer can use plantcv_distance.distance_map. Never
silently choose an elevation or expose unsupported connectivity/method controls.
Leave unseeded components zero and report unresolved coverage. Preserve all
marker pixels/IDs and the original foreground union separately. Recovered
per-ID masks are watershed assignments; individual pre-contact outer outlines
are unknown. Flag shared interfaces, validity/canvas clipping and measurement
eligibility; do not claim complete hidden outlines or measure eroded cores as
the original seed. Avoid allocating K full-frame masks; bound per-label metadata.

Preflight combined retained inputs, distance/watershed workspace, labels and
uncertainty rasters; keep dense work CUDA. Guard versions/storage and callbacks,
withhold incomplete cancellation/mutation/cap results, and test actual streams/
peak. Compare the chosen watershed only to the actual pinned skimage operation.
Engineering fixtures must cover isolated exterior conservation, touching lobes,
unseeded components, invalid/empty markers, diagonal four-connectivity, validity
holes, canvas boundaries and cancellation. Split uncertainty-radius/eligibility
policy descendants from watershed so editing them reuses partition products.
Then add a separately named experimental native working recipe, honest controls,
coordinate-bound project/batch/export and focused Qt/cache tests. Scientific
gain, original SMPTS parity and laboratory gates remain distinct and open.

Bayes mask coordinates are now reviewed with258 compatibility tests. Broader
input formats remain open: a local read-only codec probe found Qt header support
for BMP/TIFF/WebP/PNM, while OpenCV also decodes HDR that Qt did not identify.
Do not remove bounded header admission or claim all source formats are covered.
Continue full F01-F22/WP0-WP6 and per-stage replay work. Run focused tests during
implementation and fresh full discovery before completed handoff. Do not rerun
update_bayes_mask_coordinate_docs.py or earlier one-shot documentation helpers.
No test or preview process remains live. Routine workspace edits remain authorized
under auto_review; the Windows apply deny-read ACLs startup failure does not
revoke approval. Keep unrelated project files intact.

Earlier calibrated seed workflow checkpoint (8 October): three additive
24-node native recipes connect automatic colour-card affine correction to a
required half-open seed ROI, corrected RGB-to-gray conversion, median/threshold/
fill/morphology/watershed and size/colour observations. Binary, mean and Otsu
variants retain the existing source operations and image-specific controls.
Labels and measurements use a declared local pixel frame; `labels_coordinate_frame`
and compact crop metadata preserve the exact translation back to the source.
Physical chip dimensions update only calibration metadata and derived size values;
pixel measurements and segmentation remain reusable. CUDA products, combined
preflight, verified helper closure and changed-node/descendant caching are retained.
Qt launcher/previews/Keep/reopen, project persistence, actual companion batch and
whole-run archives are implemented. The seven-module compatibility gate passes
100 tests in 207.409s; the six register checks pass separately. Actual1440x900
launcher, seed crop/labels and calibrated-size views were inspected. Bounded review
covers supplied source projections and root execution evidence only.
The selected public fixtures match all seed intermediates/labels/diagnostics and
colour histograms; physical size observations use source metadata and declared
numeric tolerance. Whole-image correction retains the one-byte tolerance. The mean
fixture's unstable ellipse size remains unavailable; no fallback is substituted.
Full F01-F22/WP0-WP6 remains active: broader computational/input domains, helper
graphs/training gaps, complete per-stage replay, original/SMPTS execution, fresh
full-suite verification before completed hand-off and laboratory gates remain open.

Frozen composed-workflow approval covers the four production files and final test
file in `artifacts/calibrated-seed-frozen-hashes.json`; the final test SHA is
`fd0ea9ebb69c99ad765f02463d790b43bc2beb5409f86e9ecb9e7946e793d452`.
The reviewer used supplied projections and root evidence, with no independent
reads/run. Final register checks pass6/6 in0.555s. No test/preview process remains
live. The canonical reference manifest remains unchanged. Do not rerun the
append-style `artifacts/update_calibrated_seed_docs.py` helper. Next bounded
implementation candidate: public multiclass Naive Bayes training; both reusable
workers returned text-only design. Inspect the actual pinned parser and SciPy
KDE covariance/bandwidth/evaluation source before implementation. Dense samples,
HSV/statistics and KDE work should stay CUDA; source failures, unsupported numeric
domains and native-versus-source tolerance need explicit treatment. Existing
frozen-model inference/export should consume the actual produced model bytes.

## Resume here

Earlier automatic-card working checkpoint (8 October): the additive nine-node
native recipe now exposes ROI-before-filter preprocessing, source-ordered chip
geometry, labelled circles/bounding mask, physical calibration metadata, chip
means, standard orientation and whole-image affine correction. Effective edits
recompute only their node and descendants; radius edits reuse geometry and
physical-size edits change only calibration metadata. Verified loaded helper
closure and fixed grid identity bind reuse. Native Qt controls/previews/Keep,
saved-project round-trip, companion batch and whole-run exports are implemented.
The nine-module compatibility gate runs 103 tests in 181.702s PASS;
the strengthened integration gate runs12 tests PASS in26.869s; after the
preview repair all18 card/shared Qt tests pass in38.226s.
Bounded review covers supplied source projections and root evidence only.
Full F01-F22/WP0-WP6, broader domains/helper graphs, complete per-stage replay,
original/SMPTS execution, fresh full-suite and scientific gates remain active.

Earlier primitive checkpoint (8 October): automatic-card CUDA primitives now
implement source-ordered contours, chip filtering, corner/grid geometry, labelled
circle and bounding masks, physical chip metadata and automatic affine correction.
Pinned OpenCV 4.11 rectangle arithmetic preserves ordered chip IDs across the
loaded OpenCV version. The focused six-module gate passes 49 tests in 147.705s.
Bounded source review covers supplied projections and root evidence only.
Automatic working-recipe/desktop/project/batch integration remains open; full
F01-F22/WP0-WP6, fresh full-suite and scientific gates remain active.

Current desktop checkpoint (8 October): correction Keep/reopen uses the exact
selected project image and verified four-file inputs. Keep stages bounded immutable
copies and merges compact metadata only after cancellation/staleness checks; Save
Project separately writes the master. Reopened edits and malformed-result guards
have reviewed regression coverage. See the latest checkpoint for gates and hashes.
Full F01-F22/WP0-WP6 stays active; routine edits remain authorized under auto_review.

Earlier bounded checkpoints (8 October): reviewed size tracing supports holes,
nested islands and disconnected regions, pinned contour order/hierarchy and all
17 public observations within supported geometry. Numerically ambiguous ellipse
orientations remain unavailable without angle normalization or relaxed tolerance.
Explicit line thickness now reaches size measurements and CUDA RGB diagnostics;
the source's hard-coded-five caliper fallback remains separate from overlay width.
One/five-pixel public observations and complete images pass, including the actual
isolated worker's one-pixel protocol on the tested fixture. The size/raster gate
passes 48 tests in 90.267s. Additive native working seed copies expose effective
thickness editing, node-local invalidation, async preview, reset and save; their
24-test API/Qt gate passes in 70.210s. Historical catalogue graphs are unchanged.
Gaussian kernels 11/13/15 use pinned float32 coefficients and withhold masks at
uncertain byte-rounding boundaries; work and actual multi-tile scratch stay bounded.
The extended-threshold gate passes 22 tests in 71.401s; the larger-kernel recipe/
desktop/cache gate passes 14 tests in 90.053s. These additions have bounded source
review. Actual 1440x1000 size-thickness and classifier views were inspected.
Frozen-model Naive Bayes has reviewed recipe/API/desktop binding and class previews.
Frozen-model K-means prediction is now implemented and independently reviewed:
37 focused tests pass in26.683s against isolated pinned PlantCV/sklearn calls.
This includes sigma5 preprocessing, ordered gray/RGB patch tiles, bounded numeric
model identity, exact integer ties and guarded fractional nearest-centre decisions.
Prepared CUDA centres are reused across tiles; actual allocation, combined work,
cancellation and input preservation are covered. The separate model/predict/masks
working recipe and desktop workflow are also reviewed;27 API/Qt/compatibility tests
pass in22.282s, including effective category edits and image/prediction cache reuse.
Canonical numeric-model export is also reviewed;24 focused model/prediction/workflow
tests pass in13.257s after the incremental8MiB serializer addition.
Reviewed training components now provide unpadded sigma5 full-patch extraction
and source-ordered unit-weight MiniBatch centre/count updates on CUDA. Eight
training-data tests and ten update tests pass against pinned public/Cython calls,
including exact binary64 values, measured storage, cancellation and mutation checks.
The prior reviewed training/watershed/register gate passes38 tests in47.640s.
Sampled sigma5 patches, exact estimator RandomState calls and byte-valued CUDA
k-means++ initialization are also reviewed, with9/7/7 focused tests passing.
Public PlantCV/sklearn calls verify row order, selected centres and exact RNG state.
The assembled training/register gate passes47 tests in18.869s.
Compact MiniBatch reassignment timing, scalar convergence and low-count planning
are now independently reviewed:10 focused tests pass in1.685s against actual
pinned estimator methods, including byte-exact updates/replacements/count resets,
RNG state and all-selected failure order. Scalar control requires certified
inertia inputs; the complete bounded byte-feature fitter is recorded below.
The combined control/RNG/update/register gate passes33 tests in5.636s.
Mutable-centre CUDA scoring/inertia and compact interval convergence are also
reviewed, with7/8 focused tests passing. Integral totals are exact; fractional
source reductions have certified bounds, and undecidable decisions withhold the
result. The scorer/interval/control/classifier/register gate passes37 tests
in7.315s. The bounded byte-feature MiniBatch fitter and public-tolerance producers
are now independently reviewed. Their strengthened combined gate passes12 tests
in5.469s against traced actual public fits, including every RNG draw, initialization,
update/reassignment/convergence decision and final outputs. A4097x243-row fixture
covers complete custom-stream allocation/cancellation; variance/shift intervals
retain source rounding and withhold undecidable decisions. Total job work and live
storage are checked before RNG/output allocation. Ordered CUDA image training
and compact fitted-model export are also implemented and independently reviewed:
8 image-training/interoperability and7 export tests pass. Source image order/BGR,
per-image patch seed1, complete fit parameters, reusable CUDA rows and combined
preflight are preserved. Exported numeric models match actual public joblib
predictions/masks; only compact centres transfer to CPU, with measured metadata
peak and incremental8MiB serialization. The metadata-only fit preflight now has9
passing fitter tests. Bounded directory selection/materialization is independently
reviewed:14 directory tests and9 ordered-image tests pass together in8.434s.
Selection preserves sorted names, independent Python random choices, duplicate
occurrences and the source's index-zero failure. Qt header metadata bounds OpenCV
decoding; all matching regular files, including decoder skips, receive byte identity.
Unique images upload once; repeated choices retain repeated CUDA feature rows.
Explicit all-reduction provenance is also reviewed:26 interval/scorer/fitter tests
pass in7.584s, including public fits at1/2/4 threads. No host count is inferred.
Native directory training, reusable selection/row caches, async controls, worker
draining and atomic model saving are independently reviewed:21 new/existing
API/Qt workflow tests pass in13.657s. Fit edits reuse rows; extraction edits retain
the file selection. Saved centres enter the existing picker and invalidate only
model descendants. Persistent training provenance is now implemented and reviewed:
one bounded artifact retains the ordered byte-identified selection, effective
settings, fixed patch seed, exact estimator RNG identity, fit plan/scalar intervals,
loaded producer/runtime identities and pinned sources. Legacy centres-only files
remain compatible; recipe selection binds the entire artifact's raw bytes.
Production PlantCV model conversion is now implemented and reviewed. Shared
standard-library codecs retain legacy import aliases; the explicit converter runs
in the source training environment and writes the same atomic numeric artifact.
Eleven conversion tests pass in6.181s against actual pinned public training,
prediction and masks;38 legacy/model/native tests pass in15.916s after the move.
Native whole-run output persistence and saved-project companion execution are now
implemented. The shared exporter records exact recipe/source/frame/runtime, ordered
node states and effective controls, compact observations, every available CUDA
product and verified raw classifier-model bytes. Desktop **Export run...** drains
its worker; project batch executes exact saved native-compatible attachments and
records failures or unavailable original/reference executors explicitly. Historical
graph records and baseline reports retain their own identities. Array groups support
256 ordered class masks; metadata supports the full 128-label colour domain.
These output archives are not per-stage replay certificates or scientific validation.
The reviewed native-output/project/register integration gate runs142 tests in109.556s:
141 pass and one existing platform-specific test skips.
Project schema6 retains editable native working recipes separately from historical
graph attachments, with safe migration from versions1-4. Seed, extended-threshold,
Bayes and K-means dialogs expose Keep recipe in project; Save Project captures the
selected controls, actual canvas layout and available verified model copies.
Project native recipes reopens or removes working selections. Historical recipes
and StageRunRecords retain their exact identities even when a working selection
shares an ID but selects a different model. Snapshot closure covers both collections.
Batch uses the same runner and records historical_graph or working_recipe rows;
working recipes have no fabricated historical stage records. Native output schema3
reads v1/v2 and additionally supports four-file correction outputs. The expanded18-module gate runs279 tests in233.278s:278 pass and
one existing platform skip. All16 new working-recipe tests pass in10.003s; the two
tests added after the expanded run are included in that final focused gate.
Bounded independent approval covers supplied source/test projections and root
execution evidence. Actual1440x900 overview/working views and menus were inspected.
Complete per-stage replay, original/SMPTS execution, remaining feature domains
and fresh full-suite/scientific gates stay open; correction project integration is implemented below.

Correction now has a shared four-file runner and whole-run output schema3,
with v1/v2 archive compatibility. All four images/masks retain raw byte identities,
decoded frames and decoder facts. Verified preparation/core/runtime identities
govern reuse and archive binding. Qt Export run saves corrected CUDA RGB and
compact immutable fit matrices, locks all input controls and drains on close.
The final eight-module gate passes 102 tests in 144.574s; bounded independent
review covers supplied source/test projections and root execution evidence.
Correction project image-to-mask bindings, assets/snapshots/batch and desktop
integration are now implemented in the newer checkpoints below. Full F01-F22/WP0-WP6 and fresh full/scientific verification stay active.

Saved per-image correction bindings now execute through the shared CUDA adapter.
Exact project image ID/locator and analytical recipe identity select the four pins;
identical RGB bytes do not merge distinct image-to-mask selections. Missing selections
produce unavailable rows; selected invalid inputs produce failed rows without origin
fallback. Other project images and baseline reports continue. Source and transport
byte reads remain bounded, including final callback-free publication audits.
Output schema4 reads versions1-3 and preserves original selection records separately
from the actual opened paths and current decoder facts. Compact origin metadata is
bound independently in checkpoint identity; local schema3 identities remain unchanged.
Batch ledger3 retains exact bindings and project image IDs, adjacent correction runs
reuse prepared CUDA inputs, and those inputs are released before generic graphs.
Project schema6 persistence/snapshots and desktop Save As/degraded recovery remain.
Bounded review covers supplied source/test projections and root execution evidence:
123 tests in411.355s (122 pass;1 platform skip), then 8 focused passes in 44.869s. The project orchestration test uses a fixture
baseline and actual CUDA correction companions; it does not validate production
baseline numerics or scientific accuracy. The desktop correction checkpoint follows.

Correction dialog Keep/reopen is implemented for the exact selected project image.
A successful current CUDA run enables Keep; changed inputs, failed/incomplete outputs
and stale worker results disable it. A frozen request carries recipe/layout and four
input records, never CUDA results. The existing persistence worker verifies bounded
source/input bytes and stages immutable copies. GUI completion checks cancellation
and exact project/image/recipe/binding state before merging compact session metadata.
Keep marks the project modified; Save Project separately writes its master.
Reopened runs use project pins and preserve original selection records separately
from actual reads. Editing a reopened mask/target starts a fresh selection using the
exact selected project RGB path; an explicit different RGB selection is preserved
and cannot be kept for that image. Closing, layout or parameter changes during Keep
discard the pending merge after the worker drains. Historical attachments remain.
Bounded independent review covers supplied source/test projections and root evidence:
87 tests in289.194s (86 pass;1 skip) before the final two reviewed UI repairs; then 31 tests pass in198.483s. The final guard reuses bounded output contracts;
it does not transfer the corrected image to CPU or rerun analysis. This checkpoint
adds desktop functionality, not scientific colour validation or full feature parity.
Full F01-F22/WP0-WP6, automatic card/helper and remaining computational domains,
complete per-stage replay, original/SMPTS execution, fresh full discovery and
independent scientific/laboratory gates remain open.
The reviewed conversion/dependency/register gate passes181 tests in81.100s.
Fresh full-suite verification stays open.
The assembled fitter/tolerance/scoring/convergence/control/RNG/initializer/update/
register gate passes67 tests in15.344s. The expanded image-training/export/full-
dependency gate historically passed121 tests in54.982s. The expanded directory/
native-workflow/dependency/register gate passes161 tests in73.954s; fresh full-suite
verification remains open.
The last full discovery ran 1149 tests in 1544.843s: 1144 passed, two UI tests failed
and three skipped. Both UI failures are repaired/reviewed with focused passes.
That full run predates current additions; fresh full verification remains required
before hand-off. Remaining numerical/drawing/threshold domains, Bayes training and other K-means domains,
automatic card/helper/SMPTS execution, complete saved-project/batch reports and
model/checkpoint bindings, and laboratory gates remain open. All F01–F22/WP0–WP6
stay in scope. Earlier full-suite evidence is historical.

Project edits remain authorized and Approve for me is active (`auto_review`).
The failing shell sandbox startup applies deny-read ACLs; the direct workspace
patch editor works. Worker reads and editor calls have also stalled, so root now
owns every file write and scoped execution. Workers return plain-text patches and
reviews only; do not retry their editor or shell paths or ask for the same approval.
Root reconfirmed the normal shell startup failure in this session even for a
HANDOFF read; this is a Windows sandbox setup error, not a denial of project-edit
authorization. The permitted root escalation runs successfully through auto_review.
Routine project edits use the direct patch editor; preserve authorization across
turns and do not request it again. No app setting change is required from the user.
Preserve the
unrelated project JSON unopened/untouched/unstaged. Full F01–F22/WP0–WP6 scope
and the same reusable worker pool remain active.

### Desktop correction Keep/reopen (2026-10-08)

Correction dialog Keep/reopen is implemented for the exact selected project image.
A successful current CUDA run enables Keep; changed inputs, failed/incomplete outputs
and stale worker results disable it. A frozen request carries recipe/layout and four
input records, never CUDA results. The existing persistence worker verifies bounded
source/input bytes and stages immutable copies. GUI completion checks cancellation
and exact project/image/recipe/binding state before merging compact session metadata.
Keep marks the project modified; Save Project separately writes its master.
Reopened runs use project pins and preserve original selection records separately
from actual reads. Editing a reopened mask/target starts a fresh selection using the
exact selected project RGB path; an explicit different RGB selection is preserved
and cannot be kept for that image. Closing, layout or parameter changes during Keep
discard the pending merge after the worker drains. Historical attachments remain.
Bounded independent review covers supplied source/test projections and root evidence:
87 tests in289.194s (86 pass;1 skip) before the final two reviewed UI repairs; then 31 tests pass in198.483s. The final guard reuses bounded output contracts;
it does not transfer the corrected image to CPU or rerun analysis. This checkpoint
adds desktop functionality, not scientific colour validation or full feature parity.
Full F01-F22/WP0-WP6, automatic card/helper and remaining computational domains,
complete per-stage replay, original/SMPTS execution, fresh full discovery and
independent scientific/laboratory gates remain open.

Root owns file I/O and execution; the same two text-only workers and independent
reviewer supplied lifecycle/state designs and bounded source/test review. Initial
five backend tests pass in24.420s. Initial11 Qt methods ran116.193s and exposed
incomplete-inventory preview handling plus two fixture errors (import/read-only
path setter). Three repaired methods pass in18.422s. The expanded five-module
gate predates the output-validation and reopen/edit repairs; the subsequent
two-module gate includes both repairs and their new regressions.
Logs: artifacts/correction-project-dialog-initial-qt-gate.log,
artifacts/correction-project-dialog-repaired-gate.log,
artifacts/correction-project-dialog-final-gate.log and
artifacts/correction-project-dialog-post-review-gate.log.

Final review is bounded to supplied implementation/test projections and root
execution evidence, without independent full-file reads or execution. Five-file
freeze (earlier shared UI freezes are historical; batch/archive11 remain unchanged):

| File | Reviewed SHA-256 |
| --- | --- |
| seedvision/persistence/correction_session.py | 69099762f926932052f439f7bb825c9ddffeda75f0aae33d263cc80dd7a26943 |
| seedvision/ui/native_correction_dialog.py | 9c0ad193b8a05a2f9e47b89824d52e363d6657390179d6b93a02552087a18e19 |
| seedvision/ui/main_window.py | c647be349e53763d037fdcbe88eefaac949429d4695b373e3e1499f45c2dbd69 |
| seedvision/ui/reference_graph_dialog.py | b6030e73d2bea1abd5b1de6c0409724b7124beb640b5bd5296308c5bb66d70cd |
| tests/test_correction_project_dialog.py | 2c8896ab5a2b33cbb75b5c20873d9be3902ab56b2295dacb85c24dbd4bb5795f |

Next: continue automatic card/helper and remaining supported computational domains,
complete replay/original/SMPTS execution and the final full/scientific gates. All
feature status axes remain partial/pending/unvalidated where previously recorded.
Do not ask again for routine project-edit permission; authorization persists.

Post-document checks: six feature-register tests pass in0.484s after reconciling
historical correction pending clauses. Scoped diff whitespace checks pass with
existing LF/CRLF notices only. All5 desktop and11 batch/archive frozen hashes
remain unchanged. Seven documents and six register rows retain independent feature
status axes; no full-goal, fresh full-suite or scientific completion is claimed.

### Historical project correction execution and archive origins (2026-10-08)

This checkpoint adds `benchmarking/project_correction.py`, exact per-image batch
lookup and schema4 whole-run output origins. The unlimited complete-feature goal
stays active, with the same two text-only workers and reviewer. Root owns file I/O.
Ordinary edits remain authorized. Recorded Windows sandbox startup ACL failures
are separate from task authorization; root escalation succeeds through auto_review.
No app setting change or repeated project-edit permission is required from the user.

The adapter snapshots immutable binding metadata, checks native-compatible recipe
mode and exact image locator/ID/SHA/frame, and reads only verified project pins.
Original files remain archival metadata. Source/asset verification uses the bounded
copy helper. After the last callback, adapter/validator producer identities, image
and asset bytes, prepared tensor signatures and CUDA/compact output signatures must
still agree. Failure evicts only the correction node. Locator producer recording
covers the selected function dependency closure and its checked global constants;
unrelated project UI default factories are outside this footprint.

Origins preserve the binding and logical project locator independently from actual
input_sources/preparation paths and current decode facts. All four SHA/size/frame
records must agree, while original path/decode strings may legitimately differ.
All three project producer fields and the origin bundle must occur together.
Schema4 reads versions1/2/3, retains local schema3 checkpoint identity, rejects malformed
metadata before array restoration and never probes input files while loading.
Ordinary native results cannot claim correction origin records.

Batch uses exact image ID plus recipe ID/version/analytical SHA. Missing selection
is unavailable; selected invalid/corrupt inputs fail their row, with no fallback.
Only typed correction-input load issues are tolerated for row-level reporting.
Baseline/source/sidecar/classifier handling remains as before. A verified matching
source asset bounds all outer image guards; an invalid descriptor remains a failed
row rather than becoming a global guard. Adjacent correction runs reuse one prepared
bundle; it is released before generic graphs and before the next source image.
Ledger3 includes project_image_id and correction_binding, preserving history.
All successful pins are audited without callbacks before atomic publication.

Evidence: initial38-case gate failed on the unrelated metadata default factory and
an outdated width-fixture transport identity; it also accidentally rediscovered24
imported fixture tests. The narrowed recorder, corrected fixture and module import
repair produce14/14 passes in168.917s. Independent projection review then closed
the unbounded outer image guard and invalid-byte-count row-isolation edges.
The expanded seven-module gate runs123 tests in411.355s (122 pass;1 platform skip).
It precedes the final two-file byte-count guard/test repair, which has 8 focused passes in 44.869s.
Logs: artifacts/project-correction-runner-before-byte-count-fix-gate.log and
artifacts/project-correction-runner-final-gate.log. The wrong-byte-count regression
first reproduced the global-abort error in1 test/8.445s, then passes after bounded
candidate verification. Only the ledger module and its test changed after the
expanded gate; the before/after hash artifacts preserve that distinction.
The actual project runner
is exercised with a fixture baseline and actual CUDA correction companion paths;
this is not a new production-baseline numerical validation. Existing platform skips
remain explicit. Focused ongoing integration evidence is not fresh full discovery.

Bounded independent approval covers supplied source/test projections and root
execution evidence, without independent full-file reads or execution. Current
eleven-file freeze (older shared-file freezes are historical):

| File | Reviewed SHA-256 |
| --- | --- |
| seedvision/persistence/correction_inputs.py | 854b4602ebbab464fb67b18dc98a24d6a3936a13dbdfd239939022b7e4ed27b5 |
| seedvision/benchmarking/native_correction_recipe.py | 424a85211878073c9d02b9b2a236aaab25366e3d177621b96444cefb25753025 |
| seedvision/benchmarking/project_correction.py | 2fbd6c1d4c344a3d6df32fba71252c79b7634075dddb7f8e882947a56af27a35 |
| seedvision/export/native_results.py | e06d378122ddebf91ce441e890ebf19b4539ac553772dc198bca94785992fe44 |
| seedvision/export/project_native_graphs.py | bfb5fe65d5a0ce4e3e5505a688856aab15f32bd74e7c0ed2a4d314fe1a3ed56a |
| seedvision/export/project_runner.py | 2dd241f3e3de28148479a8cb66486b1f866ce8cc569eaf9eedd79c4695a9e556 |
| tests/test_project_correction_runner.py | 5b1177503fa3e92691593e96fa401df0da5ea42e2c966cff0217c47f6c31a185 |
| tests/test_native_correction_files.py | a4d5cfbe7814583bd7eae5e872043e3b4e84a5648c3bdb15f4c29ab881e92030 |
| tests/test_project_native_recipes.py | 70275e5d007fa50ec7ff6e18b25e6acce27813491e67f223c10abcf6909aefd5 |
| tests/test_reference_model_assets.py | 1428a41e09e7bf9084e89c183f26ab59d2a8a37c17afd6cbf7887ecafb304281 |
| tests/test_native_result_export.py | 3dbe45a5d43f9cf35411a1615a63130b48aa57a786c2a0ccd3bd86bb1e022bff |


Historical next-step plan, now implemented above: correction dialog Keep/reopen uses an explicit current project
image ID/locator, never SHA-only lookup. Keep requires a successful current run;
picker edits and stale worker results disable it. Unsaved image metadata and bounded
assets use the existing persistence worker. Persist compact immutable records only,
not CUDA results. Reopened runs use project pins and preserve separate origins.
The same worker pool has supplied the request/state and focused Qt-test designs.
Keep project now appears with an explicit selected image and successful current run.

Seven documents and six register rows record this checkpoint; feature status axes
remain partial/pending/unvalidated. Full F01-F22/WP0-WP6, automatic card detection,
remaining numerical/input domains, per-stage replay, original/SMPTS execution,
fresh full discovery and scientific/laboratory validation are still open.
Post-document checks: six register tests pass in0.528s; scoped diff whitespace
checks pass with existing LF/CRLF notices only. All11 frozen source/test hashes
remain unchanged. No complete-goal or fresh full-suite claim is made.

### Per-image correction project inputs (2026-10-08)

Historical persistence checkpoint; batch execution and archive origins are now
implemented in the newer checkpoints above, including correction Keep/reopen.

Previous goal turn made concrete progress: shared runner/export review,102 passing
tests, seven documentation updates and six passing register checks. This turn adds
actual schema6 persistence/capture/snapshot/UI passthrough. The unlimited complete-
feature goal stays active; the same two text-only workers and reviewer were reused.
Root owns all file I/O. No renewed permission or commit/push was requested.

`persistence/correction_inputs.py` defines frozen ordered four-role assets and
per-image bindings keyed by exact image ID and analytical recipe SHA. Source RGB
hash/frame must match that exact project image record. Same-byte images are not
interchangeable. Layout/presentation edits preserve the analytical binding. The
union of historical and working selected correction recipes determines retention;
history and stage-run records are not fabricated or rewritten.

Origin path/SHA/bytes/frame/decode stay separate from project-relative transport.
Frame/decode are historical declarations, not an independent decode certificate.
Pure metadata validation checks canonical paths, exact roles, paired frames and
pixel limits; at most256 bindings and1GiB of unique raw inputs are allowed, with
256MiB per file. Streamed hash/copy reads at most selected bytes+1, using1MiB chunks.
Sources, temporary files and target copies are verified before immutable publication;
final callback audits use no later caller callbacks. A failed multi-asset capture
may leave unreferenced immutable objects, but cannot publish a changed master.

Capture defaults preserve every declared pin. `existing=None` means current binding
records are already pinned; initial/reselected origin capture requires the explicit
prior tuple (`()` for wholly new records). Missing/corrupt bound paths never fall
back to originals. Byte verification/copy needs no full-resolution CPU decode or
new CUDA allocation. Actual shared CUDA correction after relocation has equal pixels.

Schema6 requires correction_bindings; schemas1-5 migrate empty and reject injected
future fields. Snapshots copy correction inputs into their verified inventory.
Image paths use images/<old64heximageid>/<sha><suffix>, preserving distinct logical
IDs for duplicate bytes; explicit old/new ID maps are injective. Restore re-roots
transport and image IDs without changing origin records or per-image masks.
Real desktop Save As and New retain/clear loaded bindings. Degraded saves keep
unresolved image/binding/selected-ID metadata; restoring the source retains the pin.

Evidence: initial17-test gate had2 parameter-key-order fixture assertions and1
platform skip; semantic payload/exact binding assertions repaired them. Expanded
105-test gate had one outdated expected-schema-field fixture; repaired. The final
six-module gate runs107 tests in131.891s:105 pass,2 platform skips (Windows symlink
privilege1314 and existing case-sensitive-filesystem requirement). Root115280/
session42983; artifacts/project-correction-persistence-final-gate.log. A subsequent
correction-specific degraded-source test plus3 entrypoint checks pass4/4 in8.494s,
root5d05b0/session57195; artifacts/project-correction-degraded-entrypoint-gate.log.
Production code and eight frozen files were unchanged; only one test method was
added after the107-case gate. Earlier hashes remain in the before-degraded-test
artifact. This is focused ongoing integration evidence, not fresh full discovery.

Bounded independent review approves supplied source/test projections and root
execution evidence only, with no independent full-file read or execution. Review
closed the helper's final-callback source audit and implicit-origin-fallback default.
Historical schema6 persistence nine-file freeze:

| File | Reviewed SHA-256 |
| --- | --- |
| seedvision/persistence/correction_inputs.py | b6528a3f5e8141e571292bbf353e88fed7d57440fe05127a6e858c2cfeca932a |
| seedvision/persistence/project_analysis.py | e72e7f28222d5a40d8eff6d8c95f386a5bc44b703e7f7af61f15e6111a3b437b |
| seedvision/persistence/snapshot.py | 76500f88187d70c6172c87168f53b0f162954ecb088027e6d41d459e47530e5c |
| seedvision/ui/main_window.py | 2f11b5d8a1928004170473af2b4f35b413b8e895869bbb2a227ee967b8c22851 |
| tests/test_project_correction_inputs.py | bfbcc9ee2c0550356ec5d6dea84189bcefe302108bb38678ac6ef4a46bc923a7 |
| tests/test_project_analysis_persistence.py | 039e1824203e9627a3306e1d0d782842de46708cff4e4ad1df4561192fedecc2 |
| tests/test_reference_graph_persistence.py | 11f3afc60cd883ff0fc89eab391c177f8d409c180d0a637fc63c2f244f52ff02 |
| tests/test_reference_model_assets.py | e2864b67261197acc37e9263c1a64b49d386b9eb6ab7bcc7f140960d04af4e8a |
| tests/test_project_native_recipes.py | eb2c7359095ca2567a2506a08ce2f97cfe054218511f316e383b3be6b68df081 |


Next implementation uses the same workers. Keep/reopen must explicitly select the
project image locator/ID, never search by matching SHA alone. Unsaved project images
need current metadata captured through the existing persistence worker. Keep is
enabled only for a successful current four-file result, and input edits invalidate it.
Reopened selections execute verified transport while preserving origin records.
Batch needs exact per-image bindings: missing selection is unavailable; selected
corrupt/undecodable input is failed with no fallback or result publication.
Current result/archive schema3 records actual opened paths and preparation strings,
which include path/device/runtime. Add a separate optional origin binding in output
schema4; do not rewrite actual paths/decode strings to pretend originals were read.
Compare original vs actual hashes/bytes/frames, not full historical decode strings
that legitimately differ by path or runtime. Bind origins independently in report
identity, update cache-hit provenance per run, and retain v1/v2/v3 archive readers.
Keep project stays hidden until this end-to-end adapter exists.

Seven docs and six feature rows record this checkpoint with24 new test references;
status axes stay partial/pending/unvalidated. Correction Keep/reopen/batch and archive
origins, automatic card detection, full numeric/input domains, per-stage replay,
original/SMPTS execution, remaining features, fresh full suite and laboratory gates
remain open. Do not mark the complete-feature goal achieved.

### Four-file correction runner and output export (2026-10-08)

Same two reusable text-only workers and independent reviewer; root owns all I/O.
Project edits remain authorized under auto_review. No renewed user confirmation
or settings change is needed. Preserve the unrelated project JSON untouched.

The new benchmark module extracts preparation from the desktop. It hashes bounded
file streams, then decodes those exact byte snapshots. Qt reads supported eight-bit
PNG/JPEG/BMP/TIFF headers before OpenCV allocation. Corresponding image/mask shapes,
equal image heights, aggregate decoded CPU bounds, 64M combined image pixels and
256M prepared CUDA bytes are enforced. A previous bundle additionally requires
verified loaded preparation code, decoder versions, four hashes/sizes/device and
tensor storage/version identity. Replacement can temporarily retain old/new bundles;
the preparation bound is per bundle, not a claim about total process peak memory.

Native correction working provenance is explicit; historical catalogue recipes are
unchanged. Four input records preserve actual opened path, SHA/size/frame and decode
facts. Paths are canonical cross-platform absolute text and are not probed when an
archive loads. Future project origin/transport bindings are separate work. The file
result composes verified preparation and tensor-core identities with actual runtime.
The final caller callback precedes the last implementation/file/input/output guards;
final bounded byte auditing makes no further caller callbacks. Correction-specific
cache/output signatures cover CUDA versions and compact matrix shape/dtype/stride/
writeability/pointer/bytes, including mutable NumPy metadata on readonly backing.

Whole-run output schema3 reads v1/v2 and adds four source records plus the composed
implementation SHA. It saves source-frame RGB uint8 and three finite float64 compact
tables. Matrices are bounded immutable bytes-backed CPU metadata, never a general
CPU-raster exception. Explicit port specs/roles are checked before array loading.
Actual Qt export locks all four selectors and drains/cancels before close. Selected
recipe save stores native graph/layout, not input files. Keep project remains hidden.

Review repairs canonical path checks, preparation/reuse provenance, final callback
ordering and matrix/output cache signatures. The initial23-test gate had one compact
manifest-role representation error, repaired. Focused publication/cache tests pass;
one intermediate injected-callback fixture used the wrong version-call count and was
replaced with final-phase mutation tests. The full current evidence follows.

Final8-module gate: 102 PASS in 144.574s, root7bd1f5/session14965;
artifacts/correction-files-export-final-integration-gate.log. This is focused
integration evidence, not fresh full discovery or scientific colour validation.
This historical freeze is superseded for shared schema6 compatibility tests.
Bounded independent approval binds these9 files; review used supplied source/test
projections and root evidence, not independent full-file inspection or execution.

| File | Reviewed SHA-256 |
| --- | --- |
| seedvision/benchmarking/native_correction_inputs.py | 3e7e111ff42c77f8dbb73e9261ecf61ef4d8782e12c8331ce61db2f977131f8c |
| seedvision/benchmarking/native_correction_recipe.py | cb6156690bceff8836e47e833a6256e4b1fd12811e6668099ac0b31b940a961d |
| seedvision/export/native_results.py | fae649bea7bd0a1cff4d3f0d201c16163e87b75799497a303fe9569111d108cf |
| seedvision/ui/native_correction_dialog.py | d454ced54dc7e145f99dd6cfb2684dd4a035c1fd37b67358135259dd31996929 |
| seedvision/ui/native_recipe_dialog.py | 1749b87870362d098ae3fad5d8b196a20acf2194b6e67b2771a7ee54515b2c36 |
| tests/test_native_correction_files.py | 9be878bbd8bd0dd2141811f7f55af7abb11332dad163010f2ab4ad6150c5d12f |
| tests/test_native_correction_recipe.py | 143bee1cd24304ad1e40b448ca0779e2225610cc511f5699265e6a1cd92a60ee |
| tests/test_native_correction_dialog.py | d6aa1b751688d3f4d62a172d241e2cf944b91b239be0e4ef824890018905dedb |
| tests/test_reference_model_assets.py | 5c440fba502b5fb87b6a3050197f63abbf328995ab1371e9d49547a9a4640094 |

Register F03.2/F19.1/F19.2/F20.1/F22.1/F22.2 gains shared runner/export mappings and
focused test references. Status axes remain partial/pending/unvalidated. Seven
documents record the checkpoint. Correction project asset/per-image source-mask
binding, snapshot closure/batch execution, per-stage replay, original/SMPTS execution,
remaining numerical/input/feature domains, Bayes training, independent laboratory
gates and fresh full discovery remain in the active complete-feature goal.
No commit/push was requested. Do not mark the goal complete.

### Editable native working recipes in projects (2026-10-08)

Same reusable two text-only workers and independent reviewer; root owns all I/O.
Approve for me remains active (auto_review), and prior project-edit authorization
persists. The Windows shell sandbox setup still fails applying deny-read ACLs;
direct project patches and scoped automatically reviewed root commands work.
No repeated user permission or app setting change is required for routine edits.

Project schema5 adds a bounded immutable native_recipes collection: one working
selection per recipe ID/version, at most16 recipes and8MiB aggregate metadata.
Versions1-4 migrate to an empty collection. Historical reference_graphs remain
exact separate RecipeSpec/StageRunRecord attachments. Model capture validates the
union, deduplicates exact selection keys and binds differing models independently,
including when history and working selections share recipe/stage IDs. Pinned bad
copies never fall back to origins. Snapshot creation/restoration requires complete
verified copies for both collections. Save As retains workspace-root semantics.

Seed, extended threshold, Bayes and K-means dialogs add Keep recipe in project.
This updates the draft project; explicit Save Project publishes controls/layout
and model copies. Project native recipes reopens/removes saved working selections;
factory launch actions prefer the saved working choice. Historical launches remain
exact. Keep neither changes the running recipe/result nor invalidates node caches.
Initial canvas overlap adjustment now has an opt-out for saved native layouts.
Explicit Qt submenu ownership fixes the initial deleted-wrapper failure. New clears
draft selections/assets. Unknown/stale metadata remains preserved and execution
reports unavailable. The separate correction dialog is not integrated by this slice.
The standalone source picker is not newly persisted; batch uses project images.

Batch ledger schema2 distinguishes historical_graph and working_recipe sources.
Both run through the same native runner with exact selection-keyed transport and
normal node-local cache reuse. Working rows have no fabricated historical records.
The production baseline settings and separate whole-run output schema remain intact.

Initial focused58-case run had four errors: the submenu lifetime defect and three
new fixture assumptions. The repaired14-test run then exposed initial canvas
overlap adjustment; after repair all14 passed7.447s. The expanded18-module gate
runs279 tests in233.278s:278 PASS/1 existing platform skip, no failures/errors;
rootbd3792/session88039, terminalef4c65,
artifacts/project-native-recipes-integration-gate.log. Two additional copy-failure
and seed/threshold/Bayes workflow checks pass2.987s (root0d6c73). The final complete
new module runs16 PASS in10.003s, rootd64200/session86812, terminal3ee060,
artifacts/project-native-recipes-complete-focused-gate.log. All processes are closed.
Actual1440x900 overview and kept native calculation plus Open/Remove menus were
rendered and inspected. The offscreen renderer loads Segoe UI explicitly; production
font behavior is unchanged. Screenshots/logs/helpers remain ignored in artifacts/.

Independent bounded approval binds the following16 files from root69bb01. Review
uses supplied source/test projections/deltas and root evidence; it is not a full-file
read or independent suite execution claim. This is a historical freeze; later
correction changes supersede the shared dialog/test hashes below.

| File | Reviewed SHA-256 |
| --- | --- |
| seedvision/persistence/native_recipes.py | 789daf155f9765f75a9fe053c9f952665be5e210f6d8c3c7638841a9807a4607 |
| seedvision/persistence/reference_models.py | 8b7443929e48da57245ab0b6b0f78df6b25a8570d0c9300a50f3a63962c3e17c |
| seedvision/persistence/project_analysis.py | 669fa3256cc6515e18a53c0fe24f2e068f0fd12f97f77406c26fd1e23c8f4b04 |
| seedvision/persistence/snapshot.py | 71f48f7cd2440875df90c2d9a27f6f60e97c72301047fc254e916288bf135b9a |
| seedvision/export/project_native_graphs.py | 64cbe1b0cb7cc41adb38af6ee136d9c6854990126cccf98632ce6c0e3204a866 |
| seedvision/export/project_runner.py | baf5c27b86ec3dd68029a5f1cca90df159dd1e0ae092deb9491c268f25cdf39e |
| seedvision/ui/native_recipe_dialog.py | e807e1dbf2f38755756be7164b04ba35bb1bcbc9178680ef942c24b517cffed4 |
| seedvision/ui/native_naive_bayes_dialog.py | 860d1f1cb8d02129a2ecbd5a0ae1132aa99202e9a58fc042832af19fcc7590a3 |
| seedvision/ui/native_kmeans_dialog.py | c896856d7f63cc93652cd56afb2ec954481190d19aeaf5ac995cc298300091f5 |
| seedvision/ui/reference_graph_dialog.py | 79844e0ff07a41e9debb0e5a479bf66d51ab11b947247167e0d97b369751e02d |
| seedvision/ui/main_window.py | eae0c8e3defcbeddd7432eceba6b3e12b73af56eef178459b55e0b5bb2ceebdc |
| seedvision/ui/pipeline_canvas.py | 1aebe66b1fa44bdcb7f716cc9dbdbd7c3caf5c31fb0b02f64576eddf44f346b8 |
| tests/test_project_native_recipes.py | 47cf36b9dcf46104b172d5df533651869451fe7f261c811e7129f2af80100d8a |
| tests/test_reference_model_assets.py | 59e0bcba0b2b6aeb2708e070f7b5382ea1b3dc7ebd09d06a342d885ceb50e56a |
| tests/test_project_analysis_persistence.py | 032fa22da373d3cdbd6cd90f53adb858faee5499c35abb67ab4bb6e707c3d23b |
| tests/test_reference_graph_persistence.py | 252b9997632f6ae53672578525a7b8e7bd12d1ee336625dfce6cfdc9612f6891 |

Register F05.2/F19.1/F19.2/F20.1/F22.1/F22.2 gains the working-recipe paths and all16
new test references. Status axes remain partial/pending/unvalidated. Seven documents
reconcile this checkpoint. The final register gate passes6 tests in0.470s,
root2bbfea, artifacts/project-native-recipes-register-final-gate.log. Scoped
diff --check passes (rootff5d54; line-ending warnings only). Current planning
documents contain no remaining claim that generic working-recipe attachment UX
or classifier-model snapshot portability is still unimplemented.
Correction project integration, per-stage replay,
original/reference/SMPTS execution, Bayes training, remaining numerical/input and
other feature domains, independent laboratory gates and fresh full discovery remain
in the active full F01-F22/WP0-WP6 goal. Do not mark the goal complete. Preserve the
unrelated project JSON unopened/untouched/unstaged. No commit/push was requested.

### Stored native-graph model portability (2026-10-08)

Same reusable two text-only workers and independent reviewer; root owns all I/O.
Routine project edits remain authorized under auto_review. Root reproduced normal
shell startup's deny-read ACL error; scoped root escalations succeeded without
renewed user permission. Direct workspace patches also succeeded. Do not ask for
the same approval again or interpret this sandbox setup failure as a policy denial.

Project schema4 adds separate reference_model_assets descriptors for selected
models in stored graph attachments. Descriptors bind recipe/version/stage, complete
selected-parameter digest, exact raw SHA/byte count and workspace-relative object.
Immutable RecipeSpec/StageRunRecord and original model analytical identity remain
unchanged. Available raw model copies publish atomically before the master; matching
objects deduplicate. Capture prunes unselected descriptors while retaining every
retained graph's binding. Missing legacy models stay savable metadata with explicit
load issues; an unavailable pinned copy remains pinned and never falls back to its
origin. Snapshots require complete verified model closure. Direct Store.save remains
metadata-only; desktop Save/Save As uses capture and materializes available copies.

The shared native runner accepts a separate stage-to-transport map and rechecks
bytes/tables before preparation or cache reuse. Original origin text remains in
selected model identity; transport-only changes reuse all node product keys.
Foreign origins require a local copy and are not interpreted as native paths.
Output schema2 retains transport metadata and reads existing v1 archives. The
desktop saved-native graph launcher and project companions use the same map.
Model picking/training clears the old override through shared _set_model_source;
Reset restores the initial recipe selection plus initial override. SaveAs keeps
workspace-root-relative dependencies even when the JSON destination folder changes.

Independent review found generic snapshot copying could read beyond selected model
size following file growth. Both creation and restoration now use descriptor-size
stat checks, reads capped at bytes+1, and repeated source/target digest checks.
Restore decodes bounded master metadata to learn caps before inventory reads, then
requires the verified master document to match. Existing folder redirection rejects
before object writes; source growth and interrupted copies cannot publish completion.

Evidence: initial expanded model gate18 PASS in9.077s (roota7f4fb/session91301,
terminal213218), then bounded model/snapshot/review gate36 PASS in12.162s
(root4176fb/session77970, terminalf57288), and the added actual training-selection/
reset worker test1 PASS in1.074s (rootdba0ab). All logs are ignored under artifacts/.
The first broader invocation ran197 tests in161.666s with one test-import error
and one existing platform skip; its existing196 cases had195 passes. The reference
inspection module expects tests on sys.path; discovery repaired invocation and
all14 cases passed28.884s (root758541/session59720, terminal657a9b).
The corrected combined17-module integration gate runs **211 tests in188.167s**:
**210 PASS/1 existing platform skip**, no failures/errors; rootd2eb92/session6220
terminalf535e2, artifacts/reference-model-assets-reviewed-integration-gate.log.
The same independent reviewer approved the bounded source/test slice against all18
frozen hashes and this evidence. All test processes are closed. Root144aa6 matched
all hashes and all20 new test references in each affected row. Scoped diff --check
passed (root1b0d37; line-ending warnings only). Final register gate:6 PASS in0.452s,
root5f9ae0, artifacts/reference-model-assets-register-final-gate.log. Final scoped
diff --check passed (roote8022e; line-ending warnings only).

Bounded independent production-source approval binds these hashes, supplied in
roote2a6a4; it covers the described portability delta and repair, not independent
execution of the suite or unrelated historical code. Final test evidence above
includes all20 new cases. No source edits follow this freeze.

| File | Reviewed SHA-256 |
| --- | --- |
| seedvision/model_io/asset_paths.py | ea4d6d95465872bf75c8710acbd1d3fcbedc8b5f4f27017bd78f4cfc724418c5 |
| seedvision/benchmarking/native_kmeans_recipe.py | bca63b2ec65052d660332488f83b560ade2ecdf20453db3db797a980086e3da2 |
| seedvision/benchmarking/native_naive_bayes_recipe.py | c3a8acf5782a9b2fee59dae4f474fa9b9d55fc5e1e1bf6079b48b28874d66b2d |
| seedvision/benchmarking/native_recipe.py | a4dfb1b915a7c6c14aa1023e725b9f0eb8dde502bbd47ab1f61467673c54e03f |
| seedvision/export/native_results.py | 5c8185d7a1dbb0660a7e92418685b6984f913cf3ef2992f4f651f64b5e798001 |
| seedvision/export/project_native_graphs.py | 497a6945d7108d0ae9cb1037c6d9bd3640ddac210e40bcaec27a8e09835b9e2c |
| seedvision/export/project_runner.py | 2d846fae35cea38f9dc0bbeab5e7b3eca007a37e102460931dfb6f4795f5a958 |
| seedvision/persistence/reference_models.py | 7fa176a74c6192ae5ea1ce3460bebfc70e8064d38a9a2b1d3fa3974c0a854842 |
| seedvision/persistence/project_analysis.py | b15c5fc49f536703cb68fa0b8642b158d57d1a312481e3bfccd2624eec989fae |
| seedvision/persistence/snapshot.py | dfb6959714f2c22beaf62c8b560d6593d8ffd36e3e10bf878ad245b90b7e2df2 |
| seedvision/ui/native_recipe_dialog.py | d330b3786c05d5509d7ee6df9f00b716c964bd7b46b0e97bb95e4ac8698053dd |
| seedvision/ui/native_naive_bayes_dialog.py | eb5727b9d23d3762a7e8aa0c1932a19c0ea439adb8881273ebcf91ddd55b65f3 |
| seedvision/ui/native_kmeans_dialog.py | 73f9d8d146cbce08643b17ae5e3322d413399e8a23efe2e7ecea87f23724986e |
| seedvision/ui/main_window.py | 9070469da38116aee2c177eb25e0f337603cc247c749c0634a51a5c0e16b20db |
| seedvision/ui/reference_graph_dialog.py | d068bdde71bbcdcf9cfa2de818cd137f132eeb3fa239049d5a971bca09b54055 |
| tests/test_reference_model_assets.py | f2ddd0e359d887d282ed2517abe5b3896a01ed0cc1beac5f0c3a8d82ebb3987b |
| tests/test_project_analysis_persistence.py | 8550a64666b7c8c6f31725934f7699808a0b754f91e3663c0ffb26610961fe55 |
| tests/test_reference_graph_persistence.py | 752baec7c1026ae380148731ccb2d1667c6647a2284cf3e2d1aa90bef01b73a2 |

Register F05.2/F19.1/F19.2/F20.1/F22.1/F22.2 gains the tested model transport,
snapshot and saved-launcher paths plus all20 concrete test references. Status axes
remain partial/pending/unvalidated. Seven project documents reconcile current
schema and portability. This slice covers stored graph attachments; standalone
working native-session attachment UX remains open. Complete per-stage replay,
original/reference/SMPTS execution, Bayes training, other numerical/input domains,
independent laboratory gates and fresh full discovery remain in the full active
F01-F22/WP0-WP6 goal. Do not mark the goal complete. Preserve the unrelated project
JSON unopened/untouched/unstaged; no commit/push was requested or performed.

### Native whole-run outputs and saved-project companions (2026-10-08)

Further implementation progress using the same two text-only workers and reviewer.
Root applied every file and ran checks. Independent approval binds the five hashes
below from supplied source bodies/deltas and root test evidence; it is not an
independent execution claim. Routine edits remain authorized under auto_review.

`seedvision/export/native_results.py` exports/loads a separate version1 native
whole-run projection: exact recipe/source/frame/runtime/preparation, ordered node
states/reasons/effective controls/cache keys, compact observations, every actual
CUDA product and raw byte-bound Bayes/K-means model assets. Tagged mappings preserve
ordering; tuple/list/record and nonfinite observations remain explicit. Public
checkpoint groups retain256 ordered class masks within each group limit. Full
128-label/12-channel/256-bin colour metadata fits the bounded report. Global array,
archive-group and metadata preflight precedes any host transfer. Publication uses
same-folder staging, flush/fsync, cancellation cleanup and a tensor-version guard
after the final callback. The loader validates semantic relationships and aggregate
inventory before loading arrays, then verifies existing checkpoint/content/model
byte identities. It returns data without executing models or recorded classes.
Scope is whole-run outputs, not StageRunRecord/per-stage replay certification.

`project_native_graphs.py` uses the same desktop path and exact saved native-compatible
RecipeSpec. Companion archives/ledger publish atomically beneath each baseline
report. Original/reference graphs receive explicit unavailable rows; failures retain
their reason and no partial graph folder. Source bytes are rechecked before runs
and publication. Historical stage/display records are retained without modification.
The baseline result releases before native execution; each native result releases
after export, leaving cache eviction free to release its tensors. Image caches clear
between sources. Actual weakrefs cover both ownership transitions.

NativeRecipeDialog adds Export run on a worker with conflicting controls disabled,
Stop/close draining, prior control restoration and stale-result invalidation. It
retains the actual live result. Baseline report schemas/crops and existing graph
metadata remain separate; no new entry point, server or mandatory ML dependency.

Repaired focused gate: **22 PASS in14.625s**, rootc78353/session6903 terminal42167d,
artifacts/native-result-export-final-focused-gate.log. Earlier integration before
the final lifetime/runtime repairs:141 tests in109.602s OK with one existing
case-sensitive-filesystem skip (rootf0ec15), artifacts/native-result-export-integration-gate.log.
The final reviewed integration gate: **142 tests in109.556s,141 PASS/1 existing skip**,
root67a888/session62815 terminaldefba4, artifacts/native-result-export-reviewed-integration-gate.log.
All test processes are closed. Roote2bcd5 reconfirmed all five reviewed source hashes;
scoped diff --check passed (rootdd2c5f; line-ending warnings only).
Final documentation/register gate:6 PASS in0.441s, root5486d2,
artifacts/native-result-export-register-final-gate.log. Final scoped diff --check
also passed (root958e6e; line-ending warnings only).
Initial memory-test fixtures were corrected to use two distinct
supported K-means/Bayes graph identities and actual available CUDA products; no
production recipe/identity rules were relaxed.

| File | Reviewed SHA-256 |
| --- | --- |
| seedvision/export/native_results.py | 5a15afc51a25b21661882e35be8b591dba22de59c047ddfc079ef6f3ae03bcdc |
| seedvision/export/project_native_graphs.py | 860144c0f57ebfb211e683063c50321422ea8a16ac3e06360c6a8ca1735db68f |
| seedvision/export/project_runner.py | 5f4c54fa195fb196d26b78a757e37d88e5809ddad0d7e430852d0a7b54808a43 |
| seedvision/ui/native_recipe_dialog.py | fff4b324bb33ae4a8b71f9f5566310f9d09bef61a90a1721ccd9775a738a2015 |
| tests/test_native_result_export.py | acf97170099bf93c0b0bbdc48d64cd8997f5f33dc208cec71e29785ac218b27f |

Register rows F05.2/F19.1/F19.2/F20.1/F22.1/F22.2 gain the tested shared paths and
all22 concrete test references; functionality/conformance/science axes remain
partial/pending/unvalidated. Planning, status, architecture and operator documents
describe the output projection and remaining work. Complete portable model-path
rebinding/snapshot closure, per-stage replay, original/reference/SMPTS execution,
Bayes training, other input/numerical domains and laboratory gates stay open.
Fresh full discovery is required before the full goal's completed hand-off. Preserve
the unrelated project JSON unopened/untouched/unstaged; no commit/push performed.

### Production PlantCV model conversion and shared codecs (2026-10-08)

Previous goal turn made progress: persistent native training provenance was
implemented/reviewed and170 dependency/register tests passed. This turn makes
further progress: same two workers supplied conversion/compatibility guidance;
root applied all files and ran tests; same reviewer approved the eight frozen
files below from supplied source bodies/deltas and test projections. No independent
execution/full-suite claim or new worker pool. Full F01–F22/WP0–WP6 remains active.

The numeric-v1 and artifact codecs now live in inert seedvision/model_io. Their
functions/schema/validation remain unchanged; only docstrings/import paths moved.
Historical CUDA/benchmarking paths are sys.modules aliases to the exact same
module/global namespaces. Native training keeps the historical persisted artifact
producer label while hashing the actual moved implementation. Fresh import/help
tests forbid desktop/ML imports; actual source/constant checks and old-global
patches still work. Native training, save, picker and cache tests pass after move.

Explicit CLI: python -m seedvision.model_io.convert_plantcv_kmeans INPUT.fit OUTPUT.json
under the source PlantCV environment from the repository. Production never relies
on the ignored reference environment. The converter alone imports source ML
packages, reads/hashes one at-most64MiB snapshot, loads that snapshot, requires
exact MiniBatchKMeans class and finite float64 centre geometry/value limits,
uniquely derives supported p1..31/C1or3 from width, and preserves centre order,
duplicates, binary64 values and signed zero without refit. File byte cap is not
a total joblib-loaded-memory bound. Source class/bytes/environment/geometry and
canonical numeric identity persist in one at-most8MiB artifact. Complete encoding
precedes same-folder temporary write/flush/fsync/os.replace; failures retain the
previous destination. Same input/output and hardlink aliases reject before loading.
The desktop picker remains JSON; no new mandatory dependency/server/GUI entry.

Actual pinned public train_kmeans saves/reloads five gray/BGR models, p1/2/3/10/31,
with real source/version SHA assertions and held-out pixel fixtures. Both classes
are present in each two-class prediction case. Converted centres match exact
binary64 hashes; complete native labels and ordered mask bytes match actual public
predict_kmeans/mask_kmeans. A fitted source fractional tie still causes native
typed withholding after conversion. Strided duplicate/signed-zero centre metadata,
actual CLI,19 invalid/oversized models, import aliases, resource bounds and atomic
failure are covered. Generated files remain ignored temporary artifacts.

Codec/native gate: **38 PASS in15.916s**, artifacts/kmeans-codec-move-focused-gate.log,
root2c26f8/session13769 terminal08dbed0. Conversion gate: **11 PASS in6.181s**,
artifacts/kmeans-conversion-focused-gate.log, root62c8f8 terminal0. Two preliminary
test-harness errors were corrected: expected source-constant exception type and
fully fitting the ambiguity-control source estimator before public prediction.
No production tolerance/domain relaxation. Final related gate: **181 PASS in81.100s**,
artifacts/kmeans-conversion-reviewed-dependency-gate.log, root01127c/session5558
terminalf59ba40; all test processes are closed. It includes six register tests.
Final register text gate: **6 PASS in0.437s**,
artifacts/kmeans-conversion-register-final-gate.log, root537790 terminal0.
All eight reviewed file hashes were reconfirmed unchanged (rootcf8a40), and
scoped git diff --check passed (root96680f; line-ending warnings only).
F05.2 now records30 mappings and215 existing test references and remains
partial_native/pending_recipe_freeze/blocked_laboratory_inputs.

| File | Reviewed SHA-256 |
| --- | --- |
| seedvision/model_io/__init__.py | 083A5BA1E4D52764D2E93ADC152AC1DC25862486994EBF49D8C49261BB6D6D40 |
| seedvision/model_io/plantcv_kmeans_model.py | EEE6FCD6BB7E97934EADB31FB9203EFD7602AD829BDC5E06BCCD1CACD286D846 |
| seedvision/model_io/kmeans_model_artifact.py | F66EE7F6B3E8B57192172A59440626D8C7FB975418466061A2437E7D71B7EC78 |
| seedvision/model_io/convert_plantcv_kmeans.py | E11B02CA484660B554FD45F74542503B70DE244F56E8E874B7DC8C41BB7AE80D |
| seedvision/cuda/plantcv_kmeans_model.py | BE1E14C75CBE38A54C8A2B2A46C4EA7E5F6301C5B540C0918CB282A350B7DAC2 |
| seedvision/benchmarking/kmeans_model_artifact.py | 3105F45F4CFDE20489F1E210AC5812E011FD9CBF74144BE1E79646B873789959 |
| seedvision/benchmarking/native_kmeans_training.py | A2CC34A3FBA625051FDE8EA851709E69EF9EACF1C7289159396EE318BBAFC33B |
| tests/test_plantcv_kmeans_conversion.py | B6F898DD11BEFA380F89364A6BE2C294B4203DA2E166BA44BAFFB28AF204C46F |

Next: complete saved-project/batch model/report/checkpoint bindings and every other
unfinished F01–F22/WP0–WP6 requirement, including Bayes training, other domains,
automatic card/helper/SMPTS execution and laboratory gates. Fresh full discovery
remains open. Preserve unrelated project JSON unopened/untouched/unstaged.

### Persistent K-means training artifacts (2026-10-08)

Historical checkpoint: pending-conversion/import-seam notes and changed file hashes
below are superseded by the reviewed production-conversion checkpoint above.

This goal turn made progress using the same reusable workers and reviewer, with
root owning all writes/commands. Project-edit authorization remains valid and
auto_review is active. No new permission question or worker pool was needed.
All F01–F22/WP0–WP6 remain in scope; the goal remains active.

The new kmeans_model_artifact codec wraps the unchanged numeric-v1 centres format
in one at-most8MiB file. At-most512KiB typed provenance includes the ordered
source manifest/digest, effective request and fixed patch seed1, all effective
fit defaults and resolved plan, scalar intervals/steps/iterations, final exact
estimator RandomState identity, loaded row/fitter/runtime identities and source
pins. Native source_n_threads remains None/all-valid-openmp-reductions-v1.
Metadata validation checks types/bounds/digests and cross-field selection,
geometry, fit/default/RNG consistency. It is historical calculation provenance,
not authenticated scientific/reference-conformance evidence. Recorded work/peak
values are bounded, while the actual fitter preflight computes their proof.
No dense fit product/cache is traversed for provenance; a sentinel test covers it.

Completion retains artifact_payload; the dialog saves the complete payload through
the existing atomic QSaveFile. Recipe asset loading accepts both formats, retains
the numeric centre digest and binds source SHA/size to the actual outer bytes.
Provenance-only changes invalidate stale selections. The loaded artifact codec
is included in recipe execution identity. Existing picker image-cache reuse and
model-descendant invalidation pass with a real training artifact.

The same reviewer approved the seven frozen hashes below from root-supplied
source bodies/deltas and test projections; no independent reads/execution.
Initial focused gate:37 PASS16.133s, artifacts/kmeans-persistent-provenance-focused-gate.log,
root7aa892/session42686 terminalb792fc0. This preceded the exact producer-set
check and dense-projection test. Expanded run:164 actual tests PASS86.157s,
artifacts/kmeans-persistent-provenance-reviewed-gate.log, roota24a84/session95773
terminal4853401; one additional loader error was a misspelled non-existent test
module (test_feature_register, correct test_peer_feature_register). This command
is not a clean gate or full-suite pass. Corrected final gate: **170 PASS in84.380s**,
artifacts/kmeans-persistent-provenance-final-gate.log, root56ef3a/session36306
terminal49f3ba0; all processes are closed. The gate includes the six register
tests and the final frozen implementation/test hashes. All seven hashes still
match review; scoped git diff --check passes with only line-ending warnings.
Final register reconciliation: **6 PASS in0.424s**,
artifacts/kmeans-persistent-provenance-register-final-gate.log, rootebb5b3 terminal0.
F05.2 now records29 mappings and204 existing test references; family status stays
partial_native/pending_recipe_freeze/blocked_laboratory_inputs.
Fresh full unittest discovery remains required before actual complete hand-off.

| File | Reviewed SHA-256 |
| --- | --- |
| seedvision/benchmarking/kmeans_model_artifact.py | D920248C099107F72480594D52E6E992CAA39D8E806B874CBF0230635FF173E1 |
| seedvision/benchmarking/native_kmeans_training.py | 9E3BF8F71BEE938C3C256F4E59ADAA9FF1D7315E958AFFCCA027C1A9D08A77A2 |
| seedvision/benchmarking/native_kmeans_recipe.py | B072AEE6FAA3E0F4597D8AB8897FCB77F1A1F87247C7BEBCE9F0464C2F73AF43 |
| seedvision/benchmarking/native_recipe.py | 8E2F188416979EDC6592FC419F7B2BBC6ED5E9DBCE259C9492D2B95445EC0C7A |
| seedvision/ui/native_kmeans_training_dialog.py | 97CE153E13302F91B31F36110F59E5B08CF8433414724C1B1BFF2983F3E46C33 |
| tests/test_kmeans_model_artifact.py | E3A363CBBFFDDBB18A466124F1AA5A8B2FCB9619EA5E6FF1A2079B8E1EBD8975 |
| tests/test_native_kmeans_training.py | EA61B074A32FCBD7A9C670E88D93F674F831D74BC1B15F7394E61128438432F6 |

Next: production PlantCV .fit conversion under its own environment, then remaining
project/batch/model/report/checkpoint bindings and every unfinished feature family.
An actual pinned public train_kmeans model probe confirms a joblib MiniBatchKMeans
with float64 centres and n_features_in_, but no patch/channel metadata. Supported
feature width uniquely determines p/C. Probe script/output remain ignored artifacts.
The numeric codec file uses stdlib but cuda/benchmarking package initializers
eagerly import desktop dependencies. A source-environment converter must solve
this shared-code import seam, avoiding a desktop sklearn/joblib dependency or
ignored reference-env requirement. No production converter is implemented yet.
Keep the unrelated project JSON unopened/untouched/unstaged. No commit/push.

### Directory training, explicit reduction provenance and native workflow (2026-10-08)

Historical checkpoint: runtime-only provenance and next-work notes below are
superseded by the persistent-artifact checkpoint above; changed hashes are historical.

Previous goal turn made progress with reviewed ordered image training/export and
the121-test dependency gate. This goal turn makes further progress: the same two
workers supplied source/cache/UI guidance, root applied all writes and ran tests,
and the same independent reviewer approved the three frozen bundles below.
No new worker pool or permission request was needed. All F01–F22/WP0–WP6 remain
in scope and the goal remains active. This is a bounded engineering checkpoint.

Directory selection preserves sorted names, independent module-level Python
random.choices before substring prefix filtering, duplicates and the source's
uninitialized idx0 accumulator failure. Every matching regular file receives SHA
and byte facts, including failed decodes. Qt reads header dimensions/format only;
OpenCV decodes unchanged uint8 gray/BGR. File64MiB/source128MiB/decoded4HW128MiB,
16Mpixels/image,4096entries/256occurrences bound the declared domain before the
relevant read/decode/upload. WebP metadata lacks a Qt image format, so its explicit
byte-component format contract retains the4HW bound. Other unsupported header,
depth/channel domains fail explicitly. Unique files upload once; repeated selected
occurrences still duplicate patch rows. CUDA input plans charge unique full
underlying storage.nbytes. Structural/aggregate/hash validation and full byte/
listing rereads precede uploads and publication; a final callback-free pass catches
callback changes to earlier files. Actual public PlantCV selection/RNG/pixels/
features/fit results and source failures pass.

The reviewed interval proof already covers every reduction order using N-1
additions; no numeric arithmetic changed. Explicit required source_n_threads=None
now records all-valid-openmp-reductions-v1. Observed positive counts retain their
value and recorded-openmp-threads-v1 provenance, with the same all-order bound.
Public fits at verified1/2/4 threads, public inertia inclusion and typed ambiguity
withholding pass. Never infer a CPU/Torch/runtime count or substitute1 for None.

The native adapter retains a verified upstream selection across fit/extraction
edits. CUDA row keys bind manifest/order, p/sampling type-value, fixed patch seed1,
device and loaded producer/runtime identity. without_rows invalidates extraction
while preserving selection. Fit edits refit existing rows; extraction misses charge
any caller-retained old row storage. Version guards surround callbacks. Export
transfers only compact centres; final assets revalidate before completion.
The native Train centres dialog exposes effective controls with source p10/n_init10,
uint32 seed and strict(0,1) fractions; all inputs disable during work. Stop and
settings changes discard late success; reject/accept/close drain QRunnable.
QSaveFile saves atomic JSON and the existing picker invalidates only model/predict/
mask descendants. Actual720x460 installed-SegoeUI render inspected at
artifacts/kmeans-training-dialog.png. Runtime profile/selection summaries are
available; complete persisted training provenance is still open.

Focused directory14/image9: **23 PASS in8.434s**, artifacts/kmeans-directory-focused-gate.log,
root955c3a terminal0. Reduction interval9/scorer8/fitter9: **26 PASS in7.584s**,
artifacts/kmeans-reduction-profile-gate.log, root19e26f terminal0.
Native adapter/Qt11 plus existing recipe10: **21 PASS in13.657s**,
artifacts/kmeans-native-training-focused-gate.log, root83ec36 terminal0.
Expanded related directory/native workflow/image/export/fit/tolerance/scoring/
convergence/control/RNG/initializer/update/full-sampled-data/model/predict/mask/
register gate: **161 PASS in73.954s**, artifacts/kmeans-directory-native-reviewed-gate.log,
rootdef351 terminal0 (initial98c381/session82929 now closed).
Final register reconciliation: **6 PASS in0.435s**,
artifacts/kmeans-directory-native-register-final-gate.log, root333ef0 terminal0.
Scoped git diff --check passes and all14 source/test hashes still match review.
The reviewer inspected source/tests and root evidence; no independent execution
or fresh full-suite claim. Full discovery remains required before actual hand-off.

| Current file | Reviewed SHA-256 |
| --- | --- |
| seedvision/benchmarking/kmeans_training_directory.py | 4440BC519CC5543DDC417D488088029489C06020994F8BC9A58684E4CB4AC894 |
| tests/test_kmeans_training_directory.py | 8D32ECB92B5E861D1E2DF1B0A1C358CEEF3BC66C51F01E18077A8BC1E6D28746 |
| seedvision/cuda/plantcv_kmeans_training.py | 4F30FFA99E99A10AB340B4DA1690ACDCB32B3FE47607CD6F240F69FB0C89308E |
| tests/test_plantcv_kmeans_training.py | A38717006BC3E3A098334EC448CA2BFB3608D5CB8962077A3F60A2E908655700 |
| seedvision/cuda/plantcv_kmeans_intervals.py | 97C3238A40C89C13E2880CC4B2ADA448FC57692501A46C4D40210DE9D1DA2079 |
| tests/test_plantcv_kmeans_intervals.py | BE785CCE1B56BD10C34F4F98C4E93269A1C48DAF4A5E0CD6703E2DEA4B7639C4 |
| seedvision/cuda/plantcv_kmeans_training_score.py | FBB5D66D6E48DC2138BD1A3E6BF7EDFA26073F55FF615DFF91B39897A328C4F8 |
| tests/test_plantcv_kmeans_training_score.py | 25ECAD482FF6D7FE2E727FA06F16957F6933A95975C0AF31ABC9B17ED995A7DE |
| seedvision/cuda/plantcv_kmeans_fit.py | 2F091B689D5BD328F8DC99BEB62E7E91EC052AF6127BBC1C9B1F89D04D0D1523 |
| tests/test_plantcv_kmeans_fit.py | 46D9DC892E8325BD7AA22F8FF0BDA4C54611C7C7A8550ABB9FF8BCE356D17C83 |
| seedvision/benchmarking/native_kmeans_training.py | 9526D0F212201B67A2BB4E27D37EED4CBF4B6D7B88A890594B3524B404A8FCF8 |
| seedvision/ui/native_kmeans_training_dialog.py | A6651E13A1F6461EC1EC230EDD6369F32FF3ABC6FEBACBCF59E7353C6F8E1FE7 |
| seedvision/ui/native_kmeans_dialog.py | BB96728B5FCDC1D2F95F758E1E67552F882CB4E7DB8C255ED3CA37B51C142D04 |
| tests/test_native_kmeans_training.py | 962B843A43CF08AE72550C1054E727F18AE859B64E4FB04E0A0625A2C8D33228 |

Next: production model conversion/import and complete persisted training provenance,
then remaining project/batch/model/report/checkpoint bindings and all unfinished
F01–F22/WP0–WP6 work. Bayes training, other sigma/bit-depth/image and numerical/
drawing/threshold domains, automatic card/helper/SMPTS execution and laboratory
gates remain open. Keep images fixtures committed and generated artifacts ignored.
Do not read/touch/stage analysis.seedfiddle-project.json. No commit/push/publication.

### Ordered CUDA image training and compact fitted-model export (2026-10-08)

Historical checkpoint: its pending directory/native workflow notes and changed
file hashes are superseded by the reviewed checkpoint above.

Previous goal turn made progress by strengthening/reviewing the complete fitter
and reconciling planning/register files. This turn implements ordered-image and
numeric-export composition. The same two workers supplied candidates; root fixed
source/API/resource seams and applied production/tests. The same reviewer approved
the six frozen files below with no actionable findings.

Metadata-only preflight_kmeans_fit_shape shares every prior calculation and checks
integer geometry before allocation. Ordered assembly accepts same-device/channel
CUDA uint8 gray/BGR, preserves occurrence order and resets sampled patch seed1 per
image. Standalone assembly exposes reusable CUDA rows; the fitting convenience
defaults to source n_init10. Combined24e9 work/128MiB data/512MiB live preflight
includes retained images, combined rows and one active extractor, then all fit
phases, before Gaussian/RNG/output allocation. All-image version guards run around
callbacks and nested stages. Undecidable numeric decisions publish no result.

Export transfers only ordered compact float64 centres. It bounds CPU tensor/row/
table/encoder/parser phases, validates the existing model, encodes incrementally,
releases its initial CPU table and reparses canonical bytes for real raw SHA/size.
Early-stopped fits, fractional/signed-zero values, every callback mutation/cancel,
and late8MiB failure pass. Actual256x2883 tracemalloc plus CPU f64 stays below the
explicit128MiB envelope. Counts/labels/row losses/features remain CUDA.

Fitter7/image7/export7: **21 PASS in14.215s**,
`artifacts/kmeans-image-training-export-gate.log`, chunk36e429 terminal0.
Strengthened image interoperability: **8 PASS in4.932s**,
`artifacts/kmeans-image-training-interoperability-gate.log`, chunk14280f terminal0.
Actual pinned public train_kmeans defaults and public patch+fit calls verify
gray/BGR odd/even/full/sampled/per-image-reset rows and fitted outputs. Exported
numeric models match public joblib predict_kmeans/mask_kmeans pixels. Multi-image
p31 customstream peak matches the whole-job envelope.
Final expanded image-training/export/fit/tolerance/scorer/interval/control/RNG/
initializer/update/full-sampled-data/model/predict/mask/register gate:
**121 PASS in54.982s**, `artifacts/kmeans-image-training-reviewed-gate.log`,
chunk792d4e terminal0 (initial chunkd6ea69/session25977 now closed).

| Current file | Reviewed SHA-256 |
| --- | --- |
| seedvision/cuda/plantcv_kmeans_fit.py | 5D924E1EAA1CF547C4E6B38B188B53100B941E8694E4BBD1BAB2268FB09A4F4D |
| tests/test_plantcv_kmeans_fit.py | B93330EB9D8CF6ED7FEC63D2A4C534EA13FED4041AB48F994F6F38C58F6FCDDB |
| seedvision/cuda/plantcv_kmeans_training.py | 208F9576D6ADEFA74AD2EC0C17D6960BBD5202360B3B0D7C94FB2EE8FEEF1DC8 |
| tests/test_plantcv_kmeans_training.py | 7DD32EA5CCC84ADFC9208363C6FA60FC287177068F08142AE1F48EC1871667C4 |
| seedvision/cuda/plantcv_kmeans_export.py | 3CAFC45BD2FD1584CB8A78D58124BB3BB37BCEAAAEF7212260AF1C0CDCA5DEE7 |
| tests/test_plantcv_kmeans_export.py | ED6F28453986AE567E5474BECC558A898ECF23542F788FC759DA8CFC1253C4B0 |

Next: directory selection/materialization and native training recipe/UI. The
colour worker's directory candidate is NOT applied: source imread skips directory/
empty invalid entries, skipped decodes need identity/byte accounting, and decoded
geometry must be bounded before allocation. Preserve sorted selection, independent
Python random.choices, duplicates, prefix-after-selection and idx0 accumulator
failure. Other sigma/bit-depth, production joblib conversion, full project/batch/
model/report/checkpoint and laboratory gates stay open. All F01–F22/WP0–WP6 remain
in scope; no fresh full-suite pass or project completion claim.

### Complete bounded byte-feature MiniBatch fitting (2026-10-08)

Historical checkpoint: its image-assembly/export next steps and fit file hashes
are superseded by the reviewed checkpoint above.

The same reusable colour/size workers supplied source-order and tolerance designs;
root applied the fitter/producers and the same independent reviewer approved the
frozen bundle and subsequent strengthening delta. This completes the declared
byte-feature k-means++ unit-weight estimator path. Image/directory ingestion,
production conversion/export, native training workflow and project/batch/scientific
bindings remain open. All other unfinished F01–F22/WP0–WP6 work stays in scope.

The initial fitter and tolerance gates each pass6 tests (2.707s/2.738s). The
strengthened pair passes **12 tests in5.469s**, `artifacts/kmeans-fit-tolerance-strengthened-gate.log`,
chunkc8a71d terminal0. Actual public MiniBatchKMeans.fit traces compare every RNG
draw byte/dtype/post-state, initialization, timer, update/reassignment/count reset,
convergence decision/state, final centre/count bytes, labels/inertia and iterations.
Resolved auto/explicit n_init, clipped batch/init size, strict validation selection,
tol0/nonzero/constant-zero and both compute_labels branches are covered, including
the source all-selected and one-step-EWA failures. Source effective thread count
remains observed provenance, never forced to1. Native/source warning categories
also agree. A4097x243/K4 fixture covers complete customstream peak/cancellation.

Public tolerance uses exact CUDA integer moments plus rational/outward bounds for
the pinned NumPy2.2.6 two-pass variance/mean and centre-shift expressions. Constant
columns prove source variance0; tol0 skips the unused producer/shift work. Dense
features, centres/counts, labels/row distances and intermediates stay CUDA; only
compact moments/counts, RNG metadata and scalar bounds reach CPU. Undecidable
numeric decisions withhold the complete fit. Every callback checks source storage
versions before later work. Whole-job20e9 work,128MiB data/CPU metadata,512MiB live
and component limits are checked before RNG/output allocation.

| Current file | Reviewed SHA-256 |
| --- | --- |
| seedvision/cuda/plantcv_kmeans_fit.py | BB4F377980B1C5F265E3E98BD2B5D690A7A67BF9A3FDE89C39FA854E4AA5F20D |
| tests/test_plantcv_kmeans_fit.py | F8333E7872535F6FFB9EAF0CC71D9077EBEA3A33D7925D14CEF6F7527E9472E1 |
| seedvision/cuda/plantcv_kmeans_tolerance.py | 9C787316DF711A937C646263FC8F07319947377A4E0A984DB8E34D5F09ADB8D5 |
| tests/test_plantcv_kmeans_tolerance.py | 660827024477F12E15F5D7191CDA355F837FC96FFF3FC2433199FDF6992B42CC |

Final assembled fitter/tolerance/scorer/interval/control/RNG/initializer/update/
register gate: **67 PASS in15.344s**, `artifacts/kmeans-fit-reviewed-gate.log`,
completion chunkcf16c9 terminal0 (initial chunk79b9b9/session4436 now closed).

Next: image/directory training assembly, compact numeric-model export and native
training bindings. Actual pinned train_kmeans (root chunkc581b9) sorts os.listdir,
uses independent Python random.choices before prefix filtering, reads unchanged
OpenCV BGR/gray bytes without colour conversion and resets patch sampling seed1
for each image; its estimator seed is separate. Preserve the idx0 accumulator
failure if the first selected occurrence is skipped/unreadable. Defaults are
patch_size10/sigma5/samplingNone/seed1/num_imgs0/n_init10. Both workers have the
source excerpts and return text designs only. Before allocating combined rows,
the wrapper needs metadata-only fit preflight and combined image/extraction/fit
resource accounting. Numeric inference model schema stays unchanged; only bounded
ordered centres may cross to CPU for model export. No fresh full-suite pass.

### Mutable-centre CUDA scoring and interval convergence (2026-10-08)

The same colour/size workers supplied text candidates and numeric designs. Root
applied the source-grounded implementation and focused tests; the same reviewer
approved the frozen scorer/interval bundle. Dense centre tables, reusable norms/
duplicate flags, score tiles, labels and row losses stay CUDA. Only compact
validity facts and the final serial-fold scalar reach CPU.

- Scoring: **7 PASS in2.329s**, `artifacts/kmeans-training-score-gate.log`,
  chunk3de4fe terminal0. Actual pinned Lloyd/inertia calls compare source labels,
  row losses and inertia with1/2/4 threads. Cases cover F1/2/3/4/5/7/8/9/31,
  byte255 labels, negative scores without clamping, duplicate first-index ties,
  all-identical fractional centres, actual4097-row/max4096-feature/256-centre
  allocation, custom streams, tile repartition and per-version norm recomputation.
  A late fractional bisector withholds the complete result. Every callback's
  source/centre storage mutation is rejected before subsequent allocation/work.
- Intervals: **8 PASS in1.407s**, `artifacts/kmeans-intervals-gate.log`,
  chunkb79789 terminal0. Actual pinned scalar convergence sequences retain every
  state bit. Point operations preserve source scalar order; nonpoints are bounded
  and overlapping improvement/tolerance decisions raise without committing state.
  Fraction endpoints use exact rational intermediates and directed conversion;
  subnormal cases and all small positive permutations/groupings are covered.

Integer byte centre facts prove every row loss and partial aggregate exact below
2**53. Fractional grouped-four/remainder losses feed an actual serial CUDA fold;
the interval helper first bounds its real sum, adds the source/GPU per-row error,
then bounds any valid source reduction. Fractional one-thread values remain
intervals; fixture agreement is not a universal compiler-equality claim. The
requested source thread count is preserved, never silently changed to1.
Global1e9 combined work, dataset128MiB, scratch64MiB and live512MiB limits precede
kernel/output allocation. Input versions are checked around callbacks/raw work.

| Current file | Reviewed SHA-256 |
| --- | --- |
| seedvision/cuda/plantcv_kmeans_training_score.py | 67A9590B274941652A51F315959A733ED193A071CFD7FF45F0B955083BEC8655 |
| tests/test_plantcv_kmeans_training_score.py | F62AFDC1CEFCC962E716439C7BE7130637EEE4BADBB533D7BF5073C66781CB54 |
| seedvision/cuda/plantcv_kmeans_intervals.py | D7BF01FAE8905F3292D1D6C5E32B14732C2B457699B51387482A0112A79C06B6 |
| tests/test_plantcv_kmeans_intervals.py | B435AF54F2A29DFA4887A6E97D55047735092DEFBFE51235E215947124CFF325 |

Final combined scorer/interval/control/existing-classifier/register gate:
**37 PASS in7.315s**, `artifacts/kmeans-training-score-reviewed-gate.log`,
chunk97d7c3 terminal0. This historical scorer checkpoint is now integrated into
the complete bounded fitter recorded above, including parameter resolution,
validation/iteration scheduling, public-tolerance producers and total-job bounds.
Other unfinished F01–F22/WP0–WP6 items remain in scope; no fresh full-suite pass.

### Compact MiniBatch control and reassignment planning (2026-10-08)

The same size worker supplied the candidate, root applied it and added actual
pinned-source tests, and the same independent reviewer approved the bounded
component. The focused gate passes **10 tests in1.685s**,
`artifacts/kmeans-control-gate.log`, chunkf5ee09 terminal0.

The final focused control/RNG/CUDA-update/register gate passes **33 tests
in5.636s**, `artifacts/kmeans-control-reviewed-gate.log`, chunkd0df55 terminal0.
Full-suite verification is reserved for genuine project completion/hand-off;
the historical full run does not cover these additions.

`seedvision/cuda/plantcv_kmeans_control.py` keeps at most256 exact float64 count
values and scalar state on CPU. The pre-update timer preserves zero-count and
10*K triggers. Convergence preserves the ignored first step, binary64 EWA order,
tolerance stop before minimum/counter changes and strict improvement comparison.
Post-update planning preserves the strict low-count threshold, source default
argsort cap and without-replacement draws from the current minibatch. Tested
tied orders agree between runtime NumPy2.5.1 and pinned2.2.6. Empty selection
consumes no RNG but still evaluates the minimum. When all centres are selected,
the tested CUDA update/write sequence and RNG advance precede the source's
min(empty) ValueError. Count digests reject mismatched/mutated compact snapshots.

| Current file | Reviewed SHA-256 |
| --- | --- |
| seedvision/cuda/plantcv_kmeans_control.py | 204B264D73258B74A7F5F654E880C57090E869605B395043A47E8509BB963B3C |
| tests/test_plantcv_kmeans_control.py | 9CA371C46E770D0995E2774CAE32589A6F18CB281C992F3F7629DE53A7A7F1C3 |

This is scalar control with certified inertia inputs, not a complete fitter.
Next: integrate mutable-centre CUDA scoring and row distances, certify the actual
source thread reduction/convergence decisions, then complete subset/validation
selection and caller scheduling. Do not clamp the source Lloyd centre-norm-minus-
dot label scores or silently replace the effective OpenMP thread count with1.
Interval control and CUDA score/inertia are now applied/reviewed in the newer
checkpoint above; full fitter orchestration remains pending.

The Windows shell sandbox failure was reconfirmed as `helper_unknown_error:
apply deny-read ACLs` before process launch. Project edits remain authorized and
`auto_review` remains active. Root's workspace patch editor works; necessary
command escalations receive automatic review. Do not request the same project-
edit permission again or retry failed worker editor/shell paths.

### Sampled patches, estimator RNG and CUDA k-means++ (2026-10-08)

The same worker pool supplied candidates; root corrected source/platform details,
applied all edits and ran focused checks. The same reviewer approved all three
bounded components. None is a claim of complete MiniBatch fitting or family parity.

- Sampled sigma5 training patches: **9 PASS in8.856s**,
  `artifacts/kmeans-sampled-data-gate.log`, session14003 terminal0. Actual isolated
  public `patch_extract` byte checks cover gray/BGR, p1/2/3/4/31, degenerate inputs,
  seeds1/7/uintmax, integer counts below/equal/above N and fractional selection.
  Integer counts clip at N while still sampling with replacement. Fractional
  zero-row requests reproduce the source's error category before native allocation.
  Row indices are drawn completely before column indices. Native NumPy2.5.1 and
  pinned2.2.6 both use legacy C-longint32 draws on Windows despite defaultint64.
  The initial worker candidate assumed defaultint64 and unsupported CUDAuint16
  advanced indexing; root probes corrected both. Same-storage int16 views gather
  without a full raster cast. CPU index/headroom16MiB and global CUDA output/work/
  phased live bounds precede Gaussian allocation. Actual p31 multi-tile peak,
  streams/repartition/cancellation and mutation rejection at every callback pass.
- Compact estimator RNG calls: **7 PASS in2.906s**,
  `artifacts/kmeans-random-schedule-gate.log`, chunk32d05a terminal0. Actual pinned
  public `MiniBatchKMeans.fit` traces replay each validation vector, conditional
  subset, weighted first choice, local uniforms, weighted batch and reassignment
  draw, comparing exact output dtype/bytes and full post-call MT19937 state.
  Validation always draws; init_size=N skips the subset draw entirely. Explicit
  unit probabilities preserve source consumption; `randint` is not a substitute.
  CPU probability/CDF/permutation/output allocations are bounded metadata; dense
  features/centres/norms/distances stay CUDA in the training components.
- CUDA unit-weight byte-valued k-means++: **7 PASS in3.248s**,
  `artifacts/kmeans-plusplus-gate.log`, chunkee138a terminal0. Actual public and
  private initializer calls plus estimator `_init_centroids` agree on selected IDs,
  float64 centre bytes, potentials and post-RNG state. Root replaced proposed
  Torch feature reductions with a bounded ushort-to-int64 squared-distance kernel,
  avoiding hidden full-tile conversions. Input128MiB and N/F limits prove every
  positive distance/prefix/potential integer below2**53. Closest/prefix/output stay
  CUDA; only small candidate IDs/potentials reach CPU. Weighted first choice, left
  search, endpoint clip and first minimum/zero/duplicate behavior are preserved.
  Global limits precede RNG/kernel work. MaxF4096, actual4097-row peak, custom streams,
  source/RNG callback storage mutations and loaded-source constants are covered.

| Current file | Reviewed SHA-256 |
| --- | --- |
| seedvision/cuda/plantcv_kmeans_sampled_data.py | 34F2DF23F3F5A6BD67E27E9F1D72A97468CBD19F5D3DF467689C1172CD1A0B41 |
| tests/test_plantcv_kmeans_sampled_data.py | AD3686BDBB135B236BDF4F791BEC22D527723F64F90EB020753AC6BE58A801E5 |
| seedvision/cuda/plantcv_kmeans_random.py | D02AD169FBD280584A2C6570FE8C5BE65EF27FCED1C3EBD4626A37C76A1146EC |
| tests/test_plantcv_kmeans_random.py | 67800CD6AF5FD48F71204E91DD81B091FCD451B073709EA476062332D2229EC5 |
| seedvision/cuda/plantcv_kmeans_plusplus.py | 3E1FEF1364280FEA507F98CB225954577CD481A78E940211EC613AE8D308B5CC |
| tests/test_plantcv_kmeans_plusplus.py | 74B746AEF9D0A9755AC266E2430202389AA54D2B8DE9ECDA276E35CA04AA13C9 |

Final assembled full/sampled training data, RNG, initialization, MiniBatch update
and reconciled-register gate: **47 PASS in18.869s**,
`artifacts/kmeans-training-assembly-gate.log`, session60828 terminal0. Scoped diff
whitespace checks pass with existing LF-to-CRLF notices only. Generated evidence
remains ignored under artifacts; no commit or publication was made.

Next work is complete fitter integration: subset/validation model selection,
CUDA mutable-centre scoring/inertia, source-ordered reassignments and convergence.
The same colour worker has supplied a scoring/inertia design; scalar controller/
reassignment planning is now applied and reviewed in the newer checkpoint above.
Source `_inertia_dense` uses four-feature groups inside each
row distance and a static OpenMP reduction across rows; full fit chooses effective
OpenMP threads. Do not silently replace that aggregate with thread1 or claim
arbitrary threaded byte equality. Scoring guard/classifier source is published in
root67f9de; pinned Cython bodies in52be32 and fit thread selection infd3b4f.
Every other unfinished F01–F22/WP0–WP6 item remains in scope. Production conversion,
other sigma/bit-depth and directory domains, desktop/project/batch/report/checkpoint
bindings and scientific validation remain open. No fresh full-suite pass is claimed.

### K-means training components (2026-10-08)

Root applied the same two workers' text-only candidates and the same independent
reviewer approved these bounded components. Full native training is still open.

- Unpadded sigma5 full-patch extraction: eight tests pass within the 30-test gate
  below. Actual isolated pinned public `patch_extract` calls cover gray/BGR,
  odd/even patches, degenerate dimensions and the maximum p31 multi-tile matrix.
  Training uses no prediction edge padding. The CUDA uint16 dataset is bounded
  at128MiB; Gaussian/feature phases have global work/live-storage preflight, actual
  peak checks, cancellation/coverage withholding and early input-version capture.
- Supplied-label unit-weight MiniBatch dense update: exact binary64 centre/count
  bytes match actual pinned public `partial_fit` and dense Cython calls. Source-row
  additions stay ordered across CUDA tiles; empty clusters preserve centres/counts,
  including negative zero. Inputs remain CUDA; maximum256x4096 centre tables have
  measured bounded scratch. Initialization, scoring, batch RNG, reassignment and
  convergence are outside this component and remain required for full fitting.
- Combined training-data/update and existing PlantCV/native watershed regressions:
  **30 PASS in45.745s**, `artifacts/kmeans-training-primitives-gate.log`, session7688
  terminal0. This gate preceded the final two review repairs. The repaired update
  gate passes **10 tests in2.468s**, `artifacts/kmeans-minibatch-review-repair-gate.log`,
  chunk52804d terminal0. Review caught empty-cluster signed-zero counts and callback
  storage mutation before raw kernels; both are fixed with regressions. Checked
  polling rejects changed inputs before every subsequent allocation/launch.
- Additive `_Program.launch_1d` validates signed32 work/scalars, uses128 threads per
  block and the current CUDA stream. The old launcher is unchanged. The sklearn
  dense-update BSD attribution is retained in `docs/SKLEARN_KMEANS_NOTICES.md`.
- Final reviewed combined training-data/update, existing PlantCV/native watershed
  and reconciled-register gate: **38 PASS in47.640s**,
  `artifacts/kmeans-training-reviewed-gate.log`, session55566 terminal0. Scoped
  whitespace checks pass with existing LF-to-CRLF notices only.

| Current file | Reviewed SHA-256 |
| --- | --- |
| seedvision/cuda/plantcv_kmeans_training_data.py | B5B3E7543D848D9A65C023872E3B9F561C6FF6672BF69E7C54464F1960D9E60E |
| tests/test_plantcv_kmeans_training_data.py | 3E6E5398A55DED82B2C7F9B1523A20F661ACDF57202D35E5396EBF96733F4425 |
| seedvision/cuda/plantcv_kmeans_minibatch.py | 4DD9DE371021668AE405276DBC8BBDA9F2D1F96FA8392013EF8FC6EF211E5B52 |
| tests/test_plantcv_kmeans_minibatch.py | DDCB666A635244897B214DCBCA5065A1BDACCC65E5498D550837723D8D9BAA98 |
| seedvision/cuda/_nvrtc.py | 647ECEB69E4516EE335CC33AFE7E0A70E35144282B8F124C236F97092BBD2CDB |

Sampled training patches, exact estimator RNG calls and byte-valued k-means++ are
now applied and reviewed in the newer checkpoint above. Root still owns writes/
execution with the same worker pool. Full fitting, other sigma domains, production
model conversion and saved-project/batch/report/checkpoint bindings remain open.
All other unfinished F01–F22/WP0–WP6 items remain in scope. No full-suite or family/
goal completion is claimed at this checkpoint.

### Frozen K-means prediction and native workflow (2026-10-08)

The same reusable workers supplied text-only candidates; root applied all writes
and ran every scoped command. The same independent reviewer approved full frozen
prediction and the additive model/predict/mask recipe/executor/desktop integration.

- Complete mask/preprocess/feature/scoring/model/prediction gate: **37 PASS in
  26.683s**, `artifacts/kmeans-complete-predict-gate.log`, session95186 terminal0.
  Actual isolated pinned public `predict_kmeans` joblib-path calls cover gray/RGB,
  patch1/2/3/10/31, degenerate dimensions, integer first ties and a reference-fitted
  synthetic model. The latter is engineering conformance only; native training is
  not implemented. Fractional ambiguous winners withhold the entire label raster.
- Workflow/API/Qt/Naive Bayes compatibility plus predictor gate: **27 PASS in
  22.282s**, `artifacts/kmeans-workflow-gate.log`, session98052 terminal0. A saved
  recipe binds a bounded numeric-centres JSON file by canonical path, raw bytes,
  numeric table digest, geometry and preprocessing; rereads precede preparation,
  direct execution and cache reuse. Gray models use prepared gray; three-channel
  models reverse RGB to source BGR on CUDA. Explicit categories produce the source
  raw union, wrapped as one adapter mask named `selected`; null produces decimal
  IDs in order. Actual parameter edits recompute masks only. Model changes retain
  prepared/image nodes; label mutation repairs prediction and masks. Invalid or
  missing assets, async cancellation and precision failures clear stale previews.
- Shared runner/dialog/seed/extended-threshold/reference/register gate: **68 PASS
  in174.011s**, `artifacts/kmeans-shared-workflow-gate.log`, session66306 terminal0.
- Follow-on canonical model export: **24 PASS in13.257s**,
  `artifacts/kmeans-model-export-gate.log`, session12857 terminal0. The same reviewer
  approved `encode_kmeans_model`: validate first, incrementally cap canonical UTF-8
  at8MiB, preserve ordered binary64/negative-zero values and numeric table digest.
  Reparsing canonical bytes acquires their own raw hash/length. Three added tests
  cover round-trip/raw identity, forged-before-encoder rejection and an actual
  738,048-value over-cap table. No production file writer or joblib converter is
  implied. This gate reruns model8, predictor6 and workflow10 at the current hashes.
- Actual1440x1000 classifier and launcher views were inspected after explicitly
  registering Segoe UI for the offscreen renderer. Images:
  `artifacts/native-kmeans-dialog-review.png` and
  `artifacts/native-kmeans-launcher-review.png`. Production font behavior is unchanged.

Features remain tiled CUDAuint16 until byte-equivalent scoring. The predictor
prepares CUDA centre tables/norms once and reuses them across every feature tile.
Whole-image work and live-storage limits precede Gaussian allocation; actual
maximum-patch multi-tile allocation passes. The initial prepared-centre reuse test
used a MagicMock that retained every CUDA tile argument; replacing that spy with a
nonretaining counter repaired the measurement without changing production limits.
The predictor's default arguments now use local literal4096/256 constants so the
existing loaded-source/default recorder can verify them; arithmetic is unchanged.
The initial workflow gate had26 passes and one test expecting `unavailable` for a
ValueError ambiguity. Its corrected assertion checks `failed` plus the explicit
`KMeansDecisionAmbiguous` reason, matching existing executor failure semantics.

| Current file | Reviewed SHA-256 |
| --- | --- |
| seedvision/cuda/plantcv_kmeans_features.py | F03CF9AB4E43F8846B45383E78E6743F84A898E18781C649AB7AAC39F8AF336D |
| seedvision/cuda/plantcv_kmeans_classifier.py | A866B0E16C82802DE121A209972C408A9504CC99D3ECDE24BE4A71B58ECF8934 |
| seedvision/cuda/plantcv_kmeans_model.py | EA54E5A5C2244003DE9B0CE62E1F866E028D0E5361F751739957587F1F6D9E86 |
| seedvision/cuda/plantcv_kmeans_predict.py | E3C6F26501B35A7AF709C01C98594E2FFF071E732C4FCC8C36C056331B8B112A |
| tests/test_plantcv_kmeans_features.py | 689FBAAE821304F68DB53EE075103A568590F16B63B6F6821893356F66BEE6ED |
| tests/test_plantcv_kmeans_classifier.py | 31C289BEB38C6F82356BC5CD9518EEF7EBB6E0697956F94ED3E0E8C68D2F3D59 |
| tests/test_plantcv_kmeans_model.py | 9509D1F5BFB7751D581F0C661789AEF4FA76DB9F6989A3FD76B2299471F418DC |
| tests/test_plantcv_kmeans_predict.py | 7CCC89120BCB52A63AD1F85E45C1613CE51FAC9AF068FA403BA85FAD40A0F59A |
| seedvision/benchmarking/native_kmeans_recipe.py | 2A292DAA56E35C384DC9C9E1BFBF0F47A47423F0E00030DA098A096E49033D4E |
| seedvision/benchmarking/native_recipe.py | E257435D3B6D77EFD1564784E8E40F9DF95347F3ED3872790D482A09616FD833 |
| seedvision/ui/native_naive_bayes_dialog.py | 371B046C860BDA734398EE3C48DBE331E28EA4D0F9EC0EDF545706A8BB65F59E |
| seedvision/ui/native_kmeans_dialog.py | 041EEE0D2F681374BBF56306E2803A2A36274068D3F549331524CC92FBB2E3B6 |
| seedvision/ui/reference_graph_dialog.py | B09E24A1F3304834A734C1A344A35CFC8160CCC040F6A829C2BB3C3553429008 |
| tests/test_native_kmeans_recipe.py | BA4E8CF2DD8AD29732EB2BF7C8E054310F5EBCF1A8D6A55EEDD35FB08520E7FC |

Next bounded work is native training and production model conversion.
The unpadded full-patch producer and unit-weight update are now applied and approved
as recorded in the newer training-components checkpoint above. Sampled extraction
and the estimator RNG schedule are the same workers' next bounded assignments.
The bounded model serializer remains approved. Training adds no prediction padding;
sigma5 full-patch extraction can reuse Gaussian patch_size1 then the feature iterator
at the training patch size. Sampled patches use independent row/column randint
draws with replacement (duplicates allowed), not sampling without replacement;
train_kmeans does not forward its estimator seed to patch_extract, whose default
seed stays1. Directory random.choices uses separate process-global state. Preserve
these source distinctions; do not claim training from the reference-fitted fixture.
Full MiniBatch fitting, other sigma/sampling domains, production models and complete
saved-project/batch/report/checkpoint binding remain open alongside every other
unfinished F01–F22/WP0–WP6 item. No fresh full-suite pass or family/goal completion
is claimed. Keep unrelated project JSON unopened/untouched/unstaged.
The reconciled feature register passes6tests in0.376s and the scoped diff whitespace
check passes with existing LF-to-CRLF notices only. Generated review outputs remain
under ignored artifacts; no commits or publication were made.

### K-means masks and Gaussian preprocessing (2026-10-08)

Root applied the same workers' mask and revised preprocessing work, then the same
independent reviewer approved the bounded deltas. All14 focused tests pass15.139s
in `artifacts/kmeans-preprocess-gate.log`; session78660 is terminal0. Earlier
session11129 is terminal1: two oracle imports treated the exported train function
as a module. The child now imports the actual module through importlib; production
did not need an arithmetic change for that test-harness error.

Mask generation preserves ordered decimal keys/union semantics and CUDA0/255
outputs, with aggregate256MiB output and bounded tiles. Its seven tests exercise
the genuine pinned public call and actual largest output allocation. Preprocessing
implements the public edge padding followed by sigma5/radius20 nearest Gaussian,
spatial axes0then1, byte rescaling, round-even and CUDAuint16. Virtual clamped halos
avoid a full padded source clone and convolution workspaces. Literal coefficient
bits are decoded inside verified functions; uncertain half-byte intervals withhold
the full raster. The64MiB tile scratch bound passes actual multi-tile allocation.
Review caught output-only work accounting; preflight now charges vertical halos
repeated for each tile column, with a tiny-tile rejection regression before allocation.
F05.2 and the seven status/planning documents now distinguish these primitives
from pending full prediction/training. Register validation passes6tests in0.371s
(`artifacts/kmeans-feature-register-gate.log`); scoped `git diff --check` passes
with existing LF-to-CRLF notices only. No files were committed or published.

| Current file | Reviewed SHA-256 |
| --- | --- |
| seedvision/cuda/plantcv_kmeans.py | 78C8D59A0F3B7094C6C58BEFA8BA6E29E339D8135DBEDC2DB5E2C4389B5D84B5 |
| tests/test_plantcv_kmeans.py | B7465BF7AED71FB2FDE1B45B82281618DA76C9B4AD7E98C56B5EA7DE1E540CCB |
| seedvision/cuda/plantcv_kmeans_preprocess.py | D6664A9D0A06AD268818FE64B05331B81B140A8FDB552EB7870CED2486D6F06C |
| tests/test_plantcv_kmeans_preprocess.py | 78595B7B42E5727FE3E72DD39BC5CBCE981D9386DF8FA6662FC509673167BCF7 |

This earlier primitive checkpoint is superseded by the frozen prediction/workflow
checkpoint above. Patch extraction, guarded scoring, numeric model identity and
the separate recipe/desktop workflow are now applied, tested and reviewed. Native
training, production model conversion and full project/batch/report/checkpoint
binding remain in scope. Root owns all edits/execution; reuse the same workers.
No fresh full-suite pass or family completion is claimed.

The permission question was checked against current tool evidence: auto_review
remains active; normal exec still fails at Windows deny-read ACL setup before a
command runs. This causes escalations to enter review but does not revoke project
authorization. Direct workspace apply_patch works. Do not ask again to approve
ordinary project edits or broaden machine permissions as an implementation step.

### Explicit thickness and larger Gaussian kernels (2026-10-08)

This current checkpoint supersedes earlier per-file inventories and open items
for thickness-one drawing and Gaussian11/13/15. Root applied all changes; the same
independent reviewer approved bounded source deltas using root execution evidence.

- Size/raster gate: **48 PASS in90.267s**, log `artifacts/size-thickness-gate.log`.
  Explicit integer width is stored with measurements; renderer mismatch rejects
  before clone. Public values/pixels match at1/5; contour/circle primitive widths
  1/2/3/5/8 match. Source fallback caliper alone stays5; every overlay uses requested
  width. Shared work/caps and cancellation remain intact.
- Extended thresholds: **22 PASS in71.401s**, log
  `artifacts/gaussian-extension-gate.log`. Pinned CV_32F coefficients11/13/15 plus
  conservative two-pass error bounds withhold uncertain byte decisions. Work1e9
  and actual three-tile64MiB scratch bounds pass. Dyadic3/5/7/9 is unchanged.
- Larger-kernel recipe/control/Qt: **14 PASS in90.053s**, log
  `artifacts/gaussian-workflow-extension-gate.log`. Version1.1.0 exposes odd3..15;
  actual15 public masks, cache-only threshold invalidation, unmocked precision
  failure/recovery and real async3-to15 edit/reset/save pass.
- Additive seed working/API/Qt: **24 PASS in70.210s**, log
  `artifacts/size-working-recipe-gate.log`. `build_native_seed_recipe` preserves
  selected original settings/topology/layout, adds effective size thickness and
  distinct `*_working` IDs/NATIVE_COMPATIBLE provenance. The reference launcher
  makes this copy; historical catalogue/attachments remain unchanged. Default5 and
  explicit1 reach measurement/drawing. Real isolated-worker1 comparison verifies
  all17 values and actual RGB pixels on the tested fixture. Async edit/reset/save
  and size-only invalidation pass. Initial23 gate had one old cancellation mock
  missing the new keyword; its updated signature asserts5 and preserves the same
  callback/exception contract. Production code did not need a repair.
- Actual1440x1000 SegoeUI thickness1 view was rendered and inspected:
  `artifacts/native-size-thickness-one-dialog-review.png`. Control, returned facts
  and actual thin diagnostic are readable. Generated files remain ignored.
- Register reconciled for F05.1/F15.1/F20.1/F22.1/F22.2; **6 PASS in0.358s**.

| Current file | Reviewed SHA-256 |
| --- | --- |
| seedvision/cuda/plantcv_size.py | 01097E1E76F6815A352A31B2FB46EBB1B026F27EA014CD23CB2E14FBE2155127 |
| seedvision/cuda/plantcv_raster_spans.py | 54E7016785E0DAB5E0C419F2193A02D9C1D903469EEF4C1B2B369AA8A404830F |
| tests/test_plantcv_size.py | 37E342369564B611B7D4FE5DDFFC761C5C62605BFAB08A5A8712D0FDC4D6BAA9 |
| tests/test_plantcv_size_diagnostic.py | FF1362A6C157F8B0E63A280A3E3C3298BC029F52FA6AD398F722D503DCBD58EA |
| seedvision/cuda/plantcv_extended_thresholds.py | F952B3F694B200EAE5F820B60BE175C50D70E91B6EA29E1815282B27A9C4CAB9 |
| tests/test_plantcv_extended_thresholds.py | F0DF5CCDC58B758485D5209225570EFFD34594B1AA701E6B3DD6B9B63F29B1BA |
| seedvision/benchmarking/native_extended_threshold_recipes.py | FDD4FEF8552CF0EFF2F46BA3BDBADE486EC3867134178E24B18732F847D7E339 |
| tests/test_native_extended_threshold_recipe.py | 90B460110C7BCD140FF8687FC48DC66F289C0BB362E3BDC8B8C673D0DB8D954E |
| tests/test_native_extended_threshold_controls.py | 4A0A23B7E9D738408BBE2F865D4FE15867895C10CF10FE421E11A6032E9E45BB |
| seedvision/benchmarking/native_recipe.py | 8B717D753D65FF543F40D68C74389799ABD39B02242292ACC39C3FC560DA673B |
| seedvision/ui/reference_graph_dialog.py | FF7D9D02631D86CAB625D9A83C9D7D040B388F35BB7699011ACD7B089D34A8E0 |
| tests/test_native_seed_working_recipe.py | 7907AFDE447CBD5F34E5B615F76E8C3325E3F56345A54184A00989C908DAE955 |
| tests/test_native_recipe.py | 0E0F93C669F57E814C644DCE038D8D5D309B053EFEDB6311ED769294FC97741E |
| tests/test_native_recipe_dialog.py | 245F24F82B9D5A57D3B3DBC62248CF774C3D69C6FC140B7330B2D064253F06B0 |

Sessions67687/68302/99473/85760/28408/99369 are terminal. Do not poll them again.
The full F01–F22/WP0–WP6 goal stays active; no fresh full-suite pass is claimed.
Keep source-only review bounds distinct from complete compatibility/science.

Historical next-step text is superseded by the K-means checkpoint above:
`mask_kmeans` and sigma5 preprocessing are now applied/reviewed. Reference
dependencies remain isolated in a child process; no desktop sys.path insertion.
Full `predict_kmeans` is still open: public prediction performs
sigma5 skimage Gaussian, round(*255) to uint16, odd/even edge-padded patch extraction
and sklearn prediction. Do not label sigma0/raw nearest-centres as that behavior.
The size worker's working-copy suggestion is now applied/reviewed; reuse the same
worker/reviewer for later project/batch/checkpoint integration. Root owns all edits
and execution; no new permission requests or replacement worker pool.

### Nested size and four threshold workflows (2026-10-08)

Nested RETR_TREE tracing/composition is applied and independently reviewed.
Its 21-test gate passes in 41.503s, including all 512 three-by-three masks,
256 seeded random masks, seven nested public-size fixtures, malformed hierarchy,
aggregate resource limits and cancellation. Dense masks remain CUDA; CPU geometry
is bounded compact contour/span metadata. The reviewed pre-ellipse size source is
`55FA4B38DC4547C3F1EB62D5D7B166DDB98C0300CDBC5815F0CB75A4D954DF77`.

Triangle is implemented and reviewed against the pinned source's histogram preamble
and final Otsu call. No-peak and unsupported Otsu decisions are explicit failures;
xstep is not exposed because it does not change the public mask calculation.
All 17 primitive tests pass in 63.676s. The four separate working RecipeSpecs,
CUDA evaluator dispatch and desktop selector are implemented and reviewed. Their
six API and six control/Qt tests pass, including all 13 effective controls, source/
tensor-version identity, cache reuse, asynchronous execution, reset and recipe save.
The historical five-recipe catalogue is unchanged.

The deterministic 4.11 ellipse compatibility path is now repaired and independently
reviewed for the guarded runtime-5.0.0 domain. The initial 51-test gate exposed
adaptive-mean graph_2's angle 180 versus pinned 0. Identical port arithmetic under
4.11 returns 0; main5 SVD/solve changes a near-zero cross coefficient's sign at the
atan2 discontinuity. The repaired path withholds the whole bundle for that ambiguity
and near-isotropy; it does not normalize angles or relax tolerance. Fourteen ordered/
reversed transformed contours match accepted center/axes/angle outputs at 1e-6.
Post-SVD/two-solve cancellation and cap-before-allocation pass. The expanded 49-case
size/native/Qt/register gate has 48 passes; its only failure was a new test expecting
() rather than the existing unavailable contract None. That assertion alone was
corrected and the affected method passes in 17.085s. Fresh full discovery is pending.

Frozen-model CUDA Naive Bayes inference is applied and independently reviewed.
Eight primitive tests pass in25.777s, including actual pinned masks, class order/duplicates,
ties/zeros/underflow/near-winner/extreme finite likelihoods, stream/tiling, bounded
model reads, cancellation and actual largest-class multi-tile allocation. The new
recipe/API/desktop binding is implemented and reviewed; training and K-means remain
open. Native threshold launcher and completed
Gaussian calculation were rendered at1440x1000 using explicitly loaded SegoeUI and
visually inspected; PNGs remain ignored under artifacts/. Root owns writes/tests;
reuse the same text-only worker pool and keep all remaining feature families active.

Reviewed checkpoint before classifier/diagnostic integration:

| File | Reviewed SHA-256 |
| --- | --- |
| `seedvision/cuda/plantcv_size.py` | `848066B470EE07896B63DECD061E166263B22AD0B45F448DFCB2EA9D6088E6B9` |
| `tests/test_plantcv_size.py` | `50BDF05AB6A1882512A14EB3B2A0EECC7FCC22F320324F4B91E10B194BB3D08B` |
| `tests/test_native_recipe.py` | `2F8E9963AEBC0BE76D2F3D947F2DCC0FA5E183BD618BEACA2AEB8E615AC304D1` |
| `seedvision/cuda/plantcv_naive_bayes.py` | `7569F9519B804440971B36751B87D893B3A9E458CFC843B24EC4FE25990087A9` |
| `tests/test_plantcv_naive_bayes.py` | `1FCFB687DDFB810DC1F825BCE6DAC659804230E17D21A1D9A4B3753E013DD2B7` |

Review was source-only using root execution evidence. It covers only the declared
guarded domains; complete ellipse/threshold/classifier families and science remain open.

Full discovery session `5685` is terminal: 1149 tests in1544.843s, 1144 passed,
two failed and three skipped. Log: `artifacts/latest-feature-unittest.log`.
The panel drag test attempted an outward/clamped move without a held-button move
event; it now drags inward with explicit LeftButton state and exact final position.
Wheel zoom now anchors to the wheel event position and ignores zero deltas.
Both targeted tests pass in3.319s and have bounded source-review approval:
`seedvision/ui/image_view.py` SHA256
`EE3378AFE727B91420C32160356543837B59722A46B34C0DFECC300F12E593D3`;
`tests/test_pipeline_ui.py` SHA256
`2219CC3DEAAFCD2DAAD58433A5DDC670FD544384F9E5EDEC7B6FDB7D5963642D`.
Skips concern the missing complete eleven-image pilot batch, opt-in full-image fill
diagnostic and POSIX case-sensitive locator. A new full suite remains required
after ongoing implementation; do not poll closed sessions or claim a full pass.

The Naive Bayes recipe/API/desktop integration is applied, focused-tested and
independently reviewed. Separate image/model stages preserve image caches on model
changes. Bounded file rereads precede preparation and direct/cache execution;
raw/table/class-order identities, nested output mutations and aliased CUDA storage
are checked. The dialog uses the actual async worker, model picker, ordered class
selector, selected-mask preview, reset/save and cancellation. Fifteen tests pass
in15.052s (eleven new classifier tests and four existing generic dialog tests).
Actual1440x1000 SegoeUI classifier and launcher views were inspected; PNGs remain
ignored under artifacts/. Reviewed six-file SHA256 inventory before the subsequent
diagnostic integration (executor/dialog hashes below supersede these two entries):

| File | Reviewed SHA-256 |
| --- | --- |
| `seedvision/benchmarking/native_naive_bayes_recipe.py` | `22F9009EAF797EA29433C46DFB0B6BC54D88E1CB850113B910C00FEC1294C891` |
| `seedvision/benchmarking/native_recipe.py` | `C9B2E03941D679138B194A89EA917B43BB1D27A2655163728A3A61A172E0FC11` |
| `seedvision/ui/native_naive_bayes_dialog.py` | `E4B8F5F71DCC3D9AD312F0E5FFDF51B1D57D4D8FCD6F9FE376DB150C94BC2A19` |
| `seedvision/ui/native_recipe_dialog.py` | `A14A32D6E39B8AA39FAC923A3D62DC9AC51BFC7D01A6AE6D242384268EDC5EC4` |
| `seedvision/ui/reference_graph_dialog.py` | `4019022E06CDEE6E03641AED3E1F1C92729DFF2E9012D9989B4EE9A4362BE8BD` |
| `tests/test_native_naive_bayes_recipe.py` | `D5D2E1659208C6121D0FA50A63018F96373604928BBDEE3D9070A0F851FC9724` |

The nonverbose thickness-five CUDA size diagnostic is now applied, integrated and
independently reviewed. Fixed16.16 thick-line strokes, closed contours, literal
OpenCV4.11 circle sine samples, source paint order and longest-path endpoints match
genuine pinned drawing/public-size pixels. The numerical/raster/diagnostic gate
passes43 tests in72.062s; repaired recipe/Qt/classifier/register gate passes36
in58.635s. The earlier broader64-test workflow gate passed145.027s before this
renderer addition. Current logs: artifacts/size-observations-diagnostic-gate.log
and artifacts/size-diagnostic-integration-gate.log. No live test sessions remain.

The initial integration image oracle retained default text labels; explicitly
setting verboseFalse/thickness5/pixel units fixed its exact-byte comparison.
Independent review found retained geometry could grow across all128 labels before
the final preflight. Collection now bounds contour points, boundary records and
contour counts incrementally, clears retained geometry on crossing and continues
accurate scalar observations. A real three-label test proves the second crossing
stops later collection while preserving all three dictionaries. Its initial setup
crossed the first boundary cap; the corrected setup fits each label individually
and now passes unchanged two-preflight/no-renderer/all-observation assertions.
Dense and exact1M-record/1,000,001-pixel turnover allocator tests stay within96MiB
scratch. The actual completed1440x1000 size view was rendered with SegoeUI and
inspected: artifacts/native-size-diagnostic-dialog-review.png (ignored).

| Current file | Reviewed SHA-256 |
| --- | --- |
| seedvision/cuda/plantcv_raster_spans.py | FBC33EAF3911C3697371650E6427108419BCC7F1384A0E28D317ED289E05D781 |
| seedvision/cuda/plantcv_size.py | 4250E88E217B89084B4B8BC68FAB611629E852094B28F61ACCB23D8A20AC6596 |
| tests/test_plantcv_size_diagnostic.py | 9CEB801F0974C241513AC163A4D87B4FFBC5127C6527644737D76F7ACCFDA5DC |
| seedvision/benchmarking/native_recipe.py | 25C168B5B99ED95EA39BF85074424804ECF58F5682D7954F0BD5D8C538CC0122 |
| seedvision/ui/native_recipe_dialog.py | 5F4AE0A0D577C5EDB6190FCDE250C3B122E4ECD1D5ED336FB7FBF8BCC7F48C40 |
| tests/test_native_recipe.py | 22198F07C51B49353E9D7718F298EAA11058E1D27440D9988F458F6BD915D7AC |
| tests/test_native_recipe_dialog.py | C3677380F15F9E941850AE482AA28DD75D58D2A695EE7E46C3B9B6C346ED3E40 |

After register/planning reconciliation, all six register checks pass in0.340s.
Scoped git diff --check passes with existing LF-to-CRLF notices. No files were
committed or published; generated renderings/logs remain ignored.

Approval is source-only and bounded; root owns execution evidence. The isolated
PlantCV worker currently uses thickness1 globally; this default5 renderer does
not yet establish equivalence to that separate protocol. Next size-worker text
candidate addresses thickness1 Circle/LINE_8 and the longest-axis fallback's
hard-coded5 distinction. Do not blindly apply its snippet: normalize_spans uses
keyword-only frame arguments, and every new path needs shared work/caps.
The colour worker is auditing larger Gaussian kernels using pinned primary source.
Reuse both workers and the same reviewer; no replacement pool or editor/shell retries.
Root continues full F01–F22/WP0–WP6. Training, complete project/batch reports and
model/checkpoint binding, other numerical/drawing/threshold domains, SMPTS execution
and scientific validation remain open. Fresh full unittest is required before
genuine hand-off; no current full pass or goal completion is claimed.

### Earlier reviewed disconnected size and bounded texture (2026-10-08)

The current focused coverage is **77 distinct passing tests across bounded gates**.
The combined size/raster/native recipe/Qt/correction/register/threshold command ran
69 actual tests successfully and two nonexistent module names produced loader errors.
The corrected correction-recipe/hue-chart modules pass seven tests in 18.304s.
After the texture repair, its expanded twelve-test gate passes in 61.008s; eleven
replace the earlier threshold tests and one adds actual CUDA allocator coverage.
No current full-suite result is claimed. The earlier 989-test result remains historical.

Size now supports multiple external contours with pinned reverse-discovery sibling
order, exact hierarchy, return-order composition and all 17 observations/metadata.
Two/three disconnected objects and frame contact match genuine isolated public calls.
Component, aggregate point/step and work limits are shared across the whole trace;
cancellation after the first contour publishes no partial result. Exact point caps
exclude the temporary repeated-start closure point. Hole/nested topology remains
unavailable. Main OpenCV 5.0 perturbs ill-conditioned ellipse points with RNG, while
pinned 4.11 uses deterministic offsets. The compact source-derived SVD guard detects
that branch and withholds the entire bundle. Sparse self-touch contour parity remains
supported. The adaptive-mean recipe's first unavailable label can therefore report
its ill-conditioned ellipse before the later hole; both outcomes withhold all size
observations. The unchanged binary/Otsu observations and live node controls pass.

Texture supports six documented GLCM properties, odd kernels 3/5/7/9, positive
integer offsets below the kernel, byte cutoffs and five SciPy borders. It returns
only complete binary masks; typed errors withhold uncertain cutoff/endpoint decisions
and unsupported contrast conversion. Internal texture maps, other properties,
even kernels and graph/UI integration remain open. Dense data/scratch stay CUDA.
Review found that a retained boolean equality plane and its reduction conversion
exceeded the intended scratch cap. Sorted keys plus lower/upper multiplicities now
compute exact sum(count squared) without that plane. Actual k9/offset1 three-tile
peak additional allocation is 24,209,920 bytes including an 8,000-byte output,
below the 134,217,728-byte scratch limit. The new allocator regression passes.
Independent review closes the resource finding and approves only the declared
binary-mask domain; no broad conformance or scientific claim is made.

| File | Reviewed SHA-256 |
| --- | --- |
| `seedvision/cuda/plantcv_size.py` | `5BE1E02B7384CD109AA6B60D54199BC10536324E801E232B6A2820FDC5FDD050` |
| `tests/test_plantcv_size.py` | `230A15C567D995FD6048511CA6D71B8C952279FCFEFD50D4362BBC4DB015F848` |
| `seedvision/cuda/plantcv_extended_thresholds.py` | `B8B47D21583069C897282C04ED797F12CB29F8061AF1B1863FD81C6871D2B251` |
| `tests/test_plantcv_extended_thresholds.py` | `F082FE4705EFFBC03183401F536294416B15021D9A81F8746ED7459085AD36E7` |

The size dialog change only describes the new supported domain; reviewed hash
`0C1D94B66B729C178F7696CBBFEDC5E82B8AF1E616409A4E6822BCC8F6B574A9`.
The recipe test's corrected unavailable-reason expectation has hash
`63D2826BD5542862F91B8BC8BAA4FBC49A24A2DBF3978C1F3AEFD7A387282645`.
Source review relied on root's execution evidence and did not run the suite.

Historical next step, now applied above: the size worker's nested RETR_TREE candidate and its
ordering/composition/cap/cancellation tests; keep unsupported outputs withheld until
the genuine pinned gate and independent review pass. The colour worker delivered a
bounded triangle wrapper/test candidate using the pinned source's final Otsu call,
fixed xstep1 and exact no-peak preflight. Both candidates are now applied and reviewed.
Do not expose xstep as a mask control because the source Otsu flag ignores its seed.
Continue all F01–F22/WP0–WP6, including complete desktop/project/batch reports and
checkpoint bindings. Reuse the same workers for text patches/reviews; root owns
all writes/execution. read_thread returns newest turns first: use turns[0] for active
command outputs, and publish chunks below its 20,000-character per-item cap.
Do not retry worker editor/shell paths or request another project-edit approval.

### Earlier reviewed size/correction integration (2026-10-08)

The fresh integrated gate passes **60 tests in 78.699s** across compact raster/size,
native seed recipes/Qt, correction RecipeSpec/Qt, hue charts and the feature register.
After register/planning reconciliation, all six register checks pass in 0.311s.
`git diff --check` passes with the existing LF-to-CRLF notices. Full discovery remains
pending for current source; the older 989-test result must not be reused as this gate.

Size now exposes all 17 actual public records, including `within_frame`'s `in_bounds`,
with exact trait/method/scale/datatype/value/label metadata and pixel units. Empty and
single-external-contour masks follow the source >5-point gate. Binary/Otsu recipe
fixtures match genuine pinned public observations; the adaptive-mean fixture has
an enclosed background hole and explicitly withholds its whole size bundle.
The diagnostic overlay and complex RETR_TREE hierarchy/composition remain open.
Label/count controls affect the returned samples and invalidate only the size node;
cancellation reaches the primitive. The native size facts view was actually rendered
at 1440x1000 with Segoe UI and inspected; ignored screenshot:
`artifacts/native-size-dialog-review.png`.

Independent source-only review approves the six frozen deltas below and closes the
active-edge P2. Search comparisons and suffix shifts are charged before insertion;
cancellation is polled during a single row. The reviewer relied on root's focused
test results, did not execute the suite, and did not broaden the scientific claim.

| File | Reviewed SHA-256 |
| --- | --- |
| `seedvision/cuda/plantcv_raster_spans.py` | `C5FCE13961A32F6C3AEE4150CF4E0F6ADB5443130DA4914BCA18F1E029565524` |
| `tests/test_plantcv_raster_spans.py` | `5BB2573F89AA55E3BE53F9E0106F2758AF298BD6BA4CDAEFB9EE1B7B93698831` |
| `seedvision/benchmarking/native_recipe.py` | `2813BABAD9E5F90F78695E54EBD21882E1D22EAD55FEB0036A7CF9E93AC627AD` |
| `tests/test_native_recipe.py` | `ACC4D474AF82F2C4656969814BB814ED306B975D00CAFCE408676F69D9922855` |
| `seedvision/ui/native_recipe_dialog.py` | `F906196E64B7DE2A5442498F0F3FABFE9BB885C8692087EBC35ADCDC97F42800` |
| `tests/test_native_recipe_dialog.py` | `69B7B2621F44A361A3ABA01054A034A2985A8B01AA5995E407B00A1FF9450188` |

The four-file correction dialog and canonical four-port API are implemented, focused
tested and reviewed. The API preserves all four public outputs, with corrected pixels
within one byte/channel on tested inputs and immutable compact matrices. CPU file
decode/preflight precedes CUDA allocation; exact hashes, device and tensor versions
govern one capped preparation slot. Actual 1440x1000 rendering was inspected in
`artifacts/native-correction-dialog-review.png`. Automatic card detection/helper
integration and complete project/batch report/checkpoint binding remain open.

The unchanged Gaussian/dual-channel candidate has fresh five-test confirmation in
21.587s and bounded independent source/test approval: module
`B82A594618EAA5E05F2FFFFD0966CD9903FE63F7114C313621910D6EBFD5ADBB`, tests
`260D7157EF7317E681D6E405003189E4A12CE25FAF2A708BD9E3D1FD7BAF3FB6`.
Approval covers declared kernels/channels/finite line domains; texture and graph/UI
integration remain outside it. The colour worker's next patch will need fresh tests
and review without reusing this source hash's approval for changed code.

Continue the complete F01–F22/WP0–WP6 goal with the same reusable workers. The size
worker is preparing a concrete bounded multiple-external-contours/no-hole patch;
the colour worker is preparing the next exact texture-threshold candidate. Root owns
all writes, execution and integration. Workers must deliver plain-text patches or
source reviews only: their shell and editor paths have stalled despite authorization.
Do not spawn a replacement pool or ask again to edit project files. Keep holes and
other unsupported topology unavailable until exact sequence/hierarchy/public-value
tests pass. Relevant tests run during implementation; full `unittest discover -s tests -v`
is required before actual hand-off. Laboratory and original SMPTS blockers remain
separate from implementation; preserve the unrelated project JSON untouched.

### Earlier native seed-recipe calculation checkpoints (2026-10-08)

The F01–F22/WP0–WP6 goal remains active. `benchmarking/native_recipe.py` now
executes the three canonical 14-stage PlantCV seed graphs using their typed
dependencies, selected parameters and the existing `PipelineAnalysisCache`.
Actual CUDA products cover median, binary/mean/supported-Otsu threshold, cleaning,
distance, ordered peaks, connected markers, negation, watershed, count and colour.
Size observations and their diagnostic overlay remain explicitly unavailable;
these runs report partial, not complete peer functionality. Colour now returns
the actual compact hue-chart data with native Qt rendering as described below.

**Run native calculation…** in the reference graph dialog opens a separate native
working view using the existing canvas/inspector. Parameters edit/reset actual
calculations; selected recipes save as UTF-8 JSON. Historical attachments remain
exact. Image decoding is recorded as the actual OpenCV decoded frame, not a
calibrated or reference-verified decoder. Dense preparation, analysis and cached
products remain CUDA-resident. Only bounded previews and compact metadata transfer.
Stop/close drain the worker; source/tensor changes reject stale cache products.
Layout-only edits reuse all nodes; analytical edits recompute only dependents.

The focused integration gate passes **44 tests in 66.254s**, including watershed,
Qt execution/cancellation/persistence, historical graph preservation and register
checks. A subsequent expanded recipe gate passes **7 tests in 22.517s**, comparing
every preprocessing/distance/peak/marker/partition stage and colour observations
with genuine pinned public calls across all three recipe arms. Actual 1440×900
Qt rendering was inspected. Runtime records include actual NVRTC and CUDA driver
versions. The fresh full-fixture watershed measurement is 23.920s and 109,250,560
additional CUDA bytes; the one-element context activation accounts for the changed
allocation measurement. Independent review and a fresh full discovery remain open.
The historical 989-test result below does not cover this changed source state.

The subsequent hue-chart integration gate passes **14 tests in 39.029s** across
native recipes, the real Qt worker/view and the new compact chart model/widget.
Chart sample/value/label rows match the actual pinned public return for all/hsv
and valid empty rgb/lab charts. The public test also pins the chart-construction
classes source and Altair version. Colour nodes now have both declared outputs
and report executed; size keeps the overall run partial. Selecting chart samples
preserves cache objects; editing calculation values clears stale charts. The actual
1440×900 Qt view was inspected with labelled hue axes and two returned samples.
Files: `benchmarking/color_histogram.py`, `ui/color_histogram.py` and their live
native-recipe integration. Independent review/full discovery remain open.

Next: finish the size/card workers' candidates, review these bounded additions,
complete diagnostics, correction graphs, project/batch reports and checkpoint
bindings, then the remaining PlantCV and explicit SMPTS paper-port operations.
Root also added `cuda/plantcv_extended_thresholds.py` and five focused tests:
all pass in 22.086s against pinned public Gaussian/dual-channel calls. Gaussian
supports exact 3/5/7/9-tap kernels, replicated borders and finite offsets +/-255.
Dual-channel covers all 15 named channels, global indices, the source CMYK C/Y
behavior, strict line comparisons, vertical/coincident lines and extra-point
handling. CUDA scratch is tiled at 65,536 pixels; cancellation/preflight are tested.
Larger Gaussian kernels, triangle/texture, classifiers and graph integration remain
open. Notices are retained in `docs/OPENCV_MATH_NOTICES.md`; independent review
and fresh full discovery remain pending.

The selected-recipe Python path is `run_native_recipe_path`; it is not yet the
complete saved-project batch/export workflow. Keep original SMPTS source and
laboratory gates separate. Project edits remain authorized under auto review;
use direct workspace patches and consolidated root execution without redundant
permission requests. Do not touch the unrelated project JSON. The reused reviewer
reported an old source-read command had waited for approval and was aborted without
output. Root interrupted/reused the same three workers to obtain status; no worktree
or replacement pool was created. Workers must send exact shell reads/tests to root
instead of issuing their own escalations. Direct small workspace patches remain
authorized. Size currently has its scaffold only; card files had not yet been
written when worker status was obtained. They are now resuming their bounded patches.

### Current peak/range/size/native-inspection batch (2026-10-07)

The full F01–F22/WP0–WP6 goal remains active. The 989-test frozen result below covers
the preceding checkpoint; current source/tests have changed and need their own full
hand-off gate. Peaks passed five focused tests and bounded independent review,
including exact ordered coordinates and every peak-mask pixel on the full fixture.
Range thresholds previously passed bounded review for raw BGR slots, HSV, CMYK
(including PlantCV's C/Y slot behavior), GRAY/GREY. The new exact LAB candidate passes
all eight range tests in 10.093s. Exhaustive pinned OpenCV comparison of all 16,777,216
byte BGR colours found zero mismatched pixels/channels; peak additional CUDA allocation
was 4,600,832 bytes at 65,536 pixels/tile. This proves byte conversion in that domain,
not complete recipes. Bounded independent LAB source/test review and diagnostic-script/report audit are approved; the reviewer did not rerun the exhaustive diagnostic. Provenance and retained
upstream notices are in `docs/OPENCV_MATH_NOTICES.md`; ignored diagnostic evidence is
`artifacts/lab-byte-domain-verification.json`. Float32 corrected-image quantization
remains a separate preparation contract.

The raw CUDA size/topology and sparse single-contour candidate now passes ten focused
tests in 2.155s. Compact drawing prerequisites pass eight tests in 1.712s; complete
observations and complex hierarchy remain unavailable pending integration and review. PlantCV
observations remain withheld pending exact contour composition and its >5-point gate.
Per-object colour statistics now have a focused-tested CUDA candidate described below,
including the original empty-HSV/ALL failure. Native checkpoint
inspection, its three producers and the register pass 43 relevant tests in 84.416s.
The nonempty-but-false support-declaration gap is repaired: shared exact declarations
reject alterations in all nine operations. Independent GPT-6.1 Sol medium re-review
approved the six frozen files and independently verified helper replacement rejection
in all three producer identities. Full verification remains pending. Saved recipe integration,
LAB review, exact size/watershed functionality, SMPTS paper-port work and scientific gates
remain open. The unrelated project JSON stays unopened/untouched/unstaged.

Root added an opt-in portable checkpoint closure and snapshot candidate:
`benchmarking/checkpoint_bundle.py`, `persistence/snapshot.py` and the **Create
snapshot with reference checkpoints…** Qt action in `ui/results.py`. The archive
preserves exact manifest/NPZ bytes, verifies every dependency and raw copy, limits
individual decoded archives and aggregate count/bytes, and withholds publication on
cancellation/missing/changed dependencies. Restored checkpoints remain in the restored
project folder's `reference-checkpoints/`; bind that store explicitly in the reference
dialog. Project graph metadata continues to report arrays unavailable by itself.
Eight new tests and existing graph-persistence/review-workflow/register coverage pass
**30 tests in 5.640s**, including source growth after preflight. Bounded independent
review approved the four frozen bundle/snapshot/Qt/test files, and root confirmed
all four hashes. The fresh full hand-off gate remains pending.

Root added `cuda/plantcv_correction.py` and six focused tests (7.082s): genuine pinned
public chip-matrix, affine and polynomial comparisons pass, with normalized matrix
tolerances and corrected-image agreement within one byte/channel on tested inputs.
Dense images/masks and tiled image calculations remain CUDA-resident. Only compact
chip/fit matrices use CPU NumPy linear algebra, with the actual version recorded.
Cancellation and invalid-matrix preflight return no partial output. Bounded independent primitive source/test review is approved at matching hashes.
The standard-card/mask primitive is also reviewed. Complete domains, automatic
detection, project/batch integration, fresh full verification and independent colour
validation remain open.

Root added `cuda/plantcv_color.py`: eight focused tests compare genuine pinned public
RGB/HSV/LAB observations, bins/traits/types/scales, circular hue, positive-only S/V,
uint8/int32 labels, binary remapping and explicit missing/empty failures. Together
with correction and the register, **20 tests pass in 32.429s** with reference source
hashes and dependency versions enforced. Dense histograms/reductions stay on CUDA;
only compact counts/observations transfer. Ordinary numeric tolerance is 1e-10 and
circular-standard-deviation tolerance is 1e-5 degrees. Bounded independent primitive review is approved. Degenerate circular-result
coverage, remaining domains, project/batch integration and full verification remain open.
The colour worker now owns standard-card orientations and circle-chip masks.

Root added `cuda/plantcv_watershed.py`, `_plantcv_watershed.cu` and `_nvrtc.py`.
The fixed CUDA queue preserves pinned scikit-image 0.25.2 default four-connectivity,
heap ties/age, initial marker order, neighbor order and label-at-push behavior.
PyTorch's bundled runtime compiler is used in memory; no toolkit installation or
external compiler is required. Dense elevation/markers/mask/queue/labels remain
CUDA-resident, with scalar progress and cancellation between bounded kernel batches.
Five watershed tests and six register tests pass in **30.946s**, including the source
wrapper pin and cancellation during both initialization and flooding. The complete
distance → ordered peaks → markers → partition test matches every stage on the
3120x2080 fixture. Watershed alone took
23.630s and 109,250,048 additional CUDA bytes for 4,777,682 foreground pixels.
The supported domain is finite contiguous float32 elevation/int32 markers/uint8 or
bool masks, default compactness=0/watershed_line=False, capped at 64M pixels and
512 MiB output/queue workspace. Bounded independent wrapper/kernel/NVRTC/test review is approved at matching hashes.
Canonical seed working graphs execute this primitive; archived native stage bindings,
complete project/batch recipes and fresh full verification remain open. Notices: `docs/SCIKIT_IMAGE_WATERSHED_NOTICES.md`.
This internal diagnostic is not original-public-call intermediate replay or science.

The updated user strategy is one reusable pool: two GPT-6 Luna xhigh implementation
workers and one GPT-6.1 Sol medium reviewer, reused through bounded follow-ups.
Root owns integration/docs. Run relevant tests during implementation and full
unittest discovery before hand-off. Sandbox process startup currently fails while
applying deny-read ACLs; scoped escalation has succeeded. The user expects Approve
for me and objected to the burst of requests. Consolidate necessary execution at root,
reuse legitimately matching approvals and do not send separate routine-read requests
from workers. Do not bypass approval boundaries or retry external publishing.
The current session now explicitly reports `approvals_reviewer: auto_review`.
Project edits remain authorized; use the direct workspace patch editor and consolidate
required execution at root. Do not ask again for the same project-edit authorization.

### Reviewed distance/marker/crop checkpoint (2026-10-06)

The complete F01–F22/WP0–WP6 goal remains active. Three Luna xhigh workers implemented
the pinned CUDA labeled-distance transform, native distance/8-connected-marker
checkpoint bridge and per-seed CUDA crops. Root integrated Qt visible-row export,
saved-project batch export, isolated PNG masks, exact result/source/frame identity,
atomic publication, Stop/close draining and CLI partial reporting. The batch is
uncommitted and has bounded independent Sol approval.

Fresh frozen discovery passes **989 tests in 1070.527 seconds: 987 passed and 2 existing skips**, with
`SEEDFIDDLE_RUN_FILL_IMAGE_DIAGNOSTIC=1`. All **220** source/test/script/configuration/
register raw SHA-256 records match before/after. Log and inventories:
`artifacts/distance-crop-discovery-corrected-full-suite.log`,
`artifacts/distance-crop-discovery-corrected-before.json` and
`artifacts/distance-crop-discovery-corrected-after.json`. The first attempt omitted `-s tests`
and had two inspection test-import errors; retain that diagnostic log, but use the
documented `.venv/Scripts/python.exe -m unittest discover -s tests -v` command.

Five distance tests include exact pinned OpenCV equality on 3120x2080 IMG_9632c.JPG.
Nine stage tests cover actual 1080p CUDA/SciPy marker partitions, explicit failure/
cancellation, archive preflight and exact selected recipe/settings provenance.
No original public watershed internal output replay is claimed. Marker peak count
limits bound compact statistics, not a measured GPU scratch envelope. Ten extraction
and twelve artifact tests cover frame/source-shape, bounded crop allocation, pixels,
masks, provenance, revisions, Qt/batch/CLI and failure nonpublication. The actual
960x640 Qt dialog was visually checked with readable Segoe UI.

The register and planning documents preserve all feature/work-package gates. Next
implementation continues pinned peaks (F10.1), multichannel range thresholds (F04.4)
and compatible size definitions (F15), with the same three Luna xhigh / Sol medium
strategy. Read-only preparations are under ignored artifacts. Then complete watershed,
colour/card correction, SMPTS paper-port branches/contact/erosion/recovery and full
saved desktop/batch recipes with caching and portable checkpoint restoration.
Original SMPTS source/license/dependencies remain unavailable. The 6 October public
capsule recheck returned no source and no fresh HTTP status; the 3 October 403 remains
a dated prior observation. Independent laboratory and scientific gates stay open.

The register verification prose was finalized after frozen discovery; no production,
test, script or configuration file changed after the verified run. Focused register
tests and documentation/catalogue checks validate that status-only finalization.
Preserve the unrelated `analysis.seedfiddle-project.json` unopened/untouched/unstaged.
Keep generated artifacts ignored, native Qt, the single launcher and CUDA residency.

### Reviewed preprocessing checkpoint (2026-10-05)

Fresh frozen discovery passes **953 tests in 1047.537 seconds: 951 passed and two
existing skips**, with `SEEDFIDDLE_RUN_FILL_IMAGE_DIAGNOSTIC=1`. All **206**
source/config/test SHA-256 records match before/after. Log:
`artifacts/peer-preprocessing-final-full-suite.log`; inventories:
`artifacts/peer-preprocessing-final-before.json` and
`artifacts/peer-preprocessing-final-after.json`. The earlier 938-test run was
diagnostic during edits and is superseded by this frozen result.

All current candidates have bounded independent Sol approval: median/morphology,
CUDA fill, four native mask-stage bindings, original mask-stage bridge, native
threshold binding, paired threshold reports, expanded original checkpoint inspector,
original preprocessing and native preprocessing. Original-chain layout and partial
final-output findings are repaired and re-reviewed. Native-chain failure prose no
longer determines cancellation; explicit mask status/event state does. Native-chain
focused coverage passes 13 tests, including original/native equality at all five
stages for binary, adaptive mean and accepted restricted Otsu. The temporary Sol
capacity failure was resolved by retry; its final review checked the frozen hashes.
The local batch is uncommitted. No external CLI review or GitHub push was retried.

The user-requested complete implementation goal remains active. The full F01–F22
matrix and register preserve unfinished functionality, conformance and science.
Next bounded work can use the same three-worker Luna xhigh / Sol medium strategy:

- F10.2 native 8-connected marker labeling can reuse `cuda.ops.connected_components`,
  with explicit peak-mask input, marker-label output and pinned SciPy partition/count
  comparison. Do not represent these as archived intermediates from the original
  public watershed call: the runner exposes final labels/observations only.
- F14 native per-seed CUDA crops can reuse `calibration.gpu_corrected_bgr`, selected
  label geometry and `analysis.crop_offset`. Preserve half-open corrected rectangles,
  explicit padding, source-to-crop transform, result/seed identity and separately
  established clipping facts. Preserve missing/partial objects without scientific
  or review promotion. See the previous `reference_preprocessing` worker's read-only
  design notes for procedural per-axis resize rounding and learned-label alignment.
- F09 must match pinned `cv2.distanceTransformWithLabels(mask, DIST_L2, maskSize=0)[0]`,
  not an assumed exact Euclidean transform. Installed OpenCV 4.11.0 documentation
  excludes precise mode for WithLabels; a tiny CPU probe returned approximately
  2.196899 at offset (1,2), versus 2.236068 for plain precise distanceTransform.
  Verify effective pinned behavior before porting; current JFA is not parity evidence.

These next-work inspections changed no source or tests. Keep the frozen current
batch intact until the next owned implementation begins. Continue to preserve the
unrelated project JSON unopened/untouched, generated artifacts ignored, native Qt,
single launcher, CUDA intermediates and node-local caching. Full recipes, remaining
operations, original SMPTS source and independent laboratory validation remain open.

### Full implementation goal and delegated work resumed (2026-10-04)

The user explicitly requested reconciliation of the feature register/project plans,
then complete feature implementation as an active goal with the previous delegated
worker strategy. The register, full F01–F22 status matrix, canonical implementation
sequence, roadmap, README and architecture now reflect the 913-test baseline and
distinguish tested local candidates from reviewed deliveries. Six register tests
and whitespace checks pass. All original scope and scientific gates remain intact.

Three Luna xhigh workers own new modules/tests only: `native_mask_stages` owns
native median/erode/dilate recipe-stage binding; `reference_preprocessing` owns the
original median→threshold→fill→erode→dilate chain; `cuda_small_objects` owns native
CUDA PlantCV-compatible small-object filtering. Root owns integration/docs and full
discovery. Sol medium review follows frozen candidates through authorized delegated
work; prior external CLI transfer and GitHub push restrictions are not retried.
Keep the unrelated project JSON unopened and generated artifacts ignored.

Root integration now verifies original median/fill/erode/dilate checkpoint images
as well as thresholds. The UI maps both image/mask previews and accepts explicit
parameter selections only when all catalogue declarations/source/topology still
match. Actual async store-binding tests and a rendered median preview pass; four
new tests and 18 graph-view tests pass. This extends the previously frozen UI/
inspection candidate; independent Sol review now approves this bounded scope
with 18 focused inspection tests passing. Fresh full discovery with
the worker batch is running. Root owns these inspector/UI files and the new
`tests/test_reference_mask_inspection.py`.

Sol also approved the bounded native mask-stage bridge including its fill extension,
CUDA small-object primitive, median/morphology primitives, native threshold bridge
and narrow paired-threshold comparisons. Original preprocessing review found two
coordinator defects: layout-only changes were rejected, and an incomplete chain
exposed its latest intermediate as `final_checkpoint`. Both are repaired, all nine
focused tests pass, and Sol's re-review plus independent runtime probe approves.
The four-operation native bridge passes six focused CUDA/original comparisons.

Diagnostic full discovery passed 938 tests in 1005.369 seconds (936 passed, two
existing skips), with the fill image diagnostic enabled. Source changed during
that run, so it is not a frozen-batch hand-off gate. A new native preprocessing
coordinator is now in focused testing/review and fresh full discovery is required
after it freezes. Its benchmark-only stage projections retain the selected parent
recipe identity and map to the two existing native bridge identities; mean/Otsu
mask operations use the mask bridge's binary recipe stage IDs. This does not supply
faithful saved desktop execution. No full-parity or scientific gate has closed.

### Resumed local integration (2026-10-04)

The preserved median, original mask-stage and paired threshold-conformance batch
passes 44 focused tests in 144.543 seconds, with actual pinned reference calls and
no skips. Added cooperative cancellation to the CUDA median primitive before
allocation, between tiles and before returning, with a regression test for input
preservation and no partial returned result; all six primitive tests pass.
The eight previously frozen candidate files remain unchanged. No external review
or GitHub transfer was retried. The unrelated project file remains untouched.

Fresh full discovery passed **913 tests in 694.380 seconds: 911 passed and two
existing skips**. Log: `artifacts/resumed-peer-full-suite.log`.
Source/config/test hashes are retained in `artifacts/resumed-peer-before-full.json`.
The candidates remain unreviewed and uncommitted. Full peer parity, frozen runnable
recipes, original SMPTS access and scientific validation remain open.

Next implementation should bind the native median/morphology primitives to exact
recipe stages and chain preprocessing checkpoints, followed by native watershed
and measurement compatibility. Keep complete F01–F22 scope. The current threshold
pair API already archives genuine original/native output comparisons. The median
boundary rejection remains in place; no further investigation is needed for this
batch. Feature-register checks pass six tests and operator guidance describes the
pending checkpoint-store inspection UI.

### Next independent local chunks (2026-10-04)

The previous goal turn made progress: the three frozen candidates passed full
881-test discovery and actual Qt preview inspection. Required Sol reviews and
GitHub push remain pending separate explicit transfer authorizations; neither
transfer has been retried. The eight prior candidate files stay frozen.

Luna xhigh workers now own three independent local chunks: exact CUDA median
filtering (`wp0_feature_audit`: new median module/tests), original median/fill/
erode/dilate checkpoint stages (`reference_runner`: new mask-stage module/tests),
and retained original/native threshold execution comparisons (`checkpoint_contracts`:
new orchestrator/tests plus the existing conformance module/tests). The latter
also has authority for a strict opt-in, identity-bound correspondence between the
three threshold bridges' different border-policy labels. Default comparisons and
all unrelated mask/instance/primitive semantics must stay strict; do not rewrite
checkpoint identities to force a match. These chunks are in development, not
independently approved or delivered. Root owns docs/integration and will require
fresh full discovery after the next source changes freeze.

Public OpenCV 4.11.0/Intel documentation and the local runtime properties clarify
the default IPP Otsu dispatch; evidence is retained in
`artifacts/otsu-default-dispatch-evidence.json`. The generic fallback is not a
substitute for default-reference behavior. No pins, dispatch or full-scope Otsu
acceptance changed. Full F01–F22/WP0–WP6 and scientific gates remain open.
Keep the unrelated project file untouched and all generated artifacts ignored.

### Frozen local candidates awaiting review (2026-10-04)

All three new chunks are frozen but remain unreviewed and uncommitted. CUDA
erosion/dilation passes 11 focused tests and root's 360 diagnostic comparisons;
native threshold stage/checkpoint binding passes 14 focused tests. Explicit
checkpoint-store inspection passes 14 verifier/Qt lifecycle tests and 18 graph-view
tests. The assigned worker repaired root's bounded environment-manifest read and
auto-deleted Qt worker findings. Root's actual Qt preview inspection
passes after removing overlapping nested-layout geometry overrides.

Fresh full discovery passed **881 tests in 588.748 seconds: 879 passed, two existing
skips**. All eight candidate feature-file hashes stayed unchanged. Log:
`artifacts/peer-pending-review-full-suite.log`; before-run hashes:
`artifacts/peer-pending-review-before-full.json`. The earlier 842-test result covers
the independently reviewed committed batch. The new run includes these candidates
but does not replace their required independent review. External Sol source-transfer approval
and GitHub push-destination approval remain separate and pending. Do not retry
either transfer or use an indirect route without a trusted reply. Keep full
F01–F22/WP0–WP6 scope and all scientific gates open. Keep the unrelated project
file unopened, untouched and unstaged.

### Pending source-review transfer approval (2026-10-04)

The next morphology chunk is frozen: 11 focused tests pass in 38.442 seconds,
and root's 180-case diagnostic batch passed 360 original public-operation/CUDA
comparisons without mismatches. Native threshold and checkpoint-inspection workers
continue local implementation. These chunks are not independently approved or
committed. Their full scope and source/resource limitations remain open.

Automatic approval review rejected the new Sol CLI review before launch because
it sends source/project-contract context to the external Codex service without
explicit payload/destination authorization. Root requested that authorization
asynchronously; no reply yet. Do not retry or route that transfer indirectly.
Earlier completed-batch Sol approvals remain valid. Continue authorized local
work; source changes require fresh full discovery. Push destination approval is
separate and still pending. Keep the unrelated project file untouched.

### Active bounded implementation chunks (2026-10-04)

The reviewed manifest, CUDA thresholds, native graph view and original-threshold
bridge are committed as `dd4899e`, `be7a77c`, `171f2a5` and `322e1fc`. The frozen
842-test run and final Sol reviews cover that completed batch. The documentation
congruence pass preserves all F01–F22, separate functionality/conformance/science
axes, the accurate native graph objective, and WP0/WP1/WP2 in-progress exit gates.
All 36 Markdown files' 161 local targets resolve; six register tests, generated
node-catalogue drift and whitespace checks pass. Historical reports retain their
dated states; the current status/roadmap/plan/architecture/operator guide agree.

Luna xhigh workers are now implementing three separate chunks: native threshold
stage/checkpoints (`checkpoint_contracts`), CUDA erosion/dilation primitives
(`wp0_feature_audit`), and explicit verified checkpoint-store/node inspection
(`reference_runner`). They own only their assigned new modules/tests, plus the
reference dialog/view tests for inspection. These chunks are not delivered or
independently reviewed yet. Root owns docs/register and final integration, runs
bounded Sol medium reviews after frozen chunks, commits approved features and
requires a fresh full suite after source changes. Preserve the unrelated untracked
project file. Push destination approval remains pending; do not retry the blocked
GitHub transfer without a trusted reply. The active full implementation goal is
not complete; original SMPTS access and laboratory inputs remain unresolved.

### Peer implementation: original-threshold stage checkpoints (2026-10-04)

`seedvision/benchmarking/reference_stage.py` binds one public PlantCV binary/mean/
Otsu call to an immutable input/output checkpoint and exact recipe stage run record.
It loads the input itself and invokes the concrete isolated runner; prebuilt results
cannot serve as execution evidence. Source/frame/recipe/upstream, source/environment
pins, selected/effective parameters and uint8 0/255 output roles must agree. Metadata
retains observations/runtime; scientific eligibility is false. Unavailable,
unsupported, failure/cancellation and valid empty states remain explicit.

Luna implemented and repaired the feature; final Sol review approves it. Thirteen
focused tests and 84 new probes pass after correcting large integer-offset rounding
and inherited-timeout cap bypass. Earlier malformed-result, actual runner and
cleanup evidence remains valid. This is a narrow stage bridge, not full recipe
execution or checkpoint graph inspection. Those and other operation contracts
remain open; full F01–F22 scope and WP0–WP6 exit gates are unchanged.

Fresh full discovery passed **842 tests in 462.632 seconds: 840 passed, two existing
skips**, including the repairs. All eleven reviewed feature-file hashes remained
unchanged. Log: `artifacts/peer-reviewed-batch-full-suite.log`; frozen hashes:
`artifacts/peer-reviewed-batch-before-full.json`. The threshold and graph-view
features are committed as `be7a77c` and `171f2a5`; the manifest is `dd4899e`.
WP0/WP1/WP2 are in progress; no exit, peer-performance or scientific gate is met.
Push remains pending explicit destination approval after automatic review rejected
the GitHub transfer. Keep the unrelated untracked project file unopened/untouched.

### Peer implementation: native reference graph view (2026-10-04)

**View → Reference graphs…** opens the reviewed `ReferenceGraphDialog` without an
image or runner. It displays all five catalogue graphs in the existing read-only
canvas/inspector and retains exact stored historical/unknown recipe versions.
Typed dependencies, branches, source functions, unselected parameters, provenance
and recorded run states remain explicit. PlantCV workflows are authored source
descriptions; SMPTS remains a paper reconstruction with unresolved choices.
Checkpoint hashes do not supply pixels; binding/restore/inspection remains open.

Luna implemented and repaired the feature; final Sol review approves it. All 24
view/read-only tests pass, including exact 8 MiB aggregate boundary, capacity-safe
selection/layout and modal cleanup. Capacity-limited views show TRANSIENT—NOT SAVED
while saved metadata stays intact; identity/schema errors still propagate. Actual
PlantCV/SMPTS Qt renders were inspected and all 52 titles fit at 100/125/150/200%
scale. Normal native card geometry and actual analytic cache objects are preserved.
The F22 graph-display objective now has this source-inspection delivery, with
checkpoint inspection and runnable frozen recipes still required. README, roadmap,
architecture, operator guide, status, canonical plan and register agree on that
boundary. Fresh full discovery is running after stage repairs; push approval is
separately pending. The unrelated untracked project file stays untouched.

### Peer implementation: CUDA threshold primitives (2026-10-04)

`seedvision/cuda/plantcv_thresholds.py` adds CUDA uint8 binary, adaptive-mean and
restricted-domain Otsu primitives without changing production recipes. Mean uses
replicated borders and pinned float32 rounding for odd kernels 3–255 and finite
offsets [-255,255]. Otsu transfers only its 256-bin int64 histogram, applies the
class-mass guard, and rejects tied/near-tied eligible partitions. The reference
runner retains default IPP. Empty-bin plateaus promise mask equivalence without
cutoff-scalar equality; full Otsu, remaining mean domains, range thresholds and
desktop recipe integration remain open. F04 and the full plan scope are intact.

Luna implemented and repaired the feature; final Sol review approves its explicit
domain. Ten focused tests pass. Root's 3,240 pinned histogram probes produced
3,132 matching accepted masks and 108 explicit rejections. Sol added border,
noncontiguous, polarity/offset, float32 half-rounding and class-mass probes. Full
2080-by-3120 k=255 fixture masks match both polarities. Sol observed about 200 MiB
extra CUDA memory at that size and 793 MiB at 4160-by-6240; this is not a hardware
envelope or scientific study. WP0/WP1/WP2 remain in progress; exits are not met.

The grouped manifest is committed as `dd4899e` and independently approved:
15 tests, 160 new Sol probes and root's 640 randomized closure probes pass. Its
metadata-only eligibility audit remains `not_certified`. The desktop reference
dialog and repaired original-threshold stage/checkpoint bridge are also approved
and frozen for separate feature commits. Graph-view/read-only tests pass 24;
the bridge passes 13 tests and 84 new review probes after repairing integer-offset
rounding and inherited-timeout validation. Checkpoint inspection and full recipes
remain open; the F22 accurate native graph objective remains required.

The 839-test full suite passed in 478.789 seconds (837 passed, two existing skips),
but predates the stage repairs. Fresh full discovery is running against the eleven
frozen reviewed feature files. Log: `artifacts/peer-reviewed-batch-full-suite.log`.
Reviewer authorization is resolved; push destination approval is separately
pending after automatic review rejected the GitHub transfer. Do not retry without
the trusted reply. Keep `analysis.seedfiddle-project.json` unopened and untouched.

### Peer implementation: reproducible source metadata preparation (2026-10-04)

`seedvision/benchmarking/reference_setup.py` rehydrates the exact raw CRLF runner
source audit and 67-package environment map from committed config/lock evidence.
Inspection performs no writes/network; materialization reports absent prerequisites
and fetches only through an explicitly supplied bounded callback. Source paths,
commit URLs, counts/bytes/hashes, retained flyr identity, atomic no-overwrite and
cleanup are checked. Bytearray/memoryview sizes precede copying; total/empty chunks
are bounded. This helper does not create/install an environment or rebuild a wheel.

Luna implemented and repaired the feature; final Sol review approved it. Nine tests
pass locally and in a temporary source-present/metadata-present root without flyr.
No-source fresh checkout passes eight tests with one correct external-source skip
and reports all 15 sources unavailable. Prior 53+50 rejection/cleanup probes passed.
Filesystem-level symlinks and real process races were not tested on this host;
mocked rejection/collision branches passed. Native defaults remain unchanged.
Reference dialog repairs, grouped manifest review and CUDA threshold development
continue. Fresh full discovery follows a frozen batch; push approval is pending.

### Peer implementation: audited reference graph catalogue (2026-10-03)

`seedvision/benchmarking/recipe_catalogue.py` supplies five deterministic immutable
source-inspection graphs from committed pins: three authored PlantCV seed workflows,
a colour-correction contract and nine-stage SMPTS paper reconstruction. Required
per-image choices are unselected, not invented defaults. Watershed internal nodes
describe one audited public call without independent replay/checkpoints; its
marker-based count observation follows flooding. SMPTS raster semantics remain
unknown where the paper does not specify them, and original capsule parity is
unavailable. All F01–F22 and extended seed methods remain in scope.

Luna implemented and repaired the catalogue; final Sol review found no defects.
Twelve focused tests pass. Source projection uses documented canonical committed
JSON, distinct from the runner's raw source-audit hash. Actual Qt view integration,
recipe execution and final scientific evidence remain pending. Persistence is
committed as `5891a97`; MainWindow is released to the dialog worker. Push destination
approval remains pending. Fresh full discovery follows a frozen feature batch.

### Peer implementation: project/snapshot graph metadata (2026-10-03)

Project schema 3 now retains immutable reference recipe attachments, exact per-stage
historical run records, layout/selection and cable state. Versions 1/2 migrate to an
empty attachment collection; v2 cannot carry injected v3 fields. Constructor/save/
load share count/byte bounds and strict project-error handling. MainWindow open,
save/save-as and session clearing retain/reset attachments correctly. Existing
snapshot create/restore carries metadata and preserves source/master tamper checks.
Native analysis-settings schema remains 20. Checkpoint arrays are explicitly
unavailable: historical hashes do not mean artifacts were restored or verified.

Luna implemented and repaired this feature; final Sol review found no material
defects. The 47 persistence/UI tests pass with one existing filesystem skip, and
107 boundary probes plus actual Open/Save/Save As/New Project checks pass. Catalogue
source-fidelity repairs are frozen for final review; the actual reference dialog
and source-preparation helper are developing. Full discovery follows their frozen
batch. Push destination approval remains pending after automatic review rejected
the push. Preserve the unrelated untracked project file.

### Peer implementation: read-only native node editor mode (2026-10-03)

`PipelineCanvas` and `PipelineInspector` now support explicit immutable `read_only`
constructor mode; native editing remains the default. Reference mode blocks topology,
parameter, enabled/reset, run-to and native optimization/procedural actions and shows
fixed values. Native-ID collisions do not leak native foreground/project/overlay
claims. Selection, dragging/layout, zoom, arrange, bundling and routing remain usable.

Luna implemented this feature; final Sol review found no defects after two repairs.
Five focused read-only tests and 21 targeted Qt tests pass. The independent visible
socket-to-empty gesture mutates editable mode and clears its connected cache, while
reference mode preserves graph facts and real `PipelineAnalysisCache.values` object
identities. Actual reference recipe views/render inspection, checkpoint artifacts
and overall F22 acceptance remain pending. MainWindow graph persistence is frozen
for separate review; do not mix that uncommitted plumbing into this feature commit.

### Peer implementation: strict, nonmutating recipe decoders (2026-10-03)

Shared recipe decoders now copy decoded dictionaries before converting child
records, so repeated public decoding preserves caller-owned JSON. Invalid mode,
source-resolution and run-state enum values consistently raise `RecipeContractError`.
Luna repaired this issue found during project persistence review; Sol independently
passed 12 recipe tests, 22 public round-trips, 60 invalid enum cases and 468 nested
JSON enum cases with no material defects. Persistence/UI retention is frozen for
re-review; catalogue has two source-fidelity repairs and actual graph views remain
in development. Full discovery follows the frozen batch; push approval is pending.

### Peer implementation: isolated original PlantCV runner (2026-10-03)

`seedvision/benchmarking/reference_runner.py` and `plantcv_worker.py` execute twelve
audited original PlantCV operations in the explicitly supplied reference Python.
The desktop does not import PlantCV/SciPy/scikit-image. Source/dependency identity,
bounded input inventory/archive/JSON, verified-source compilation instead of cached
bytecode, timeout/cancellation/reaping/cleanup and effective parameter contracts are
checked. RGB/BGR boundary conversions and normalized RGB chip/polynomial matrix
semantics are explicit; automatic observations/calibration state are retained.
Availability IDs are `plantcv_reference` and unavailable `smpts_reference`.

Luna implemented and repaired the runner; final Sol review found no material defects.
Twenty focused tests pass in the actual isolated environment, with no skips. The
last matrix probe agrees with pinned source to 2.69e-15 and all 48 chip means match.
This is an operation runner foundation. Task-specific recipe execution, archived
stage outputs/native conformance, graph checkpoint inspection and final studies
remain pending. Ignored reference source/environment artifacts are prerequisites;
a fresh checkout without them reports unavailable. Full discovery follows the
frozen current F22 batch. Push remains pending destination approval after the
automatic approval-review rejection; completed feature commits remain local.

### Peer implementation: checkpoint conformance reports (2026-10-03)

`seedvision/benchmarking/conformance.py` compares exact-identity checkpoint pairs
for primitive arrays, masks and label-invariant instance partitions. Stage scope
requires the same upstream checkpoint; pipeline scope retains both upstream hashes.
Raster roles and relevant semantics are checked on every raster comparison route.
Integer values/bounds/errors remain exact; float tolerances are reference-relative
and preserve ordinary equality boundaries while handling overflow explicitly.
Bounded reports include common-evaluator metrics and matched IDs; independent
scientific accuracy is always `not_evaluated` by this layer.

Luna implemented and repaired the feature; final Sol review found no material
defects. Thirteen focused tests and 33 final numeric probes pass. Prior independent
review verified 60 common-evaluator/audit comparisons and the shared allocation
guard. There are no PlantCV-native conformance results yet: this is the comparison
API, not completed backend compatibility. Reference-runner repair, F22 catalogue/UI
and graph persistence remain in progress. Fresh full discovery remains required.

### Peer implementation: bounded shared instance assignment (2026-10-03)

The common evaluator and Hungarian solver now preflight the actual dummy-column
cost matrix, optional evaluator weights and scratch against 64 MiB before dense
allocation. In-place negation avoids a second weights-sized temporary. Oversized
components raise `MemoryError`; supported matching retains its cardinality/IoU/tie
semantics. Luna implemented and repaired the feature; Sol final review found no
material defects. Five focused budget tests pass; 25 procedural-fit tests and
existing matching/metric checks pass. Independent final probes preserve exact HEAD
behavior for 400 solver and 150 evaluator cases; a 1600-by-1600 near-cap probe
peaked at 63,380,800 bytes. Conformance still has a tolerance-boundary repair under
way; runner has a colour-matrix metadata repair. F22 catalogue/read-only UI/graph
metadata persistence are being developed. Full discovery follows the frozen batch.

### Peer implementation: versioned recipe graph contracts (2026-10-03)

`seedvision/benchmarking/recipes.py` supplies strict immutable recipe, provenance,
typed stage/port/dependency, parameter, condition and run records. It projects into
the existing `PipelineGraph`, without a second graph engine. Exact graph validation
rejects altered topology, parameters/source facts, stage-run key swaps, inactive
duplicates and stale identities. Layout/display edits preserve analytical hashes;
explicit upstream expectations are checked when supplied. Unknown/unavailable
sources cannot be marked executed, empty or bypassed. Pinned original provenance
requires source and environment pins.

Luna implemented this feature; final Sol review found no material defects after
six repairs. Twelve focused tests and 142 independent probes pass. Audited recipe
catalogues, F22 desktop views/checkpoint inspection and project/snapshot persistence
remain pending. The 738-test full suite passed before the current repair batch;
root must run fresh discovery when chunks are frozen. Commit `a8b5023` is local;
automatic approval review blocked pushing to the configured GitHub destination
because trusted authorization did not name that exact destination. A request for
destination approval is pending. Keep unrelated `analysis.seedfiddle-project.json`
untouched and excluded.

### Peer implementation: exact checkpoint integer ranges (2026-10-03)

Checkpoint raster and image ranges now preserve integer bounds and compare exact
Python scalar extrema. Values above `2**53` cannot pass a rounded float64 bound.
Luna implemented this repair; Sol independently passed all 24 checkpoint tests
and 74 publication/load/round-trip probes, including int64/uint64 extremes,
finite floats and empty arrays. The prior 738-test full run passed (736 passed,
two existing skips) before this repair batch; a fresh full run is still required
after the current runner, recipe and conformance repairs are frozen. This repair
does not establish reference conformance or scientific accuracy.

### Peer implementation: immutable checkpoint feature (2026-10-03)

The WP0 audit feature and F22 objective are committed and pushed as `8373e3f` and
`e86a4b0`. The completed WP1 checkpoint foundation is
`seedvision/benchmarking/checkpoints.py`, exported through its package `__init__`.
Use explicit `RasterRole` declarations for frame-bound rasters; unmarked arrays
are compact auxiliary data. Publication requires explicit benchmark host-transfer
consent and checks the whole inventory plus exact stored-NPZ size before transfer.
Loads use one bounded immutable archive snapshot and immutable array backing;
stage replay requires exact identity, execution eligibility and measurement contract.
Atomic content-address publication handles cancellation and concurrent losers.

Luna implemented this chunk; independent Sol reviews found and repaired aggregate
preflight, archive reopen, raster semantics, immutability, empty arrays, concurrent
cleanup, archive-size preflight and reserved NumPy-name defects. Final Sol review
found no unresolved material defects, all 23 focused tests pass, and 2,381 estimator
plus 350 exact/tight-cap probes pass. Logs remain under ignored `artifacts/`.
No production analysis behavior or application launch point changed.

The runner reviewer performed full discovery during other checkpoint edits: 704
tests, one stale error-message assertion failure and two skips. The current focused
checkpoint suite fixes that assertion; root must run fresh complete discovery when
all current chunks are frozen. Runner repairs and recipe/conformance infrastructure
remain under review/development. F22 actual graphs, reference persistence/UI and
WP2–WP6 remain pending; the active implementation goal is not complete.

### Peer implementation: source, scope and engineering baseline (2026-10-03)

Implementation has started under the user's requested Luna xhigh worker / Sol
medium review workflow, with a separate commit for each completed feature.
The F22 documentation objective is committed as `e86a4b0`.
`docs/plantcv_smpts_features.json` expands all F01–F22 into operation contracts,
source/native/test mappings, gaps and independent functionality/conformance/science
status. `config/peer_reference_sources.json` pins PlantCV 4.11.3 at
`77e3322fd0cd5a9bef18bb99402f20c8f2db87a9`, selected original source hashes and
67 isolated dependency archives. The binary-only requirements lock needs the exact
retained flyr wheel under ignored `artifacts/reference-wheels/`; missing external
artifacts make provisioning unavailable, never authorize an unpinned source build.
The binary-only dry run verified all 67 exact wheel versions and hashes.

`config/peer_baseline.json` freezes the default native graph and committed IMG_9405
source at `e86a4b0`, its unavailable-policy result revision, native Python/dependency/
CUDA/device provenance and raw CRLF versus Git-blob hashes. This fixture has no
reviewed benchmark truth and no Foreground references; the capture is engineering
evidence only. Original SMPTS capsule access returned HTTP 403; the paper and
ambiguities are recorded, and original execution/parity remains unavailable.

Sol review found and corrected a copied source hash, a native mapping, the flyr
source-build gap and missing baseline runtime/checkout identity. Independent Sol
re-review found no remaining material defects; six focused register checks pass.
The latest broad suite ran 684 tests in 308.261 seconds: 682 passed, two existing
skips. Log: `artifacts/peer-infrastructure-full-suite.log`. Later runner and
checkpoint-repair tests were not all present at that discovery and need their
focused checks plus a fresh final suite. WP0/WP1 remain in progress; exact selected
reference recipes and archived outputs, F22 views and WP2–WP6 are still outstanding.
Checkpoint, isolated runner and recipe-contract chunks are under development/review;
their presence in the working tree is not a completed peer-functionality claim.
Keep `analysis.seedfiddle-project.json` untouched and excluded from commits.

### Reference implementation graph objective (2026-10-03)

Added the user's explicit objective that PlantCV/SMPTS reference implementation
graphs be accurately displayed in the native node editor (F22). The implementation
plan section 5.2 defines topology, branches, parameters, intermediate inspection,
provenance, unresolved-source handling, persistence/cache behavior and actual Qt
visual acceptance. WP1–WP3 and the definition of completion carry this requirement;
the roadmap, architecture, comparison, status and index use the F01–F22 scope.
This is pending implementation, not a new graph capability. Documentation baseline
`dfda336` was committed and pushed after the prior entry below. Verification for
this documentation-only follow-up: full discovery ran 670 tests in 347.608 seconds,
668 passed and 2 skipped; log `artifacts/reference-graph-plan-full-suite.log`.
Focused checks passed for all 36 Markdown documents' local links, F01–F22 and
WP0–WP6 inventories, objective/completion linkage, catalogue drift and whitespace.
No implementation changes, commit or push were made for this follow-up.

### Accepted PlantCV–SMPTS peer-functionality direction (2026-10-03)

The user requested revision and incorporation of two external reports from
`SeedFiddle_Codex_Handoff.zip`. Their canonical revised versions are now
`docs/PLANTCV_SMPTS_COMPARISON.md` and `docs/PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md`.
The root roadmap, current status, architecture, operator guide, scientific protocol
and documentation index are aligned with this accepted extension. Existing baseline
engineering remains implemented; WP0–WP6 peer-functionality work remains pending.
Start with a scoped feature audit and reference/replay infrastructure, alongside
laboratory inputs. Preserve SeedFiddle's architecture and scientific release gates.
Original SMPTS code has not been recovered or executed; paper ports must be named
separately. Proposed backend IDs, paths and YAML are not current executable features.

The integration baseline is `094aa7d`; the previous reviewed implementation was
`f244df8`. This is a documentation-only change. The supplied copies of existing
project files were substantively identical and were not used to overwrite the
repository. `analysis.seedfiddle-project.json` is an unrelated untracked user file.
Post-integration congruence pass: inspected the scope/status hierarchy across all
36 project Markdown documents (34 tracked baseline files and two new reports),
checked local links, and reconciled current guides plus learned, shape, library and
optimization records. Dated audits and implementation history retain their original
evidence; promoted brainstorm items explicitly defer to the accepted roadmap.
No broken local Markdown links remain. The generated node catalogue check passes;
no generated catalogue or runtime/schema changes were needed.

Full unittest discovery on 3 October: 670 run in 325.881 seconds, 668 passed and 2
skipped (historical pilot batch and Windows/POSIX filesystem case). Log:
`artifacts/peer-direction-full-suite.log`. Focused documentation link, F01–F21 and
WP0–WP6 inventory/status checks, catalogue drift and final whitespace checks pass.
No comparative algorithm or laboratory experiment was run. No commit/push was
performed.


### Documentation consolidation (2026-09-29)

The canonical present-tense summary is now
[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md), with all documentation
classified in [`docs/README.md`](docs/README.md). `README.md`, `PLAN.md`, the
operator and architecture guides, package-specific READMEs, implementation
records, and dated audits were reconciled with revision `f244df8`. Dated sections
below remain chronological evidence and may describe controls, failures, fixture
availability, or uncommitted state that was accurate only at that time.

Current engineering state: the full image/reference/annotation/analysis/review/
export/snapshot workflow is implemented. Scientific performance remains
unvalidated pending the laboratory inputs in
`docs/SCIENTIFIC_VALIDATION_PROTOCOL.md`. Latest full local verification on 29 September ran
670 tests in 383.737 seconds: 668 passed and 2 skipped.

### Image zoom and hilum picker (2026-09-19)

Image zoom buttons now retain the viewport center even when the pointer is
over the toolbar; mouse-wheel zoom remains anchored to the image pointer.
The hilum picker ends after one completed placement and also cancels whenever
the selected seed ID changes, including the signal-blocked next-empty path.
Regression tests cover toolbar and wheel zoom, one-shot placement, and both
seed-selection paths. Full unittest discovery ran 670 tests: 668 passed and 2 skipped.

### Seed annotation review flow (2026-09-19)

The palette now has Next unannotated, which cycles painted seed IDs still
missing a reviewed condition or explicit reviewed shape. Seed condition has a
bold group title and a divider; No defects and the defect choices disable one
another. Outline, Full length visible, and Pose share one row. A Complete
outline automatically checks and disables Full length visible; changing away
from Complete returns it to an editable, unchecked choice. Exclude from
modelling sits beside Apply + save at the top of the seed editor.

Load matching reference now prefers the current saved applied seed mask over
the bundled machine pre-annotation. The bundled mask is used only when no
applied seed mask exists. Restoring the applied mask remains undoable. Focused
UI/import tests passed; full unittest discovery passed 669 tests with 2 skips.

### Annotation palette cleanup (2026-09-19)

The annotation palette is now owned by the main window and may be dragged beyond
the image view, within the central application area. Seed visibility uses a
Show all/selected dropdown beside Clear seed/Clear all. The automatic-result
draft importer and optional-metadata collapse were removed. Tool instructions
are in button tooltips; annotation pixel counts are in the Project node details.
Condition, seed shape, exclusion, and hilum controls were compacted. Selecting
explicit Outline and Pose marks shape metadata reviewed; the unchecked Exclude
from modelling control writes an exclusion reason only when selected. Hilum uses
Pick location and Clear location, with only the location marker displayed;
direction is still derived internally for saved metadata. See focused UI tests and the full-suite log under ignored
`artifacts/annotation-panel-full-suite.log`.


### Current review remediation (2026-09-18)

Read [the remediation register](docs/CRITICAL_REVIEW_REMEDIATION.md) and
[operator guide](docs/OPERATOR_GUIDE.md) before relying on older chronological
notes below. The current behavior supersedes historical claims about annotation
markers overriding segmentation, raw/resolved foreground features, legacy count
exports, eleven-fixture inventories and scientific validation tiers.

The working tree includes the earlier complete node-optimization implementation
plus the critical-review corrections. Preserve both. Main additions are shared
production settings, safe learning feature/checkpoint/export contracts, global
metrics and provenance audits, bounded dataset/evaluation/history storage,
coordinate-bound references and explicit legacy migration, source-content cache
invalidation, authoritative results/review/export, portable snapshots, responsive
persistence/shutdown, annotation redo and workflow/layout fixes. The single native
`seed_vision.py` entry point and node-local CUDA cache architecture remain intact.

Reference schema is now 6. Unknown legacy coordinate frames are withheld until
reviewed; transforms transport masks and invalidate clipped complete outlines.
Feature recipe v2 is raw foreground colour; ambiguous v1 foreground recipes need
re-export/retraining. Checkpoints use restricted tensor/primitive loading. Missing
foreground is unavailable rather than zero. Physical result measurements and
unvalidated per-object traits are withheld. Counts/coverage never certify science.

The exact missing `images/IMG_9689c.JPG` was recovered from Git and matches the
recorded SHA-256. Synthetic tests remain engineering evidence only. Remaining
scientific inputs are listed in the register and validation protocol: intended
use/tolerances, reviewed grouped real data, physical calibration/acquisition facts,
trait definitions and operator validation. No data were promoted to reviewed truth.

Final full suite: 667 tests in 294.339 seconds, 665 passed and 2 skipped
(unavailable complete historical pilot batch; POSIX case-sensitive filesystem
test on Windows). Focused checks, compilation, catalogue drift and whitespace
checks pass. Tests/logs and screenshots are under ignored `artifacts/critical-*`.
See the remediation register for detailed evidence. No generated output belongs in `images/`.
New Windows CPU/Qt and optional CUDA workflow definitions are local changes;
hosted CI has not run here. No commit or push was requested or performed.


Repository: <https://github.com/efedo/SeedFiddle>

Primary branch: `main`

Launch point: `seed_vision.py`

This is an on-premise PySide6 desktop application for counting, separating,
measuring, and broadly classifying soybean and lupin seeds in calibrated lab
photographs. Images may contain densely touching or overlapping seeds.

Hard constraints:

- No local server, installer, or bundled executable.
- One top-level Python launch file; implementation modules may remain under
  `seedvision/`.
- A virtual environment is optional, but the launcher can create and populate
  `.venv`.
- Full-raster analysis is GPU-first PyTorch CUDA. Keep tensors and cached
  intermediates on the GPU until display or compact metadata requires transfer.

## Latest update — 2026-09-18: reference-driven node optimization

- Implemented the node optimization compliance work. Read
  `docs/NODE_OPTIMIZATION.md` for usage, objective definitions, conditional
  targets, persistence and limitations. The historical review now has an
  implementation-status table for O01–O11.
- New `seedvision/optimization/` registry/search/production evaluator covers
  all 48 cards: 38 computational capabilities, 269 searchable controls before
  input-dependent exclusions, and ten explicit exemptions. Procedural search
  covers all 35 eligible controls; Reference edges covers 19. Physical facts,
  scoring rules, reference policies and annotation geometry remain fixed.
- `seedvision/ui/optimization.py` provides one native planner/worker/review
  path for every node and the Analysis project command. It fits shared settings
  across selected images in dependency order using actual `analyze_path`
  calculations, isolated caches, true-producer invalidation and CUDA reductions.
  Learned nodes tune decoders; model-weight training remains separate.
- Fixed the procedural surrogate-input problem and moving edge targets.
  Inactive semantic/trace/oval inputs are excluded with reasons. Complete,
  shape-reviewed, connected reference filtering is shared with live analysis.
  The obsolete procedural texture fallback control is visibly disabled.
  Gradient-cutoff diagnostics score their own outputs; shadow/highlight
  controls require corresponding reviewed rasters.
- Runs are explicitly in-sample adaptation, not independent validation.
  Diagnostic-target NPZ imports are source/transform-bound; partial labels and
  per-image exhaustive-coverage declarations have distinct scoring behavior.
  No real-image accuracy improvement is claimed.
- Before/after overlays and per-image results precede application. Cancellation,
  source/annotation changes and stale graphs prevent partial installation.
  Atomic JSON journals, timestamped history, guarded rollback and Save As
  companion copying support recovery. Apply/restore marks the project dirty:
  save normally to persist its node parameters.
- Validation: focused, Qt and two-image real CUDA integration tests pass.
  The production integration verifies that applying one shared proposal
  reproduces final per-image scores. Optional-node smoke tests and native dialog
  renders are under ignored `artifacts/optimization-*`; these are correctness
  checks, not accuracy benchmarks. Focused optimizer/UI/CUDA tests: 66 passed
  in 25.281 seconds. Final full discovery: 638 tests in 536.445 seconds,
  635 passed, two skipped, and one pre-existing missing-image fixture failure
  (`images/IMG_9689c.JPG`). Logs are
  `artifacts/optimization-focused-final.log` and
  `artifacts/optimization-full-suite-final-2026-09-18.log`. An additional final
  25-test contract/UI run passed after adding reference-context provenance.
- Existing missing fixture `images/IMG_9689c.JPG` remains unchanged. No generated
  artifacts or image fixtures were added to the source changes. No commit/push
  was requested.

## Previous update — 2026-09-09: node optimization compliance review

- Additional review-only work is recorded in
  `docs/NODE_OPTIMIZATION_COMPLIANCE_REVIEW_2026_09_09.md`. It assesses the
  requirement that every annotation-optimizable node expose fitting and that
  a project command sequentially optimize all eligible nodes. The application
  is not compliant: only Reference edges and Procedural seed separation have
  node-local fit actions, both are partial, and no project coordinator exists.
- The report inventories all 29 active and 19 unused cards, distinguishes
  direct/downstream/conditional supervision from legitimate exemptions, and
  provides 11 prioritized findings plus an implementation and acceptance plan.
- Fresh synthetic probes show four default procedural search parameters are
  ineffective with the worker's omitted semantic/trace inputs. The edge fitter
  can improve its loss with identical predictions by changing the interior
  buffer used to define its evaluation targets. The procedural reference
  texture weight is also bypassed when resolved material input is supplied.
  These are findings, not completed fixes.
- Validation: 43 focused optimizer/UI tests passed in 5.688 seconds. Full
  unittest discovery ran 611 tests in 315.771 seconds: 608 passed, 2 skipped,
  and the same missing `images/IMG_9689c.JPG` fixture caused one failure.
  Actual Qt inspector/menu inventory and counterexample evidence are under
  ignored `artifacts/node-optimization-review-2026-09-09/`; standalone review
  harnesses and test logs also remain under ignored `artifacts/`.
- No application implementation or committed image fixtures were changed;
  no commit or push was requested. Read this addendum alongside the original
  critical application review before implementing optimization changes.

## Previous update — 2026-09-08: comprehensive critical review

- Review-only work is recorded in
  `docs/CRITICAL_APPLICATION_REVIEW_2026_09_08.md`, covering code, actual Qt
  layouts/workflows, scientific merit, persistence, learning/evaluation,
  functionality, and a staged improvement plan. There are 30 prioritized
  items, including 13 P1 items for the affected workflows. No application
  implementation was changed and no commit/push was requested.
- Confirmed reproductions include loose-workspace draft loss on close,
  exported/live foreground feature mismatch, partial-patch boundary
  supervision, interrupted learning-sample replacement, evaluation accepting
  an audit-invalid split, inflated library validation tiers, and stale raw
  source caching. See the report for evidence, scope, and acceptance criteria;
  these are findings to repair, not completed fixes.
- Actual Qt rendering used Fusion/Segoe UI at 1100×700, 1366×768, and
  1920×1080. An isolated real-worker IMG_9670c run yielded zero instances with
  no Foreground references; a rerun using unreviewed bundled interiors as
  material examples only yielded 15 procedural instances versus zero legacy
  proposals. These are diagnostic observations, not independent accuracy
  measurements. User projects and committed image fixtures were untouched.
- Full unittest discovery: 611 run, 608 passed, 2 skipped, 1 existing failure
  for the absent `images/IMG_9689c.JPG` reference-manifest source; 293.406 s.
  Log: ignored `artifacts/critical-review-full-suite-2026-09-08.log`.
  Reproduction scripts, JSON observations, isolated persistence, and Qt
  screenshots are under ignored `artifacts/critical-review-2026-09-08/` and
  `artifacts/critical_review_*.py`. The report links the exact evidence files.

## Previous update — 2026-09-08: compact seed annotation controls and centroids

- Seed selector is now `Seed:` / colour swatch / four-digit numeric spinner
  (1–9999) / `Next empty`. `Show selected only` appears once, on the next row;
  the second existing-seed dropdown is removed. The red `empty` notice remains.
  Next empty reports capacity rather than selecting an occupied ID when all
  9999 IDs are used. Existing uint16 annotation archives are not renumbered.
- Removed routine visible boundary-supervision and coat-vocabulary messages;
  contextual instructions are tooltips. Invalid legacy shape metadata still
  shows a visible warning. Condition, shape, painting, and Apply/save mechanics
  are retained.
- `seedvision/annotation/geometry.py` calculates area centroids of each painted
  ID in bounded row chunks. All pixels count, including disconnected fragments.
  A partial outline uses the **visible painted area's centroid**, not an
  inferred complete-seed centre. The editable CPU annotation raster is already
  available; no CUDA analysis raster is downloaded for this operation.
- Image overlays show constant-screen-size, white crosshairs with dark halos
  at these centroids. They obey annotation visibility, selected-only filtering,
  and independent annotation opacity. Cached geometry survives display-only
  changes, is invalidated by mask edits/replacement, and is cleared on image
  switches. Crosshairs never become manual detection markers or model inputs.
  Brush commits refresh centroid geometry **before** emitting the edit signal,
  so synchronous hilum-inspector listeners cannot read the previous stroke's
  cached centre. A real Qt brush-event regression verifies this ordering.
- Hilum direction is now the unit vector **from the painted seed centroid to
  the hilum point**, not a user-dragged vector. Pick/drag places or moves the
  landmark; the inspector shows its derived angle (clockwise from image-right).
  Direction is unknown at the centroid or without a landmark. It updates with
  mask edits/undo and is derived again when applying annotations, using the
  existing archive metadata fields. Merely displaying old archives does not
  rewrite their saved metadata. Manual direction controls are removed.
- Regression coverage: `tests/test_annotation_centres.py` exercises geometry,
  filtering/opacity/cache independence, compact controls, edit/undo/save
  round-trips, brush-event ordering, missing direction, and ID exhaustion.
  A dense synthetic 6240×4160 raster with 6040 IDs took 0.547 s for its initial
  centroid calculation; display-only changes reuse those compact coordinates.
  Qt visual checks are in
  ignored `artifacts/annotation-controls-compact.png` and
  `artifacts/annotation-centre-crosshairs.png`. All **58 focused tests passed**
  (16.725 s), logged in `artifacts/annotation-controls-focused-2026-09-08.log`.
  Final full discovery: **611 run, 608 passed, 2 skipped, 1 pre-existing
  missing-fixture failure** (276.125 s). The unchanged reference manifest still
  points to absent `images/IMG_9689c.JPG`; the fixture check remains intact.
  Log: `artifacts/full-suite-2026-09-08-annotation-controls-verified.log`.
  No commit/push requested in this turn; earlier local changes
  and user-owned draft/project files remain untouched.

## Previous update — 2026-09-08: reference-gradient input and viewer controls

- Checkpointed preceding work as `95e8032` before this request. No push requested.
  Local `analysis7.seedfiddle-project.json`, `seedfiddle_prompts.md`, and the
  pre-existing deletion of `prompts.txt` were deliberately not included.
- Reference edges consumes `edge_gradients.magnitude`, not generic thinned
  ridges. Continuous CUDA gradient support multiplies Physical compatibility;
  normalization also starts from float gradient support. The node applies its
  own NMS/hysteresis afterwards. Raw prototype calculations and conservative
  subtraction policy are unchanged. Settings format is now **20**; versions
  1–19 migrate the old ridge input and preserve disconnected/suspended branches.
- Toolbar: Run pipeline; independent checked/depressed Image/Pipeline view
  toggles; removed duplicate top-level fit/actual-size buttons (menu shortcuts
  remain); analytical opacity slider maximum 102 px instead of 170 px. Moved
  material/seed visibility controls out of the painting panel, after Annotate
  seed instances, with their own annotation-only opacity slider. These controls
  neither edit masks nor invalidate calculation caches.
- Pipeline header uses viewport width, includes Fit and 100%, and collapses
  editing commands into a Tools menu on narrow panes. Selected nodes have a
  saturated blue header/body and a 5 px cosmetic cyan outline; their status
  colour remains visible as a separate strip.
- Material colour overlays are ordered Foreground probability/HSV, Background
  probability/HSV, Other probability/HSV, then foreground-vs-nonseed excess.
  Other HSV reads the actual Other fit stored under the historical
  `excluded_component_*` metadata names. It does not refit or use Background's
  configurable chroma/frequency weights. Missing fits show an explicit message.
- See `tests/test_reference_edge_toolbar_revision.py` for display independence,
  narrow-header controls, overlay/socket ownership, and numeric production-vs-HSV
  agreement. `docs/LOCAL_EDGE_NORMALIZATION.md` now describes continuous support.
- Verification: 75 focused graph/migration/gradient/toolbar checks passed;
  final UI/project/toolbar checks **95 passed** (75.670 s). Full `unittest`
  discovery: **602 run, 599 passed, 2 skipped, 1 pre-existing failure**
  (283.942 s). The sole failure remains the committed reference manifest's
  missing `images/IMG_9689c.JPG`; no fixture or manifest was removed or hidden.
  Logs are in ignored `artifacts/full-suite-2026-09-08.log` and
  `artifacts/final-ui-checks-2026-09-08.log`. Qt visual checks used actual Windows
  fonts at 1360 and 1920 px; pipeline controls were also tested at 420 px.

## New Windows computer setup

1. Install Git and a 64-bit Python version allowed by
   `config/runtime_dependencies.json`. The current supported range is Python
   3.12 through 3.14; Python 3.12 is the recommended baseline. Do not use a
   newer unsupported interpreter merely because it is the newest release.
2. Clone and enter the repository:

   ```powershell
   git clone https://github.com/efedo/SeedFiddle.git
   Set-Location .\SeedFiddle
   ```

3. Create the optional project venv, install the pinned runtime packages, and
   launch:

   ```powershell
   py -3.12 .\seed_vision.py --environment venv --bootstrap auto
   ```

   The launcher installs PySide6, NumPy, OpenCV, and the configured CUDA-enabled
   PyTorch wheel. Online bootstrap requires normal package-index access.
   `--offline` works only after compatible wheel files have been placed in
   `wheels/`; wheel binaries are intentionally not committed.

4. Verify the runtime and GPU:

   ```powershell
   .\.venv\Scripts\python.exe .\seed_vision.py --diagnostics
   ```

   Confirm that PyTorch is compatible and the acceleration line reports CUDA.
   The NVIDIA driver must support the configured PyTorch CUDA wheel; a separate
   system CUDA toolkit is normally unnecessary.

5. Run the test suite:

   ```powershell
   $env:QT_QPA_PLATFORM = "offscreen"
   .\.venv\Scripts\python.exe -m unittest discover -s tests
   ```

6. Launch later with:

   ```powershell
   .\.venv\Scripts\python.exe .\seed_vision.py --no-bootstrap
   ```

If the venv bootstrap fails, delete only the incomplete `.venv`, install the
recommended supported Python version, and rerun step 3. Never delete the
repository or `images/` while troubleshooting the environment.

## Current implementation

- **The complete right-pane node-control catalogue is audited and grouped by
  operation.** All 271 controls across 38 configurable active/toolbox nodes are
  assigned to 99 exhaustive node-owned sections. Graph construction rejects a
  missing, duplicate, unknown, or unsectioned control; tests also require every
  computational setting to be either visible or explicitly documented as a
  private/compatibility field. The audit exposed three previously fixed
  procedural controls (reference-surface occupancy weight and sparse/packed
  marker-area priors), corrected misleading control descriptions, added
  calibration/wavelet edit-time validation. See `docs/NODE_CONTROL_AUDIT.md`.
- **Project is the sole root of every per-image graph.** It replaces the former
  Raw images and Manual annotations pseudo-roots and exposes one typed **Raw
  image** plus one typed **Annotations** bundle. The annotation bundle retains
  separate Background, Foreground, Other, annotated-instance, and manual-centre
  layers internally. The right pane shows the current master path, saved or
  modified state, available/unresolved image counts, selected image and
  dimensions, species, and applied annotation counts. Metadata now consumes
  the Project image context, so every active and restored toolbox node has
  Project in its ancestry. Settings schema version 9 migrates versions 1--8 by
  folding the two retired roots and their wires into Project.
- **Seed-boundary confirmation** is now an active, proposal-independent edge
  diagnostic rather than a consumer of provisional/annotated instance shapes.
  Curved trace pixels perform a reverse centre transform over seed-scaled
  normal radii and bounded tangent offsets (required because a true ellipse
  centre is not generally on each boundary normal). Local vote maxima gather
  compatible ridge coordinates; CUDA moment reduction estimates orientation,
  then a compact radius/axis-ratio bank is scored against 16 perimeter sectors
  for ridge coverage and tangent agreement. Implausible sizes/axis ratios and
  duplicate centres are rejected before output. **Most likely edge ovals**
  draws the retained compact geometry, while **Oval-derived seed-centre
  probability** rasterizes only retained oval scores and applies the configured
  seed-relative centre blur. Raw centre votes remain a separate pre-validation
  diagnostic. The full raster remains lazy/on-device and compact oval geometry
  transfers only when its vector overlay is selected. Candidate-instance
  changes invalidate only candidate consumers and cannot alter these fits.
- The overlay selector is a node-first menu whose headings exactly match the
  active pipeline cards and inspector titles. It has no checkable children,
  synthetic Viewer/None group, or Qt mnemonic ampersands. Every selectable
  overlay is mirrored by an identically labelled output connector on its owner
  node, including dynamically generated colour/pattern overlays; a Qt contract
  test selects every owner and checks all three surfaces.
- The Project annotation bundle combines the former Reference layers and Manual
  seed-centre inputs while preserving each evidence type and its image-local
  persistence.
- **Material colour probabilities** now combines the foreground and background
  colour cards and controls without combining their calculations. Foreground,
  Background, and Other remain independent evidence rasters with their prior
  equations and cached intermediates. **Use background colour analysis** turns
  off only Background/Other fitting; mandatory-reference Foreground fitting
  remains active. Version-4 settings migrate both former node records and
  connections into this combined node.
- **Material noise probabilities** now combines the former foreground- and
  background-noise cards without combining their calculations. All 20 numerical
  controls, explicit independent Background/Other and Foreground enable switches,
  the Foreground/Background/Other noise overlays, distinct maximum versus first-
  tertile defaults, CUDA cache products, and internal timings are retained.
  Changing the combined card invalidates both internal products; version-5 and
  older profiles merge their former node records, enabled states, and wires into
  this owner.
- Node-editor wires now receive stable deterministic per-wire colours from a
  shared source-node palette. Unobstructed bundled cable trunks and branches use
  smooth cubic curves; obstacle-avoiding routes retain rounded corners.
- Ruler evidence rendering extends imperial display ticks to the exterior edge
  without changing detected coordinates or scale calculations. The summary
  text has a dark translucent contrast plate so it remains readable on pale
  images.
- The **File** menu now separates portable analysis settings from complete
  project masters. `.seedfiddle-settings.json` profiles contain every graph
  parameter, enabled/active/toolbox state, and authored connection state, but
  deliberately exclude node layout, images, species, annotations, manual
  centres, statuses, and caches. `.seedfiddle-project.json` masters embed that
  profile plus ordered source-image records and fingerprints, species/current
  selection, node positions, selected node/overlay, and the two cable-display
  flags. They reference rather than embed the full-resolution source-bound
  reference-region and manual-centre sidecars. Project shortcuts are `Ctrl+N`
  New, `Ctrl+Shift+O` Open, `Ctrl+S` Save, and `Ctrl+Shift+S` Save As; `Ctrl+O`
  remains Open images. Recent masters use QSettings. Open validates the full
  schema and graph compatibility before mutation, clears every prior per-image
  draft/cache/history, and restores available images without leaking unlisted
  deterministic sidecars. Missing/changed source records and unresolved listed
  sidecar references remain byte-for-byte in a degraded master on Save.
  Hash-changed sidecars load only through their explicit manifest path and own
  source binding; corrected-coordinate masks wait for current calibrated shape
  validation. A successful user edit switches that image to its canonical
  autosaved sidecar. Version-one masters deliberately reference these canonical
  mutable per-image sidecars rather than snapshotting or versioning them; two
  masters pointing at one sidecar share its current annotation content, while
  the recorded SHA-256 detects changes on the next open. Dirty project
  replacement/close and unapplied drafts use
  Save/Discard/Cancel; autosave failures remain dirty and are retried before a
  master can be saved. Project/profile mutation is disabled during background
  analysis, training, or fitting.
- Colour-card/swatch detection, colour balance, projective deskew, ruler
  detection, 5 cm scale overlay, absolute scale, and dual Petri-dish rims.
  Ruler assignment now treats tick detection as a separate evidence stage:
  narrow high-contrast bands along both ruler edges are evaluated independently,
  then terminally bounded, regularly spaced tick families are fitted without
  averaging the competing reverse scale, barcode, text, and plastic outline
  into one profile. A separate **Ruler detection evidence** overlay draws the
  likely outline green, metric ticks red, imperial ticks orange, metric-number
  components dark red, and imperial-number components dark orange. Absolute
  scale is accepted only from a reliable metric tick pitch; endpoints and an
  imperial competitor never silently establish authoritative pixels/mm. Tick
  transverse lengths are measured before semantic assignment and clustered
  into minor/intermediate/unit classes. The measured hierarchy can shift a
  preliminary lattice phase (fixing the former one-sixteenth `IMG_9533`
  imperial error), recurring long ticks define 10 mm and 1 inch dividers, and nearby
  printed glyph groups are associated with inferred number/unit labels. Metric
  and imperial tick families produce separate pixels/mm estimates; their
  symmetric percentage disagreement is shown as a sanity check and discounts
  confidence rather than being averaged away. Each scale also reports a length-
  hierarchy consistency score and is rejected unless the longest measured
  ticks correspond to its major increments. Tick assignment now
  samples only the compact ruler strip at full source resolution, follows every
  local maximum across the complete 151-dash metric and 97-dash imperial
  lattices, and pools the printed length hierarchy so glare-weakened marks are
  mapped instead of dropped. The second family must occupy the opposite edge.
  An expanded strip fits the transparent-plastic top and bottom edges outward
  from their respective tick roots, then independently fits left and right ends
  outside the terminal ticks. Their intersections form a mildly projective
  quadrilateral, so residual skew is not forced into a rectangle and the
  neighbouring colour-card border cannot become the ruler outline. See
  `docs/RULER_EVIDENCE_DETECTION.md`. The Reference seed
  scale overlay draws a dark-haloed yellow maximum-width fit around every actual
  selected reference component and places its bold diameter text immediately
  above it on an opaque dark contrast plate. Local foreground-core refinement prevents attached shadows from
  inflating this initial estimate. Applied annotated instances override it when
  at least two complete masks exist: the configurable largest fraction defaults
  to 25%, image-cutoff/disconnected masks are excluded, and the overlay draws
  every measured chord plus a histogram and final mean.
- Petri-dish search uses the ruler-calibrated, editable expected 96 mm outer
  diameter as a soft tie-breaker and fallback. A coherent full-circle visual
  rim pair overrides an inconsistent ruler scale, preventing the oversized
  failures seen on dense `IMG_9533.JPG` and sparse `IMG_9546.JPG` while still
  using physical scale when visual evidence is ambiguous. A bounded rotated-
  ellipse refinement retains mild camera-view ovality (up to a 1.15 axis ratio);
  circular downstream consumers use the larger semi-axis so they cannot clip
  seeds near the shorter projected rim.
- Native Qt node graph with inline Blueprint-style controls, dependency-aware
  caching, node timings, progress colours, purple adjacent-node highlighting,
  manual overlap, automatic non-overlapping arrangement, zoom controls, and
  node-driven viewer overlays/intermediates. Every authored datum has its own
  succinct labelled and typed input/output socket; all active and toolbox
  connections use explicit endpoints. Users can drag to restore supported
  connections, drag a connected input into empty space to disconnect it, or
  right-click an edge. Plain left-drag over a connection pans instead of
  selecting its large path bounds; Ctrl-click explicitly selects an edge for
  Delete/Backspace. Right-clicking an enabled implemented node offers **Run to
  node**, which executes only that node and its enabled graph ancestors while
  retaining node-local caches. Two independent default-off toolbar options simplify dense
  wiring without changing the DAG: **Bundle cables** gives compatible wires from
  one source a shared initial trunk before they branch toward their destinations,
  while **Route around nodes** uses rounded detours (and tighter necessary bends)
  to keep connections out of intervening node cards. Disconnection bypasses enabled consumers
  and their dependents; reconnection restores only the cards automatically
  suspended by that missing input, leaving deliberately disabled experimental
  nodes off. The **Project** input exposes a typed image-local annotation bundle
  to every calculation that consumes authored evidence.
  Physical and non-physical edge training
  masks are regenerated from the annotated IDs and are not separately painted
  graph inputs. The one-line toolbar includes an
  unused-node toolbox; **Circle candidates**, **Distance-peak candidates**,
  **Calibration residual risk**, the
  directional surface-darkness gradient branch and its lightening/darkening
  derivative upper cutoffs, and every distance-dependent node are preserved
  there with their authored connections but excluded from the default DAG and
  calculations.
- The complete 53-node active/toolbox catalogue has an independent direct-input
  contract test. Corrected-image, dish-region, absolute-scale, seed-diameter,
  mask, proposal, and reference dependencies are explicit wherever the runtime
  reads them. In particular, **Edge gradients** consumes **Layout detection**'s
  dish region and **Oriented edge traces** consumes **Seed scale estimate**;
  scale-only changes rebuild traces while retaining cached ridges. Oriented
  traces expose their selected source in the node and overlay legend and can use
  generic ridges, the authoritative supported reference ridge, the optional
  conservative net-reference ridge, or the normalized supported reference ridge. Trace-gap
  bridges additionally require tight tangent alignment, the same directed
  gradient side, and a facing endpoint, preventing the former gap-4 collapse of
  adjacent parallel seed rims into one component. An explicit `off | prefer |
  require` convexity policy now rejects S-shaped candidate bridges using an
  editable angular ambiguity tolerance while accepting either C-shaped
  winding. It is intentionally a local initial-link constraint rather than a
  false claim of globally proving arbitrary trace components convex; policy or
  tolerance changes rebuild only trace state and dependents. Legacy wires
  to unused illumination, tangent, reflectance, radial, and image-quality data
  were removed, and the U-Net/StarDist lighting input is represented as separate
  flattened-grayscale, shadow, and highlight channels.
- Overlay selection and opacity are compact top-toolbar controls. The main
  selector groups indented layers under disabled owning-node headings, and a
  synchronized selector directly below the selected node title lists only that
  node's overlays. Material-reference painting and seed-instance annotation are
  top-toolbar modes that reveal one contextual control
  panel over the image; the right inspector contains only image metadata, the
  selected `Node:` controls, its local overlay selector and explanation, and
  applicable summaries.
  The image-pane toolbar always shows the current filename, and opening an
  image from any entry point selects and scrolls to the matching row in the
  left image list. The redundant **Add images…** button below the list was
  removed; toolbar/File-menu **Open images…** remains the single multi-file
  add-and-open command. Below the list, an always-visible **Analysis activity**
  card names the executing node, total/node elapsed time, queued reruns, long
  intervals without a node boundary, and cooperative cancellation of a
  superseded revision. A scrollable **References and annotations** inventory
  shows the exact canonical or project-bound sidecar, discovery/load/withheld/
  pending-calibration state, saved corrected dimensions, material pixel counts,
  annotated seed ID/pixel counts, unapplied draft state, and any bundled
  pre-annotation association for the current image.
- A separate **Annotate seed instances** mode records distinct-colour,
  full-resolution integer seed IDs. It provides a freehand brush, magnetic edge
  tracing, adaptive smart fill, shape-guided fill, and an eraser. Assisted tools
  show debounced live previews and commit on click: edge trace uses click anchors
  and a magnetic path preview, while both fill tools preview their regions even
  when the active seed has no prior marks. A **Show selected seed
  only** checkbox filters the display to one cropped ID and changing the ID
  recentres without altering zoom; invisible neighbouring labels are protected
  from paint and erase. The display is a single viewport-aware graphics item
  that requests exact 512-pixel source tiles as Qt exposes them; it never scales
  the instance labels through a whole-image display proxy. Tile-local indexed
  palettes preserve one categorical mask pixel per image pixel, and a bounded
  512-entry/48 MiB LRU cache keeps pan, zoom, edits, and selected-only filtering
  responsive without unbounded image-size-dependent storage. Trace edge, Smart
  fill, and Shape fill independently select thinned
  ridges (default), oriented traces, combined ridge-priority evidence, physical-edge
  probability, or broad edge magnitude. Smart fill and Shape fill additionally
  offer net physical-edge evidence, using the live display-only physical-minus-
  scaled-non-physical coefficient owned by the instance-derived edge node. Trace uses a banded continuity-aware
  live-wire seam, local optional directed/undirected tangents, and anchored
  component support so it cannot switch to a stronger parallel boundary. Smart
  fill uses OpenCV's native floating-range neighbour comparison, preserves other
  IDs, reaches the chosen one-pixel edge frontier, and exposes bounded
  tunnelling/tolerance/distance options for patterned seeds. **Maximum distance
  from cursor** defaults to 0.60 estimated seed diameter and is a true radial
  click-to-pixel limit (not an instance width); the separate maximum-added-pixel
  control remains an independent area safety cap. **Fall-off half-life**
  defaults to 0.40 seed diameter and applies exact pressure
  `2^(-distance/half-life)` to both the allowed inward-neighbour Lab-step limit
  and continuous edge threshold, respecting selected 4/8 connectivity;
  this strengthens stopping evidence with distance without creating fractional
  labels or blocking uniform/no-edge regions. Tunnelling carves one
  local weak-edge passage rather than globally weakening the ROI. A leaked flood
  that reaches its cursor-distance limit is replaced by a smooth local star-convex edge
  contour. Freehand drags use a lightweight vector stroke preview and
  rebuild the annotation raster once on release; they never rebuild the
  analysis overlays. Trace commits are exact one-full-resolution-pixel
  categorical lines, independent of the brush radius and without antialias
  expansion. The ordered trace session retains its cyan first anchor; returning
  to it previews and fills only the closed interior by default, while protecting
  labels belonging to other seeds. The combined evidence is the fixed maximum
  of generic thinned ridges and binary oriented traces plus physical-edge
  probability and broad edge magnitude at 45% strength; it is not a separately
  learned model. Assisted algorithms crop edge work to their local cursor
  region instead of rebuilding or normalizing full-resolution rasters on every
  event.
  Shape fill automatically fits a rotated ellipse prior, but the prior is only
  an approximate maximum extent. Its contour search is unrestricted inward for
  partially occluded seeds. Outward displacement has exponential pressure with
  a default 0.05-seed-diameter half-life and a hard 0.10-diameter cutoff. It uses
  fixed 8-neighbour growth internally and deliberately exposes no tangent,
  tunnelling, cursor-radius, radial-falloff, or connectivity controls. The old
  plain Snap shape tool has been removed.
  Draft/apply/revert state is per image and independent of the
  binary colour-reference masks; applied labels become authoritative markers
  for the active procedural watershed and also constrain the dormant
  provisional instance branch without overriding foreground probability.
  **New seed** advances to the next available ID without changing the active
  Brush/Trace/Shape-fill/Smart-fill/Eraser tool, its option page, or its
  configured values.
  A **Start from result** selector expands any available procedural,
  U-Net/watershed, or StarDist labels into a full corrected-image editable
  draft. The initializing method is preserved in export notes; the draft stays
  explicitly unreviewed and requires correction of every automated error.
- GPU foreground/background colour probabilities, symmetric three-band noise
  profiles with directional texture continuation, shared edge gradients,
  directed/undirected tangents, ridge thinning, and oriented edge traces.
  Foreground and background colour scores are independent evidence models and
  are deliberately not forced to sum to one: a pale seed may validly resemble
  both, and Other/unmodelled evidence must remain possible. Painted foreground
  and safely inset annotated-seed prototypes are fused as positive evidence by
  probabilistic union, so a near-zero automatic branch can no longer veto a
  strong reference match. The shared multifeature material-prototype node is a
  different model: its available Background/Foreground/Other class scores are
  jointly normalized with a fixed unknown mass, so their sum is at most one.
  Background texture positives are sampled directly from the exterior annulus
  in its own coordinate frame; they no longer disappear when projected into the
  smaller dish crop. The nominal texture/colour blend is reduced continuously
  when fitted class profiles have weak separation, preventing texture from
  overwhelming clearer colour evidence. Directional rays are integrated on
  CUDA into each class noise probability but
  are no longer materialized as individual viewer overlays. Foreground noise
  defaults to **1st tertile**, the exact linearly interpolated one-third
  quantile across ray directions; this is more conservative than median while
  avoiding minimum's all-direction requirement. Its CUDA reducer interpolates
  two `kthvalue` order statistics rather than allocating a sorted full-raster
  direction bank. Background noise retains its maximum default. The combined
  Material colour/noise nodes also expose direct-bright **Other colour
  probability** and **Other noise probability** overlays when Other examples
  have been painted.
  The colour view is raw target-model Lab membership; the noise view is raw
  target-only texture compatibility and receives no semantic counterclass or
  colour blend. These diagnostics remain blank without Other paint and are
  separate from the multifeature Reference Other-material prototype
  probability. Each node also exposes a viewer-only **Foreground vs
  Background/Other excess** diagnostic. At every pixel the stronger of
  Background and Other is compared with Foreground; positive Foreground excess
  is blue, positive Background/Other excess is red, and ties are black. The
  sibling Non-seed scores use a maximum rather than an addition because the raw
  evidence maps are independent and may legitimately overlap. These views do
  not enter material decisions, node caches, or downstream invalidation.
- **Hue only** displays corrected hue at fixed neutral brightness/chroma, with
  achromatic pixels shown neutral gray. **Wavelet decomposition** is a four-
  level stationary B3-spline à trous decomposition with selectable detail
  layers and a residual; all layers stay at full image resolution and the four
  details plus residual reconstruct the original corrected RGB tensor exactly.
- The **Project annotations** bundle contains the mutually
  exclusive Background/Foreground/Other material layer, integer annotated seed
  instances, and source-coordinate manual centre points. Painting a
  class clears the other two at that pixel; Other supplies a competing learned
  distribution to both material models and attenuates a class only where it fits
  better, preserving colours shared with legitimate positive evidence. It feeds
  both colour models and both class-specific noise
  models, so applying a painted edit invalidates every true graph dependent.
  Each noise classifier learns its positive and negative texture distributions
  directly from the applicable painted areas when present, using colour
  pseudo-labels only for an unpainted class. Painted coordinates are not forced
  to exact colour- or noise-probability zero or one. The material panel's
  **Include annotated seeds as Foreground** option defaults on. When enabled,
  it adds only safely inset interiors from applied seed IDs to the Foreground
  colour, directional-noise, and material-prototype sources; contours and their
  uncertainty band remain excluded. Painted Background, Other, and exclusion
  evidence takes precedence, manual Foreground remains additive, and unapplied
  drafts or display-only visibility changes have no analytical effect. There
  is no separately painted boundary layer: every complete annotated instance
  contributes its one-pixel contour as physical-edge supervision, while only strong edge/ridge
  candidates safely inset from that contour contribute sparse non-physical-edge
  supervision. Flat interior and the contour uncertainty band remain unlabelled.
  These derived examples fit image-local Physical and Non-physical tangent-strip
  prototype banks from corrected Lab context and polarity-neutral edge geometry.
  Their spatially broad outputs are explicitly labelled **prototype
  compatibility** and retained only as diagnostics: the two-channel view uses
  magenta for overlap, the excess view shows the unscaled signed margin, and
  **Net physical-edge prototype compatibility** shows
  `max(Pphysical - k × Pnonphysical, 0)` with a black floor.
  The Reference edges node's **Non-physical subtraction weight** defaults to
  0.5 and supports 0--2 without modifying either source compatibility raster.
  The authoritative **Reference-edge probability** is
  `thinned_true_edge_support × Pphysical` at full
  resolution. Descriptor/restoration halos are therefore exactly zero away
  from a real ridge. Assisted fill, curve confirmation, learned-instance
  inputs, and procedural separation receive only this supported field, its
  normalized/thinned derivatives, or separately edge-supported Physical and
  Non-physical channels; raw compatibility never serves as a barrier.
  The previous supported subtraction is retained separately as **Conservative
  net physical-edge evidence**, with explicit Smart/Shape-fill and thinned-ridge
  choices. Only this conservative branch and the raw Net compatibility diagnostic
  depend on the subtraction weight.
  Local normalization acts on the thinned true-edge support before reapplying
  Physical probability, estimates a seed-scale winsorized local RMS envelope,
  and applies a bounded one-sided gain that can enhance but does not attenuate
  above-floor supported evidence. The exact full-resolution ridge mask is
  reapplied after interpolation. The per-pixel gain denominator is capped by
  current support so a neighbouring strong arc cannot suppress its weak
  continuation. A smooth absolute gate keeps zero-ridge regions at zero, and
  ridge hysteresis admits enhanced weak maxima only when they connect to a
  strong seed. Its default
  radius, target support, maximum gain, and floor are
  respectively 0.30 seed diameter, 0.35, 2.5x, and 0.04. The full rationale,
  equations, safeguards, and validation plan are in
  `docs/LOCAL_EDGE_NORMALIZATION.md`. The node owns those four controls plus
  independent NMS step, low/high threshold, hysteresis-reach, and working-size
  controls. Raw compatibility diagnostics remain viewable but are not
  operational edge sources.
  The formerly black continuous normalized overlay was a
  display-quantization bug: its `[0, 1]` raster is now multiplied to display
  range before conversion to `uint8`; its derivative overlays were already
  generated through a separate correctly scaled path. When no annotated seed
  exists, the semantic classifier is neutral instead of treating every generic
  image edge as a physical reference. Learning export derives both boundary
  targets and their sparse validity raster from the instance labels.
- Active **Procedural seed separation** combines seed-material evidence,
  generic edge/ridge candidates, supported instance-derived semantic evidence,
  normalized reference-edge ridges, and coherent convex oriented traces into
  its physical-boundary cost. Edge-supported class channels and the normalized
  authoritative field gate/discount the candidate without admitting broad
  descriptor halos; the gate is neutral with no annotations. Centre markers
  come from smoothed material geometry and seed-interior depth, not sensor/noise,
  shadow, or internal-boundary peaks. Each centre is evaluated at several
  material thresholds (default five). Hard minimum area, maximum width,
  concavity, thin-protrusion, solidity, axis-ratio, and maximum-area checks
  reject impossible candidates, including candidates originating from
  annotation markers. Complete annotations above the hard minimum are retained
  as exact authored shapes instead of being expanded by watershed; tiny legacy
  centre marks still seed a generated candidate that must pass the same hard
  geometry checks. Soft minimum-area and maximum-width bands
  lower scores before their hard cutoffs. A score-ordered overlap-aware pass
  selects the final non-overlapping combination and calculates explicit
  per-instance confidence. Its eight overlays expose material likelihood/mask,
  boundary cost, centre likelihood, instance identities, confidence, convex-hull
  concavity, and unselected alternatives coloured by relative score. Clicking
  an instance in the main result selects it, outlines it in yellow, selects the
  procedural node, and reports its area, maximum side-to-side width and width
  fraction, concavity, protrusion, solidity, axis ratio, confidence, and source
  in the inspector. GPU rasters are resized before the one
  essential bounded CPU topology transfer; CPU labels and evidence stay at the
  bounded working size and Qt scales them only for display. The compact result
  is node-cached, and
  distinct painted instance IDs suppress nearby automatic markers.
  The **Project annotations** graph input and its inspector editor allow
  per-image placement and adjustment of the actual watershed markers rather
  than the final region centroids. Automatic markers are hollow cyan, manual
  or replacement markers solid yellow, annotation-authoritative markers locked
  magenta, and rejected manual points red with their reason. Click adds, drag
  moves, and right-click/Delete removes; Escape cancels the active drag or exits
  editing. **Augment automatic** retains discovery while suppressing nearby
  duplicates. **Replace automatic** uses the editable list as the complete
  automatic-marker set but never removes annotation markers. Switching to
  Replace seeds the list from every currently surviving non-annotation marker;
  dragging or deleting an automatic marker makes the same conversion so its
  former maximum cannot silently reappear. Image-local Undo and Reset are
  available. Completed gestures alone autosave a compact SHA/dimension-bound
  source-coordinate sidecar under `projects/manual-seed-centres/`; the fresh
  calibration transform maps exact source floats to corrected and crop-local
  coordinates on every run. The per-image input invalidates only procedural
  inference and its dependents, and editing stops while analysis runs or when
  the image/node/result changes.
  Its inspector also offers a bounded annotation-guided parameter fit. Trial
  segmentations explicitly receive no annotated watershed markers, and are
  compared with the applied masks using deterministic one-to-one overlap
  matching. False-positive pixels use a bounded exponential distance cost:
  their cost is zero on the target, small immediately outside it, reaches the
  editable 2.0 amplitude at the default 0.50-seed-diameter scale, and rises
  exponentially farther away. Matched results measure distance from their own
  annotated target (not the annotation union), while missed seed area remains
  1.0 and unmatched instances retain a small count term. Raw FP counts remain
  available alongside distance-weighted FP equivalents. **Annotations cover
  whole dish** defaults off, so partial-review
  metrics explicitly ignore wholly disjoint predictions. When checked, every
  predicted object is evaluated, including standalone background false objects.
  Search results are proposals only; accepting one uses a single atomic graph
  update and invalidates the procedural node and dependents once. The fit is
  image-local and in-sample, but accepted settings are global to every image;
  neither mode is held-out validation.
- **Layout detection** now also owns the perimeter Background reference instead
  of exposing a redundant separate node. Its ruler-calibrated outer-rim buffer
  defaults to 0.35 cm and its independently
  adjustable median-colour band thickness defaults to 0.5 cm; both are shown
  exactly in a dedicated overlay, with a nominal-dish scale fallback. The
  overlay includes an opacity-independent swatch and hex label for the selected
  median starting background colour. The Material colour probabilities node's
  Background colour probability overlay
  also evaluates the exact fitted positive Background Lab model across
  this bounded GPU annulus and displays it beside the unchanged dish crop; its
  annulus is outline-only so the probability values remain legible. Background
  painting now defaults **Keep perimeter-matched source** on, combining only the
  colour-filtered ring pixels and their Lab samples with manual Background
  marks. Similar-coloured pixels inside the dish are evaluated by the resulting
  model but are not recruited as references. In the painting view, only the
  retained ring is cyan while manual marks stay green. Unchecking the control
  removes the complete perimeter-derived source;
  without paint, the colour model falls back to its generic light/low-chroma
  automatic selection.
- **Hierarchical material evidence** is the sole authoritative Seed-versus-
  Non-seed decision. Raw Foreground, Background, Other, directional texture,
  and valid multiclass prototype maps remain independent and need not sum to
  one. The node publishes resolved Seed, resolved Non-seed, conflict/ambiguity,
  and unknown masses that sum to one, plus a second Background/Other decision
  with explicit subtype ambiguity and unknown. Other and Background are
  parallel positive Non-seed classes; neither suppresses the other. Automatic
  Foreground/perimeter authority decays exponentially with reviewed coverage.
  Every raw source also earns reliability from its target median versus the
  90th-percentile reviewed top-level non-target response; Background and Other
  calibrate only against Foreground, never one another. Reliability scales
  contribution without changing the displayed raw map. The audited binary
  Seed threshold is 0.68. Material prototypes publish probabilities only with
  at least two material banks; physical/non-physical edge probabilities require
  both edge banks. Procedural occupancy, restored distance candidates,
  provisional assignments, seed-interior smoothing, and curve semantic sides
  now consume the resolved decision. See
  `docs/MATERIAL_EVIDENCE_REDESIGN.md` and
  `docs/EVIDENCE_AND_ANALYSIS_FORENSIC_AUDIT.md`; the read-only saved-sidecar
  diagnostic is `tools/audit_material_evidence.py`.
- A six-output active multiscale node supplies fine/medium/coarse surrounding RMS
  darkness and Lab-colour noise energy. The maximum one-sided lightening and
  darkening CIE L* surface slopes, their query-to-target directions, and the two
  dependent derivative upper cutoffs are disabled and preserved in the
  unused-node toolbox. These raw masks are intentionally distinct from the
  learned foreground/background noise profiles.
- Two annotation layers: categorical material references
  (Background/Foreground/Other) and labelled seed instances. The material layer
  enforces exclusivity while it is painted and normalizes older mask state when
  it is read. Other samples fit independent positive colour-frequency and
  texture distributions. They are no longer subtracted from either Foreground
  or Background, in the raster calculation or colour-profile inspector. The
  compact editor uses class selectors plus shared Paint/Eraser/Clear-layer,
  Apply/Revert, brush-radius, and independent Materials/Seed-instances
  display-only visibility controls; longer
  explanations live in tooltips. Its drag handle repositions the panel, and
  clearing a class restores Paint mode so that it can immediately be redrawn.
  A context-aware **Undo** button and `Ctrl+Z` retain 20 unapplied commands per
  image in separate reference and seed-instance histories. Each drag, assisted
  click, clear, imported label map, or calculated starting draft is atomic;
  categorical peers and instance provenance are restored together. Histories
  use compressed changed tiles rather than full-resolution snapshots and are
  rebased by Apply, Revert, or validated disk restoration.
  Seed-instance IDs are summarized in one streaming row-run scan with exact
  categorical 8-connectivity. The editor warns, without blocking Apply, when
  the selected ID or any other ID occupies disconnected areas; the compact
  summary is cached by immutable/copy-on-write label-array identity.
  Smart fill calls its floating-range Lab control **Neighbour colour step**:
  candidates are compared with accepted touching pixels (L at the displayed
  step, a*/b* at 72%), not with the initial click. The independent
  **Click-origin colour range** compares every newly added pixel directly with
  the clicked pixel on the same channel scale, preventing accumulated gradual
  drift. Its conservative default is 72; it is a hard range independent of
  radial fall-off, while existing pixels of the active instance remain
  preserved. **Edge barrier threshold**
  is the minimum selected edge strength that stops growth; lower values stop
  on weaker evidence. Tooltips also document the one local tunnelling passage.
  The additional **Shape fill** tool performs a bounded centre/scale/axis-ratio/
  axial-rotation search, treats the fitted ellipse only as an approximate outer
  prior, and optimizes an independent cyclic contour using edge, Lab-contrast,
  and interior-core evidence. It
  previews prior versus refined boundary and refuses low-coverage, long-gap,
  overlapping, oversized, or irrelevant evidence instead of stamping an oval.
  While Shape fill is active, each mouse-wheel notch changes its visible oval
  size preference by 5% and does not zoom the image. The refined contour
  validates the fitted evidence but is not a hard mask around the canonical
  Smart-fill result. The shape prior imposes no inward restriction for occlusion,
  while outward
  extension decays with a 0.05-diameter half-life and stops at a hard
  0.10-diameter cutoff. Shape fill reuses Smart fill's neighbour Lab step, edge
  barrier, maximum-pixel cap, and frontier inclusion with fixed 8-connectivity,
  but not its tunnelling, cursor-distance, or radial-falloff controls.
  The Lab probability models now have node-owned, full-image-pane HSV
  diagnostics for both colour classes. Hue and saturation are the visible axes;
  a toolbar Value slider scans exact brightness slices, starts at the dominant
  visible fitted mode, and has a **Peak** reset. Achromatic membership is shown
  once in a neutral swatch rather than projected across every hue. The
  underlying colours are not dimmed. The contextual editor keeps natural row
  heights and scrolls vertically when the split image viewport is short.
  **Apply + save** for either a reference draft or a seed-instance draft writes
  the material raster and integer seed IDs/provenance to one atomic, version-2
  categorical NPZ
  under ignored `projects/reference-regions/`; unapplied strokes are never
  persisted. The explicit **Save applied reference regions** command remains
  available. An automatic-save failure warns without rolling back the valid
  in-memory applied state. First opening an image in a window restores the
  immutable applied snapshot only after its stored SHA-256 and dimensions match
  the current source; mismatch/corruption warns and installs nothing. Switching
  away and back preserves newer in-memory applied state. When no analysis cache
  exists for the returning image, the viewer now renders the authoritative
  material and seed-instance layers directly over the source instead of clearing
  them while it waits for a later analysis. An empty snapshot can intentionally
  replace an older nonempty archive. Version-1 archives are
  still validated and loaded, but their retired manually painted boundary
  raster is intentionally ignored rather than restored or migrated into
  semantic training evidence.
  **Load reference mask…** first looks for the current photograph in the
  version-1 `seed-instance-references/manifest.json` bundle. Bundled `uint16`
  masks stay in immutable source-image coordinates and are accepted only after
  their safe relative paths, source and mask SHA-256 digests, source dimensions,
  declared ID count, and categorical values validate. The current calibration
  homography maps them into corrected coordinates with nearest-neighbour
  interpolation, and loading is refused if any nonzero ID disappears. A valid
  mask becomes a replacement-confirmed, undoable, unapplied draft; manual
  corrected-coordinate PNG/TIFF/NPZ import remains available through the
  explicit **Choose mask file…** action even when a matching bundle exists. The
  committed bundle covers all eleven image fixtures and includes the two isolated
  reference seeds outside each dish. Every entry is explicitly
  `reviewed: false`: these machine-prepared masks are editing bootstraps, not
  accepted training or evaluation ground truth, until every contour is reviewed
  at full resolution and applied.
- **Reference texture prototypes** is an active CUDA-first, image-local
  procedural classifier fed by the corrected image, detected dish region, seed scale,
  material references and annotated instance IDs, shared gradient/ridge/tangent fields, and all six raw
  frequency-noise masks. Independent controls retain up to 64 robust coverage
  medoids per material class by default (Background, Foreground, and Other) and
  256 per instance-derived edge class (Physical edge and Non-physical edge).
  The edge capacity supports up to 1,024; GPU fitting assignments and full-raster
  evaluation are chunked to bound temporary memory independently of that limit.
  Material features combine Lab, multiscale energy, local residual, edge, ridge,
  and density evidence. Edge features use narrow tangent-aligned strips with
  explicit interior, centre-edge, and exterior pools, signed cross-edge Lab
  contrast, local residuals, edge/ridge support, and axial tangent coherence.
  Reviewed instance geometry supplies the outward normal only while fitting;
  inference uses image tangents and tests both polarities, avoiding annotation
  leakage. Material matching defaults to a 960-pixel working limit, while edge
  matching adaptively targets 28 pixels per seed diameter up to an independent
  2,048-pixel hard cap. It produces Foreground/seed-surface, Background, Other,
  Physical-edge, and Non-physical-edge likelihoods without hard-writing reviewed
  pixels. Gaussian-kernel matches are calibrated in two stages: the unsharpened
  best match determines known-versus-unknown confidence, while separate exposed
  material and edge class contrasts sharpen only relative Background/
  Foreground/Other or Physical/Non-physical competition.
  This avoids the former middling ceiling for ordinary in-class descriptors
  while preserving weak-match unknown mass and exact subtype ties. The
  calibration receives no masks or reference coordinates, and reviewed pixels
  are never hard-written. Every
  probability/likelihood overlay now states its low and high display colours in
  the image legend (direct maps are black-to-bright, while inverse background
  evidence is explicitly labelled white-low/black-high).
  The seed-surface result has adjustable inputs to the active procedural
  watershed and the dormant seed-interior branch. The image pane can replace
  the photograph with a scrollable, zoomable collage of every medoid, grouped
  by class with source counts and support percentages. The displayed squares
  are context only; tangent-aligned edge thumbnails mark the centre sample in
  yellow and two polarity-neutral side strips in cyan. When no annotated
  instance exists, physical/non-physical
  semantic output is neutral; unknown regions retain neutral generic edge/ridge
  support in the procedural separator but do not receive a physical label.
  Complete applied instance IDs always derive a one-pixel Physical-edge contour.
  Strong edge/ridge candidates safely inset beyond the configurable buffer
  become sparse Non-physical-edge examples; flat interior and the uncertainty
  band remain unlabelled. These derived masks are regenerated from the persisted
  IDs and current evidence rather than stored separately.
- Full-image CUDA jobs use one dedicated worker and coalesce newer requests.
  A newer request cooperatively cancels the superseded analysis at the next
  timed node boundary instead of invisibly waiting for the obsolete revision to
  finish; the current long-running node remains visibly identified while it
  reaches that safe boundary. Worker cancellation and every worker exception
  have explicit Qt signals, and completed-result installation has its own GUI
  error boundary so a post-worker display exception cannot leave an orphaned
  busy/idle state.
  Per-image node caches are LRU-bounded to three images and 2 GiB of reachable
  CUDA storage; eviction and failures release GPU ownership and lazy CPU mirrors.
  Overlay downloads survive only until the next viewer render. Painted masks
  use immutable applied arrays; categorical peers are copied lazily on the first
  dab that can replace them, rather than when the panel merely opens. Both reference and instance freehand tools use incremental vector
  previews rather than rebuilding the corrected base image during pointer moves.
  Corrected-coordinate reference archives are fingerprint/schema checked, then
  held as visibly pending until calibration confirms their dimensions. The
  completion transition now adopts the corrected viewer raster before binding
  those masks. This fixes the IMG_9533 failure in which valid 6508×4294 masks
  were installed against the still-raw 6240×4160 scene, raising an uncaught
  Qt-slot shape error after the worker had already disappeared from the active
  task table. Applied archives derive their saved dimensions from their
  categorical rasters/current corrected result rather than whichever raw or
  diagnostic pixmap happens to be displayed.
- Desktop launches configure a per-user rotating diagnostic log at
  `%LOCALAPPDATA%\Seed Fiddle\logs\seed-fiddle.log` on Windows, retaining five
  5 MiB backups, plus a `seed-fiddle-crash.log` faulthandler target. Analysis,
  learning, procedural-fit, edge-fit, and GUI result-installation boundaries
  record full tracebacks before emitting concise Qt failure messages. Uncaught
  main/thread exceptions and Qt warnings/errors are also captured. Failure
  dialogs and **Help → Runtime summary** expose the absolute log paths. Before
  this addition, caught worker/installation exceptions were reduced to one-line
  dialog strings and no persistent application log existed.
- **Grayscale & local lighting** estimates a valid-mask-aware broad field,
  flattens grayscale with a clipped log ratio, and maps standardized local
  deviations through separate nonlinear shadow/highlight sigmoids. All rasters
  remain GPU-resident and selectable as overlays.
- The dormant **Circle candidates** toolbox node has authored inputs from shared
  edge magnitude, sensor/noise, flattened grayscale, shadow, and highlight
  products. Its CUDA ring bank weights their boundary transitions independently,
  while remaining disabled and outside the default DAG. A selected restored
  circle node can be moved back to **Unused nodes** and restored again later.
- Foreground references use a per-pixel Lab colour-frequency table, never a
  regional average or painted-pixel probability override. Automatic background
  colour is anchored to the retained outside-dish annulus distribution, which
  remains additive to painted Background references by default.
- The combined Material colour node exposes separate **Maximum reference colour
  modes** controls. Foreground retains coverage-preserving quantized
  Lab frequency cells, whereas background fits adaptive robust Lab mixture
  components. The background default is 32, its supported range is 1--256, and
  full-image membership is evaluated four modes at a time to keep peak CUDA
  memory bounded. Positive components use the strongest weighted membership,
  not an additive clamp: an 11-fixture check caught the additive implementation
  pathologically saturating seed surfaces when capacity rose from 4 to 32. The
  auxiliary competing Other distribution remains capped at 64 modes so raising
  positive background capacity cannot multiply it into an unexpectedly
  expensive model.
- Painted foreground colour modes are capped with coverage-preserving selection,
  not a top-frequency truncation. The former truncation allowed a larger varied
  reference mask to evict an existing smaller colour mode and collapse its
  membership. The HSV diagnostic now uses an exact full-pane hue/saturation
  slice at the selected Value, keeps visually neutral membership in a separate
  swatch, and labels only the leading modes near the selected Value instead of
  covering the plot with unrelated frequency cells.
- Foreground colour is now mandatory user-supervised evidence. Painted
  Foreground and, when enabled, safely inset applied seed-instance interiors fit
  the coverage-preserving Lab frequency model. Isolated reference-seed fits are
  scale-only. The background-distance/Otsu start, automatic colour source,
  automatic/reviewed viewer split, and self-refinement rounds are retired.
  With no authored Foreground source the colour/noise probabilities are zero and
  the Qt node reports that the reference is missing.
- Foreground, Background, and Other remain independent raw evidence scores, not
  calibrated complements. Class annotation coordinates calibrate source
  reliability and select training samples but never hard-mask, subtract, or
  overwrite any raw raster. Material evidence alone normalizes Seed versus
  Non-seed, Ambiguity, and Unknown; Background and Other are symmetric positive
  Non-seed evidence and their overlap becomes subtype ambiguity rather than a
  special Foreground suppressor.
- Every probability raster now shares the valid crop reaching the outer edge of
  the configured perimeter Background band. Only the binary seed-proposal gate
  remains limited to the interior analysis region.
- The redundant visible **Seed-interior probability** node has been retired.
  Downstream diagnostics consume the authoritative **Resolved Seed
  probability** from Material evidence decision, and the procedural node no
  longer publishes a duplicate Seed-material-likelihood overlay. A private
  smoothed raster may still be reused inside dormant advanced diagnostics, but
  it is not a graph node or selectable overlay.
- The application is branded **Seed Fiddle** and uses a tightly framed rounded
  soybean plus a high-contrast, separately readable fiddle bow under
  `seedvision/assets/seed_vision_icon.png`; a 32-pixel occupancy regression test
  prevents the taskbar mark from shrinking back into excess transparent space.
- The 11 low-resolution test photographs under `images/` are committed and are
  required by image-backed tests. Generated diagnostics under `artifacts/` are
  ignored.

The 2026-08-23 overlay/evidence/gradient cleanup made the following contracts
explicit:

- The visible toolbar overlay selector is a node-first `QMenu`; each node opens
  a right-arrow submenu of only its owned overlays. The old grouped combo is a
  hidden state/accessibility model. Physical/non-physical prototype rasters are
  owned by **Reference texture prototypes**; their combined and net products are
  owned by **Instance-derived edge probabilities**. A large red image-pane
  **Calculating …** banner tracks the selected overlay's actual owner/status.
- The retired **Foreground strength**, automatic Foreground, and reviewed-
  Foreground overlays are not selectable. The old foreground-feature raster is
  zero-valued compatibility data and has no live application-pipeline consumer;
  advanced seed-interior analysis receives resolved Foreground probability.
- Directed and undirected tangent pass-through nodes were merged into **Edge
  gradients**. That node now exposes derivative method, source-fusion rule,
  original/residual/detail-band inclusion, blur, chroma weight, normalization,
  gamma, and wavelet-detail gain. The stationary wavelet node is an authored
  upstream dependency and its signed details plus residual exactly reconstruct
  the corrected crop.
- **Oriented edge traces** names its ridge source and exposes an explicit trace
  diameter multiplier linked to the master seed diameter. Non-generic reference,
  net-reference, and normalized-net ridge sources retain short semantic
  fragments instead of collapsing to black under the generic minimum-length and
  junction filters.
- Procedural concavity display contains only exterior-connected convex-hull
  pockets between a candidate outline and its hull; seed interior and enclosed
  holes are excluded. **Next empty seed** replaces the ambiguous **New seed**
  label.
- Smart fill keeps tunnelling disabled in the UI. After click-origin clipping,
  star-convex recovery, and pixel limiting, newly added pixels are re-filtered to
  the click-connected component while pre-existing active-seed paint is
  preserved. This prevents detached narrow trails from surviving the new total
  click-origin colour range.
- The selected pipeline card now uses a saturated dark-blue border/body. Normal
  completed Background-colour and Procedural nodes use the standard completed
  status colour instead of warning orange.

The full suite passed 444 tests on Python 3.12.10 with PyTorch CUDA on an RTX
3070 on 2026-08-18. The 393 non-pipeline-UI tests passed normally (with one
POSIX-only portability regression skipped on Windows); the 51
pipeline-UI tests passed with the developer machine's stale ignored
reference-region autoload disabled so its modal warning could not block the
offscreen runner. This includes the comprehensive graph-contract correction,
exact perimeter-band background-probability display, SHA-bound reference
persistence, full-canvas soybean/bow icon replacement, unified
32-mode default background-colour capacity with bounded-memory evaluation,
selected-seed-only annotation display, selectable assisted-tool edge evidence,
continuity-aware Trace Edge, leak-recovering Smart Fill, parallel-safe
oriented trace-gap linking, and winding-independent local convexity preference
and requirement modes. It also covers exact first-tertile foreground-noise
direction integration as the foreground default, plus the independently cached,
tool-selectable internal physical-reference ridge and its CUDA NMS/hysteresis controls.
It also locks exact one-pixel open Trace Edge commits and protected inward fills
when an instance contour returns to its first anchor, annotation-derived
Physical-edge/Non-physical-edge evidence, and the non-mutating
asymmetric procedural-parameter fit plus its Qt lifecycle and cache invalidation.
The current redesign retires manual boundary painting and version-2 persistence
stores material references plus instance IDs only; version-1 boundary rasters
are validated but ignored. Complete instances now supply physical contours and
sparse safely inset internal edge candidates automatically, with neutral
semantic support wherever annotation evidence is absent.

The complete suite was run again on 2026-08-22 after adding local net-edge
normalization: 473 tests ran, 470 passed, and 2 platform/fixture-dependent tests
were skipped. The sole failure is the known user-worktree deletion of
`images/IMG_9689c.JPG`; its manifest-integrity test correctly reports that the
referenced fixture is absent. All 189 focused normalization, GPU-layer, graph,
cache, annotation-tool, procedural, settings-persistence, and Qt pipeline-UI
tests passed.

The suite was rerun after the probability/ruler/ellipse/wavelet/procedural/UI
redesign on 2026-08-22: 479 tests ran, 476 passed, 2 platform-dependent tests
were skipped, and the same single manifest-integrity test failed solely because
the still-deleted committed `images/IMG_9689c.JPG` fixture is absent. The 147
focused computational tests and all 55 pipeline-UI integration tests pass; the
settings-persistence suite also passes after accounting for the selectable
reference-ridge input to Oriented edge traces.

The suite was rerun after the analysis-lifecycle/reference-association repair on
2026-08-22: 482 tests ran, 479 passed, and 2 platform-dependent tests were
skipped. The sole failure remains the intentional/user-worktree deletion of
tracked `images/IMG_9689c.JPG`; its manifest-integrity test correctly reports
the missing source fixture. All 97 focused timing, project, persistence, and Qt
pipeline-UI tests pass. A live full-resolution IMG_9533 run also passed while an
initial run was deliberately superseded: cooperative cancellation completed,
the valid 6508×4294 archive installed all 11 IDs, the reference-dependent rerun
finished, and the controller returned to an idle state with no pending bundle.

The suite was rerun after measured tick-hierarchy alignment, mildly projective
ruler outlines, annotation-derived seed scale, and exterior-annulus noise repair
on 2026-08-23: 496 tests ran, 493 passed, and 2 fixture/platform-dependent tests were skipped.
The sole failure is still the user-worktree deletion of tracked
`images/IMG_9689c.JPG`; the reference-manifest integrity test correctly reports
that missing source. All 14 calibration tests and the expanded Qt ruler/seed-
scale overlay integration test pass. The real `IMG_9546.JPG` regression maps
all 151 metric and 97 imperial dashes, infers their measured length classes and
unit dividers, reports 18.000 versus 18.268 px/mm (1.48% disagreement), and
constrains four independently sloped outline sides outside and close to those
terminal rows. `IMG_9533.JPG` additionally locks the one-sixteenth imperial
phase correction, all five semantic length classes, full-inch endpoints, and
right-to-left 6…0 labels. A negative synthetic test shortens all centimetre
ticks and confirms that periodic spacing alone cannot pass the major-increment
sanity gate. The two focused Qt overlay regressions pass after adding a dark
halo to every isolated-reference circle and a dark plate behind its label.
Saved `IMG_9533.JPG` annotations override the 119.1 px refined isolated-
reference estimate with a 178.2 px mean of the selected four largest complete
annotated widths. Its exterior-ring median Background-noise score rises from
122 to 224 while its annotated-seed median falls from 184 to 130. The isolated-
fit GrabCut seed is fixed explicitly so these values and downstream seed counts
are independent of prior OpenCV operations in the process.

The suite was rerun after adding Smart fill's independent click-origin colour
range on 2026-08-23: 497 tests ran, 494 passed, and 2 fixture/platform-dependent
tests were skipped. The sole failure remains the user-worktree deletion of
tracked `images/IMG_9689c.JPG`; its reference-manifest integrity test correctly
reports the missing source. All 30 focused annotation, Shape-fill isolation,
and Qt control/session-state tests pass.

The suite was rerun after unifying annotation edge-source contracts and adding
persistent error diagnostics on 2026-08-23: 498 tests ran, 495 passed, and 2
fixture/platform-dependent tests were skipped. The sole failure is still the
user-worktree deletion of tracked `images/IMG_9689c.JPG`; its reference-manifest
integrity test correctly reports the missing source. The exhaustive Qt source
regression selects every Trace/Smart-fill/Shape-fill value, including the two
formerly mismatched normalized sources, and the rotating-log regression verifies
that a GUI-installation-style exception retains its complete traceback.

The suite was rerun after the overlay/evidence/gradient cleanup on 2026-08-23:
502 tests ran, 499 passed, and 2 platform-dependent tests were skipped. The sole
failure remains the pre-existing user-worktree deletion of tracked
`images/IMG_9689c.JPG`; its reference-manifest integrity test correctly reports
that missing source. All 246 focused annotation, procedural, graph, Qt,
persistence, material-evidence, visualization, and image-backed cache tests
pass. Runtime diagnostics report the supported Python 3.12/PySide6/NumPy/
OpenCV/PyTorch environment ready with CUDA on the RTX 3070.

The suite was rerun after the node/overlay/cable consolidation on 2026-08-24:
503 tests ran, 500 passed, and 2 platform-dependent tests were skipped. The sole
failure remains the pre-existing deletion of tracked `images/IMG_9689c.JPG`;
the reference-manifest integrity test correctly rejects that missing source.
All 122 focused graph, settings-migration, cable-routing, overlay-contract, and
manual-centre Qt tests pass, as do the remaining computational and UI tests in
the full run.

The suite was rerun after combining the foreground/background directional-noise
cards on 2026-08-24: 506 tests ran, 503 passed, and 2 platform-dependent tests
were skipped. The sole failure remains the same pre-existing deletion of
tracked `images/IMG_9689c.JPG`; the reference-manifest integrity test correctly
rejects that missing source. All 166 focused graph, UI, settings-migration,
overlay, and device-cache regressions pass, as do all 65 broader pilot,
material-evidence, project, inspector, learning, and procedural-fit integration
tests.

## 2026-08-24 pipeline consolidation and evidence audit

The visible graph now uses calculation owners instead of legacy wrapper cards:
metadata is a peer of Raw images; colour-card work belongs to **Deskew and
colour balance**; ruler geometry and absolute scale belong to **Ruler detection
and scale**; generic ridges belong to **Edge gradients**; and semantic ridges
belong to **Reference edges**. Settings versions 1--6 migrate into the merged
owners while genuinely unknown nodes, controls, and wires remain hard errors.

The material-noise audit found and removed two unintended colour dependencies:
colour thresholds formerly invented texture pseudo-labels, and the final
texture result was blended with colour probability. Material-noise outputs are
now texture-only. The surviving graph dependency carries only the safely inset,
option-controlled annotated-Foreground reference region so cache invalidation
remains honest. The edge-prototype audit likewise found that absolute ridge
amplitude was embedded in the descriptor and multiplied into the class output.
The descriptor now contains semantic strip appearance only; gradient strength
is introduced downstream when constructing boundary/ridge products.

Normalized-reference trace doubling was a bilinear restoration halo: every
positive halo pixel was treated as a separate ridge. Semantic ridge sources are
now re-thinned along their live normal before component linking. Shared edge
magnitude is tested to change with derivative method, and graph invalidation is
tested to stop at upstream owners and recompute only the changed node and its
descendants.

Automatic procedural, provisional, U-Net, and StarDist inference no longer
receive painted instance IDs. Procedural fitting uses annotation-independent
image/material/edge evidence and treats masks only as scoring targets.
Reference-edge fitting trains and evaluates on disjoint alternating instance
IDs. See `docs/PIPELINE_CONSOLIDATION_AND_EVIDENCE_AUDIT.md` for the full chain-
by-chain findings, equations/ownership decisions, and regression inventory.

The 2026-08-24 validation covered all 511 tests in clean process groups: 508
passed, 2 fixture/platform-dependent tests were skipped, and the sole failure
is the established user-worktree deletion of `images/IMG_9689c.JPG`; its
reference-manifest integrity test correctly rejects the missing source. The 19
real-image pilot tests pass, as do all 331 post-pilot graph, Qt, persistence,
procedural, reference, shape-fill, and visualization tests. A single monolithic
process exhibited retained-state degradation after the learning tests (a
17-second clean fixture analysis grew beyond 30 minutes while still consuming
CPU); isolated process groups complete normally and expose no application
pipeline stall.

The 2026-08-26 node-control audit validation ran 518 tests in one process: 515
passed, 2 platform/fixture-dependent tests were skipped, and the sole failure
remains the established user-worktree deletion of `images/IMG_9689c.JPG`.
Before that full run, all 120 focused graph/inspector/UI/settings-persistence
tests and all 65 procedural-fit/learning tests passed. `git diff --check` also
passes; its output contains only the repository's existing LF-to-CRLF notices.

The 2026-08-26 Project-root migration passed all 119 focused pipeline-model,
pipeline-UI, and analysis-settings tests, plus the broader 233-test graph/UI/
project/reference/visualization group. A full-suite run was also attempted, but
the open desktop workload had the 8 GiB GPU at 99% utilization with about
7.9 GiB allocated; the real-image dense-lupin pilot continued to make forward
progress through existing directional-noise and boundary-tracing CUDA stages
but could not finish in a reasonable hand-off window. Timed stack traces showed
no wait in Project routing, persistence, annotation loading, or raw-image cache
invalidation. The only completed-suite failure before that contention point was
the established missing `images/IMG_9689c.JPG` fixture.

## Learned instance-segmentation implementation

Work continues on branch `codex/learned-instance-segmentation`. Two independent
disabled-by-default active-DAG nodes now provide native PyTorch learned
segmentation:

- **U-Net + watershed instances** predicts interior, physical boundary,
  apparent/non-physical coat-pattern boundary, centre plus normalized distance,
  and auxiliary dense-error probability. Complete instance masks derive their
  physical contour and sparse safely inset internal edge candidates directly;
  the intervening uncertainty band and flat interior are invalid for the
  non-physical-boundary loss. Its pattern-aware marker watershed can also
  consume applied painted instance IDs.
- **StarDist seed instances** predicts object probability, checkpointed radial
  distances, and auxiliary radial-error probability, with spatially indexed
  polygon NMS and local rasterization.

Both nodes use self-describing checkpoints, canonical seed-scale normalization,
overlap-blended CUDA inference, separate forward/decoder caches, typed controls,
node-owned overlays, and explicit missing-checkpoint warnings. Checkpoint
binaries under `models/` are ignored and neither node is enabled automatically.

The new `seedvision/learning/` package includes manifest/annotation I/O,
content-addressed target caches, exact StarDist target construction,
deterministic augmentation, training/early stopping, quantitative instance and
dense-head metrics, validation-only decoder search, frozen test evaluation,
comparison/contact-sheet rendering, unlabelled fixture review, and an
experimental U-Net-gated StarDist decoder. Applied complete masks can be
saved and loaded at full corrected resolution, then exported from the Learning
menu with explicit group, split, revision, author, and review metadata. The same
menu audits datasets and launches new or checkpoint-refinement training in the
serial CUDA worker, with epoch progress and optional checkpoint activation.

The controlled 72-image/1,474-instance synthetic engineering set produced a
locked-test U-Net F1 of 1.000, PQ 0.820, count error 0, physical-boundary F1
0.989, and pattern-boundary F1 0.839. StarDist locked-test F1 was 0.996 and PQ
0.825, but validation and real-image false-object behaviour was substantially
worse. These are not scientific results.

Visual review on all eleven real fixtures shows that simulator-only weights do
not transfer: neutral-species U-Net counts are severely low except on sparse
`IMG_9670c` (15 predicted versus 16 visibly present), while StarDist strongly
overpredicts pattern/rim/background objects. The repo still has no complete
human-reviewed real instance masks, so no real accuracy can be calculated and
publication readiness cannot be claimed. See
`docs/LEARNED_INSTANCE_SEGMENTATION.md` for the protocol and
`docs/LEARNED_SEGMENTATION_RESULTS.md` for exact metrics, timings, fixture-by-
fixture visual findings, and the required real-data path.

## Procedural separation outcome and next approach

The complete procedural proof of concept is implemented, but visual QA says it
is an annotation bootstrap rather than a validated counter. It usefully
separates pale round seeds and gives plausible dense round-seed partitions, but
strongly bicoloured elongated lupins still show obvious false splits at coat
transitions and merges at contacts. The sparse `IMG_9670c.JPG` dish visibly has
16 seeds and receives 18 automatic instances. Automatic fixture counts are
diagnostic only: `IMG_0002c` 621, `IMG_9632c` 130, `IMG_9636c` 92, `IMG_9641c`
90, `IMG_9666c` 129, `IMG_9667c` 138, `IMG_9668c` 84, `IMG_9670c` 18,
`IMG_9685c` 116, `IMG_9689c` 58, and `IMG_9974c` 563.

After vectorizing label statistics, the procedural node takes approximately
0.41–0.87 seconds on the RTX 3070 for the tested 856–1758 px dish crops; the
complete active analysis takes approximately 2.1–6.3 seconds. Further manual
threshold tuning did not resolve the core ambiguity because true contacts and
within-seed coat boundaries can have equally strong edges. The next justified
step is to review masks produced by this node and train a small
species-conditioned boundary/instance model (U-Net plus watershed or StarDist).
Do not start with a transformer: eleven unlabelled fixtures do not support it.

## Net reference-edge subtraction and prototype diagnostics

The Reference edges node owns a raw cached diagnostic margin,
`max(Pphysical - weight * Pnonphysical, 0)`, and a distinct authoritative
Reference-edge probability formed by multiplying Pphysical by the generic
full-resolution thinned true-edge support. The supported margin is separately
available as **Conservative net physical-edge evidence**. Its **Non-physical subtraction
weight** is the first full inspector control and first inline control (default
0.5) in the **Optional conservative net evidence** section. Assisted Smart/Shape
fill can explicitly choose either supported field; the normalized probability,
canonical ridge, and default downstream reference input use Physical probability
without the extra subtraction. Changing only the weight reuses upstream learned
prototype banks and cannot change authoritative or normalized probability.

Reference texture prototypes also publishes a **Reference prototype source
footprints** overlay. Material markers show the retained medoid pixel and its
context radius. Edge markers show the three tangent-aligned interior/centre/
exterior lines and the five actual bilinear samples on each, with dot sizes
encoding the 1:2:3:2:1 pooling weights. The prototype collage and inspector
help now state explicitly that thumbnail patches are presentation context, not
fitted templates, and that no filled region between the strip guides is
matched.

## Overlay and normalized-reference-edge cleanup

The foreground binary proposal viewer mode was removed: it rendered the old
baseline foreground mask rather than the material-decision node's own product
and had no active consumer. The Background/Other subtype ambiguity and unknown
maps were valid computed products but were missing from the image viewer's
renderable-mode registry; they are now normal probability overlays with their
original node ports and legends. **Boundary confidence and normals** remains a
dormant legacy toolbox diagnostic. It is disabled and shelved by default with
no active segmentation consumer, so its bypass maps are intentionally zero;
the node details and overlay legends now say so explicitly.

The separate Perimeter background reference node has been retired. Layout
detection now owns its two annulus controls, typed output port, overlay, status,
cache unit, and downstream connections. Settings-profile format 10 migrates the
old node's parameters and rewires its output to Layout detection. The Reference
edges viewer calls the continuous product **Normalized reference-edge
probability** and no longer exposes raw prototype compatibility as an
assisted-tool source. Supported and normalized products share the same blue
scale for direct comparison.

Normalized reference-edge support is one-sided: local gain is clamped to
`[1, maximum_gain]`, so normalization cannot make above-floor ridge support
weaker. The gain denominator is capped at current pixel support so a strong arc
inside the seed-scale window cannot block amplification of its immediately
adjacent weak continuation. The smooth absolute gate reaches full weight at
the configured floor, and high/low ridge hysteresis—not local intensity
leakage—decides whether an enhanced weak maximum belongs to a coherent ridge.

The 2026-08-27 focused graph/UI/persistence/material/cache/edge run passed all
209 tests. Full discovery ran 524 tests in 307 seconds: 521 passed, 2 expected
environment/platform tests were skipped, and the sole failure remains the
pre-existing committed deletion of `images/IMG_9689c.JPG` while the reference
manifest still names that source fixture. `git diff --check` passes apart from
the repository's existing LF-to-CRLF notices.

## Material-prototype probability calibration

The Reference texture prototypes node no longer treats its Gaussian-kernel
similarities as posterior masses. That interpretation gave an ordinary
one-robust-scale in-class descriptor a score near `exp(-0.5)` and then divided
it again by all class scores plus unknown mass, creating a second artificial
middling ceiling. The new global calibration separates unsharpened
known-material confidence from relative class competition, with an exposed
**Material class contrast** default of 4.0. Uniformly weak matches remain
unknown and exact Background/Other ties remain ties.

The transform receives only score rasters and the valid-image mask, never
reference masks or coordinates. A regression test gives identical score
vectors at two nominal locations and requires bit-identical output, alongside
tie and uniformly-weak cases. On the saved `IMG_9405.JPG` sidecar, painted Other
pixels changed from mean/median Reference Other-material probability
0.536/0.541 to 0.775/0.828; competing Foreground and Background means there are
0.087 and 0.032. Painted Foreground and Background target means are 0.810 and
0.706. These are learned global responses, not hard-written reference values.
Settings format 11 persists the contrast control and version 10 or older
profiles adopt the current 4.0 default. The prototype cache signature includes
the control.

## Edge-prototype probability calibration

Physical-edge and Non-physical-edge prototype similarities now use the same
two-stage, mask-blind calibration as material prototypes. The former direct
`score / (physical + nonphysical + 0.10)` conversion treated Gaussian-kernel
similarity as posterior mass and imposed an artificial middling ceiling.
Unsharpened strongest similarity now determines known-edge confidence, while
the exposed **Edge class contrast** (default 4.0) affects only relative class
competition. Exact ties remain ties and uniformly weak matches retain mostly
unknown mass. The transform receives descriptor scores and a generic valid
strip mask, never annotation targets or coordinates, so reviewed pixels cannot
be hard-written. The leakage-safe held-out-instance edge fitter includes the
new contrast in its bounded optimization. Settings format 12 persists it;
version 11 and older profiles adopt 4.0. Prototype cache identity includes both
material and edge contrast controls.

The focused graph/UI/persistence/material/edge/cache/pilot run passed all 216
tests in 190 seconds. Full discovery ran 527 tests in 289 seconds: 524 passed,
2 platform/environment tests were skipped, and the sole failure is still the
pre-existing missing `images/IMG_9689c.JPG` fixture referenced by the committed
instance-reference manifest. `git diff --check` reports only the repository's
existing LF-to-CRLF notices.

## Material-noise class-balance redesign

Material-noise classification now uses nine fixed, texture-only descriptor
channels: residual RMS plus principal- and cross-axis local variation at fine,
medium, and coarse scales. This adds oriented multiscale structure without
creating a second general-purpose prototype bank alongside Reference texture
prototypes. Each class is now a target-only compatibility model: the robust
target distribution and its 95th-percentile positive distance define the
global half-support distance, and no semantic counterclass enters another
class's calibration or pixel score. Cross-class reliability and all semantic
contrast occur exclusively in Material evidence decision. Reference pixels
remain training examples and are never hard-overridden.

Cross-class invariance regressions change sibling annotations to genuinely
different texture types—not merely duplicated pixels—and require bit-identical
Foreground, Background, and Other raw rasters while still requiring a changed
target reference to change its own result. The saved `IMG_9405.JPG` audit now
reports painted-Other mean/median Other-texture compatibility 0.892/0.918 and
Foreground-texture compatibility 0.455/0.486. Painted Background can
legitimately score as both Background (0.947 mean) and Other (0.655 mean),
while painted Foreground scores 0.793 mean as Foreground and 0.094 mean as
Other. These are learned target-only responses, not mask-written values.

All 65 visualization-layer tests, 139 focused material/pipeline/UI/settings
tests, and all 19 real-image pilot tests pass after the target-only change. The
pilot run took 513 seconds under simultaneous desktop GPU load. After removing
the obsolete counterclass fields from the public texture profile, final full
discovery ran 534 tests in 414 seconds: 531 passed, 2 expected tests were
skipped, and the sole failure remains the pre-existing missing
`images/IMG_9689c.JPG` source still named by the committed reference manifest.

## Semantic reference-seed traits

Reference seed instances can now carry one optional species-specific,
mutually-exclusive coat-pattern label plus independently overlapping condition
labels. `Lupinus mutabilis` currently exposes White, Banded light, Banded dark,
and Other coat patterns; the shared condition vocabulary is Immature, Split,
Wrinkled, and Stained. A separate Reviewed control distinguishes an explicitly
sound/absent condition example from an unreviewed seed, so unchecked conditions
are never silently used as negatives. The selected seed's controls live in the
existing seed-instance editor and are enabled only after that seed ID has a
painted mask.

Reference-region format 3 stores the semantic records and their species
vocabulary as strict scalar JSON metadata beside the full-resolution uint16
instance raster, preserving `allow_pickle=False`. Versions 1 and 2 still load
with empty semantic labels. Orphan IDs, malformed identifiers, duplicate
conditions, and positive conditions without explicit review are rejected. The
project/reference association panel reports saved coat and condition-review
counts. Semantic-only Apply + save invalidates only the new diagnostic branch;
painting or deleting instance pixels retains the established full annotation
dependency behavior.

The terminal **Reference seed traits** node learns separate image-local Lab,
multiscale texture, edge/ridge, and local-residual prototype banks from safely
inset labelled seed material. Coat probabilities compete only with one another
and are renormalized to sum to one on the resolved seed-material domain when at
least two classes have support. Each condition is a separate
reviewed-present-versus-reviewed-absent probability and may overlap other
conditions. Missing calibration yields black/unavailable output. The node only
consumes the current resolved seed mask and has no downstream connection, so it
cannot alter Background, Foreground, Other, or physical/non-physical edge
calculations. Painted semantic labels train the models but never overwrite
their output pixels. Settings-profile format 14 adds the node; version 13 and
older profiles adopt its terminal default configuration and connections.

Clean-process verification covered all 544 tests: 541 passed, 2 expected
fixture/platform cases skipped, and the sole failure remains the established
missing tracked `images/IMG_9689c.JPG` file named by the committed reference
manifest. All 19 real-image pilots and the semantic archive/editor/cache/
normalization regressions pass. A monolithic discovery process again exhibited
the documented retained-state slowdown after earlier suites; clean process
groups completed normally. `git diff --check` reports only the repository's
existing LF-to-CRLF notices.

## Signed probability-excess diagnostics

The combined **Material colour probabilities** and **Material noise
probabilities** nodes now each publish a display-only signed comparison of raw
Foreground evidence with the strongest raw Background/Other response. For
each pixel, `d = Foreground - max(Background, Other)`; blue is `max(d, 0)`, red
is `max(-d, 0)`, and a tie is black. The comparison deliberately neither adds
the two independent Non-seed compatibilities nor normalizes the three raw
classes. It reuses the existing lazy rasters only when selected and creates no
new calculation, cached tensor, graph dependency, or downstream evidence.

**Reference edges** similarly publishes **Physical vs non-physical edge
excess**, an unscaled direct margin with Physical blue and Non-physical red.
It complements the overlap-preserving magenta-capable comparison and is
deliberately independent of the separate weighted/clamped Net physical-edge
calculation.

All 112 visualization/model calculation tests and all 64 pipeline-UI tests pass
in clean processes. Repository-wide clean-process coverage completed every
test except the two-pass dense-lupin pilot, which remained compute-bound while
the host GPU was externally saturated at 99% and 7.9/8.0 GiB; it was stopped
without an exception. The other 18 pilot methods pass separately. Across the
544 uniquely completed tests, 541 passed, 2 expected platform tests skipped,
and the sole failure remains the established missing tracked
`images/IMG_9689c.JPG` fixture named by the committed reference manifest.
`compileall` passes and `git diff --check` reports only the repository's
existing LF-to-CRLF notices.

## Edge-supported reference-edge authority

Raw Physical, Non-physical, and weighted-net edge descriptor responses are now
explicitly presented as **prototype compatibility** diagnostics. They remain
available for inspecting how the strip descriptors classify context, but no
fill or procedural boundary path consumes their spatially broad fields.

The canonical full-resolution product is now exactly
`thinned_true_edge_support * Pphysical`.
The Reference edges node exposes this as **Reference-edge probability** and
receives its true-edge ridge through an explicit typed graph connection. Its
normalized and hysteresis-thinned derivatives are masked back to the exact
full-resolution true-edge footprint after resizing, so descriptor and
interpolation halos cannot create barriers. Assisted fills map their persisted
`net_physical` source to this supported field, and procedural separation now
receives the canonical product through its own typed input instead of
reconstructing an unweighted raw-class difference. Curve and learned branches
that require separate edge classes receive edge-supported Physical and
Non-physical compatibility maps.

The previous lambda-based supported margin remains a separate optional product,
**Conservative net physical-edge evidence**, selected explicitly by assisted
fills with `conservative_net_physical`. The existing `net_reference_ridges`
trace source now names its actual conservative ridge, while `reference_ridges`
and the normalized branch use authoritative Physical probability. Existing
normalized storage/port IDs retain their legacy `net` spelling for saved-graph
compatibility; their UI labels say Reference-edge probability. Neither the
prototype classifier, unknown confidence, annotation masks, nor fill growth
algorithm is changed by this split.

Settings-profile format 15 adds the true-edge-support input and authoritative
procedural-edge connection; version-14 and older profiles adopt both current
connections during migration. Formula, zero-off-ridge, cache, graph, migration,
fill-source, procedural, visualization, and UI regressions pass. Full discovery
ran 546 tests in 315 seconds: 543 passed, 2 expected platform/environment tests
were skipped, and the sole failure remains the pre-existing missing
`images/IMG_9689c.JPG` fixture referenced by the committed instance-reference
manifest. `compileall` and `git diff --check` pass; the latter reports only the
repository's existing LF-to-CRLF notices.

## Annotation-guided Reference edges fit application

Starting the Reference edges fit now explicitly confirms a fit-and-apply
operation. A genuinely improving best proposal is applied directly to the live
node when the worker completes; the former second default-No question no longer
silently discards a completed proposal. The node is selected and its inspector
controls are rebuilt from the applied values, while the status bar enumerates
every old-to-new setting change. The application log records the image, loss
change, and complete fitted-value mapping.

Applying these controls now also invalidates the combined Reference texture
prototype cache, which currently owns the raw physical/non-physical class-map
calculation, plus all of that producer's dependents. This prevents updated live
settings from being paired with resident pre-fit prototype rasters. A UI
regression covers automatic application, live node values, producer-cache
invalidation, and recomputation dispatch. Fit progress reports best-so-far loss
rather than whichever candidate happened to run most recently. The 10 focused
fit/inspector/application tests pass. All 528 non-pilot tests ran in 202 seconds:
525 passed, 2 expected platform/environment tests skipped, and the sole failure
remains the pre-existing missing `images/IMG_9689c.JPG` fixture. A monolithic
discovery run was stopped after roughly 29 minutes in the already documented
retained-state real-image pilot slowdown; it was still CPU-active, several
pilots had completed, and no additional failure marker had appeared.

## 2026-08-31 authoritative Physical probability and conservative evidence

Reference-edge probability is `R * Pphysical`, where `R` is the existing
full-resolution thinned true-edge support. Its normalized and thinned
derivatives use the same Physical probability and are bit-identical when only
the Non-physical subtraction weight changes. The classifier still retains its
known-versus-unknown confidence; no conditional ratio or second subtraction
is applied to the authoritative probability.

The previous `R * max(Pphysical - lambda * Pnonphysical, 0)` is retained as
**Conservative net physical-edge evidence**, with a separate blue overlay,
typed output, Smart/Shape-fill source, and thinned ridge. Trace sources can
select the conservative ridge explicitly. Both products are exactly zero off
the true-edge footprint. The canonical probability shares its lazy GPU raster
with the already supported Physical class input. Raw prototype compatibility,
annotation handling, and the fill growth algorithms are unchanged. Stable
graph/tool IDs preserve existing settings profiles without a schema bump.

The full suite ran in two clean process groups: 530 non-pilot tests in 179
seconds and all 19 real-image pilots in 99 seconds. Across all 549 tests,
546 passed, 2 expected platform/environment cases skipped, and the sole failure
remains the pre-existing missing `images/IMG_9689c.JPG` source named by the
committed reference manifest. Formula, lambda-invariance, cache reuse,
zero-off-ridge, source-adapter/fallback, overlay-port naming, UI, fit, and
settings-persistence checks pass. `compileall` and `git diff --check` pass.

## Layout vessel selection

Layout detection's inspector begins with a **Seed vessel type** dropdown pinned
to **Glass Petri-dish**. This disabled, single-option capability indicator is
deliberately separate from computational parameters: other vessel/surface
detectors are not implemented, and opening the inspector cannot change settings
or invalidate analysis caches. The existing Petri-dish calculations, profiles,
and geometry are unchanged. A Qt regression covers ordering, locking, node
switches, and absence of parameter-change emissions.

Verification: both focused inspector tests pass, and the rendered settings panel
was checked with the application's Windows font. Full discovery in clean
non-pilot/pilot processes ran all 550 tests: 547 passed, 2 expected tests skipped,
and the only failure remains the missing `images/IMG_9689c.JPG` manifest fixture.
All 19 real-image pilots pass. Compilation and `git diff --check` pass.

## Species libraries and reference dimensions/shape implementation

`docs/SPECIES_REFERENCE_LIBRARIES.md` and
`docs/REFERENCE_SEED_DIMENSIONS_AND_SHAPE.md` are now implementation records,
not unstarted proposals. One shared immutable library framework publishes and
resolves source-balanced foreground colour/noise, material prototypes, aligned
edge prototypes, seed traits, shape summaries, and joint dimensions/shape
banks. Projects pin an exact ID/version/content hash; runtime assembly excludes
the current source; missing pins offer exact bundle import and never follow
latest. The manager supports build, grouped validation, publish, fork,
import/export, retirement, and pinning.

Shape reference metadata, robust 2-D measurements, coherent-boundary
uncertainty, physical-seed repeated-view grouping, pose-conditioned contour
families, species/lineage/accession/lot partial pooling, overlays, and opt-in
geometry consumers are implemented. Uncalibrated masks contribute only
dimensionless shape. The legacy scalar diameter is preserved as a compatibility
adapter. Intrinsic 3-D recovery remains explicitly deferred pending paired views
or physical thickness data.

Library extraction now calls the production descriptor seams and uses bounded
source/seed-balanced compact transfers rather than approximate CPU features.
Descriptor schema IDs were advanced so older approximate artifacts cannot be
silently interpreted as compatible. The focused library/shape/persistence/
pipeline/UI/procedural/visualization regression set passes 264 tests. After the
project-schema expectation was updated, final full discovery ran 565 tests in
287 seconds: 562 passed, 2 expected tests skipped, and the sole failure is the
established absent `images/IMG_9689c.JPG` source named by the
instance-reference manifest.

## Procedural centres, reference-error diagnostics, and annotation/library UX

Procedural centre likelihood now consumes the proposal-independent,
fit-validated oval-centre probability from Seed-boundary confirmation. The
editable positive-only weight defaults to 0.85 and probabilistic-union fusion
can boost a retained oval centre without suppressing the existing material,
interior-depth, and flattened-grayscale fallback. The raw centre-vote field,
raw peak markers, and fit-validated oval-centre probability have explicit names
and remain separate diagnostics. The graph dependency is typed, cached, and
causes only the procedural branch and its dependents to be invalidated.

Procedural seed separation also exposes **Reference underreach / overreach
cost**. After automatic inference is complete, annotated and predicted labels
are globally matched one-to-one above a minimum IoU, with explicit unmatched
choices. Blue represents underreach; red distance-weighted overreach; amber
missed references; magenta incorrect concavity pockets. The exact same pixel
cost implementation serves fitting and display. The annotation raster is never
passed into centre discovery, watershed, candidate selection, shape filtering,
or confidence. Shared matching and cost controls affect the fitting objective
and post-inference comparison, never the automatic prediction directly.

The seed-trait editor no longer permits the invalid combination that caused the
opaque project-save failure: a seed cannot be included in shape modelling while
Outline or Pose is Unknown. Outline and Pose now precede the inclusion control,
an inline status explains eligibility, and old invalid in-memory records remain
editable. Persistence errors name the exact seed and repair action; project-save
errors explicitly state that saving is paused and no in-memory data was lost.

The species-library manager is reorganized into three numbered workflows:
**Use a library**, **Create a version**, and **Inspect selected version**.
Pin/import/export/retire actions, reviewed source selection, optional biological
context, forked-source retention, immutable publication, product coverage, and
validation are now grouped and explained where they are used. Selection counts,
dynamic action labels, fork requirements, tooltips, and build status are visible
without switching between disconnected tabs. Analysis-settings format 17 adds
the new procedural controls and connections; version 16 and older profiles
adopt authored defaults.

## Annotation and shape-controls revision — 2026-09-05

See `docs/ANNOTATION_SHAPE_AND_EDGE_REVIEW_2026_09_05.md` for the rationale,
measurement semantics, compatibility notes and punctate-edge classifier audit.
Detected ruler now also shows the independently measured imperial span. The
annotation panel has a draggable title bar and resize grip; the tool-settings
stack sizes to the active page (hidden fill pages caused the large brush gaps).
Existing-ID selection, a red empty marker, adjacent selected-only control,
exclusive No defects versus defect labels, and an on-image hilum target/vector
editor are implemented. Physical seed ID is no longer exposed but old values
are preserved.

Maximum span uses all explicitly complete or full-length-visible annotations,
never a top fraction. Partial full-length masks contribute size only, not shape.
Complete reviewed masks additionally supply mean internal concavity and signed,
seed-balanced turning/curvature distributions. A typed compact summary feeds
soft trace/boundary priors; upstream gradient/ridge caching is preserved.
The Reviewed measurements, size–ovality, uncertainty and new curvature overlays
now explain their quantities. Uncertainty is sensitivity, not a validated CI.

Fixed the reversed major-axis comparison in OpenCV ellipse-angle conversion
(it rotated bodies 90 degrees and corrupted robust-fit residuals). Added inward/
outward perturbations to uncertainty, raster-consistent convex-hull concavity,
and cropped per-seed morphology. Shape-library descriptor schemas are now v3;
old shape banks are disabled with a rebuild warning, while other products stay
usable. Old shape observations are not silently promoted during fork recovery.
Reference archives are v5 (v1–v4 still load), analysis settings v18 (retired top-
fraction parameter migrates away). Existing saved source data is not overwritten.

`tests/test_annotation_shape_revision.py` covers the new UI, geometry, migration,
no-mask hilum editing, shape-bank compatibility, and curvature/cache contracts.
Generated UI screenshots are only in ignored `artifacts/`.

Validation: final full discovery ran 585 tests in 273.069 seconds: 582 passed,
2 skipped, and the sole failure remains the pre-existing missing manifest image
`images/IMG_9689c.JPG`. Compileall and scoped diff whitespace checks passed.

## Procedural reference assignment and cost repair — 2026-09-05

See `docs/PROCEDURAL_REFERENCE_MATCHING_AND_COSTS.md`. Replaced greedy raw-area
matching with component-local global IoU assignment and unmatched choices.
Incidental contacts no longer turn neighbouring unreviewed seeds into costly
false correspondences. Missed references are now visible and have a lower
default pixel cost (0.5). Incorrect exterior-connected candidate-concavity
pockets incur a +2 surcharge; correct natural indentations and enclosed holes
do not. Pixel loss uses fixed reviewed area, so omission weights actually affect
the score. Overlay and fitter share summed pixel costs, with blue/red/amber/
magenta diagnostics and a fixed cost-4 alpha scale. Cost knobs cannot be fitted
away and cannot alter automatic predictions directly. The existing fit-action
controls now alias the node's persisted values; settings schema 19 adds the
new matching, missing, concavity, and coverage defaults.

Validation: 135 focused tests pass. Full discovery ran 595 tests in 268.691
seconds: 592 passed, 2 skipped, and the only failure is the pre-existing missing
`images/IMG_9689c.JPG` manifest fixture. Compilation, diff whitespace checks and
CUDA runtime diagnostics pass. The synthetic cost-overlay contact sheet was
visually checked and is retained only in ignored `artifacts/`.

## First actions for the next agent

Read `AGENTS.md`, inspect `git status`, run diagnostics and the test suite, then
continue from the user's newest request. Preserve unrelated user changes and
do not commit generated artifacts or a local virtual environment.

## Automatic-card preprocessing and shared adaptive-mean checkpoint (8 October)

The dependent conformance/register/working-recipe/Qt/extended-threshold gate also
passes all32 tests in89.823s (`artifacts/card-preprocess-compatibility-gate.log`).
All8 current source/test hashes and the previous5 desktop/11 batch hashes remain
unchanged; scoped diff whitespace checks pass. The full implementation goal stays
active, and fresh full unittest discovery remains required at completed handoff.

CUDA automatic-card preprocessing is implemented for BGR grayscale, exact 11-tap byte Gaussian blur and inverse adaptive mean/Gaussian kernels3..255. Only SHA-verified compact Gaussian taps use CPU OpenCV; all raster filtering stays on CUDA. Non-dyadic Gaussian binary decisions require outward exact-rational error certificates. Shared adaptive mean now checks scalar-double, SIMD-float32 and SHIFT23 arithmetic, including full-frame singleton-axis reduction. Backend-sensitive binary decisions explicitly withhold output. The actual pinned public detector is executed only through its preprocessing boundary in this gate; complete contour/card geometry, automatic correction recipe/API/Qt integration and scientific colour validation remain open. Focused final gate: Ran 57 tests in 191.942s; bounded independent review covers supplied computational/guard projections and root execution evidence only. Fresh full-suite verification remains required at completed handoff.

This is a progress checkpoint; F03.2 remains partial/pending/unvalidated.
The final gate is artifacts/card-preprocess-reviewed-gate.log. Earlier gates:
12 tests/10 passes had two fixture expectation errors (one certified random
mask and one genuinely uncertain public fixture); repaired13 pass43.693s.
Independent review then reproduced the scalar-tail mean bug. The first shared
repair gate ran55 tests/54 passes183.598s; its only failure incorrectly included
an executed upstream median node in the descendants expected to be skipped.
The final gate includes the repaired assertion, fresh recovery, singleton
full-frame dispatch, fractional offsets and both polarities. No full-suite or
complete detector/scientific claim is made. Root owns all I/O; the same three
text-only workers were reused. Routine project-edit authorization persists;
auto_review is active, direct patches and scoped root execution work. No
settings change or renewed approval is required for normal workspace edits.

Resume with a contour-only CUDA boundary collector and compact TREE/simple
geometry, then the complete automatic detector and additive working recipe.
The existing sparse tracer retains every transition (CHAIN_APPROX_NONE), so
CHAIN_APPROX_SIMPLE still needs an ordered direction-compression pass. Preserve
strict source area/aspect/solidity and median filters; transformed grid centres
(not original rectangle centres) must each map strictly inside one chip.
Do not clamp minAreaRect dimensions. Canonical peer source manifest remains
27f90c3507a01a3cabc00bd934cbbab8e28d95aa91387ce876775850d5013854.
Original comparison runtime is OpenCV4.11.0; root runtime is OpenCV5.0.0.
All127 compact sigma-zero CV32F kernel identities are byte-verified.

- `seedvision/cuda/plantcv_card_preprocess.py`: `0bbf28ad0c9d4fc88d4de07ca7a415421dd910847c166118f6c4c3526bf4127a`
- `seedvision/cuda/plantcv_card_kernel_pins.py`: `53fac5b83029e43678c24f0165f9afb9171f88870861ef31388002f87c08607e`
- `seedvision/cuda/plantcv_thresholds.py`: `de85e85e969267e81957875e681323f4733a9b548d5c222cc4dee9a2dadcb164`
- `seedvision/benchmarking/native_threshold_stage.py`: `00fe1771a6de2989147da5623001d973b59255efe2bafba0c82f748e9b8f0d91`
- `tests/test_plantcv_card_preprocess.py`: `ec7c06a3af98cdd3c7f9fe632ca15e927a35159af913725cca29f46a1a92e7b8`
- `tests/test_plantcv_thresholds.py`: `df2082567e0b2a3483200925def243469738a2317047cec1bba7d16c0f7372e9`
- `tests/test_native_threshold_stage.py`: `2082d114c5bf8458c04eb747825fe7fbdb47a92f1c7f74acf586d7705932d825`
- `tests/test_native_recipe.py`: `f5682ec301360b27badd4ac7ddc681912cd2a899911eb3165929759de7135f75`

## 8 October — automatic card geometry and affine primitives

Final checkpoint audit: all49 numerical tests pass in147.705s; after documentation
updates all6 feature-register tests pass in0.527s. The scoped diff/whitespace check
is clean. All6 new and8 prior preprocessing source/test hashes remain frozen,
and the canonical peer manifest hash is unchanged. The existing reviewer gives
bounded approval for the6 supplied frozen files and root evidence, with no
independent file reads or execution. No tests/processes remain live. Next is the
additive automatic-card working recipe and its verified helper closure, then
native Qt/project/batch/export integration; the full goal remains active.

Automatic colour-card CUDA primitives are implemented in bounded supported
domains in `cuda/plantcv_card_detection.py`. RETR_TREE/SIMPLE contours come from
the new contour-only CUDA boundary collector; no dense CPU image or mask is used.
The collector preserves cyclic direction changes, repeated visits, source order
and hierarchy, with distinct empty, unsupported, resource and cancellation cases.
Sparse topology retains the existing point/row/component/trace/work limits; its
Python state is not claimed to fit the packed boundary byte count. CUDA extraction
scratch is preflighted at 256 MiB and tested on an actual custom stream.

The detector preserves strict contour area/aspect/solidity and median-area filters,
white-corner ordering, source int32 transform WITHOUT a perspective divide,
strict transformed-center membership, inclusive clipped bounding rectangles and
ordered overlapping circle labels10..240. It retains the source case where23
detected contours still generate24 grid masks. ROI bounds are explicitly half-open
xyxy. Interior bounds are compared with actual `pcv.roi.rectangle` effective crops;
frame-edge bounds are compared with rectangular source Objects directly. Named
or custom physical chip scaling is returned as metadata without changing global
application units. Reusable filters and full float64 labels/uint8 bounding masks
stay on CUDA; only sparse geometry, four sampled pixels and compact fit matrices
cross to CPU. Input/filter/output versions and final callback mutation checks
discard incomplete results; compact arrays are immutable bytes-backed views.

`cuda/plantcv_minrect.py` ports pinned4.11 rotating-caliper float32 operation
boundaries and `RotatedRect::points`, retaining last-equal-area selection and
source corner order without angle normalization. The loaded OpenCV5 min-area
rectangle sign convention and far-corner arithmetic caused actual label and ULP
failures; these were fixed rather than relaxing geometry comparisons. The fixed
4x6 grid box is pinned to its exact32 bytes of float32 metadata and SHA256
39c9018bf2b16570e4c2871891f031b76f5cdfb52131b1673b26e0672cec0ac6.
Actual OpenCV/NumPy runtime and reference-source identities are returned.
Loaded hull generation remains a runtime dependency with bounded reference tests.
Aggregate geometry preflight charges all incoming SIMPLE points and contour count
before candidate hulls, retained-centers geometry and24-center membership tests.
Circle/retained-storage/filter work limits precede full output allocation.

`auto_correct_color_cuda` uses the detected mask for chip means, standard matrix
pos=3 and the source affine fit, then corrects the whole image. It is separate
from the existing four-file polynomial correction workflow. Compact matrices are
immutable and corrected full-resolution BGR remains on CUDA. Pinned public
auto-correction tests require source means within2e-15, exact standard matrices
and at most one byte/channel corrected-image difference.

Focused final six-module gate: 49 tests PASS in 147.705s,
`artifacts/automatic-card-reviewed-gate.log`. This includes all512 binary3x3
masks, nested/seam/thin/random contour arrays and hierarchy,109 compact geometry
fixtures with exact hull/rectangle/corner comparisons, actual full public
detect/mask/auto calls, both adaptive methods, four right-angle rotations plus13
degrees, equal white-corner ties, named/custom scales, interior/edge/clipped ROI,
overlap and radius-zero labels,23-contour source behavior, numerical uncertainty,
resource preflight, cancellation/mutation including reusable filters/final masks,
changed grid metadata and actual custom-stream allocation.

Bounded independent approval uses supplied computational/guard projections,
frozen hashes and root execution evidence. The reviewer did not independently
read files or execute tests. This establishes bounded primitive functionality and
tested reference conformance, not scientific colour accuracy, arbitrary runtime
conformance, a loaded working-recipe certificate or complete feature parity.
Next work is additive automatic-card working-recipe/Qt binding, loaded callable
closure covering contour/minrect/grid/preprocessing/correction helpers, and shared
saved-project/batch/export integration. Existing four-file correction bindings,
historical catalogue identities and the canonical peer source manifest remain.
Full F01-F22/WP0-WP6 and fresh full unittest/scientific gates remain open.

Frozen source/test SHA256:

```json
{
  "seedvision/cuda/plantcv_contours.py": "e252e4e9bf0d94091088ff9ea449242095e7e8ed388b1e51d6927a8b664c580f",
  "seedvision/cuda/plantcv_minrect.py": "23839438d05046ca3e72eea492990ab630c207c49a561111125d97c15b680a24",
  "seedvision/cuda/plantcv_card_detection.py": "0722eb52da4d04b1475084ac5677a96e6c2873accd785d4255e2764a03086859",
  "tests/test_plantcv_contours.py": "12eeace8a5a2b34130c45cc7d6d8abc9d3f2aeee745e0b8e1144ab1147576d76",
  "tests/test_plantcv_minrect.py": "6e9f45e7955dea0e0da687005dbb4d4884d0fc57cc8cc6f49def90a668a908db",
  "tests/test_plantcv_card_detection.py": "c88c1771bb1ad274875fb0090219c0e53b1b63f4357762bf381aa7bffbead6b4"
}
```

## 8 October — automatic-card working recipe and desktop/project integration

`plantcv_automatic_color_card` v1.0.0 is an additive editable native graph.
RGB input converts once to reusable BGR; ROI cropping precedes grayscale,
Gaussian11 and inverse adaptive filtering. Separate geometry, circle/bounding
mask, physical scale, chip means, standard matrix and affine nodes preserve
source channel order. Standard position3 matches public auto-correction; other
positions explicitly compose the public standard matrix and affine functions.
Physical dimensions produce mm-per-pixel metadata without changing image pixels
or global application units. Radius changes reuse geometry; layout edits reuse
all analytical products. Compact geometry/matrices are deeply immutable finite
JSON tuples, while full images/masks/reusable filters remain CUDA tensors.

Loaded callable identity includes the card factory, detector, contour/minrect,
preprocessing/Gaussian pins, colour-card and correction helpers alongside the
shared runner closure. The exact immutable grid byte hash is verified separately.
Input versions are captured before the first callback; accepted cached/new
products are checked after callbacks, including the final callback. Mutations
discard the result/cache; failed or uncertain stages leave empty outputs and
unavailable descendants. Dirty descendants are evicted before replacement
allocation. The graph-owned32N+256MiB CUDA bound includes25N accepted products,
temporary corrected BGR, tile scratch and bounded contour extraction; old result
objects held independently by callers are outside cache ownership. Checks occur
before CUDA upload and direct-run allocation/runtime loading.

The reference overview offers Automatic colour card. The shared async canvas
supports effective controls, ROI-local/BGR previews, float64 chip-label colours,
compact scale facts, Keep/reopen and export. Project schema6 and the existing
batch/whole-run archive preserve exact working selections separately from
historical attachments, without inventing historical stage records. The canonical
peer source manifest and historical catalogue remain unchanged. Four-file
polynomial correction remains a separate workflow.

Root verification: 103 tests in 181.702s PASS,
`artifacts/automatic-card-working-compatibility-gate.log`; the final strengthened
integration gate passes 12 tests PASS in 26.869s,
`artifacts/automatic-card-working-final-gate.log`. Actual pinned public
default-Gaussian auto-correction and ROI calls require exact labels/bbox/corners/
centres/target matrices, source means within2e-15 and corrected image within one
byte/channel. Tests cover every exposed control branch, layout-only reuse,
altered cached masks/compact geometry, failure/recovery, first/final callback
mutation, cancellation, fixed-grid/code changes, pre-upload resource rejection,
custom-stream fresh/replacement memory, actual async Qt and saved project/batch/
export round-trips. Earlier20-test gate had18 passes and two fixture errors
(wrong Keep method and path/execution callback counting); both were repaired.
The repaired ten-test gate passed in19.469s before expansion. The final gate also
covers actual Gaussian uncertainty/recovery and mid-mask cancellation plus a
callback that mutates an input and raises; the result and cache are withheld.

Bounded independent review covers supplied computational/guard/dispatch/cache
projections, frozen hashes and root execution evidence. No independent full-file
reads or execution occurred. This is tested bounded functionality/reference
conformance, not scientific colour validation, arbitrary-runtime conformance,
complete per-stage replay or full feature parity. Full F01-F22/WP0-WP6 remains
active; fresh full discovery is required at completed handoff.

Current frozen source/test SHA256 (prior minrect/detector/shared-runner/UI freezes
are historical; no unmodified prior freezes are promoted):

```json
{
  "seedvision/benchmarking/native_card_recipe.py": "247e207f77afb816fc09866ca9b3a246b5c23ea5726492d4a51532711bfd86cb",
  "seedvision/benchmarking/native_recipe.py": "d450b628f21469cd0d04a364b961a7c235d33afdb9cf01e92da0ca75e072f61a",
  "seedvision/cuda/plantcv_card_detection.py": "5dc66f325b7421f8b8b9d8c707343de5ac1ea25718d6b9ddd0e9c7151a995e2c",
  "seedvision/cuda/plantcv_minrect.py": "4dea0ddd857343e7b08cdc3888dc64d0cd7be78d70d98a3556957ec92c83f0be",
  "seedvision/ui/native_recipe_dialog.py": "97fee23e677f5f1214558e6aefaeb6a1643589875ad254d4e9c413114b97ecb9",
  "seedvision/ui/reference_graph_dialog.py": "b09ce41d68104740d19da80c76b04ad18b1686a62574a055cd8088d918cc222e",
  "tests/test_native_card_recipe.py": "4d5ef223976c1247142f443002fb111e9d2215effa52b8648d9a2e8a9df6784d"
}
```

Post-review display verification: all18 card/shared Qt tests pass in38.226s
(`artifacts/automatic-card-working-preview-gate.log`). Actual1440x900 overview,
ROI-labelled-mask and scale views were rendered with Windows fonts and inspected.
The preview now fits its current label dimensions, preserving aspect ratio and
showing all24 chip circles. This display-only repair changes the shared UI hash;
the other six current source/test hashes remain unchanged. The103-test gate
predates the preview repair; the18-test gate includes the12 strengthened card
tests and all shared Qt regressions. Six register tests pass after reconciliation.
Full F01-F22/WP0-WP6 and fresh full-suite/scientific gates remain active.

Final display review binds `native_recipe_dialog.py` to
97fee23e677f5f1214558e6aefaeb6a1643589875ad254d4e9c413114b97ecb9.
The reviewer approved the supplied display projection and root evidence, without
independent reads/execution or dynamic-resize verification. Final register gate:
6 tests PASS in0.524s. Scoped diff whitespace checks pass with the existing CRLF
notice only; all7 current frozen source/test hashes and the canonical peer source
manifest remain unchanged. No test/preview processes remain live. Continue the
remaining helper/input domains and computational features, complete replay and
original/SMPTS integration; full-suite/scientific gates stay open. Routine project
edits remain authorized under auto_review; do not ask for the same approval again.
