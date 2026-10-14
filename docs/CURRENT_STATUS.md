# Current project status

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

Current evidence: artifacts/complete-software-focus-run-v3.json, artifacts/complete-software-repair-qc-v2.json, artifacts/complete-software-repair-root-visual-qa-v1.json, artifacts/complete-software-projection-review-v4.json, artifacts/complete-software-frozen-hashes-v6.json and artifacts/complete-software-documentation-update-v4.json. See [Benchmark reports](BENCHMARK_REPORTS.md) for existing launch-point actions and limits. The single seed_vision.py/native PySide6 architecture, CUDA residency, node-local cache and user-authorized full goal remain intact and active.
## Historical verification and planning

Prior records below keep their dated scope and nonpassing history; current predicates above supersede their former next-work statements.

## Prior V2 finite software reconciliation (historical)

39 of the 40 operation rows implement their declared guarded software domains. F09.2 remains an unresolved original SMPTS source question: no distance-transform stage is invented. PlantCV conformance is scoped to pinned witnessed domains; paper-port engineering evidence stays distinct from blocked original parity. Separately named native/enhanced methods do not acquire PlantCV identity. Scientific statuses are unchanged.

The final concrete gaps are implemented: desktop public-reference exception/cancellation and withheld-delivery QC; actual resource/version/cache/synchronization/peak/logical-transfer and correction-time reports; prespecified absolute and paired connected-group comparisons with complete failure/abstention/stratum accounting; and typed frozen Bayes/K-means model/training-evidence binding to audited access and known consumed image identities. Unknown source membership stays ineligible. See [Benchmark reports](BENCHMARK_REPORTS.md) for existing launch-point actions and exact limits.

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

Cancellation runs between evidence rows, fitting iterations and calibration candidates. Validated bounded frozen JSON is saved atomically; a failure or stop before commit preserves previous bytes. A late stop preserves valid saved output and skips activation. Explicit activation validates the artifact and graph transaction first, then invalidates only classifier/dependents. Serial Qt tasks guard progress/completion by identity; Stop, close and graph changes retain honest outcomes. Existing U-Net/StarDist training remains compatible. Usage and exact manifest fields: [Native seed-trait training](TRAIT_CLASSIFIER_TRAINING.md).

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

All6 register checks PASS1.209s,zero skips; scoped implementation closure is artifacts/polynomial-integrity-closure-v1.json. The separate one-shot candidate driver is artifacts/capture-polynomial-runtime-candidate-v1.py. Consult artifacts/polynomial-runtime-candidate-status-v1.json for execution state; candidate/diff remain unreviewed until explicit reconciliation. Do not restart a LIVE/terminal driver or edit source/tests/config/capture scripts while it is LIVE.

## F05 acceptance reconciliation (10 October, historical)

The current extended-threshold gate passes 49 tests in 218.422s, zero skips: 22 primitive, 8 recipe, 6 control/Qt and 13 saved-replay checks. Runtime/test files, 39 preceding freezes and fixed baselines remain unchanged. This is a focused gate, not full unittest or science.

The F05 contracts now reflect mask-only extended outputs, even Gaussian requests, source-coordinate Bayes masks of differing dimensions, guarded positive constant-channel roundoff covariance, and implemented directory/project/Qt/SVG paths. Historical hash-matched classifier gates retain their exact scope and execution freshness; they are not promoted to a fresh whole-pipeline gate.

The acceptance audit is docs/PLANTCV_SMPTS_ACCEPTANCE_AUDIT.json: 40 requirements are listed; F05 has 12 criteria: 9 scoped, 2 pending training reconciliations and 1 external scientific gate. The other 38 operation rows await detailed reconciliation. All 40 coarse statuses, latest runtime implementation and immutable acceptance baseline are preserved. Each scoped criterion states current/historical evidence, exact selected profile, domain, unmet predicate and dependency. No whole F05/domain or scientific completion is claimed.

Next: reconcile F01.1/F01.2/F02.1/F03.1/F03.2, then remaining registered requirements and explicit F05 domains. L01-L04 and original SMPTS source/license/dependencies remain external gates. Full F01-F22/WP0-WP6 stays active. Focused register results and closure are recorded in artifacts/f05-domain-reconciliation-register-final-v1.log and artifacts/f05-domain-reconciliation-closure-v1.json.

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

## Closed finite public seed acceptance and desktop layout checkpoint (historical)

Eight fixed actual public chains now retain exact deterministic inputs, masks, watershed locals/labels, size/color observations, histogram data, selected/effective controls and source/package/build identities. Binary light/dark, mean requested8/effective9 and Otsu complete. Explicit7 on nonempty/empty labels executes size but preserves the public color UnboundLocalError as partial. Default empty invokes/verifies watershed once and skips size/color. Gray RGB retains the float16 JSON color failure. Four cases complete,one is empty,three are partial; partial/empty attempts publish no complete manifest.

The immutable expected record SHA is5f84f7f8e8d9b71e6f0704a262c6002e4ed382a16cc13511566596b0cf87eb6a. Fresh tests reconstruct actual calls and never refresh expected evidence. Transient durations/UUIDs/checkpoint addresses and shuffled palette/display ordering are outside exact repeatability; full per-run evidence retains them. Actual BGR/RGB relation and sorted palette multiset remain checked/bound. Controlled-copy projection checks are normalization tests only. Prepared RGB/gray use one local OpenCV snapshot with actual_public_reader=False; no public-reader parity follows.

The preview now fits its actual widget on resize, keeps aspect ratio and clears retained state on image replacement/invalidation. Long paths and hashes yield horizontal space. Root inspected eight readable offscreen views at1440x900/1050x650: saved complete reopen, BGR/RGB products, running controls and a fresh genuine partial outcome matching the baseline. All requested dimensions and preview bounds hold. Installed-font override exists only in the ignored evidence driver; native application font/calculations/CUDA caches are unchanged.2605 QTimer ticks establish measured responsiveness for one run,not a latency guarantee.

Final compatibility29 tests PASS661.965s,zero skips, plus separate same-size replacement1 PASS0.497s:30 distinct passing IDs across two executions,not a single combined30-test gate. The earlier13-test run had12 passes/one width failure17.879s; the repaired13-test gate passes15.499s and overlaps final coverage. Runtime is unchanged after29; only the extra focused test was added. Bounded supplied-projection review approves the six-file freeze; no independent reviewer I/O/tests/visual inspection. Evidence: artifacts/public-seed-acceptance-frozen-hashes-v1.json, public-seed-acceptance-review-v1.json and public-capture-layout-evidence-v1/visual-evidence.json. All24 other preceding frozen files remain unchanged; the prior UI freeze stays historical and is explicitly superseded.

Forty operation statuses remain unchanged,48 native replay operations,13 public runner operations and148 referenced test files. All6 final feature-register checks PASS0.848s,zero skips; closure: artifacts/public-seed-acceptance-closure-v1.json. F01-F22/WP0-WP6 remains active. Next: actual public color-card corrected capture/acceptance, then broader method/input/default/resource domains. Original SMPTS, L01-L04 laboratory inputs, scientific validation and completed-goal full unittest remain open.

## Closed public reference capture reader and desktop checkpoint (historical)

The source-graph view now launches a separate public seed capture for exact selected historical binary, mean and Otsu recipes. Honest calculation controls create an immutable request. Public preprocessing, one watershed capture and independent size/color calls run off the UI thread. Captured locals remain artifacts of one call; native calculations, CUDA caches and historical graph records remain unchanged.

The complete-manifest reader verifies all ten typed checkpoints against trusted caller roots/digest, exact source/frame/recipe/measurement/stage/role identities, actual source pins and runtimes, label facts and unchanged size/color label contexts. It rederives the authored maximum-label or explicit positive loop bound. Numeric-array descriptors use the remaining aggregate budget before each load; metadata/codec/loader/public/process scratch and RSS are outside that bound. The local OpenCV decoder prepares RGB and color-derived gray explicitly with actual_public_reader=False.

Saved captures restore the exact selected recipe after verification. Worker generations and source stamps reject stale completion; Stop/close drain without stale previews. Actual observations/charts and compact artifacts appear as source facts, with original-pixel hashes and at most4Mi aggregate sampled preview pixels. Raw BGR is adapted only for Qt display. Cancellation/error before the recipe sidecar commit retracts only this attempt's exact hash-verified manifest; changed replacement files and individual checkpoints survive. After commit, the completed stored revision persists even if the UI discards a stale result.

Final compatibility58 tests PASS269.004s,zero skips:24 new reader/session/desktop checks and34 existing graph/persistence/entrypoint checks. Actual selected chromatic binary capture executes eight public calls; stored capture loading also exercises asynchronous Qt delivery. The initial8-test/setup-error run76.258s retains a genuine public color float16-observation JSON failure on gray RGB. The24-test run138.879s had19 passes/five bound-method fault-fixture failures; a test-only repair precedes the final green gate. These failed runs remain history. Bounded review approves the supplied five-file freeze/root evidence, with no independent reviewer I/O/tests/visual inspection and no root visual-inspection claim. Evidence: artifacts/public-reference-capture-frozen-hashes-v1.json, public-reference-capture-review-v1.json and public-reference-capture-compatibility-v3.log. All20 prior coordinator/capture/recipe/reader frozen files remain unchanged.

Forty operation statuses remain unchanged,48 native replay operations,13 public runner operations and146 referenced test files. All6 final feature-register checks PASS1.920s,zero skips; closure: artifacts/public-reference-capture-closure-v1.json. F01-F22/WP0-WP6 stays active; next is a finite actual binary/mean/Otsu whole-chain outcome matrix and representative complete/partial public Qt rendering. Broader input/default/resource domains, original SMPTS, L01-L04 laboratory inputs, scientific validation and completed-goal full unittest remain open.

## Closed public seed-chain coordinator checkpoint (historical)

The benchmark coordinator now binds prepared gray and RGB checkpoints to the exact source, frame and selected authored recipe before any public call. It reuses the five existing preprocessing bridges, calls watershed.capture once, then runs actual public size and color independently. Captured locals remain artifacts of that one watershed call. Size/color consume the original prepared RGB and content-verified published labels; their labels_context raster is explicitly an input context. Actual source observations, charts, raw BGR diagnostic and separate RGB adapter retain provenance.

Unselected n_labels uses int(public_labels.max()), exactly the authored rule. Explicit positive selections pass unchanged, even on an empty raster; actual public errors remain failures. Marker count, distinct positive final count and maximum label ID are separate facts. Default zero labels skip both analyses honestly. Successful individual checkpoints survive a partial result; only a completed required call/capture set publishes the final atomic manifest. Cancellation and changed inputs withhold publication; each independent analysis reloads the immutable original labels.

Validation has35 distinct passing check IDs across two executions: the35-test compatibility run had34 passes/one fault-fixture error in771.932s,zero skips, and the corrected fixture passes its focused1-test repair in84.939s. There is no single green35-test gate. The initial9-test PASS204.395s predates final guards/counters/new actual-empty checks. Actual public coverage is a selected binary eight-call chain plus empty default/explicit7 requests; existing preprocessing checks cover binary/mean/Otsu. Fault injection checks orchestration contracts and is not a public numerical oracle. Bounded review approves the supplied final two-file freeze and root evidence; no independent reviewer I/O/tests/visual inspection. Evidence: artifacts/reference-seed-chain-frozen-hashes-v1.json, reference-seed-chain-review-v1.json, reference-seed-chain-compatibility-v1.log and reference-seed-chain-repair-v1.log. All18 prior capture/recipe/reader frozen files and canonical peer manifest remain unchanged.

Forty operation statuses remain unchanged,48 native replay operations,13 public runner operations and145 referenced test files. All6 final feature-register checks PASS1.826s,zero skips; closure: artifacts/reference-seed-chain-closure-v1.json. Native application/Qt/cache behavior is unchanged. The typed manifest reader and asynchronous Qt capture view are next; mean/Otsu whole-chain and broader public method/input/default acceptance remain open. Full F01-F22/WP0-WP6, completed-goal full unittest, original SMPTS and L01-L04 laboratory inputs remain active/open; scientific validation is false.

## Closed public watershed capture checkpoint (historical)

A separate segmentation.watershed.capture reference operation executes the exact pinned PlantCV function once and captures its actual distance, coordinates, peaks, markers, negative distance, labels, BGR diagnostic and shuffled palette. RGB is a declared adapter of the same diagnostic. No public helper is replaced or intermediate recalculated. The ordinary labels-only operation and historical authored recipe definitions stay unchanged. Empty coordinates use an exact typed zero-length transport record; general NPZ raster rules remain strict.

Public source/runtime/helper/adapter identities are verified before and after the call. A malformed capture retains only independently copied, validated public labels and an unavailable reason; public-call errors propagate. Existing profiler/trace behavior and actual public RNG advancement are preserved. Cancellation withholds publication and cleans transport. Retained numeric arrays are admitted at42P+3 bytes and16P+3 elements with actual descriptor checks; metadata, validation scratch and public/process peak memory are outside that bound.

Final68 focused/compatibility tests PASS313.624s,zero skips, including15 new tests and53 existing runner/preprocessing/mask/stage checks. The prior13-test gate PASS21.376s predates final runtime/fallback-label fixes and overlaps these68. The exact four-file freeze has bounded reviewer approval from supplied projections/root execution; no independent reviewer I/O, tests or visual inspection. Canonical peer manifest and14 prior recipe/reader frozen files remain unchanged. Evidence: artifacts/reference-watershed-capture-frozen-hashes-v1.json, reference-watershed-capture-review-v1.json and reference-watershed-capture-compatibility-v1.log.

All eight register/planning mirrors now record this API checkpoint:40 operation statuses remain unchanged,48 native replay operations,13 public runner operations and144 referenced test files. All6 capture register checks PASS0.833s,zero skips; closure: artifacts/reference-watershed-capture-closure-v1.json. Full F01-F22/WP0-WP6 stays active. Next: source whole-chain coordinator, typed checkpoints and captured-artifact links to one public call, followed by asynchronous Qt integration. Broader public method/input/default domains, completed-goal full unittest, original SMPTS and L01-L04 laboratory inputs remain open; no scientific validation or whole-chain/desktop completion is claimed.

## Closed recipe acceptance checkpoint (historical)

Fresh session4085 is terminal/exit0: all4 tests PASS in2736.266s,zero skips. The immutable baseline verifies223 selected profiles (69 canonical,56 median ROI,48 reader,48 reader-plus-median and2 correction drivers),3747 ordered nodes, declared outputs/statuses, cold/warm cache/key equality and fully verified v5 archives. Seventy first-representative operation/serialized-port replay contracts cover all48 supported replay operations with source/model origins hidden. External image/model providers and four-input correction remain outside generic replay.

The known partial mean ellipse boundary is excluded from223 and verified separately. Selected plain9/-2 and calibrated21/-1 mean chains match actual pinned public raster/color products and size observations under the existing1e-6 tolerance. Baseline SHA b8fefc1b5d3bf0d595bfa243303bcc194f181d49f36cf05502d69654ace4c8f2; test SHA4804505a6dcea146bb316b2c495e08e57e0ae6487e308da222e3d3538f8c222d. Tests never regenerate expected evidence. The exact two-file freeze has bounded review approval from supplied projections/root evidence, with no independent reviewer I/O, tests or visual inspection. Fourteen final/prior-reader frozen files,13 fixture/exporter source files and capture helper identities are verified unchanged; runtime application code is unchanged.

All eight register/planning mirrors now record this checkpoint. Forty operation statuses and48 replay operations remain unchanged;143 referenced test files include causal recipe acceptance references. All6 structural register checks PASS2.094s,zero skips. Evidence: artifacts/recipe-acceptance-frozen-hashes-v1.json, recipe-acceptance-final-root-evidence-v1.json, recipe-acceptance-review-v1.json and recipe-acceptance-focused-v1.log; closure is recorded in recipe-acceptance-closure-v1.json. Failed v1/v2/initial-mean history remains in the candidate audit and preceding handoff history. No defaults, broader-domain, all223 public-source parity, maximum-resource or scientific claim follows from selected native lifecycle agreement.

Next: extend actual pinned public whole-chain reference capture and acceptance. Full F01-F22/WP0-WP6 stays active; remaining method/input domains, completed-goal full unittest, original SMPTS source and L01-L04 laboratory inputs remain open.

## Closed reader acceptance checkpoint (historical)

Reader acceptance checkpoint (10 October). A frozen28-fixture record pins the ordered source bytes/SHA, public rgb/gray outputs and native color-derived/codec-gray outputs, frames, uint8 dtypes and pixel digests. Oracle, fixture builder, decoder builds, input helper and preparation code identities are bound. Torch2.13.0+cu126/CUDA12.6/RTX3070 is the measured acceptance environment, not a universal device guarantee.

Public46 successes/10 failures and native48 successes/8 rejections remain distinct outcomes. The two Unicode native successes match the identical ASCII input while both public path reads fail; no path-outcome parity is claimed. All48 successful native mode cases reuse identical warm preparations and pixels.24 actual color-to-codec switches preserve RGB tensor identity and contents; first-mode switch checks are not applicable.19 fixtures produce different grayscale pixels.

Fresh2 tests PASS21.645s,zero skips, independently reconstruct the full fixed record and never regenerate expected evidence. Missing or changed baseline bytes reject. Bounded review approves the supplied test/evidence freeze (artifacts/reader-acceptance-frozen-hashes-v1.json); all generated records stay in ignored artifacts/. The preceding ten-file reader implementation is unchanged. No independent reviewer I/O/tests/visual inspection, maximum-domain, complete-family or scientific claim. All six final feature-register checks PASS0.791s,zero skips. Forty operation mirrors/current table rows,142 referenced test-file hashes,two frozen test/evidence files and the preceding unchanged ten-file reader implementation are verified. Closure: artifacts/reader-acceptance-closure-v1.json.

F01-F22/WP0-WP6 remains active:40 registered operations,48 replay contracts and142 referenced test files. F01.1 stays partial_native/pending_recipe_freeze/not_applicable. Next: finite complete-recipe cold/warm/archive/replay output acceptance. Remaining method/input domains and all-recipe output freeze, completed-goal full unittest, original SMPTS and L01-L04 stay open.

## Reviewed reader implementation checkpoint (historical)

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
Small-object ROI checkpoint (10 October). CUDA fill now filters independent nonempty half-open rectangles, counts4-connected objects locally and preserves exterior pixels. Twenty-eight additive seed/calibrated/debug variants keep historical definitions and independent erosion/dilation controls. Both desktop action orders, dragged layout, Keep/reopen, project batch, node-local caching and source-independent composed replay are verified.

125 pinned public mask-value comparisons pass. Public explicit-ROI replacement returns NumPy platform integer dtype; native BinaryMask remains uint8 with identical0/255 values. Typed-output parity for that source quirk is not claimed. Tiny ROI retains full-canvas admission. Strict callback/storage guards cover original/private buffers; the historical bridge now verifies the actual imported callback module aliases as well as helper code.

The20-test focused gate PASS400.342s and146-test affected gate PASS949.067s precede the final nonnumerical alias correction. The final19-test gate PASS425.616s,zero skips, retests all12 new ROI checks and the affected bridge/helper checks.166 distinct code-test IDs have passing executions across these gates; no single combined green166-test gate is claimed. Initial11 tests had10 passes/one layout-hash assertion failure and remain nonpassing history.

Eight final files are frozen in artifacts/fill-roi-frozen-hashes-v2.json; six stayed unchanged during the last two-file correction. Bounded review approves supplied projections/root execution. No independent reviewer reads/tests/visual inspection or new maximum-resource/scientific claim is made. All six final feature-register checks pass in0.853s,zero skips. Forty row mirrors,139 referenced test-file hashes,eight final frozen files and the unchanged canonical peer manifest are verified; scoped whitespace passes. Closure: artifacts/fill-roi-closure-v1.json.

F01-F22/WP0-WP6 remains active with40 registered operations,48 saved-stage replay operations and139 referenced test files. F07.2 remains partial_native/pending_recipe_freeze/not_validated. Next software gap: expose existing median ROI in working graph/replay controls while preserving boundary rejection rules. Remaining method/domain acceptance, all-recipe reference-output freeze and completed-goal full unittest stay open. Original SMPTS source and L01-L04 laboratory inputs remain external gates.

## Previous rectangular morphology checkpoint (historical)
Rectangular morphology checkpoint (10 October). Erosion and dilation now expose null or explicit nonempty half-open ROI bounds in fourteen additive seed/calibrated/diagnostic working variants. CUDA filters the crop independently and replaces it in a full-canvas copy; exterior pixels stay unchanged. Kernel2..255, iterations1..64, node-local invalidation, dragged layout, Keep/reopen, project batch and source-independent composed replay are tested.

All104 public outputs match the verified PlantCV4.11.3/OpenCV4.11.0.86 wrapper/helper environment. Controls use helper slice endpoints; the public rectangle constructor stores x+w-1/y+h-1 endpoints, which its helper then treats as exclusive. Arbitrary contours and empty/out-of-frame rectangles are outside the native domain. Tiny ROI retains full-canvas admission. Shared callback/storage guards and caller-independent source recompilation are tested; no new maximum-resource or visual-inspection claim is made.

Focused26 tests PASS159.366s and repaired export24 tests PASS20.988s,zero skips. The broader171-test run had170 passes and one stale schema1 test assertion; it remains nonpassing history. The repaired assertion requires schema2 and the exact revision contract.197 distinct code-test IDs have passing executions across these gates; this is not a single combined green197-test run. The ten-file freeze is artifacts/morphology-roi-frozen-hashes-v1.json. Bounded review approved supplied projections/root evidence; no independent reads/tests/visual inspection.

All six feature-register checks pass in0.770s,zero skips. Forty row mirrors,138 referenced test-file hashes, the ten-file tested freeze and unchanged canonical peer manifest are verified.

F01-F22/WP0-WP6 remains active:40 registered operations,48 saved-stage replay operations and138 referenced test files. F07.3 remains partial_native/pending_recipe_freeze/not_validated. Remaining operation/domain acceptance, all-recipe/reference-output freeze and completed-goal full unittest are open. Original SMPTS source and L01-L04 laboratory inputs remain external gates. Prior source freezes retain their historical scope; shared graph/helper/UI sources changed in this checkpoint.

## Previous prediction review revision checkpoint (historical)
Prediction review revision checkpoint (10 October). Review/report schema2 now binds instance confidences, candidate scores/selections and rejected manual-centre coordinates/reasons alongside the source, labels, settings and checkpoint. The same detached numeric snapshots supply identity and exported values. Changes produce a new revision; old decisions remain attached to their prior revision. Malformed compact facts reject before review attachment.

The final affected gate passes all33 tests in6.436s,zero skips, including10 new checks. Coverage includes actual desktop review loading, saved-project batch serialization and CUDA crop publication. The two-file change has bounded review approval based on supplied projections/root results; no independent reviewer reads/tests or visual inspection is claimed. The earlier10/23 passing gates overlap these33 tests.

All six feature-register checks pass in0.853s. Forty matrix rows,137 referenced test-file hashes, the two-file tested freeze and the unchanged ten-file Otsu freeze are verified; scoped whitespace passes.

F01-F22/WP0-WP6 remains active with40 registered operations,48 saved-stage replay operations and137 referenced test-file hashes. The omitted-prediction-fact software gap is closed; broader eligibility/method integration, remaining explicit operation/domain acceptance, all-recipe/reference-output freeze and fresh completed-goal full unittest verification remain open. Original SMPTS source and L01-L04 laboratory inputs remain external gates.

## Previous binary and Otsu checkpoint (historical)
Binary and Otsu cutoff checkpoint (10 October). Fixed threshold now has512 exact public mask checks spanning every uint8 value, integer cutoff and polarity (131072 decisions), including a noncontiguous CUDA input. Otsu exposes native-selected cutoff facts from the same single histogram as its mask, preserving historical mask-only ports. Cache reuse, fact integrity, required v5 archive facts, strict v1-4 legacy loading and saved-input mask-plus-fact replay are implemented; older cutoffs are explicitly unrecorded.

The pinned reference adapter retains the actual public PlantCV mask and separately captures the cv2.threshold scalar after exact mask verification. Combined conformance requires both mask and cutoff equality; scalar differences fail even when masks agree. Sixteen Otsu mask cases and two exact scalar pairs pass. Constants, thin/singleton, imbalanced and empty-bin cases are covered; occupied ties/near-ties remain withheld. Normal async desktop facts and replay after deleting the source pass. No visual-inspection or new full-resolution/resource-domain claim is made.

The final15 focused checks pass in57.416s and128 affected checks pass in548.421s,zero skips. Freeze: artifacts/otsu-cutoff-frozen-hashes-v2.json (10 files). The initial13-test run had12 passes and one incorrect test result attribute; it remains nonpassing history. Bounded review uses supplied projections/root evidence only; no independent reads/tests/visual inspection.

F01-F22/WP0-WP6 remains active and partial_native, with40 registered operations,48 saved-stage replay operations and136 referenced test-file hashes. Next: remaining operation/domain acceptance and all-recipe/reference-output freeze. Original SMPTS source/license/dependencies, L01-L04 laboratory inputs and a fresh full unittest suite at completed-goal handoff remain open. Exact mask/cutoff evidence is not scientific validation.

## Previous adaptive threshold checkpoint (historical)
Adaptive threshold kernel checkpoint (9 October). CUDA mean accepts integer requests3..255 and Gaussian3..15, rounding even sizes upward to odd exactly as PlantCV does. Requested values remain in recipe/settings/cache identity; effective values drive actual computation and resource admission and appear in reference/native checkpoints, replay work details and desktop facts. Live tooltips explain the rule without changing the historical catalogue. Existing precision certificates and card-specific controls retain their domains.

The pinned public checks cover756 structured mean masks,54 thin/singleton masks,4 kernel-effect cases and72 Gaussian masks. Requested254 still withholds the scalar/SIMD rounding counterexample with no published checkpoint. Requested14 uses effective15 for Gaussian work/scratch limits. Normal async mean4 and Gaussian4-to14, distinct requested4/5 cache identity, exact reference/native selected/effective metadata, even4 conformance, and both Gaussian archived stages after deleting their origin are verified. A below-minimum typed edit rejects before execution and preserves valid cache; inconsistent recorded controls reject at export before an archive exists. No additional full-resolution even-kernel domain is claimed.

All39 final affected checks pass in157.063s with zero skips under artifacts/adaptive-even-kernels-frozen-hashes-v4.json (13 files). The earlier93-,39- and23-test runs remain nonpassing history; repaired fixtures and the strict conformance domain string did not weaken mask comparisons or precision guards. Review uses supplied projections/root evidence only; no independent reviewer reads/tests/visual inspection. Final log: artifacts/adaptive-even-kernels-final-green-v7.log.

F01-F22/WP0-WP6 remains active and partial_native. The register/matrix still covers40 operations and48 saved-stage replay operations, now with135 referenced test-file hashes. Next software work: remaining binary/Otsu method-domain acceptance and selected-cutoff adapter metadata, then complete operation/domain and all-recipe/reference-output freeze. Original SMPTS source/license/dependencies, L01-L04 laboratory inputs and fresh full unittest verification at completed-goal handoff remain open. No scientific or original-code parity is claimed.

## Previous image input checkpoint (historical)
Image input and decoder checkpoint (9 October). Native recipes now decode the exact bounded immutable byte snapshot identified by source size/SHA-256. The separately pinned PlantCV 4.11.3 readimage(mode=rgb) contract supplies BGR uint8; CUDA derives RGB and pinned grayscale from the same color-decoded raster. Alpha dropping, 16-bit quantization, EXIF orientation and first-image behavior remain explicit. Cache identity binds canonical path/bytes/device, helper/preparation/converter/guard code and actual decoder versions/build/flags. Initial path-entry, fresh preparation and cached-return callbacks share strict bool, source, metadata and tensor checks. Detached source/frame/preparation facts prevent stale reuse; final source/code checks precede cache publication. Node-local recomputation remains intact.

The final eleven focused checks pass in 33.189s and eighteen compatibility checks pass in 107.482s, with zero skips. They cover source/cache replacement, cancellation phases, caller-exception identity, mutation priority, private CPU headers/pixels and CUDA buffers, all nine preparation arms, early model behavior, saved projects/batch, node-local reuse and actual async Qt paths. Four reviewed files remain frozen in artifacts/native-image-inputs-frozen-hashes-v1.json. Review approval uses supplied projections and root evidence; no independent reviewer reads, tests or visual inspection.

The saved 32-row public reader matrix has 27 successful exact full BGR codec comparisons and four matching invalid-input outcomes. CUDA RGB/gray are exact on the 23 small public-success cases plus the four additional committed/long-row/singleton cases. PNG/JPEG/BMP/TIFF, gray/palette/alpha/depth variants, JPEG EXIF 2–8, CMYK JPEG, first TIFF page and misleading extensions are covered. The remaining Unicode filename is an explicit native path extension: pinned OpenCV4.11 Windows imread fails that path, while Python byte loading succeeds and matches the identical ASCII-path codec input. Path-outcome parity and other readimage modes are not claimed.

Six fresh preparation children start at zero allocated CUDA bytes. The unchanged 2080x3120 image peaks at 47,000,064 bytes within a 48,572,928-byte preparation envelope and 330,820,096-byte graph admission, both with no callback and a strict False callback. The unchanged 4160x6240 image peaks at 183,281,664 within 184,854,528 and 973,290,496 respectively. A 1x65553 row and one pixel also match. The generated 7000x8000 image decodes but exceeds the actual graph byte budget, with zero CUDA uploads and zero peak allocation. Traced host peaks are 11,449,487–179,440,754 bytes; encoded caps, traced peaks and conservative preparation envelopes are not decoder memory/process RSS guarantees or maximum-domain proof.

Root inspected normal async source-stage views at 1440x900 and 1050x650 using explicitly registered Windows Segoe UI 9. Provided status, source hash/frame and the image preview are readable; repeated runs reuse prepared tensors and all completed nodes. The selected CJK filename glyph is a square in the offscreen font label, so this evidence does not claim all Unicode font rendering. Numeric/cold probe v1 stopped afterward on a wrong provided-versus-executed source-node assertion; functional UI v2 captures lacked registered fonts. These remain nonpassing/superseded history; isolated UI v3 and final evidence are artifacts/native-image-inputs-final-evidence-v3.json. Application code was unchanged by the probe repairs.

There are still 48 declared saved-stage replay operations and 40 registered operation rows. This checkpoint closes the selected decoder matrix, not complete method/input-domain acceptance. Next: execute the remaining operation/domain inventory and final all-recipe/reference-output freeze. Original SMPTS source/license/dependencies, L01–L04 laboratory inputs and a fresh full unittest suite at completed-goal handoff remain open. F01–F22/WP0–WP6 stays active and partial_native; no original-code or scientific parity is claimed.

## Previous crop checkpoint (historical)

Calibrated seed crop replay checkpoint (9 October). calibrated_seed.crop now replays only verified saved corrected-RGB, source-frame, card-geometry and card-ROI inputs. There are 48 declared replay operations. Exact half-open seed bounds, controls, direct and card-ancestor producer keys, source/parent/local frames, raster headers and fixed metadata bind before upload. Original source images and upstream correction/detection are not opened or recomputed. Both local CUDA rasters and every metadata field are compared; a bounded but wrong overlap count is a numerical mismatch.

The full 13-test crop gate passes in 259.160s with zero skips. The broader 68-test compatibility gate has 67 passes and one public cancellation-exception error in 745.081s and remains nonpassing history. A narrow exception-only adapter restores AutomaticCardCancelled; all six affected checks pass in 164.153s under the final v4 freeze. They include initial/mid/final cancellation, caller-exception identity, mutation priority, saved-input guards, all eight calibrated/diagnostic graphs and actual async Qt replay. Numerical calculations are unchanged by the exception adapter; a fresh full unittest suite remains required at the completed-goal handoff.

Six independent pinned OpenCV 4.11.0/NumPy 2.2.6 oracles match saved-input RGB slices, grayscale bytes, inclusive overlap counts and all frame/translation metadata. Six fresh-process replays start with zero allocated CUDA bytes and fit their plans, including the first runtime audit. The 2080x3120 full crop peaks at 59,008,000 bytes within 88,244,736 planned; its 96x128 crop peaks at 19,912,704 within 20,435,456. Full-width rows, partial/shifted equal-size crops and one pixel also match. Contiguous row/full/singleton slices may alias admitted parent storage; admission conservatively charges a fresh RGB copy. Traced host peaks are 24,602,703–212,857,363 bytes, not process RSS limits or maximum-domain proof.

The large parent is a generated 2080x3120 fixture made from the committed image plus a controlled card/seed patch; committed images are unchanged. Median is intentionally unselected in the full/alias/singleton source runs, so the crop executes while those whole graphs remain partial. The ordinary partial/shifted seed graphs execute completely. Arbitrary parent IDs/transforms are covered by the focused tests, rather than the cold probes.

Two root-inspected normal async initial-run/stage-selection/replay Qt captures at 1440x900 and 1050x650 use Segoe UI 9. Bounds, local-frame translation, overlap and RGB previews are readable; active results and archive hashes are preserved. Evidence: artifacts/native-crop-replay-probe-v1/evidence.json. Five-file freeze: artifacts/native-crop-replay-frozen-hashes-v4.json. Bounded review: artifacts/native-crop-replay-review-checkpoint.json; approval uses supplied projections and root evidence, with no independent reviewer reads, tests or visual inspection.

All 40 registered F01–F22 operation rows are mapped in docs/PLANTCV_SMPTS_METHOD_DOMAIN_MATRIX.md and its JSON inventory, with contracts, source functions, native mappings, domain axes, test candidates and required acceptance work. This is an audit map, not completed method/domain acceptance. Next engineering work is the decoding/channel/orientation/format freeze, explicit remaining method-domain acceptance and final recipe/reference-output freeze. Original SMPTS source/license/dependencies, L01–L04 laboratory inputs and the completed-goal full suite remain open. F01–F22/WP0–WP6 stays active and partial_native; reconstruction agreement does not establish original-code or scientific parity.

Earlier checkpoints follow; their operation counts, freezes and evidence retain historical scope.

## Previous paper checkpoint (historical)

Paper saved-stage replay checkpoint (9 October). Eight provisional smpts_paper.* operations now replay all nine calculated paper stages, including both component stages, bringing the declared replay count to 47. Exact controls, producer keys, typed ports, raster headers, coordinate frames, inventories, contact IDs, crop bounds/offsets/parents and provisional metadata are bound before upload. Only the selected stage executes; original image paths and ancestor calculations are not opened or recomputed. Packed crops may exceed source pixel count because component boxes can overlap. Contacts uploads its direct label dependency, even though the calculation uses its shape.

The final calculation gate passes all 13 paper tests in 198.543s with zero skips. The shared 53-test gate has 52 passes and one tuple/list fixture assertion failure; the repaired assertion passes in 15.240s. Those logs retain their respective scope. Ten affected contract/desktop checks pass in 46.684s after the selector repair. The final contrasting-active/archive Qt regression passes in 7.660s after failure/cancellation coverage and visible-label cleanup. Replay crop selection reads replay buffers; node reselection restores active crops, and failed/cancelled replay hides the selector. Calculation code is unchanged from the final 13-test and cold gates.

Fourteen fresh-process replays match from zero allocated CUDA bytes and fit their allocation plans, including the first runtime audit. The committed 2080x3120 fixture has 503 source components, 31 selected parents and 1118 final crops. An isolated pinned OpenCV 4.11.0.86/NumPy 2.2.6 oracle matches all nine rasters, all packed RGB/mask crops and their inventories/lineage. Full mean replay peaks at 242,178,048 bytes within 913,182,208 planned; packed crops peak at 142,010,368 within 209,440,768. Controlled, singleton, empty and overlapping-box cases also pass. Traced host peaks are 21,383,714-292,383,624 bytes; these are not process RSS limits or maximum-domain evidence.

Six final actual async Qt views at 1440x900 and 1050x650 use the normal initial run and stage-selection handlers with Segoe UI 9. Root inspected contact/crop summaries, coloured labels, individual replay crops and corrected button text; active runs and archives are preserved. Final freeze: artifacts/native-paper-replay-frozen-hashes-v8.json. Oracle/cold evidence: artifacts/native-paper-replay-final-probe-v2/evidence.json. Final visuals: artifacts/native-paper-replay-final-ui-v5/evidence.json. Bounded review: artifacts/native-paper-replay-review-checkpoint.json; the reviewer used supplied projections/root evidence and performed no independent reads, tests or visual inspection. Earlier failed schema adapters and superseded captures remain historical evidence.

F01-F22/WP0-WP6 remains active and partial_native. Next: calibrated_seed.crop saved-input replay, the complete registered method/input/domain matrix and final recipe freeze. Original SMPTS source/license/dependencies, L01-L04 laboratory inputs and fresh completed-goal full unittest discovery remain open. Reconstruction agreement does not establish original-code conformance or scientific validation.


Recovery partition, annotation and measurements replay checkpoint (9 October). All six experimental `recovery.*` operations now support verified saved-input replay, bringing the declared operation count to 39. The four generic upstream calculations also replay, covering all ten calculated nodes in the recovery graph. Partition preserves nine typed CUDA products and input/output aliases; annotation preserves the shared 0/1 uncertainty band; measurements selects zero arrays and consumes archived annotation metadata. Original image paths and ancestor calculations are not executed. Measurement replay explicitly reports that upstream pixels and statistics are not recomputed.

The affected 60-test gate passes in 607.405s with zero skips. Two added final checks pass in 39.583s: mutations of every nested partition tensor followed by a caller exception withhold publication, and all four upstream stages match. Calculation source is unchanged after the shared gate. Coverage includes exact raster/compact comparisons, malformed aliases/headers/encodings, finite elevation, all uncertainty radii, both policies, disappeared IDs, empty/singleton/unseeded inputs, source/cache behavior, coordinated frozen-record/snapshot replacement and actual recovery/generic/card/model Qt replay. The initial 15-test gate had 13 passes and two fixture defects; it remains nonpassing history. The repaired four-test gate passes in 44.453s.

Four independent controlled reference cases pass: 259x513 full/ROI touching cores, 1x1 singleton and 2x3 empty input. Pinned public PlantCV threshold/erosion, SciPy four-connected IDs, OpenCV chamfer and skimage watershed match all partition arrays. NumPy boundary records/traits and SciPy four-neighbor dilation match annotation and measurement records. Full support retains three active IDs; ROI retains two plus a disappeared inventory row. Both have two interface pixels. The withhold/flag policies permit one/two cores respectively. This is computational engineering evidence, not original SMPTS or scientific validation.

Twelve fresh-process replays match from zero allocated CUDA bytes, selecting 9/10/0 arrays for partition/annotation/measurements. The first audit's 13,312 retained bytes are included. Full/ROI partition peaks at 4,405,248 bytes within 17,869,312 planned; annotation at 3,739,648 within 12,313,600. Measurements peaks at 29,184 within its 65,536-byte audit reserve. Singleton/empty partition peaks are 286,720/284,672 and annotation 34,816/34,304, all admitted. Traced host peaks are 19,663,770–22,520,649 bytes; encoded caps are not RSS guarantees. Partition admission uses conservative D=P before upload; active IDs L and original inventory I remain distinct. The downstream source limit is 8,064 objects, while the existing whole-report 8 MiB/one-million-traversal archive limits remain separate from the 64 MiB per-product replay decoder. Maximum whole-archive capacity is not established.

Six actual async Qt renders at 1440x900 and 1050x650 use Windows fonts and the normal source setter. Root inspected distinct partition colours, the nonzero uncertainty preview, eligibility/unavailable observations and source captions; active runs and archives are preserved. Evidence: `artifacts/native-recovery-final-stages-probe-v2/evidence.json`; final seven-file freeze: `artifacts/native-recovery-final-stages-frozen-hashes-v2.json`; bounded review: `artifacts/native-recovery-final-stages-review-checkpoint.json`. The reviewer approved supplied projections and root execution/visual evidence, with no independent reads, tests or visual inspection. Earlier freezes and the merged-core first probe retain historical scope.

F01–F22/WP0–WP6 remains active and partial_native. Next: the eight provisional paper-helper operation contracts (including both component stages), the complete registered method/input/domain matrix and final recipe freeze. Original SMPTS source/license/dependencies, L01–L04 laboratory inputs and fresh completed-goal full unittest discovery remain open.

Earlier recovery clipping checkpoint follows.

Recovery clipping saved-stage replay checkpoint (9 October). `recovery.clip_cores` now replays verified archived markers and validity in the full source frame. There are 36 verified declared replay operations, including three experimental recovery stages. Every clipped marker pixel is compared; surviving sparse IDs remain unchanged. Canonical producer keys, controls, headers, core inventory and source/ROI metadata are bound before upload. Ancestor numerical calculations and original image paths are not executed or opened. Clipping retains the upstream 32,768-core domain and does not impose the downstream 8,064-object metadata limit.

Recovery object records, resource plans and final annotation statuses now use detached primitive-field snapshots. Private captured baselines also detect replacement of a retained snapshot during a callback. In-place frozen-field changes, including changes followed by caller exceptions, withhold publication. The first focused gate passes 11 tests in 27.121s. The seven-module shared gate has 115 passes and one desktop-summary failure in 908.560s; it remains nonpassing history. The summary was hidden for a raster-only output and was repaired by rendering available summaries. All four affected actual async Qt replay tests then pass in 29.839s. The broader source/calculation checks include full-resolution recovery, cache reuse, generic/card/model replay and all nine card variants. Fresh full unittest discovery remains required at completed-goal handoff.

Five controlled input/reference cases pass: 259x513 full/ROI, 1x1 singleton, 32,768 sparse cores in a 1x98,304 image and all-invalid saved support. Pinned public PlantCV threshold/erosion and vectorized SciPy four-connected first-pixel IDs establish the core inputs; NumPy conditional clipping supplies the independent expected result. The all-invalid case explicitly consumes saved support without recomputing upstream ROI pixels. Five fresh processes start with zero allocated CUDA bytes and select three unique direct arrays. Full/ROI/all-invalid peaks are 2,940,416 bytes within 2,991,616 planned; singleton is 31,232 within 69,120; maximum inventory is 2,176,512 within 2,228,736. The first runtime audit's 13,312 bytes are included. Traced host peaks are 17,531,643–60,219,099 bytes; encoded caps are not RSS guarantees. This is controlled engineering evidence, not complete raster-domain, original SMPTS or scientific validation.

Two actual async Qt renders at 1440x900 and 1050x650 use Windows fonts and the normal source setter. Root inspected the readable clipping summary, full-frame preview and source caption. Active runs and recursive archive hashes are preserved. Evidence: artifacts/native-recovery-clip-probe-v1/evidence.json; seven-file freeze: artifacts/native-recovery-clip-frozen-hashes.json; bounded review: artifacts/native-recovery-clip-review-checkpoint.json. Review approves supplied projections and root execution/visual evidence only; no independent reads, tests or visual inspection are claimed. The earlier core/validity 93-test gate and six cold probes remain historical in artifacts/native-recovery-stage-replay-probe-v1/evidence.json after the shared recovery source changed.

F01–F22/WP0–WP6 remains active and partial_native. Next: partition, annotation and measurement replay, paper helper contracts, the complete registered method/input/domain matrix and recipe freeze. Original SMPTS source/license/dependencies, L01–L04 laboratory inputs and fresh completed-goal full unittest discovery remain open. Routine project edits remain authorized under auto_review; do not ask again.

Earlier colour-card checkpoint follows.

Colour-card saved-stage replay checkpoint (9 October). All eight `color_card.*` stages now replay verified archived direct inputs: BGR, preprocessing, geometry, masks, scale, means, standard targets and affine correction. There are 33 verified declared replay operations. Canonical producer keys, controls and source/ROI frames bind nine card graphs: standalone, four calibrated and four calibrated diagnostic variants. The three plain seed diagnostic variants contain no card stages; the earlier foundation planning statement of seven card diagnostic compositions was incorrect.

The final affected ten-module gate passes 129 tests in 725.227s with zero skips, including22 card integration tests. It verifies every raster/compact output, deleted origins, all nine variants, explicit full-bounds ROI identity, label validation before uint8 narrowing, strict/final cancellation and private CUDA/CPU/header/field mutation guards. The earlier181-test shared gate had179 passes and two public mutation-error-family failures; it is nonpassing history. Those failures were repaired at the public detector/correction boundary and passed the focused3-test gate in5.090s before the final129-test gate. Calibrated/range/diagnostic/cache/UI regressions passed in the earlier shared gate. The repair preserves public AutomaticCardError compatibility, unrelated exceptions and numerical calculations.

Controlled620x460 full-frame and380x540 ROI workflows agree with pinned public PlantCV preprocessing, geometry, masks, scales and standard targets. Chip means use2e-15 absolute tolerance; the declared affine pixel tolerance is one byte, with zero difference in both probes. All sixteen fresh-process stage replays match from zero allocated CUDA bytes and fit their per-stage plans, including the first audit. Scale selects no arrays and peaks at29,184 bytes within65,536; standard selects one855,600-byte BGR array and peaks at885,248 within929,792. The largest observed tensor-allocation peak is19,490,304 bytes for affine within20,529,664 planned. Traced host peaks range15,952,201–20,284,769 bytes; encoded caps are not RSS guarantees. These are controlled engineering domains, not maximum-domain or scientific evidence.

Sixteen actual async Qt renders at1440x900 and1050x650 use Windows fonts and the normal source setter, preserve active runs/archives and retain correct requested sizes. Root inspected all eight operations represented across both sizes: compact summaries and raster previews. Initial renders with a directly assigned source path are superseded caption evidence. Source/recipe/runtime/oracle/probe/output/frame and recursive archive hashes are recorded in artifacts/native-card-stage-replay-probe-v1/evidence.json. Ten-file freeze: artifacts/native-card-stage-replay-frozen-hashes.json. Bounded review approves supplied projections and root execution/visual evidence only; no independent reads, tests or visual inspection are claimed. Earlier source freezes retain historical scope when shared modules change.

F01–F22/WP0–WP6 stays active and partial_native. Next: recovery core-label/validity replay, remaining recovery and paper helper contracts, then the complete registered method/input/domain matrix and recipe freeze. Original SMPTS source/license/dependencies, L01–L04 laboratory inputs and fresh completed-goal full unittest discovery remain open. Routine project edits remain authorized under auto_review; do not ask again.

Archived-model saved-stage replay checkpoint (9 October). `analysis.naive_bayes_classifier`, `analysis.kmeans_predict` and `analysis.kmeans_masks` now replay verified archived inputs. There are 25 declared verified replay operations. Bayes TSV and legacy/provenance K-means model bytes remain host-only; original model paths are not resolved or opened. Prediction binds canonical image/model ancestry and both image headers while selecting only the active raster. Masks replay verifies label/model lineage and identities without recomputing upstream numerical predictions or parsing raw model bytes. Every returned mask pixel, ordered class key/name and model-identity leaf is compared on its typed contract.

The final 11-module focused gate passes 110 tests in 160.816s with zero skips. It covers public primitive comparisons, source/model/private CUDA buffer and nested-output mutation guards including callback exceptions, strict/final cancellation, deleted origins, both model formats, absent/duplicate/large categories, malformed lineage/headers, numerical mismatches, node-local caches and actual desktop replay. The actual 56-class small-mask regression admits allocator rounding. The 76-test earlier gate is historical; an initial fixture/module gate and the first three probe attempts are nonpassing history. The third probe exposed and led to correction of the comparison reduction's int64 temporary budget. Saved archive integers retain their existing 256-bit bound; direct category calculation permits larger absent IDs.

Controlled source/public/reference probes: Bayes259x513 matches all three pinned public masks and398,616 replay values, selecting834,788 bytes. Replay adds3,303,936 CUDA bytes within19,654,144 planned;944,128 preexisting bytes are separate. Gray131x259/p2/legacy and RGB131x259/p3/envelope K-means labels and all masks match pinned public PlantCV. Predictions add1,175,552/3,570,688 CUDA bytes within68,226,560/68,817,920 planned. Masks select135,716 bytes, compare101,803 values and add546,816 within680,960 planned. A separate fresh-process masks replay starts at zero allocated and adds560,128 bytes, including its first runtime audit. All classes in these probes are nonempty. Traced host peaks are13-14MiB; encoded/raw metadata caps do not guarantee RSS. These are controlled engineering domains, not corpus, maximum-domain or scientific evidence.

Source, recipe, model, archive, output, runtime and oracle-script hashes plus six actual async Qt views at1440x900/1050x650 with explicitly loaded Windows fonts are recorded in artifacts/native-model-stage-replay-probe-v4/evidence.json. Root inspected the class selectors, masks and coloured label previews; active runs and archives were preserved. The initial font-missing renders remain superseded probe evidence. Final source/test freeze: artifacts/native-model-stage-replay-frozen-hashes.json; gate: artifacts/native-model-stage-replay-final-shared-gate.log. Bounded review approves the supplied fifteen-file freeze using supplied projections/root execution and visual evidence; the reviewer performed no independent reads/tests/visual inspection. Earlier freezes retain historical scope when shared modules change.

F01-F22/WP0-WP6 remains active and partial_native. Next software work: remaining card, recovery and paper helper replay, then the complete method/input/domain matrix and recipe freeze. Original SMPTS source/license/dependencies and L01-L04 laboratory gates remain open. Fresh full unittest discovery is required at completed-goal handoff. Routine project edits remain authorized under auto_review; do not ask again. The Windows sandbox launcher may fail with `apply deny-read ACLs`; narrow already-authorized tool retries use automatic review. This is an execution setup failure, not renewed project-edit authorization.

Earlier verified checkpoints follow.

Archived-model replay foundation (9 October). Bounded primitive contracts now bind Bayes TSV and both legacy/provenance K-means JSON encodings to archived byte length/SHA, parsed numerical identity, source selection and ordered class-mask headers. Original model path text is validated without resolving or opening it. Native K-means prediction and replay preparation share allocation-free shape/work/storage planning; replay charges active image, RGB-to-BGR conversion, expected/actual products, primitive buffers, comparison tile and audit reserve separately. Raw model/metadata caps do not guarantee process RSS.

The final focused five-module gate passes 53 tests in 27.355s with zero skips, including 19 new foundation tests and existing pinned public prediction/preprocessing, cache/API and actual desktop classifier tests. Four supplied frozen source/test hashes have bounded review approval using root execution evidence; no independent reads/tests are claimed. Evidence: artifacts/native-model-replay-foundation-reviewed-gate.log, native-model-replay-foundation-frozen-hashes.json and native-model-replay-foundation-review-checkpoint.json.

This is a helper foundation. Saved-stage classifier dispatch, producer/frame binding, replay callback/private-buffer guards, every-mask comparison and replay UI integration remain unfinished. The verified replay count remains 22. Next: connect these contracts for analysis.naive_bayes_classifier, analysis.kmeans_predict and analysis.kmeans_masks, then complete remaining card/recovery/paper replay and the method/input/domain matrix. F01-F22/WP0-WP6 remains active and partial_native; original SMPTS and laboratory gates plus fresh completed-goal full discovery remain open. Routine project edits remain authorized under auto_review; do not ask again.

Earlier verified checkpoints follow.

Size and units saved-stage replay checkpoint (9 October). `analysis.size` now replays original RGB/labels and all seventeen observation records in source or verified seed-ROI frames, comparing the optional CUDA diagnostic separately. Exact source descriptors, nominal datatypes and finite local coordinates are preserved. `calibrated_seed.size_units` replays the compact physical conversion with zero selected arrays and binds the saved size/frame and card-scale producers. Its conversion allocates no CUDA tensors; the shared runtime audit has a separately admitted 64 KiB reserve. Upstream label maxima and card measurements are not recomputed by the units stage. There are now 22 declared replay operations.

The shared 15-module gate passes 185 tests in 1155.051s before the final audit-budget refinement. The final affected size/unit and existing planner/contract gate passes 28 tests in 86.761s after that refinement (zero skips; overlapping gates). Nine final source/test files have bounded supplied-projection/root-evidence review approval. Earlier freezes remain historical when shared modules change.

The controlled direct-input 1027x1025 graph uses 128 actually computed labels with independently prepared, hashed gray/RGB arrays; it makes no single-image decoding claim and changes no upstream outputs. All seventeen size variables agree with pinned public PlantCV under the declared 1e-6 numerical tolerances and ten-decimal longest-path check; the diagnostic agrees byte-for-byte. The public explicit-one empty-mask behavior agrees too. Source replay selects three arrays (10,526,750 bytes), compares 3,171,593 values and adds 21,663,232 CUDA bytes within 268,960,128 planned; 13,312 preexisting bytes are separate. Whole source replay traces 63,354,720 additional peak host bytes. Physical/pixel units select no arrays and add 15,872/15,872 CUDA bytes within the separate 65,536-byte shared-audit reserve; an isolated cold audit measured 29,184 bytes. Encoded metadata caps do not guarantee Python RSS. These are controlled engineering domains, not maximum-domain, corpus or scientific evidence.

Freeze: artifacts/native-size-replay-frozen-hashes.json; final gates: artifacts/native-size-replay-final-shared-gate.log and native-size-replay-final-gate.log; source/oracle/recipe/archive/output/runtime hashes and actual Qt views: artifacts/native-size-replay-probe-final-v2/evidence.json. Actual async size/unit summaries and the size image were inspected at 1440x900 and 1050x650 with Windows fonts; the active run and every archive were preserved. Review uses supplied projections and root execution/visual evidence, with no independent file reads/tests/visual inspection claimed.

The F01-F22/WP0-WP6 goal remains active and partial_native. Next: archived-model classifier replay, then remaining card, recovery and paper helper contracts and the complete method/input/domain audit. Original SMPTS source/license/dependencies and L01-L04 laboratory gates remain open. Fresh full unittest discovery is required before completed-goal handoff.

Earlier verified checkpoints follow.

Colour saved-stage replay checkpoint (9 October). `analysis.color` now supports typed saved-input replay in source and verified calibrated seed-ROI frames. All RGB/LAB/HSV/ALL histograms, bins, statistics, label records and returned hue-chart fields are compared; explicit nonfinite tags preserve NaN/inf outcomes. The native Qt view shows the returned chart and concise observations while preserving the active run. There are now 20 declared replay operations.

The shared 15-module gate passes 179 tests in 979.683s before the final character-length precheck; the final affected colour gate passes 14 tests in 125.179s after that refinement (zero skips; overlapping gates). Six final source/test files have bounded supplied-projection/root-evidence review approval. Earlier freezes remain historical when shared modules change.

The controlled 257x513 selected-port fixture uses 128 labels and ALL, crossing two full tiles plus 769 pixels. All 634,380 compact leaves match saved-stage replay. Actual pinned public histograms/bins agree exactly; finite statistics use the declared 1e-10/1e-5 circular-std tolerances. Replay adds 12,357,120 CUDA bytes within 151,093,376 planned, with 13,312 preexisting separate; selected arrays total 922,887 bytes. Host compact parsing/decoding/snapshots/comparison trace 44,363,468 additional peak bytes; whole replay traces 75,029,055. Process working-set/lifetime-peak readings include imports and the CUDA runtime. Encoded metadata caps do not promise Python RSS. This fixture verifies colour on controlled selected inputs, not its modified upstream watershed history, maximum raster domain, corpus or scientific accuracy.

F01-F22/WP0-WP6 remains active and partial_native. Next: typed size/unit observation replay, then remaining model, card, recovery and paper helper contracts and the full method/input/domain audit. Original SMPTS source/license/dependencies and laboratory inputs remain separate open gates. Fresh full unittest discovery is required before completed-goal handoff.

Earlier checkpoint history follows.

Calibrated range workflow checkpoint (9 October). The additive `plantcv_calibrated_seed_range` graph corrects the colour card, crops an explicit seed ROI, applies channel-range selection, smooths its binary mask, and executes fill/morphology/watershed plus size/colour observations. Traits use the original corrected cropped RGB; the range masked RGB is a pre-median diagnostic. This is a user-selected composition of pinned operations, not a recovered historical fixed-range workflow or scientific validation.

The final 11-module gate passes 143 tests in 647.331s, including 11 new integration tests, with zero skips. All nine prior calibrated/diagnostic recipe serializations remain exact. Tests cover five spaces plus partial ranges, complete pinned corrected-ROI outputs/physical observations, range/median/ROI/units cache boundaries, empty-result recovery, fresh/cached/debug admission, excess input backing, final callback/runtime mutation, actual Qt replay/Keep/reopen and saved-project/batch/archive round trips.

Actual custom-stream source/ROI resource probes and async Qt views pass. Range and mask-median saved-input replay verify local frame/translation while preserving the active run and archive. Six source/test files have bounded supplied-projection/root-evidence review approval. See [all requested feature statuses](FEATURE_IMPLEMENTATION_STATUS.md).

F01-F22/WP0-WP6 remains active and partial_native. Remaining model/helper/paper/observation replay and the full method/input/domain freeze remain open, alongside original SMPTS source and laboratory gates. Fresh full unittest discovery is required before completed-goal handoff. Earlier source freezes retain historical scope when shared modules change.

Earlier checkpoint history follows.

Custom-range graph and saved-stage replay checkpoint (9 October). The separate `plantcv_threshold_custom_range` recipe now executes the existing pinned CUDA primitive with real RGB/HSV/LAB/CMYK/GRAY and inclusive byte-bound controls. Both the binary mask and masked original RGB are retained, exported and replayed. RGB bounds use pinned B,G,R storage-slot semantics; CMYK retains the source C/Y swap. The four existing extended recipe identities and historical reference catalogue remain unchanged.

The final shared gate passes 133 tests in 220.734s (11 new range tests; zero skipped). Actual 887x887 LAB/CMYK probes select 353,079/214,138 pixels and match both pinned public outputs and isolated saved-input replay. Replay adds 29,599,232/52,102,656 CUDA bytes within 234,924,160 planned; 13,312 preexisting bytes are separate. These are controlled engineering domains, not maximum-domain, corpus or scientific evidence. There are now 19 declared saved-stage replay operations.

See [all requested feature statuses](FEATURE_IMPLEMENTATION_STATUS.md). The 15-case source matrix covers five spaces, partial ranges, equal endpoints and reversed bounds. Tests also cover strict malformed controls, cache reuse, selected-only archive loading, wrong valid outputs as mismatch, cancellation, final owned-output storage mutation, runtime changes, empty batch/export and actual Qt/Keep/replay. The eight frozen source/test files have bounded supplied-projection/root-evidence review approval. Fresh completed-goal full-suite and scientific gates remain open.

Earlier checkpoint history follows.

SMPTS executable paper reconstruction checkpoint (8 October). The additive `smpts_paper_fixed_mean` graph now executes the paper-supported branch/filter/intersection/contact/erosion/extraction sequence. Defaults25/5/13 come from the paper; offset/polarity, fixed cutoff, connectivity, axis-aligned boxes, box-area cutoff, border, erosion shape and crop rules remain declared assumptions. The historical unresolved graph is unchanged.

The 111-test relevant gate passes in 117.973s (19 new tests;0 skipped). The 2080x3120 committed fixture matches nine pinned OpenCV substep rasters and all 1118 packed crops. This verifies the declared reconstruction, not original SMPTS conformance or scientific accuracy. The full F01-F22/WP0-WP6 goal remains active; full discovery is still required before completed handoff.

The declared default fixture produces503 source components,31 suspected contacts and 1118 relocalized crops (34,124,820 packed bytes). Additional CUDA peak is229,198,848 bytes within 616,660,992 planned. Outputs include preserved unselected pixels, explicit disappearance and split-parent lineage. Overlapping boxes carry isolated masks. Empty extraction succeeds with zero crops; cap/stall/mutation/cancellation failures publish no incomplete descendants.

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

Current desktop checkpoint (8 October): correction Keep/reopen retains verified
four-file selections for the exact project image. Keep merges compact session
metadata after worker capture and stale/cancel checks; Save Project saves the
master. Reviewed evidence and remaining full-scope gates are recorded below.

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

**Status date:** 8 October 2026
**Engineering baseline:** `094aa7d`; prior implementation review `f244df8`
**Historical frozen full verification (before the current batch):** 989 tests in 1070.527 seconds: 987 passed and 2 existing skips.
The full-image CUDA fill diagnostic was enabled. All 220 source/test/script/
configuration/register hashes remained unchanged throughout discovery. Log:
`artifacts/distance-crop-discovery-corrected-full-suite.log`; matching before/after inventories use the
same prefix. The documented `unittest discover -s tests -v` command was used.

The feature register and planning documents account for all F01–F22 and WP0–WP6.
The current local batch adds reviewed CUDA median, erosion/dilation and 4-connected
small-object removal; exact original/native stage bindings; paired threshold
reports; original five-stage preprocessing; and verified checkpoint inspection for
seven original operations. Independent Sol review approved their supported domains.
Original-chain layout-only rejection and partial-final-output defects are repaired
and approved. The selected-parameter Qt viewer was tested through actual asynchronous
store binding and a rendered preview.

Focused evidence: six native stage tests, eight fill tests including a 3120×2080
original-reference comparison, nine original-chain tests, and 18 independently run
inspection tests pass. The new native five-stage coordinator passes all 13 tests,
including binary/mean/accepted-Otsu pixel equality at every preprocessing stage.
Its independent Sol review is complete. It uses explicit stage-specific native bridge
projections and preserves the parent selected identity and stage mapping; it is a
benchmark API, not complete saved desktop recipe execution. The current batch remains
uncommitted. Historical test counts and review evidence are retained in
[HANDOFF.md](../HANDOFF.md).

The distance/marker/crop batch has bounded independent Sol approval and passes
fresh frozen full discovery. Five distance tests include exact pinned OpenCV equality
on the 3120x2080 fixture mask. Nine stage tests include 1080p CUDA/SciPy marker
partition agreement, exact recipe/settings identity and publication-budget preflight.
Crop extraction passes ten tests; twelve artifact tests cover PNG pixels, isolated
masks, coordinates, stale revisions, Qt filtered rows, Stop/close, saved-project batch
publication and CLI partial reporting. The Qt review dialog was visually checked.
Native crop artifacts bind source/result identity and publish atomically. These
bounded implementations advance F09/F10/F14/F19/F20; complete peer recipes and
original SMPTS extraction semantics remain open. The local batch is uncommitted.

The follow-on peak/range/size/native-inspection batch is in progress and is not
covered by the preceding 989-test frozen run. Peaks have five focused tests and
bounded review; previously reviewed range modes now have a new exact LAB candidate.
All eight range tests pass in 10.093 seconds, and all 16,777,216 byte BGR colours match
pinned OpenCV LAB conversion with zero differing pixels/channels. Additional CUDA
allocation peaked at 4,600,832 bytes for 65,536-pixel tiles. Bounded independent
LAB source/test review and diagnostic-script/report audit are approved at matching
frozen hashes. The reviewer did not rerun the diagnostic; this conversion evidence
does not establish complete recipe parity.
The CUDA size/topology foundation returns all 17 PlantCV observations for supported
empty, single, disconnected and nested contours with holes. Compact CPU geometry
uses bounded contour/spans metadata; dense masks/reductions stay CUDA. Genuine pinned
contour order, hierarchy, public values and metadata agree on the tested domain.
Nested tracing/composition has bounded independent review and 21 passing focused
tests, including exhaustive small masks and genuine pinned public observations.
The deterministic ellipse compatibility path is now reviewed for its guarded
runtime-5.0.0 domain. An initial 180/0 angle mismatch revealed solver sign noise;
that atan2 boundary and near-isotropy now withhold the whole bundle without angle
normalization or relaxed tolerance. Fourteen transformed/reversed contour comparisons,
post-solver cancellation and preallocation caps pass. The 49-case size/native/Qt/
register gate has 48 passes plus its corrected targeted test passing separately.
Eight frozen-model Naive Bayes primitive tests and bounded review also pass. Its
recipe/API/desktop integration is now reviewed with fifteen passing tests, including
asset rereads, image-cache preservation, actual class previews and cancellation.
Explicit size thickness and nonverbose diagnostics are reviewed and integrated.
Widths one/five match public values/pixels, including the actual isolated worker's
one-pixel fixture. The size/raster gate passes 48 tests; native working/API/Qt
passes 24, including effective editing, size-only invalidation, preview/reset/save.
Complete ellipse/drawing/classifier domains remain open. Texture's allocation
regression stays below128MiB. Fresh full discovery is required before hand-off.

An opt-in portable checkpoint candidate preserves selected archived outputs and their
complete upstream closure in snapshots. The new **Create snapshot with reference
checkpoints…** action selects the source store explicitly; restoration retains it in
the restored project folder's `reference-checkpoints/`. Eight new tests plus existing
snapshot/register coverage pass 30 tests in 5.640 seconds. Raw bytes, dependency
identities and original eligibility are preserved. Bounded independent review approved
the four frozen bundle/snapshot/Qt/test files, with hashes confirmed by root. Fresh full
verification remains pending; project metadata alone does not bind or verify that store.

Native chip-matrix extraction, affine correction and polynomial correction candidates
pass six focused tests in 7.082 seconds, including genuine pinned public calls.
Corrected pixels agree within one byte/channel on tested inputs. Image/mask reductions
and tiled pixel calculations remain on CUDA; only compact fit metadata crosses to
CPU NumPy. Bounded independent primitive source/test review is approved; standard-card
orientations/masks are also reviewed. Full domains, automatic detection, project/batch
integration, fresh full verification and independent colour-accuracy evidence remain open.

Per-label RGB/HSV/LAB analysis passes eight focused tests against pinned public
observations, including bins/traits/types/scales, circular hue and positive-only S/V.
The combined colour/correction/register gate passes 20 tests in 32.429 seconds with
source hashes and dependency versions enforced. CUDA histograms use bounded tiles;
only compact counts/observations transfer to CPU. Missing/empty HSV/ALL calls retain
explicit reference-failure states. Bounded independent primitive review is approved. Degenerate hue coverage,
full project/batch integration and fresh full verification remain open.

The new CUDA priority-queue watershed candidate matches all stages of the isolated
pinned distance → ordered peaks → connected markers → final labels chain on the
3120x2080 fixture. Watershed took 23.630 seconds and 109,250,048 bytes additional CUDA
allocation for 4,777,682 foreground pixels. Five watershed tests and six register tests
pass in 30.946 seconds, including wrapper-pin checks and cancellation during both
initialization and flooding. PyTorch's bundled runtime compiler preserves the pinned
default queue semantics without a separate toolkit. Bounded independent wrapper/kernel/NVRTC/test review is approved at matching hashes.
Archived checkpoint bindings and complete recipe outputs remain open; live graph
execution is described below. The historical full suite does not cover these files.

Seed Fiddle already provides a local desktop workflow for loading, calibration
diagnostics, annotation, procedural/optional learned proposals, adaptation, review,
export and portable project snapshots. Complete peer recipes and operation coverage
remain unfinished. Original SMPTS conformance requires unavailable source material;
scientific promotion requires reviewed independent laboratory data, traceable physical
and colour measurements, trait definitions and prespecified acceptance limits.

The canonical native seed-calculation candidate now executes the three authored
14-stage graphs with selected controls, actual typed fan-in and node-local caching.
**Run native calculation…** opens a separate editable working view; it supports
real asynchronous CUDA execution, bounded previews, Stop/worker-draining close and
UTF-8 selected-recipe save. The same `run_native_recipe_path` API accepts saved
recipes. Historical reference attachments remain unchanged. A 44-test integration
gate passes in 66.254s; the subsequently expanded seven-test recipe gate passes in
22.517s, including complete binary/mean/supported-Otsu chains against pinned public
calls. Actual 1440×900 Qt rendering was inspected. Actual decoder/NVRTC/driver and
loaded-code identities are retained. A subsequent 14-test gate in 39.029s adds the
returned hue-chart data and native Qt display: exact public-return rows, valid empty
RGB/LAB charts, sample selection without recomputation and stale-chart clearing.
Supported size observations now appear in the inspector with honest label/count
controls and immutable metadata. Unsupported topology and the overlay keep runs partial.
Bounded source/test reviews are approved. Complete project batch/report/checkpoint
integration and fresh full discovery remain open.

## Complete peer-feature status (reconciled 8 October 2026)

Candidate means implemented and locally tested but not yet delivered. Review status is
recorded separately. Functionality, reference agreement and scientific validity remain
independent; engineering tests do not close F01–F22 acceptance.

| Feature | Current implementation | Remaining work |
|---|---|---|
| F01 — Image decoding, ROI and coordinate preparation | **Partial native.** Loading, source identity, corrected coordinates and ROI transport exist. | Freeze reference decoding, orientation, channels and ROI contracts. |
| F02 — Spatial calibration | **Partial native.** Ruler/dish scale estimates, diagnostics and conversion machinery exist. | Reference calibration contracts and independent dimensional validation. |
| F03 — Neutral balance and color-card correction | **Partial native / bounded primitives and working view reviewed.** CUDA chip means, affine/polynomial correction and standard-card matrices/masks match tested pinned public calls. The four-port native RecipeSpec, shared four-file runner and API/Qt whole-run export have bounded review and focused public-call/cache/provenance tests; prior view rendering was inspected. | Automatic-card working recipe/Qt/project/batch integration, broader input domains, fresh full gate and colour validation. |
| F04 — Selected threshold recipes | **Bounded implementations reviewed; LAB and native recipe candidates tested.** CUDA binary, adaptive mean, restricted Otsu and BGR-slot/HSV/CMYK/gray ranges; exact LAB passes eight range tests and exhaustive byte-colour comparison. All three threshold arms now execute in the native seed working view. | Candidate review/full gate, remaining domains, range-stage integration and complete Otsu behavior. |
| F05 — Alternative foreground methods and classifiers | **Partial native / thresholds, frozen-model Bayes and bounded K-means prediction/training workflows reviewed.** Extended thresholds pass22 primitive and14 workflow tests. Bayes has eight inference primitive and fifteen initial workflow tests, plus bounded sample-table and binary image/mask CUDA training, typed project provenance and native asynchronous controls. The current 207-test compatibility gate includes15 new directory integration cases and15 existing sample-table cases. K-means now includes compact model identity, full prediction/masks, full/sampled sigma5 patches, complete MiniBatch fitting, directory selection/materialization, numeric export and native training controls/cache/atomic save and reviewed persistent training provenance. Artifact selection binds complete outer bytes and accepts legacy numeric files. Reviewed source-environment conversion shares standard-library codecs and matches public trained-model predictions/masks. Explicit all-reduction provenance avoids inferred source thread counts. The reviewed conversion/dependency/register gate passes181 tests in81.100s; actual Qt training view inspected. | Complete texture/remaining Gaussian and K-means training domains, Remaining Bayes training domains, complete per-stage report/checkpoint binding, held-out science and fresh full discovery. |
| F06 — SMPTS two-branch foreground construction | **Gap / original conformance blocked.** SMPTS paper graph describes two branches and combination. | Executable reconstructed branches and explicit choices; original source needed for verified reproduction. |
| F07 — Mask cleaning and morphology | **Bounded implementations reviewed.** CUDA median, erosion/dilation and 4-connected small-object removal; four native stage bindings and original/native five-stage benchmark preprocessing. | Remaining ROI domains and complete desktop recipe integration. |
| F08 — Connected components and suspected contacts | **Partial native.** Component labels, statistics and bounds exist. | Reference conformance and SMPTS rectangle contact-screen rules. |
| F09 — Distance transform | **Bounded primitive and stage binding reviewed.** CUDA fixed-point labeled-transform map matches pinned OpenCV on varied masks and the full fixture. | Complete recipe integration and downstream sensitivity. |
| F10 — Peak and marker generation | **Bounded primitives reviewed; live recipe integration tested.** Pinned ordered coordinates/masks and CUDA 8-connected markers feed the canonical native calculation; marker checkpoints exist. | Archived peak-stage binding, remaining domains and complete project/batch integration. |
| F11 — Watershed partitioning | **Partial native / bounded default mode reviewed.** CUDA priority heap matches pinned edge cases and the full-resolution reference chain; canonical native seed graphs execute it. | Fresh full gate, diagnostics, remaining domains, archived checkpoint bindings and project/batch reports. |
| F12 — SMPTS targeted erosion splitting | **Gap / original conformance blocked.** General erosion primitives exist. | SMPTS targeted contact splitting, kernel and decision semantics. |
| F13 — Contour-preserving recovery enhancement | **Gap.** Existing partitioning infrastructure is reusable. | Eroded-core to original-contour recovery and comparative evaluation. |
| F14 — Individual extraction and coordinate mapping | **Bounded native crop/export workflow reviewed.** CUDA per-seed image/mask pairs, half-open rectangles, padding/clipping and source transforms; Qt and batch artifact export. | Complete peer-recipe extraction and original SMPTS semantics. |
| F15 — PlantCV size and shape compatibility | **Partial native / supported observations and diagnostics reviewed.** All17 records and actual public images match at1/5, including the real isolated worker1 fixture. Size/raster48 and working/API/Qt24 gates pass. Effective thickness control preserves historical graphs and size-only caching. Shared limits and96MiB paint scratch remain tested. | Complete ellipse/numerical/drawing domains and protocol coverage, full discovery and laboratory validation. |
| F16 — Canonical research geometry | **Existing native functionality.** Stated acceptance clauses, guarded pooling, source/frame lineage and persisted diagnostics are verified in recorded domains. | Independent accuracy, repeatability and physical measurement validation (L03). |
| F17 — PlantCV per-object color distributions and statistics | **Partial native / bounded observations and chart UI reviewed.** Eight public-reference tests cover numerical observations. Returned hue-chart rows and native display are tested, including empty RGB/LAB, cache-preserving sample selection and repaired failure/cancel rerun clearing. | Degenerate circular results, complete domains, project/batch integration and fresh full gate. |
| F18 — Hilum, pattern and condition features | **Partial native.** Reviewed annotations, supplied-evidence CUDA classifier training/save/activation/cancellation, inference/cache/report consumers and portable frozen-model snapshots are implemented in the declared guarded contract. | Independent expert definitions/truth and locked biological agreement, reliability and abstention acceptance (L04). |
| F19 — QC, method selection and export | **Partial native.** Authoritative review, versioned native archives, source/frame/calibration and model provenance exist; trait-model snapshot closure is verified. | Finite audit of all supported backend IDs, QC revision, desktop/batch record equality, nonzero crop lineage and failed/unavailable preservation. |
| F20 — Batch execution, cache, cancellation and resource control | **Partial native / exact saved native companion execution reviewed.** Batch uses the shared native path and saved selections, source checks, node-local caches and explicit failure/unavailable ledgers; baseline and completed graph CUDA ownership releases before subsequent work. | Original/reference batch execution, correction-session project integration and resource/correction-time study. |
| F21 — Scientific and conformance evaluation | **Infrastructure implemented / tested candidates.** Checkpoint replay, array/mask/partition comparison, grouped manifests and genuine paired threshold reports exist. | Complete operation/recipe coverage, reviewed corpus, grouped uncertainty and comparative study. |
| F22 — Faithful PlantCV and SMPTS graphs in the native node editor | **Graph inspection, portable checkpoints and bounded native views reviewed.** Historical views persist; working graphs have honest controls, caching, project save/reopen and shared whole-run output export. Saved historical and working native selections execute as distinct project companions while preserving historical records. | Remaining numerical/drawing/classifier domains, SMPTS/original execution, correction-session project integration, per-stage replay and fresh full gate. |

## Accepted extension: peer functionality

The project now targets selected PlantCV–SMPTS seed-analysis capabilities under
[the implementation and benchmark plan](PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md).
The [comparison report](PLANTCV_SMPTS_COMPARISON.md) explains the scope and evidence.
WP0/WP1/WP2/WP5 are in progress. The [operation-level register](plantcv_smpts_features.json)
maps F01–F22 to source operations, existing code/tests, gaps and acceptance criteria.
PlantCV 4.11.3 source and the isolated reference environment are pinned in
[the source record](../config/peer_reference_sources.json); the default native
unavailable-policy baseline is captured in [the baseline record](../config/peer_baseline.json).
Exact task-specific reference recipes and outputs still need freezing. Original
SMPTS source access remains unavailable; a reconstruction cannot establish original-code
parity. The benchmark checkpoint API now publishes immutable bounded records and
replays exact source/frame/recipe/backend/environment identities and checks the
expected measurement contract when one is supplied.
Raster roles enforce shape/dtype/semantics, and all size limits precede explicit
benchmark host transfers. Its 24 focused tests pass after independent review,
including exact integer range enforcement above `2**53`.
Versioned recipe contracts now serialize provenance, stages, typed ports,
dependencies, parameters, conditions and run identities, and project them into
the existing `PipelineGraph`. Analytical identity excludes display/layout changes;
extra, missing, miswired and inactive nodes are rejected. Twelve focused tests and
independent Sol review cover this infrastructure. The reviewed deterministic
catalogue now provides three authored PlantCV seed workflows, a colour-correction
contract and a nine-stage SMPTS paper reconstruction. Original PlantCV operations
map to pinned source; watershed internals describe one public call, without
independent replay. Required per-image values remain unselected, so these are
nonrunnable source-inspection graphs rather than frozen benchmark recipes. Twelve
catalogue tests pass. **View → Reference graphs…** now displays all five graphs in
the existing native canvas/inspector. Exact stored versions, stale identities,
unselected parameters and unresolved provenance remain inspectable. The 24 focused
view/read-only tests and final Sol review pass; actual Qt render inspection and
all 52 titles at 100/125/150/200% scale pass. Capacity-limited views remain explicitly
transient, preserving saved metadata. Display operations preserve actual native
cache objects and native editing remains the default. Explicit store binding and
verified pixel inspection exist. Portable restoration and the separate native
seed working view are focused-tested candidates; complete recipe acceptance remains open.
Project schema6 retains historical recipe/run/layout attachments, separate working selections and model assets through
desktop open/save and snapshot round-trips, while safely reading versions1-5.
Forty-seven persistence/UI tests pass with one existing filesystem skip. Historical
checkpoint hashes are retained without embedded arrays. The tested inspection
candidate can verify arrays in a user-selected local store; portable restoration
is still separate work.
The reviewed conformance API now compares primitive arrays, masks and instance
partitions from exact-identity checkpoints. It retains stage/pipeline upstream
provenance, checks raster semantics, uses exact integer errors and reference-relative
float tolerances, and records bounded common-evaluator matched-ID audits. Thirteen
focused tests pass. These reports measure computational agreement; this API always
records independent scientific accuracy as not evaluated. Actual original/native threshold comparisons now run through the tested pair
candidate. Other operations and full recipe orchestration remain open. The reviewed
isolated PlantCV runner executes twelve pinned original operations: binary/mean/Otsu
thresholds, median filtering, erosion/dilation/fill, watershed, size/colour analysis
and manual/automatic colour correction. Twenty focused tests pass with the exact
reference environment. It validates source/dependency identity, inputs/outputs,
RGB boundary semantics, timeout/cancellation and cleanup. Missing ignored source
or environment artifacts report unavailable. The reviewed source-preparation helper
reconstructs exact runner metadata from committed pins, inspects missing prerequisites
without writes/network and supports explicit bounded callback fetching. Nine focused
tests pass, including a source-present root with no flyr wheel. It does not install
the isolated environment or rebuild the retained wheel. Original SMPTS remains unavailable;
desktop/batch recipe orchestration and retained native-reference corpus comparisons
are still pending. The reviewed `run_reference_stage` API now loads an exact input
checkpoint, invokes one pinned public binary/mean/Otsu threshold operation and
archives the validated output with a stage run record. It binds source/frame,
recipe/upstream identity, selected/effective parameters, raw source/environment
pins, raster roles, observations and runtime. Unavailable/unsupported/failed/empty
states remain distinct; scientific eligibility stays false. All 13 focused tests
and 84 final Sol probes pass. It is not a full recipe interpreter or graph preview.
WP2 now has reviewed CUDA binary, adaptive-mean and restricted-domain Otsu primitives.
Inputs and output masks stay on CUDA; only Otsu's 256-bin histogram crosses to CPU.
Mean supports odd kernels 3–255 and finite offsets [-255, 255]. Default pinned IPP
Otsu behavior differs on ambiguous partitions, which the helper rejects. Root's
3,132 accepted histogram probes match the pinned masks and 108 reject explicitly;
mask-equivalent empty-bin plateaus do not promise scalar-cutoff equality. Complete
F04 scope and desktop recipe integration remain open. Actual threshold-pair
reports are implemented; complete comparative corpus coverage remains open. WP3/WP4/WP6 retain their open gates, with original SMPTS parity blocked. Faithful
display of pinned reference implementation graphs in the native node editor remains
the explicit F22 objective. Its reviewed source-inspection delivery covers topology,
parameters, provenance, Qt visual checks and metadata persistence; bounded original-operation checkpoint inspection is reviewed; runnable recipes and portable
array restoration remain required. Existing native methods remain the
engineering baseline. Functional coverage,
numerical conformance, scientific performance and operational cost require separate
proof. Laboratory data collection continues alongside WP0/WP1/WP2/WP5.

## Capability status

| Area | Current state | Remaining evidence or work |
|---|---|---|
| PlantCV–SMPTS peer functionality | WP0/WP1/WP2/WP5 in progress; pinned source/environment and native baseline, reviewed infrastructure and restricted native threshold primitives. | Freeze actual reference recipes and outputs, integrate usable paths and complete WP1–WP6 gates; original SMPTS parity is blocked and no peer-performance claim is established. |
| Runtime and desktop shell | Implemented. One `seed_vision.py` launcher, native PySide6 UI, optional local environment bootstrap, diagnostics, cancellation, and responsive persistence. | Validate the frozen dependency and hardware envelope intended for laboratory deployment. |
| Analysis graph | Implemented. The generated catalogue covers 48 active/toolbox cards. Node local caching and dependency based invalidation are preserved. | New controls must continue to affect calculations and receive optimization coverage or a stated exemption. |
| Calibration and image evidence | Engineering implementation complete for ruler/card detection, deskew, neutral balance, layout, material, edge, texture, lighting, and quality evidence. | Supply traceable card/camera geometry and independent measurements before enabling or claiming validated physical or colorimetric output. |
| Material references | Implemented with Background, Foreground, and Other classes, exclusions, draft/apply/revert history, and source bound persistence. | Independently review reference policy on representative acquisition sessions. |
| Seed annotation | Implemented with brush, edge trace, shape fill, smart fill, eraser, undo/redo, saved reference loading, seed condition/shape metadata, and one shot hilum placement. | Produce complete independently reviewed masks and trait labels. |
| Procedural instances | Implemented as a transparent CUDA first evidence plus bounded CPU topology path and retained annotation bootstrap/fallback. | Establish count, split/merge, boundary, and correction time performance on held out real images. |
| Learned instances | U-Net/watershed and StarDist nodes, training, checkpoints, tiled inference, evaluation, and audits are implemented but disabled by default. | Retrain and evaluate on reviewed real data. Existing results establish synthetic engineering behavior only. |
| Node optimization | Implemented for all 38 computational cards, with 10 explicit exemptions, fixed objectives, project sequencing, proposal review, history, and rollback. | Treat every run as in-sample adaptation; independent performance still needs a frozen evaluation protocol. |
| Species libraries | Implemented with immutable versioning, content pins, source exclusion, import/export, validation metadata, and product specific coverage. | Build and validate reviewed libraries from representative sources. Coverage is not accuracy. |
| Results and export | Implemented with explicit availability, method selection, per-instance review, immutable result revisions, CSV/JSON/annotated-image export, and batch parity. | Physical units and unvalidated traits remain withheld until their validation gates pass. |
| Recovery and provenance | Implemented with source/checkpoint hashes, coordinate bound reference schema 6, project schema6 (versions1-5 migration; working recipes/model assets/correction input bindings), analysis settings schema 20, and immutable portable snapshots. Reference graph metadata persists; a tested candidate verifies arrays from an explicitly selected store. | Unknown legacy coordinate frames require explicit alignment review; portable reference checkpoint-array restoration remains pending. |

## Current annotation behavior

- The application toolbar presents **Annotate: Materials / Seeds**. The annotation
  palette belongs to the application window, can sit beside the image, and retains
  its position.
- Material and seed rasters are shown and edited only over the deskewed corrected
  image. Source, gamut, prototype collage, and other non-deskewed views suppress
  annotations and active drawing previews because those coordinates do not align.
- Seed selection provides **Next empty** and **Next unannotated**. **Show** selects
  all or the current seed. Clear actions share the same compact row.
- **Load matching reference** first restores the current saved applied seed mask,
  then falls back to the bundled source-bound pre-annotation, then offers an
  explicit corrected-coordinate file. A replacement remains an undoable draft.
- **Seed condition** separates **No defects** from defect labels. The two sides are
  mutually exclusive. Outline, **Full length visible**, and pose share one row;
  Complete forces full length visible on and disables it.
- **Exclude from modelling** is an unchecked confirmation control beside
  **Apply + save**. Hilum metadata exposes only **Pick location** and
  **Clear location**. Picking ends after one placement and is cancelled when the
  selected seed changes; direction is derived internally from the painted centroid.
- Toolbar zoom buttons preserve the viewport centre. Wheel zoom remains anchored
  beneath the pointer.

## Validation evidence

The latest full local run used:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

The latest reviewed distance/marker/crop run on 6 October 2026 passes **989 tests in 1070.527 seconds: 987 passed and 2 existing skips**,
with the full-image CUDA fill diagnostic enabled and all **220** source/test/script/
configuration/register hashes unchanged. Log: `artifacts/distance-crop-discovery-corrected-full-suite.log`.
The first attempt omitted the documented `-s tests` discovery option and finished
with two test-import errors; its log is retained as diagnostic evidence in
`artifacts/distance-crop-final-full-suite.log`. Corrected discovery includes those
inspection modules. The preceding reviewed preprocessing run passed 953 tests in
1047.537 seconds with two existing skips and 206 unchanged hashes. Historical
842/881/913/938 results remain in HANDOFF.md and the canonical implementation plan.
Engineering verification does not establish full conformance or scientific accuracy.

The prior frozen infrastructure run on 3 October 2026 ran 738 tests in 343.357
seconds: 736 passed and two existing skips. Log:
`artifacts/peer-infrastructure-final-full-suite.log`. That run included proposed
runner/conformance modules in the working tree; their separate reviews and commits
are recorded above. A passing suite does not replace independent review or close
delivery gates.

During the reference-graph objective follow-up on 3 October 2026, it ran 670 tests
in 347.608 seconds: 668 passed and 2 skipped. The checkout baseline is `dfda336`;
no production code changed. The earlier documentation integration ran in 325.881
seconds; the prior 29 September run took 383.737 seconds. The skips cover an unavailable
complete historical pilot batch and a POSIX case-sensitive filesystem check on
Windows. The node catalogue drift check and `git diff --check` also passed for this documentation integration. The test log is
`artifacts/reference-graph-plan-full-suite.log` (ignored generated output). Software and synthetic tests demonstrate engineering
contracts; they do not establish scientific performance.

## Work needed for laboratory readiness

1. Agree on intended use, error tolerances, rejection policy, and acceptable
   correction time before opening a final test set.
2. Collect and adjudicate representative original images and complete instance
   masks with biological and acquisition group identities.
3. Record traceable ruler/card/camera geometry and independent specimen and colour
   measurements across the image plane.
4. Define coat, condition, damage, wrinkling, visibility, and exclusion labels with
   positive, negative, and ambiguous examples; measure reviewer agreement.
5. Train or adapt only on development data, freeze the complete recipe, and run the
   independent protocol described in
   [SCIENTIFIC_VALIDATION_PROTOCOL.md](SCIENTIFIC_VALIDATION_PROTOCOL.md).
6. Run representative operator, accessibility, resource, interruption, and recovery
   studies on the target laboratory hardware.

## Reading order

Use [README.md](../README.md) to start, [OPERATOR_GUIDE.md](OPERATOR_GUIDE.md)
for operation, [ARCHITECTURE.md](ARCHITECTURE.md) for software contracts,
[NODE_CATALOGUE.md](NODE_CATALOGUE.md) for generated node/control truth, and
[SCIENTIFIC_VALIDATION_PROTOCOL.md](SCIENTIFIC_VALIDATION_PROTOCOL.md) for the
remaining scientific gate. The [documentation index](README.md) classifies all
other records as current guidance, implementation records, historical audits, or
future ideas.

Post-review display verification: all18 card/shared Qt tests pass in38.226s
(`artifacts/automatic-card-working-preview-gate.log`). Actual1440x900 overview,
ROI-labelled-mask and scale views were rendered with Windows fonts and inspected.
The preview now fits its current label dimensions, preserving aspect ratio and
showing all24 chip circles. This display-only repair changes the shared UI hash;
the other six current source/test hashes remain unchanged. The103-test gate
predates the preview repair; the18-test gate includes the12 strengthened card
tests and all shared Qt regressions. Six register tests pass after reconciliation.
Full F01-F22/WP0-WP6 and fresh full-suite/scientific gates remain active.

## Feature register at the calibrated workflow checkpoint

These are family/operation status axes from the full register. A partial implementation remains partial even when its selected engineering fixture passes.

| ID | Feature | Functionality | Conformance | Scientific validation |
| --- | --- | --- | --- | --- |
| F01 | Image decoding, ROI and coordinate preparation | partial_native | not_run, pending_recipe_freeze | not_applicable, not_validated |
| F02 | Spatial calibration | partial_native | pending_recipe_freeze | blocked_laboratory_inputs |
| F03 | Neutral balance and color-card correction | partial_native | not_applicable, pending_test | blocked_laboratory_inputs |
| F04 | Selected threshold recipes | partial_native | pending_recipe_freeze | not_validated |
| F05 | Alternative foreground methods and classifiers | partial_native | pending_recipe_freeze, pending_test | blocked_laboratory_inputs, not_validated |
| F06 | SMPTS two-branch foreground construction | gap | blocked_source | not_validated |
| F07 | Mask cleaning and morphology | partial_native | pending_recipe_freeze | not_validated |
| F08 | Connected components and suspected contacts | partial_native | blocked_source, pending_recipe_freeze | not_validated |
| F09 | Distance transform | partial_native | blocked_source, pending_recipe_freeze | not_applicable, not_validated |
| F10 | Peak and marker generation | partial_native | pending_recipe_freeze | not_validated |
| F11 | Watershed partitioning | partial_native | pending_recipe_freeze, pending_test | not_validated |
| F12 | SMPTS targeted erosion splitting | gap | blocked_source | not_validated |
| F13 | Contour-preserving recovery enhancement | gap | not_applicable | blocked_laboratory_inputs |
| F14 | Individual extraction and coordinate mapping | partial_native | blocked_source | not_validated |
| F15 | PlantCV size and shape compatibility | partial_native | not_run | blocked_laboratory_inputs |
| F16 | Canonical research geometry | reused_native | not_applicable | blocked_laboratory_inputs |
| F17 | PlantCV per-object color distributions and statistics | partial_native | pending_test | blocked_laboratory_inputs |
| F18 | Hilum, pattern and condition features | partial_native | not_applicable | blocked_laboratory_inputs |
| F19 | QC, method selection and export | partial_native | pending_test | not_validated |
| F20 | Batch execution, cache, cancellation and resource control | partial_native | pending_recipe_freeze, pending_test | not_validated |
| F21 | Scientific and conformance evaluation | partial_native | pending_test | blocked_laboratory_inputs, not_validated |
| F22 | Faithful PlantCV and SMPTS graphs in the native node editor | partial_native | pending_test | not_validated |
