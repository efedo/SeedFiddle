# Native seed-trait training

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

Public reference QC (13 October): shared QC-v4 records actual complete/incomplete public captures and preserves exact acknowledgment through private project snapshots. This does not approve trait labels or establish biological validity. Classifier evaluation still needs explicit linkage of audited grouping/access, actual training evidence and frozen model identity.

Catalogue source fidelity (12 October): the independently authored fixed catalogue and Qt injection guard reject altered audited-function definitions. This identity check does not approve trait labels or establish biological validity. F19 acceptance concerns the evaluated recipe/prediction request; it does not require unrelated training/library-management terminal features.

Recipe batch QC (12 October): actual row outcomes and distinct incomplete batch publication records retain exact QC/acknowledgment dependencies through private snapshot restoration. Historical successful row evidence does not verify current archive availability, approve trait labels or install predictions. Training producer boundaries and independent biological acceptance remain open.

Correction attempt QC (12 October): actual four-file and saved-project correction attempts preserve failure and cancellation facts with returned source/target metadata separate from unverified outputs and scientific eligibility. Exact acknowledgments survive original removal and private snapshot/resnapshot. These records never approve trait labels or install historical predictions. Batch/training boundaries and independent biological acceptance remain open.

Recipe attempt QC (12 October): generic native recipe failure and cancellation preserve bounded selected-file observations and actual invocation facts with unknown archive/node/measurement/science facts. The shared viewer and mixed v1/v2 private snapshot history preserve exact acknowledgments after original removal. These records never authenticate operators, install historical predictions or approve seed/trait labels. Correction/batch producers, wider training boundaries and independent biological acceptance remain open.

Project QC history (12 October): explicitly retained native/reference QC views and multiple exact analytical-revision acknowledgment events survive private snapshot restoration and original artifact removal. Masters sharing a folder remain isolated. Actual restored MainWindow QC reopening preserves current analysis and annotation drafts; historical records do not authenticate accounts, install predictions or approve seed/trait labels. Whole-recipe terminal QC, wider training boundaries and independent biological acceptance remain open.

Shared recipe QC (12 October): whole verified native/paper exports and recorded selected reference inspections retain exact provenance and outcomes. Separate acknowledgment binds only that QC revision and recorded account/time; measurement/scientific eligibility remains unknown. Loading QC never installs predictions, authenticates authorship or approves seed/trait labels. Project/snapshot QC transport, wider training boundaries and independent biological acceptance remain open.

Terminal snapshot evidence (12 October): native execution/preflight terminal records and recorded acknowledgment events survive verified private project restoration and original workspace removal. Historical acknowledgment does not authenticate an account, install predictions or approve seed/trait labels. Actual trait/model snapshot consumers retain their contracts. Shared recipe QC, wider training boundaries and independent biological acceptance remain open.

Desktop terminal/history evidence (12 October): rejected profile and annotation imports retain selected-file attempt evidence and preserve the tested current graph/drafts. Historical terminal acknowledgment/export does not install predictions or approve seeds/traits. Completed revision-v7 and frozen classifier/training contracts retain their prior meaning. Wider training boundaries, terminal snapshot transport and independent biological acceptance remain open.

Native preflight terminal records (12 October): malformed supplied requests and scoped preparation failures retain separate immutable attempt evidence, while uncaptured request/image/calibration/prediction facts remain unknown. Acknowledgment of a failed attempt cannot approve seed or trait predictions. Actual malformed catalogue and Qt preparation witnesses pass; earlier desktop boundaries and historical transport remain open. Completed revision-v7, frozen classifier/training contracts and independent biological acceptance retain their prior meaning.

Native execution terminal records (12 October): failed or cancelled analysis now has separate immutable supplied-request evidence with uncompleted prediction/calibration facts explicitly unknown. Reviewing or acknowledging that terminal record cannot approve seed or trait predictions. Completed result revision-v7, frozen classifier/training contracts and independent biological acceptance requirements retain their prior meaning.

Native backend outcome integration (12 October): revision-v7 binds explicit instance producer states alongside actual stored trait predictions. A missing or failed learned instance branch can retain a supported procedural result, with both outcomes visible. The classifier consumes the actual selected result; successful empty instances remain empty. Cancellation and fatal CUDA errors still abort the run. Frozen trait model, training and biological validation requirements are unchanged.

Portable result identity (12 October): revision-v6 uses the verified frozen model bytes and actual prediction facts while preserving the local model path and executable settings in report evidence. Exact saved decisions can survive verified snapshot/restore; changed bytes, state, threshold or inputs cannot inherit the prior approval. Restored decisions remain private unless you explicitly save a new workspace review. Opaque legacy/missing selections and historical revision contracts receive no inferred approval migration. Root192 distinct checks pass; this is software portability evidence, with biological validation still required.

Saved desktop review decisions now reach batch reports for the same unchanged native analysis in the same project location. Reports bind the captured inputs and actual stored predictions under revision-v5; raw project-save and route facts remain separate evidence. Layout-only saves keep that review identity. Annotation, biological-context or selected-model byte changes invalidate prior approvals while preserving their historical files. Relocated model paths still affect identity; portable inference does not yet imply portable review approval.

Use **Learning → Train object-level seed traits…** to choose an evidence manifest, output file and training controls. The job shares the desktop's serial CUDA worker. Cancel and Stop request a stop before the next calculation or file commit. **Learning → Activate frozen seed-trait model…** also activates an existing model. Activation validates the artifact and graph before changing the Object-level seed traits node.

Saving and activation are separate. The default saves without activating. Save to a new file when a model is already active. A failed or cancelled job before publication preserves the previous output; cancellation after publication keeps the valid saved file and skips activation. A graph change during training also skips automatic activation. Only the classifier and its dependents are invalidated on successful activation; reusable upstream caches remain available.

Definitions, truth, permissions and biological grouping must be supplied. The program checks their consistency and isolation; it does not establish their biological authority. Calibration diagnostics and predictive probabilities do not establish scientific validity. Independent locked testing remains required.

The UTF-8 JSON manifest has exactly these root fields:

```json
{
  "format": "seedfiddle-trait-training",
  "version": 1,
  "feature_schema": "seed-classifier-pixel-v1",
  "vocabulary": {
    "revision": "your-definition-revision",
    "definition_source": "your supplied definition reference",
    "definition_source_sha256": "replace with 64 lowercase hexadecimal characters",
    "traits": [
      {"identifier": "class_a", "definition": "Your explicit definition", "evidence_reference": "Your definition evidence"},
      {"identifier": "class_b", "definition": "Your second explicit definition", "evidence_reference": "Your definition evidence"}
    ]
  },
  "examples": []
}
```

This shows the envelope, not a ready-to-train dataset. Supply 2–64 unique trait definitions and 2–10,000 examples, including permitted examples in both splits and every defined class in training. Definition IDs use lowercase letters, digits and underscores, beginning with a letter. Replace the digest placeholder with your supplied definition-source digest. That digest is recorded as supplied provenance, not independently verified expert approval.

Each example has exactly these fields:

```json
{
  "example_id": "unique-example",
  "physical_id": "your-stable-physical-seed-id",
  "group_id": "your-biological-lot-and-capture-group",
  "trait_id": "class_a",
  "truth_provenance": "your adjudicated label reference",
  "permitted_use": true,
  "split": "train",
  "input": {
    "kind": "reference_seed",
    "image_path": "capture.png",
    "image_sha256": "replace with the actual source-file SHA256",
    "archive_path": "projects/reference-regions/your-archive.npz",
    "archive_sha256": "replace with the actual archive-file SHA256",
    "seed_id": 1
  }
}
```

Splits are `train` and `calibration`. Example, physical-seed and group identities cannot overlap splits. All captures from the same biological group must retain that group identity. The program also rejects identical actual source/selected-mask evidence across splits, even when declared identities differ. It cannot independently establish whether supplied groups represent independent biological samples.

Paths resolve relative to the manifest directory; absolute paths also work. The reference archive must contain the requested seed, a recorded source-to-corrected transform and a reviewed complete outline without an exclusion. Known physical-seed annotations must agree with the example. Canvas-border seeds and pixels outside decoded source support reject. Archive/source identity and known annotation-generation identity must match the exact consumed source snapshot; absent generation history remains unknown.

Alternatively, `input` can contain exactly:

```json
{
  "kind": "compact_features",
  "values": [0.1, 0, 0.1, 0.1, 0.1, 0.2, 0, 0.2, 0.2, 0.2, 0.3, 0, 0.3, 0.3, 0.3, 2.772588722239781, 0, 1, 0, 4],
  "provenance": "Your explicit feature-source reference"
}
```

This illustrative row describes a uniform-colour 4×4 square. The twenty finite values must follow the exact schema: red, green and blue each contribute mean, population standard deviation, q10, q50 and q90; then log pixel area, log bounding-box aspect, bounding-box fill, moment anisotropy and grid perimeter divided by square-root area. Colours are unit RGB; area and perimeter refer to visible corrected-frame pixels. Geometry is resolution-dependent and carries no physical scale. Supplied compact rows record supplied provenance without claiming verified extractor lineage.

Set `permitted_use` to false to retain an excluded fact in the audit. Its `input` can be null; the program never interprets or opens denied input contents. Fact identities still participate in split-isolation checks.

Manifest input is bounded to 16 MiB. Each compressed image snapshot is bounded to 512 MiB and each compressed archive to 256 MiB; oversized inputs reject without resizing. Full-resolution decoded images use the existing CPU decoder, while correction, feature rasters and fitting stay on CUDA. The frozen finite JSON model is bounded to 4 MiB and contains definitions, controls, split/input audits, evidence identities, weights, normalization, calibration and development diagnostics. It contains no executable checkpoint. Loading never retrains it.

## Portable project snapshots

A selected Object-level seed traits model is included even while its node is disabled. Create Snapshot validates the exact finite JSON bytes within4MiB and records byte count, raw SHA256 and canonical model SHA256. The snapshot contains a content-addressed object; Restore Snapshot verifies it and changes the stored selection to a path within the restored workspace. Neither operation retrains the model or changes the original model. Model paths resolve relative to the learning/workspace root, matching native and saved-project batch analysis.

An explicit `None` selection creates no model asset. Missing, invalid, oversized, changed or inconsistent selected models stop completion. Earlier snapshots without a selected trait model remain readable. Earlier snapshots that selected a trait model without storing it in the new inventory cannot supply complete portable closure and are rejected. Restored supplied definitions and audit facts remain supplied provenance; they do not establish expert approval or biological accuracy.

## Reading stored predictions in review

The results table’s Predicted trait column shows the stored supplied class ID or an explicit abstention reason. Hover for the supplied definition/evidence, model identity and maximum predictive probability where available. That probability does not establish correctness. The summary counts assigned, abstained and unavailable stored rows. Reviewed annotation traits remain independent and are never replaced by these predictions.

Opening review does not fit or run the classifier. Changing the abstention threshold and recomputing can change assignments while preserving probabilities. Saved decisions apply only to their exact prediction revision; earlier review bytes remain historical. A probability exactly equal to the selected threshold qualifies, including an exact1 at threshold1.
