# Critical application review: remediation and remaining validation

Updated 2026-09-18. This register follows all 30 findings in
[the original critical review](CRITICAL_APPLICATION_REVIEW_2026_09_08.md).
The original report is historical evidence; this document records current behavior.
The earlier [node optimization implementation](NODE_OPTIMIZATION.md) is preserved.

## Outcome and interpretation

The independently actionable integrity, result-workflow, evaluation, persistence,
UI and resource corrections described below are implemented. Scientific findings
are mitigated in the software and have a prepared validation protocol; they are
**not closed as experimentally validated**. No synthetic test, review flag, source
count, or passing software suite certifies laboratory performance.

“Implemented” means the stated software correction is present. “Prepared; input
required” means the application now avoids unsupported claims and the remaining
experiment needs actual specimens, reference measurements or operator decisions.
Optional refinements and empirical limits are stated explicitly after the register.

## Finding-by-finding register

| ID | Priority | Current status and correction |
|---|---|---|
| R01 | P1 | Implemented. Loose-workspace close resolves unapplied drafts explicitly; failed sidecar saves block close. Cancel preserves edits. |
| R02 | P1 | Implemented. Live/exported features share `pipeline_evidence`; foreground means raw colour evidence. Version 2 identifies this recipe. Ambiguous v1 foreground checkpoints/datasets require re-export and retraining. |
| R03 | P1 | Implemented. Physical contour consumers require connected, complete, shape-reviewed, nonexcluded instances. Partial interiors remain material evidence. Learning export refuses incomplete instance truth. Clipped reprojected contours lose shape eligibility. |
| R04 | P1 | Implemented. Both evaluation paths audit before inference and reject invalid splits. Reports state factual review/provenance and always leave scientific certification false. |
| R05 | P1 | Implemented. Source counts establish provisional coverage, never validated performance. Coverage is per product/class/group; unavailable distances are null. Legacy validated tiers are labeled as unvalidated performance in the manager. |
| R06 | P1 | Implemented. Checkpoints use restricted `weights_only=True` loading and validate dictionary/tensor structure. No unrestricted-pickle fallback. |
| R07 | P1 | Implemented. Learning exports write unique component revisions, verify hashes/audit, then switch the manifest last. Interrupted writes cannot mix old labels with new features. |
| R08 | P1 | Implemented. Reference schema 6 stores the source-to-corrected transform. Matrix transport replaces shape-based resizing. Unknown legacy frames are withheld until explicit visual alignment review; original archives are backed up. Clipped shapes and off-canvas landmarks are downgraded. |
| R09 | P1 | Implemented. Permanent results summary; explicit active method and availability; per-instance accepted/excluded/unreviewed table and ID search; JSON/CSV/annotated-image export with immutable revision identity. Batch uses the same saved production settings and references. Physical units and unvalidated object traits are withheld. |
| R10 | P1 | Prepared; input required. Physical result fields are withheld and pixel-space measurements remain available. Geometry screening checks both axes at multiple locations against declared error bounds. Known card/camera geometry, independent dimensions and an acquisition envelope are needed to implement and certify an appropriate physical calibration. |
| R11 | P1 | Prepared; input required. Reviewed, grouped benchmark protocol and auditing are ready. No machine-generated or bundled unreviewed labels were promoted to human truth. A real adjudicated benchmark remains necessary. |
| R12 | P1 | Implemented safeguards; experiment pending. Development, frozen and reference-assisted evaluation protocols are explicit. Independent evaluation requires source/preprocessing/target-access provenance, reviewed annotations, checkpoint membership and frozen decoder selection evidence. GUI feature export declares image-local adaptation and does not offer locked test export. |
| R13 | P1 | Implemented. Missing foreground evidence yields an unavailable/blocked result with a reference affordance; a valid empty result remains a distinct zero. No legacy count fallback. |
| R14 | P2 | Implemented. Parameter forms wrap rows/labels, expose accessible names and keyboard buddies, and offer collapsible sections. Laptop, 150% and 200% scale renders were inspected; the minimum is now 900x520 logical pixels and initial size is bounded to the screen; operator accessibility validation remains separate. |
| R15 | P2 | Implemented core workflow. Image workspace is the default; results are always reachable; Ctrl+K finds and frames a node and neighbors. Both views cannot leave a persistent blank workspace. Expert graph remains available. Parameter sections collapse independently. |
| R16 | P2 | Implemented. Tool choice and Apply/save precede visible trait and shape metadata. The application-owned palette can move beyond the image view. Palette position and splitter state persist; undo/redo and drawing shortcuts support repeated corrections. Applying edits remains distinct from approving shape/trait metadata. |
| R17 | P2 | Implemented. Unknown/unassigned default, explicit reviewed species for conditioned training, capture-group inheritance without filename invention, and same-source species-conflict auditing. Actual biological labels still need a reviewer. |
| R18 | P2 | Implemented. Explicit verified, immutable portable snapshot packages preserve image/reference/model/library copies separately from shared mutable working sidecars. Restoring creates a new working master. Result review records retain UTC/account-attributed append-only history; supplementary vocabulary/configuration and optimization/review evidence are archived. |
| R19 | P2 | Implemented. Source and checkpoint byte digests govern cache reuse and result provenance. Learned results retain the checkpoint hash actually used; replacing it prevents stale export. Replace-in-place and mid-computation changes invalidate results; references are not silently reassigned to new image bytes. Ordinary node changes preserve upstream caching. |
| R20 | P2 | Implemented. Stop command, cooperative worker/tile cancellation, Qt-timer worker-draining shutdown, and progress-bearing background project/reference/learning/result persistence replace blocking close waits and bulk GUI-thread writes. Individual GPU kernels and atomic filesystem operations finish before cancellation. |
| R21 | P2 | Implemented semantics; calibration needs data. UI/results distinguish support and ranking scores from correctness probabilities. Reliability bins, Brier score and ECE tooling are available; no calibrated accuracy claim is emitted. |
| R22 | P2 | Prepared; input required. Alternative candidates, rejected manual centers and reasons are inspectable/exported. Stratified error tooling includes rejected true seeds. Quantifying size/shape/occlusion bias needs representative reviewed specimens and agreement on relevant strata. |
| R23 | P2 | Implemented terminology; measurement validation needs data. UI describes neutral colour balance. The acquisition protocol requires known reference values and independent checks before making colorimetric claims. |
| R24 | P2 | Prepared; input required. Per-instance reporting explicitly abstains from unvalidated biological trait assignments and inferred full shape. Trait definitions, adjudication, disagreement and object-level evaluation are specified in the protocol. |
| R25 | P2 | Implemented bounded extraction. Shared production-settings and feature contracts remove duplicated execution recipes; results, frame migration, work control, persistence jobs, snapshots and evaluation/resource policies have separate modules. The large legacy controller remains a candidate for incremental refactoring, not a rewrite prerequisite. |
| R26 | P2 | Implemented. Exact missing fixture recovered from Git with its recorded SHA-256; no reference assertions weakened. New independent integrity/workflow/evaluation regressions, Windows CPU/Qt CI and optional configured CUDA-runner checks. Hosted CI has not been executed in this local session. |
| R27 | P2 | Implemented. Global maximum-cardinality/IoU matching, bounded sparse overlap components, explicit empty/unavailable metric policy, micro/image/group summaries and component/source/content identities. Checkpoints retain development membership; decoder reports are checked for selection overlap. |
| R28 | P2 | Implemented bounds. Byte-budgeted training LRU, disk-budgeted evaluation prediction spool with deterministic cleanup, lazy labels, bounded thumbnails and compressed byte-budgeted undo/redo histories. Real-corpus throughput and resource envelopes still require the representative corpus. |
| R29 | P3 | Implemented. Current operator/architecture guides, archived historical README, generated node/control catalogue with drift check, scientific validation protocol and this status register supersede historical implementation notes and obsolete batch examples. |
| R30 | P3 | Implemented core conveniences. Redo, drawing/stop/result/search/view-export shortcuts, seed-ID and decision filters, next matching reviewed/unreviewed seed, previous/next annotation, per-image computation/save status icons, completed-method comparison and persistent layout. Accessibility labels and wrapping were improved. Screen-reader/operator conformance is not claimed. |

## Verification

- **Final full unittest suite: 667 tests run; 665 passed, 2 skipped**, in
  294.339 seconds. Command: `.venv/Scripts/python.exe -m unittest discover -s tests -v`.
  Local log: `artifacts/critical-remediation-final-pass.log`.
- The two skips are the unavailable complete eleven-image historical pilot batch
  and a POSIX case-sensitive filesystem test on Windows.
- Focused integrity regressions: **20 passed**, including small CPU end-to-end U-Net,
  StarDist and hybrid evaluation, strict JSON, restricted pickle rejection, decoder
  overlap rejection, source/checkpoint replacement, spool budgets, cancellation and
  clipped-frame eligibility. The pickle test additionally verifies the restricted
  loader's specific rejection, rather than accepting an unrelated schema failure.
- Focused UI/workflow regressions: **76 passed** after the screen-size correction;
  final annotation-layout/workflow checks: **22 passed**. Result-dialog interactions
  verify decision filtering, next-match navigation, comparison and review history.
- Actual saved-project batch smoke test passed with the optional species library
  absent; unavailable foreground exported as unavailable rather than zero.
- Generated catalogue drift check, Python compilation and `git diff --check` passed.
- Native offscreen Qt renders were inspected at 1366x768 and 1280x800 logical pixels
  (150% scale), and 960x540 logical pixels at 200% (1920x1080 physical). All 48 node
  inspectors stayed within the laptop window; largest inspector minimum width was
  307 logical pixels. Viewport export with its interpretation key and pixel scale
  was also rendered and inspected. These checks do not certify accessibility.
- Exact restored fixture: `images/IMG_9689c.JPG`, 402495 bytes, SHA-256
  `9c256238eb376322f7a229aeac964366bb3ed94159b5389869d69f7100e9e28e`.

Local logs/screenshots are under ignored `artifacts/critical-*`; generated artifacts
are not committed fixtures. Intermediate failures were corrected and covered by the
final passing run. Hosted CI has not run in this local session. Software tests and
synthetic data do not establish laboratory counting, geometry or trait accuracy.

## Compatibility and operational limits

- Back up existing projects before migration. Unknown old annotation frames are
  deliberately withheld; changing a version number or resizing masks is not migration.
- Ambiguous v1 foreground feature recipes require re-export/retraining. Restricted
  checkpoint loading intentionally rejects arbitrary pickled Python objects.
- Portable snapshots relocate verified analytical dependencies. Supplementary review
  and optimization evidence is retained with its original identity in the snapshot
  index; it is not automatically applied to relocated result/source identities.
- View export preserves the viewport pixels and appends the overlay name, interpretation
  key and a pixel-space scale bar. Physical units remain withheld.
- Cancellation is cooperative; it cannot interrupt a running CUDA kernel or safely
  abandon the middle of a filesystem commit. Progress dialogs keep Qt responsive.
- Independent evaluation checks recorded provenance; an honest upstream export and
  study conduct are still necessary. Interactive annotation-adapted exports cannot
  be relabeled as independent evidence.
- Formal accessibility certification, realistic-corpus resource envelopes and
  operator correction-time acceptance require representative users, hardware and data.
  Engineering layout checks cannot substitute for those studies.

## Inputs needed to finish scientific findings

The [scientific validation protocol](SCIENTIFIC_VALIDATION_PROTOCOL.md) gives the
collection, adjudication, freeze and analysis procedure. The remaining inputs are:

1. **P1: intended use and tolerances.** Decisions the application supports; acceptable
   count, boundary, dimension and trait errors; acceptable correction time.
2. **P1: reviewed benchmark.** Representative original images, whole-region reviewed
   instance masks, species/lot/session/physical-seed grouping, empty/negative controls,
   reviewer identity/revision and independently adjudicated examples.
3. **P1: physical calibration.** Exact reference-card geometry, camera/lens/acquisition
   details, coplanarity/height constraints, horizontal/vertical measurements throughout
   the dish, and independent specimen dimensions. These select the correct calibration
   model instead of guessing a card aspect ratio or camera model.
4. **P2: colour and traits.** Traceable colour-reference values, expert trait definitions,
   trait truth and agreement/adjudication policy, and the biologically relevant bias strata.
5. **P2/P3: operator validation.** Target hardware, typical project sizes, operators and
   accessibility requirements for performance, correction-time and usability studies.

No original fixture is needed from the user: the missing file was recovered exactly.
