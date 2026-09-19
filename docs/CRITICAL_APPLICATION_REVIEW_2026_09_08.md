# Seed Fiddle: critical application review

**Review date:** 8 September 2026  
**Reviewed revision:** `60a4b8bb3af1c34f54c4945231699fafa496a694`  
**Scope:** application behavior, native desktop interface, scientific methods, validation, persistence, performance, and maintainability.  
**Disposition:** review report only; no application fixes made.

**Additional review, 9 September 2026:** the [node optimization compliance review](D:/Programming/SeedFiddle/docs/NODE_OPTIMIZATION_COMPLIANCE_REVIEW_2026_09_09.md) assesses all 48 active/toolbox cards against the requirement for annotation-guided optimization and a project-wide sequential command. It adds 11 prioritized findings, including incomplete fitting coverage, ineffective procedural trials, and a changing edge-evaluation target.

Seed Fiddle is a substantial research and annotation environment, with useful image diagnostics and unusually extensive engineering regression coverage. It is not yet a dependable, complete scientific counting and measurement product. The principal problems are broader than segmentation accuracy: ordinary unsaved work can be lost, training and inference disagree about an input channel, partial annotations can become incorrect boundary supervision, and validation reports can overstate the evidence supporting them.

The highest-value next work is to repair those correctness and data-integrity problems, make one complete review-and-export workflow dependable, and establish independent real-image validation. Adding further diagnostic nodes or model complexity should follow evidence that they improve that workflow.

## 1. What was examined and what the evidence establishes

The working tree was clean at the start. The review began with `HANDOFF.md` and `AGENTS.md`, followed by the launcher, project/profile/reference persistence, graph execution and caching, Qt controller/view/inspector/annotation components, calibration, material and edge models, procedural separation, measurement and shape models, reference libraries, and learning export/training/evaluation. Tests and previous scientific/engineering reports were cross-checked against current behavior.

This was a risk-based review of the major implementation paths, not a claim that every line or all parameter combinations were exhaustively verified. An AST inventory counted **76 implementation Python files and 72,609 lines** under `seedvision/`. The current default graph contains **29 active cards, 19 unused cards, and 179 active connections**. There are **309 parameter entries** across active and unused cards; this count includes entries that are not ordinary numeric controls. These are current measurements, not the older catalogue counts in the handoff.

### Fresh verification

| Check | Result |
|---|---|
| Runtime diagnostics | Ready; Python 3.12.10, PySide6 6.11.1, NumPy 2.5.1, OpenCV 5.0.0.93, PyTorch 2.13.0+cu126 |
| Acceleration | NVIDIA GeForce RTX 3070, CUDA available |
| Full `unittest` discovery | **611 run: 608 passed, 2 skipped, 1 failed; 293.406 seconds** |
| Failure | Reference manifest names absent `images/IMG_9689c.JPG` |
| Skips | Complete eleven-image ruler batch unavailable; case-sensitive-filesystem test inapplicable on this host |
| Actual Qt widget rendering | Fusion style with Windows Segoe UI fonts; 1100×700, 1366×768, and 1920×1080 logical window sizes |
| Fresh sparse-image analysis | Actual background worker completed; zero procedural instances without Foreground examples |
| Material-supported sparse analysis | Actual Apply/save and worker rerun completed; 15 procedural instances; legacy proposal count remained zero |
| Additional reproductions | Loose-workspace close, feature-channel mismatch, partial boundary supervision, export interruption, invalid evaluation gate, library tier assignment, source-cache staleness |

The UI checks used the actual PySide6 widgets, actions, Qt event processing, and production background worker, with persistence redirected to an isolated directory under `artifacts/`. They were not browser mockups. They do **not** establish physical-monitor DPI behavior, screen-reader accessibility, or measured usability with lab operators. A first worker harness attempt was interrupted and replaced with an event loop that yields to Python worker threads; its delay is not counted as an application performance defect.

The sparse fixture was `IMG_9670c.JPG`, 3120×2080 source pixels. A fresh run took approximately **13 seconds** including worker startup and result installation; ruler detection accounted for about **5.7 seconds**. For the second run, interiors from the **unreviewed** bundled mask were eroded with a 9×9 kernel and used solely as positive **material** examples. No instance labels or annotation markers were supplied to the separator. This diagnostic rerun took approximately **6.4 seconds** and produced 15 instances against 16 visually identifiable dish seeds. It also left several visible silhouettes incompletely covered. This is an illustrative failure case, **not** a held-out accuracy estimate or a hand-validated mask benchmark.

### Evidence files

All generated evidence remains in ignored `artifacts/`; no fixture was changed.

- [Full test log](D:/Programming/SeedFiddle/artifacts/critical-review-full-suite-2026-09-08.log)
- [Reproduction results](D:/Programming/SeedFiddle/artifacts/critical-review-2026-09-08/reproductions.json) and [reproduction script](D:/Programming/SeedFiddle/artifacts/critical_review_probes.py)
- [Qt review harness](D:/Programming/SeedFiddle/artifacts/critical_review_ui.py) and [final worker log](D:/Programming/SeedFiddle/artifacts/critical-review-ui-foreground.log)
- [UI observations](D:/Programming/SeedFiddle/artifacts/critical-review-2026-09-08/ui-observations.json), [fresh analysis](D:/Programming/SeedFiddle/artifacts/critical-review-2026-09-08/analysis-observations.json), and [material-supported analysis](D:/Programming/SeedFiddle/artifacts/critical-review-2026-09-08/foreground-observations.json)

These links are local evidence, not committed assets. Preserve the evidence directory separately if the report needs to travel with its screenshots and logs.

## 2. Priority register

**P0:** immediate, broadly applicable catastrophic failure; none established in this review.  
**P1:** required before relying on the affected workflow or making scientific claims. Some concern normal editing; others apply specifically to learning, imported checkpoints, or quantitative use.  
**P2:** important correctness, usability, robustness, or maintainability work for the next development cycle.  
**P3:** useful refinement after the main workflow is dependable.

“Confirmed” means demonstrated by code and/or a bounded reproduction. “Gap” identifies an absent capability or insufficient evidence, not a claim that every existing result is wrong. Suggested redesigns below are recommendations; acceptance criteria describe how to decide whether the improvement is complete.

| ID | Priority | Improvement | Evidence type |
|---|---|---|---|
| R01 | P1 | Protect unapplied work in loose-workspace sessions | Confirmed data-loss path |
| R02 | P1 | Make exported and live model features identical | Confirmed calculation mismatch |
| R03 | P1 | Exclude incomplete/unreviewed contours from physical-edge training | Confirmed supervision-contract defect |
| R04 | P1 | Enforce evaluation audits and remove automatic scientific certification | Confirmed validation defect |
| R05 | P1 | Replace source-count-based library validation tiers | Confirmed misleading tier assignment |
| R06 | P1 | Stop unrestricted checkpoint unpickling | Confirmed unsafe loading path |
| R07 | P1 | Make learning-sample replacement transactional | Confirmed partial-write corruption |
| R08 | P1 | Bind annotation rasters to their calibration coordinate frame | Confirmed schema/transport gap |
| R09 | P1 | Complete an authoritative results and export workflow | Confirmed capability gap and legacy mismatch |
| R10 | P1 | Establish metrologically valid geometry and scale | Scientific gap in measurement assumptions |
| R11 | P1 | Establish a reviewed, independent real-image benchmark | Scientific validation gap |
| R12 | P1 | Separate annotation-assisted adaptation from independent evaluation | Scientific protocol/provenance gap |
| R13 | P1 | Distinguish missing evidence from a valid zero-count result | Confirmed fresh-user workflow failure |
| R14 | P2 | Fix clipped inspector content and narrow-window layouts | Visually confirmed |
| R15 | P2 | Make the graph navigable and optional for routine work | Visually confirmed usability problem |
| R16 | P2 | Put annotation actions before optional trait metadata | Visually confirmed usability problem |
| R17 | P2 | Make species and biological context explicit and reliable | Confirmed default/provenance risk |
| R18 | P2 | Offer immutable project annotation revisions and portable recovery | Documented limitation with scientific consequences |
| R19 | P2 | Invalidate source caches when file contents change | Confirmed stale-read reproduction |
| R20 | P2 | Add responsive cancellation, save progress, and resource controls | Confirmed lifecycle limitations; performance recommendation |
| R21 | P2 | Use honest score and uncertainty terminology | Scientific interpretation gap |
| R22 | P2 | Quantify size-prior, shape-filter, and occlusion biases | Scientific validation gap |
| R23 | P2 | Separate neutral balance from validated color measurement | Confirmed method limitation |
| R24 | P2 | Validate biological traits and define per-seed reporting | Scientific and product gap |
| R25 | P2 | Refactor large controllers and duplicated computation contracts | Measured maintainability problem |
| R26 | P2 | Restore a green fixture baseline and strengthen independent tests | Confirmed test gap |
| R27 | P2 | Harden evaluation metrics and dataset provenance | Confirmed implementation limitations |
| R28 | P2 | Bound learning/annotation memory at realistic dataset sizes | Confirmed unbounded cache; scaling not benchmarked |
| R29 | P3 | Consolidate current documentation and remove obsolete instructions | Confirmed documentation drift |
| R30 | P3 | Add operator conveniences and accessibility verification | Suggested improvements |

## 3. Required correctness and integrity improvements

### R01 — Unapplied annotations can be lost when closing a loose workspace

**Evidence:** `MainWindow.__init__` starts with `_project_tracking_enabled = False` (`main_window.py:1206` vicinity). `closeEvent` only resolves unapplied drafts and prompts about saving inside that flag's branch (`main_window.py:13501`). The ordinary startup image-folder workflow therefore receives different protection from an explicit New/Open/Save Project session.

**Reproduction:** A real window loaded a small image, received an actual reference-edit slot call, and had a nonempty dirty set. Closing returned `True`, invoked neither draft resolution nor a warning, and wrote no archive. The probe would have answered Cancel had a warning appeared.

**Impact:** A user can paint useful work and close the application without being told that it will disappear. The project's existence should not determine whether annotations deserve protection.

**Required:** Always resolve dirty material/instance drafts and failed autosaves before close or workspace replacement. Keep project-master saving separate from reference saving. Test loose startup, explicit projects, multiple dirty images, failed saves, and Cancel. Success means every edited draft is either persisted or explicitly discarded by the user.

### R02 — Learning export and live inference use different foreground channels

**Evidence:** `learning/export.py:20` maps the model channel `foreground_colour` to `result.foreground_probability`, which is the resolved material decision after fusion. Live `segmentation/baseline.py:2172` maps that same channel to `foreground_colour_probability`, the retained raw foreground color response. The assignment separating raw and resolved products occurs around `baseline.py:1780`.

**Reproduction:** The mapping probe distinguishes the two sources. On the material-supported real-image run, their mean absolute raster difference was **128.55 on the displayed 0–255 scale**, so this is not merely naming drift.

**Impact:** Training/evaluation on exported features can look successful while desktop inference receives a different distribution. Existing feature-spec equality checks cannot detect this because channel names are unchanged.

**Required:** Centralize feature assembly at one production interface and test exported-versus-live tensors on the same analysis, for every supported channel. Version the semantic change, record which representation is intended, and rebuild or explicitly mark affected datasets/checkpoints incompatible. Do not assume that renaming a variable repairs existing learned artifacts.

### R03 — Applying a partial seed patch turns its artificial border into physical-edge supervision

**Evidence:** The UI supports unknown/partial outlines and applies their rasters (`main_window.py:9677`). `instance_boundary_references` explicitly treats every applied positive ID as a reviewed complete region (`annotation/instance_references.py:48`). The CUDA edge-prototype path accepts only the label raster, not outline/review metadata, and derives normals and training boundaries from it (`cuda/layers.py:6067`).

**Reproduction:** A 16×16 interior patch produces **60 physical-edge training pixels**. The derivation has no argument with which to exclude its incomplete outline. The current brush instruction also encourages painting an “interior mark,” which conflicts with the boundary model's completeness assumption.

**Impact:** Scribbles, partly traced outlines, visible fragments, and machine-generated masks applied for editing can teach the physical-edge classifier incorrect borders. Shape-model eligibility safeguards do not protect this separate consumer.

**Required:** Separate material-positive scribbles, partial visible-instance annotations, and reviewed complete contours. Pass an explicit eligible-ID set or boundary-validity mask to every edge-learning/export/library path. For partial outlines, retain only explicitly confirmed physical arcs; artificial crop/occlusion closures need an unknown region. Test that changing an ID's eligibility changes only its eligible consumers and does not convert an incomplete border into training truth.

### R04 — Evaluation can report scientific validation even when the split audit fails

**Evidence:** `evaluate_checkpoint` loads a manifest and checks feature-spec equality but never calls `audit_manifest` (`learning/evaluation.py:223`). Its `scientifically_validated` field depends on a manifest boolean, `split == "test"`, and reviewed flags (`evaluation.py:361`). It does not verify checkpoint training membership, duplicate source content, or the provenance of decoder selection.

**Reproduction:** A real manifest contained the same capture group in train and test. The real audit returned invalid with an explicit leakage error. The real evaluator still completed and wrote `scientifically_validated: true`. Only the forward model and decoder were mocked to keep this gate test small; manifest handling, audit, metrics, and report generation were real.

**Required:** Audit before inference; reject leakage and incompatible evidence. Bind evaluation to immutable checkpoint, dataset-content, and decoder-selection identifiers. Report factual states such as “reviewed test split evaluated,” with audit outcomes and limitations. A software boolean cannot certify scientific validity. Test that invalid grouping, repeated images under new IDs, and training/test overlap cannot receive a validated label.

### R05 — Library tiers imply validation without demonstrating performance

**Evidence:** `reference_library/validation.py:42` chooses tiers from source count; two capture groups anywhere in the library can promote a particular product to `MULTI_CONTEXT_VALIDATED`. Distance diagnostics do not gate the tier. Missing cross-group comparisons can return zero, which is indistinguishable from a perfect numerical result without additional context.

**Reproduction:** A foreground-color product contributed by three sources, all in group A, became `multi_context_validated` because an unrelated fourth source was in group B. Its reported product group count was **1**, leave-one-source-out distance **10,000**, cross-group distance **0**, and publication eligibility **true**.

**Required:** Compute evidence coverage separately for each product, class, and context. Use terms such as “three-source coverage” until appropriate held-out performance is established. Represent unavailable metrics as unavailable, with fold/support counts. Publishing a provisional artifact may remain useful, but publication and validation must be separate states. Ensure product-level failures cannot be hidden by a clean global warning list.

### R06 — Imported checkpoints are loaded with unrestricted pickle execution

**Evidence:** `learning/checkpoint.py:62` calls `torch.load(..., weights_only=False)` before validating the checkpoint's schema. This path is reachable from checkpoint selection for training/refinement and inference.

**Impact:** A checkpoint from another person or downloaded source can execute code with the desktop user's privileges. No malicious checkpoint was constructed or executed during this review. PyTorch's own guidance warns that unrestricted loading uses unpickling and requires trusted input. [PyTorch loading documentation](https://docs.pytorch.org/docs/stable/generated/torch.load.html).

**Required:** Prefer tensor/state-dictionary loading with constrained primitive metadata, `weights_only=True` where compatible, and strict validation of the accepted payload. Test all application-generated checkpoint families against the constrained loader. If a legacy format requires unrestricted loading, provide an explicit, exceptional trusted-file conversion path. Restricted loading reduces this attack surface; it should not be described as a universal security sandbox.

### R07 — “Atomic” learning export can leave old labels paired with new features

**Evidence:** `export_learning_sample` replaces the final feature NPZ before writing the labels, optional rasters, display image, and finally the manifest (`learning/data.py:235`). The operation is not transactional across those files, despite its docstring.

**Reproduction:** During replacement of an existing sample, an injected label-write failure left the old manifest unchanged but changed the feature mean from **0 to 1**. A later reader therefore receives a mixed revision.

**Required:** Validate the complete proposed manifest, including group/split rules, before writing. Stage every component in a new revision directory, validate it, then atomically switch a manifest/reference to that complete revision. Preserve the old sample if any step fails. Add failure injection at each write/rename boundary, including replacement and disk-full scenarios.

### R08 — Saved raster annotations are not bound to the transform that defined their coordinates

**Evidence:** Reference archives persist source digest and corrected dimensions, but no source-to-corrected homography or calibration-recipe identity (`persistence/reference_regions.py:179`). `_aligned_reference_mask` and `_aligned_instance_annotations` resize mismatched rasters to the new dimensions with nearest-neighbor interpolation (`baseline.py:2645`, `2693`). Equal dimensions are treated as aligned. In contrast, manual centers explicitly retain source coordinates and are reprojected through the current transform.

**Impact:** Changing deskew/perspective settings or upgrading calibration can invalidate pixel correspondence even when image shape remains the same. Resizing is not the geometric transformation from an old calibrated frame to a new one. Existing archive shape checks prevent some mistakes but cannot detect same-size frame changes.

**Required:** Store source-coordinate labels or the exact original transform plus coordinate-space/schema identity. Reproject categorical masks using `H_new × inverse(H_old)`, validating labels, landmarks, and boundaries; alternatively withhold them until the user resolves the mismatch. Do not silently substitute resizing. Add same-size translation/rotation and changed-canvas tests, not just dimension checks.

### R09 — There is no complete authoritative results/export contract

**Evidence:** The desktop exposes learning export and mask saving but no general per-seed measurement/count/trait report export. `seedvision/export/__init__.py` is only a package docstring. The documented `scripts/analyze_pilot.py` writes `result.count` and circles from `result.proposals`; `BaselineAnalysis.count` is the length of the legacy proposals, not the active procedural output. The real material-supported run had **15 procedural instances and zero legacy proposals**. The batch script also calls `analyze_path` without loading project settings or applied references.

The large count summary is visible only when selecting legacy `identification` or `output` cards (`main_window.py:12969`), both outside the normal active workflow. The procedural inspector does expose its own count, but that is not a unified results surface.

**Required:** Define the authoritative result source explicitly. Provide an always-accessible image result summary, selectable per-instance table, review status, accepted/excluded count, units, and export to CSV plus structured JSON and optional annotated image. Include method/settings/source/calibration/library/model/annotation revision identifiers. Export visible 2-D measurements separately from inferred shape parameters. Make batch execution consume the same project recipe and references; a missing result must never silently become a legacy zero.

### R10 — Geometric correction is not yet a validated metric rectification

**Evidence:** `_projective_deskew_matrix` uses average *observed* opposing side lengths to choose the destination rectangle (`calibration/image.py:2831`). It has no known physical card aspect ratio. A foreshortened rectangular reference can consequently remain foreshortened while receiving a “perspective corrected” state. Downstream dimensions use one scalar pixels/mm derived from a ruler direction. A coherent tick lattice establishes a useful directional length reference, not proof of isotropic metric scale across the seed plane.

The project also needs explicit treatment of lens distortion, card/ruler coplanarity, seed elevation, and geometric differences between the rim and the measured seed face. These are measurement assumptions, not evidence that every nearly overhead fixture is badly distorted. Planar homography methods require the relevant geometric correspondences and plane assumptions. [OpenCV homography tutorial](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html).

**Required:** Use known physical reference geometry or a validated camera/acquisition model; verify horizontal and vertical scale at several dish locations. Compare measured seed dimensions with an independent physical/reference imaging method. Propagate or bound scale and pose errors. Withhold quantitative physical measurements outside the validated capture envelope, while retaining pixel-space review.

### R11 — Real-world scientific performance remains unestablished

**Evidence:** Bundled instance masks are marked unreviewed; the committed learning manifest contains one unreviewed training sample and is scientifically ineligible. Earlier learned-model results explicitly demonstrate synthetic-to-real failure. The missing fixture further reduces the reproducible baseline. This review found no committed independent real-image benchmark sufficient to establish claimed counting, boundary, dimension, or trait performance; it does not claim that no privately reviewed annotations exist on the user's machine.

**Required:** Build a versioned, human-reviewed benchmark, stratified by species, lot, imaging session, density, contact/overlap, pattern, glare, and size. Include empty dishes and non-seed objects. Review whole evaluation regions, including false positives, and adjudicate a subset with a second annotator. Hold out entire biological/capture groups and freeze the final test set. Synthetic tests remain valuable engineering checks but cannot establish laboratory accuracy.

Choose several metrics appropriate to the task rather than treating count agreement or pixel overlap alone as success. Image-analysis validation guidance explicitly distinguishes object detection, segmentation, robustness, and reliability. [Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z).

### R12 — Annotation-assisted analysis needs an explicit evaluation protocol

**Evidence:** Applied instance interiors can train material evidence; their contours train edge prototypes; their shapes can set scale and priors. Procedural fitting removes direct instance markers, but its cached upstream evidence may already have learned from those same annotations. `cuda/material.py:109` estimates source reliability on the fitted reference regions themselves. Reference-edge fitting withholds alternating instance IDs, but repeatedly uses that fold to select settings (`reference_edge_fit.py:139`, `277`). These are different forms of adaptation, not independent final tests.

**Required:** Define and label three modes: image-local interactive adaptation, frozen pretrained inference, and few-shot/reference-assisted inference. Record exactly which annotations each mode may use. In a final evaluation, held-out targets must not affect preprocessing, scale estimation, library assembly, feature export, or hyperparameter selection unless that reference access is part of the declared deployment protocol. Test the entire pipeline for leakage, not just the final decoder. A tuning fold is validation data, not a locked test.

### R13 — A missing Foreground reference is presented as a completed zero-result analysis

**Evidence:** Mandatory Foreground evidence is correctly zero without an authored source. The fresh sparse-image run nevertheless ended with “Generated 0 reviewable procedural instances” and a completed procedural node. Its diagnostic warning text was assigned to `warning_label`, which is permanently hidden (`main_window.py:4895`, `6423`). No visible prompt led the operator from the zero result to the missing reference.

**Required:** Distinguish “not enough evidence to analyze” from “analyzed an empty region.” Keep calibration/diagnostic stages usable, but block or mark downstream results unavailable when mandatory evidence is missing. Show a durable readiness summary with a direct action to supply the needed reference. Show pertinent analysis/calibration warnings next to the result, not only in logs, tooltips, or a different node. Test that an empty dish and an unconfigured seed image cannot produce the same result state.

## 4. Desktop and operational improvements

### R14 — Inspector clipping obstructs ordinary use

At 1366×768 **and** 1920×1080, long node titles, descriptions, field labels, and controls extended beyond the right pane. Horizontal scrolling exists, so the fields are not necessarily permanently inaccessible; the operator must repeatedly move sideways to read a setting and then edit it. The default side-panel sizing and unwrapped forms are the relevant seams (`main_window.py:4759`, `pipeline_inspector.py` form construction).

**Improve:** Reflow labels above controls when narrow; wrap titles and explanations; use sensible expanding editors; remember splitter widths. Test every inspector at its supported minimum width with real fonts, long enum labels, keyboard focus, and 125–200% DPI. Acceptance means key labels and values are readable together without horizontal scrolling.

![The actual 1366×768 window: clipped inspector and an initially cable-dominated graph](D:/Programming/SeedFiddle/artifacts/critical-review-2026-09-08/02-open-sparse-1366.png)

### R15 — Routine users encounter the pipeline's implementation complexity immediately

The initial selected node is the advanced reference-dimensions/shape stage, and the graph initially shows a portion of a large network dominated by cables. Fitting the entire graph makes fine detail too small to read. Both Image and Pipeline can also be switched off, leaving the workspace blank; this is currently allowed and tested.

**Improve:** Default to the image and a short workflow summary: acquisition → references → analysis → review → export. Preserve the full graph as an expert workspace. Add node search, “show selected node and neighbors,” collapsible operation groups, and a simple return-to-result action. Prevent an unexplained blank workspace or display a recovery affordance. Keep typed ports and dependency-aware caching; simplify presentation rather than removing useful diagnostics.

### R16 — Annotation metadata dominates the panel before the drawing tools

In the actual annotation view, the floating panel was **430×638 pixels** within an **838×688-pixel** image widget at laptop size. Coat/condition/outline/pose/hilum controls preceded Brush, Trace, Fill, and their options. Apply/save was below the visible portion. An operator correcting hundreds of seeds must repeatedly manage a large overlay and scroll past metadata that may be irrelevant to the current task.

**Improve:** Put seed selection, tool choice, Undo, and Apply/save first and keep commit state visible. Collapse optional biological/shape metadata. Offer a docked annotation panel or a smaller movable palette, remember its position, and provide a compact keyboard-assisted review sequence. Clearly distinguish “save editing progress” from “approve for a scientific consumer,” especially after R03. Verify full paint → undo → switch ID → apply → reopen flows without panel hunting.

![Actual annotation view: tools are below optional metadata and the panel covers much of the image area](D:/Programming/SeedFiddle/artifacts/critical-review-2026-09-08/09-annotation-image-only.png)

### R17 — Species defaults can silently mislabel analyses and learning exports

The initial species selector displayed **Soybean** on the lupin-like sparse fixture; this is a default selection, not an inferred identification. Learning export uses the current selector and defaults the capture group to the filename stem (`main_window.py:9871`). Related captures can therefore acquire different default groups, and a project can lack biologically meaningful grouping despite valid strings.

**Improve:** Start with Unknown/unassigned unless verified project metadata exists. Make the scope of a species/context change explicit. Require reviewed species metadata for species-conditioned training and inherit biological/capture groups from recorded context, not just a filename. Flag conflicting same-source species labels and unknown provenance. This does not require automatic species recognition.

### R18 — Project saving is not a frozen scientific snapshot

The project format deliberately references canonical mutable sidecars. Two masters can therefore observe the same later annotation edits, even after Save As. Hash-change detection is valuable, but it does not recreate the old annotation revision. Missing image records are preserved safely, yet restoring a portable working set is still a manifest/path-management task.

**Improve:** Offer an explicit immutable analysis snapshot or content-addressed annotation revision, distinct from the working project. Include a portable project-package option with verified copies/references, relocation/relink controls, and an auditable history of annotation approval. Preserve existing shared-sidecar behavior for users who want it; make its scope visible when saving or forking a project.

### R19 — Source caching ignores changed file contents

`analyze_path` reuses `raw.image` when the resolved path matches and no explicit source dirty flag is set (`baseline.py:508`). A reproduction analyzed an image with mean value 10, replaced the file with mean 240, and obtained mean **10** again from the cached path.

**Improve:** Track size/mtime and a content digest policy at source ownership boundaries. On change, invalidate that image and its dependents, re-check annotation bindings, and notify the user. Preserve upstream caches for ordinary display or node-only edits. Test replace-in-place, same-name relinking, and external edits between image switches.

### R20 — Cancellation and persistence need operator-visible responsiveness

The serial CUDA worker, coalesced revisions, activity card, and node-boundary cancellation are sound choices. However, the normal analysis workflow lacks a straightforward Stop action, and cancellation waits for the current node. `closeEvent` waits ten seconds and then calls an unbounded `waitForDone()` on the GUI thread (`main_window.py:13560` vicinity). Compression, hashing, and learning export also occur synchronously in UI paths.

**Improve:** Add Stop with explicit cancellation state, bounded cooperative checks inside long loops, and asynchronous shutdown/progress that continues processing Qt events. Move heavy persistence/export work to suitable workers with immutable input snapshots. Surface peak memory and actionable CUDA-allocation errors. Benchmark cold/warm execution, node edits, overlay switching, source switching, save, and cancellation latency separately. This review's small timing sample is not a latency guarantee.

## 5. Scientific merit of the pipeline

The overall approach is scientifically plausible as an **interactive hypothesis and annotation system**. It separates material identity from geometric boundary support, exposes intermediates, uses seed-relative scales, and combines complementary cues. None of those properties alone establishes accuracy. The main concern is whether the resulting interpretation, prior strength, and uncertainty are supported by independent data.

| Pipeline portion | Merit worth preserving | Main limitation and required evidence |
|---|---|---|
| Reference card and ruler | Explicit evidence overlays; metric/imperial disagreement; refusal to derive scale from endpoints alone | R10/R23: establish metric geometry and color validity; test missing/wrong card, glare, tilt, and spatial scale variation |
| Dish localization | Soft physical-size prior with visual rim evidence; explicit geometry | Predetermined vessel/acquisition envelope and approximate elliptical handling need boundary-seed/crop sensitivity tests |
| Foreground/background/other color | Independently inspectable class support; positive-reference learning avoids arbitrary hard label overwrites | R13/R21: missing-reference readiness; coverage bias and uncalibrated support values |
| Multiscale noise/texture | Useful complement to color on patterned or near-background seeds | Image noise, focus, compression, illumination, and biological texture can be confounded; test incremental benefit |
| Material fusion | Explicit ambiguity and unknown mass; avoids forcing background and Other to compete incorrectly | Normalized masses are not empirical correctness probabilities; correlated sources and in-sample reliability need validation |
| Gradients, ridges, oriented traces | Shared device-resident calculation, explicit geometry, tangible boundary diagnostics | Internal coat edges and true contacts can have similar local evidence; local convexity is a bias, not a proof of seed identity |
| Reference edge prototypes | Tangent-aligned strips and supported edges are a reasonable targeted classifier | R03/R12: contour eligibility and adaptation leakage; negative examples are largely derived from heuristics |
| Oval/center evidence | Proposal-independent fits can recover useful center support under weak edges | Oval/convex assumptions can suppress atypical, damaged, or partly visible seeds; quantify failure by shape class |
| Procedural separation | Transparent bounded watershed, alternatives, manual marker correction, inspectable rejection | R22: false merges/splits, extent underreach, prior-driven selection, and confidence calibration remain unresolved |
| U-Net + watershed | Suitable compact architecture; separate boundary/pattern targets and tiled inference | R02/R11/R12: export mismatch, domain shift, feature provenance, independent real data |
| StarDist | Useful compact representation for sufficiently star-convex visible objects | It assumes a representable radial outline; it cannot recover an arbitrary occluded outline from a single photograph. The original method validates star-convex polygon prediction, not this seed domain. [Original StarDist paper](https://arxiv.org/abs/1806.03535) |
| Shape measurements and libraries | Complete/partial/pose distinctions, dimensionless fallback, source balancing, and explicit sensitivity are valuable | R08/R10/R22: coordinate and metric validity, correlated repeated seeds, pose bias, and sparse hierarchy support |
| Traits | Explicit reviewed negatives and separate overlapping conditions are appropriate | R24: semantic truth, biological interpretation, and per-seed summary accuracy remain unestablished |

### R21 — “Probability,” “confidence,” and “uncertainty” require tighter semantics

Gaussian/prototype compatibility, weighted material masses, procedural geometry scores, and learned sigmoid outputs are not automatically calibrated frequencies of correct classifications. Summing to one is insufficient. Fusing related color/texture/edge signals may double-count correlated evidence. The current held-in reference reliability is useful adaptively but optimistic as an estimate of generalization.

The learned uncertainty heads predict current prediction-error targets (`learning/losses.py:80` vicinity), while shape uncertainty perturbs boundaries and supplied scale uncertainty. Neither is automatically a calibrated confidence interval or a measure of all model/domain uncertainty. The shape documentation already correctly calls this sensitivity; preserve that distinction.

**Improve:** Call unvalidated outputs support/compatibility scores, document units and normalization, and distinguish diagnostic uncertainty sources. Measure reliability diagrams, Brier score, and risk versus retained coverage on independent groups before exposing probability-of-correctness claims. Modern learned classifiers can be miscalibrated even with strong accuracy. [Guo et al., calibration study](https://proceedings.mlr.press/v70/guo17a.html).

### R22 — Size and shape priors can bias the population being counted

The procedural path uses one processing diameter and hard/soft bounds on area, span, solidity, concavity, protrusion, axis ratio, and overlap (`procedural.py:123`). Those are reasonable regularizers for ordinary seeds. They can reject exactly the small, immature, split, wrinkled, or unusual seeds that biological phenotyping needs to retain. Partial visibility changes observable area and outline; rejecting it as geometrically implausible is not equivalent to proving no seed exists.

**Improve:** Separate detection validity from phenotype atypicality. Keep excluded/rejected objects reviewable with a reason and uncertainty. Compare single-diameter and distribution-aware priors, full-resolution and working-resolution boundaries, and seed-size strata. Report count, visible-face shape, and any inferred full-seed geometry as distinct outcomes. Never infer total overlapping objects or 3-D dimensions from unsupported single-view completion.

### R23 — The color correction is neutral balancing, not full colorimetric calibration

`_neutral_balance_gains` identifies low-chroma observed swatches and applies channel gains (`calibration/image.py:2803`); correction includes gamut compression. It does not fit known reference values for the colored patches, characterize the camera response, or report residual color error. This can improve appearance and local consistency, but cross-image stain/coat-color measurements need additional evidence.

**Improve:** Document supported acquisition, card identity, lighting, exposure, focus, and white-balance assumptions. If quantitative color transfer matters, validate a suitable correction model on held-out patches/captures and report residuals in a specified color space. Do not fit extra correction parameters solely to improve appearance. Acquisition sensitivity is real in seed analysis: the SeedCounter evaluation explicitly studied different devices and lighting conditions. [SeedCounter evaluation](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01990/full).

### R24 — Trait outputs need biological ground truth and an object-level contract

The species-specific annotation vocabulary and independently reviewed condition negatives are good foundations. Pixel-level resemblance to labeled material is still an incomplete basis for a per-seed conclusion such as immature, stained, split, or wrinkled. Shadows, hilum structure, coat pattern, damage, and focus can produce similar appearance. The dedicated classification package is currently a placeholder; trait diagnostics elsewhere should not be mistaken for a finished specimen-level classifier.

**Improve:** Define each trait operationally with illustrated annotation guidance and expert review. Measure inter-rater agreement and class-specific performance. Define aggregation from visible pixels to an instance, minimum visible area, abstention, and how mixed/unknown evidence is exported. Report “visible-face appearance” where that is what was measured; avoid interpreting appearance as viability, composition, or whole-seed condition without an independent study.

## 6. Engineering, evaluation, and sustainability

### R25 — Large files and duplicated contracts make regression prevention expensive

Measured sizes include `main_window.py` **13,614 lines**, `cuda/layers.py` **8,287**, and `image_view.py` **6,177**. `analyze_image` spans **1,879 lines**. The default graph builder spans **3,773 lines**, although much is declarative. Settings, graph ports, execution, feature mappings, overlays, and status text remain connected by manually maintained identifiers and adapters. R02 is a concrete example of such drift.

**Improve:** Extract project/session lifecycle, annotation state, worker coordination, and results presentation behind narrow interfaces. Give node computation a typed input/output contract shared with graph, export, and tests. Generate repetitive UI/overlay bindings where feasible. Preserve the single `seed_vision.py` launcher, native Qt architecture, GPU ownership, and node-local invalidation. Avoid a rewrite; move one bounded responsibility at a time with meaningful compatibility checks.

### R26 — A large passing suite is valuable but the baseline is not green

The missing reference fixture remains a real reproducibility failure, and one related batch test is skipped. Documentation has normalized this state across many updates. No `.github` workflow directory was present in the reviewed tree. Existing tests strongly cover UI contracts, migration, synthetic geometry, and cache behavior, but the fresh review reproductions demonstrate gaps at cross-module and workflow boundaries.

**Improve:** Recover the exact intended fixture through its provenance, or make an explicit reviewed dataset-version change; do not silently remove the assertion or replace the photograph with a merely similar file. Require green CPU/headless contract checks plus an appropriate CUDA integration run. Add regressions for R01–R08/R13/R19, independent expected results, interrupted saves, and lifecycle transitions. Add metamorphic tests for rotation, scale, source replacement, annotation eligibility, and display-only changes. An exposed-control audit should test numerical effect where appropriate, not only that a control is listed and serialized.

### R27 — Evaluation and dataset provenance need further hardening

Beyond R04, `evaluate_instances` uses descending-IoU greedy matching (`learning/metrics.py:183` vicinity). At the usual 0.5 threshold, exclusivity limits ambiguity, but its configurable lower thresholds can require a maximum-cardinality/global assignment to avoid undercounting matches. Boundary tolerances are fixed in pixels and need interpretation after canonical resizing. Empty-reference cases and unavailable metrics require explicit reporting policy.

Learning samples store paths and grouping strings, not immutable source/content/calibration/feature-generation identities. Checkpoint training metadata hashes the manifest JSON, but that does not fingerprint files subsequently changed under the same paths. Audit grouping cannot detect the same image exported under another identifier/group. The export UI defaults to a filename group, even when richer biological context exists.

**Improve:** Use global matching for the advertised threshold range; test ambiguous assignments and empty dishes. Record metric definitions and both micro/image/group summaries. Hash all learning components and record preprocessing, calibration, annotation eligibility, model/library recipe, and biological source identities. Preserve checkpoint training-source membership so evaluation can audit overlap directly. Report unavailable measurements as explicit missing values, not suggestive zeroes.

### R28 — Dataset-scale memory behavior is not bounded by the image cache policy

The application's GPU image cache is bounded, which is a strength. `SeedTileDataset._cache`, however, retains complete float32 feature stacks, labels, and targets for every loaded sample (`learning/data.py:436`, `526`). Evaluation retains predictions for all samples before decoding/search (`evaluation.py:229`). Per-image annotation/draft histories also have a different lifetime from the bounded GPU analysis cache.

**Improve:** Use bounded sample caches or storage suited to tile access; stream ordinary evaluation; explicitly budget retained outputs needed for decoder search. Measure host RAM, GPU memory, disk use, and interactive latency at realistic source resolution and dataset sizes. One 18-channel 1600×1600 float32 stack alone is about **176 MiB**, before labels and targets. The existing small synthetic set does not demonstrate scalability to a useful real training corpus.

### R29 — Documentation mixes current behavior with obsolete implementation history

The handoff and README retain claims that applied instances directly seed/override automatic segmentation, while the current procedural call explicitly passes `seed_instance_annotations=None` (`baseline.py:2114`). Earlier sections refer to thinned-ridge inputs where the latest Reference edges revision uses gradient magnitude. Old counts, timings, catalogue totals, and eleven-fixture statements coexist with newer behavior. The default Foreground requirement also makes the old batch walkthrough incomplete.

**Improve:** Keep a short authoritative operator guide and current architecture/method reference. Move chronological details to a changelog. Mark every example's dataset version, input references, settings, and model/commit provenance. Generate node/control catalogues from the running graph. Fix visible encoding artifacts and retire screenshots/examples that no longer reproduce.

### R30 — Add convenience features after correctness and workflow repair

Useful follow-ons include Redo; keyboard shortcuts for common annotation tools and next-reviewed/next-unreviewed instance; a searchable seed list; per-image review/computation/save status in the image list; comparison of alternative methods; and export of an exact view with legend and scale. These should serve a defined operator workflow, rather than adding more independent panels.

Accessibility needs direct testing of tab order, focus visibility, screen readers, keyboard access to graph operations, and high-DPI rendering. Complement categorical colors with IDs, patterns, or selection cues. The fixed minimum 1100×700 logical window also deserves testing on small screens at high scaling. No accessibility conformance claim is justified by the offscreen render checks alone.

## 7. What should be preserved

- **Native, local architecture.** There is no need to introduce a server, installer, or bundled executable to address these findings.
- **GPU ownership and lazy presentation.** Device-resident raster intermediates, bounded transfers for topology, and cached display products are appropriate for this workload.
- **Dependency-aware recomputation.** Existing focused regressions test cache and display independence; fixes should strengthen the actual contracts instead of replacing them with blanket reruns.
- **Strong reference persistence safeguards.** Source hashes, strict NPZ parsing without pickle, categorical validation, exact project-sidecar association, and preservation of unresolved records are worth retaining.
- **Annotation tools.** Tiled labels, protected neighboring IDs, compressed undo history, magnetic tracing, previewed fills, and per-image drafts form a useful correction environment once approval and save semantics are repaired.
- **Scientific distinctions already present.** Unknown evidence, partial-review scoring, pose metadata, reviewed-negative traits, provisional model warnings, and explicit limits on synthetic claims are good foundations. Apply them consistently across all consumers and reporting paths.
- **Transparent diagnostics.** The rich graph is valuable for method development and failure analysis. It needs a simpler routine workflow around it, not removal.

## 8. Proposed validation program

The following is a suggested sequence, not a claim that a particular sample count guarantees success. Choose study sizes and tolerances from the laboratory's intended decisions and observed variability.

1. **Define the measurements and review unit.** Specify dish count versus visible-seed count, treatment of overlapping/cutoff objects, physical versus visible boundaries, per-seed length/area, and each trait. Record a capture protocol and exclusion rules before tuning.
2. **Build reviewed reference data.** Use a representative pilot to refine annotation guidance; have two operators independently review a subset and adjudicate disagreements. Keep full-dish truth for final counting metrics, including background false positives. Group repeated captures of the same seed/lot and imaging session.
3. **Freeze development partitions.** Create grouped training, calibration/validation, and locked test partitions with immutable content hashes. Explicitly separate image-local reference-assisted operation from frozen automatic operation. Do not use the locked test to select preprocessing or shape priors.
4. **Compare simple baselines and ablations.** Evaluate a simple material/geometry baseline, the present procedural system, a color-only small U-Net, the richer feature model, and StarDist where shape assumptions apply. Remove one evidence branch at a time. Include no-library versus held-out library, annotation-assisted versus frozen, and scale-prior variants. Prefer the smallest pipeline that gives repeatable benefit. SmartGrain provides a relevant example of explicitly defined seed-shape measurement; it is not evidence of performance on these crowded photographs. [SmartGrain study](https://academic.oup.com/plphys/article/160/4/1871/6109568).
5. **Measure more than counts.** Report object precision/recall/F1, absolute and signed count error, split/merge rates, matched IoU/PQ, boundary error in physically interpretable units, and measurements versus independent references. Add trait confusion matrices, abstention coverage, probability reliability where claimed, and operator correction time. Count agreement can conceal one missed seed and one false object.
6. **Quantify uncertainty by independent group.** Use intervals that respect repeated seeds/captures and clustered lots. Do not treat millions of neighboring pixels as independent biological samples. Report subgroup performance and failure examples, including rare defects that hard geometric filters may suppress.
7. **Test acquisition and operational robustness.** Change lighting, exposure, orientation, focus, density, ruler/card visibility, camera/session, and image resolution within the intended envelope. Test corrupted/missing sources, source replacement, disk-write failures, memory pressure, cancellation, and project recovery.
8. **Run an operator acceptance study.** Observe actual lab users completing a fresh-image → reference → analysis → correction → approval → export → reopen task. Measure completion rate, correction/save mistakes, and time per image. This resolves questions that static rendering cannot.

Acceptance thresholds should be predeclared. Examples include the maximum allowed counting bias, physical length error, missed-defect rate, and correction time; the report deliberately does not invent universal scientific thresholds for the laboratory.

## 9. Recommended delivery order

| Delivery stage | Work | Exit condition |
|---|---|---|
| 1. Protect work and prevent misleading outputs | R01, R02, R03, R04, R05, R06, R07, R13 | Reproductions become regression tests; no silent draft loss, unsafe ordinary checkpoint load, mixed sample revisions, false validation states, or missing-evidence zero result |
| 2. Make a result reproducible | R08, R09, R17, R18, R19, R27 | One source/recipe/reference revision produces an explicit result that can be reviewed, exported, reopened, and traced |
| 3. Make routine review usable | R14, R15, R16, R20 | Lab-size layouts are readable; drawing/commit actions remain accessible; long work can be cancelled and saved with progress |
| 4. Establish scientific evidence | R10, R11, R12, R21, R22, R23, R24 | Grouped real-image and physical-measurement validation supports the actual claims, with failures and uncertainty reported |
| 5. Sustain and scale | R25, R26, R28, R29, R30 | Green reproducible checks, bounded resource use, current documentation, and maintainable ownership boundaries |

Some work can overlap: reviewed-data collection should begin early while correctness fixes are made. Architecture cleanup should follow the seams exposed by those fixes. A larger model or additional handcrafted diagnostic is justified only when an ablation or operator study shows a material benefit.

## 10. Reproduction commands

Run from the repository root using the existing environment:

```powershell
.\.venv\Scripts\python.exe .\seed_vision.py --diagnostics
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe .\artifacts\critical_review_probes.py
.\.venv\Scripts\python.exe .\artifacts\critical_review_ui.py --analyze --foreground-probe
```

The UI harness uses an isolated store under its evidence directory. To reproduce the **fresh** missing-reference state, use a new isolated evidence directory or preserve/rename the prior harness-created store before rerunning; otherwise the material examples saved by its second pass will legitimately autoload. Do not remove real project sidecars or alter `images/` to reset this experiment.

The proof scripts are diagnostic reproductions, not fixes or a replacement for the application's test suite. In particular, the evaluation gate test intentionally mocks model inference; the feature, data, and UI findings document exactly what was exercised. No model was retrained, no malicious model was loaded, no scientific accuracy claim was inferred from the unreviewed bundle, and no user project or source fixture was modified.
