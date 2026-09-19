# Reference-driven node optimization

Implemented 2026-09-18. This guide describes the implementation that addresses
the historical [node compliance review](NODE_OPTIMIZATION_COMPLIANCE_REVIEW_2026_09_09.md).

## Run an optimization

1. Open and save a project. Apply reference edits first; images with unapplied
   reference or instance drafts are excluded. Shape references must be marked
   **shape reviewed**, **complete**, connected, and without a shape-exclusion
   reason. Partial outlines are not silently treated as complete objects.
2. Select **Analysis → Optimize eligible nodes from project references…**, or
   the optimization button on an individual computational node. Both use the
   same planner, production evaluator, search, result review and application path.
3. Select images and nodes. The planner shows reference counts, objective types,
   missing prerequisites and fixed/searchable controls (hover a node). Inactive
   toolbox nodes are not activated. Enable an optional branch before fitting it.
   An enabled learned branch needs a compatible checkpoint.
4. Declare **Every seed reviewed** separately for each image only when coverage
   really is exhaustive. Otherwise disjoint predicted instances are unscored;
   leakage from matched predictions still counts as error. Coverage declarations
   are saved against the filtered annotation hash and reset when those labels change.
5. Choose passes and a per-node evaluation budget. Zero means a complete sweep
   of every currently eligible control. A nonzero budget includes the baseline
   and records exhaustion explicitly. It may end before every control is tried.
6. Inspect the proposal, per-image losses and before/after instance overlays.
   Apply installs one shared set of project parameters. Cancel or reject keeps
   existing settings. Save the project to persist accepted parameter values.

Runs are explicitly **in-sample project adaptation, not independent validation**.
References may train intermediate classifiers and score the resulting pipeline.
An improvement therefore says nothing by itself about accuracy on new images.
Model-weight training remains a separate Learning operation; these commands tune
decoder/analysis controls, never silently train a neural network.

## Coverage and objectives

The shared registry classifies all 48 cards and all 309 controls: 38 computational
cards have optimization capabilities, covering 269 searchable controls before
run-specific exclusions. Ten cards have explicit exemptions. A new unknown node
or unsupported control kind raises an error rather than silently losing coverage.

| Nodes | Objective / required references |
|---|---|
| Deskew and colour balance; Layout detection; Material evidence decision | Foreground/nonseed material probabilities against painted material classes or complete seed silhouettes. Deskew geometry stays frozen; colour balance is searchable. |
| Material colour probabilities | Foreground and background colour probabilities against reviewed material regions. |
| Material noise probabilities | Foreground and background noise probabilities against material regions. |
| Multiscale darkness and colour noise; Reference texture prototypes | Seed-surface/background-texture probabilities against material regions. |
| Wavelet decomposition; Edge gradients; Reference edges | Physical-edge, supported edge, normalized net edge and ridge outputs against frozen contour/interior targets. |
| Reference seed dimensions and shape; Directional surface darkness gradients; Oriented edge traces; Seed-boundary confirmation | Downstream production procedural instance loss against complete reviewed instances. |
| Lightening/darkening gradient ceilings | Their own filtered gradient magnitude against fixed contour/interior targets. These optional diagnostic outputs currently have no instance-separation consumer. |
| Grayscale and local lighting | Downstream instance loss; shadow/highlight controls additionally require reviewed shadow/highlight target rasters. |
| Procedural seed separation | Actual production instance loss. All 35 eligible controls are accessible, including integer controls and formerly omitted surface/oval/shape/candidate controls. |
| U-Net instances; StarDist instances | Instance loss from production decoders in their native GUI units, reusing frozen inference outputs where inputs are unchanged. |
| Reference seed traits | Reviewed coat and condition labels. Unreviewed conditions are not negative examples. |
| Wrinkling | Reviewed wrinkled/not-wrinkled condition labels on complete instances; a weak region-level target, not a wrinkle tracing annotation. |
| Boundary confidence and normals | Boundary magnitude against fixed contours/interiors. |
| Circle candidates; Distance candidates; Identification | One-to-one reference-centre matching, recall, localization and proposal confidence. Unmatched detections count only under exhaustive coverage. |
| Instance masks | Legacy instance labels against complete references. |
| Touching split; Contact graph | Visible adjacency of complete reference instances, or imported split/contact targets. Visible adjacency does not assert hidden overlap order. Split fitting needs a positive touching example. |
| Ellipse likelihood | Contour targets, or imported ellipse-strength targets. |
| Assignment confidence; Proposal disagreement | Correctness/error of a frozen baseline assignment relative to reviewed identities, or imported confidence/risk targets. |
| Image quality; Pattern decomposition; Colour probabilities; Radial profile; Coat damage; Calibration residuals | Explicit reviewed output targets. Seed outlines alone do not establish these diagnostic classes. |

The exemptions are Project, Species/metadata, Species reference library, Ruler
detection, Hue only, Measurements, Review, Classification, Aggregation and Output.
Their present controls represent facts, fixed conversions, or unimplemented
features rather than a fit-worthy calculation. Library blend/model controls are
optimized at their consumers; library identity remains fixed.

Physical dimensions, scoring penalties, coverage declarations, training-source
selectors, checkpoint identities, resolution/tiling limits and annotation
coordinate transforms remain fixed. Ordinary seed references do not supply
independent ruler/perspective geometry. No geometry fit is inferred from them.
The legacy procedural `reference_texture_weight` is disabled in the inspector:
production uses resolved material evidence, so that fallback weight would have
no effect. Missing/zero semantic, ridge, trace and oval evidence produces explicit
per-run exclusions rather than ineffective trials. The lighting deviation,
shadow, highlight and softness controls similarly require their own output labels.

## Objective and search conventions

- Scalar probability outputs use class-balanced binary log loss at fixed reviewed
  coordinates; NaN pixels are ignored. Scores are reduced on the output's CUDA
  device. Only compact scalar results are transferred back.
- Contour targets and their interior buffer are frozen before candidate search.
  Altering a training-buffer parameter cannot redefine the evaluation target.
- Instance matching uses the existing one-to-one asymmetric instance loss, fixed
  scoring weights/seed scale, and a corrected-image evaluation grid bounded to
  1024 pixels on its longest side. A reference that disappears at this scale
  makes that image ineligible; it is not silently dropped from a candidate.
- Each image has equal weight. Per-image scores are retained. No independent
  train/test split or group-generalization claim is implied by the aggregate.
- Search is deterministic coordinate descent: bounded numeric steps, integer
  candidates, boolean alternatives and categorical choices. Ancestor proposals
  are accepted within the isolated run before descendants are optimized. This
  is not a guarantee of a global optimum.
- Final before/after procedural instance scores detect cross-node regressions
  where complete instance references and the procedural branch are available.
  The review reports individual regressions even when the mean improves. If no
  such final metric exists it is displayed as unavailable; local output scores
  remain in the node details. Applying a local improvement is a user decision.

## Reviewed diagnostic targets

Use **Analysis → Import reviewed optimization targets…** after analyzing the
current image. The file is a non-pickled `.npz` archive, bounded to 1 GiB and 256
arrays. It is bound to the original source file and the exact corrected-image
transform. Values are float rasters in `[0,1]`, at the full corrected-image
height/width, with NaN for unreviewed pixels. Do not use the dish-crop dimensions.

Metadata keys:

- `source_sha256`: SHA-256 of the original image bytes.
- `transform_sha256`: `seedvision.optimization.runtime.array_digest()` of the
  analysis's `calibration.affine_matrix`, including dtype and shape.

Supported target keys:

| Node | Keys (prefix each with `node_id__`) |
|---|---|
| `illumination_decomposition` | `shadow_likelihood`, `highlight_likelihood` |
| `image_quality` | `sensor_noise`, `image_quality_risk` |
| `pattern_decomposition` | `pattern__plain`, `pattern__spots`, `pattern__mottled`, `pattern__patches`, `pattern__striped`, `pattern__bicolour` |
| `colour_probabilities` | `colour__white`, `colour__yellow`, `colour__green`, `colour__red`, `colour__brown`, `colour__black` |
| `touching_split` | `touching_split_likelihood` |
| `ellipse_likelihood` | `ellipse_likelihood` |
| `assignment_confidence` | `instance_assignment_confidence` |
| `radial_profile` | `radial_profile_residual` |
| `proposal_disagreement` | `proposal_disagreement` |
| `contact_graph` | `contact_pairs` |
| `coat_damage` | `coat_damage_likelihood` |
| `calibration_residuals` | `calibration_residual_risk` |

`contact_pairs` is the exception to raster shape: an N×3 array of complete
reference-instance ID, another reference-instance ID, and reviewed 0/1 contact.
IDs must exist in the eligible references. Self-pairs and duplicate unordered
pairs are rejected. Missing pairs are unreviewed, not negative examples.

Example construction in a Python session with the current `analysis` result and
an independently reviewed `shadow_target` array:

```python
import numpy as np
from seedvision.optimization.runtime import array_digest, file_digest

np.savez_compressed(
    "reviewed-targets.npz",
    source_sha256=file_digest(analysis.image_path),
    transform_sha256=array_digest(analysis.calibration.affine_matrix),
    illumination_decomposition__shadow_likelihood=shadow_target,
)
```

Imported files remain external references. Keep them available; changed source
files, calibration or target files are checked rather than silently accepted.
Persisted raster targets can be reopened without a pre-existing analysis cache.

## Isolation, persistence and recovery

Candidate settings and caches are isolated from the live graph. Every candidate
invalidates its true producer and downstream consumers; Reference edges also
invalidates the shared Reference texture prototypes producer. Unrelated GPU
intermediates are reused. Per-image caches are bounded by a three-image LRU.
Probability scoring does not download full-resolution rasters. Existing bounded
CPU topology/matching and small Qt preview transfers retain their normal roles.

Cancellation, invalid candidates, source changes and stale project/annotation
snapshots cannot install a partial proposal. Successful application validates
all changes before mutating the live graph. Journal-write failures roll back
settings. The application remains a single native PySide6 process launched by
`seed_vision.py`.

For `example.seedfiddle-project.json`, companion files use the stem
`example.seedfiddle-project`:

- `.optimization.json`: latest proposal and disposition.
- `.optimization-<UTC timestamp>.json`: retained run records.
- `.optimization-policy.json`: per-image coverage declarations and label hashes.
- `.optimization-targets.json`: source hashes mapped to reviewed target files.
- `.optimization-failed-<UTC timestamp>.json`: failure summaries.

Records include original settings, trials, exclusions, per-image scores, proposed
changes, source/reference/transform/checkpoint/library provenance and final
disposition. **Save As** copies the companion records and resolves imported
target paths relative to the original project. Moving projects outside the app
also requires moving their companion files and referenced targets appropriately.

**Analysis → Optimization history / restore previous settings…** lets you select
a run record and restore its pre-run values. Restore refuses to overwrite later
edits to any control on an affected node. It marks the project dirty; save after
restoring. Other node settings are left intact.

## Verification

Contract/search tests cover catalogue classification, typed candidates, frozen
image membership, sequential dependence, shared aggregation, true-owner cache
invalidation, cancellation, invalid-candidate recovery and atomic application.
Qt tests cover node actions, menu dispatch, successful progress-dialog teardown,
stale results, rollback protection and Save As companions. A CUDA integration
test fits two generated images through the real colour → edge → procedural path
and verifies that applying the shared proposal reproduces final scores.

Generated-image smoke tests additionally exercise optional diagnostic adapters.
They validate wiring and execution, not biological accuracy. Real held-out
annotated benchmarks are still required before making accuracy claims.

Validation on 18 September 2026: 66 focused tests passed. Full unittest discovery
ran 638 tests: 635 passed, two skipped, and the pre-existing absent
`images/IMG_9689c.JPG` fixture caused the sole failure. Native plan/result dialogs
were rendered and visually inspected. The committed image fixtures were unchanged.
