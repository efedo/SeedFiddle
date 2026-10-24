# Current project status

Status as of 23 October 2026. This page summarises what is implemented and what
still blocks laboratory use. The roadmap is [PLAN.md](../PLAN.md); the operator view
is the [operator guide](OPERATOR_GUIDE.md).

**In short:** the desktop workflow, the analysis graph, persistence, export, batch
execution and the PlantCV–SMPTS peer pipelines are implemented and covered by a
649-unit test suite. None of the analytical outputs has been validated against
independently reviewed laboratory data. Physical units and trait predictions are
withheld or labelled unvalidated, and procedural counting is not yet competitive
with counting by eye.

## Capability status

| Area | Current state | Remaining evidence or work |
|---|---|---|
| Runtime and desktop shell | Implemented. One `seed_vision.py` launcher, native PySide6 UI, optional local environment bootstrap, diagnostics, cancellation, and responsive persistence. Image and Pipeline views side by side; menus File/Edit/Analysis/Learning/Help. | Validate the frozen dependency and hardware envelope intended for laboratory deployment. |
| Analysis graph | Implemented. Default seed analysis plus 18 factory pipelines; pipelines load Locked and can be saved as custom pipeline files. The generated [node catalogue](NODE_CATALOGUE.md) lists every card. Node local caching and dependency based invalidation are preserved. | New controls must continue to affect calculations and receive optimization coverage or a stated exemption. |
| Calibration and image evidence | Engineering implementation complete for ruler/card detection, deskew, neutral balance, layout, material, edge, texture, lighting, and quality evidence. | Supply traceable card/camera geometry and independent measurements before enabling or claiming validated physical or colorimetric output. |
| Material references | Implemented with Background, Foreground, and Other classes, exclusions, draft/apply/revert history, and source bound persistence. | Independently review reference policy on representative acquisition sessions. |
| Seed annotation | Implemented with brush, edge trace, shape fill, smart fill, eraser, undo/redo, saved reference loading, seed condition/shape metadata, and one shot hilum placement. | Produce complete independently reviewed masks and trait labels. |
| Procedural instances | Implemented as a transparent CUDA-first evidence path with bounded CPU topology, used as annotation bootstrap and fallback. Undercounts touching seeds substantially (see [seed-detection evaluation](SEED_DETECTION_EVALUATION.md)). | Establish count, split/merge, boundary, and correction time performance on held out real images. |
| Learned instances | U-Net/watershed and StarDist nodes, training, checkpoints, tiled inference, evaluation, and audits are implemented but disabled by default. | Retrain and evaluate on reviewed real data. Existing results establish synthetic engineering behavior only. |
| Node optimization | Implemented for 38 computational nodes, with 12 explicit exemptions (`optimization/registry.py`), fixed objectives, project sequencing, proposal review, history, and rollback. | Treat every run as in-sample adaptation; independent performance still needs a frozen evaluation protocol. |
| Species libraries | Implemented with immutable versioning, content pins, source exclusion, import/export, validation metadata, and product specific coverage. | Build and validate reviewed libraries from representative sources. Coverage is not accuracy. |
| Results and export | Implemented with explicit availability, method selection, per-instance review, immutable result revisions, CSV/JSON/annotated-image export, and batch parity. | Physical units and unvalidated traits remain withheld until their validation gates pass. |
| Recovery and provenance | Implemented with source/checkpoint hashes, coordinate-bound reference archives (v7), project documents (v7, reading v1–v6), analysis settings (v23) and immutable portable snapshots. | Unknown legacy coordinate frames require explicit alignment review. |

## PlantCV–SMPTS peer functionality (F01–F22)

All 22 capability families are implemented within their declared software domains.
"Implemented" means the operation runs natively with tested guards; it does not
mean numerical parity with every PlantCV input or scientific validity. Original
SMPTS source code is unavailable, so SMPTS stages are a reconstruction from the
paper. The laboratory gates L01–L04 are defined in the
[scientific validation protocol](SCIENTIFIC_VALIDATION_PROTOCOL.md).

| Family | Current software scope | Source/laboratory requirement |
|---|---|---|
| F01 Image decoding, ROI and coordinate preparation | Implemented declared guarded domain | No additional external input for software; scientific promotion remains separate. |
| F02 Spatial calibration | Implemented declared guarded domain | L03 independent physical accuracy |
| F03 Neutral balance and color-card correction | Implemented declared guarded domain | L03 held-out neutral/color measurements; L03 held-out color patches and sessions |
| F04 Selected threshold recipes | Implemented declared guarded domain | No additional external input for software; scientific promotion remains separate. |
| F05 Alternative foreground methods and classifiers | Implemented declared guarded domain | L01/L02 independent prespecified grouped classifier evaluation |
| F06 SMPTS two-branch foreground construction | Implemented declared guarded domain | Original SMPTS source/license/dependencies; Original SMPTS source and unresolved original semantics |
| F07 Mask cleaning and morphology | Implemented declared guarded domain | No additional external input for software; scientific promotion remains separate. |
| F08 Connected components and suspected contacts | Implemented declared guarded domain | Original SMPTS rectangle-screen semantics |
| F09 Distance transform | F09.1 implemented; F09.2 unresolved source question | Original SMPTS source: confirm exact distance-stage presence or absence |
| F10 Peak and marker generation | Implemented declared guarded domain | No additional external input for software; scientific promotion remains separate. |
| F11 Watershed partitioning | Implemented declared guarded domain | No additional external input for software; scientific promotion remains separate. |
| F12 SMPTS targeted erosion splitting | Implemented declared guarded domain | Original SMPTS source; L01/L02 independent split/merge and erosion errors |
| F13 Contour-preserving recovery enhancement | Implemented declared guarded domain | L01/L02 independent contact/boundary/area/dimension study |
| F14 Individual extraction and coordinate mapping | Implemented declared guarded domain | Original SMPTS source parity |
| F15 PlantCV size and shape compatibility | Implemented declared guarded domain | No additional external input for software; scientific promotion remains separate. |
| F16 Canonical research geometry | Implemented declared guarded domain | L03 independent measurement/coverage study |
| F17 PlantCV per-object color distributions and statistics | Implemented declared guarded domain | L03 independent held-out color measurements |
| F18 Hilum, pattern and condition features | Implemented declared guarded domain | L04 expert definitions/adjudication/agreement and independent locked testing; L04 independent expert truth and representative locked reliability/abstention testing |
| F19 QC, method selection and export | Implemented declared guarded domain | No additional external input for software; scientific promotion remains separate. |
| F20 Batch execution, cache, cancellation and resource control | Implemented declared guarded domain | L04 representative operator correction-time/workflow study |
| F21 Scientific and conformance evaluation | Implemented declared guarded domain | L01/L02 independent prespecified locked study; L03/L04 for relevant endpoints |
| F22 Faithful PlantCV and SMPTS graphs in the native node editor | Implemented declared guarded domain; main workspace verified | Original SMPTS source remains visibly unresolved |

Operation-level detail: [feature register](plantcv_smpts_features.json) and
[method and input-domain inventory](PLANTCV_SMPTS_METHOD_DOMAIN_MATRIX.md).

## Annotation behaviour

- **Materials** and **Seeds** on the toolbar open the reference annotation panel,
  docked beside the image. Annotations are shown and edited only on the corrected,
  deskewed image; other views hide them because their coordinates do not align.
- Seed selection offers **Next empty** and **Next unannotated**; **Show** selects all
  seeds or the current one.
- **Learning → Seed outlines → Import bundled seed outlines…** restores source-bound
  pre-annotations as an undoable draft; **Import seed outlines from file…** accepts
  a categorical label image in corrected coordinates.
- **Seed condition** keeps **No defects** and defect labels mutually exclusive.
  A Complete outline forces **Full length visible**.
- **Exclude from modelling** sits beside **Apply + save**. Hilum metadata is a
  one-shot **Pick location** / **Clear location**; direction is derived from the
  painted centroid.
- Toolbar zoom preserves the viewport centre; wheel zoom stays anchored under the
  pointer.

## Verification

`python -m tests --full` runs 649 functional units in three isolated processes
and passed on 23 October 2026 with four expected skips (an absent eleven-image
pilot batch, an opt-in fill diagnostic, a case-sensitive-filesystem check and a
symlink-privilege check). The four frozen numerical acceptance baselines and 223
frozen recipe profiles are unchanged. See [TESTING.md](TESTING.md).

## Work needed for laboratory readiness

1. Agree on intended use, error tolerances, rejection policy and acceptable
   correction time before opening a final test set.
2. Collect and adjudicate representative original images and complete instance
   masks with biological and acquisition group identities.
3. Record traceable ruler/card/camera geometry and independent specimen and colour
   measurements across the image plane.
4. Define coat, condition, damage, wrinkling, visibility and exclusion labels with
   positive, negative and ambiguous examples; measure reviewer agreement.
5. Train or adapt only on development data, freeze the complete recipe and run the
   independent protocol in [SCIENTIFIC_VALIDATION_PROTOCOL.md](SCIENTIFIC_VALIDATION_PROTOCOL.md).
6. Run representative operator, accessibility, resource, interruption and recovery
   studies on the target laboratory hardware.
