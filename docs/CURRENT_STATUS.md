# Current project status

Status as of 6 November 2026. This page summarises what is implemented and what
still blocks laboratory use. The roadmap is [PLAN.md](../PLAN.md); the operator view
is the [operator guide](OPERATOR_GUIDE.md).

**In short:** the desktop workflow, the analysis graph, persistence, export and
batch execution are implemented and covered by a 317-unit test suite. None of the analytical outputs has been validated against
independently reviewed laboratory data. Physical units and trait predictions are
withheld or labelled unvalidated, and procedural counting is not yet competitive
with counting by eye.

## Capability status

| Area | Current state | Remaining evidence or work |
|---|---|---|
| Runtime and desktop shell | Implemented. One `seed_vision.py` launcher, native PySide6 UI, optional local environment bootstrap, diagnostics, cancellation, and responsive persistence. Image and Pipeline views side by side; menus File/Edit/Analysis/Learning/Help. | Validate the frozen dependency and hardware envelope intended for laboratory deployment. |
| Analysis graph | Implemented. Procedural seed analysis and Cellpose seed analysis on the shared graph; any of their nodes can be added again as an independent instance with its own controls and cache. The PlantCV and SMPTS reproductions were removed in November 2026. Default pipelines stay locked, editing one offers a named copy that the project keeps, and copies can be saved as custom pipeline files. The generated [node catalogue](NODE_CATALOGUE.md) lists every card. Node local caching and dependency based invalidation are preserved. | New controls must continue to affect calculations and receive optimization coverage or a stated exemption. |
| Calibration and image evidence | Engineering implementation complete for ruler/card detection, deskew, colour correction against the chart's reference values (grey balance as the fallback), layout, material, edge, texture, lighting, and quality evidence. | Supply traceable card/camera geometry and independent measurements before enabling or claiming validated physical or colorimetric output. Photos without a vessel are analysed around their confident seeds, less the chart and ruler. With fewer than three confident seeds (most seed piles) the whole image is used, and label text, logos, chart frames and the parts of a ruler beyond its measured span then draw false proposals. Such a pile with no vessel also has no measured seed size: the size is a fixed fraction of the best-fitting circle, often well off (PI542972: 126 px against about 280 px seeds). |
| Material references | Implemented with Background, Foreground, and Other classes, exclusions, draft/apply/revert history, and source bound persistence. Automatic initial references come from confident seeds (Foreground), the visible vessel wall, chart and ruler (Other), and the perimeter ring, seed rings or image border (Background); painting outweighs them. | Independently review reference policy on representative acquisition sessions. |
| Seed annotation | Implemented with seed picking, brush, edge trace, shape fill, smart fill, eraser, undo/redo, saved reference loading, seed condition/shape metadata, and one shot hilum placement. | Produce complete independently reviewed masks and trait labels. |
| Procedural instances | Implemented as a transparent CUDA-first evidence path with bounded CPU topology, used as annotation bootstrap and fallback. Undercounts touching seeds substantially (see [seed-detection evaluation](SEED_DETECTION_EVALUATION.md)). | Establish count, split/merge, boundary, and correction time performance on held out real images. |
| Learned instances | U-Net/watershed and StarDist nodes, training, checkpoints, tiled inference, evaluation, and audits are implemented but disabled by default. **Cellpose seed analysis** runs a zero-shot **Cellpose-SAM** node (optional `cellpose` package, pretrained `cpsam_v2` weights), with cached flows so mask-recovery edits skip inference. | Retrain and evaluate on reviewed real data. Existing U-Net/StarDist results establish synthetic engineering behavior only; Cellpose-SAM has been inspected by eye, not scored. |
| Node optimization | Implemented for 39 computational nodes, with 12 explicit exemptions (`optimization/registry.py`), fixed objectives, project sequencing, proposal review, history, and rollback. | Treat every run as in-sample adaptation; independent performance still needs a frozen evaluation protocol. |
| Species libraries | Implemented with immutable versioning, content pins, source exclusion, import/export, validation metadata, and product specific coverage. | Build and validate reviewed libraries from representative sources. Coverage is not accuracy. |
| Results and export | Implemented with explicit availability, method selection, per-instance review, immutable result revisions, CSV/JSON/annotated-image export, and batch parity. | Physical units and unvalidated traits remain withheld until their validation gates pass. |
| Recovery and provenance | Implemented with source/checkpoint hashes, coordinate-bound reference archives (v7), project documents (v8), pipeline files (v2), analysis settings (v26) and immutable portable snapshots; each reader refuses earlier versions. | Unknown legacy coordinate frames require explicit alignment review. |

## Annotation behaviour

- **Materials** and **Seeds** on the toolbar open the reference annotation panel,
  a movable overlay that floats over the workspace and is as tall as its controls.
  Its title bar names the mode, drags the panel, and holds **Undo**, a roll-up
  control and a close button that turns the mode off. Annotations are shown and edited only on the corrected,
  deskewed image; other views hide them because their coordinates do not align.
- Seed selection offers **Next empty**, **Next unannotated** and the **Select** tool,
  which makes a clicked painted seed current and returns to the previous tool;
  **Show** selects all seeds or the current one.
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

`python -m tests --full` runs 317 functional units in three isolated processes; a
case-sensitive filesystem check is expected to skip on Windows. See
[TESTING.md](TESTING.md).

## Work needed for laboratory readiness

1. Agree on intended use, error tolerances, rejection policy and acceptable
   correction time before opening a final test set.
2. Collect and adjudicate representative original images and complete instance
   masks with biological and acquisition group identities.
3. Record traceable ruler/card/camera geometry and independent specimen and colour
   measurements across the image plane.
4. Define coat, condition, damage, wrinkling, visibility and exclusion labels with
   positive, negative and ambiguous examples; measure reviewer agreement.
5. Train or adapt only on development data, freeze the complete pipeline settings and run the
   independent protocol in [SCIENTIFIC_VALIDATION_PROTOCOL.md](SCIENTIFIC_VALIDATION_PROTOCOL.md).
6. Run representative operator, accessibility, resource, interruption and recovery
   studies on the target laboratory hardware.
