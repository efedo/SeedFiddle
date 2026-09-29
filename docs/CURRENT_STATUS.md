# Current project status

**Status date:** 29 September 2026
**Documented revision:** `f244df8` on `codex/learned-instance-segmentation`
**Latest complete local verification:** 670 tests run; 668 passed, 2 skipped

Seed Fiddle is an engineering complete local desktop workflow for image loading,
calibration diagnostics, material and seed annotation, procedural and optional
learned instance proposals, reference driven parameter adaptation, result review,
export, and portable project snapshots. It remains a research and review tool.
Counting accuracy, physical measurement accuracy, colour accuracy, biological
trait validity, and operator usability have not been established on an independent
representative laboratory corpus.

## Capability status

| Area | Current state | Remaining evidence or work |
|---|---|---|
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
| Recovery and provenance | Implemented with source/checkpoint hashes, coordinate bound reference schema 6, project schema 2, analysis settings schema 20, and immutable portable snapshots. | Unknown legacy coordinate frames still require explicit alignment review. |

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

It ran 670 tests in 383.737 seconds on 29 September 2026: 668 passed and 2
skipped. The skips cover an unavailable
complete historical pilot batch and a POSIX case-sensitive filesystem check on
Windows. The node catalogue drift check and `git diff --check` also passed at the
documented revision. Software and synthetic tests demonstrate engineering
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
