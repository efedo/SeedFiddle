# Current project status

**Status date:** 3 October 2026
**Engineering baseline:** `094aa7d`; prior implementation review `f244df8`
**Latest complete local verification:** 738 tests run; 736 passed, 2 skipped,
before the current review-repair batch. In-progress chunks require a fresh final
suite; focused checkpoint and recipe suites now pass 24 and 12 tests respectively.

Seed Fiddle has an implemented baseline local desktop workflow for image loading,
calibration diagnostics, material and seed annotation, procedural and optional
learned instance proposals, reference driven parameter adaptation, result review,
export, and portable project snapshots. It remains a research and review tool.
Counting accuracy, physical measurement accuracy, colour accuracy, biological
trait validity, and operator usability have not been established on an independent
representative laboratory corpus.

## Accepted extension: peer functionality

The project now targets selected PlantCV–SMPTS seed-analysis capabilities under
[the implementation and benchmark plan](PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md).
The [comparison report](PLANTCV_SMPTS_COMPARISON.md) explains the scope and evidence.
WP0/WP1 are in progress. The [operation-level register](plantcv_smpts_features.json)
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
catalogue tests pass. Desktop views and checkpoint inspection remain pending. The
existing canvas/inspector now have a reviewed read-only mode for reference facts;
five focused Qt tests verify blocked analytical/topology actions, native-ID fact
isolation and display-only operations preserving actual cache objects. Native
editing remains the default. This mode is a prerequisite for the actual F22 views.
Project schema 3 now retains strict immutable recipe/run/layout attachments through
desktop open/save and snapshot round-trips, while safely reading versions 1/2.
Forty-seven persistence/UI tests pass with one existing filesystem skip. Historical
checkpoint hashes are retained, but their arrays remain explicitly unavailable;
artifact binding, restoration and inspection are separate work.
The reviewed conformance API now compares primitive arrays, masks and instance
partitions from exact-identity checkpoints. It retains stage/pipeline upstream
provenance, checks raster semantics, uses exact integer errors and reference-relative
float tolerances, and records bounded common-evaluator matched-ID audits. Thirteen
focused tests pass. These reports measure computational agreement; this API always
records independent scientific accuracy as not evaluated. Actual native-reference
conformance runs and recipe orchestration remain separate work. The reviewed
isolated PlantCV runner executes twelve pinned original operations: binary/mean/Otsu
thresholds, median filtering, erosion/dilation/fill, watershed, size/colour analysis
and manual/automatic colour correction. Twenty focused tests pass with the exact
reference environment. It validates source/dependency identity, inputs/outputs,
RGB boundary semantics, timeout/cancellation and cleanup. Missing ignored source
or environment artifacts report unavailable. Original SMPTS remains unavailable;
desktop/batch recipe orchestration and archived comparison runs are still pending.
WP2–WP6 remain pending. Faithful display of pinned reference implementation graphs in the
native node editor is also an explicit pending objective (F22), with topology,
parameter, provenance and Qt visual acceptance. Existing native methods remain the
engineering baseline. Functional coverage,
numerical conformance, scientific performance and operational cost require separate
proof. Laboratory data collection continues alongside WP0/WP1.

## Capability status

| Area | Current state | Remaining evidence or work |
|---|---|---|
| PlantCV–SMPTS peer functionality | WP0/WP1 in progress; operation-level register, PlantCV source/environment pins and native engineering baseline captured. | Freeze actual reference recipes and outputs, finish reference/replay infrastructure and WP2–WP6 gates; no parity claim yet. |
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
| Recovery and provenance | Implemented with source/checkpoint hashes, coordinate bound reference schema 6, project schema 3 (v1/v2 migration), analysis settings schema 20, and immutable portable snapshots. Reference graph metadata persists; checkpoint arrays are explicitly unavailable. | Unknown legacy coordinate frames require explicit alignment review; reference checkpoint artifact binding/restore remains pending. |

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

The latest frozen infrastructure run on 3 October 2026 ran 738 tests in 343.357
seconds: 736 passed and two existing skips. Log:
`artifacts/peer-infrastructure-final-full-suite.log`. Later review repairs require
a fresh complete run. This includes proposed runner/conformance modules in the
working tree; a passing suite does not replace independent review or close their
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
