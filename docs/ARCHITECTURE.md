# Architecture and analysis contracts

How the code is organised and which module owns which guarantee. Binding rules for
changes are in [AGENTS.md](../AGENTS.md); the user-facing behaviour is in the
[operator guide](OPERATOR_GUIDE.md).

## Packages

| Package | Responsibility |
|---|---|
| `seed_vision.py`, `seedvision/bootstrap.py`, `application.py`, `diagnostics.py` | Single launcher, optional `.venv` bootstrap, runtime diagnostics, CLI training/evaluation entry points. |
| `pipeline/` | Toolkit-independent graph model (`model.py`: `PipelineGraph`, `PipelineNode`, `ParameterSpec`), graph-to-settings contract (`settings.py`), shared project inputs (`project_context.py`), editable factory workflows (`workflows.py`), added catalogue nodes, per-node backend execution, and `node_aliases.py`, which maps engine dirty-node names to graph nodes. |
| `segmentation/` | `baseline.py` production orchestration with node-local caching; procedural instance separation and review proposals. |
| `cuda/`, `calibration/`, `measurement/`, `visualization/` | Device-resident evidence and PlantCV-compatible CUDA operations, ruler/card/vessel detection, reviewed geometry, display layers. |
| `annotation/` | Annotation tools, eligibility rules and exact coordinate transport. |
| `benchmarking/` | PlantCV/SMPTS recipe specifications and catalogue, isolated reference runner, checkpoints, conformance, grouped study manifests and comparisons, native recipe execution. |
| `learning/`, `classification/`, `model_io/` | U-Net/StarDist training and evaluation, the optional pretrained Cellpose-SAM branch, trait classification, standard-library model formats and PlantCV model conversion. |
| `persistence/` | Projects, snapshots, reference archives, sidecars, settings profiles, run and QC history, image decoding (`image_decoding.read_image`, safe for non-ASCII paths). |
| `export/` | Authoritative result reports, CSV/JSON/image and crop export, native whole-run archives, the saved-project batch runner. |
| `reference_library/`, `optimization/` | Immutable species libraries; reference-driven node optimisation. |
| `ui/` | PySide6 application (see [Workflow and persistence](#workflow-and-persistence)). |

## Pipeline graph and caching

The typed graph owns every calculation parameter. `ParameterSpec` validates values
(finite numbers, bounds, choices) and `PipelineGraph.set_parameters` invalidates
the edited node and its dependents only, preserving bypass state. The calculation
engine in `segmentation/baseline.py` caches results per node and recomputes a node
only when its own inputs or parameters change; display-only state (overlay,
opacity, card positions, layout files) never invalidates analytical caches.
Engine stages whose names differ from graph node IDs are mapped in
`pipeline/node_aliases.py`, and `tests/test_node_vocabulary.py` keeps the two
vocabularies aligned.

Every exposed control must change a calculation. Each computational node declares
an optimisation capability or an explicit exemption, and
[NODE_CATALOGUE.md](NODE_CATALOGUE.md) is generated from the live registry.

Factory PlantCV and SMPTS pipelines are `RecipeSpec` descriptions
(`benchmarking/recipes.py`, `recipe_catalogue.py`) projected into the same
`PipelineGraph` and executed by `pipeline/backend_execution.py` (native CUDA or the
isolated PlantCV runner, per node). They share the project-context nodes and the
node-local cache.

## Data and computation ownership

- `segmentation/baseline.py`: source decoding/content identity, calibrated coordinates,
  node-local recomputation and production orchestration. Source changes invalidate
  image results; ordinary parameter edits invalidate only their dependents.
- `cuda/`, `calibration/`, `measurement/`: device-resident evidence and geometry.
  CPU transfer is limited to display, compact metadata and documented bounded topology.
- `learning/features.py`: shared live/export feature channels. Versioned feature specs
  identify raw foreground evidence; unsupported ambiguous recipes are rejected.
- `annotation/eligibility.py` and `annotation/coordinates.py`: consumer-specific shape
  eligibility and exact recorded-frame transport. Drawing progress is not scientific
  approval; complete contours clipped during transport lose shape eligibility.
- `reference_library/`: immutable library artifacts, source/group-balanced products
  and coverage audits. Coverage tiers never prove performance.
- `optimization/`: shared capability registry, fixed reference objectives, dependency
  order search and production evaluator. Optimization proposals have provenance and
  explicit review/application; they are in-sample adaptation.

## Workflow and persistence

`ui/main_window.py` remains the native coordinator. Bounded responsibilities are
extracted into `ui/results.py` (results, run history, snapshots, toolbar),
`pipeline_canvas.py`, `pipeline_inspector.py`, `pipeline_workflow.py` (factory
pipeline loading and node actions), `pipeline_documents.py` (Lock and pipeline/layout
files), `help_menu.py`, `reference_frames.py`, `work_control.py`,
`persistence_jobs.py` and `optimization.py`. Workers compute/save while Qt remains
responsive. Shutdown polls worker completion; cancellation occurs at safe boundaries
and between learned-model tiles.

`persistence/project_analysis.py` stores fingerprinted project masters that reference
per-image annotation sidecars. A saved project's sidecars live in its own
`name.seedfiddle-annotations/` folder (`persistence/project_annotations.py`): edits go
to its `unsaved/` subfolder and Save copies them in, so projects never share
annotations. The loose workspace and untitled projects write to the shared
`projects/` folders instead. `persistence/annotation_transfer.py` moves chosen
annotations between projects as a zip bundle matched by image SHA-256.
`persistence/snapshot.py` separately packages verified
content copies and restores a relocated working master. Supplementary audit evidence
is retained with its original identity; restoration does not silently reassign
review decisions or optimization targets to changed source/result identities.

Project documents (version 7, reading 1–6) retain the loaded pipeline workflow,
added node instances, immutable reference-graph metadata, editable native working
recipes and bounded model-asset descriptors. Analysis settings are version 25.
Historical checkpoint digests do not imply restored or verified arrays.

`export/results.py` defines authoritative method selection, unavailable-vs-empty
state, pixel-space object rows, exact result revisions and JSON/CSV/image export.
Physical units and unvalidated traits abstain. `export/project_runner.py` loads the
same saved production recipe and source-bound references as the desktop. The legacy
`BaselineAnalysis.count` property remains a proposal-count compatibility API and
must not be used for new result reporting.

`export/native_results.py` writes a separate v4 native whole-run projection and reads v1/v2/v3. Tagged
ordered metadata preserves tuple/list/mapping/record distinctions and explicit
nonfinite observations; named output bindings reference bounded checkpoint groups.
Only an explicit save transfers dense CUDA products to CPU. All descriptors, total
bytes, group inventories and report size are checked first. The loader checks
semantic relationships and aggregate inventory before restoring any arrays, then
verifies the existing checkpoint hashes and raw model-byte identity. It returns
data without executing models or instantiating recorded Python classes. Partial
nodes retain only actual products; output persistence does not establish conformance.

`export/project_native_graphs.py` executes exact saved native attachments and working recipes
through the same path as the desktop. It atomically publishes a companion ledger
and output archives under the baseline report. Original/reference graphs receive
explicit unavailable rows. Schema2 ledger rows identify historical versus working recipes;
working recipes contain no invented run history. Each result releases ownership after export, so
normal node-cache eviction can free its tensors; baseline tensors release before
native execution and the image cache clears before the next source. Separate
`persistence/reference_models.py` descriptors bind each exact historical/working model
selection to verified workspace-relative copies. Transport paths are excluded
from analytical identity and every run rereads selected bytes before cache reuse.
Snapshots require complete model closure; model inventory and copy reads are
capped by descriptor length even after preflight. This does not add per-stage
replay certificates or scientific eligibility.

## Annotation and display coordinates

Reference and seed annotations are stored against the corrected, deskewed image
frame. The application suppresses annotation rasters, centres, brush outlines,
assisted previews, and hilum editing whenever the source image, gamut slice,
prototype collage, or another non-deskewed view is displayed. This avoids a
visually plausible but geometrically incorrect overlay.

The main window owns the reference annotation panel, which floats over the
central workspace while an annotation mode is active; the image view positions it,
clamps it to the workspace and moves it by its title bar, and the window saves
its position. The panel takes no splitter space, so it never narrows the viewer.
Otherwise the image view owns only scene content and zoom/pan behaviour. Toolbar zoom preserves the viewport
centre; wheel zoom preserves the scene point beneath the pointer. Hilum placement
is a one-shot interaction tied to the current seed ID and derives its saved
direction from the painted seed centroid.

## Learning and independent evaluation

`learning/data.py` publishes immutable component revisions before atomically changing
a manifest. Audits detect changed components, duplicate source/feature content,
group leakage and conflicting species labels. Training caches are byte bounded;
evaluation predictions use a budgeted temporary spool with deterministic cleanup.
Checkpoint loading is restricted, and cached learned models use content hashes.

`learning/evaluation_protocol.py` records immutable component/source membership and
checks checkpoint/decoder selection overlap. Development, frozen and reference-
assisted protocols have distinct target-access provenance. Recorded provenance is
necessary but cannot prove study conduct; reviewed labels do not automatically
certify scientific validity. `learning/metrics.py` uses global object assignment,
explicit empty/unavailable policies and micro/image/group reporting. Empirical
calibration and biological validation follow the separate laboratory protocol.

## PlantCV–SMPTS comparison infrastructure

- `benchmarking/reference_runner.py` runs selected original PlantCV 4.11.3
  operations in the pinned, isolated environment under `artifacts/`, validating
  source and dependency identity, inputs, outputs, timeouts and cancellation.
  Missing environments report *unavailable*; there is no silent native fallback.
- `benchmarking/checkpoints.py` publishes immutable, source-bound stage checkpoints;
  `conformance.py` compares arrays, masks and partitions between checkpoints with
  exact integer errors and declared float tolerances. These reports measure
  computational agreement only.
- `benchmarking/manifest.py` and the grouped-comparison modules describe locked
  study splits, check leakage across source/group/access chains and compute paired
  grouped comparisons.
- `export/native_results.py` writes native whole-run archives (v4; reads v1–v3)
  with tagged metadata and checkpoint groups, checking every size limit before any
  CUDA-to-CPU transfer. `export/project_native_graphs.py` runs saved pipelines in
  batch through the desktop path and records unavailable reference executors
  explicitly.

The [implementation plan](PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md) is the design
specification; the [feature register](plantcv_smpts_features.json) maps every
operation to code, tests and acceptance status.

## Extending the application

Keep controls tied to actual calculations. Add the node's optimisation capability
or an explicit non-optimisable reason, then regenerate the catalogue. Preserve
native Qt, one launcher and device-resident reusable rasters. Do not edit reviewed
calculation sources listed in the provenance records without a reviewed provenance
update (see [Testing](TESTING.md#provenance-checks)). Run focused regressions and
the full suite. Use reviewed real data for scientific claims; fixture and synthetic
tests establish engineering properties, not laboratory accuracy.
