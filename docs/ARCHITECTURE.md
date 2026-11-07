# Architecture and analysis contracts

How the code is organised and which module owns which guarantee. Binding rules for
changes are in [AGENTS.md](../AGENTS.md); the user-facing behaviour is in the
[operator guide](OPERATOR_GUIDE.md).

## Packages

| Package | Responsibility |
|---|---|
| `seed_vision.py`, `seedvision/bootstrap.py`, `application.py`, `diagnostics.py` | Single launcher, optional `.venv` bootstrap, runtime diagnostics, CLI training/evaluation entry points. |
| `pipeline/` | Toolkit-independent graph model (`model.py`: `PipelineGraph`, `PipelineNode`, `ParameterSpec`), graph-to-settings contract (`settings.py`), shared project inputs (`project_context.py`), the two default pipelines (`default_pipelines.py`), added node instances (`catalogue_nodes.py`), and `node_aliases.py`, which maps engine dirty-node names to graph nodes. |
| `segmentation/` | `baseline.py` production orchestration with node-local caching; procedural instance separation and review proposals. |
| `cuda/`, `calibration/`, `measurement/`, `visualization/` | Device-resident evidence and CUDA operations, chart colour correction (`cuda/colour_correction.py`), ruler/card/vessel detection, reviewed geometry, display layers. |
| `annotation/` | Annotation tools, eligibility rules and exact coordinate transport. |
| `benchmarking/` | Locked study manifests, grouped method comparisons and the benchmark command line. `reference_runner.py`, the former isolated PlantCV runner, is kept but has no caller. |
| `learning/`, `classification/` | U-Net/StarDist training and evaluation, the optional pretrained Cellpose-SAM branch, trait classification. |
| `persistence/` | Projects, pipeline files, snapshots, reference archives, sidecars, settings profiles, run history, image decoding (`image_decoding.read_image`, safe for non-ASCII paths). |
| `export/` | Authoritative result reports, CSV/JSON/image and crop export, the saved-project batch runner and its added-node reports. |
| `reference_library/`, `optimization/` | Immutable species libraries; reference-driven node optimisation. |
| `ui/` | PySide6 application (see [Workflow and persistence](#workflow-and-persistence)). |

## Pipeline graph and caching

The typed graph owns every calculation parameter. `ParameterSpec` validates values
(finite numbers, bounds, choices) and `PipelineGraph.set_parameters` invalidates
the edited node and its dependents only, preserving bypass state. The calculation
engine in `segmentation/baseline.py` caches results per node and recomputes a node
only when its own inputs or parameters change; display-only state (overlay,
opacity, card positions) never invalidates analytical caches.
Engine stages whose names differ from graph node IDs are mapped in
`pipeline/node_aliases.py`, and `tests/test_node_vocabulary.py` keeps the two
vocabularies aligned.

Every exposed control must change a calculation. Each computational node declares
an optimisation capability or an explicit exemption, and
[NODE_CATALOGUE.md](NODE_CATALOGUE.md) is generated from the live registry.

The two default pipelines share one graph and differ only in which nodes are
active (`pipeline/default_pipelines.py`). An added node instance
(`pipeline/catalogue_nodes.py`) is an independent copy of one default node with its
own parameters and cache; it runs only the nodes it needs through the same engine.

## Data and computation ownership

- `segmentation/baseline.py`: source decoding/content identity, calibrated coordinates,
  node-local recomputation and production orchestration. Source changes invalidate
  image results; ordinary parameter edits invalidate only their dependents. A run to
  one node passes the other enabled nodes as `retained_nodes`: their cached results
  stand, and the cache's `stale_nodes` records those that a change has outdated, for
  the next run that includes them. Cancelling a run keeps its finished results and
  restores its dirty nodes; a failure discards the image's cache.
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
extracted into `ui/results.py` (results, snapshots, toolbar),
`pipeline_canvas.py`, `pipeline_inspector.py`, `pipeline_workflow.py` (the Pipeline
list and added-node output preview), `pipeline_documents.py` (the default-pipeline lock,
named copies and pipeline files), `image_list.py` (Images list rows and their
thumbnails), `overlay_header.py` (the overlay selector and opacity row, docked in the
header of whichever image display is showing), `help_menu.py`, `reference_frames.py`, `work_control.py`,
`persistence_jobs.py` and `optimization.py`. Workers compute/save while Qt remains
responsive. Shutdown polls worker completion; cancellation occurs at safe boundaries
and between learned-model tiles. Thumbnails decode only for painted rows, on their own
low-priority thread so that they never count as analysis activity, and are held only
in a bounded in-memory cache.

`persistence/project_analysis.py` stores fingerprinted project masters that reference
per-image annotation sidecars. A saved project's sidecars live in its own
`name.seedfiddle-annotations/` folder (`persistence/project_annotations.py`): edits go
to its `unsaved/` subfolder and Save copies them in, so projects never share
annotations. An untitled project writes to the shared
`projects/` folders instead. `persistence/annotation_transfer.py` moves chosen
annotations between projects as a zip bundle matched by image SHA-256.
`persistence/snapshot.py` separately packages verified
content copies and restores a relocated working master. Supplementary audit evidence
is retained with its original identity; restoration does not silently reassign
review decisions or optimization targets to changed source/result identities.

Project documents (version 8) retain the analysis settings, added node instances
and the project's named pipeline copies. Pipeline files are version 2, snapshot
indexes version 2, analysis settings version 26, reference-region archives version 7,
seed-centre files version 2 and learning feature specifications version 2. Each
reader accepts only its current version and refuses earlier files rather than
migrating them.

`export/results.py` defines authoritative method selection, unavailable-vs-empty
state, pixel-space object rows, exact result revisions and JSON/CSV/image export.
Physical units and unvalidated traits abstain. `export/project_runner.py` loads the
same saved analysis settings and source-bound references as the desktop, and
`export/node_instances.py` writes each added node instance's outputs beside the
baseline report. The legacy
`BaselineAnalysis.count` property remains a proposal-count compatibility API and
must not be used for new result reporting.

## Annotation and display coordinates

Reference and seed annotations are stored against the corrected, deskewed image
frame. The application suppresses annotation rasters, centres, brush outlines,
assisted previews, and hilum editing whenever the source image, gamut slice,
prototype collage, or another non-deskewed view is displayed. This avoids a
visually plausible but geometrically incorrect overlay.

The main window owns the reference annotation panel, which floats over the
central workspace while an annotation mode is active; the image view positions it,
clamps it to the workspace and moves it by its title bar, and the window saves
its position. The panel (`ui/annotation_panel.py`) bounds its own height by its
contents and asks to be placed again when they change. The panel takes no
splitter space, so it never narrows the viewer.
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

## Benchmark studies

`benchmarking/manifest.py` and `grouped_comparison.py` describe locked study
splits, check leakage across source/group/access chains and compute paired grouped
comparisons; `benchmark_cli.py` runs a prespecified comparison from the launcher.
These reports measure agreement between methods on declared observations, not
scientific validity.

## Extending the application

Keep controls tied to actual calculations. Add the node's optimisation capability
or an explicit non-optimisable reason, then regenerate the catalogue. Preserve
native Qt, one launcher and device-resident reusable rasters. Run focused
regressions and the full suite. Use reviewed real data for scientific claims; fixture and synthetic
tests establish engineering properties, not laboratory accuracy.
