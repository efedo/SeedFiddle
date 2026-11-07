# SeedFiddle hand-off

Orientation for the next developer or agent. Current as of 6 November 2026
(`main`). Read [AGENTS.md](AGENTS.md) for the binding rules; this file explains
the context behind them.

## What the project is

A native PySide6 desktop application (`seed_vision.py` is the only launch point)
for counting, measuring and classifying soybean and lupin seeds in calibrated
photographs. Analyses are typed node graphs with node-local caching on a CUDA
device. The two default pipelines, Procedural seed analysis and Cellpose seed
analysis, share one graph and differ in which nodes are active
(`seedvision/pipeline/default_pipelines.py`). Any default node can also be added
as an independent instance with its own controls and cache
(`seedvision/pipeline/catalogue_nodes.py`). The earlier PlantCV and SMPTS
reproductions and their recipe engine were removed in November 2026. See
[README.md](README.md) for the user view and
[docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md) for what is and is not done.

## Rules that are easy to break

- One launcher, native Qt, no server, installer or bundled executable.
- Full-resolution rasters and reusable intermediates stay on the CUDA device;
  transfer to CPU only for Qt display or compact metadata.
- Editing one node must recompute only that node and its dependents
  (`segmentation/baseline.py`, `pipeline/model.py`). Display-only changes must
  not invalidate caches.
- Every exposed control must change a calculation. A new computational node
  needs an optimisation capability or an explicit exemption, and the node
  catalogue must be regenerated (`scripts/generate_node_catalogue.py`).
- `images/` holds committed fixtures. Generated output goes to ignored
  `artifacts/`.
- Run focused tests for each change and `python -m tests --full` before hand-off.

## Repository map

| Path | Contents |
|---|---|
| `seed_vision.py` | Launcher: optional `.venv` bootstrap, diagnostics, CLI training/evaluation entry points. |
| `seedvision/ui/` | Qt application. `main_window.py` is the coordinator; `results.py`, `pipeline_canvas.py`, `pipeline_inspector.py`, `pipeline_workflow.py`, `pipeline_documents.py`, `help_menu.py` and the dialogs own bounded pieces. |
| `seedvision/pipeline/` | `PipelineGraph`/`PipelineNode`/`ParameterSpec` model, the default pipelines, added node instances, project context, `node_aliases.py` (engine dirty-node vocabulary). |
| `seedvision/segmentation/` | `baseline.py` production orchestration and procedural instance separation. |
| `seedvision/cuda/`, `calibration/`, `measurement/` | Device-resident evidence, chart colour correction (`cuda/colour_correction.py`), calibration and geometry. |
| `seedvision/benchmarking/` | Study manifests, grouped method comparisons and the benchmark command line; the unused `reference_runner.py` (see Open work). |
| `seedvision/learning/` | U-Net/StarDist training, datasets, evaluation protocols; the optional Cellpose-SAM node (`cellpose_sam.py`). |
| `seedvision/persistence/`, `export/` | Projects, snapshots, sidecars, image decoding (`image_decoding.read_image`), result reports, batch runner. |
| `seedvision/reference_library/`, `optimization/`, `annotation/` | Species libraries, reference-driven node optimisation, annotation tools. |
| `tests/` | Parallel unittest runner (`python -m tests`) and 115 test modules. |
| `config/` | Pinned runtime dependencies and trait vocabulary; the PlantCV lock and `peer_*.json` records belong to the unused reference runner. |
| `scripts/` | Batch runner, node-catalogue generator, `prune_artifacts.py`. |
| `docs/` | Guides, contracts and records; start at [docs/README.md](docs/README.md). |

## Run and test

```powershell
.\.venv\Scripts\python.exe .\seed_vision.py --no-bootstrap
.\.venv\Scripts\python.exe -m tests                  # core subset
.\.venv\Scripts\python.exe -m tests tests.test_gui_layout   # focused module
.\.venv\Scripts\python.exe -m tests --full           # 317 units, 3 workers
```

Wall time over 15 minutes is reported as an advisory warning only; a run fails on
test outcomes or an unaccounted unit, and a worker that writes no journal entry for
`--stall-seconds` (default 900) is stopped. Run one full suite at a time per GPU.
Details are in [docs/TESTING.md](docs/TESTING.md).

`artifacts/` is git-ignored generated output; tests need nothing in it.
`scripts/prune_artifacts.py --delete` clears old `test-runs/` and `test-temp/`
directories.

## Gotchas

- **Closed test windows stay alive.** Their own signal connections keep them alive,
  so the test runner deletes the ones each test leaves behind, and a full-suite
  worker ends itself without interpreter shutdown; see
  [docs/TESTING.md](docs/TESTING.md).
- **Annotation storage is per project.** A saved project's manual centres and
  reference layers live in `name.seedfiddle-annotations/` beside its project file;
  edits go to its `unsaved/` subfolder until Save copies them in. The main window
  swaps `_reference_region_store` and `_manual_seed_centre_store` per project
  (`_install_annotation_stores`), so UI code must find an image's file through
  `_project_reference_sidecar_paths` / `_project_manual_centre_sidecar_paths`, not
  by constructing a store on the workspace root. Only untitled projects use the
  shared `projects/` folders.
- **Deferred callbacks.** Use `seedvision.ui.deferred.run_soon(owner, callback)`,
  not `QTimer.singleShot`; `tests/test_deferred_callbacks.py` enforces this under
  `seedvision/`. `singleShot(0, callback)` outlives its target, and an exception it
  raises during a static dialog such as `QMessageBox.warning()` can crash the
  process. The context overload `singleShot(0, owner, callback)` loses the exception
  silently instead.
- **Non-ASCII paths.** `cv2.imread` fails on them on Windows; always use
  `seedvision.persistence.image_decoding.read_image`.
- **Offscreen Qt screenshots** render boxes unless `QT_QPA_FONTDIR=C:/Windows/Fonts`.
  `QAction` lives in `PySide6.QtGui`, not `QtWidgets`.
- **UI text limits.** `tests/test_ui_text_limits.py` caps command labels at 45
  characters and tooltips at 450.

## Open work

- **Seed detection quality.** Procedural separation undercounts touching seeds
  (about 5× on the benchmark image). Against Cellpose-SAM masks as a stand-in
  reference, its markers mostly land on seeds; the loss is in its instance filters
  (area and concavity limits scaled by the seed-diameter estimate, which runs about
  1.4× the seeds' equivalent diameter on packed images), so a different centre
  source does not help. The fitted seed ovals of Seed-boundary confirmation find
  roughly half the seeds on packed images and most on sparse ones. Cellpose-SAM
  looked right by eye and is now an optional, disabled-by-default node whose
  package the launcher installs unless `--no-cellpose` is given (see the operator
  guide's Cellpose-SAM section). Reviewed reference masks are the blocker
  for any quantitative claim. See
  [docs/SEED_DETECTION_EVALUATION.md](docs/SEED_DETECTION_EVALUATION.md).
- **Laboratory gates L01–L04** (reviewed corpus, physical/colour measurement,
  trait definitions, operator study) are external and still open; see
  [PLAN.md](PLAN.md) and
  [docs/SCIENTIFIC_VALIDATION_PROTOCOL.md](docs/SCIENTIFIC_VALIDATION_PROTOCOL.md).
- **Unused reference runner.** `seedvision/benchmarking/reference_runner.py`, which
  ran original PlantCV 4.11.3 in an isolated Python 3.12 interpreter, has had no
  caller or test since PlantCV was removed. It is kept, with
  `config/plantcv-reference-requirements.txt`, the `config/peer_*.json` records,
  `scripts/repair_reference_lock.py` and that script's check in
  `tests/test_bootstrap.py`, until it is either retired or given a new use.
- **GUI follow-ups** not done in the October pass: an icon strip for annotation
  tools (no icon assets exist).

## History

The chronological implementation log that used to fill this file, and the copies
of it pasted into `PLAN.md`, `docs/CURRENT_STATUS.md`, the operator guide and
other documents, were removed in the October 2026 documentation cleanup. They
remain in git history (for example `git show 2e5b398:HANDOFF.md`). Dated audits
and reviews are kept under [docs/archive/](docs/archive/).
