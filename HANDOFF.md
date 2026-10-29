# SeedFiddle hand-off

Orientation for the next developer or agent. Current as of 23 October 2026
(`main`, after the review-fix, GUI-simplification and documentation-cleanup
commits). Read [AGENTS.md](AGENTS.md) for the binding rules; this file explains
the context behind them.

## What the project is

A native PySide6 desktop application (`seed_vision.py` is the only launch point)
for counting, measuring and classifying soybean and lupin seeds in calibrated
photographs. Analyses are typed node graphs with node-local caching on a CUDA
device. Besides the Default seed analysis there are 18 factory pipelines that
reproduce PlantCV and SMPTS seed workflows natively (the "peer-functionality"
work, families F01–F22). See [README.md](README.md) for the user view and
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
| `seedvision/pipeline/` | `PipelineGraph`/`PipelineNode`/`ParameterSpec` model, factory workflows, project context, `node_aliases.py` (engine dirty-node vocabulary). |
| `seedvision/segmentation/` | `baseline.py` production orchestration and procedural instance separation. |
| `seedvision/cuda/`, `calibration/`, `measurement/` | Device-resident evidence, calibration and geometry. |
| `seedvision/benchmarking/` | PlantCV/SMPTS recipes, isolated reference runner, checkpoints, conformance and grouped comparisons. |
| `seedvision/learning/` | U-Net/StarDist training, datasets, evaluation protocols; the optional Cellpose-SAM node (`cellpose_sam.py`). |
| `seedvision/persistence/`, `export/` | Projects, snapshots, sidecars, image decoding (`image_decoding.read_image`), result reports, batch runner. |
| `seedvision/reference_library/`, `optimization/`, `annotation/` | Species libraries, reference-driven node optimisation, annotation tools. |
| `tests/` | Parallel unittest runner (`python -m tests`) and ~250 test modules. |
| `config/` | Pinned runtime dependencies, PlantCV reference sources, fixture manifest, trait vocabulary. |
| `scripts/` | Batch runner, node-catalogue generator, `prune_artifacts.py`. |
| `docs/` | Guides, contracts and records; start at [docs/README.md](docs/README.md). |

## Run and test

```powershell
.\.venv\Scripts\python.exe .\seed_vision.py --no-bootstrap
.\.venv\Scripts\python.exe -m tests                  # core subset
.\.venv\Scripts\python.exe -m tests tests.test_gui_layout   # focused module
.\.venv\Scripts\python.exe -m tests --full           # 649 units, ~14 min, 3 workers
```

The last full run (23 October 2026) passed all 649 units with the 4 expected
skips. Wall time over 15 minutes is reported as an advisory warning only; a run
fails on test outcomes or an unaccounted unit, and a worker that writes no
journal entry for `--stall-seconds` (default 900) is stopped. Details are in
[docs/TESTING.md](docs/TESTING.md).

### `artifacts/` holds irreplaceable test inputs

`artifacts/` is git-ignored but **not** disposable. The full suite needs these
entries, which exist nowhere else unless you keep a fixture bundle
(`--fixtures-from`):

- the frozen acceptance baselines `recipe-acceptance-baseline-v1/v2.json`,
  `public-seed-acceptance-baseline-v1.json`, `reader-acceptance-baseline-v1.json`;
- the provenance records (`*-code-provenance-*.json`, `*-provenance-transition-*.json`,
  `pipeline-workspace-*.json`) listed in `config/test_fixtures.json`;
- `plantcv-reference-env/`, `reference-sources/`, `reference-wheels/` (pinned
  PlantCV 4.11.3 environment and sources);
- `learned/`, `method-qc/`, `branding/` and the two `test-suite-cleanup-full-v2*.json`
  files referenced by tests.

Everything else there is generated. `scripts/prune_artifacts.py --delete` clears
old `test-runs/` and `test-temp/` directories and never touches the list above.

## Gotchas

- **Frozen provenance.** `tests/acceptance_provenance.py` hashes the working-tree
  bytes of reviewed calculation sources (for example
  `seedvision/benchmarking/reference_runner.py`). Do not edit those files without
  a reviewed provenance update; a cosmetic change fails the suite.
- **Line endings.** Provenance digests are of the raw working-tree bytes, and the
  reviewed sources are a mix of LF and CRLF files. With `core.autocrlf=true`, a
  `git checkout`/`git restore` of an LF file rewrites it as CRLF and fails the
  acceptance tests without any content change. Restore such a file by
  converting it back to LF, not with git.
- **Non-ASCII paths.** `cv2.imread` fails on them on Windows; always use
  `seedvision.persistence.image_decoding.read_image`.
- **Offscreen Qt screenshots** render boxes unless `QT_QPA_FONTDIR=C:/Windows/Fonts`.
  `QAction` lives in `PySide6.QtGui`, not `QtWidgets`.
- **UI text limits.** `tests/test_ui_text_limits.py` caps command labels at 45
  characters and tooltips at 450.
- **Unreachable dialogs.** `NativeRecipeDialog` and its Naive Bayes, K-means and
  correction subclasses, and `PublicReferenceCaptureDialog`, are no longer opened
  from the application since the Reference window was removed. Only tests use them.
  Either reconnect them or retire them.

## Open work

- **Seed detection quality.** Procedural separation undercounts touching seeds
  (about 5× on the benchmark image). Against Cellpose-SAM masks as a stand-in
  reference, its markers mostly land on seeds; the loss is in its instance filters
  (area and concavity limits scaled by the seed-diameter estimate, which runs about
  1.4× the seeds' equivalent diameter on packed images), so a different centre
  source does not help. The fitted seed ovals of Seed-boundary confirmation find
  roughly half the seeds on packed images and most on sparse ones. Cellpose-SAM
  looked right by eye and is now an optional, disabled-by-default node (see the
  operator guide's Cellpose-SAM section). Reviewed reference masks are the blocker
  for any quantitative claim. See
  [docs/SEED_DETECTION_EVALUATION.md](docs/SEED_DETECTION_EVALUATION.md).
- **Laboratory gates L01–L04** (reviewed corpus, physical/colour measurement,
  trait definitions, operator study) are external and still open; see
  [PLAN.md](PLAN.md) and
  [docs/SCIENTIFIC_VALIDATION_PROTOCOL.md](docs/SCIENTIFIC_VALIDATION_PROTOCOL.md).
- **Original SMPTS source** is unavailable, so SMPTS stages are a paper
  reconstruction and F09.2 is unresolved.
- **GUI follow-ups** not done in the October pass: an icon strip for annotation
  tools (no icon assets exist).

## History

The chronological implementation log that used to fill this file, and the copies
of it pasted into `PLAN.md`, `docs/CURRENT_STATUS.md`, the operator guide and
other documents, were removed in the October 2026 documentation cleanup. They
remain in git history (for example `git show 2e5b398:HANDOFF.md`). Dated audits
and reviews are kept under [docs/archive/](docs/archive/).
