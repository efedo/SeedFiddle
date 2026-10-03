# Seed Fiddle

Seed Fiddle is a local PySide6 application for reference-assisted seed-image
analysis, annotation, instance review and reproducible result export. It uses
PyTorch CUDA for full-resolution image evidence and retains node-local reusable
intermediates on the device. There is no application server or bundled executable.

**Project status (3 October 2026):** the existing engineering workflow is
implemented through reviewed result export and portable snapshots. The latest
full local suite ran 670 tests: 668 passed and 2 were skipped. See the
[current project status](docs/CURRENT_STATUS.md) for the capability matrix,
annotation behavior, schema versions, and remaining validation work.

**New project direction:** achieve peer functionality for selected PlantCV and
SMPTS seed-analysis capabilities within SeedFiddle. The
[revised comparison](docs/PLANTCV_SMPTS_COMPARISON.md) and
[implementation/benchmark plan](docs/PLANTCV_SMPTS_IMPLEMENTATION_PLAN.md) define
the accepted extension. Capability auditing and comparison infrastructure come
first; peer algorithms, adapters and parity claims are not yet delivered.

**Current limits:** outputs are proposals for review. Real-image counting accuracy,
physical measurement accuracy and biological trait validity remain unestablished.
Physical result units and unvalidated per-seed traits are withheld. Neutral colour
balance is not validated colorimetry. Read the
[scientific validation protocol](docs/SCIENTIFIC_VALIDATION_PROTOCOL.md) before
planning quantitative laboratory use.

Most project code was authored by AI and has not had comprehensive human review.
Independently review and validate it before relying on results.

## Start

Python 3.12 x64 is the baseline runtime. With compatible dependencies available:

```powershell
python .\seed_vision.py
python .\seed_vision.py --diagnostics
```

The single launcher can create a project-local virtual environment or install
missing packages after confirmation. `--no-bootstrap` suppresses bootstrap;
`--offline` requires compatible wheels in `wheels/`. Runtime requirements are in
[requirements-runtime.txt](requirements-runtime.txt). Full-raster analysis is
GPU-first; the UI reports the active device and fallback.

## Work through an image

1. Open images or a saved project. Verify acquisition, species and biological
   context; species starts as Unknown/unassigned.
2. Supply appropriate Foreground/Background/Other references. Missing required
   evidence means unavailable, not a zero seed count.
3. Run the pipeline. Inspect intermediate overlays or use the optional expert
   graph. Reference-driven node and project optimization propose settings for
   review; fitting to the same references is adaptation, not validation.
4. Use **Review / export results** (`Ctrl+R`) to inspect seed IDs and mark accepted,
   excluded or unreviewed instances. Filter decisions and compare completed methods.
5. Export JSON, CSV and an annotated image, or save the displayed viewport with its
   pixel-scale key. Save a project for continued editing; create an immutable
   portable snapshot to freeze verified analytical dependencies and audit evidence.

Stop uses `Ctrl+.`. Annotation Undo/Redo use `Ctrl+Z` / `Ctrl+Shift+Z`; node search
uses `Ctrl+K`. See the guide for drawing and review shortcuts.

## Current documentation

- [Current project status](docs/CURRENT_STATUS.md): implemented capabilities,
  latest verification, current annotation behavior, and remaining work.
- [Documentation index](docs/README.md): current guides, implementation records,
  historical audits, and research ideas.
- [Implementation and validation roadmap](PLAN.md): completed baseline phases,
  the accepted peer-functionality extension and remaining laboratory milestones.
- [Operator guide](docs/OPERATOR_GUIDE.md): workflow, shortcuts, projects, snapshots,
  results, learning and migration.
- [Architecture and contracts](docs/ARCHITECTURE.md): ownership, caching and provenance.
- [Generated node/control catalogue](docs/NODE_CATALOGUE.md).
- [Node optimization](docs/NODE_OPTIMIZATION.md): objectives, eligibility and exemptions.
- [Critical-review remediation](docs/CRITICAL_REVIEW_REMEDIATION.md): all 30 findings,
  corrections, test evidence, limitations and remaining laboratory inputs.
- [Scientific validation protocol](docs/SCIENTIFIC_VALIDATION_PROTOCOL.md).

The [historical README](README_HISTORY.md), dated audits and experiment reports
preserve their original findings. They are not current run instructions or validated
performance claims. [HANDOFF.md](HANDOFF.md) contains agent constraints and dated
implementation history.

## Compatibility and batch work

Reference schema 6 binds rasters to calibration coordinates. Unknown legacy frames
require explicit alignment review; never migrate them by resizing or changing a
schema number. Raw-foreground feature recipe v2 replaces ambiguous v1 foreground
recipes, which require re-export/retraining. Checkpoint loading is restricted to
supported tensor/primitive payloads.

Batch execution consumes the saved desktop recipe and fingerprinted references:

```powershell
python scripts/analyze_pilot.py --project projects/analysis.seedfiddle-project.json --output-dir artifacts/project-results
```

Working masters retain shared mutable sidecars. Use an explicit snapshot when an
immutable revision is required. Source/checkpoint content changes invalidate cached
results and require recomputation or reference review.

## Development and checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/generate_node_catalogue.py --check
```

`images/` holds committed fixtures. Generated logs, screenshots and exports belong
under ignored `artifacts/`. Windows CPU/Qt CI and an optional configured CUDA runner
are defined in [.github/workflows/tests.yml](.github/workflows/tests.yml).

Desktop diagnostic logs rotate under `%LOCALAPPDATA%\Seed Fiddle\logs\` on Windows.
**Help → Runtime summary** shows the actual log and crash-log paths. Software tests
and synthetic examples do not substitute for independent laboratory validation.

## License

[MIT License](LICENSE).
