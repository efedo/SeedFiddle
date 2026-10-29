# Seed Fiddle

Seed Fiddle is an on-premises desktop application for reviewing, counting,
measuring and broadly classifying soybean and lupin seeds in calibrated laboratory
photographs. It is a native PySide6 application launched from a single script,
with full-resolution analysis on a CUDA GPU through PyTorch.

Analyses are typed node graphs ("pipelines"). The Default seed analysis combines
calibration, material and edge evidence, procedural instance separation and
optional learned (U-Net/StarDist) and pretrained Cellpose-SAM branches. Eighteen further factory pipelines
reproduce selected PlantCV and SMPTS seed workflows natively, so methods can be
compared on the same image.

> **Status.** The software engineering is extensive and tested. The analytical
> methods have **not** been validated against independently reviewed laboratory
> data: physical units and trait predictions are withheld or labelled unvalidated,
> and procedural counting undercounts touching seeds. Most of the code was written
> by AI and has not had comprehensive human review. See
> [Current status](docs/CURRENT_STATUS.md) and the
> [seed-detection evaluation](docs/SEED_DETECTION_EVALUATION.md) before relying on
> any result.

## Install and launch

Requirements: Windows, 64-bit Python 3.12 (the baseline), a supported NVIDIA GPU,
a compatible NVIDIA driver, and CUDA-enabled PyTorch that successfully executes a
small CUDA calculation. Python 3.13–3.15 are permitted by the runtime gate when
compatible dependencies are already installed. The baseline installation pins
are not a promise that bootstrap can provision a new Python 3.15 environment. A
separate CUDA toolkit is not needed.

```powershell
git clone https://github.com/efedo/SeedFiddle.git
Set-Location .\SeedFiddle
py -3.12 .\seed_vision.py --environment venv --bootstrap auto
```

The first launch can create `.venv` and install the pinned packages from
[requirements-runtime.txt](requirements-runtime.txt) after asking. Later launches:

```powershell
.\.venv\Scripts\python.exe .\seed_vision.py --no-bootstrap
.\.venv\Scripts\python.exe .\seed_vision.py --diagnostics
```

`--diagnostics` reports PyTorch build compatibility separately from an actual GPU
calculation; it returns nonzero until both are ready. `--offline` installs only
from wheels you place in `wheels/` (binaries are not committed). A CPU-only or
ROCm PyTorch build is incompatible; a CUDA build with an unavailable GPU or
driver is a hardware/runtime problem and is not repeatedly reinstalled.

### NVIDIA CUDA and Python 3.15

Use the same interpreter for pip, diagnostics, analysis, and launch. The preferred
CUDA 12.6 pin is a baseline option for supported NVIDIA GPUs, but CUDA 12.6 does
not cover Blackwell GPUs. On Windows with Python 3.15, the CUDA 13.2 wheel is an
option for Blackwell hardware and requires a driver compatible with CUDA 13.2.
CUDA 12.6 can be used where supported by the GPU and installed driver; choose the
wheel family for the hardware/driver rather than downgrading an already-working
newer CUDA build.

Install one of these into the selected interpreter:

```powershell
python -m pip install --upgrade "torch==2.15.0+cu132" --index-url https://download.pytorch.org/whl/cu132
python -m pip install --upgrade "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
```

Verify package consistency and run a real CUDA calculation before launching:

```powershell
python -m pip check
python -c "import torch; x=torch.tensor([1.0,2.0,3.0],device='cuda'); y=((x+2)*3).sum(); torch.cuda.synchronize(); print(torch.cuda.get_device_name(0), y.item())"
python seed_vision.py --environment current --no-bootstrap --diagnostics
python seed_vision.py --environment current --no-bootstrap
```

The runtime gate accepts any in-range NVIDIA CUDA PyTorch build that passes the
calculation, including a working newer CUDA family; the preferred install pin is
not an equality requirement.

### Use an existing Python environment

The application runtime and the PlantCV reference environment have different
purposes. The application uses the dependencies already installed in the selected
interpreter; the frozen PlantCV environment remains an isolated Python 3.12
environment for reproducing reference calculations. Its lock file is byte-pinned:
the working-tree bytes must match the CRLF SHA-256 recorded in the provenance, even
though Git stores the text normalized. `.gitattributes` requests CRLF for fresh
checkouts, including when `core.autocrlf=false`. An existing checkout or a
downloaded copy that still has LF endings may need a one-time repair.

Run these commands with the interpreter that contains the dependencies you
installed:

```powershell
python scripts/repair_reference_lock.py
python scripts/repair_reference_lock.py --repair
python seed_vision.py --environment current --no-bootstrap --diagnostics
python seed_vision.py --environment current --no-bootstrap
```

The first lock command checks only; use `--repair` only when CRLF conversion alone
matches the existing pinned checksum. `--environment current` keeps the explicitly
selected interpreter, and `--no-bootstrap` prevents installation using the
baseline pins. The diagnostics command checks runtime compatibility before
launching; the final command launches Seed Fiddle.

## The window

- **Images** (left): the project's images. Hover an image for its status; the ✕
  removes it from the project.
- **Image** and **Pipeline** views (centre): the selected overlay on the image, and
  the node graph. Toggle either with `Ctrl+1` / `Ctrl+2`; both are visible by default.
- **Reference annotation** panel: a movable overlay over the image when you choose
  **Materials** or **Seeds** on the toolbar.
- **Right panel**: the result summary and **Results…** at the top, then two tabs:
  **Node** (inspector for the selected pipeline node) and **Image** (file,
  dimensions, species, analysis device and the calibration evidence found).

Menus: **File** (projects, images, snapshots, import/export), **Edit** (annotation
undo/redo, annotation modes, `Ctrl+K` node search), **Analysis** (run, stop,
results, run history, optimisation), **Learning** (seed outlines, datasets,
training, species libraries) and **Help** (operator guide `F1`, keyboard shortcuts,
runtime summary).

## Work through an image

1. **File → Open images…** (`Ctrl+O`) or open a saved project. Set the verified
   species in the **Image** tab; species is operator metadata and is never inferred.
2. Choose a pipeline from the selector at the top of the Pipeline view; more are
   under **… → Other pipelines**. A loaded pipeline is **Locked**: clear **Lock** to
   edit node parameters or wiring.
3. Paint material references (**Materials**: Background, Foreground, Other) and,
   where useful, seed outlines (**Seeds**). Missing required evidence makes a result
   *unavailable*; it is never reported as zero seeds.
4. **Run pipeline** (`Ctrl+R`); **Stop** with `Ctrl+.`. Changing one node recomputes
   only that node and its dependents.
5. **Results…** (`Ctrl+Shift+E`): accept or exclude each seed, compare instance
   methods, and export JSON, CSV, an annotated image or per-seed crops.
6. **File → Save project** (`Ctrl+S`). Use **File → Snapshots → Create snapshot…**
   for an immutable, verified copy of a project and its dependencies.

The [operator guide](docs/OPERATOR_GUIDE.md) covers every step in detail.

## Batch processing

A saved project can be re-run without the desktop, using the same recipe and
reference archives:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_pilot.py --project projects\analysis.seedfiddle-project.json --output-dir artifacts\project-results
```

Add `--seed-crop-padding-px 4` to also write per-seed image/mask crops.

## Development

```powershell
.\.venv\Scripts\python.exe -m tests            # core regression subset (24 units)
.\.venv\Scripts\python.exe -m tests --full     # all 649 units, about 14 minutes
.\.venv\Scripts\python.exe scripts\generate_node_catalogue.py --check
```

The full suite needs CUDA, the pinned PlantCV reference environment and reviewed
fixtures under `artifacts/`; see [Testing](docs/TESTING.md). Read
[AGENTS.md](AGENTS.md) and [HANDOFF.md](HANDOFF.md) before changing the code. The
[documentation index](docs/README.md) lists everything else.

Logs rotate under `%LOCALAPPDATA%\Seed Fiddle\logs\`; **Help → Runtime summary**
shows the exact paths.

## License

[MIT](LICENSE).
