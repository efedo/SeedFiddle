# Operator guide

How to use Seed Fiddle day to day. **Help → Operator guide** (`F1`) opens this file.
For installation see the [README](../README.md); for what the results can and
cannot support see [Current status](CURRENT_STATUS.md).

> Seed Fiddle's outputs are not yet validated against independent laboratory data.
> Treat counts, outlines, measurements and trait predictions as proposals to review.
> Physical units and unvalidated traits are withheld, and an uncalibrated score is
> not a probability of being correct.

## Runtime and CUDA readiness

Normal desktop, batch-analysis, and GPU-learning operation requires a supported
NVIDIA GPU, a compatible NVIDIA driver, an NVIDIA CUDA-enabled PyTorch build, and
a successful small CUDA calculation. Startup diagnostics check the PyTorch build
and actual device execution separately. CPU-only and ROCm builds are rejected;
when a valid NVIDIA CUDA build cannot use the GPU, diagnostics report the hardware
or driver failure instead of repeatedly reinstalling PyTorch.

Python 3.12 remains the application baseline. Python 3.15 is accepted by the
runtime gate when compatible dependencies are already installed; the baseline
pins are not a promise that a fresh Python 3.15 environment can be provisioned.
Use the same Python interpreter for pip, diagnostics, analysis and launch. On
Windows with Python 3.15, these are supported CUDA wheel installation options:

```powershell
python -m pip install --upgrade "torch==2.15.0+cu132" --index-url https://download.pytorch.org/whl/cu132
python -m pip install --upgrade "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
```

CUDA 13.2 supports Blackwell GPUs but requires a compatible newer NVIDIA driver.
CUDA 12.6 is an option for GPUs and drivers it supports, but does not cover
Blackwell. A working newer CUDA-enabled PyTorch build is accepted; Seed Fiddle
does not replace it just to match the preferred CUDA 12.6 installation pin.
Confirm dependencies and run a real device calculation:

```powershell
python -m pip check
python -c "import torch; x=torch.tensor([1.0,2.0,3.0],device='cuda'); y=((x+2)*3).sum(); torch.cuda.synchronize(); print(torch.cuda.get_device_name(0), y.item())"
python seed_vision.py --environment current --no-bootstrap --diagnostics
python seed_vision.py --environment current --no-bootstrap
```

`--environment current` keeps the explicitly selected interpreter and
`--no-bootstrap` prevents package installation. Diagnostics remain available when
readiness fails and return a nonzero status, with package-build and device-execution
details and suggested next steps. A launch that may install packages also adds the
optional Cellpose-SAM package when it is missing (see [Cellpose-SAM](#cellpose-sam));
`--no-cellpose` leaves it out.

## The window

| Area | Purpose |
|---|---|
| Toolbar | **Run pipeline**, **Stop**, **Image** / **Pipeline** view toggles, overlay controls, and **Materials** / **Seeds** annotation modes. |
| Images (left) | Images in the project. Icons show calculated, computed and unsaved states; hover for the text status. The ✕ on hover removes an image from the project. |
| Image view | The selected overlay on the current image. Drag to pan, scroll to zoom. |
| Pipeline view | The node graph for the current image. Both views are visible by default; `Ctrl+1` and `Ctrl+2` toggle each, `Ctrl+3` shows both. |
| Reference annotation panel | Floats over the image while **Materials** or **Seeds** is active; drag its title to move it and the corner grip to resize it. |
| Right panel | Result summary, **Results…** and, when Foreground evidence is missing, **Paint Foreground references…** at the top; below, the **Node** tab (inspector for the selected node) and the **Image** tab (file, dimensions, species, analysis device and calibration evidence). |

**Help → Keyboard shortcuts** lists every shortcut; the common ones are at the end
of this guide. Splitter sizes and the annotation panel position are remembered.

## Projects and images

- **File → New project** (`Ctrl+N`), **Open project…** (`Ctrl+Shift+O`),
  **Open recent project**, **Save project** (`Ctrl+S`) and **Save project as…**.
- **File → Open images…** (`Ctrl+O`) adds images. Set the verified species in the
  **Image** tab. Species starts *Unknown / unassigned* and is never inferred.
- For an immutable revision use **File → Snapshots → Create snapshot…**: it copies
  and verifies images, annotations, manual centres, enabled model checkpoints and
  the pinned species library, and writes its completion index last. **Create
  snapshot with reference checkpoints…** also includes saved pipeline checkpoints.
  **Restore snapshot…** creates a separate verified working copy and reports its
  path; open that with **File → Open project…**. Missing or modified components
  prevent restoration.
- **File → Import / export** loads or saves an analysis-settings profile, exports
  or imports annotations (below) and saves the current image viewport
  (`Ctrl+Alt+E`).
- Replacing an image file in place invalidates its cached calculations. Existing
  annotations will not silently bind to the new bytes: restore the original file or
  start a fresh project and re-validate the references.

### What is saved where

For a project saved as `name.seedfiddle-project.json`:

| What | Where | Written |
|---|---|---|
| Node settings and wiring, the image list with file fingerprints, species, canvas layout, recipes | `name.seedfiddle-project.json` | On **Save project** |
| Manual seed centres and reference layers (materials, seed outlines and seed traits), one file per image | `name.seedfiddle-annotations/manual-seed-centres/` and `…/reference-regions/` beside the project file | On **Save project** |
| Annotation edits made since the last save | `name.seedfiddle-annotations/unsaved/` | As you edit |
| Optimisation targets, policy and history | `name.seedfiddle-project.optimization*.json` beside the project file | When the optimiser runs |
| Models and native recipe inputs | `native-models/` beneath the workspace folder | On **Save project** |

Each project owns its annotations, so editing an image in one project never changes
another. Annotation edits count as unsaved project changes: **Save project** copies
them into the project's folder, and discarding the changes deletes them. If Seed
Fiddle closes unexpectedly, opening the project offers to restore them.

Without a saved project (the loose workspace at startup, or a new untitled
project) annotation edits are written as you make them to the workspace's shared
`projects/manual-seed-centres/` and `projects/reference-regions/` folders, and the
first **Save project** copies them into the new project's folder. A project saved
by an earlier version still points at those shared folders. It opens marked as
modified, and the next save copies its files into its own folder. The shared files
are left in place.

A project with annotations must be saved inside the workspace folder, because its
annotation folder sits beside the project file. Move the project file, its
annotation folder and `native-models/` together.

### Moving annotations between projects

**File → Import / export → Export annotations…** lists the images that have manual
seed centres or reference layers. Choose the images and kinds and save a
`.seedfiddle-annotations.zip` file. In the other project, **Import annotations…**
matches the file's annotations to images by exact file contents, so the same image
files must be in that project. Imported annotations replace the same kind of
annotation on that image and stay unsaved until you save the project.

## Pipelines

Every image is analysed by a pipeline: a typed graph whose root inputs are
**Project → Image selector → Species and metadata**, plus **Species reference
library**. These four nodes cannot be removed. The **Image selector** node and the
Images list choose the same current image.

### Choosing and locking

Choose a pipeline from the selector at the top of the Pipeline view: **Default seed
analysis**, the PlantCV colour-card correction, the three PlantCV seed workflows
(fixed binary, adaptive mean, Otsu) and the SMPTS paper reconstruction. Thirteen
more are under **… → Other pipelines** (extended thresholds, Naive Bayes, K-means,
calibrated and recovery variants).

A loaded pipeline is **Locked**: running, backend choice, layout and inspection
work, but parameters, wiring and bypass state cannot change. Clear **Lock** to edit.
Factory pipelines are never overwritten; re-selecting one restores its original
values.

### Running

- **Run pipeline** (`Ctrl+R`) calculates the current image; **Stop** (`Ctrl+.`)
  requests cancellation at the next safe boundary and discards obsolete outputs.
  Right-click a node for **Run to node**, which runs it and its enabled upstream
  nodes only. On a disabled node it reads **Enable and run to node** and enables the
  node first; if the node cannot be enabled (a missing optional package or
  checkpoint, an unconnected input or a disabled upstream node) a message says why.
  While **Lock** is set it reads **Run to node (clear Lock to enable it)**.
- A disabled node's status says why it is off: its own reason, *requires* a named
  disabled upstream node, *disabled by disconnected input*, or a missing optional
  package.
- While running, the activity panel shows the current node, whether it was
  recomputed or reused, and the elapsed time. A failed, partial or unavailable
  result shows its reason and selects the affected node. Read that reason before
  changing controls.
- Changing a parameter recomputes only that node and its dependents on the next run.
  Moving cards, choosing overlays or changing opacity never invalidates results.
- **Backend:** a global **Native / PlantCV** choice applies to every node; a node's
  own **Backend** setting can override it, or **Inherit**. PlantCV execution needs the
  pinned local reference environment; if it is unavailable the node reports that
  instead of silently running natively. Stages without a public PlantCV equivalent
  are native only, and the six watershed stages share one choice.
- Full-resolution analysis stays on the GPU. If a request exceeds a precision,
  memory or work limit it is reported as unsupported rather than run at a lower
  resolution.

### Inspecting and editing nodes

Select a node to show it in the **Node** tab: **How it works**, its parameters in
grouped sections (collapse a section with its arrow), **Reset** to restore the
loaded values, and **Enabled** where the node can be bypassed. Supported nodes
offer **Optimise this node…** (see [Optimisation](#optimisation)).

On the canvas:

- Node cards show their main outputs. Viewer-only outputs are folded behind a
  "▸ n more views" row; click it to expand or fold them. Connected outputs always
  show.
- With the pipeline unlocked, **Add node** inserts a catalogue node. **… → Remove
  selected added node** removes an added node and its private inputs.
- **Tools** offers **Auto arrange**, **Move to unused**, **Bundle cables** and **Route
  around nodes**; **Fit**, **100%** and **−** control the canvas zoom.
- `Ctrl+K` finds a node by name and frames it with its neighbours.

### Pipeline-specific inputs

- **Foreground region.** Working binary, adaptive-mean and Otsu threshold nodes and
  the SMPTS intersection node accept a rectangular region. Clear **Lock**, then use
  **Select foreground region…** to draw it, or enter it in the node's settings as
  `[left, top, right, bottom]` in that node's input pixels (right and bottom
  excluded); `null` means the whole frame. Foreground outside the rectangle is set
  to background.
- **Models.** Naive Bayes and K-means nodes have **Select model…** and **Train…**
  buttons. A selected model file must stay available and unchanged; reselecting a
  changed file recalculates its masks while reusing the prepared image. In the
  K-means category-mask node, **cat list** chooses clusters: `null` returns every
  cluster, `[0, 2]` returns one union mask.
- **Colour correction.** The built-in **Deskew and colour balance** node rectifies
  geometry and balances RGB channels from neutral patches. The PlantCV colour-card
  pipelines instead fit source patch colours to target colours (an affine transform
  for the automatic card route, a nine-term polynomial for four-input **Correct
  colour**). They complement each other; neither replaces the other.
- **Calibrated seed pipelines** crop an explicit seed region after card correction.
  Labels and coordinates in those outputs are local to the crop
  (parent xy = local xy + crop origin). The **Chip size calibration** stage converts
  sizes to card units when a chip size is set; otherwise sizes stay in pixels.

### Saving pipelines and layouts

The Pipeline header has **Save**, **Export** and the **…** menu:

- **Save** writes a custom pipeline file (calculation settings, node IDs, wiring,
  enabled state, backend choices, layout and referenced assets, but no pixels or
  results). It needs a name different from any factory. **… → Load pipeline…**
  opens one; referenced assets that are missing are reported.
- **Load layout…** / **Save layout as…** change node positions only and keep cached
  results.
- **Export run…** writes the completed run (including partial runs) to a new folder
  with `native-results.json`, output checkpoints and any selected model bytes.
- **Recorded run QC…** reviews and acknowledges recorded run outcomes.
- **Replay selected saved stage…** reruns one supported node from an exported run
  folder and reports match or mismatch with a fresh preview. A computational match
  does not certify source conformance or laboratory accuracy.
- **Source and availability…** explains a factory pipeline's provenance and which
  stages are unresolved (for example, original SMPTS source is unavailable).
- **Import legacy recipe…** loads an older saved recipe file.

## Viewing outputs

The **Overlay** menu lists **Source image**, **Calibrated image and annotations**,
and every available output grouped by node; type in its filter field to narrow the
list. **Opacity** blends a mask or label output over a matching RGB image (same
dimensions and frame). HSV controls appear only where they are meaningful.

Toolbar zoom keeps the viewport centre; the mouse wheel zooms about the pointer.
`F` fits the image and `1` shows actual size.

**Save current image viewport** (`Ctrl+Alt+E`) saves the displayed pixels with a
caption giving the overlay name, interpretation key and a pixel-scale bar. It does
not assert a physical length; use result export for structured measurements.

## Annotating references

Annotations are stored in corrected, deskewed image coordinates. They are shown,
and drawing is enabled, only while a deskewed view is displayed; the source image
and other views hide them because their coordinates do not align. Run Default seed
analysis first if the corrected frame is not yet available.

### Material references

Choose **Materials** (or **Paint Foreground references…** when a result reports
missing Foreground evidence), pick Background, Foreground or Other, and paint the region.
Use the exclusion controls only where a class model should ignore an area. **Apply
+ save** applies the draft to analysis and keeps it with the image's annotations
(see [What is saved where](#what-is-saved-where)). If writing it fails, the applied
state stays in memory and the failure is reported; **Save project** retries.

### Seed outlines

Choose **Seeds**. Then:

1. Select a seed ID, or use **Next empty** / **Next unannotated** (painted seeds whose
   condition or shape review is incomplete). **Show** displays all seeds or the
   current one. `Alt+Left` / `Alt+Right` step through annotated seeds.
2. Paint with Brush (`Ctrl+Alt+B`), Trace edge (`Ctrl+Alt+T`), Shape fill, Smart fill
   (`Ctrl+Alt+F`) or Eraser; each tool's tooltip explains it. **Clear seed** and
   **Clear all** sit beside the visibility control.
3. Set **Seed condition**: **No defects** and the defect choices are mutually
   exclusive. Choose Outline and Pose. A Complete outline checks and locks **Full
   length visible**; an incomplete outline can set it for size-only use.
4. Leave **Exclude from modelling** clear unless the seed must be omitted from model
   and library contributions.
5. **Pick location** places the hilum once on the current seed (Escape cancels);
   **Clear location** removes it. Direction is derived from the painted centroid.
6. **Apply + save**.

`Ctrl+Z` / `Ctrl+Shift+Z` undo and redo annotation edits. Applying or reprojecting
annotations is a history boundary, and history has count and size limits.

| State or command | Meaning |
|---|---|
| Draft | Editable or imported labels, separate from those used by analysis. |
| **Apply + save** | Uses the labels in analysis and keeps them with the image's annotations. In a saved project they are recorded by **Save project**. |
| **Learning → Seed outlines → Import bundled seed outlines…** | Loads the source-bound bundled labels for this image as an undoable draft. Imported outlines are not ground truth until reviewed. |
| **Import seed outlines from file…** | Loads a lossless categorical PNG, TIFF or NPZ whose size exactly matches the corrected image. |
| **Export seed outlines…** | Writes the applied labels as a 16-bit PNG in corrected coordinates. Apply a draft first to include it. |

Only connected, complete, shape-reviewed, non-excluded outlines supervise physical
contours, optimisation and training. Saving progress does not approve an
incomplete contour.

### Manual seed centres

Select the Procedural seed separation node and choose **Edit centres on image**.
Click empty space to add a centre, drag a marker to move it, and right-click or
press Delete to remove it. Each gesture is kept with the image's annotations
(recorded by **Save project** in a saved project) and recomputes only procedural
separation and what depends on it.

| Marker | Meaning |
|---|---|
| Hollow cyan circle | Automatic centre found by the procedural node |
| Filled yellow circle | Manual centre you added or moved |
| Grey × | Automatic centre you moved or removed; detection stays suppressed there. Delete it to restore detection. |
| Magenta square | Centre locked by an applied seed annotation |

**Mode** sets how manual centres combine with automatic detection. **Augment
automatic** keeps detection on and adds your centres, for correcting a few seeds.
**Replace automatic** turns detection off so only your manual centres (and
annotation-locked ones) seed instances, for marking every seed by hand. Replace
starts from your manual centres and never copies automatic ones.

## Results and export

Open **Results…** (`Ctrl+Shift+E`, or the button in the right panel).

- The table lists proposed, accepted, excluded and unreviewed seeds. Mark each seed;
  **Save review decisions** stores them against the exact source, recipe,
  calibration, annotation and prediction revision. Decisions record UTC time and
  the local account as an audit trail, not authentication.
- Filter by decision or seed ID and use **Next matching seed (Alt+N)** to step
  through a subset. Double-click a row to inspect that seed.
- **Show rejected candidates** reveals unselected procedural candidates, useful for
  finding missed atypical seeds.
- **Compare instance methods…** reports proposal counts for each available method.
  Inspect their overlays; count agreement alone is not validation.
- **Export JSON, CSV and annotated image…** writes a new result revision. Visible
  2-D pixel extents and areas are reported separately from inferred full shapes.
- **Export crops for visible rows…** writes, for each visible seed, a corrected image
  crop, an isolated mask and `seed-crops.json` with coordinates and provenance. A
  padded crop may contain neighbouring seeds; their IDs are recorded separately.
  Only a completed export is published.
- The **Predicted trait** column shows a stored trait class or an abstention reason;
  see [seed-trait training](TRAIT_CLASSIFIER_TRAINING.md).

**Analysis → Run history…** lists terminal records of failed, cancelled and
completed runs, which can be reviewed, acknowledged or exported as JSON.

*Unavailable* means mandatory evidence or an instance result is missing. It is
never reported as zero seeds. After **Run to node** on a node before every instance
node, the count reads *Not run*: nothing failed, and **Run pipeline** produces the
count.

## Optimisation

Save the project, apply the intended references and clear **Lock**. Use
**Analysis → Optimise → Optimise all pipeline nodes…** to search every eligible
node in pipeline order, or a node's own **Optimise this node…** button to search
only that node. A node stops early once a full pass improves nothing at the
displayed precision.

The plan shows eligible and excluded seeds and images, with reasons, and each
node's readiness. Only complete, connected, shape-reviewed, non-excluded outlines
are targets; apply or discard drafts first. Current-image and mixed
library/current-image modes work without a library pin; **Species library only**
needs a compatible pinned library and reports a missing or incompatible pin rather
than substituting another version.

Choose images and a budget, then review the proposed changes before applying them
together. Cancelling or changing inputs withholds the proposal. **Optimisation
history** reviews and rolls back earlier runs; **Import optimisation targets**
loads saved targets. Optimisation is in-sample adaptation to these references, not
independent validation. Details: [Node optimisation](NODE_OPTIMIZATION.md).

## References and recovery

New reference archives record the source-to-corrected transform, so a calibration
change reprojects labels exactly. Archives with an unknown legacy transform are
withheld. **Analysis → Reference alignment…** offers an explicit overlay review when
the canvases match; different canvases need the original transform or new
annotations. Keep the original archive.

Closing waits for running work, then asks to save, discard or cancel any drafts.
A failed save keeps edits in memory. Annotation edits not yet saved to a project
are also kept in its `unsaved/` folder, which the next open offers to restore.

## Learning

- **Learning → Seed outlines → Add image to training dataset…** exports a normalised
  crop, model features, seed labels, derived boundary targets and a manifest entry.
  It reports incomplete, unreviewed or excluded outlines and unapplied drafts first,
  and refuses to omit them silently. Export needs verified species and an explicit
  biological or capture group; related captures must share a split.
- **Audit learning dataset…** verifies component hashes, duplicate content and
  grouping.
- **Train or refine U-Net / StarDist…** trains a learned instance model. Learned
  nodes are disabled by default and need a compatible checkpoint. See
  [learned instance segmentation](LEARNED_INSTANCE_SEGMENTATION.md).
- **Train object-level seed traits…** and **Activate frozen seed-trait model…**: see
  [seed-trait training](TRAIT_CLASSIFIER_TRAINING.md).
- **Train PlantCV probability model…** trains a Naive Bayes model from a sampled RGB
  table or from image and mask directories, and saves the exact PlantCV TSV
  (optionally with hue/saturation/value density plots as SVG). **Train frozen
  K-means centres…** trains centres from a folder of images and saves them with
  their training provenance in one JSON file. Defaults match PlantCV.
- **Species reference libraries…** builds, validates, publishes, forks, imports,
  exports and pins immutable species libraries. Coverage reports inventory evidence,
  not accuracy. See [species reference libraries](SPECIES_REFERENCE_LIBRARIES.md).

To use an existing PlantCV K-means `.fit` model, convert it with the Python
environment that trained it, then select the JSON with **Select model…**:

```powershell
python -m seedvision.model_io.convert_plantcv_kmeans model.fit model.json
```

Feature specification v2 means raw Foreground colour evidence in both inference
and export. Version 1 foreground recipes are ambiguous and must be re-exported and
retrained. Checkpoints load only tensor and primitive state through PyTorch's
restricted loader.

## Cellpose-SAM

**Cellpose-SAM seed instances** (Seed detection category of the Default seed
analysis) runs the pretrained Cellpose-SAM model zero-shot on the calibrated colour
crop. It needs no training or checkpoint, but it needs the optional `cellpose`
package, which `seed_vision.py` installs when it is missing unless `--no-cellpose`
is given. In the default `--bootstrap ask` mode it lists the packages and asks
(Enter accepts); `--bootstrap auto` installs without asking into the project
`.venv`, or into the current interpreter with `--environment current`;
`--no-bootstrap` never installs. A first launch that creates `.venv` lists it with
the required packages. The install:

- adds torchvision from the installed PyTorch build's wheel index (`cu126`,
  `cu132` and so on), pinned against that build so pip picks the torchvision
  release made for it and leaves PyTorch alone;
- adds Cellpose's other dependencies at the `cellpose_dependencies` pins in
  `config/runtime_dependencies.json`;
- adds `cellpose` last and without dependencies, because its list names
  `opencv-python-headless`, which would replace the `cv2` module of the GUI OpenCV
  build.

`--offline` takes all of these from `wheels/`. A failed install is reported and
logged, and Seed Fiddle starts without the node. `--diagnostics` lists `cellpose`
as *OPTIONAL* while it is missing, and reports a *Conflict* with the repair
commands when `opencv-python-headless` is installed beside `opencv-python`.

To run it:

1. Clear **Lock**, select the node and tick **Enabled**, or right-click it and
   choose **Enable and run to node**. If the package is missing, a message says so
   and the node stays disabled; its status then reads *needs the optional cellpose
   package*.
2. **Run pipeline**. The first run downloads the `cpsam_v2` weights (about
   1.2 GB) into `~/.cellpose/models`, or `CELLPOSE_LOCAL_MODELS_PATH` if set, and
   loading them takes several seconds. Later runs reuse the loaded model.
3. Choose **Cellpose-SAM seed instances**, **cell probability** or **instance
   confidence** from the **Overlay** menu.

Results use the first enabled instance node that completed, in the order U-Net,
StarDist, Cellpose-SAM, procedural. With the learned nodes at their default
(disabled), an enabled Cellpose-SAM node therefore supplies the reviewed seeds and
measurements; disable it to return to procedural separation.

Controls:

- **Model**: a Cellpose pretrained name (`cpsam_v2`) or the path of a Cellpose
  weights file, absolute or relative to the project folder. **Rescale to estimated
  seed diameter** resizes the crop so the seed-scale estimate maps to Cellpose's
  30-pixel training diameter; off, the model relies on its own scale invariance.
  Changing either reruns the network.
- **Mask recovery**: **Cell-probability threshold** (logit; lower admits fainter
  pixels), **Flow-error threshold** (higher accepts masks whose flows agree less
  with their shape) and **Minimum instance area** (a fraction of the squared seed
  diameter). Objects centred outside the dish are dropped and the rest are clipped
  to it. These controls re-decode the cached flows without rerunning the network.

The output is unvalidated: no reviewed reference masks exist yet, so treat every
instance as a proposal to review.

## Batch processing

A saved project can be re-run without the desktop:

```powershell
python scripts/analyze_pilot.py --project projects/analysis.seedfiddle-project.json --output-dir artifacts/project-results
```

The batch uses the same saved recipe, reference archives and saved pipelines as the
desktop, and rejects unresolved fingerprints and unknown reference frames. Add
`--seed-crop-padding-px 4` to write per-seed crops. Saved PlantCV/SMPTS pipelines
produce `native_graphs/native-graphs.json` and per-graph output folders in each
report.

## Keyboard shortcuts

| Keys | Action |
|---|---|
| `Ctrl+R` / `Ctrl+.` | Run pipeline / Stop |
| `Ctrl+Shift+E` | Results |
| `Ctrl+1` / `Ctrl+2` / `Ctrl+3` | Toggle Image view / toggle Pipeline view / show both |
| `F` / `1` | Fit image / actual size |
| `Ctrl+K` | Find pipeline node |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo annotation edit |
| `Alt+Left` / `Alt+Right` | Previous / next annotated seed |
| `Ctrl+Alt+B` / `Ctrl+Alt+T` / `Ctrl+Alt+F` | Brush / edge trace / smart fill |
| `Alt+N` | Next matching seed (Results) |
| `Ctrl+N`, `Ctrl+Shift+O`, `Ctrl+S`, `Ctrl+Shift+S` | New, open, save, save project as |
| `Ctrl+O` | Open images |
| `Ctrl+Alt+E` | Save current image viewport |
| `F1` | This guide |
