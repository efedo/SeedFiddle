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
| Toolbar | **Run pipeline**, **Stop**, **Image** / **Pipeline** view toggles, **HSV value** for colour-gamut overlays, and **Materials** / **Seeds** annotation modes with their visibility and opacity. |
| Images (left) | Images in the project, each with a thumbnail. The filled row is the image open in the viewer. A click or arrow key only moves the outlined list cursor; double-click or `Enter` opens that image. A badge on the thumbnail shows that an image is computing, computed, or needs attention; no badge means analysis is still required. Each row also names its image's species: the species saved for that image, otherwise the project's, and *Unknown* until one is chosen. It sits under the name in a narrow panel, abbreviating the genus if it must (*L. mexicanus*), and becomes its own column once the panel is dragged to about 300 pixels wide. Hover for the text status and the full species. The **+** button above the list adds images. A crossed thumbnail means the file could not be read, and its tooltip gives the reason. The ✕ on hover removes an image from the project. |
| Image view | The selected overlay on the current image. Drag to pan, scroll to zoom. Its dark header holds **Overlay** and **Overlay opacity** on the left, the file name, and the zoom buttons on the right. |
| Pipeline view | The node graph for the current image. Both views are visible by default; `Ctrl+1` and `Ctrl+2` toggle each, `Ctrl+3` shows both. |
| Reference annotation panel | Floats over the image while **Materials** or **Seeds** is active and is as tall as its controls. Drag its title bar to move it and the corner grip to resize it. The title bar also holds **Undo**, a chevron that rolls the panel up to its title bar and back, and **×**, which finishes annotating as turning the mode off on the toolbar does (unapplied edits stay as a draft). |
| Right panel | Result summary, **Results…** and, when Foreground evidence is missing, **Paint Foreground references…** at the top; below, the **Node** tab (inspector for the selected node) and the **Image** tab (file, dimensions, species, analysis device and calibration evidence). |

**Help → Keyboard shortcuts** lists every shortcut; the common ones are at the end
of this guide. Splitter sizes and the annotation panel position are remembered.

## Projects and images

- Seed Fiddle always works in a project. It starts in a new, empty, untitled
  project; images are added to the open project and saved with it.
- **File → New project** (`Ctrl+N`), **Open project…** (`Ctrl+Shift+O`), **Save
  project** (`Ctrl+S`) and **Save project as…**. The eight most recent projects are
  listed by number near the bottom of the **File** menu; choose one to open it
  again. Same-named projects show their folder, and hovering shows the full path.
- **File → Open images…** (`Ctrl+O`) or the **+** button above the Images list adds
  images to the project. The **Image** tab
  shows the open image's verified species, and changing it changes that image only.
  An image with no species of its own follows the project's. The first species you
  choose becomes the project's, so a one-species project sets it once; with no image
  open, the selector sets the project's. Species starts *Unknown* and is never
  inferred.
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
| Node settings, added nodes, named pipeline copies, the image list with file fingerprints, species and canvas layout | `name.seedfiddle-project.json` | On **Save project** |
| Manual seed centres and reference layers (materials, seed outlines and seed traits), one file per image | `name.seedfiddle-annotations/manual-seed-centres/` and `…/reference-regions/` beside the project file | On **Save project** |
| Annotation edits made since the last save | `name.seedfiddle-annotations/unsaved/` | As you edit |
| Optimisation targets, policy and history | `name.seedfiddle-project.optimization*.json` beside the project file | When the optimiser runs |

Each project owns its annotations, so editing an image in one project never changes
another. Annotation edits count as unsaved project changes: **Save project** copies
them into the project's folder, and discarding the changes deletes them. If Seed
Fiddle closes unexpectedly, opening the project offers to restore them.

Until a new project is first saved, annotation edits are written as you make them
to the workspace's shared `projects/manual-seed-centres/` and
`projects/reference-regions/` folders, and the first **Save project** copies them
into the new project's folder. A project saved by an earlier version still points
at those shared folders. It opens marked as modified, and the next save copies its
files into its own folder. The shared files are left in place. A saved project loads
only the annotations it records: a shared file for an image added to it later is
named under **References and annotations**, but not loaded.

A project with annotations must be saved inside the workspace folder, because its
annotation folder sits beside the project file. Move the project file and its
annotation folder together.

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

Choose a pipeline from the **Pipeline** list at the top of the Pipeline view. Under
**Default pipelines** it offers **Procedural seed analysis** and **Cellpose seed
analysis**; your own editable copies are listed under **Saved in this project**.

**Procedural seed analysis** separates seeds from material, edge and shape evidence
built from automatic and painted references, and includes the optional U-Net and StarDist
nodes. **Cellpose seed analysis** holds [Cellpose-SAM](#cellpose-sam) and only the
nodes it needs: inputs, deskew and colour correction, ruler, layout, and seed scale. To
compare both on one graph, make an editable copy of Procedural seed analysis and add
Cellpose-SAM from **Add node**.

Choosing the other default pipeline loads it fresh and arranges its nodes
automatically; default pipelines are never overwritten. A project or pipeline file
opens with the layout it was saved with. Returning to the default pipeline the graph
came from keeps the project's own graph, including its layout; changing between the
two, or leaving an editable copy of one, restores the factory values and layout, and
the edits stay in the copy. Changing between them clears the open result until you
run again; nodes they share stay calculated.

Default pipelines are always locked and cannot be renamed. Running, inspecting
outputs and arranging nodes work. Parameter editors, wiring and the **Enabled** checkbox are read-only,
and the **Node** tab says so and links to making a copy.

**Add node**, **Move to unused**, **… → Remove selected added node** and the rename
button beside the list stay available. Using one says that default pipelines cannot
be edited and asks whether to create an editable copy, as do **Enable and run to
node**, the background toolbar checkboxes, loading a settings profile and activating
a trait model. **Yes** asks for a name, then loads the copy under **Saved in this
project** and carries out what you asked for; nothing needs recalculating. **No**, or
cancelling the name, leaves the default pipeline unchanged. A blank name, one already
used in the project and the name of a default pipeline are refused, and the prompt
asks again.

On an editable copy the rename button renames it. The project file keeps every copy
and reopens on the one you were using; choose one in the list to return to it.

### Running

- **Run pipeline** (`Ctrl+R`) calculates the current image; **Stop** (`Ctrl+.`)
  takes effect before the next GPU operation, or within the long CPU-only steps.
  The image's finished node results are kept, so the next run recalculates only what
  was still owed.
  Right-click a node for **Run to node**, which runs that node, plus any enabled
  upstream node that has changed or has no result yet. Other nodes keep the results
  they have. Nodes that depend on a changed node show *Out of date*, and the next
  **Run pipeline** recalculates them. If a change moves the dish crop, the kept
  results no longer line up, so the core stages run on and every other node waits
  for the next **Run pipeline**.
  On a disabled node it reads **Enable and run to node** and enables the
  node first; if the node cannot be enabled (a missing optional package or
  checkpoint, an unconnected input or a disabled upstream node) a message says why.
  On a default pipeline it offers an editable copy first.
- A disabled node's status says why it is off: its own reason, *requires* a named
  disabled upstream node, *disabled by disconnected input*, or a missing optional
  package.
- While running, the activity panel shows the current node, whether it was
  recomputed or reused, and the elapsed time. A failed, partial or unavailable
  result shows its reason and selects the affected node. Read that reason before
  changing controls.
- Changing a parameter recomputes only that node and its dependents on the next run.
  Moving cards, choosing overlays or changing opacity never invalidates results.
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
- **Add node** inserts another copy of a default node, with its own settings and
  results; a check marks each node the pipeline already uses. **… → Remove selected added node** removes an added node and its
  private inputs. On a default pipeline both offer an editable copy first.
- **Auto arrange** lays the nodes out in dependency columns. **Move to unused**,
  **Bundle cables** and **Route around nodes** sit beside it, or under **Tools** when
  the pane is narrow; **Fit**, **100%** and **−** control the canvas zoom.
- `Ctrl+K` finds a node by name and frames it with its neighbours.

### Colour chart and ruler

- **Colour chart.** The chart may lie anywhere in the photograph and at any
  angle. Its patches are found as a grid of uniform squares, and the chart is
  identified from the order of its colours in each channel, which exposure,
  white balance and tone curve do not change: the Datacolor SpyderCHECKR 24 and
  the X-Rite ColorChecker Classic 24 (also the layout of the CameraTrax
  24ColorCard). Patches are numbered in the chart's own upright reading order
  however it lies: ColorChecker greys are 19–24 from white to black, as X-Rite
  numbers them; SpyderCHECKR greys run down the first column (1, 5, 9, 13, 17,
  21). A chart cut by the frame is numbered from its visible patches. Any other
  grid of at least 12 patches, three or more each way, with several saturated
  colours and a wide range of lightness, is reported as an unrecognised chart
  and numbered row by row as it lies in the image.
- **Ruler.** The ruler may also lie anywhere and at any angle, outside the chart.
  The strongest pairs of parallel edges a ruler-width apart are measured in turn
  until one carries a complete metric tick train of **Expected visible span**
  (**Span mm**, on **Ruler detection and scale**; default 150 mm), with a whole
  centimetre at each end. When no band does, as for a ruler cut by the frame or
  the chart, partly covered, or shorter or longer than the span, each band is
  measured again over its longest visible run of millimetre ticks: at least 3 cm
  with nearly every tick present, so barcodes and textures do not qualify.
  Scales printed without 5 mm marks are accepted. An end of the run with blank
  ruler beyond it must be a centimetre mark; an end cut by the frame, the chart
  or an object may fall anywhere. The imperial scale is not sought in this case,
  and tick labels appear only when the zero end is in view. The scale is the
  least-squares slope of tick position against tick index, so it resolves a
  fraction of a pixel per millimetre. Tick labels count from the end where the
  metric ticks run along the ruler's upper edge, viewed along the ruler; this
  affects only the labels, not the scale. When neither pass finds a scale, a band
  whose millimetre ticks still form a regular train along the expected span is
  shown as the ruler outline, without a scale. Any other band, such as a printed
  sample label, a box edge or graph paper, is not reported as a ruler.

### Colour correction

- **Colour correction.** The **Deskew and colour correction** node rectifies the
  image from the colour chart and, by default, corrects its colours against the
  chart's published reference values (**Chart reference fit**): every patch is
  matched to its reference colour, so colour balance, grey balance and levels are
  corrected together. The Datacolor SpyderCHECKR 24 and X-Rite ColorChecker
  Classic are known, at any orientation. The neutral patches carry extra weight so
  greys stay grey. The node status reports the chart and the mean colour error
  (CIEDE2000) of the fitted patches and of each patch held out of its own fit.
  When the chart is not recognised or too few patches are usable, the node falls
  back to grey balance (channel gains from the neutral patches) and warns.
  **Correction model** set to **Grey balance only** always uses the gains.
### Saving pipelines

Three compact icons sit beside the **Pipeline** list: the pencil renames the pipeline,
the folder loads one from a file and the disk saves one to a file. Hover over an icon
for its name. **Export** and the **…** menu follow them.

- The disk writes a custom pipeline file (calculation settings, added nodes and
  layout, but no pixels or results). It needs a name different from any default
  pipeline, and work carries on in an editable copy of that name. The folder opens a
  file as an editable copy, numbering its name if the project already has that one.
  Neither offers a copy first, because neither changes a default pipeline.
- Node positions are saved with the project and with each pipeline file, and are
  restored when either is opened. Moving nodes never invalidates cached results.
- **Export** writes the selected added node's result report to a folder you choose;
  with no added node selected it opens **Results…**.
- The **…** menu holds **Remove selected added node**, and **Export run…** when a
  narrow pane hides the **Export** button.

## Viewing outputs

The overlay controls sit at the left of the image view's header, and stay there when
a pipeline's node output or the source image replaces the image view. The **Overlay**
menu lists **Source image**, **Calibrated image and annotations**, and every
available output grouped by node; type in its filter field to narrow the list.
**Overlay opacity** blends a mask or label output over a matching RGB image (same
dimensions and frame). The toolbar's HSV controls appear only where they are
meaningful. The header belongs to the image pane, so hiding that pane hides it too.

When the pane narrows, the header gives up width in this order: the opacity slider and
label shorten, the overlay name is shortened, the labels and percentage go, the file
name is shortened with an ellipsis in its middle, and in the narrowest panes the slider
goes. The zoom buttons always stay. Hover for the full overlay name or file path, and
widen the pane to get a hidden slider back.

The header's zoom buttons keep the viewport centre; the mouse wheel zooms about the
pointer. `F` fits the image and `1` shows actual size.

The **Procedural centre likelihood** overlay rings every watershed marker proposal.
Plain rings are the markers that became seeds. Orange crossed rings are proposals
that left no seed; hover one for its reason, and read the totals per reason in the
legend.

**Save current image viewport** (`Ctrl+Alt+E`) saves the displayed pixels with a
caption giving the overlay name, interpretation key and a pixel-scale bar. It does
not assert a physical length; use result export for structured measurements.

## Annotating references

Annotations are stored in corrected, deskewed image coordinates. They are shown,
and drawing is enabled, only while a deskewed view is displayed; the source image
and other views hide them because their coordinates do not align. Run Default seed
analysis first if the corrected frame is not yet available.

### Material references

Every image starts with automatic references, so material evidence works before
anything is painted. The **Automatic material references** node shows them: green
Foreground, orange Background and magenta Other.

- **Foreground**: the inset interiors of confident seeds. These are the isolated
  reference seeds when the reference region holds them. Otherwise they are seeds that
  **Reference seed dimensions and shape** finds isolated, compact and of consistent
  size, inside the vessel or, when no vessel rim is visible, anywhere outside the
  chart and ruler. The search tolerates shading and a patchy backdrop, ignores dust,
  and needs at least a quarter of a seed width of clear backdrop around each seed, so
  a pile or a tightly packed tray gives no confident seeds rather than wrong ones.
  The clear ring must look like the backdrop, so a hilum on a piled seed or a mark
  on a box wall does not count, and thin strokes such as printed characters or a
  gloss ring do not either. Blobs only a few pixels wide, such as barcode bars
  broken up by the mask cleaning, are too small to judge and are not taken.
- **Other**: the vessel wall, when its rim is visible around most of the fitted edge,
  and the detected colour chart and ruler. Seeds heaped against the wall are left out
  of it: seed-coloured blobs in the wall band stay unlabelled, while the rim's own thin
  lines remain Other.
- **Background**: the ring outside the vessel. When there is no vessel, or the ring's
  colour differs from the surface around the confident seeds (a lined dish, say),
  rings around those seeds are used instead, without their darkest part (cast
  shadows). With neither, a band along the image border is used.

Without a visible vessel, the confident seeds also set the analysis area: the
area they span, grown by two seed widths or a quarter of that span, whichever is
larger, less the chart and ruler. Seeds well beyond the outermost confident ones
can fall outside it. With fewer than three confident seeds, as in most piles, the
whole image less the chart and ruler is analysed, and labels, logos, the chart's
frame and any part of a ruler beyond its measured span can then draw proposals.
The views then draw the analysed area, dashed, instead of the fitted circle, and
**Layout detection** reports no vessel. With no seed measured at all, the seed
size is a fixed fraction of that circle and **Reference seed dimensions and
shape** says so rather than marking a seed.

Painting adds to these and outweighs them: the more you paint of a class, the less
its automatic area counts, down to the node's floor weight, and painted pixels are
never overridden. Turn a class off in the node to rely on painting alone.

To paint, choose **Materials** (or **Paint Foreground references…** when a result
reports missing Foreground evidence), pick Background, Foreground or Other, and paint
the region.
Use the exclusion controls only where a class model should ignore an area. **Apply
+ save** applies the draft to analysis and keeps it with the image's annotations
(see [What is saved where](#what-is-saved-where)). If writing it fails, the applied
state stays in memory and the failure is reported; **Save project** retries.

### Seed outlines

Choose **Seeds**. Then:

1. Select a seed ID, or use **Next empty** / **Next unannotated** (painted seeds whose
   condition or shape review is incomplete). **Select** then a click on a painted
   seed makes it current and returns to the previous tool; a click where no seed
   is painted changes nothing. **Show** displays all seeds or the current one.
   `Alt+Left` / `Alt+Right` step through annotated seeds.
2. Paint with Brush (`Ctrl+Alt+B`), Trace edge (`Ctrl+Alt+T`), Shape fill, Smart fill
   (`Ctrl+Alt+F`) or Eraser; each tool's tooltip explains it. **Clear seed** and
   **Clear all** sit beside the visibility control.
3. Under **Seed traits**, set **Condition**: **No defects** and the defect choices
   are mutually exclusive. Choose Outline and Pose. A Complete outline checks and
   locks **Full length visible**; an incomplete outline can set it for size-only use.
4. Leave **Exclude from modelling** (beside **Apply + save**) clear unless the seed
   must be omitted from model and library contributions.
5. On the **Hilum** row, **Pick location** places the hilum once on the current seed
   (Escape cancels); **Clear location** removes it. Direction is derived from the
   painted centroid.
6. **Apply + save**.

A cross marks each painted seed's centroid. It scales with the image and spans a
quarter of the median seed length, so crowded seeds stay visible when zoomed out.

`Ctrl+Z` / `Ctrl+Shift+Z` undo and redo annotation edits. Applying or reprojecting
annotations is a history boundary, and history has count and size limits.

| State or command | Meaning |
|---|---|
| Draft | Editable or imported labels, separate from those used by analysis. |
| **Apply + save** | Uses the labels in analysis and keeps them with the image's annotations. In a saved project they are recorded by **Save project**. |
| **Learning → Seed outlines → Import bundled seed outlines…** | Loads the source-bound bundled labels for this image as an undoable draft. Imported outlines are not ground truth until reviewed. |
| **Import seed outlines from file…** | Loads a lossless categorical PNG, TIFF or NPZ whose size exactly matches the corrected image. |
| **Copy Cellpose-SAM seeds to outlines** | Turns this image's [Cellpose-SAM](#cellpose-sam) seeds into an undoable draft. When the image already has outlines, **Add missing seeds** keeps them and adds only Cellpose seeds that are mostly not outlined yet (on unoutlined pixels, with new IDs); **Replace outlines** discards them. |
| **Export seed outlines…** | Writes the applied labels as a 16-bit PNG in corrected coordinates. Apply a draft first to include it. |

Optimisation and species libraries use only connected, complete, shape-reviewed,
non-excluded outlines. Training and reference texture prototypes use every connected,
non-excluded outline, reviewed or not: an unlabelled seed would train as background, and
reviewed seeds alone can sit in one part of the image. Saving progress does not approve
an incomplete contour.

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

Markers keep a constant screen size while you edit, so they stay easy to grab.
Outside the editor they scale with the image and span a quarter of the
estimated seed diameter.

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

On an image whose run did not complete, **Results…** says how the run ended and why.

*Unavailable* means mandatory evidence or an instance result is missing. It is
never reported as zero seeds. After **Run to node** on a node before every instance
node, the count stays if nothing it depends on changed. Otherwise it reads *Not run*:
nothing failed, and **Run pipeline** produces the count.

## Optimisation

Save the project, apply the intended references and work on an editable copy. Use
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

Reference archives record the source-to-corrected transform, so a calibration
change reprojects labels exactly. Seed Fiddle opens only the current file formats:
a project, settings profile, reference archive or seed-centre file written by an
earlier format is refused with an unsupported-version message rather than migrated.

Closing waits for running work, then asks to save, discard or cancel any drafts.
A failed save keeps edits in memory. Annotation edits not yet saved to a project
are also kept in its `unsaved/` folder, which the next open offers to restore.

## Learning

- **Train or refine U-Net / StarDist…** builds its own dataset and trains on it. Its
  table lists every project image with seed outlines, with its seed count and species,
  and assigns **Training** and **Validation**: about one image in five per species,
  never a species' largest. Every image starts in its own group. Give related captures
  the same group, and they keep one split. **Test** keeps an image out of training and
  decoder tuning, for the
  [training experiment](LEARNED_INSTANCE_SEGMENTATION.md#training-experiment) to score.
  An image with disconnected or excluded seeds
  or an unapplied draft stays listed with the reason and cannot be used, because a seed
  left out would train as background. Tick **Reviewed** only for an image a person
  checked seed by seed; seeds need no per-seed review.
- Train asks before using images not marked **Reviewed** and records that choice with
  the model. It then analyses any image that needs it, using only the pipeline stages
  the model's inputs come from, saves normalised crops, model features, seed labels and
  derived boundary targets, verifies the dataset's hashes, duplicate content and
  grouping, and starts training. **Cancel** works at any point: before training it
  returns to the image you were on and trains nothing; during training, where **Stop**
  does the same, it ends after the current batch, keeps the best checkpoint so far and
  activates nothing. Each run, cancelled or not, writes its own folder under
  `learning-data/`, which can be deleted once it is no longer wanted. Learned nodes are
  disabled by default and need a compatible checkpoint.
  See [learned instance segmentation](LEARNED_INSTANCE_SEGMENTATION.md).
- **Train object-level seed traits…**, **Import seed-trait model…** and **Export
  seed-trait model…**: see [seed-trait training](TRAIT_CLASSIFIER_TRAINING.md).
- **Species reference libraries…** creates, uses, imports and exports species
  libraries. A library describes one species: **Species** at the top of the window,
  set to the open image's species, picks the species every tab works on.
  - **Libraries** lists that species' versions. With none yet, **Create a library…**
    leads to the next tab. **Use for this project** pins the selected version; only a
    library of the open image's species can be pinned. **Export…**, **Retire** and
    **Import library file…** complete the set.
  - **Create a library** lists the project's images of that species, then those with
    no species set; creating the library records the checked ones as that species,
    so a new project need not set **Species** image by image first. Images with
    saved annotations are checked, and images without them are shown greyed. Give
    the version a name, optionally start from an existing version (keeping its
    images, each removable), and choose **Create *species* library**. The images are
    checked before anything is published, and a version never changes afterwards.
    Optional lineage, accession, seed-lot and repeated-photo identifiers are under
    **Source details**.
  - **Details** shows each evidence product's coverage and validation. Coverage
    counts evidence, not accuracy.

  A project pins one library. On an image of another species the desktop marks the
  library node blocked and analyses without the library; batch runs and optimisation
  refuse that image. See [species reference libraries](SPECIES_REFERENCE_LIBRARIES.md).

Feature specification v2 means raw Foreground colour evidence in both inference
and export. Datasets and checkpoints recorded with version 1 are refused: export
the dataset again and retrain. Checkpoints load only tensor and primitive state
through PyTorch's restricted loader.

## Cellpose-SAM

**Cellpose-SAM seed instances** is the seed-detection node of **Cellpose seed
analysis**. It runs the pretrained Cellpose-SAM model zero-shot on the calibrated colour
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

1. Choose **Cellpose seed analysis** in the **Pipeline** list. If the package is
   missing, a message says so and the list stays where it was.
2. **Run pipeline**. The first run downloads the `cpsam_v2` weights (about
   1.2 GB) into `~/.cellpose/models`, or `CELLPOSE_LOCAL_MODELS_PATH` if set, and
   loading them takes several seconds. Later runs reuse the loaded model.
3. Choose **Cellpose-SAM seed instances**, **cell probability** or **instance
   confidence** from the **Overlay** menu.

To run it beside procedural separation, add it from **Add node** on an editable copy
of Procedural seed analysis, then tick **Enabled**, or right-click it and choose
**Enable and run to node**. Results use the first enabled instance node that
completed, in the order U-Net, StarDist, Cellpose-SAM, procedural, so on such a copy
Cellpose-SAM supplies the reviewed seeds and measurements until you disable it.

Controls:

- **Model**: a Cellpose pretrained name (`cpsam_v2`) or the path of a Cellpose
  weights file, absolute or relative to the project folder. **Rescale to estimated
  seed diameter** resizes the crop so the seed-scale estimate maps to Cellpose's
  30-pixel training diameter; off, the model relies on its own scale invariance.
  Changing either reruns the network.
- **Mask recovery**: **Cell-probability threshold** (logit; lower admits fainter
  pixels), **Flow-error threshold** (higher accepts masks whose flows agree less
  with their shape) and **Minimum instance area** (a fraction of the squared seed
  diameter). Objects centred outside the dish (or, without a vessel, the analysis
  area) are dropped and the rest are clipped to it. These controls re-decode the cached flows without rerunning the network.

The output is unvalidated: no reviewed reference masks exist yet, so treat every
instance as a proposal to review. To start a reviewed reference from it, use
**Learning → Seed outlines → Copy Cellpose-SAM seeds to outlines** (see
[Seed outlines](#seed-outlines)), correct the draft and apply it. Seeds whose centre
lies outside the detected dish are missing from the copy, so check the dish outline
first.

## Batch processing

A saved project can be re-run without the desktop:

```powershell
python scripts/analyze_pilot.py --project projects/analysis.seedfiddle-project.json --output-dir artifacts/project-results
```

The batch uses the same saved settings, added nodes and reference archives as the
desktop, and rejects unresolved fingerprints and unknown reference frames. Add
`--seed-crop-padding-px 4` to write per-seed crops. Added nodes write their results
to a `catalogue_nodes/` folder in each report.

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
