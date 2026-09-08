# SeedFiddle hand-off

## Resume here

Repository: <https://github.com/efedo/SeedFiddle>

Primary branch: `main`

Launch point: `seed_vision.py`

This is an on-premise PySide6 desktop application for counting, separating,
measuring, and broadly classifying soybean and lupin seeds in calibrated lab
photographs. Images may contain densely touching or overlapping seeds.

Hard constraints:

- No local server, installer, or bundled executable.
- One top-level Python launch file; implementation modules may remain under
  `seedvision/`.
- A virtual environment is optional, but the launcher can create and populate
  `.venv`.
- Full-raster analysis is GPU-first PyTorch CUDA. Keep tensors and cached
  intermediates on the GPU until display or compact metadata requires transfer.

## Latest update — 2026-09-08: compact seed annotation controls and centroids

- Seed selector is now `Seed:` / colour swatch / four-digit numeric spinner
  (1–9999) / `Next empty`. `Show selected only` appears once, on the next row;
  the second existing-seed dropdown is removed. The red `empty` notice remains.
  Next empty reports capacity rather than selecting an occupied ID when all
  9999 IDs are used. Existing uint16 annotation archives are not renumbered.
- Removed routine visible boundary-supervision and coat-vocabulary messages;
  contextual instructions are tooltips. Invalid legacy shape metadata still
  shows a visible warning. Condition, shape, painting, and Apply/save mechanics
  are retained.
- `seedvision/annotation/geometry.py` calculates area centroids of each painted
  ID in bounded row chunks. All pixels count, including disconnected fragments.
  A partial outline uses the **visible painted area's centroid**, not an
  inferred complete-seed centre. The editable CPU annotation raster is already
  available; no CUDA analysis raster is downloaded for this operation.
- Image overlays show constant-screen-size, white crosshairs with dark halos
  at these centroids. They obey annotation visibility, selected-only filtering,
  and independent annotation opacity. Cached geometry survives display-only
  changes, is invalidated by mask edits/replacement, and is cleared on image
  switches. Crosshairs never become manual detection markers or model inputs.
  Brush commits refresh centroid geometry **before** emitting the edit signal,
  so synchronous hilum-inspector listeners cannot read the previous stroke's
  cached centre. A real Qt brush-event regression verifies this ordering.
- Hilum direction is now the unit vector **from the painted seed centroid to
  the hilum point**, not a user-dragged vector. Pick/drag places or moves the
  landmark; the inspector shows its derived angle (clockwise from image-right).
  Direction is unknown at the centroid or without a landmark. It updates with
  mask edits/undo and is derived again when applying annotations, using the
  existing archive metadata fields. Merely displaying old archives does not
  rewrite their saved metadata. Manual direction controls are removed.
- Regression coverage: `tests/test_annotation_centres.py` exercises geometry,
  filtering/opacity/cache independence, compact controls, edit/undo/save
  round-trips, brush-event ordering, missing direction, and ID exhaustion.
  A dense synthetic 6240×4160 raster with 6040 IDs took 0.547 s for its initial
  centroid calculation; display-only changes reuse those compact coordinates.
  Qt visual checks are in
  ignored `artifacts/annotation-controls-compact.png` and
  `artifacts/annotation-centre-crosshairs.png`. All **58 focused tests passed**
  (16.725 s), logged in `artifacts/annotation-controls-focused-2026-09-08.log`.
  Final full discovery: **611 run, 608 passed, 2 skipped, 1 pre-existing
  missing-fixture failure** (276.125 s). The unchanged reference manifest still
  points to absent `images/IMG_9689c.JPG`; the fixture check remains intact.
  Log: `artifacts/full-suite-2026-09-08-annotation-controls-verified.log`.
  No commit/push requested in this turn; earlier local changes
  and user-owned draft/project files remain untouched.

## Previous update — 2026-09-08: reference-gradient input and viewer controls

- Checkpointed preceding work as `95e8032` before this request. No push requested.
  Local `analysis7.seedfiddle-project.json`, `seedfiddle_prompts.md`, and the
  pre-existing deletion of `prompts.txt` were deliberately not included.
- Reference edges consumes `edge_gradients.magnitude`, not generic thinned
  ridges. Continuous CUDA gradient support multiplies Physical compatibility;
  normalization also starts from float gradient support. The node applies its
  own NMS/hysteresis afterwards. Raw prototype calculations and conservative
  subtraction policy are unchanged. Settings format is now **20**; versions
  1–19 migrate the old ridge input and preserve disconnected/suspended branches.
- Toolbar: Run pipeline; independent checked/depressed Image/Pipeline view
  toggles; removed duplicate top-level fit/actual-size buttons (menu shortcuts
  remain); analytical opacity slider maximum 102 px instead of 170 px. Moved
  material/seed visibility controls out of the painting panel, after Annotate
  seed instances, with their own annotation-only opacity slider. These controls
  neither edit masks nor invalidate calculation caches.
- Pipeline header uses viewport width, includes Fit and 100%, and collapses
  editing commands into a Tools menu on narrow panes. Selected nodes have a
  saturated blue header/body and a 5 px cosmetic cyan outline; their status
  colour remains visible as a separate strip.
- Material colour overlays are ordered Foreground probability/HSV, Background
  probability/HSV, Other probability/HSV, then foreground-vs-nonseed excess.
  Other HSV reads the actual Other fit stored under the historical
  `excluded_component_*` metadata names. It does not refit or use Background's
  configurable chroma/frequency weights. Missing fits show an explicit message.
- See `tests/test_reference_edge_toolbar_revision.py` for display independence,
  narrow-header controls, overlay/socket ownership, and numeric production-vs-HSV
  agreement. `docs/LOCAL_EDGE_NORMALIZATION.md` now describes continuous support.
- Verification: 75 focused graph/migration/gradient/toolbar checks passed;
  final UI/project/toolbar checks **95 passed** (75.670 s). Full `unittest`
  discovery: **602 run, 599 passed, 2 skipped, 1 pre-existing failure**
  (283.942 s). The sole failure remains the committed reference manifest's
  missing `images/IMG_9689c.JPG`; no fixture or manifest was removed or hidden.
  Logs are in ignored `artifacts/full-suite-2026-09-08.log` and
  `artifacts/final-ui-checks-2026-09-08.log`. Qt visual checks used actual Windows
  fonts at 1360 and 1920 px; pipeline controls were also tested at 420 px.

## New Windows computer setup

1. Install Git and a 64-bit Python version allowed by
   `config/runtime_dependencies.json`. The current supported range is Python
   3.12 through 3.14; Python 3.12 is the recommended baseline. Do not use a
   newer unsupported interpreter merely because it is the newest release.
2. Clone and enter the repository:

   ```powershell
   git clone https://github.com/efedo/SeedFiddle.git
   Set-Location .\SeedFiddle
   ```

3. Create the optional project venv, install the pinned runtime packages, and
   launch:

   ```powershell
   py -3.12 .\seed_vision.py --environment venv --bootstrap auto
   ```

   The launcher installs PySide6, NumPy, OpenCV, and the configured CUDA-enabled
   PyTorch wheel. Online bootstrap requires normal package-index access.
   `--offline` works only after compatible wheel files have been placed in
   `wheels/`; wheel binaries are intentionally not committed.

4. Verify the runtime and GPU:

   ```powershell
   .\.venv\Scripts\python.exe .\seed_vision.py --diagnostics
   ```

   Confirm that PyTorch is compatible and the acceleration line reports CUDA.
   The NVIDIA driver must support the configured PyTorch CUDA wheel; a separate
   system CUDA toolkit is normally unnecessary.

5. Run the test suite:

   ```powershell
   $env:QT_QPA_PLATFORM = "offscreen"
   .\.venv\Scripts\python.exe -m unittest discover -s tests
   ```

6. Launch later with:

   ```powershell
   .\.venv\Scripts\python.exe .\seed_vision.py --no-bootstrap
   ```

If the venv bootstrap fails, delete only the incomplete `.venv`, install the
recommended supported Python version, and rerun step 3. Never delete the
repository or `images/` while troubleshooting the environment.

## Current implementation

- **The complete right-pane node-control catalogue is audited and grouped by
  operation.** All 271 controls across 38 configurable active/toolbox nodes are
  assigned to 99 exhaustive node-owned sections. Graph construction rejects a
  missing, duplicate, unknown, or unsectioned control; tests also require every
  computational setting to be either visible or explicitly documented as a
  private/compatibility field. The audit exposed three previously fixed
  procedural controls (reference-surface occupancy weight and sparse/packed
  marker-area priors), corrected misleading control descriptions, added
  calibration/wavelet edit-time validation. See `docs/NODE_CONTROL_AUDIT.md`.
- **Project is the sole root of every per-image graph.** It replaces the former
  Raw images and Manual annotations pseudo-roots and exposes one typed **Raw
  image** plus one typed **Annotations** bundle. The annotation bundle retains
  separate Background, Foreground, Other, annotated-instance, and manual-centre
  layers internally. The right pane shows the current master path, saved or
  modified state, available/unresolved image counts, selected image and
  dimensions, species, and applied annotation counts. Metadata now consumes
  the Project image context, so every active and restored toolbox node has
  Project in its ancestry. Settings schema version 9 migrates versions 1--8 by
  folding the two retired roots and their wires into Project.
- **Seed-boundary confirmation** is now an active, proposal-independent edge
  diagnostic rather than a consumer of provisional/annotated instance shapes.
  Curved trace pixels perform a reverse centre transform over seed-scaled
  normal radii and bounded tangent offsets (required because a true ellipse
  centre is not generally on each boundary normal). Local vote maxima gather
  compatible ridge coordinates; CUDA moment reduction estimates orientation,
  then a compact radius/axis-ratio bank is scored against 16 perimeter sectors
  for ridge coverage and tangent agreement. Implausible sizes/axis ratios and
  duplicate centres are rejected before output. **Most likely edge ovals**
  draws the retained compact geometry, while **Oval-derived seed-centre
  probability** rasterizes only retained oval scores and applies the configured
  seed-relative centre blur. Raw centre votes remain a separate pre-validation
  diagnostic. The full raster remains lazy/on-device and compact oval geometry
  transfers only when its vector overlay is selected. Candidate-instance
  changes invalidate only candidate consumers and cannot alter these fits.
- The overlay selector is a node-first menu whose headings exactly match the
  active pipeline cards and inspector titles. It has no checkable children,
  synthetic Viewer/None group, or Qt mnemonic ampersands. Every selectable
  overlay is mirrored by an identically labelled output connector on its owner
  node, including dynamically generated colour/pattern overlays; a Qt contract
  test selects every owner and checks all three surfaces.
- The Project annotation bundle combines the former Reference layers and Manual
  seed-centre inputs while preserving each evidence type and its image-local
  persistence.
- **Material colour probabilities** now combines the foreground and background
  colour cards and controls without combining their calculations. Foreground,
  Background, and Other remain independent evidence rasters with their prior
  equations and cached intermediates. **Use background colour analysis** turns
  off only Background/Other fitting; mandatory-reference Foreground fitting
  remains active. Version-4 settings migrate both former node records and
  connections into this combined node.
- **Material noise probabilities** now combines the former foreground- and
  background-noise cards without combining their calculations. All 20 numerical
  controls, explicit independent Background/Other and Foreground enable switches,
  the Foreground/Background/Other noise overlays, distinct maximum versus first-
  tertile defaults, CUDA cache products, and internal timings are retained.
  Changing the combined card invalidates both internal products; version-5 and
  older profiles merge their former node records, enabled states, and wires into
  this owner.
- Node-editor wires now receive stable deterministic per-wire colours from a
  shared source-node palette. Unobstructed bundled cable trunks and branches use
  smooth cubic curves; obstacle-avoiding routes retain rounded corners.
- Ruler evidence rendering extends imperial display ticks to the exterior edge
  without changing detected coordinates or scale calculations. The summary
  text has a dark translucent contrast plate so it remains readable on pale
  images.
- The **File** menu now separates portable analysis settings from complete
  project masters. `.seedfiddle-settings.json` profiles contain every graph
  parameter, enabled/active/toolbox state, and authored connection state, but
  deliberately exclude node layout, images, species, annotations, manual
  centres, statuses, and caches. `.seedfiddle-project.json` masters embed that
  profile plus ordered source-image records and fingerprints, species/current
  selection, node positions, selected node/overlay, and the two cable-display
  flags. They reference rather than embed the full-resolution source-bound
  reference-region and manual-centre sidecars. Project shortcuts are `Ctrl+N`
  New, `Ctrl+Shift+O` Open, `Ctrl+S` Save, and `Ctrl+Shift+S` Save As; `Ctrl+O`
  remains Open images. Recent masters use QSettings. Open validates the full
  schema and graph compatibility before mutation, clears every prior per-image
  draft/cache/history, and restores available images without leaking unlisted
  deterministic sidecars. Missing/changed source records and unresolved listed
  sidecar references remain byte-for-byte in a degraded master on Save.
  Hash-changed sidecars load only through their explicit manifest path and own
  source binding; corrected-coordinate masks wait for current calibrated shape
  validation. A successful user edit switches that image to its canonical
  autosaved sidecar. Version-one masters deliberately reference these canonical
  mutable per-image sidecars rather than snapshotting or versioning them; two
  masters pointing at one sidecar share its current annotation content, while
  the recorded SHA-256 detects changes on the next open. Dirty project
  replacement/close and unapplied drafts use
  Save/Discard/Cancel; autosave failures remain dirty and are retried before a
  master can be saved. Project/profile mutation is disabled during background
  analysis, training, or fitting.
- Colour-card/swatch detection, colour balance, projective deskew, ruler
  detection, 5 cm scale overlay, absolute scale, and dual Petri-dish rims.
  Ruler assignment now treats tick detection as a separate evidence stage:
  narrow high-contrast bands along both ruler edges are evaluated independently,
  then terminally bounded, regularly spaced tick families are fitted without
  averaging the competing reverse scale, barcode, text, and plastic outline
  into one profile. A separate **Ruler detection evidence** overlay draws the
  likely outline green, metric ticks red, imperial ticks orange, metric-number
  components dark red, and imperial-number components dark orange. Absolute
  scale is accepted only from a reliable metric tick pitch; endpoints and an
  imperial competitor never silently establish authoritative pixels/mm. Tick
  transverse lengths are measured before semantic assignment and clustered
  into minor/intermediate/unit classes. The measured hierarchy can shift a
  preliminary lattice phase (fixing the former one-sixteenth `IMG_9533`
  imperial error), recurring long ticks define 10 mm and 1 inch dividers, and nearby
  printed glyph groups are associated with inferred number/unit labels. Metric
  and imperial tick families produce separate pixels/mm estimates; their
  symmetric percentage disagreement is shown as a sanity check and discounts
  confidence rather than being averaged away. Each scale also reports a length-
  hierarchy consistency score and is rejected unless the longest measured
  ticks correspond to its major increments. Tick assignment now
  samples only the compact ruler strip at full source resolution, follows every
  local maximum across the complete 151-dash metric and 97-dash imperial
  lattices, and pools the printed length hierarchy so glare-weakened marks are
  mapped instead of dropped. The second family must occupy the opposite edge.
  An expanded strip fits the transparent-plastic top and bottom edges outward
  from their respective tick roots, then independently fits left and right ends
  outside the terminal ticks. Their intersections form a mildly projective
  quadrilateral, so residual skew is not forced into a rectangle and the
  neighbouring colour-card border cannot become the ruler outline. See
  `docs/RULER_EVIDENCE_DETECTION.md`. The Reference seed
  scale overlay draws a dark-haloed yellow maximum-width fit around every actual
  selected reference component and places its bold diameter text immediately
  above it on an opaque dark contrast plate. Local foreground-core refinement prevents attached shadows from
  inflating this initial estimate. Applied annotated instances override it when
  at least two complete masks exist: the configurable largest fraction defaults
  to 25%, image-cutoff/disconnected masks are excluded, and the overlay draws
  every measured chord plus a histogram and final mean.
- Petri-dish search uses the ruler-calibrated, editable expected 96 mm outer
  diameter as a soft tie-breaker and fallback. A coherent full-circle visual
  rim pair overrides an inconsistent ruler scale, preventing the oversized
  failures seen on dense `IMG_9533.JPG` and sparse `IMG_9546.JPG` while still
  using physical scale when visual evidence is ambiguous. A bounded rotated-
  ellipse refinement retains mild camera-view ovality (up to a 1.15 axis ratio);
  circular downstream consumers use the larger semi-axis so they cannot clip
  seeds near the shorter projected rim.
- Native Qt node graph with inline Blueprint-style controls, dependency-aware
  caching, node timings, progress colours, purple adjacent-node highlighting,
  manual overlap, automatic non-overlapping arrangement, zoom controls, and
  node-driven viewer overlays/intermediates. Every authored datum has its own
  succinct labelled and typed input/output socket; all active and toolbox
  connections use explicit endpoints. Users can drag to restore supported
  connections, drag a connected input into empty space to disconnect it, or
  right-click an edge. Plain left-drag over a connection pans instead of
  selecting its large path bounds; Ctrl-click explicitly selects an edge for
  Delete/Backspace. Right-clicking an enabled implemented node offers **Run to
  node**, which executes only that node and its enabled graph ancestors while
  retaining node-local caches. Two independent default-off toolbar options simplify dense
  wiring without changing the DAG: **Bundle cables** gives compatible wires from
  one source a shared initial trunk before they branch toward their destinations,
  while **Route around nodes** uses rounded detours (and tighter necessary bends)
  to keep connections out of intervening node cards. Disconnection bypasses enabled consumers
  and their dependents; reconnection restores only the cards automatically
  suspended by that missing input, leaving deliberately disabled experimental
  nodes off. The **Project** input exposes a typed image-local annotation bundle
  to every calculation that consumes authored evidence.
  Physical and non-physical edge training
  masks are regenerated from the annotated IDs and are not separately painted
  graph inputs. The one-line toolbar includes an
  unused-node toolbox; **Circle candidates**, **Distance-peak candidates**,
  **Calibration residual risk**, the
  directional surface-darkness gradient branch and its lightening/darkening
  derivative upper cutoffs, and every distance-dependent node are preserved
  there with their authored connections but excluded from the default DAG and
  calculations.
- The complete 53-node active/toolbox catalogue has an independent direct-input
  contract test. Corrected-image, dish-region, absolute-scale, seed-diameter,
  mask, proposal, and reference dependencies are explicit wherever the runtime
  reads them. In particular, **Edge gradients** consumes **Layout detection**'s
  dish region and **Oriented edge traces** consumes **Seed scale estimate**;
  scale-only changes rebuild traces while retaining cached ridges. Oriented
  traces expose their selected source in the node and overlay legend and can use
  generic ridges, the authoritative supported reference ridge, the optional
  conservative net-reference ridge, or the normalized supported reference ridge. Trace-gap
  bridges additionally require tight tangent alignment, the same directed
  gradient side, and a facing endpoint, preventing the former gap-4 collapse of
  adjacent parallel seed rims into one component. An explicit `off | prefer |
  require` convexity policy now rejects S-shaped candidate bridges using an
  editable angular ambiguity tolerance while accepting either C-shaped
  winding. It is intentionally a local initial-link constraint rather than a
  false claim of globally proving arbitrary trace components convex; policy or
  tolerance changes rebuild only trace state and dependents. Legacy wires
  to unused illumination, tangent, reflectance, radial, and image-quality data
  were removed, and the U-Net/StarDist lighting input is represented as separate
  flattened-grayscale, shadow, and highlight channels.
- Overlay selection and opacity are compact top-toolbar controls. The main
  selector groups indented layers under disabled owning-node headings, and a
  synchronized selector directly below the selected node title lists only that
  node's overlays. Material-reference painting and seed-instance annotation are
  top-toolbar modes that reveal one contextual control
  panel over the image; the right inspector contains only image metadata, the
  selected `Node:` controls, its local overlay selector and explanation, and
  applicable summaries.
  The image-pane toolbar always shows the current filename, and opening an
  image from any entry point selects and scrolls to the matching row in the
  left image list. The redundant **Add images…** button below the list was
  removed; toolbar/File-menu **Open images…** remains the single multi-file
  add-and-open command. Below the list, an always-visible **Analysis activity**
  card names the executing node, total/node elapsed time, queued reruns, long
  intervals without a node boundary, and cooperative cancellation of a
  superseded revision. A scrollable **References and annotations** inventory
  shows the exact canonical or project-bound sidecar, discovery/load/withheld/
  pending-calibration state, saved corrected dimensions, material pixel counts,
  annotated seed ID/pixel counts, unapplied draft state, and any bundled
  pre-annotation association for the current image.
- A separate **Annotate seed instances** mode records distinct-colour,
  full-resolution integer seed IDs. It provides a freehand brush, magnetic edge
  tracing, adaptive smart fill, shape-guided fill, and an eraser. Assisted tools
  show debounced live previews and commit on click: edge trace uses click anchors
  and a magnetic path preview, while both fill tools preview their regions even
  when the active seed has no prior marks. A **Show selected seed
  only** checkbox filters the display to one cropped ID and changing the ID
  recentres without altering zoom; invisible neighbouring labels are protected
  from paint and erase. The display is a single viewport-aware graphics item
  that requests exact 512-pixel source tiles as Qt exposes them; it never scales
  the instance labels through a whole-image display proxy. Tile-local indexed
  palettes preserve one categorical mask pixel per image pixel, and a bounded
  512-entry/48 MiB LRU cache keeps pan, zoom, edits, and selected-only filtering
  responsive without unbounded image-size-dependent storage. Trace edge, Smart
  fill, and Shape fill independently select thinned
  ridges (default), oriented traces, combined ridge-priority evidence, physical-edge
  probability, or broad edge magnitude. Smart fill and Shape fill additionally
  offer net physical-edge evidence, using the live display-only physical-minus-
  scaled-non-physical coefficient owned by the instance-derived edge node. Trace uses a banded continuity-aware
  live-wire seam, local optional directed/undirected tangents, and anchored
  component support so it cannot switch to a stronger parallel boundary. Smart
  fill uses OpenCV's native floating-range neighbour comparison, preserves other
  IDs, reaches the chosen one-pixel edge frontier, and exposes bounded
  tunnelling/tolerance/distance options for patterned seeds. **Maximum distance
  from cursor** defaults to 0.60 estimated seed diameter and is a true radial
  click-to-pixel limit (not an instance width); the separate maximum-added-pixel
  control remains an independent area safety cap. **Fall-off half-life**
  defaults to 0.40 seed diameter and applies exact pressure
  `2^(-distance/half-life)` to both the allowed inward-neighbour Lab-step limit
  and continuous edge threshold, respecting selected 4/8 connectivity;
  this strengthens stopping evidence with distance without creating fractional
  labels or blocking uniform/no-edge regions. Tunnelling carves one
  local weak-edge passage rather than globally weakening the ROI. A leaked flood
  that reaches its cursor-distance limit is replaced by a smooth local star-convex edge
  contour. Freehand drags use a lightweight vector stroke preview and
  rebuild the annotation raster once on release; they never rebuild the
  analysis overlays. Trace commits are exact one-full-resolution-pixel
  categorical lines, independent of the brush radius and without antialias
  expansion. The ordered trace session retains its cyan first anchor; returning
  to it previews and fills only the closed interior by default, while protecting
  labels belonging to other seeds. The combined evidence is the fixed maximum
  of generic thinned ridges and binary oriented traces plus physical-edge
  probability and broad edge magnitude at 45% strength; it is not a separately
  learned model. Assisted algorithms crop edge work to their local cursor
  region instead of rebuilding or normalizing full-resolution rasters on every
  event.
  Shape fill automatically fits a rotated ellipse prior, but the prior is only
  an approximate maximum extent. Its contour search is unrestricted inward for
  partially occluded seeds. Outward displacement has exponential pressure with
  a default 0.05-seed-diameter half-life and a hard 0.10-diameter cutoff. It uses
  fixed 8-neighbour growth internally and deliberately exposes no tangent,
  tunnelling, cursor-radius, radial-falloff, or connectivity controls. The old
  plain Snap shape tool has been removed.
  Draft/apply/revert state is per image and independent of the
  binary colour-reference masks; applied labels become authoritative markers
  for the active procedural watershed and also constrain the dormant
  provisional instance branch without overriding foreground probability.
  **New seed** advances to the next available ID without changing the active
  Brush/Trace/Shape-fill/Smart-fill/Eraser tool, its option page, or its
  configured values.
  A **Start from result** selector expands any available procedural,
  U-Net/watershed, or StarDist labels into a full corrected-image editable
  draft. The initializing method is preserved in export notes; the draft stays
  explicitly unreviewed and requires correction of every automated error.
- GPU foreground/background colour probabilities, symmetric three-band noise
  profiles with directional texture continuation, shared edge gradients,
  directed/undirected tangents, ridge thinning, and oriented edge traces.
  Foreground and background colour scores are independent evidence models and
  are deliberately not forced to sum to one: a pale seed may validly resemble
  both, and Other/unmodelled evidence must remain possible. Painted foreground
  and safely inset annotated-seed prototypes are fused as positive evidence by
  probabilistic union, so a near-zero automatic branch can no longer veto a
  strong reference match. The shared multifeature material-prototype node is a
  different model: its available Background/Foreground/Other class scores are
  jointly normalized with a fixed unknown mass, so their sum is at most one.
  Background texture positives are sampled directly from the exterior annulus
  in its own coordinate frame; they no longer disappear when projected into the
  smaller dish crop. The nominal texture/colour blend is reduced continuously
  when fitted class profiles have weak separation, preventing texture from
  overwhelming clearer colour evidence. Directional rays are integrated on
  CUDA into each class noise probability but
  are no longer materialized as individual viewer overlays. Foreground noise
  defaults to **1st tertile**, the exact linearly interpolated one-third
  quantile across ray directions; this is more conservative than median while
  avoiding minimum's all-direction requirement. Its CUDA reducer interpolates
  two `kthvalue` order statistics rather than allocating a sorted full-raster
  direction bank. Background noise retains its maximum default. The combined
  Material colour/noise nodes also expose direct-bright **Other colour
  probability** and **Other noise probability** overlays when Other examples
  have been painted.
  The colour view is raw target-model Lab membership; the noise view is raw
  target-only texture compatibility and receives no semantic counterclass or
  colour blend. These diagnostics remain blank without Other paint and are
  separate from the multifeature Reference Other-material prototype
  probability. Each node also exposes a viewer-only **Foreground vs
  Background/Other excess** diagnostic. At every pixel the stronger of
  Background and Other is compared with Foreground; positive Foreground excess
  is blue, positive Background/Other excess is red, and ties are black. The
  sibling Non-seed scores use a maximum rather than an addition because the raw
  evidence maps are independent and may legitimately overlap. These views do
  not enter material decisions, node caches, or downstream invalidation.
- **Hue only** displays corrected hue at fixed neutral brightness/chroma, with
  achromatic pixels shown neutral gray. **Wavelet decomposition** is a four-
  level stationary B3-spline à trous decomposition with selectable detail
  layers and a residual; all layers stay at full image resolution and the four
  details plus residual reconstruct the original corrected RGB tensor exactly.
- The **Project annotations** bundle contains the mutually
  exclusive Background/Foreground/Other material layer, integer annotated seed
  instances, and source-coordinate manual centre points. Painting a
  class clears the other two at that pixel; Other supplies a competing learned
  distribution to both material models and attenuates a class only where it fits
  better, preserving colours shared with legitimate positive evidence. It feeds
  both colour models and both class-specific noise
  models, so applying a painted edit invalidates every true graph dependent.
  Each noise classifier learns its positive and negative texture distributions
  directly from the applicable painted areas when present, using colour
  pseudo-labels only for an unpainted class. Painted coordinates are not forced
  to exact colour- or noise-probability zero or one. The material panel's
  **Include annotated seeds as Foreground** option defaults on. When enabled,
  it adds only safely inset interiors from applied seed IDs to the Foreground
  colour, directional-noise, and material-prototype sources; contours and their
  uncertainty band remain excluded. Painted Background, Other, and exclusion
  evidence takes precedence, manual Foreground remains additive, and unapplied
  drafts or display-only visibility changes have no analytical effect. There
  is no separately painted boundary layer: every complete annotated instance
  contributes its one-pixel contour as physical-edge supervision, while only strong edge/ridge
  candidates safely inset from that contour contribute sparse non-physical-edge
  supervision. Flat interior and the contour uncertainty band remain unlabelled.
  These derived examples fit image-local Physical and Non-physical tangent-strip
  prototype banks from corrected Lab context and polarity-neutral edge geometry.
  Their spatially broad outputs are explicitly labelled **prototype
  compatibility** and retained only as diagnostics: the two-channel view uses
  magenta for overlap, the excess view shows the unscaled signed margin, and
  **Net physical-edge prototype compatibility** shows
  `max(Pphysical - k × Pnonphysical, 0)` with a black floor.
  The Reference edges node's **Non-physical subtraction weight** defaults to
  0.5 and supports 0--2 without modifying either source compatibility raster.
  The authoritative **Reference-edge probability** is
  `thinned_true_edge_support × Pphysical` at full
  resolution. Descriptor/restoration halos are therefore exactly zero away
  from a real ridge. Assisted fill, curve confirmation, learned-instance
  inputs, and procedural separation receive only this supported field, its
  normalized/thinned derivatives, or separately edge-supported Physical and
  Non-physical channels; raw compatibility never serves as a barrier.
  The previous supported subtraction is retained separately as **Conservative
  net physical-edge evidence**, with explicit Smart/Shape-fill and thinned-ridge
  choices. Only this conservative branch and the raw Net compatibility diagnostic
  depend on the subtraction weight.
  Local normalization acts on the thinned true-edge support before reapplying
  Physical probability, estimates a seed-scale winsorized local RMS envelope,
  and applies a bounded one-sided gain that can enhance but does not attenuate
  above-floor supported evidence. The exact full-resolution ridge mask is
  reapplied after interpolation. The per-pixel gain denominator is capped by
  current support so a neighbouring strong arc cannot suppress its weak
  continuation. A smooth absolute gate keeps zero-ridge regions at zero, and
  ridge hysteresis admits enhanced weak maxima only when they connect to a
  strong seed. Its default
  radius, target support, maximum gain, and floor are
  respectively 0.30 seed diameter, 0.35, 2.5x, and 0.04. The full rationale,
  equations, safeguards, and validation plan are in
  `docs/LOCAL_EDGE_NORMALIZATION.md`. The node owns those four controls plus
  independent NMS step, low/high threshold, hysteresis-reach, and working-size
  controls. Raw compatibility diagnostics remain viewable but are not
  operational edge sources.
  The formerly black continuous normalized overlay was a
  display-quantization bug: its `[0, 1]` raster is now multiplied to display
  range before conversion to `uint8`; its derivative overlays were already
  generated through a separate correctly scaled path. When no annotated seed
  exists, the semantic classifier is neutral instead of treating every generic
  image edge as a physical reference. Learning export derives both boundary
  targets and their sparse validity raster from the instance labels.
- Active **Procedural seed separation** combines seed-material evidence,
  generic edge/ridge candidates, supported instance-derived semantic evidence,
  normalized reference-edge ridges, and coherent convex oriented traces into
  its physical-boundary cost. Edge-supported class channels and the normalized
  authoritative field gate/discount the candidate without admitting broad
  descriptor halos; the gate is neutral with no annotations. Centre markers
  come from smoothed material geometry and seed-interior depth, not sensor/noise,
  shadow, or internal-boundary peaks. Each centre is evaluated at several
  material thresholds (default five). Hard minimum area, maximum width,
  concavity, thin-protrusion, solidity, axis-ratio, and maximum-area checks
  reject impossible candidates, including candidates originating from
  annotation markers. Complete annotations above the hard minimum are retained
  as exact authored shapes instead of being expanded by watershed; tiny legacy
  centre marks still seed a generated candidate that must pass the same hard
  geometry checks. Soft minimum-area and maximum-width bands
  lower scores before their hard cutoffs. A score-ordered overlap-aware pass
  selects the final non-overlapping combination and calculates explicit
  per-instance confidence. Its eight overlays expose material likelihood/mask,
  boundary cost, centre likelihood, instance identities, confidence, convex-hull
  concavity, and unselected alternatives coloured by relative score. Clicking
  an instance in the main result selects it, outlines it in yellow, selects the
  procedural node, and reports its area, maximum side-to-side width and width
  fraction, concavity, protrusion, solidity, axis ratio, confidence, and source
  in the inspector. GPU rasters are resized before the one
  essential bounded CPU topology transfer; CPU labels and evidence stay at the
  bounded working size and Qt scales them only for display. The compact result
  is node-cached, and
  distinct painted instance IDs suppress nearby automatic markers.
  The **Project annotations** graph input and its inspector editor allow
  per-image placement and adjustment of the actual watershed markers rather
  than the final region centroids. Automatic markers are hollow cyan, manual
  or replacement markers solid yellow, annotation-authoritative markers locked
  magenta, and rejected manual points red with their reason. Click adds, drag
  moves, and right-click/Delete removes; Escape cancels the active drag or exits
  editing. **Augment automatic** retains discovery while suppressing nearby
  duplicates. **Replace automatic** uses the editable list as the complete
  automatic-marker set but never removes annotation markers. Switching to
  Replace seeds the list from every currently surviving non-annotation marker;
  dragging or deleting an automatic marker makes the same conversion so its
  former maximum cannot silently reappear. Image-local Undo and Reset are
  available. Completed gestures alone autosave a compact SHA/dimension-bound
  source-coordinate sidecar under `projects/manual-seed-centres/`; the fresh
  calibration transform maps exact source floats to corrected and crop-local
  coordinates on every run. The per-image input invalidates only procedural
  inference and its dependents, and editing stops while analysis runs or when
  the image/node/result changes.
  Its inspector also offers a bounded annotation-guided parameter fit. Trial
  segmentations explicitly receive no annotated watershed markers, and are
  compared with the applied masks using deterministic one-to-one overlap
  matching. False-positive pixels use a bounded exponential distance cost:
  their cost is zero on the target, small immediately outside it, reaches the
  editable 2.0 amplitude at the default 0.50-seed-diameter scale, and rises
  exponentially farther away. Matched results measure distance from their own
  annotated target (not the annotation union), while missed seed area remains
  1.0 and unmatched instances retain a small count term. Raw FP counts remain
  available alongside distance-weighted FP equivalents. **Annotations cover
  whole dish** defaults off, so partial-review
  metrics explicitly ignore wholly disjoint predictions. When checked, every
  predicted object is evaluated, including standalone background false objects.
  Search results are proposals only; accepting one uses a single atomic graph
  update and invalidates the procedural node and dependents once. The fit is
  image-local and in-sample, but accepted settings are global to every image;
  neither mode is held-out validation.
- **Layout detection** now also owns the perimeter Background reference instead
  of exposing a redundant separate node. Its ruler-calibrated outer-rim buffer
  defaults to 0.35 cm and its independently
  adjustable median-colour band thickness defaults to 0.5 cm; both are shown
  exactly in a dedicated overlay, with a nominal-dish scale fallback. The
  overlay includes an opacity-independent swatch and hex label for the selected
  median starting background colour. The Material colour probabilities node's
  Background colour probability overlay
  also evaluates the exact fitted positive Background Lab model across
  this bounded GPU annulus and displays it beside the unchanged dish crop; its
  annulus is outline-only so the probability values remain legible. Background
  painting now defaults **Keep perimeter-matched source** on, combining only the
  colour-filtered ring pixels and their Lab samples with manual Background
  marks. Similar-coloured pixels inside the dish are evaluated by the resulting
  model but are not recruited as references. In the painting view, only the
  retained ring is cyan while manual marks stay green. Unchecking the control
  removes the complete perimeter-derived source;
  without paint, the colour model falls back to its generic light/low-chroma
  automatic selection.
- **Hierarchical material evidence** is the sole authoritative Seed-versus-
  Non-seed decision. Raw Foreground, Background, Other, directional texture,
  and valid multiclass prototype maps remain independent and need not sum to
  one. The node publishes resolved Seed, resolved Non-seed, conflict/ambiguity,
  and unknown masses that sum to one, plus a second Background/Other decision
  with explicit subtype ambiguity and unknown. Other and Background are
  parallel positive Non-seed classes; neither suppresses the other. Automatic
  Foreground/perimeter authority decays exponentially with reviewed coverage.
  Every raw source also earns reliability from its target median versus the
  90th-percentile reviewed top-level non-target response; Background and Other
  calibrate only against Foreground, never one another. Reliability scales
  contribution without changing the displayed raw map. The audited binary
  Seed threshold is 0.68. Material prototypes publish probabilities only with
  at least two material banks; physical/non-physical edge probabilities require
  both edge banks. Procedural occupancy, restored distance candidates,
  provisional assignments, seed-interior smoothing, and curve semantic sides
  now consume the resolved decision. See
  `docs/MATERIAL_EVIDENCE_REDESIGN.md` and
  `docs/EVIDENCE_AND_ANALYSIS_FORENSIC_AUDIT.md`; the read-only saved-sidecar
  diagnostic is `tools/audit_material_evidence.py`.
- A six-output active multiscale node supplies fine/medium/coarse surrounding RMS
  darkness and Lab-colour noise energy. The maximum one-sided lightening and
  darkening CIE L* surface slopes, their query-to-target directions, and the two
  dependent derivative upper cutoffs are disabled and preserved in the
  unused-node toolbox. These raw masks are intentionally distinct from the
  learned foreground/background noise profiles.
- Two annotation layers: categorical material references
  (Background/Foreground/Other) and labelled seed instances. The material layer
  enforces exclusivity while it is painted and normalizes older mask state when
  it is read. Other samples fit independent positive colour-frequency and
  texture distributions. They are no longer subtracted from either Foreground
  or Background, in the raster calculation or colour-profile inspector. The
  compact editor uses class selectors plus shared Paint/Eraser/Clear-layer,
  Apply/Revert, brush-radius, and independent Materials/Seed-instances
  display-only visibility controls; longer
  explanations live in tooltips. Its drag handle repositions the panel, and
  clearing a class restores Paint mode so that it can immediately be redrawn.
  A context-aware **Undo** button and `Ctrl+Z` retain 20 unapplied commands per
  image in separate reference and seed-instance histories. Each drag, assisted
  click, clear, imported label map, or calculated starting draft is atomic;
  categorical peers and instance provenance are restored together. Histories
  use compressed changed tiles rather than full-resolution snapshots and are
  rebased by Apply, Revert, or validated disk restoration.
  Seed-instance IDs are summarized in one streaming row-run scan with exact
  categorical 8-connectivity. The editor warns, without blocking Apply, when
  the selected ID or any other ID occupies disconnected areas; the compact
  summary is cached by immutable/copy-on-write label-array identity.
  Smart fill calls its floating-range Lab control **Neighbour colour step**:
  candidates are compared with accepted touching pixels (L at the displayed
  step, a*/b* at 72%), not with the initial click. The independent
  **Click-origin colour range** compares every newly added pixel directly with
  the clicked pixel on the same channel scale, preventing accumulated gradual
  drift. Its conservative default is 72; it is a hard range independent of
  radial fall-off, while existing pixels of the active instance remain
  preserved. **Edge barrier threshold**
  is the minimum selected edge strength that stops growth; lower values stop
  on weaker evidence. Tooltips also document the one local tunnelling passage.
  The additional **Shape fill** tool performs a bounded centre/scale/axis-ratio/
  axial-rotation search, treats the fitted ellipse only as an approximate outer
  prior, and optimizes an independent cyclic contour using edge, Lab-contrast,
  and interior-core evidence. It
  previews prior versus refined boundary and refuses low-coverage, long-gap,
  overlapping, oversized, or irrelevant evidence instead of stamping an oval.
  While Shape fill is active, each mouse-wheel notch changes its visible oval
  size preference by 5% and does not zoom the image. The refined contour
  validates the fitted evidence but is not a hard mask around the canonical
  Smart-fill result. The shape prior imposes no inward restriction for occlusion,
  while outward
  extension decays with a 0.05-diameter half-life and stops at a hard
  0.10-diameter cutoff. Shape fill reuses Smart fill's neighbour Lab step, edge
  barrier, maximum-pixel cap, and frontier inclusion with fixed 8-connectivity,
  but not its tunnelling, cursor-distance, or radial-falloff controls.
  The Lab probability models now have node-owned, full-image-pane HSV
  diagnostics for both colour classes. Hue and saturation are the visible axes;
  a toolbar Value slider scans exact brightness slices, starts at the dominant
  visible fitted mode, and has a **Peak** reset. Achromatic membership is shown
  once in a neutral swatch rather than projected across every hue. The
  underlying colours are not dimmed. The contextual editor keeps natural row
  heights and scrolls vertically when the split image viewport is short.
  **Apply + save** for either a reference draft or a seed-instance draft writes
  the material raster and integer seed IDs/provenance to one atomic, version-2
  categorical NPZ
  under ignored `projects/reference-regions/`; unapplied strokes are never
  persisted. The explicit **Save applied reference regions** command remains
  available. An automatic-save failure warns without rolling back the valid
  in-memory applied state. First opening an image in a window restores the
  immutable applied snapshot only after its stored SHA-256 and dimensions match
  the current source; mismatch/corruption warns and installs nothing. Switching
  away and back preserves newer in-memory applied state. When no analysis cache
  exists for the returning image, the viewer now renders the authoritative
  material and seed-instance layers directly over the source instead of clearing
  them while it waits for a later analysis. An empty snapshot can intentionally
  replace an older nonempty archive. Version-1 archives are
  still validated and loaded, but their retired manually painted boundary
  raster is intentionally ignored rather than restored or migrated into
  semantic training evidence.
  **Load reference mask…** first looks for the current photograph in the
  version-1 `seed-instance-references/manifest.json` bundle. Bundled `uint16`
  masks stay in immutable source-image coordinates and are accepted only after
  their safe relative paths, source and mask SHA-256 digests, source dimensions,
  declared ID count, and categorical values validate. The current calibration
  homography maps them into corrected coordinates with nearest-neighbour
  interpolation, and loading is refused if any nonzero ID disappears. A valid
  mask becomes a replacement-confirmed, undoable, unapplied draft; manual
  corrected-coordinate PNG/TIFF/NPZ import remains available through the
  explicit **Choose mask file…** action even when a matching bundle exists. The
  committed bundle covers all eleven image fixtures and includes the two isolated
  reference seeds outside each dish. Every entry is explicitly
  `reviewed: false`: these machine-prepared masks are editing bootstraps, not
  accepted training or evaluation ground truth, until every contour is reviewed
  at full resolution and applied.
- **Reference texture prototypes** is an active CUDA-first, image-local
  procedural classifier fed by the corrected image, detected dish region, seed scale,
  material references and annotated instance IDs, shared gradient/ridge/tangent fields, and all six raw
  frequency-noise masks. Independent controls retain up to 64 robust coverage
  medoids per material class by default (Background, Foreground, and Other) and
  256 per instance-derived edge class (Physical edge and Non-physical edge).
  The edge capacity supports up to 1,024; GPU fitting assignments and full-raster
  evaluation are chunked to bound temporary memory independently of that limit.
  Material features combine Lab, multiscale energy, local residual, edge, ridge,
  and density evidence. Edge features use narrow tangent-aligned strips with
  explicit interior, centre-edge, and exterior pools, signed cross-edge Lab
  contrast, local residuals, edge/ridge support, and axial tangent coherence.
  Reviewed instance geometry supplies the outward normal only while fitting;
  inference uses image tangents and tests both polarities, avoiding annotation
  leakage. Material matching defaults to a 960-pixel working limit, while edge
  matching adaptively targets 28 pixels per seed diameter up to an independent
  2,048-pixel hard cap. It produces Foreground/seed-surface, Background, Other,
  Physical-edge, and Non-physical-edge likelihoods without hard-writing reviewed
  pixels. Gaussian-kernel matches are calibrated in two stages: the unsharpened
  best match determines known-versus-unknown confidence, while separate exposed
  material and edge class contrasts sharpen only relative Background/
  Foreground/Other or Physical/Non-physical competition.
  This avoids the former middling ceiling for ordinary in-class descriptors
  while preserving weak-match unknown mass and exact subtype ties. The
  calibration receives no masks or reference coordinates, and reviewed pixels
  are never hard-written. Every
  probability/likelihood overlay now states its low and high display colours in
  the image legend (direct maps are black-to-bright, while inverse background
  evidence is explicitly labelled white-low/black-high).
  The seed-surface result has adjustable inputs to the active procedural
  watershed and the dormant seed-interior branch. The image pane can replace
  the photograph with a scrollable, zoomable collage of every medoid, grouped
  by class with source counts and support percentages. The displayed squares
  are context only; tangent-aligned edge thumbnails mark the centre sample in
  yellow and two polarity-neutral side strips in cyan. When no annotated
  instance exists, physical/non-physical
  semantic output is neutral; unknown regions retain neutral generic edge/ridge
  support in the procedural separator but do not receive a physical label.
  Complete applied instance IDs always derive a one-pixel Physical-edge contour.
  Strong edge/ridge candidates safely inset beyond the configurable buffer
  become sparse Non-physical-edge examples; flat interior and the uncertainty
  band remain unlabelled. These derived masks are regenerated from the persisted
  IDs and current evidence rather than stored separately.
- Full-image CUDA jobs use one dedicated worker and coalesce newer requests.
  A newer request cooperatively cancels the superseded analysis at the next
  timed node boundary instead of invisibly waiting for the obsolete revision to
  finish; the current long-running node remains visibly identified while it
  reaches that safe boundary. Worker cancellation and every worker exception
  have explicit Qt signals, and completed-result installation has its own GUI
  error boundary so a post-worker display exception cannot leave an orphaned
  busy/idle state.
  Per-image node caches are LRU-bounded to three images and 2 GiB of reachable
  CUDA storage; eviction and failures release GPU ownership and lazy CPU mirrors.
  Overlay downloads survive only until the next viewer render. Painted masks
  use immutable applied arrays; categorical peers are copied lazily on the first
  dab that can replace them, rather than when the panel merely opens. Both reference and instance freehand tools use incremental vector
  previews rather than rebuilding the corrected base image during pointer moves.
  Corrected-coordinate reference archives are fingerprint/schema checked, then
  held as visibly pending until calibration confirms their dimensions. The
  completion transition now adopts the corrected viewer raster before binding
  those masks. This fixes the IMG_9533 failure in which valid 6508×4294 masks
  were installed against the still-raw 6240×4160 scene, raising an uncaught
  Qt-slot shape error after the worker had already disappeared from the active
  task table. Applied archives derive their saved dimensions from their
  categorical rasters/current corrected result rather than whichever raw or
  diagnostic pixmap happens to be displayed.
- Desktop launches configure a per-user rotating diagnostic log at
  `%LOCALAPPDATA%\Seed Fiddle\logs\seed-fiddle.log` on Windows, retaining five
  5 MiB backups, plus a `seed-fiddle-crash.log` faulthandler target. Analysis,
  learning, procedural-fit, edge-fit, and GUI result-installation boundaries
  record full tracebacks before emitting concise Qt failure messages. Uncaught
  main/thread exceptions and Qt warnings/errors are also captured. Failure
  dialogs and **Help → Runtime summary** expose the absolute log paths. Before
  this addition, caught worker/installation exceptions were reduced to one-line
  dialog strings and no persistent application log existed.
- **Grayscale & local lighting** estimates a valid-mask-aware broad field,
  flattens grayscale with a clipped log ratio, and maps standardized local
  deviations through separate nonlinear shadow/highlight sigmoids. All rasters
  remain GPU-resident and selectable as overlays.
- The dormant **Circle candidates** toolbox node has authored inputs from shared
  edge magnitude, sensor/noise, flattened grayscale, shadow, and highlight
  products. Its CUDA ring bank weights their boundary transitions independently,
  while remaining disabled and outside the default DAG. A selected restored
  circle node can be moved back to **Unused nodes** and restored again later.
- Foreground references use a per-pixel Lab colour-frequency table, never a
  regional average or painted-pixel probability override. Automatic background
  colour is anchored to the retained outside-dish annulus distribution, which
  remains additive to painted Background references by default.
- The combined Material colour node exposes separate **Maximum reference colour
  modes** controls. Foreground retains coverage-preserving quantized
  Lab frequency cells, whereas background fits adaptive robust Lab mixture
  components. The background default is 32, its supported range is 1--256, and
  full-image membership is evaluated four modes at a time to keep peak CUDA
  memory bounded. Positive components use the strongest weighted membership,
  not an additive clamp: an 11-fixture check caught the additive implementation
  pathologically saturating seed surfaces when capacity rose from 4 to 32. The
  auxiliary competing Other distribution remains capped at 64 modes so raising
  positive background capacity cannot multiply it into an unexpectedly
  expensive model.
- Painted foreground colour modes are capped with coverage-preserving selection,
  not a top-frequency truncation. The former truncation allowed a larger varied
  reference mask to evict an existing smaller colour mode and collapse its
  membership. The HSV diagnostic now uses an exact full-pane hue/saturation
  slice at the selected Value, keeps visually neutral membership in a separate
  swatch, and labels only the leading modes near the selected Value instead of
  covering the plot with unrelated frequency cells.
- Foreground colour is now mandatory user-supervised evidence. Painted
  Foreground and, when enabled, safely inset applied seed-instance interiors fit
  the coverage-preserving Lab frequency model. Isolated reference-seed fits are
  scale-only. The background-distance/Otsu start, automatic colour source,
  automatic/reviewed viewer split, and self-refinement rounds are retired.
  With no authored Foreground source the colour/noise probabilities are zero and
  the Qt node reports that the reference is missing.
- Foreground, Background, and Other remain independent raw evidence scores, not
  calibrated complements. Class annotation coordinates calibrate source
  reliability and select training samples but never hard-mask, subtract, or
  overwrite any raw raster. Material evidence alone normalizes Seed versus
  Non-seed, Ambiguity, and Unknown; Background and Other are symmetric positive
  Non-seed evidence and their overlap becomes subtype ambiguity rather than a
  special Foreground suppressor.
- Every probability raster now shares the valid crop reaching the outer edge of
  the configured perimeter Background band. Only the binary seed-proposal gate
  remains limited to the interior analysis region.
- The redundant visible **Seed-interior probability** node has been retired.
  Downstream diagnostics consume the authoritative **Resolved Seed
  probability** from Material evidence decision, and the procedural node no
  longer publishes a duplicate Seed-material-likelihood overlay. A private
  smoothed raster may still be reused inside dormant advanced diagnostics, but
  it is not a graph node or selectable overlay.
- The application is branded **Seed Fiddle** and uses a tightly framed rounded
  soybean plus a high-contrast, separately readable fiddle bow under
  `seedvision/assets/seed_vision_icon.png`; a 32-pixel occupancy regression test
  prevents the taskbar mark from shrinking back into excess transparent space.
- The 11 low-resolution test photographs under `images/` are committed and are
  required by image-backed tests. Generated diagnostics under `artifacts/` are
  ignored.

The 2026-08-23 overlay/evidence/gradient cleanup made the following contracts
explicit:

- The visible toolbar overlay selector is a node-first `QMenu`; each node opens
  a right-arrow submenu of only its owned overlays. The old grouped combo is a
  hidden state/accessibility model. Physical/non-physical prototype rasters are
  owned by **Reference texture prototypes**; their combined and net products are
  owned by **Instance-derived edge probabilities**. A large red image-pane
  **Calculating …** banner tracks the selected overlay's actual owner/status.
- The retired **Foreground strength**, automatic Foreground, and reviewed-
  Foreground overlays are not selectable. The old foreground-feature raster is
  zero-valued compatibility data and has no live application-pipeline consumer;
  advanced seed-interior analysis receives resolved Foreground probability.
- Directed and undirected tangent pass-through nodes were merged into **Edge
  gradients**. That node now exposes derivative method, source-fusion rule,
  original/residual/detail-band inclusion, blur, chroma weight, normalization,
  gamma, and wavelet-detail gain. The stationary wavelet node is an authored
  upstream dependency and its signed details plus residual exactly reconstruct
  the corrected crop.
- **Oriented edge traces** names its ridge source and exposes an explicit trace
  diameter multiplier linked to the master seed diameter. Non-generic reference,
  net-reference, and normalized-net ridge sources retain short semantic
  fragments instead of collapsing to black under the generic minimum-length and
  junction filters.
- Procedural concavity display contains only exterior-connected convex-hull
  pockets between a candidate outline and its hull; seed interior and enclosed
  holes are excluded. **Next empty seed** replaces the ambiguous **New seed**
  label.
- Smart fill keeps tunnelling disabled in the UI. After click-origin clipping,
  star-convex recovery, and pixel limiting, newly added pixels are re-filtered to
  the click-connected component while pre-existing active-seed paint is
  preserved. This prevents detached narrow trails from surviving the new total
  click-origin colour range.
- The selected pipeline card now uses a saturated dark-blue border/body. Normal
  completed Background-colour and Procedural nodes use the standard completed
  status colour instead of warning orange.

The full suite passed 444 tests on Python 3.12.10 with PyTorch CUDA on an RTX
3070 on 2026-08-18. The 393 non-pipeline-UI tests passed normally (with one
POSIX-only portability regression skipped on Windows); the 51
pipeline-UI tests passed with the developer machine's stale ignored
reference-region autoload disabled so its modal warning could not block the
offscreen runner. This includes the comprehensive graph-contract correction,
exact perimeter-band background-probability display, SHA-bound reference
persistence, full-canvas soybean/bow icon replacement, unified
32-mode default background-colour capacity with bounded-memory evaluation,
selected-seed-only annotation display, selectable assisted-tool edge evidence,
continuity-aware Trace Edge, leak-recovering Smart Fill, parallel-safe
oriented trace-gap linking, and winding-independent local convexity preference
and requirement modes. It also covers exact first-tertile foreground-noise
direction integration as the foreground default, plus the independently cached,
tool-selectable internal physical-reference ridge and its CUDA NMS/hysteresis controls.
It also locks exact one-pixel open Trace Edge commits and protected inward fills
when an instance contour returns to its first anchor, annotation-derived
Physical-edge/Non-physical-edge evidence, and the non-mutating
asymmetric procedural-parameter fit plus its Qt lifecycle and cache invalidation.
The current redesign retires manual boundary painting and version-2 persistence
stores material references plus instance IDs only; version-1 boundary rasters
are validated but ignored. Complete instances now supply physical contours and
sparse safely inset internal edge candidates automatically, with neutral
semantic support wherever annotation evidence is absent.

The complete suite was run again on 2026-08-22 after adding local net-edge
normalization: 473 tests ran, 470 passed, and 2 platform/fixture-dependent tests
were skipped. The sole failure is the known user-worktree deletion of
`images/IMG_9689c.JPG`; its manifest-integrity test correctly reports that the
referenced fixture is absent. All 189 focused normalization, GPU-layer, graph,
cache, annotation-tool, procedural, settings-persistence, and Qt pipeline-UI
tests passed.

The suite was rerun after the probability/ruler/ellipse/wavelet/procedural/UI
redesign on 2026-08-22: 479 tests ran, 476 passed, 2 platform-dependent tests
were skipped, and the same single manifest-integrity test failed solely because
the still-deleted committed `images/IMG_9689c.JPG` fixture is absent. The 147
focused computational tests and all 55 pipeline-UI integration tests pass; the
settings-persistence suite also passes after accounting for the selectable
reference-ridge input to Oriented edge traces.

The suite was rerun after the analysis-lifecycle/reference-association repair on
2026-08-22: 482 tests ran, 479 passed, and 2 platform-dependent tests were
skipped. The sole failure remains the intentional/user-worktree deletion of
tracked `images/IMG_9689c.JPG`; its manifest-integrity test correctly reports
the missing source fixture. All 97 focused timing, project, persistence, and Qt
pipeline-UI tests pass. A live full-resolution IMG_9533 run also passed while an
initial run was deliberately superseded: cooperative cancellation completed,
the valid 6508×4294 archive installed all 11 IDs, the reference-dependent rerun
finished, and the controller returned to an idle state with no pending bundle.

The suite was rerun after measured tick-hierarchy alignment, mildly projective
ruler outlines, annotation-derived seed scale, and exterior-annulus noise repair
on 2026-08-23: 496 tests ran, 493 passed, and 2 fixture/platform-dependent tests were skipped.
The sole failure is still the user-worktree deletion of tracked
`images/IMG_9689c.JPG`; the reference-manifest integrity test correctly reports
that missing source. All 14 calibration tests and the expanded Qt ruler/seed-
scale overlay integration test pass. The real `IMG_9546.JPG` regression maps
all 151 metric and 97 imperial dashes, infers their measured length classes and
unit dividers, reports 18.000 versus 18.268 px/mm (1.48% disagreement), and
constrains four independently sloped outline sides outside and close to those
terminal rows. `IMG_9533.JPG` additionally locks the one-sixteenth imperial
phase correction, all five semantic length classes, full-inch endpoints, and
right-to-left 6…0 labels. A negative synthetic test shortens all centimetre
ticks and confirms that periodic spacing alone cannot pass the major-increment
sanity gate. The two focused Qt overlay regressions pass after adding a dark
halo to every isolated-reference circle and a dark plate behind its label.
Saved `IMG_9533.JPG` annotations override the 119.1 px refined isolated-
reference estimate with a 178.2 px mean of the selected four largest complete
annotated widths. Its exterior-ring median Background-noise score rises from
122 to 224 while its annotated-seed median falls from 184 to 130. The isolated-
fit GrabCut seed is fixed explicitly so these values and downstream seed counts
are independent of prior OpenCV operations in the process.

The suite was rerun after adding Smart fill's independent click-origin colour
range on 2026-08-23: 497 tests ran, 494 passed, and 2 fixture/platform-dependent
tests were skipped. The sole failure remains the user-worktree deletion of
tracked `images/IMG_9689c.JPG`; its reference-manifest integrity test correctly
reports the missing source. All 30 focused annotation, Shape-fill isolation,
and Qt control/session-state tests pass.

The suite was rerun after unifying annotation edge-source contracts and adding
persistent error diagnostics on 2026-08-23: 498 tests ran, 495 passed, and 2
fixture/platform-dependent tests were skipped. The sole failure is still the
user-worktree deletion of tracked `images/IMG_9689c.JPG`; its reference-manifest
integrity test correctly reports the missing source. The exhaustive Qt source
regression selects every Trace/Smart-fill/Shape-fill value, including the two
formerly mismatched normalized sources, and the rotating-log regression verifies
that a GUI-installation-style exception retains its complete traceback.

The suite was rerun after the overlay/evidence/gradient cleanup on 2026-08-23:
502 tests ran, 499 passed, and 2 platform-dependent tests were skipped. The sole
failure remains the pre-existing user-worktree deletion of tracked
`images/IMG_9689c.JPG`; its reference-manifest integrity test correctly reports
that missing source. All 246 focused annotation, procedural, graph, Qt,
persistence, material-evidence, visualization, and image-backed cache tests
pass. Runtime diagnostics report the supported Python 3.12/PySide6/NumPy/
OpenCV/PyTorch environment ready with CUDA on the RTX 3070.

The suite was rerun after the node/overlay/cable consolidation on 2026-08-24:
503 tests ran, 500 passed, and 2 platform-dependent tests were skipped. The sole
failure remains the pre-existing deletion of tracked `images/IMG_9689c.JPG`;
the reference-manifest integrity test correctly rejects that missing source.
All 122 focused graph, settings-migration, cable-routing, overlay-contract, and
manual-centre Qt tests pass, as do the remaining computational and UI tests in
the full run.

The suite was rerun after combining the foreground/background directional-noise
cards on 2026-08-24: 506 tests ran, 503 passed, and 2 platform-dependent tests
were skipped. The sole failure remains the same pre-existing deletion of
tracked `images/IMG_9689c.JPG`; the reference-manifest integrity test correctly
rejects that missing source. All 166 focused graph, UI, settings-migration,
overlay, and device-cache regressions pass, as do all 65 broader pilot,
material-evidence, project, inspector, learning, and procedural-fit integration
tests.

## 2026-08-24 pipeline consolidation and evidence audit

The visible graph now uses calculation owners instead of legacy wrapper cards:
metadata is a peer of Raw images; colour-card work belongs to **Deskew and
colour balance**; ruler geometry and absolute scale belong to **Ruler detection
and scale**; generic ridges belong to **Edge gradients**; and semantic ridges
belong to **Reference edges**. Settings versions 1--6 migrate into the merged
owners while genuinely unknown nodes, controls, and wires remain hard errors.

The material-noise audit found and removed two unintended colour dependencies:
colour thresholds formerly invented texture pseudo-labels, and the final
texture result was blended with colour probability. Material-noise outputs are
now texture-only. The surviving graph dependency carries only the safely inset,
option-controlled annotated-Foreground reference region so cache invalidation
remains honest. The edge-prototype audit likewise found that absolute ridge
amplitude was embedded in the descriptor and multiplied into the class output.
The descriptor now contains semantic strip appearance only; gradient strength
is introduced downstream when constructing boundary/ridge products.

Normalized-reference trace doubling was a bilinear restoration halo: every
positive halo pixel was treated as a separate ridge. Semantic ridge sources are
now re-thinned along their live normal before component linking. Shared edge
magnitude is tested to change with derivative method, and graph invalidation is
tested to stop at upstream owners and recompute only the changed node and its
descendants.

Automatic procedural, provisional, U-Net, and StarDist inference no longer
receive painted instance IDs. Procedural fitting uses annotation-independent
image/material/edge evidence and treats masks only as scoring targets.
Reference-edge fitting trains and evaluates on disjoint alternating instance
IDs. See `docs/PIPELINE_CONSOLIDATION_AND_EVIDENCE_AUDIT.md` for the full chain-
by-chain findings, equations/ownership decisions, and regression inventory.

The 2026-08-24 validation covered all 511 tests in clean process groups: 508
passed, 2 fixture/platform-dependent tests were skipped, and the sole failure
is the established user-worktree deletion of `images/IMG_9689c.JPG`; its
reference-manifest integrity test correctly rejects the missing source. The 19
real-image pilot tests pass, as do all 331 post-pilot graph, Qt, persistence,
procedural, reference, shape-fill, and visualization tests. A single monolithic
process exhibited retained-state degradation after the learning tests (a
17-second clean fixture analysis grew beyond 30 minutes while still consuming
CPU); isolated process groups complete normally and expose no application
pipeline stall.

The 2026-08-26 node-control audit validation ran 518 tests in one process: 515
passed, 2 platform/fixture-dependent tests were skipped, and the sole failure
remains the established user-worktree deletion of `images/IMG_9689c.JPG`.
Before that full run, all 120 focused graph/inspector/UI/settings-persistence
tests and all 65 procedural-fit/learning tests passed. `git diff --check` also
passes; its output contains only the repository's existing LF-to-CRLF notices.

The 2026-08-26 Project-root migration passed all 119 focused pipeline-model,
pipeline-UI, and analysis-settings tests, plus the broader 233-test graph/UI/
project/reference/visualization group. A full-suite run was also attempted, but
the open desktop workload had the 8 GiB GPU at 99% utilization with about
7.9 GiB allocated; the real-image dense-lupin pilot continued to make forward
progress through existing directional-noise and boundary-tracing CUDA stages
but could not finish in a reasonable hand-off window. Timed stack traces showed
no wait in Project routing, persistence, annotation loading, or raw-image cache
invalidation. The only completed-suite failure before that contention point was
the established missing `images/IMG_9689c.JPG` fixture.

## Learned instance-segmentation implementation

Work continues on branch `codex/learned-instance-segmentation`. Two independent
disabled-by-default active-DAG nodes now provide native PyTorch learned
segmentation:

- **U-Net + watershed instances** predicts interior, physical boundary,
  apparent/non-physical coat-pattern boundary, centre plus normalized distance,
  and auxiliary dense-error probability. Complete instance masks derive their
  physical contour and sparse safely inset internal edge candidates directly;
  the intervening uncertainty band and flat interior are invalid for the
  non-physical-boundary loss. Its pattern-aware marker watershed can also
  consume applied painted instance IDs.
- **StarDist seed instances** predicts object probability, checkpointed radial
  distances, and auxiliary radial-error probability, with spatially indexed
  polygon NMS and local rasterization.

Both nodes use self-describing checkpoints, canonical seed-scale normalization,
overlap-blended CUDA inference, separate forward/decoder caches, typed controls,
node-owned overlays, and explicit missing-checkpoint warnings. Checkpoint
binaries under `models/` are ignored and neither node is enabled automatically.

The new `seedvision/learning/` package includes manifest/annotation I/O,
content-addressed target caches, exact StarDist target construction,
deterministic augmentation, training/early stopping, quantitative instance and
dense-head metrics, validation-only decoder search, frozen test evaluation,
comparison/contact-sheet rendering, unlabelled fixture review, and an
experimental U-Net-gated StarDist decoder. Applied complete masks can be
saved and loaded at full corrected resolution, then exported from the Learning
menu with explicit group, split, revision, author, and review metadata. The same
menu audits datasets and launches new or checkpoint-refinement training in the
serial CUDA worker, with epoch progress and optional checkpoint activation.

The controlled 72-image/1,474-instance synthetic engineering set produced a
locked-test U-Net F1 of 1.000, PQ 0.820, count error 0, physical-boundary F1
0.989, and pattern-boundary F1 0.839. StarDist locked-test F1 was 0.996 and PQ
0.825, but validation and real-image false-object behaviour was substantially
worse. These are not scientific results.

Visual review on all eleven real fixtures shows that simulator-only weights do
not transfer: neutral-species U-Net counts are severely low except on sparse
`IMG_9670c` (15 predicted versus 16 visibly present), while StarDist strongly
overpredicts pattern/rim/background objects. The repo still has no complete
human-reviewed real instance masks, so no real accuracy can be calculated and
publication readiness cannot be claimed. See
`docs/LEARNED_INSTANCE_SEGMENTATION.md` for the protocol and
`docs/LEARNED_SEGMENTATION_RESULTS.md` for exact metrics, timings, fixture-by-
fixture visual findings, and the required real-data path.

## Procedural separation outcome and next approach

The complete procedural proof of concept is implemented, but visual QA says it
is an annotation bootstrap rather than a validated counter. It usefully
separates pale round seeds and gives plausible dense round-seed partitions, but
strongly bicoloured elongated lupins still show obvious false splits at coat
transitions and merges at contacts. The sparse `IMG_9670c.JPG` dish visibly has
16 seeds and receives 18 automatic instances. Automatic fixture counts are
diagnostic only: `IMG_0002c` 621, `IMG_9632c` 130, `IMG_9636c` 92, `IMG_9641c`
90, `IMG_9666c` 129, `IMG_9667c` 138, `IMG_9668c` 84, `IMG_9670c` 18,
`IMG_9685c` 116, `IMG_9689c` 58, and `IMG_9974c` 563.

After vectorizing label statistics, the procedural node takes approximately
0.41–0.87 seconds on the RTX 3070 for the tested 856–1758 px dish crops; the
complete active analysis takes approximately 2.1–6.3 seconds. Further manual
threshold tuning did not resolve the core ambiguity because true contacts and
within-seed coat boundaries can have equally strong edges. The next justified
step is to review masks produced by this node and train a small
species-conditioned boundary/instance model (U-Net plus watershed or StarDist).
Do not start with a transformer: eleven unlabelled fixtures do not support it.

## Net reference-edge subtraction and prototype diagnostics

The Reference edges node owns a raw cached diagnostic margin,
`max(Pphysical - weight * Pnonphysical, 0)`, and a distinct authoritative
Reference-edge probability formed by multiplying Pphysical by the generic
full-resolution thinned true-edge support. The supported margin is separately
available as **Conservative net physical-edge evidence**. Its **Non-physical subtraction
weight** is the first full inspector control and first inline control (default
0.5) in the **Optional conservative net evidence** section. Assisted Smart/Shape
fill can explicitly choose either supported field; the normalized probability,
canonical ridge, and default downstream reference input use Physical probability
without the extra subtraction. Changing only the weight reuses upstream learned
prototype banks and cannot change authoritative or normalized probability.

Reference texture prototypes also publishes a **Reference prototype source
footprints** overlay. Material markers show the retained medoid pixel and its
context radius. Edge markers show the three tangent-aligned interior/centre/
exterior lines and the five actual bilinear samples on each, with dot sizes
encoding the 1:2:3:2:1 pooling weights. The prototype collage and inspector
help now state explicitly that thumbnail patches are presentation context, not
fitted templates, and that no filled region between the strip guides is
matched.

## Overlay and normalized-reference-edge cleanup

The foreground binary proposal viewer mode was removed: it rendered the old
baseline foreground mask rather than the material-decision node's own product
and had no active consumer. The Background/Other subtype ambiguity and unknown
maps were valid computed products but were missing from the image viewer's
renderable-mode registry; they are now normal probability overlays with their
original node ports and legends. **Boundary confidence and normals** remains a
dormant legacy toolbox diagnostic. It is disabled and shelved by default with
no active segmentation consumer, so its bypass maps are intentionally zero;
the node details and overlay legends now say so explicitly.

The separate Perimeter background reference node has been retired. Layout
detection now owns its two annulus controls, typed output port, overlay, status,
cache unit, and downstream connections. Settings-profile format 10 migrates the
old node's parameters and rewires its output to Layout detection. The Reference
edges viewer calls the continuous product **Normalized reference-edge
probability** and no longer exposes raw prototype compatibility as an
assisted-tool source. Supported and normalized products share the same blue
scale for direct comparison.

Normalized reference-edge support is one-sided: local gain is clamped to
`[1, maximum_gain]`, so normalization cannot make above-floor ridge support
weaker. The gain denominator is capped at current pixel support so a strong arc
inside the seed-scale window cannot block amplification of its immediately
adjacent weak continuation. The smooth absolute gate reaches full weight at
the configured floor, and high/low ridge hysteresis—not local intensity
leakage—decides whether an enhanced weak maximum belongs to a coherent ridge.

The 2026-08-27 focused graph/UI/persistence/material/cache/edge run passed all
209 tests. Full discovery ran 524 tests in 307 seconds: 521 passed, 2 expected
environment/platform tests were skipped, and the sole failure remains the
pre-existing committed deletion of `images/IMG_9689c.JPG` while the reference
manifest still names that source fixture. `git diff --check` passes apart from
the repository's existing LF-to-CRLF notices.

## Material-prototype probability calibration

The Reference texture prototypes node no longer treats its Gaussian-kernel
similarities as posterior masses. That interpretation gave an ordinary
one-robust-scale in-class descriptor a score near `exp(-0.5)` and then divided
it again by all class scores plus unknown mass, creating a second artificial
middling ceiling. The new global calibration separates unsharpened
known-material confidence from relative class competition, with an exposed
**Material class contrast** default of 4.0. Uniformly weak matches remain
unknown and exact Background/Other ties remain ties.

The transform receives only score rasters and the valid-image mask, never
reference masks or coordinates. A regression test gives identical score
vectors at two nominal locations and requires bit-identical output, alongside
tie and uniformly-weak cases. On the saved `IMG_9405.JPG` sidecar, painted Other
pixels changed from mean/median Reference Other-material probability
0.536/0.541 to 0.775/0.828; competing Foreground and Background means there are
0.087 and 0.032. Painted Foreground and Background target means are 0.810 and
0.706. These are learned global responses, not hard-written reference values.
Settings format 11 persists the contrast control and version 10 or older
profiles adopt the current 4.0 default. The prototype cache signature includes
the control.

## Edge-prototype probability calibration

Physical-edge and Non-physical-edge prototype similarities now use the same
two-stage, mask-blind calibration as material prototypes. The former direct
`score / (physical + nonphysical + 0.10)` conversion treated Gaussian-kernel
similarity as posterior mass and imposed an artificial middling ceiling.
Unsharpened strongest similarity now determines known-edge confidence, while
the exposed **Edge class contrast** (default 4.0) affects only relative class
competition. Exact ties remain ties and uniformly weak matches retain mostly
unknown mass. The transform receives descriptor scores and a generic valid
strip mask, never annotation targets or coordinates, so reviewed pixels cannot
be hard-written. The leakage-safe held-out-instance edge fitter includes the
new contrast in its bounded optimization. Settings format 12 persists it;
version 11 and older profiles adopt 4.0. Prototype cache identity includes both
material and edge contrast controls.

The focused graph/UI/persistence/material/edge/cache/pilot run passed all 216
tests in 190 seconds. Full discovery ran 527 tests in 289 seconds: 524 passed,
2 platform/environment tests were skipped, and the sole failure is still the
pre-existing missing `images/IMG_9689c.JPG` fixture referenced by the committed
instance-reference manifest. `git diff --check` reports only the repository's
existing LF-to-CRLF notices.

## Material-noise class-balance redesign

Material-noise classification now uses nine fixed, texture-only descriptor
channels: residual RMS plus principal- and cross-axis local variation at fine,
medium, and coarse scales. This adds oriented multiscale structure without
creating a second general-purpose prototype bank alongside Reference texture
prototypes. Each class is now a target-only compatibility model: the robust
target distribution and its 95th-percentile positive distance define the
global half-support distance, and no semantic counterclass enters another
class's calibration or pixel score. Cross-class reliability and all semantic
contrast occur exclusively in Material evidence decision. Reference pixels
remain training examples and are never hard-overridden.

Cross-class invariance regressions change sibling annotations to genuinely
different texture types—not merely duplicated pixels—and require bit-identical
Foreground, Background, and Other raw rasters while still requiring a changed
target reference to change its own result. The saved `IMG_9405.JPG` audit now
reports painted-Other mean/median Other-texture compatibility 0.892/0.918 and
Foreground-texture compatibility 0.455/0.486. Painted Background can
legitimately score as both Background (0.947 mean) and Other (0.655 mean),
while painted Foreground scores 0.793 mean as Foreground and 0.094 mean as
Other. These are learned target-only responses, not mask-written values.

All 65 visualization-layer tests, 139 focused material/pipeline/UI/settings
tests, and all 19 real-image pilot tests pass after the target-only change. The
pilot run took 513 seconds under simultaneous desktop GPU load. After removing
the obsolete counterclass fields from the public texture profile, final full
discovery ran 534 tests in 414 seconds: 531 passed, 2 expected tests were
skipped, and the sole failure remains the pre-existing missing
`images/IMG_9689c.JPG` source still named by the committed reference manifest.

## Semantic reference-seed traits

Reference seed instances can now carry one optional species-specific,
mutually-exclusive coat-pattern label plus independently overlapping condition
labels. `Lupinus mutabilis` currently exposes White, Banded light, Banded dark,
and Other coat patterns; the shared condition vocabulary is Immature, Split,
Wrinkled, and Stained. A separate Reviewed control distinguishes an explicitly
sound/absent condition example from an unreviewed seed, so unchecked conditions
are never silently used as negatives. The selected seed's controls live in the
existing seed-instance editor and are enabled only after that seed ID has a
painted mask.

Reference-region format 3 stores the semantic records and their species
vocabulary as strict scalar JSON metadata beside the full-resolution uint16
instance raster, preserving `allow_pickle=False`. Versions 1 and 2 still load
with empty semantic labels. Orphan IDs, malformed identifiers, duplicate
conditions, and positive conditions without explicit review are rejected. The
project/reference association panel reports saved coat and condition-review
counts. Semantic-only Apply + save invalidates only the new diagnostic branch;
painting or deleting instance pixels retains the established full annotation
dependency behavior.

The terminal **Reference seed traits** node learns separate image-local Lab,
multiscale texture, edge/ridge, and local-residual prototype banks from safely
inset labelled seed material. Coat probabilities compete only with one another
and are renormalized to sum to one on the resolved seed-material domain when at
least two classes have support. Each condition is a separate
reviewed-present-versus-reviewed-absent probability and may overlap other
conditions. Missing calibration yields black/unavailable output. The node only
consumes the current resolved seed mask and has no downstream connection, so it
cannot alter Background, Foreground, Other, or physical/non-physical edge
calculations. Painted semantic labels train the models but never overwrite
their output pixels. Settings-profile format 14 adds the node; version 13 and
older profiles adopt its terminal default configuration and connections.

Clean-process verification covered all 544 tests: 541 passed, 2 expected
fixture/platform cases skipped, and the sole failure remains the established
missing tracked `images/IMG_9689c.JPG` file named by the committed reference
manifest. All 19 real-image pilots and the semantic archive/editor/cache/
normalization regressions pass. A monolithic discovery process again exhibited
the documented retained-state slowdown after earlier suites; clean process
groups completed normally. `git diff --check` reports only the repository's
existing LF-to-CRLF notices.

## Signed probability-excess diagnostics

The combined **Material colour probabilities** and **Material noise
probabilities** nodes now each publish a display-only signed comparison of raw
Foreground evidence with the strongest raw Background/Other response. For
each pixel, `d = Foreground - max(Background, Other)`; blue is `max(d, 0)`, red
is `max(-d, 0)`, and a tie is black. The comparison deliberately neither adds
the two independent Non-seed compatibilities nor normalizes the three raw
classes. It reuses the existing lazy rasters only when selected and creates no
new calculation, cached tensor, graph dependency, or downstream evidence.

**Reference edges** similarly publishes **Physical vs non-physical edge
excess**, an unscaled direct margin with Physical blue and Non-physical red.
It complements the overlap-preserving magenta-capable comparison and is
deliberately independent of the separate weighted/clamped Net physical-edge
calculation.

All 112 visualization/model calculation tests and all 64 pipeline-UI tests pass
in clean processes. Repository-wide clean-process coverage completed every
test except the two-pass dense-lupin pilot, which remained compute-bound while
the host GPU was externally saturated at 99% and 7.9/8.0 GiB; it was stopped
without an exception. The other 18 pilot methods pass separately. Across the
544 uniquely completed tests, 541 passed, 2 expected platform tests skipped,
and the sole failure remains the established missing tracked
`images/IMG_9689c.JPG` fixture named by the committed reference manifest.
`compileall` passes and `git diff --check` reports only the repository's
existing LF-to-CRLF notices.

## Edge-supported reference-edge authority

Raw Physical, Non-physical, and weighted-net edge descriptor responses are now
explicitly presented as **prototype compatibility** diagnostics. They remain
available for inspecting how the strip descriptors classify context, but no
fill or procedural boundary path consumes their spatially broad fields.

The canonical full-resolution product is now exactly
`thinned_true_edge_support * Pphysical`.
The Reference edges node exposes this as **Reference-edge probability** and
receives its true-edge ridge through an explicit typed graph connection. Its
normalized and hysteresis-thinned derivatives are masked back to the exact
full-resolution true-edge footprint after resizing, so descriptor and
interpolation halos cannot create barriers. Assisted fills map their persisted
`net_physical` source to this supported field, and procedural separation now
receives the canonical product through its own typed input instead of
reconstructing an unweighted raw-class difference. Curve and learned branches
that require separate edge classes receive edge-supported Physical and
Non-physical compatibility maps.

The previous lambda-based supported margin remains a separate optional product,
**Conservative net physical-edge evidence**, selected explicitly by assisted
fills with `conservative_net_physical`. The existing `net_reference_ridges`
trace source now names its actual conservative ridge, while `reference_ridges`
and the normalized branch use authoritative Physical probability. Existing
normalized storage/port IDs retain their legacy `net` spelling for saved-graph
compatibility; their UI labels say Reference-edge probability. Neither the
prototype classifier, unknown confidence, annotation masks, nor fill growth
algorithm is changed by this split.

Settings-profile format 15 adds the true-edge-support input and authoritative
procedural-edge connection; version-14 and older profiles adopt both current
connections during migration. Formula, zero-off-ridge, cache, graph, migration,
fill-source, procedural, visualization, and UI regressions pass. Full discovery
ran 546 tests in 315 seconds: 543 passed, 2 expected platform/environment tests
were skipped, and the sole failure remains the pre-existing missing
`images/IMG_9689c.JPG` fixture referenced by the committed instance-reference
manifest. `compileall` and `git diff --check` pass; the latter reports only the
repository's existing LF-to-CRLF notices.

## Annotation-guided Reference edges fit application

Starting the Reference edges fit now explicitly confirms a fit-and-apply
operation. A genuinely improving best proposal is applied directly to the live
node when the worker completes; the former second default-No question no longer
silently discards a completed proposal. The node is selected and its inspector
controls are rebuilt from the applied values, while the status bar enumerates
every old-to-new setting change. The application log records the image, loss
change, and complete fitted-value mapping.

Applying these controls now also invalidates the combined Reference texture
prototype cache, which currently owns the raw physical/non-physical class-map
calculation, plus all of that producer's dependents. This prevents updated live
settings from being paired with resident pre-fit prototype rasters. A UI
regression covers automatic application, live node values, producer-cache
invalidation, and recomputation dispatch. Fit progress reports best-so-far loss
rather than whichever candidate happened to run most recently. The 10 focused
fit/inspector/application tests pass. All 528 non-pilot tests ran in 202 seconds:
525 passed, 2 expected platform/environment tests skipped, and the sole failure
remains the pre-existing missing `images/IMG_9689c.JPG` fixture. A monolithic
discovery run was stopped after roughly 29 minutes in the already documented
retained-state real-image pilot slowdown; it was still CPU-active, several
pilots had completed, and no additional failure marker had appeared.

## 2026-08-31 authoritative Physical probability and conservative evidence

Reference-edge probability is `R * Pphysical`, where `R` is the existing
full-resolution thinned true-edge support. Its normalized and thinned
derivatives use the same Physical probability and are bit-identical when only
the Non-physical subtraction weight changes. The classifier still retains its
known-versus-unknown confidence; no conditional ratio or second subtraction
is applied to the authoritative probability.

The previous `R * max(Pphysical - lambda * Pnonphysical, 0)` is retained as
**Conservative net physical-edge evidence**, with a separate blue overlay,
typed output, Smart/Shape-fill source, and thinned ridge. Trace sources can
select the conservative ridge explicitly. Both products are exactly zero off
the true-edge footprint. The canonical probability shares its lazy GPU raster
with the already supported Physical class input. Raw prototype compatibility,
annotation handling, and the fill growth algorithms are unchanged. Stable
graph/tool IDs preserve existing settings profiles without a schema bump.

The full suite ran in two clean process groups: 530 non-pilot tests in 179
seconds and all 19 real-image pilots in 99 seconds. Across all 549 tests,
546 passed, 2 expected platform/environment cases skipped, and the sole failure
remains the pre-existing missing `images/IMG_9689c.JPG` source named by the
committed reference manifest. Formula, lambda-invariance, cache reuse,
zero-off-ridge, source-adapter/fallback, overlay-port naming, UI, fit, and
settings-persistence checks pass. `compileall` and `git diff --check` pass.

## Layout vessel selection

Layout detection's inspector begins with a **Seed vessel type** dropdown pinned
to **Glass Petri-dish**. This disabled, single-option capability indicator is
deliberately separate from computational parameters: other vessel/surface
detectors are not implemented, and opening the inspector cannot change settings
or invalidate analysis caches. The existing Petri-dish calculations, profiles,
and geometry are unchanged. A Qt regression covers ordering, locking, node
switches, and absence of parameter-change emissions.

Verification: both focused inspector tests pass, and the rendered settings panel
was checked with the application's Windows font. Full discovery in clean
non-pilot/pilot processes ran all 550 tests: 547 passed, 2 expected tests skipped,
and the only failure remains the missing `images/IMG_9689c.JPG` manifest fixture.
All 19 real-image pilots pass. Compilation and `git diff --check` pass.

## Species libraries and reference dimensions/shape implementation

`docs/SPECIES_REFERENCE_LIBRARIES.md` and
`docs/REFERENCE_SEED_DIMENSIONS_AND_SHAPE.md` are now implementation records,
not unstarted proposals. One shared immutable library framework publishes and
resolves source-balanced foreground colour/noise, material prototypes, aligned
edge prototypes, seed traits, shape summaries, and joint dimensions/shape
banks. Projects pin an exact ID/version/content hash; runtime assembly excludes
the current source; missing pins offer exact bundle import and never follow
latest. The manager supports build, grouped validation, publish, fork,
import/export, retirement, and pinning.

Shape reference metadata, robust 2-D measurements, coherent-boundary
uncertainty, physical-seed repeated-view grouping, pose-conditioned contour
families, species/lineage/accession/lot partial pooling, overlays, and opt-in
geometry consumers are implemented. Uncalibrated masks contribute only
dimensionless shape. The legacy scalar diameter is preserved as a compatibility
adapter. Intrinsic 3-D recovery remains explicitly deferred pending paired views
or physical thickness data.

Library extraction now calls the production descriptor seams and uses bounded
source/seed-balanced compact transfers rather than approximate CPU features.
Descriptor schema IDs were advanced so older approximate artifacts cannot be
silently interpreted as compatible. The focused library/shape/persistence/
pipeline/UI/procedural/visualization regression set passes 264 tests. After the
project-schema expectation was updated, final full discovery ran 565 tests in
287 seconds: 562 passed, 2 expected tests skipped, and the sole failure is the
established absent `images/IMG_9689c.JPG` source named by the
instance-reference manifest.

## Procedural centres, reference-error diagnostics, and annotation/library UX

Procedural centre likelihood now consumes the proposal-independent,
fit-validated oval-centre probability from Seed-boundary confirmation. The
editable positive-only weight defaults to 0.85 and probabilistic-union fusion
can boost a retained oval centre without suppressing the existing material,
interior-depth, and flattened-grayscale fallback. The raw centre-vote field,
raw peak markers, and fit-validated oval-centre probability have explicit names
and remain separate diagnostics. The graph dependency is typed, cached, and
causes only the procedural branch and its dependents to be invalidated.

Procedural seed separation also exposes **Reference underreach / overreach
cost**. After automatic inference is complete, annotated and predicted labels
are globally matched one-to-one above a minimum IoU, with explicit unmatched
choices. Blue represents underreach; red distance-weighted overreach; amber
missed references; magenta incorrect concavity pockets. The exact same pixel
cost implementation serves fitting and display. The annotation raster is never
passed into centre discovery, watershed, candidate selection, shape filtering,
or confidence. Shared matching and cost controls affect the fitting objective
and post-inference comparison, never the automatic prediction directly.

The seed-trait editor no longer permits the invalid combination that caused the
opaque project-save failure: a seed cannot be included in shape modelling while
Outline or Pose is Unknown. Outline and Pose now precede the inclusion control,
an inline status explains eligibility, and old invalid in-memory records remain
editable. Persistence errors name the exact seed and repair action; project-save
errors explicitly state that saving is paused and no in-memory data was lost.

The species-library manager is reorganized into three numbered workflows:
**Use a library**, **Create a version**, and **Inspect selected version**.
Pin/import/export/retire actions, reviewed source selection, optional biological
context, forked-source retention, immutable publication, product coverage, and
validation are now grouped and explained where they are used. Selection counts,
dynamic action labels, fork requirements, tooltips, and build status are visible
without switching between disconnected tabs. Analysis-settings format 17 adds
the new procedural controls and connections; version 16 and older profiles
adopt authored defaults.

## Annotation and shape-controls revision — 2026-09-05

See `docs/ANNOTATION_SHAPE_AND_EDGE_REVIEW_2026_09_05.md` for the rationale,
measurement semantics, compatibility notes and punctate-edge classifier audit.
Detected ruler now also shows the independently measured imperial span. The
annotation panel has a draggable title bar and resize grip; the tool-settings
stack sizes to the active page (hidden fill pages caused the large brush gaps).
Existing-ID selection, a red empty marker, adjacent selected-only control,
exclusive No defects versus defect labels, and an on-image hilum target/vector
editor are implemented. Physical seed ID is no longer exposed but old values
are preserved.

Maximum span uses all explicitly complete or full-length-visible annotations,
never a top fraction. Partial full-length masks contribute size only, not shape.
Complete reviewed masks additionally supply mean internal concavity and signed,
seed-balanced turning/curvature distributions. A typed compact summary feeds
soft trace/boundary priors; upstream gradient/ridge caching is preserved.
The Reviewed measurements, size–ovality, uncertainty and new curvature overlays
now explain their quantities. Uncertainty is sensitivity, not a validated CI.

Fixed the reversed major-axis comparison in OpenCV ellipse-angle conversion
(it rotated bodies 90 degrees and corrupted robust-fit residuals). Added inward/
outward perturbations to uncertainty, raster-consistent convex-hull concavity,
and cropped per-seed morphology. Shape-library descriptor schemas are now v3;
old shape banks are disabled with a rebuild warning, while other products stay
usable. Old shape observations are not silently promoted during fork recovery.
Reference archives are v5 (v1–v4 still load), analysis settings v18 (retired top-
fraction parameter migrates away). Existing saved source data is not overwritten.

`tests/test_annotation_shape_revision.py` covers the new UI, geometry, migration,
no-mask hilum editing, shape-bank compatibility, and curvature/cache contracts.
Generated UI screenshots are only in ignored `artifacts/`.

Validation: final full discovery ran 585 tests in 273.069 seconds: 582 passed,
2 skipped, and the sole failure remains the pre-existing missing manifest image
`images/IMG_9689c.JPG`. Compileall and scoped diff whitespace checks passed.

## Procedural reference assignment and cost repair — 2026-09-05

See `docs/PROCEDURAL_REFERENCE_MATCHING_AND_COSTS.md`. Replaced greedy raw-area
matching with component-local global IoU assignment and unmatched choices.
Incidental contacts no longer turn neighbouring unreviewed seeds into costly
false correspondences. Missed references are now visible and have a lower
default pixel cost (0.5). Incorrect exterior-connected candidate-concavity
pockets incur a +2 surcharge; correct natural indentations and enclosed holes
do not. Pixel loss uses fixed reviewed area, so omission weights actually affect
the score. Overlay and fitter share summed pixel costs, with blue/red/amber/
magenta diagnostics and a fixed cost-4 alpha scale. Cost knobs cannot be fitted
away and cannot alter automatic predictions directly. The existing fit-action
controls now alias the node's persisted values; settings schema 19 adds the
new matching, missing, concavity, and coverage defaults.

Validation: 135 focused tests pass. Full discovery ran 595 tests in 268.691
seconds: 592 passed, 2 skipped, and the only failure is the pre-existing missing
`images/IMG_9689c.JPG` manifest fixture. Compilation, diff whitespace checks and
CUDA runtime diagnostics pass. The synthetic cost-overlay contact sheet was
visually checked and is retained only in ignored `artifacts/`.

## First actions for the next agent

Read `AGENTS.md`, inspect `git status`, run diagnostics and the test suite, then
continue from the user's newest request. Preserve unrelated user changes and
do not commit generated artifacts or a local virtual environment.
