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

- The overlay selector is a node-first menu whose headings exactly match the
  active pipeline cards and inspector titles. It has no checkable children,
  synthetic Viewer/None group, or Qt mnemonic ampersands. Every selectable
  overlay is mirrored by an identically labelled output connector on its owner
  node, including dynamically generated colour/pattern overlays; a Qt contract
  test selects every owner and checks all three surfaces.
- **Manual annotations** now combines the former Reference layers and Manual
  seed centres inputs. Its explicit Background, Foreground, Other, Annotated
  seeds, and Manual seed centres connectors preserve the separate evidence
  types and their image-local persistence.
- **Material colour probabilities** now combines the foreground and background
  colour cards and controls without combining their calculations. Foreground,
  Background, and Other remain independent evidence rasters with their prior
  equations and cached intermediates. **Use background colour analysis** turns
  off only Background/Other fitting; mandatory-reference Foreground fitting
  remains active. Version-4 settings migrate both former node records and
  connections into this combined node.
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
  nodes off. One **Manual annotations** input exposes separately typed
  Background, Foreground, Other, Annotated seeds, and Manual seed centres
  outputs, with explicit connections to every calculation that consumes them.
  Physical and non-physical edge training
  masks are regenerated from the annotated IDs and are not separately painted
  graph inputs. The one-line toolbar includes an
  unused-node toolbox; **Circle candidates**, **Distance-peak candidates**,
  **Calibration residual risk**, the
  directional surface-darkness gradient branch and its lightening/darkening
  derivative upper cutoffs, and every distance-dependent node are preserved
  there with their authored connections but excluded from the default DAG and
  calculations.
- The complete 54-node active/toolbox catalogue has an independent direct-input
  contract test. Corrected-image, dish-region, absolute-scale, seed-diameter,
  mask, proposal, and reference dependencies are explicit wherever the runtime
  reads them. In particular, **Edge gradients** consumes **Layout detection**'s
  dish region and **Oriented edge traces** consumes **Seed scale estimate**;
  scale-only changes rebuild traces while retaining cached ridges. Oriented
  traces expose their selected source in the node and overlay legend and can use
  generic ridges, physical-reference ridges, raw net-reference ridges, or the
  locally normalized net-reference ridges. Trace-gap
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
  direction bank. Background noise retains its maximum default. The Background
  colour/noise nodes also expose direct-bright **Other colour probability** and
  **Other noise probability** overlays when Other examples have been painted.
  The colour view is the raw competing Lab membership already used by the
  background model; the noise view fits Other against painted Background and
  Foreground texture, blends 72% three-band texture with 28% Other-colour
  probability, and applies the background node's directional integration.
  These diagnostics remain blank without Other paint and are separate from the
  multifeature Reference Other-material prototype probability.
- **Hue only** displays corrected hue at fixed neutral brightness/chroma, with
  achromatic pixels shown neutral gray. **Wavelet decomposition** is a four-
  level stationary B3-spline à trous decomposition with selectable detail
  layers and a residual; all layers stay at full image resolution and the four
  details plus residual reconstruct the original corrected RGB tensor exactly.
- **Manual annotations** is an explicit active input node for the mutually
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
  These derived examples fit an image-local **Instance-derived edge probabilities**
  classifier from corrected Lab values, edge magnitude, tangent coherence, and thinned ridges.
  Its physical/non-physical outputs feed boundary confirmation and procedural
  watershed. Two derived, viewer-only diagnostics materialize those existing
  lazy rasters only when selected: **Physical blue / non-physical red** uses
  magenta for overlapping support, while **Net physical-edge probability**
  displays `max(physical - k × non-physical, 0)` in blue with a black floor.
  Its node-owned **Internal-edge subtraction** coefficient defaults to 0.5 and
  supports 0--2 without modifying either source raster. The subtraction is a
  positive evidence margin, not a calibrated posterior. A
  separately cached **Thinned reference edge ridge** node applies
  normal-direction NMS and CUDA hysteresis to the continuous physical-edge and
  raw-net fields. It also publishes **Locally normalized net physical edge**
  and a thinned normalized ridge. The normalized path separates the
  physical-versus-internal margin from their shared absolute support, estimates
  a seed-scale winsorized local RMS envelope, applies bounded symmetric gain,
  and finally applies an absolute smooth floor so quiet-region noise is not
  promoted. Its default radius, target support, maximum gain, and floor are
  respectively 0.30 seed diameter, 0.35, 2.5x, and 0.04. The full rationale,
  equations, safeguards, and validation plan are in
  `docs/LOCAL_EDGE_NORMALIZATION.md`. The node owns those four controls plus
  independent NMS step, low/high threshold, hysteresis-reach, and working-size
  controls. Raw Physical, Non-physical, Net, and raw thinned-ridge diagnostics
  remain unchanged. The formerly black continuous normalized overlay was a
  display-quantization bug: its `[0, 1]` raster is now multiplied to display
  range before conversion to `uint8`; its derivative overlays were already
  generated through a separate correctly scaled path. When no annotated seed
  exists, the semantic classifier is neutral instead of treating every generic
  image edge as a physical reference. Learning export derives both boundary
  targets and their sparse validity raster from the instance labels.
- Active **Procedural seed separation** combines seed-material evidence,
  generic edge/ridge candidates, instance-derived semantic margin, separately
  thinned physical ridges, and coherent convex oriented traces into its
  physical-boundary cost. The locally normalized net margin gates the whole
  candidate field while raw non-physical probability remains an explicit
  negative-evidence discount; the gate is neutral with no annotations. Centre markers
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
  The **Manual annotations** graph input and its inspector editor allow
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
- **Perimeter background reference** is a separate cached active node. Its
  ruler-calibrated outer-rim buffer defaults to 0.35 cm and its independently
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
  pixels. The three material classes compete through one shared normalization
  with reserved unknown mass rather than three unrelated high scores; this
  improves the ambiguous pale-seed/spot/background-texture case. Every
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
- Both probability nodes expose **Maximum reference colour modes** as the
  user-facing capacity control. Foreground retains coverage-preserving quantized
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
- The independent seed-interior branch remains visible but disabled. It now
  consumes foreground colour probability, learned foreground-noise probability,
  reference-prototype seed-surface probability, and background evidence;
  independent editable weights control both learned texture and reference
  texture before background support is incorporated. The
  distance, instance, final boundary, review, measurement, classification,
  aggregation, and output branch is outside the active DAG in **Unused nodes**.
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
selectable Thinned reference edge ridge and its CUDA NMS/hysteresis controls.
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

## First actions for the next agent

Read `AGENTS.md`, inspect `git status`, run diagnostics and the test suite, then
continue from the user's newest request. Preserve unrelated user changes and
do not commit generated artifacts or a local virtual environment.
