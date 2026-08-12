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

- Colour-card/swatch detection, colour balance, projective deskew, ruler
  detection, 5 cm scale overlay, absolute scale, and dual Petri-dish rims.
- Petri-dish search uses the ruler-calibrated, editable expected 96 mm outer
  diameter to prevent crowded seed-mass edges from replacing the physical rim;
  all eleven committed fixtures have visually verified inner and outer edges.
- Native Qt node graph with inline Blueprint-style controls, dependency-aware
  caching, node timings, progress colours, purple adjacent-node highlighting,
  manual overlap, automatic non-overlapping arrangement, zoom controls, and
  node-driven viewer overlays/intermediates. Every authored datum has its own
  succinct labelled and typed input/output socket; all active and toolbox
  connections use explicit endpoints. Users can drag to restore supported
  connections, drag a connected input into empty space to disconnect it, or
  right-click/select-delete an edge. Disconnection bypasses enabled consumers
  and their dependents; reconnection restores only the cards automatically
  suspended by that missing input, leaving deliberately disabled experimental
  nodes off. An explicit Painted seed instances input exposes annotation
  dependencies that were previously implicit. The one-line toolbar includes an
  unused-node toolbox; **Circle candidates**, **Distance-peak candidates**,
  **Calibration residual risk**, the
  directional surface-darkness gradient branch and its lightening/darkening
  derivative upper cutoffs, and every distance-dependent node are preserved
  there with their authored connections but excluded from the default DAG and
  calculations.
- Overlay selection and opacity are compact top-toolbar controls. The main
  selector groups indented layers under disabled owning-node headings, and a
  synchronized selector directly below the selected node title lists only that
  node's overlays. Material-reference and boundary-reference painting are
  top-toolbar modes that reveal one contextual control
  panel over the image; the right inspector contains only image metadata, the
  selected `Node:` controls, its local overlay selector and explanation, and
  applicable summaries.
- A separate **Annotate seed instances** mode records distinct-colour,
  full-resolution integer seed IDs. It provides a freehand brush, magnetic edge
  tracing, edge-supported circle/ellipse snapping, adaptive smart fill, and an
  eraser. Assisted tools now show debounced live previews and commit on click:
  edge trace uses click anchors and a magnetic path preview, shape snap shows a
  nominal shape cursor plus the fitted boundary, and smart fill previews its
  region even when the active seed has no prior marks. Trace/shape tools
  optionally use directed or undirected edge tangents. Smart fill uses OpenCV's
  native floating-range neighbour comparison, preserves other IDs,
  stops at edges, and exposes bounded tunnelling/tolerance/growth options for
  patterned seeds. Freehand drags use a lightweight vector stroke preview and
  rebuild the annotation raster once on release; they never rebuild the
  analysis overlays. Assisted algorithms crop edge work to their local cursor
  region instead of rebuilding or normalizing full-resolution rasters on every
  event.
  Draft/apply/revert state is per image and independent of the
  binary colour-reference masks; applied labels become authoritative markers
  for the active procedural watershed and also constrain the dormant
  provisional instance branch without overriding foreground probability.
  A **Start from result** selector expands any available procedural,
  U-Net/watershed, or StarDist labels into a full corrected-image editable
  draft. The initializing method is preserved in export notes; the draft stays
  explicitly unreviewed and requires correction of every automated error.
- GPU foreground/background colour probabilities, symmetric three-band noise
  profiles with directional texture continuation, shared edge gradients,
  directed/undirected tangents, ridge thinning, and oriented edge traces.
  Directional rays are integrated on CUDA into each class noise probability but
  are no longer materialized as individual viewer overlays.
- **Painted material references** is an explicit active input node for one
  mutually exclusive Background/Foreground/Other categorical layer. Painting a
  class clears the other two at that pixel; Other supplies negative evidence to
  both material models. It feeds both colour models and both class-specific noise
  models, so applying a painted edit invalidates every true graph dependent.
  Each noise classifier learns its positive and negative texture distributions
  directly from the applicable painted areas when present, using colour
  pseudo-labels only for an unpainted class. Painted coordinates are not forced
  to exact noise-probability zero or one. A separate **Painted boundary
  references** input holds mutually exclusive Physical edge/Non-edge review
  evidence. Its smart boundary brush previews and commits either class to nearby
  analysed edges, and learning export persists both values and their sparse
  validity raster.
- Active **Procedural seed separation** combines seed-material evidence,
  edge/sensor/ridge/shadow physical-boundary cost, scale-aware centre markers,
  marker-controlled watershed, and explicit per-instance confidence. Its six
  overlays expose material likelihood/mask, boundary cost, centre likelihood,
  instance identities, and confidence. GPU rasters are resized before the one
  essential bounded CPU topology transfer; CPU labels and evidence stay at the
  bounded working size and Qt scales them only for display. The compact result
  is node-cached, and
  distinct painted instance IDs suppress nearby automatic markers.
- **Perimeter background reference** is a separate cached active node. Its
  ruler-calibrated outer-rim buffer defaults to 0.35 cm and its independently
  adjustable median-colour band thickness defaults to 0.5 cm; both are shown
  exactly in a dedicated overlay, with a nominal-dish scale fallback. The
  overlay includes an opacity-independent swatch and hex label for the selected
  median starting background colour.
- A six-output active multiscale node supplies fine/medium/coarse surrounding RMS
  darkness and Lab-colour noise energy. The maximum one-sided lightening and
  darkening CIE L* surface slopes, their query-to-target directions, and the two
  dependent derivative upper cutoffs are disabled and preserved in the
  unused-node toolbox. These raw masks are intentionally distinct from the
  learned foreground/background noise profiles.
- Three annotation layers: categorical material references
  (Background/Foreground/Other), categorical boundary references (Physical
  edge/Non-edge), and labelled seed instances. Each categorical layer enforces
  exclusivity while it is painted and normalizes older four-mask state when it
  is read. Other samples fit negative colour-frequency and texture distributions
  for both material models; they never hard-zero their painted coordinates. The
  compact editor uses class selectors plus shared Paint/Eraser/Clear-layer,
  Apply/Revert, brush-radius, and painted-overlay visibility controls; longer
  explanations live in tooltips. It also retains Lab
  probability models shown as saturation-projected chromatic HSV
  hue/tint/shade contour slices for both colour classes, preceded by an exact
  neutral white-to-black strip. This preserves pale tinted reference modes
  without projecting neutral evidence across unrelated saturated hues. The
  underlying colours are not dimmed. The contextual editor keeps natural row
  heights and scrolls vertically when the split image viewport is short.
- Full-image CUDA jobs use one dedicated worker and coalesce newer requests.
  Per-image node caches are LRU-bounded to three images and 2 GiB of reachable
  CUDA storage; eviction and failures release GPU ownership and lazy CPU mirrors.
  Overlay downloads survive only until the next viewer render. Painted masks
  use immutable applied arrays; categorical peers are copied lazily on the first
  dab that can replace them, rather than when the panel merely opens. Both reference and instance freehand tools use incremental vector
  previews rather than rebuilding the corrected base image during pointer moves.
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
  colour is anchored to the retained outside-dish annulus distribution.
- Automatic foreground colour is now independently anchored by the accepted
  isolated reference-seed components above the ruler. Their individual Lab
  frequencies are compared against the exterior-background distribution and
  combined with seed-scale surface-lightness evidence. The former automatic
  high-confidence self-fit remains only as an explicitly labelled fallback when
  no isolated reference seed survives. Selecting **Foreground colour
  probability** shows the leading starting modes as swatches, hex RGB values,
  frequencies, source type, and source-pixel count in the inspector.
- The independent seed-interior branch remains visible but disabled. It now
  consumes foreground colour probability, learned foreground-noise probability,
  and background evidence; an editable texture weight controls the colour/noise
  blend before background support is incorporated. The
  distance, instance, final boundary, review, measurement, classification,
  aggregation, and output branch is outside the active DAG in **Unused nodes**.
- The application is branded **Seed Fiddle** and uses the seed-and-fiddle-bow
  icon under `seedvision/assets/seed_vision_icon.png`.
- The 11 low-resolution test photographs under `images/` are committed and are
  required by image-backed tests. Generated diagnostics under `artifacts/` are
  ignored.

The full suite passed 156 tests on Python 3.12.10 with PyTorch CUDA on an RTX
3070 on 2026-08-12 after the typed editable-node-wiring and independent
automatic-foreground-reference implementations and visual/performance QA.

## Learned instance-segmentation implementation

Work continues on branch `codex/learned-instance-segmentation`. Two independent
disabled-by-default active-DAG nodes now provide native PyTorch learned
segmentation:

- **U-Net + watershed instances** predicts interior, physical boundary,
  apparent/non-physical coat-pattern boundary, centre plus normalized distance,
  and auxiliary dense-error probability. Its pattern-aware marker watershed can
  also consume applied painted instance IDs.
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
