# Seed Fiddle implementation plan

## Objective

Build an on-premises desktop application that counts, measures, and broadly
classifies soybean and lupin seeds in calibrated laboratory photographs. Each
photograph contains one known species. Touching and partially overlapping seeds
must be handled as separate visible instances whenever a person can distinguish
them.

Initial species:

- Soybean
- *Lupinus mutabilis*
- *Lupinus polyphyllus*
- *Lupinus mexicanus*

## Deployment contract

- `seed_vision.py` is the only launch point.
- The application is unbundled Python source; it is never packaged as an
  executable and has no installer.
- PySide6 provides the desktop interface.
- No server, listening port, cloud service, telemetry, automatic update, or
  mandatory network access is used.
- Supporting Python modules, configuration, model weights, and optional offline
  wheels live beside the launch file.
- A project-local `.venv` is optional. On first launch, the standard-library-only
  bootstrapper can either create/use `.venv` or install missing dependencies into
  the current interpreter, after explicit confirmation.
- The bootstrapper prefers a supplied local wheelhouse and never silently falls
  back to the Python user-site directory.
- When PyTorch is not already supplied, online bootstrap uses the official CUDA
  12.6 wheel index; a compatible existing PyTorch build is retained.
- Python 3.12 x64 is the baseline runtime. Python 3.13 and 3.14 x64 may be used
  when the complete pinned dependency set has been validated with them.

## Product workflow

1. Create or open a local project.
2. Import one or more original-resolution photographs.
3. Assign image metadata: species, lot, capture information, and notes.
4. Detect the scale and colour reference.
5. Rectify perspective and normalize colour.
6. Segment every visible, distinguishable seed into an instance mask.
7. Review and edit masks: add, delete, split, merge, and redraw.
8. Calculate measurements for complete masks.
9. Classify visible-face coat traits, wrinkling, and damage.
10. Review low-confidence or ungradable seeds.
11. Export counts, proportions, measurements, labels, and annotated images.
12. Retain corrections as versioned training data.

## Trait schema v0.1

Common per-seed fields:

- Instance identifier and visible mask
- Known species
- Primary coat colour
- Pattern and optional secondary colour
- Wrinkling: 0 none, 1 slight, 2 clear, 3 severe, or ungradable
- Damage flags: cracking, missing/peeling coat, chipping/breakage, discoloration
- Global visible-damage score: 0 sound, 1 minor, 2 moderate, 3 severe, or
  ungradable
- Occlusion: none, touching only, partial, or heavy
- Measurement validity
- Model confidence and review state

Soybean begins with yellow, greenish, green/"blue", red, brown, white, black,
and other as primary colours. Pattern fields avoid creating a separate combined
class for every bicolour combination. Lupin vocabularies are configured per
species and include other/unknown.

## Measurements

For complete seed masks:

- Maximum and minimum Feret diameters
- Projected area
- Perimeter
- Equivalent diameter
- Aspect ratio
- Circularity
- Roundness
- Solidity/convexity

Touching seeds remain measurable when their complete boundaries are separated.
Partially occluded seeds are counted but complete-shape measurements are withheld.
All measurements and classifications describe the visible face only.

## Reporting rules

- Trait proportions use the number of seeds gradable for that trait as their
  denominator.
- Ungradable and uncertain seeds are reported separately.
- Every percentage includes coverage, for example: "40% white; colour gradable
  for 97% of visible seeds."
- Summary statistics distinguish all visible seeds from the subset eligible for
  physical measurements.

## Technical architecture

```text
seed_vision.py                 standard-library launcher/bootstrapper
seedvision/bootstrap.py       environment checks and dependency provisioning
seedvision/application.py     delayed GUI entry
seedvision/pipeline/          typed DAG, parameters, status, and invalidation
seedvision/ui/                PySide6 image review and native node canvas
seedvision/calibration/       reference detection and calibration
seedvision/segmentation/      seed instance segmentation
seedvision/visualization/     lazy CUDA diagnostic, boundary, and trait maps
seedvision/classification/    species-conditioned trait models
seedvision/measurement/       calibrated geometry
seedvision/persistence/       local project and SQLite storage
seedvision/export/            CSV, JSON, and annotated imagery
```

GPU work runs outside the Qt GUI thread and communicates through Qt signals.
Calibration, foreground/background colour and noise estimation, shared gradients,
edge tracing, image-quality products, and the retained experimental analysis
stages use a CUDA-first PyTorch backend. Node tensors and intermediate
overlays remain on GPU and are reused by downstream stages; only a selected Qt
overlay, the corrected display image, or compact metadata/geometry is
materialized on CPU. OpenCV remains the image decoder and compatibility layer,
so no custom CUDA-enabled OpenCV build is required.

### Vessel-layout extension path

The current layout result is explicitly tagged as `petri_dish` and preserves
both concentric physical glass edges. Downstream stages use the upper/outer rim
for the seed-analysis region, complete vessel extent, and outside-background
sampling region instead of assuming that every future vessel is one circle.

Future layout work will introduce a common vessel-geometry interface with those
regions plus vessel-specific overlay primitives. Detector dispatch can then add:

- blue weigh boats, using colour-assisted polygon and rounded-corner segmentation;
- watch glasses, using paired circular, elliptical, or general conic glass-edge fits;
- seeds resting on paper, using an unbounded surface with an explicit analysis region.

These alternative detectors are planned extension points only. The current
implementation deliberately selects the dual-edge Petri-dish detector.

## Delivery phases

### Phase 1 — application foundation (complete)

- Launcher, diagnostics, optional environment bootstrap
- Modular PySide6 shell and zoomable image canvas
- Local project conventions and configuration loading
- Automated tests for bootstrap decisions without installing packages

Exit: `python seed_vision.py` can validate/provision its environment and open the
desktop shell.

### Phase 2 — pilot image and annotation workflow (in progress)

- Import and preserve original images
- Species and lot metadata
- Native Qt visual pipeline from raw images to final output
- Typed node parameters, status, per-node CUDA/CPU calculation timing, and
  downstream cache invalidation
- Independent direct-input contracts for all 53 active/toolbox nodes, with
  corrected-image, dish-region, physical/seed scale, mask, proposal, and
  reference dependencies represented explicitly and cache invalidation verified
  against the graph
- Explicit dish-layout, seed-scale, symmetric foreground/background colour and
  noise-probability stages, plus shared edge-gradient/ridge/trace diagnostics
  with visible method explanations and per-run diagnostics
- Image-local multi-prototype pattern matching from all material, boundary, and
  instance-reference sublayers, with separate Background, Foreground, Other,
  Physical-edge, and Non-edge banks; global probability rasters; and a full-pane
  grouped collage of every retained prototype
- Toolbox-preserved, disabled one-sided maximum lightening/darkening
  surface-slope diagnostics and derivative upper cutoffs, plus active raw
  fine/medium/coarse surrounding RMS masks for darkness and Lab colour noise
- Node-selected access to most implemented algorithm settings, with bounded
  editors, concise Blueprint-style controls on every configurable graph node,
  tooltip descriptions, composite validation, shared edge-field controls, and
  downstream cache invalidation
- Review-oriented procedural automatic masks with explicit confidence; the
  former distance-based masks remain in the unused-node toolbox
- Selectable foreground/background colour and noise, image-quality,
  undirected-edge, directed-edge, ridge, and trace overlays
- Fifteen authored CUDA-first diagnostic/trait products, with unfinished
  distance-dependent products retained outside the active DAG: soft interior
  (fusing foreground colour, learned foreground noise, and background evidence), boundary
  normals, touching split, ellipse support, proposal disagreement, assignment
  confidence, contact graph, illumination decomposition, image quality,
  per-seed radial profile, wrinkling, coat damage, pattern probabilities,
  colour probabilities, and calibration residual risk
- Dormant per-seed provisional trait summaries and lot-level broad class proportions
- Per-image manual background/foreground references, independent exclusions
  for each probability layer, painted-overlay visibility, automatic fallback,
  and a bypassable background-colour/noise branch
- Separate per-image foreground-reference masks and distinct-colour seed-instance
  annotations, each with independent draft/apply/revert state; freehand,
  live-preview/click-to-apply magnetic edge tracing, nominal-cursor plus
  edge-supported circle/ellipse snapping, adaptive neighbour-relative smart fill
  from marked or unmarked seeds with bounded tunnelling, and erasing are available;
  freehand drags use a lightweight vector preview and one annotation-raster
  refresh on release, while assisted previews operate on local cursor regions;
  applied instance IDs constrain the active procedural watershed as
  authoritative markers and also constrain the dormant provisional branch
- Mask editing and autosave
- Versioned annotation representation

Exit: representative samples can be annotated entirely inside Seed Fiddle.

Current pilot status: 11 low-resolution images load directly from `images/`;
the application detects colour swatches and the ruler, deskews and
neutral-balances the image, assigns pixels per millimetre, and detects both
physical Petri-dish edges in a background thread. The active DAG now continues
from validated calibration and diagnostic evidence—symmetric foreground/background
colour and three-band noise probabilities, illumination/image-quality products,
simple grayscale flattening, nonlinear local shadow/highlight products, and a
shared CUDA edge-gradient/ridge/trace path—into a review-oriented procedural
watershed with explicit instance confidence. Independent active diagnostics also
provide one-sided maximum lightening/darkening L* surface slopes and six raw
multiscale darkness/colour noise masks. The editable strong-slope cutoff views
remain authored but have moved to the unused-node toolbox.
Distance-peak candidates and
all of their DAG descendants—identification, provisional instances, final
boundary confirmation, review, measurement, classification, aggregation, and
output—are preserved with their wiring in the unused-node toolbox. Circle
candidates are preserved there independently. When explicitly restored, the
circle bank consumes cached shared edge magnitude and sensor/noise-boundary
evidence plus flattened-grayscale, local-shadow, and local-highlight boundaries
through separate weights. The user can move the restored node back to the
toolbox and add it again later; this experiment does not make circle candidates
part of the default separation workflow.

Foreground/background reference masks and their independent exclusion masks
have per-image draft/apply/revert state. Painted foreground colours remain
individual Lab frequency bins and never force the probability under the brush
to one. Painted reference/exclusion colours can be hidden while inspecting an
overlay. Seed-instance IDs remain separate from foreground references. Their
annotation mode supports freehand marks, magnetic edge tracing, shape snapping,
and locally adaptive smart fill for partially annotated patterned seeds. The
active **Procedural seed separation** node now implements the classical proof
of concept: colour/noise material fusion, scale-aware coat-hole filling, a
fused edge/sensor/ridge/shadow boundary cost, material/depth/ring centre
markers, marker-controlled watershed, and per-instance confidence. Painted
instance IDs are authoritative markers. The node transfers only GPU-resized
bounded rasters for the CPU topology step and caches the full result.

### Procedural validation outcome and learned-model milestone

All eleven fixtures were rendered as corrected image, material evidence/mask,
boundary cost, centre likelihood, and coloured instance overlay. Pale round and
dense round samples are useful annotation starting points. Strongly bicoloured
elongated lupins still have obvious coat-pattern false splits and contact merges;
the sparse 16-seed regression image produces 18 automatic instances. Adjusting
material morphology, marker spacing, centre evidence, orientation voting, and
area priors did not remove this ambiguity consistently across species. The
optimized node takes about 0.41–0.87 seconds per 856–1758 px dish crop on the
RTX 3070.

The next milestone is therefore:

1. Review and correct procedural masks in representative isolated, touching,
   patterned, dense, and partially occluded strata.
2. Persist those masks as versioned training/validation data with species and
   review state.
3. Train a small species-conditioned boundary/instance model—initially U-Net
   plus seeded watershed or StarDist—using tiled full-resolution images and
   realistic synthetic contacts. Do not start with a transformer given the
   present data volume.
4. Evaluate count error, boundary F1/IoU, split/merge error, confidence
   calibration, and correction time on held-out lots.
5. Keep the procedural result as a transparent fallback and annotation
   bootstrap; replace automatic count reporting only when the learned model
   meets agreed acceptance criteria.

### Phase 3 — calibration and measurements (started)

- Detect the supplied 4×6 colour reference (implemented)
- Detect ruler location and long-axis orientation independently (implemented)
- Combine reference orientations for rotational deskew (implemented)
- Neutral-swatch colour-balance correction (implemented)
- Ruler tick/known-span scale assignment in pixels per millimetre (implemented)
- Full projective and target-value colourimetric correction (pending reference specification)
- Calibrated measurements and validity rules
- Calibration diagnostics and failure states

Exit: unobscured seed measurements reproduce a small manual reference set within
an agreed tolerance.

Mask geometry calculations are implemented for maximum/minimum Feret diameter,
area, perimeter, equivalent diameter, aspect ratio, circularity, roundness,
solidity, and convexity. Automatic calibration now feeds a physical scale into
the corrected image and exposes it to these measurement functions. All 11 pilot
images resolve a ruler scale; the half- and full-resolution groups produce the
expected approximately twofold pixels-per-millimetre separation.

### Phase 4 — segmentation proof of concept

- Compare controlled-image classical segmentation with learned instance
  segmentation
- Tile large images
- Train on corrected real masks plus realistic synthetic overlaps
- Evaluate isolated, touching, and overlapping strata separately

Exit: useful counting accuracy with difficult instances visibly referred for
review.

### Phase 5 — trait models

- Calibrated soybean colour baseline
- Species-specific lupin vocabularies derived from real images
- Wrinkling, damage flags, and global damage models
- Confidence calibration and ungradable handling

Exit: broad trait proportions meet pilot acceptance criteria.

### Phase 6 — validation and lab release

- Hold out complete lots and capture sessions
- Validate count error, trait macro-F1, measurement error, and review time
- Freeze dependency/model versions and assemble an offline wheelhouse
- Document environment creation and recovery

Exit: reproducible on-premises lab workflow with no installation or server
requirement.

## Immediate data milestone

The 11 low-resolution pilot images establish the controlled layout and span
sparse through extremely packed samples. Next, add original-resolution images
covering soybean colour groups, all three lupin species, defects, and difficult
overlaps. Image-to-species identities and the colour-card/ruler specifications
will be used to draft the annotation guide and validate calibration.
