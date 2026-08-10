# Seed Vision implementation plan

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
Calibration, foreground/background estimation, distance and circle proposals,
shared gradients, boundary tracing, provisional masks, and the fifteen advanced
raster products use a CUDA-first PyTorch backend. Node tensors and intermediate
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
- Explicit dish-layout, seed-scale, foreground-mask, distance-peak,
  Hough-circle, and candidate-fusion stages with visible method explanations
  and per-run diagnostics
- Node-selected access to most implemented algorithm settings, with bounded
  editors, concise Blueprint-style controls on every configurable graph node,
  tooltip descriptions, composite validation, shared edge-field controls, and
  downstream cache invalidation
- Preliminary automatic masks (implemented as provisional watershed masks)
- Selectable proposal, instance-mask, colour-background,
  noise-frequency-background, undirected-edge, directed-edge, and seed-edge
  curve overlays
- Fifteen CUDA-first diagnostic/trait products: soft interior, boundary
  normals, touching split, ellipse support, proposal disagreement, assignment
  confidence, contact graph, illumination decomposition, image quality,
  per-seed radial profile, wrinkling, coat damage, pattern probabilities,
  colour probabilities, and calibration residual risk
- Per-seed provisional trait summaries and lot-level broad class proportions
- Per-image manual background reference points, automatic fallback, and a
  bypassable background-colour/noise branch
- Mask editing and autosave
- Versioned annotation representation

Exit: representative samples can be annotated entirely inside Seed Vision.

Current pilot status: 11 low-resolution images load directly from `images/`;
the application detects colour swatches and the ruler, deskews and
neutral-balances the image, assigns pixels per millimetre, detects the Petri
dish, and creates fused classical review proposals in a background thread. A
movable, zoomable Qt node canvas exposes the full classical proposal path:
dish Hough detection, isolated-reference seed scaling, Lab foreground masking,
distance-transform peaks, Hough circle candidates, and confidence-weighted
fusion. Its inspector explains each method and each parameter inline, while
per-run node status reports the actual threshold, mask coverage, candidate
counts, and fused total. The image viewer can switch between unique spatially contrasting
instance colours, dark-to-light background-colour likelihood, and a continuous
noise-frequency refinement trained from colour-derived pseudo-labels. It also
provides continuous full-palette hue representations of both axial/undirected
and polarity-aware directed edge tangents, sharing one edge-strength map.
The directed field also feeds a seed-edge curve likelihood that combines edge
strength, seed-radius arc continuity, directed tangent agreement, and an
inferred seed interior that darkens toward the boundary. Background and tangent
evidence branch directly from the corrected image; only the curve-radius test
also consumes the global seed scale from identification.
Automatic background colour can be overridden with user-selected reference
patches anywhere on the corrected image; the background branch can also be
bypassed without disabling seed proposals, provisional masks, or edge maps.
Directed tangents use lightness polarity and place the brighter side on their
right. Each overlay is represented by a live-status branch in the pipeline;
selecting an overlay node prepares that layer in Image review. The sparse
16-seed image is the first count regression target. The masks are provisional;
interactive correction and autosave remain to be built.

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
