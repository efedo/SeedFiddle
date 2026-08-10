# Seed Vision

Seed Vision is a local PySide6 application for counting, measuring, and broadly
classifying soybean and lupin seeds in calibrated laboratory photographs.

The project now includes a PyTorch CUDA proposal pipeline and selectable
analysis overlays in the desktop interface. See [`PLAN.md`](PLAN.md) for the
agreed scope and delivery sequence.

## Launch

With compatible dependencies already available:

```powershell
python .\seed_vision.py
```

The launcher can create a project-local virtual environment or install missing
packages into the current interpreter after confirmation. It never creates a
server or bundles the application as an executable.

Python 3.12 x64 is the baseline runtime.

Useful noninteractive checks:

```powershell
python .\seed_vision.py --diagnostics
python .\seed_vision.py --no-bootstrap
python .\seed_vision.py --environment venv --bootstrap auto
python .\seed_vision.py --environment current --bootstrap auto
python .\seed_vision.py --offline
```

`--offline` requires compatible wheel files in `wheels/`.

The window automatically lists supported files in `images/`. Select an image
and choose **Run to seed identification**. The pipeline first detects the 4×6
colour-card grid and ruler independently, estimates their orientations, applies
a planar projective deskew and neutral-swatch colour balance, and resolves the
0-to-terminal ruler-dash span and minor ticks into pixels per millimetre. Dish detection and seed proposals then
run on that corrected image. The complete workflow runs in a worker thread, so
the interface remains responsive.

After analysis, **Viewer overlay** selects the calibrated image and scale bar,
detected swatches, detected ruler, proposal outlines, provisional instance
colour masks, foreground strength/mask, distance transform, separate proposal
branches, background-colour likelihood, 24 directional background texture rays
at the default 15° spacing, their merged likelihood, or undirected/directed
edge tangents. Instance colours are unique
within the image and assigned to make spatial neighbours contrast strongly.
Background match is deliberately
displayed dark (high match) to light (low match). The refined layer uses
confident pixels from the colour layer as per-image pseudo-labels, learns
fine/medium/coarse band-pass energy profiles for background and non-background,
evaluates their continuation along one-sided rays, and merges the directional
maps with the selected mean, maximum, minimum, or median rule. Both edge overlays use brightness for
maximum multichannel edge strength. The undirected option maps axial tangent
orientation across the complete hue wheel, with 0° and 180° equivalent. The
directed option uses 0° and 360° equivalence and orients tangents so that the
brighter side of an edge lies on their right. Directed polarity is naturally
less stable for edges whose two sides have nearly equal perceptual lightness.
The seed-boundary branch retains continuous shared gradients on CUDA, thins
them with non-maximum suppression and hysteresis, and links compatible pixels
into oriented traces while bridging short gaps and splitting junctions. Dense
radius/arc sampling, normal-based centre voting, provisional circle and ellipse
fits, foreground/background side agreement, and optional inward-lightness
polarity then confirm likely boundaries. Troubleshooting overlays expose every
stage, including rejection reasons and fitted vector geometry.
The opacity slider applies to every overlay.

Fifteen additional CUDA-first analysis products are represented as live nodes:
soft seed-interior probability; boundary confidence/normals; touching-seed
split likelihood; multiscale ellipse support; proposal-source disagreement;
instance-assignment confidence; an occlusion/contact graph; illumination,
shadow, reflectance, and glare decomposition; image-quality diagnostics;
per-seed radial profiles; wrinkling; coat damage; broad pattern probabilities;
broad colour probabilities; and calibration residual risk. Selecting a node
shows its primary overlay. Troubleshooting intermediates—including individual
quality maps and every colour/pattern class probability—remain selectable in
the overlay list. Broad colour/pattern proportions and mean condition scores
are also accumulated per provisional seed.

To override the automatic class estimates, run an image once and use **Paint
background** and/or **Paint foreground**. Left-drag paints a full-resolution
binary reference mask. The Paint/Eraser buttons select the left-drag operation,
while right-drag remains a temporary eraser shortcut. A live image-coordinate
outline shows the exact brush footprint set by the radius slider. Painting is
only a draft and performs no image analysis.
Choose **Apply reference masks** once both masks are complete to rerun only the
affected nodes and their dependents, or **Revert edits** to restore the last
confirmed masks. **Clear** also edits only the draft until it is applied.

Every painted pixel contributes to a robust multimodal CIE Lab distribution.
Colour modes are weighted by their frequency within the painted mask, with
editable frequency influence, colour tolerance, clustering, and iterative
refinement controls on the foreground and background nodes. Thus references
affect similarly coloured pixels throughout the dish, while the exact painted
pixels remain hard semantic constraints. Turn off **Use background colour
analysis** (or bypass the **Background colour** node) when no reliable
background reference is available. Seed proposals, provisional masks, and edge
diagnostics still run; the colour and noise-frequency background maps are
omitted.

Selecting either the **Background colour** or **Foreground segmentation** node
displays its fitted multimodal membership contours over a CIE Lab gamut slice.
The foreground diagnostic uses painted modes when supplied and otherwise
summarizes the current automatic high-confidence foreground colours.

## Visual pipeline

The central workspace can show **Image review** and the native Qt **Pipeline**
canvas simultaneously in a splitter, or emphasize either view. The graph runs
from raw images on the left to final output on the right. Nodes can be moved,
the canvas can be panned and zoomed, and node colours report idle, running,
complete, warning, planned, bypassed, and failed states.
The inspector's **How it works** explanation starts collapsed whenever a node is
selected and can be expanded with its disclosure button.
Every node has a bottom calculation-time footer. CUDA stages use queued GPU
events and resolve all durations with one synchronization at the end of the run,
so timing does not serialize the pipeline at every node; reused nodes retain
their most recent measured time. During a run, nodes begin blue and independently
turn green as their worker stage completes; progress messages are scoped to the
current image and pipeline revision.

Selecting any node opens a visible **How it works** explanation in the right
inspector and selects its corresponding image overlay. Intermediate raster
products remain available in the overlay list. Parameter effects are compact
tooltips on both the field label and editor. Changing a setting immediately
recomputes that node and its true downstream stages while cached independent
upstream evidence is reused. Each configurable node also exposes one to four
high-value Boolean, choice, integer, or floating-point controls directly on its
graph card; the inspector remains the complete settings editor. Cards size to
their actual ports and controls, and the canvas places or nudges them into the
nearest non-overlapping vertical slot. The
calibration nodes expose neutral-balance enablement, nominal visible ruler span,
minor-tick interval, and maximum accepted deskew. Mask review, trait
classification, aggregation, and export remain visible as planned stages while
they are implemented.

### How seed proposals are produced

The formerly compressed layout-to-identification span is represented by these
live pipeline nodes:

1. **Layout detection** downsizes the corrected image to at most 1600 px,
   blurs its grayscale representation, and uses gradient-aligned CUDA ring voting to find
   the primary Petri-dish edge. A dense concentric radial profile then resolves the
   lower/inner and upper/outer glass edges. Candidate radii are 16–29% of image height.
   A weak expected position/radius prior selects among detected circles and supplies the
   shown confidence; failure to detect a circle remains a failure. Both edges are shown
   on the layout node, while the outer edge defines every downstream analysis region,
   the complete vessel extent, and the start of the outside-background sampling band.
2. **Seed scale estimate** examines the fixed isolated-reference region above
   the ruler in CIE Lab. Colour-difference components are morphologically
   cleaned and filtered by edge contact, area, aspect ratio, and size relative
   to the dish. The corrected median equivalent diameter of up to three accepted
   components becomes the global seed diameter; if none survive, the fallback
   is 16% of dish radius.
3. **Dish foreground mask** estimates dish background from low-chroma,
   sufficiently light pixels inside the configured outer-rim inset. Weighted Lab
   distance supplies foreground strength. The applied threshold is
   `max(8, Otsu threshold × Foreground threshold)`, followed by seed-scale
   elliptical opening and closing.
   The **Background colour** diagnostic outlines the exact outer-rim sampling
   annulus used for its initial prior (or the inside-rim fallback when necessary).
   Its node inspector shows 25%, 50%, 75%, and 90% fitted membership contours
   over a local CIE Lab colour-gamut slice, with learned mode frequencies.
   The **Background noise profile** applies its learned three-band texture
   classifier to that same surrounding annulus and displays the resulting noise
   likelihood alongside the dish-resident refined map.
4. **Distance-peak candidates** finds local maxima of the foreground distance
   transform. This can suggest multiple centres inside touching foreground.
   In parallel, **Circle candidates** applies a normalized CUDA convolutional ring bank with
   candidate radii 22–62% of the global diameter.
5. **Seed identification** confidence-weights nearby circle and distance
   candidates into one proposal, retains sufficiently separated candidates,
   and sorts the provisional results spatially.

After a run, these nodes report the selected dish geometry, seed-scale source,
actual foreground threshold and coverage, both candidate counts, and fused
proposal count. Their settings are located on the stage they affect:
**Reference correction** on Seed scale estimate; **Dish area used** and
**Foreground threshold** on Dish foreground mask; **Circle strictness** on
Circle candidates; and **Duplicate merge distance** on Seed identification.

Most implemented numeric settings are editable by selecting their owning node.
The inspector exposes the implemented settings on their owning nodes, including:

- dish detection resolution, Hough thresholds, radius bounds, and selection prior;
- reference ROI bounds, colour distance, component filtering, correction, and fallback;
- foreground background-percentiles, Lab weighting, threshold, and morphology;
- distance-map smoothing, peak neighbourhood/depth, and proposal radius;
- circle edge/accumulator thresholds, radius bounds, centre spacing, and GPU working resolution;
- fusion distance and candidate confidence weights;
- manual/automatic background sampling, reported colour ranges, texture bands,
  directed-ray geometry, direction integration, and GPU working resolution;
- edge blur, chroma weighting, normalization, and display gamma;
- watershed marker and instance-extent limits; and
- ridge NMS/hysteresis, oriented trace linking and gap controls, dense radius
  and arc sampling, circle/ellipse fits, centre voting, semantic sides,
  optional lightness boost, and rejection confidence; and
- CUDA working resolution/device policy plus seed-relative controls for all
  fifteen advanced diagnostic and trait products.

Every editor has a bounded range and a tooltip explanation. Composite settings
are checked before acceptance: minimum/maximum radii and extents must remain
ordered, reference rectangles must remain valid, and noise pseudo-label cutoffs
cannot cross. Changing a setting invalidates cached results on the affected
downstream branch. Directed and undirected tangents share one continuous
gradient computation. Final-fit changes reuse cached GPU ridges and traces;
trace changes reuse cached GPU ridges.

The calibration path is represented by separate **Colour-card swatches**,
**Ruler detection**, **Deskew & colour balance**, and **Absolute ruler scale**
nodes. The colour-card detector rectifies the outer card, searches RGB edge
support around every expected swatch boundary on GPU, and pools those responses
across shared grid rows and columns. This locks the displayed quadrilaterals to
the printed solid edges while requiring only one compact profile transfer back
to the CPU. Selecting the nodes prepares the matching reference diagnostic or calibrated
image in **Image review**. Ruler endpoints represent the large 0 and configured
terminal scale dashes rather than the ends of the plastic ruler body. The result is stored as pixels per millimetre
and displayed as a physical scale bar; batch CSV reports include swatch count,
deskew angle, scale, and scale confidence.

The implemented diagnostic layers are also first-class pipeline nodes:
background colour, noise-frequency refinement, instance colour masks, shared
edge gradients, undirected/directed tangents, thinned ridges, oriented traces,
seed-boundary confirmation, and the fifteen CUDA-first
analysis products described above.
The background-colour node is bypassable, and its automatic colour estimate can
be replaced per image by a user-painted binary reference mask in Image review.
Their node headers follow analysis running/completion/failure state. Selecting
one of these nodes selects the corresponding layer for the **Image review** tab.

Background colour and **Edge gradients** derive directly from **Deskew & colour
balance**, rather than from seed identification. The shared gradient outputs
feed both tangent displays and the ridge/trace branch. **Seed-boundary
confirmation** also receives global seed scale, background/foreground
probability, and provisional instances as soft, inspectable evidence.

These are deliberately approximate review proposals and provisional watershed
masks, not validated counts or reviewed instance masks. Packed, touching
samples are expected to need corrections and will supply the training data for
the learned instance model. Background and edge values are image-derived
likelihood scores rather than probabilities calibrated against labelled data.
The noise profile falls back to the colour likelihood when the image does not
provide enough confident examples of both classes.

## Pilot batch report

To generate a CSV summary and annotated images for every pilot image:

```powershell
.\.venv\Scripts\python.exe .\scripts\analyze_pilot.py .\images
```

Outputs are written to `artifacts/pilot/baseline/`.

## Dependency tiers

PySide6, NumPy, OpenCV, and PyTorch are core runtime dependencies. The bootstrap
manifest installs PyTorch from its CUDA wheel index. The ordinary
`opencv-python` wheel is used for image decoding and a few compact compatibility
operations. Full-raster calibration, proposal, foreground/background, edge,
boundary, mask, and advanced analyses use PyTorch CUDA. Authoritative tensors
and intermediate overlays stay on the GPU. Only a selected Qt overlay, the
corrected display image, or compact result metadata/geometry is transferred to
CPU. The right panel reports the exact active tensor device and any fallback.
