# Seed Fiddle

> [!CAUTION]
> Most of this project's code was authored by AI and has not yet undergone
> comprehensive human review. Use the software and its analysis results with
> caution, and independently review and validate them before relying on them.

Seed Fiddle is a local PySide6 application for counting, measuring, and broadly
classifying soybean and lupin seeds in calibrated laboratory photographs.

The project now includes a PyTorch CUDA evidence pipeline, a review-oriented
procedural instance-separation node, and selectable analysis overlays in the
desktop interface. See [`PLAN.md`](PLAN.md) for the
agreed scope and delivery sequence.

It also includes disabled-by-default native PyTorch implementations of a
five-logical-head U-Net plus pattern-aware watershed and 2-D StarDist. Their
training, validation, fixture-review, and data-export paths are implemented,
but no checkpoint has been validated for scientific use on real photographs.
See [`docs/LEARNED_INSTANCE_SEGMENTATION.md`](docs/LEARNED_INSTANCE_SEGMENTATION.md)
for the protocol and
[`docs/LEARNED_SEGMENTATION_RESULTS.md`](docs/LEARNED_SEGMENTATION_RESULTS.md)
for the current evidence and limitations.

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

## Learned instance models

Applied, complete seed-instance masks can be exported from **File > Export
applied seed labels for learningâ€¦**. The export stores the exact corrected,
canonical-scale feature stack, display image, labels, species condition, and an
unreviewed manifest record. Interior marks or partly filled seeds are not valid
training masks; update reviewer, capture/lot group, split, and revision metadata
before training.

The launcher provides reproducible headless actions. For example:

```powershell
python .\seed_vision.py --learning-audit D:\data\seed-masks\manifest.json

python .\seed_vision.py --train-learned-model unet_watershed `
  --learning-manifest D:\data\seed-masks\manifest.json `
  --model-output D:\models\unet.pt

python .\seed_vision.py --train-learned-model stardist `
  --learning-manifest D:\data\seed-masks\manifest.json `
  --model-output D:\models\stardist.pt --training-rays 32

python .\seed_vision.py --evaluate-learned-model D:\models\unet.pt `
  --learning-manifest D:\data\seed-masks\manifest.json `
  --learning-split validation --optimize-decoder `
  --evaluation-output D:\reports\unet-validation

python .\seed_vision.py --evaluate-learned-model D:\models\unet.pt `
  --learning-manifest D:\data\seed-masks\manifest.json `
  --learning-split test `
  --decoder-settings D:\reports\unet-validation\evaluation.json `
  --evaluation-output D:\reports\unet-test
```

Decoder search is rejected outside the validation split. Synthetic-data and
unlabelled-fixture commands exist for software/domain-shift checks, but their
reports always remain scientifically invalid. Checkpoint files are ignored by
Git and must be distributed with their immutable manifest, provenance, training
report, and held-out evaluation report.

The window automatically lists supported files in `images/`. Select an image
and choose **Run active pipeline**. The pipeline first detects the 4×6
colour-card grid and ruler independently, estimates their orientations, applies
a planar projective deskew and neutral-swatch colour balance, and resolves the
0-to-terminal ruler-dash span and minor ticks into pixels per millimetre. Dish,
colour/noise, image-quality, and edge diagnostics then run on that corrected
image. When ruler scale is available, the dish circle
bank uses the editable expected 96 mm exterior diameter to reject smaller
circular seed-mass boundaries and constrain the exterior glass edge before
resolving its inner partner. The
image-relative radius controls remain the fallback when physical scale is
unavailable. The complete workflow runs in a worker thread, so
the interface remains responsive.

After analysis, the top-bar **Overlay** chooser groups indented image layers
beneath their owning pipeline-node names and selects the calibrated image and scale bar,
detected swatches, detected ruler, foreground colour probability, background
colour probability, foreground/background noise probability, image-quality
products, 24 directional background texture rays
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
The active edge branch retains continuous shared gradients on CUDA, thins
them with non-maximum suppression and hysteresis, and links compatible pixels
into oriented traces while bridging short gaps and splitting junctions. The
preserved **Directional surface darkness gradients** node samples one-sided
rays over a seed-relative radius and retains separate maximum lightening and
darkening CIE L* slopes plus their query-to-target directions. Its two optional
derivative upper-cutoff nodes set responses above an editable raw L*/pixel
ceiling exactly to zero, deliberately removing strong edge-scale changes from
the weak-surface views. All three nodes now live in **Unused nodes**, are
disabled by default, and contribute no default calculation or overlay.
**Multiscale darkness & colour noise** supplies six non-learned local RMS
energy masks: fine, medium, and coarse bands for L* darkness variation and Lab
a*/b* colour variation. These are separate from the learned foreground/background
noise-probability classifiers. The former distance-candidate, instance, final
boundary-confirmation, review,
measurement, classification, aggregation, and output branch is preserved in
**Unused nodes** but does not calculate or expose overlays by default.
The adjacent opacity slider applies to every overlay. Selecting a node also
places a synchronized node-local overlay chooser directly below its title in
the right inspector; nodes without image products show a disabled placeholder.

Additional CUDA-first analysis products are authored as nodes. The soft
seed-interior probability now receives both foreground colour probability and
the learned foreground-noise probability, blends them with an adjustable
texture weight, and then incorporates inverse background evidence. Other nodes
provide boundary confidence/normals; touching-seed
split likelihood; multiscale ellipse support; proposal-source disagreement;
instance-assignment confidence; an occlusion/contact graph; simple flattened
grayscale plus local-lighting-aware nonlinear shadow/highlight maps; broad
illumination, reflectance, and glare decomposition; image-quality diagnostics;
per-seed radial profiles; wrinkling; coat damage; broad pattern probabilities;
broad colour probabilities; and calibration residual risk. Independent
illumination, image-quality, calibration-residual, foreground/noise, and edge
diagnostics remain in the active graph. Products that depend on the removed
distance branch are retained in the toolbox for future development.

To override the automatic class estimates, run an image once and use the
top-bar **Paint background** and/or **Paint foreground references** controls. Either command
opens a contextual panel over the image with both reference summaries, separate
**Exclude from background** and **Exclude from foreground** masks, clear
buttons, brush controls, and confirmation actions. Exclusion strokes are
negative training examples: separate colour-frequency and texture distributions
are fitted from them and matching evidence is downweighted throughout the image.
The painted coordinates are not overwritten or forced to zero, and exclusions
do not force pixels into the opposite class. Left-drag paints a full-resolution
binary mask. Every painted layer has a direct **Erase** button; the shared
Paint/Eraser buttons select the left-drag operation, while right-drag remains a
temporary eraser shortcut. A
live image-coordinate outline shows the exact brush footprint set by the
radius slider. Painting is only a draft and performs no image analysis.
Choose **Apply reference masks** once both masks are complete to rerun only the
affected nodes and their dependents, or **Revert edits** to restore the last
confirmed masks. **Clear** also edits only the draft until it is applied. Turn
off **Show painted reference/exclusion areas** to inspect an analysis overlay
without the painted colours; seed-instance annotation colours remain visible.

Use the separate **Annotate seed instances** mode to give each known seed a
stable, distinct colour ID. Freehand **Brush** and **Eraser** drags show an
immediate vector stroke, then rebuild only the annotation raster on release;
they do not rebuild the analysis overlays. The assisted tools use a debounced
live preview and click-to-apply interaction: **Trace edge**
sets an initial anchor, previews a magnetic path as the cursor moves, and applies
each segment on the next click; **Snap shape** shows both a nominal reference
cursor and its nearest edge-supported circle/ellipse fit; **Smart fill** previews
its proposed region and can start from an unmarked seed or extend a partial
annotation. Trace and shape tools can ignore tangents or
incorporate the calculated directed or undirected edge tangents with an
adjustable influence. Smart fill compares each candidate with already accepted
touching pixels rather than a fixed starting colour, stops at calculated edges,
never overwrites another seed ID, and offers bounded growth, four/eight-neighbour
connectivity, colour tolerance, edge threshold, and five tunnelling strengths.
Its native floating-range flood compares candidates with accepted neighbours,
not an absolute initial colour.
This locally adaptive rule can cross gradual light/dark seed pattern changes;
tunnelling progressively relaxes the local edge and colour barriers while the
configured radius and pixel limits keep growth bounded.
Each seed ID is shown in a stable, distinct colour;
**New seed** advances to another identity, while the selector permits revising
an existing one. These integer labels are stored independently from foreground
colour references, so they never force foreground probability. Applying them
invalidates **Procedural seed separation** as well as the dormant **Instance
colour masks** branch. Each distinct painted ID becomes an authoritative
procedural watershed marker and suppresses a nearby automatic duplicate without
altering foreground probability. Right-drag remains a temporary eraser for
every annotation tool.

Every painted foreground pixel contributes to a quantized CIE Lab
colour-frequency table, so light and dark seed patterns remain separate instead
of being averaged into a regional colour. Editable frequency influence, colour
tolerance, and bounded refinement controls determine how that evidence affects
similarly coloured pixels throughout the dish. Painted foreground pixels use
the same probability equation as every matching unpainted pixel and are never
forced to probability one. Painted background references remain explicit
semantic constraints. Turn off **Use background colour
analysis** (or bypass the **Background colour probability** node) when no reliable
background reference is available. Independent foreground-noise, image-quality,
and edge diagnostics still run; the colour and noise-frequency background maps
are omitted.

Selecting either the **Background colour probability** or **Foreground colour
probability** node
displays its fitted multimodal membership contours over an HSV-rendered
hue/tint/shade projection like a colour-picker square: white at the top, pure
hues through the middle, and black at the bottom. Contours project across the
unshown saturation dimension, then evaluate every sampled colour with the
unchanged CIE Lab mixture model. The underlying colours are never dimmed by
membership probability.
The foreground diagnostic uses painted modes when supplied and otherwise
summarizes the current automatic high-confidence foreground colours.

## Visual pipeline

The central workspace can show **Image review** and the native Qt **Pipeline**
canvas simultaneously in a splitter, or emphasize either view. The active graph
runs from raw images on the left to calibration and diagnostic products on the
right. Nodes can be moved and
intentionally overlapped. A one-line graph toolbar shows zoom controls and can
automatically arrange the graph into non-overlapping dependency columns with
barycentric ordering to reduce crossings. Its **Unused nodes** toolbox preserves
experimental nodes outside the executable DAG and can explicitly restore one
with its authored wiring. The image viewer has its own fit,
100%, zoom, and percentage controls. Both canvases can be panned and zoomed,
and node colours report idle, running,
complete, warning, planned, bypassed, and failed states.
The inspector's **How it works** explanation starts collapsed whenever a node is
selected and can be expanded with its disclosure button.
Every node has a bottom calculation-time footer. CUDA stages use queued GPU
events and resolve all durations with one synchronization at the end of the run,
so timing does not serialize the pipeline at every node; reused nodes retain
their most recent measured time. During a run, nodes begin blue and independently
turn green as their worker stage completes; progress messages are scoped to the
current image and pipeline revision.

Selecting any node identifies it with a compact **Node:** heading in the right
inspector, offers a collapsed **How it works** explanation, and selects its
corresponding image overlay. Intermediate raster
products remain available in the overlay list. Parameter effects are compact
tooltips on both the field label and editor. Changing a setting immediately
recomputes that node and its true downstream stages while cached independent
upstream evidence is reused. Direct neighbours of the selected node receive a
purple outline. Each configurable node also exposes one to four
high-value Boolean, choice, integer, or floating-point controls directly on its
graph card; the inspector remains the complete settings editor. Cards size to
their actual ports and controls. The complete Settings panel includes a Reset
button that restores the selected node's authored defaults. The
calibration nodes expose neutral-balance enablement, nominal visible ruler span,
minor-tick interval, and maximum accepted deskew. Mask review, trait
classification, aggregation, and export are retained in **Unused nodes** while
they are implemented.

### Active analysis and procedural seed separation

The active layout-to-diagnostic span is represented by these live pipeline nodes:

1. **Layout detection** downsizes the corrected image to at most 1600 px,
   blurs its grayscale representation, and uses gradient-aligned CUDA ring voting to find
   the primary Petri-dish edge. A dense concentric radial profile then resolves the
   lower/inner and upper/outer glass edges. Candidate radii are 16–29% of image height.
   A weak expected position/radius prior selects among detected circles and supplies the
   shown confidence; failure to detect a circle remains a failure. Both edges are shown
   on the layout node, while the outer edge defines every downstream analysis region,
   and the complete vessel extent.
2. **Perimeter background reference** starts an automatic colour-reference
   annulus 0.35 cm beyond the detected outer edge by default. Its independently
   adjustable thickness defaults to 0.5 cm. Both controls use ruler-calibrated
   pixels per millimetre, with the nominal 48 mm dish radius as a fallback when
   ruler scale is unavailable. The dedicated overlay shows the exact buffer and
   band; if too little outer annulus is visible, an orange inside-rim fallback
   preserves both configured distances.
3. **Seed scale estimate** examines the fixed isolated-reference region above
   the ruler in CIE Lab. Colour-difference components are morphologically
   cleaned and filtered by edge contact, area, aspect ratio, and size relative
   to the dish. The corrected median equivalent diameter of up to three accepted
   components becomes the global seed diameter; if none survive, the fallback
   is 16% of dish radius.
4. **Foreground colour probability** retains individual Lab samples from its compact
   rim-adjacent prior and fits a multimodal background distribution without
   allowing in-dish seed colours to redefine it. Weighted Lab distance supplies
   foreground strength. The applied threshold is
   `max(8, Otsu threshold × Foreground threshold)`, followed by seed-scale
   elliptical opening and closing.
   The **Background colour probability** diagnostic consumes the separately
   controlled perimeter reference and also outlines that exact buffered annulus
   (or the inside-rim fallback when necessary).
   Its node inspector shows 25%, 50%, 75%, and 90% fitted membership contours
   over an HSV-rendered hue/tint/shade projection, with learned mode frequencies.
   The **Background noise probability** applies its learned three-band texture
   classifier to that same surrounding annulus and displays the resulting noise
   likelihood alongside the dish-resident refined map.
5. **Foreground noise probability** learns a matching three-band profile from
   foreground-colour pseudo-labels. It continues texture evidence across
   patterned coats without hard-forcing painted reference pixels.
6. **Edge gradients**, **Thinned edge ridges**, and **Oriented edge traces**
   retain the strongest current evidence for visible seed boundaries without
   depending on an unvalidated centre proposal.
7. **Procedural seed separation** fuses foreground colour/noise and inverse
   background into a material likelihood, fills only enclosed seed-sized coat
   holes, and excludes a seed-relative dish margin. Shared edge magnitude,
   sensor/noise transitions, thinned ridges, and local shadow form a physical
   boundary cost. Smoothed material, boundary depth, and annular boundary
   support produce automatic markers; distinct painted instance IDs replace
   nearby automatic markers. OpenCV marker-controlled watershed performs the
   bounded topological partition, and marker/boundary/area evidence supplies a
   per-instance confidence. Six owned overlays expose every stage. GPU inputs
   are resized before this topology-only CPU transfer, and the result is cached
   at the node.
8. The disabled **Directional surface darkness gradients** toolbox node searches
   independent one-sided rays and emits lightening/darkening magnitude and
   direction products. Its lightening and darkening derivative cutoff nodes
   retain only raw slopes at or below their editable L*/pixel ceilings,
   suppressing strong edges. The complete three-node branch is preserved with
   its connections in **Unused nodes**.
9. **Multiscale darkness & colour noise** builds a seed-relative Gaussian
   pyramid and exposes surrounding fine/medium/coarse RMS energy separately for
   L* darkness and Lab chroma. These masks are raw diagnostic evidence and do
   not learn foreground/background classes.

The preserved **Circle candidates**, **Distance-peak candidates**, directional
surface-darkness and derivative upper-cutoff nodes, plus
every active-DAG descendant of the distance node, live in the graph toolbar's
**Unused nodes** toolbox. They contribute no status, overlay, dependency, or
calculation to the current graph. Restoring a node also restores any authored
connections whose two endpoints are currently active; each restored unfinished
node remains disabled until explicitly enabled. If **Circle candidates** is
restored, its CUDA ring bank now consumes the cached **Edge gradients** magnitude
and boundary transitions from **Image-quality diagnostics**' sensor/noise map,
flattened grayscale, local shadow, and local highlight likelihoods through
separate editable weights; it no longer hides a private grayscale-edge
calculation behind those graph inputs. Select a restored **Circle candidates**
node and choose **Move to unused** to remove it and all incident connections;
it can be added again from **Unused nodes** later.

The independent seed-interior branch remains visible but disabled; its authored
inputs include foreground colour, foreground noise, and background evidence.
Disabling
any active node immediately disables its active dependents; toolbox branches
can be rebuilt explicitly as their required upstream nodes are restored.

After a run, active nodes report the selected dish geometry, seed-scale source,
foreground threshold and coverage, colour/noise profile separation, edge-trace
support, and image-quality diagnostics. Their settings remain located on the
stage whose calculation they affect.

Most implemented numeric settings are editable by selecting their owning node.
The inspector exposes the implemented settings on their owning nodes, including:

- dish detection resolution, Hough thresholds, radius bounds, and selection prior;
- reference ROI bounds, colour distance, component filtering, correction, and fallback;
- perimeter-background buffer and reference-band thickness in centimetres;
- foreground background-percentiles, Lab weighting, threshold, and morphology;
- dormant distance-map smoothing, peak neighbourhood/depth, and proposal radius;
- dormant circle edge/accumulator thresholds, edge-magnitude, sensor/noise,
  flattened-grayscale, shadow, and highlight weights, radius bounds, centre
  spacing, and GPU working resolution after
  restoring **Circle candidates** from the toolbox;
- local-lighting field scale, grayscale flattening gain, deviation context,
  nonlinear shadow/highlight thresholds, and transition softness;
- dormant distance/circle fusion confidence after restoring those nodes;
- manual/automatic background sampling, reported colour ranges, texture bands,
  directed-ray geometry, direction integration, and GPU working resolution;
- edge blur, chroma weighting, normalization, and display gamma;
- one-sided surface-gradient blur, ray length, angular/sample resolution,
  normalization, and GPU working size; after restoring the optional cutoff
  nodes, independent lightening/darkening upper L*/pixel cutoffs;
- fine/medium/coarse frequency-noise scales, surrounding RMS context,
  normalization, display gamma, and GPU working size;
- dormant provisional instance-extent limits; and
- procedural material threshold/morphology, dish margin, boundary fusion,
  centre-evidence weights and spacing, marker acceptance, topology resolution,
  and calibrated area confidence limits;
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

The active implemented evidence layers are first-class pipeline nodes:
foreground/background colour and noise probability, shared edge gradients,
undirected/directed tangents, one-sided lightening/darkening surface gradients,
six multiscale darkness/colour noise masks,
thinned ridges, oriented traces, illumination, image quality, and calibration
residuals. The active procedural node consumes those products and exposes
review-oriented seed identities and confidence. The former distance-based
instance, boundary, review, and output nodes remain serialized in the toolbox
alongside the optional strong-slope cutoff views.
The background-colour node is bypassable, and its automatic colour estimate can
be replaced per image by a user-painted binary reference mask in Image review.
Their node headers follow analysis running/completion/failure state. Selecting
one of these nodes selects the corresponding layer for the **Image review** tab.

Background colour probability and **Edge gradients** derive directly from **Deskew & colour
balance**, rather than from seed identification. The shared gradient outputs
feed both tangent displays and the active ridge/trace branch.

The procedural output is not a validated count or reviewed instance mask.
Visual QA across all eleven fixtures found useful separation for pale round
seeds and plausible partitions for densely packed round seeds, but it also
found obvious false splits and merges on strongly bicoloured elongated lupins.
The sparse `IMG_9670c.JPG` fixture visibly contains 16 dish seeds but receives
18 automatic instances. On the RTX 3070 test system, the cached procedural node
itself takes about 0.41–0.87 seconds for the evaluated 856–1758 px dish crops;
the complete active pipeline takes about 2.1–6.3 seconds. These results establish
an honest classical baseline and a useful annotation bootstrap, not production
counting accuracy. High-accuracy automation now requires manually reviewed
instance masks and a small species-conditioned boundary/instance model (for
example U-Net plus watershed or StarDist); a transformer is not justified by
the present dataset size. Patterned, touching samples remain correction and
training data for that model.
Background, foreground, noise, and edge values are image-derived
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
boundary, and advanced evidence analyses use PyTorch CUDA. Authoritative tensors
and intermediate overlays stay on the GPU. The procedural node is the documented
exception: it resizes its required evidence on the tensor device, transfers a
bounded working raster for OpenCV watershed topology, and restores integer
labels for Qt review. Selected overlays, the corrected display image, and
compact result metadata/geometry also transfer to CPU. The right panel reports
the exact active tensor device and any fallback.

## License

Seed Fiddle is available under the [MIT License](LICENSE).
