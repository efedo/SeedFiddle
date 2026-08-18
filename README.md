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

The complete desktop workflow is under **Learning**:

1. Run the image analysis, choose **Annotate seed instances**, optionally
   initialize the draft from a procedural/U-Net/StarDist result, and correct
   every seed. Every visible seed must have one ID; interior scribbles and
   partly filled seeds are not training masks.
2. **Load reference mask…** in the seed-annotation panel first looks for a
   source-bound mask in `seed-instance-references/`. It verifies the source and
   mask SHA-256 digests, then warps categorical IDs into the current corrected
   image with nearest-neighbour interpolation and installs an undoable draft.
   **Choose mask file…** explicitly selects a corrected-coordinate PNG, TIFF,
   or NPZ even when a matching bundled reference exists.
   **Save applied seed-label mask…** preserves the corrected-coordinate
   `uint16` draft for later editing. The repository bundle covers all eleven
   committed photographs, including the two isolated reference seeds outside
   each dish. Its entries are deliberately marked `reviewed: false`: they are
   machine-prepared pre-annotations that must be corrected and accepted at full
   resolution before training or scientific evaluation.
3. **Export applied labels to learning datasetâ€¦** stores the canonical-scale
   feature stack, display image, labels, species condition, lot/capture group,
   train/validation/test split, annotation revision, reviewer, and review state
   in a versioned manifest. Seed Fiddle prevents one group from crossing
   splits. Unreviewed exports are retained for correction but rejected by the
   trainer.
4. **Audit learning datasetâ€¦** checks files, dimensions, IDs, disconnected
   masks, review coverage, and group leakage.
5. **Train or refine U-Net / StarDistâ€¦** trains a new checkpoint or continues
   from an explicitly selected compatible checkpoint on the serial CUDA
   worker. It requires reviewed training and independently grouped validation
   samples, saves the best validation-loss checkpoint plus its JSON report,
   can be cancelled between batches, and can activate that checkpoint in the
   corresponding pipeline node.

The U-Net's interior, physical boundary, centre, distance, and uncertainty
targets are derived from the reviewed instance masks. The physical target is
the instance contour; its apparent coat-pattern boundary target uses only strong
edge/ridge candidates safely inset from that contour. Flat interior and the
uncertainty band are excluded by the sparse validity mask rather than fabricated
as negative labels.
StarDist object and radial targets are derived directly from the same reviewed
instance masks. Neither architecture learns from the painted foreground or
background reference layers as labels; those are inputs to the exported feature
stack.

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
products, the background texture model's directionally integrated likelihood,
or undirected/directed
edge tangents. Instance colours are unique
within the image and assigned to make spatial neighbours contrast strongly.
Background match is deliberately
displayed dark (high match) to light (low match). The refined layer uses
confident pixels from the colour layer as per-image pseudo-labels, learns
fine/medium/coarse band-pass energy profiles for background and non-background,
evaluates their continuation along one-sided rays, and merges the directional
maps with the selected mean, maximum, minimum, median, or first-tertile rule.
Foreground noise defaults to the exact one-third quantile, so roughly two
thirds of ray directions must retain at least the reported support; background
noise retains its maximum default. The bounded CUDA implementation interpolates
two order statistics without sorting a second full directional raster bank.
Both edge overlays use brightness for
maximum multichannel edge strength. The undirected option maps axial tangent
orientation across the complete hue wheel, with 0° and 180° equivalent. The
directed option uses 0° and 360° equivalence and orients tangents so that the
brighter side of an edge lies on their right. Directed polarity is naturally
less stable for edges whose two sides have nearly equal perceptual lightness.
The active edge branch retains continuous shared gradients on CUDA, thins
them with non-maximum suppression and hysteresis, and links compatible pixels
into oriented traces while bridging short gaps and splitting junctions. Its
convexity policy can leave legacy tangent-only links unchanged, prefer
single-turn C-shaped gap continuations, or require every non-ambiguous gap
link to avoid an S-shaped inflection. Both clockwise and counter-clockwise
curvature are accepted. The
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
broad colour probabilities; and calibration residual risk. Calibration residual
risk and products that depend on the removed distance branch are retained in
the unused-node toolbox for future development. Independent illumination,
image-quality, foreground/noise, and edge diagnostics remain active.

Reference annotation is divided into two independent full-resolution layers:

1. **Material references** is one categorical mask with mutually exclusive
   **Background**, **Foreground**, and **Other** classes. Painting one class at
   a pixel removes either of the other two there. Other means neither ordinary
   dish background nor seed foreground; it supplies a competing learned model
   to both fitted colour/texture classes instead of hard-setting their output
   pixels. It reduces a class only where Other fits better, so glass or neutral
   colours shared with a positive class cannot veto legitimate evidence.
2. **Seed instance annotations** is the separate labelled integer layer
   described below; every seed receives its own stable colour ID. Its viewer
   overlay is drawn from viewport-requested 512-pixel tiles at one mask pixel
   per image pixel, so zooming and panning never expose the former 2,048-pixel
   display proxy. Tile-local indexed palettes retain exact categorical edges
   while a dual 512-entry/48 MiB LRU bound prevents image size from creating an
   unbounded display cache.

There is no separate painted boundary-reference layer. A complete applied seed
instance supplies its one-pixel physical contour automatically. Strong
edge/ridge candidates safely inset from that contour supply sparse
**Non-physical edge** examples; ordinary flat seed interior is left unlabelled.
This keeps the two classes geometrically consistent with the instance mask and
prevents a stale manual boundary raster from disagreeing with it.

The active **Reference texture prototypes** node turns those reviewed examples
into an image-local, non-neural pattern matcher. Its independent capacity
controls default to 64 coverage-preserving medoids for each material class
(Background, Foreground, and Other) and 256 for each instance-derived edge
class (Physical edge and Non-physical edge), instead of averaging a class into
one texture. The edge capacity can be increased to 1,024; fitting assignments
and full-image evaluation are chunked so this does not allocate a
samples-by-prototypes or prototypes-by-full-raster tensor. Material prototypes
combine Lab colour, fine/medium/coarse darkness and colour energy, local
residuals, edge magnitude, ridges, and local edge density. Edge prototypes
instead use narrow tangent-aligned strips with separately pooled interior,
centre-edge, and exterior regions, signed interior-minus-exterior Lab contrast,
local residuals, edge/ridge support, and axial tangent coherence. Annotated
instance geometry supplies the true outward normal while fitting a physical
edge, including opposite normals at a contact between two seed IDs. Global
inference never receives that annotation geometry: it uses image-derived
tangents and evaluates both normal polarities. Material matching retains its
independent 960-pixel default working limit, while edge matching adaptively
targets 28 working pixels per seed diameter up to its separate 2,048-pixel
default hard cap so small seeds retain substantially more boundary detail. The
node evaluates every prototype throughout the dish on CUDA and publishes
Foreground/seed-surface, Background, Other, Physical-edge, and Non-physical-edge
likelihoods. Competing classes attenuate a positive match only when they fit
more specifically, and reviewed coordinates are never assigned an output
probability directly.

Selecting the node exposes a full-image-pane **Reference texture prototype
collage** containing every retained medoid, grouped and colour-coded by class,
with cluster support percentages and source reference counts. The squares are
context thumbnails rather than spatial templates. Edge thumbnails mark the
actual centre sample in yellow and the two polarity-neutral side strips in cyan.
Its three material likelihood overlays remain available beside the existing
physical/non-physical overlays. Separate maximum material and edge prototypes,
minimum cluster support, fit iterations, similarity tolerance, seed-relative
context, thumbnail size, ridge influence, strip geometry, and separate
material/edge working dimensions are editable and computationally active. With no applied
references the expensive feature fit is skipped. When no instance annotations
exist, the physical/non-physical classifier is neutral: unknown regions retain
neutral generic edge/ridge support in the procedural separator, but are not
misrepresented as learned physical-boundary evidence. This is per-image procedural adaptation, not a
trained cross-image model; it is intended to improve annotation bootstrapping
and provide inspectable evidence for later learned-model training.

Applied seed-instance annotations always provide the boundary-class examples.
Each labelled instance's one-pixel contour becomes **Physical edge** evidence;
only strong edge/ridge candidates beyond the configurable safe interior buffer
become **Non-physical edge** evidence. The intervening uncertainty band and
unstructured interior remain unlabelled. These masks are regenerated from the
saved instance IDs and current image evidence and are never stored as a second,
potentially stale annotation layer.

The top-bar **Material references** and **Annotate seed instances** commands
open the applicable compact contextual controls over
the image. Drag the **Move painting controls** bar to reposition this panel.
Repeated explanatory text is kept in tooltips. Shared **Paint**,
**Eraser**, **Clear layer**, brush-radius, **Apply**, and **Revert** controls
replace per-class editing rows; right-drag remains a temporary eraser. A live
image-coordinate outline shows the exact brush footprint. Freehand
reference strokes use lightweight vector previews and rebuild their mask
overlays only once on release. Painting remains a draft and performs no image
analysis until **Apply + save**. The always-visible **Show: Materials | Seed
instances** checkboxes independently hide either painted group
without changing, applying, saving, or invalidating their data. For example,
leave only Seed instances checked to review labels without material marks.
Clearing the selected class returns the shared tool to Paint so the empty layer
can immediately be redrawn.

The fixed panel header also provides a context-aware **Undo** button
(`Ctrl+Z`). Seed Fiddle retains the latest 20 unapplied commands for each image,
independently for the categorical reference layers and seed-instance labels.
One freehand drag, assisted-tool click, clear command, imported label map, or
automatic starting draft is one undo step. Undo restores mutually exclusive
peer material classes together (Background/Foreground/Other), including the
previous seed-label provenance. **Apply + save** and **Revert**
establish a new history boundary; disk-restored references begin with an empty
queue.

**Apply + save** confirms the current material or seed-instance draft and
atomically stores both applied annotation layers together in the
ignored `projects/reference-regions/` workspace area. This automatic save occurs
only after Apply; an unfinished stroke or other unapplied draft is never written.
The explicit **Save applied references** command remains available. Seed Fiddle
automatically restores the archive the next time the same image is opened, but
only after its stored SHA-256 fingerprint and dimensions match the current
image. A changed, corrupt, or incompatible image/archive produces a warning and
no reference pixels are loaded. If automatic saving fails, the newly applied
state remains available in memory and a prominent warning explains that it is
not yet durable.

The current reference-region archive is version 2 and contains only the
categorical material raster and annotated seed IDs/provenance. Version-1
archives remain readable after the same fingerprint validation, but their
retired manually painted Physical-edge/Non-edge raster is deliberately ignored.

Use the separate **Annotate seed instances** mode to give each known seed a
stable, distinct colour ID. Freehand **Brush** and **Eraser** drags show an
immediate vector stroke, then rebuild only the annotation raster on release;
they do not rebuild the analysis overlays. The assisted tools use a debounced
live preview and click-to-apply interaction: **Trace edge**
sets an initial anchor, previews a magnetic path as the cursor moves, and applies
each segment on the next click; **Smart fill** previews its proposed region and
can start from an unmarked seed or extend a partial annotation. **Shape fill**
adds a geometric prior without stamping a nominal oval:
it searches centre, size, axis ratio, and every axial ellipse rotation, then uses
that fitted shape only as a prior for a separately optimized closed contour along
the selected real edge evidence. The mouse wheel changes its visible **Oval size
preference** in 5% steps instead of zooming while the tool is active. The
refined edge contour validates the fit and remains a preview/confidence datum;
it is not a hard selection mask. Selection uses Smart fill's neighbour-relative
Lab growth and edge-frontier logic under one-sided ellipse pressure with fixed
natural 8-neighbour connectivity. The fitted ellipse is approximately a maximum
extent, not a minimum: the shape prior imposes no inward limit where a seed is
occluded, while
outward movement receives an exponential penalty with a default 0.05-seed-
diameter half-life and is prohibited beyond 0.10 seed diameter. Shape fill does
not use tangents or tunnelling, and its one-sided extent prior replaces Smart
fill's cursor-distance and radial-falloff controls. Its preview shows a dotted
prior, the refined boundary, and the proposed mask. Insufficient edge coverage,
sector coverage, continuity, confidence, or excessive overlap produces an
explicit refusal and paints nothing. **Show selected seed only** hides every
other instance mark; choosing another Seed ID recentres the image on that seed
without changing zoom. Hidden neighbouring IDs are protected from both the
brush and eraser.

Trace edge, Smart fill, and Shape fill expose the exact **Edge evidence** they consume:
**Thinned edge ridges** (the precise default), **Thinned reference edge ridge**,
**Oriented edge traces**, **Combined (ridge priority)**,
**Physical-edge probability**, or the broader **Edge magnitude** raster. Smart
fill and Shape fill additionally offer **Net physical-edge probability**
(physical minus the node's adjustable scaled non-physical evidence). The
reference-ridge choice applies
independently adjustable normal-direction non-maximum suppression and CUDA
high/low hysteresis to the broad, instance-trained physical-edge probability;
it therefore retains instance-derived classification while placing the barrier on a
local edge centreline. The combined option is a fixed per-pixel maximum of
generic thinned ridges and binary oriented-trace support, with physical-edge
probability and broad edge magnitude retained at 45% strength. Trace edge uses a
bounded continuity-aware live-wire seam,
locks to support connecting its two anchors when available, and evaluates
directed/undirected tangent evidence against each local move. It therefore does
not jump from the selected boundary to a stronger parallel one. Open trace
segments are committed as exact one-full-resolution-pixel categorical lines,
independent of the freehand brush radius and without an antialias fringe. The
first anchor remains marked in cyan; returning to it previews and, by default,
fills only the closed polygon interior while preserving other seed IDs. Smart fill
compares each candidate with already accepted touching pixels rather than a
fixed starting colour, stops at the selected edge raster, never overwrites
another seed ID, and offers bounded growth, four/eight-neighbour connectivity,
**Neighbour colour step** controls the maximum floating-range OpenCV-Lab change
from an accepted pixel to a touching candidate (L uses the displayed value and
a*/b* use 72%); it is not distance from the initially clicked colour. **Edge
barrier threshold** is the minimum selected edge strength that blocks growth,
so lower values stop on weaker evidence and higher values permit more growth.
**Maximum distance from cursor** is a hard circular limit on newly filled
pixels and defaults to 0.60 estimated seed diameter. It is a radius from the
click, not the instance's side-to-side width; with a centred click, the largest
permitted span is therefore 1.20 seed diameters. **Fall-off half-life** defaults
to 0.40 seed diameter and applies the exact radial extension pressure
`p(d) = 2^(-d/h)`. At distance `h`, both the allowed inward-neighbour Lab step
and the continuous edge-barrier threshold are half their cursor values, so
progressively weaker colour transitions or edges stop outward growth. The
minimum Lab step is taken over inward touching neighbours permitted by the
selected four/eight-neighbour connectivity. Zero edge evidence and uniform
colour remain passable to the hard distance limit, and the output remains a
categorical label rather than a feathered mask. Smart fill also exposes a
separate maximum-added-pixel safety limit and five tunnelling strengths. A one-pixel
barrier frontier lets the annotation meet a selected thinned ridge instead of
stopping on the shoulder of a blurred magnitude band. Tunnelling opens one
bounded passage through the nearest admissible weak barrier; it does not weaken
every edge or broaden the colour tolerance. If a small contact gap nevertheless
lets the neighbour-relative flood reach its cursor-distance limit, a local smooth
star-convex edge contour recovers one seed instead of accepting the leaked
radius-sized region.
Every seed ID is checked with exact categorical 8-neighbour connectivity. A
non-blocking warning identifies the selected seed's number of disconnected
areas, or lists other IDs requiring review; Apply remains available because a
disconnection can occasionally be intentional during an unfinished edit.
The **Start from result** selector can expand any available **Procedural seed
separation**, **U-Net + watershed**, or **StarDist** label result into a
full-resolution editable draft. This is an annotation bootstrap, never an
automatic review decision: every omission, duplicate, split, merge, rim
fragment, and contour must still be corrected by a person. The selected method
is retained as draft provenance and written into the unreviewed learning-export
notes.
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

Both colour-probability nodes call their capacity control **Maximum reference
colour modes**, but this shared name does not imply identical estimators.
Foreground modes are coverage-preserving quantized colour-frequency cells;
background modes are adaptively fitted robust Lab mixture components. The
background default is 32 modes and can be increased to 256 when unusually
complex reference colours justify the additional calculation time. Background
membership is evaluated in small mode groups so increasing this capacity does
not create a full image-by-mode GPU tensor. Adaptive background components are
combined as a strongest-weighted-mode fuzzy union rather than added together;
consequently, allowing more representatives cannot inflate probability merely
because several fitted modes overlap.

Selecting either the **Background colour probability** or **Foreground colour
probability** node offers an **Accepted ... colours (HSV)** overlay. It replaces
the image temporarily with a full-size, exact HSV slice: hue runs horizontally,
saturation vertically, and the toolbar **HSV value** slider scans brightness.
The initial slice follows the strongest visible fitted colour mode; **Peak**
returns to it after manual scanning. A separate neutral swatch reports the exact
achromatic probability at the selected value, since neutral colours have no
meaningful hue. The 25%, 50%, 75%, and 90% contours are still evaluated by the
unchanged CIE Lab mixture model, and the represented colours are never dimmed by
membership probability. This exact three-coordinate view avoids the ambiguous
hidden-saturation projection used by the former miniature inspector plot.
The foreground diagnostic uses painted modes when supplied. Otherwise it uses
the independently detected isolated reference seeds above the ruler as its
starting colour-frequency samples. The foreground inspector lists the eight
most frequent starting modes as colour swatches, hexadecimal RGB values, and
frequencies. If no reference seed survives detection, the inspector explicitly
labels the older high-confidence-image-pixel diagnostic fit as a fallback.

The active graph exposes one **Reference layers** input node with distinct typed
outputs for Background, Foreground, Other, and Annotated seeds. Every
calculation that consumes one of these sublayers has a corresponding
connector. Painted material areas supply direct colour/texture evidence;
colour-probability pseudo-labels remain only a fallback where a class has no
painted samples. Painting never overwrites the resulting probability at the
brush coordinates.

Complete annotated instances automatically fit the image-local physical versus
non-physical edge classifier using the corrected Lab image, continuous edge
magnitude, directed/undirected tangent coherence, and thinned ridges. Their
contours are physical examples; safely inset edge/ridge candidates are sparse
non-physical examples. Its physical and non-physical overlays feed final curve
confirmation and procedural watershed boundary cost and remain available as
optional channels to compatible learned-model checkpoints. With no annotated
instances, the semantic output is neutral rather than a generic-edge fallback.

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
including when a node-graph pan begins over a connection path. Connections use
right-click to disconnect or Ctrl-click followed by Delete/Backspace; ordinary
left-drag is reserved for navigation unless it begins on a node or socket.
Node colours report idle, running,
complete, warning, planned, bypassed, and failed states.
The inspector's **How it works** explanation starts collapsed whenever a node is
selected and can be expanded with its disclosure button.
Every node has a bottom calculation-time footer. CUDA stages use queued GPU
events and resolve all durations with one synchronization at the end of the run,
so timing does not serialize the pipeline at every node; reused nodes retain
their most recent measured time. During a run, nodes begin blue and independently
turn green as their worker stage completes; progress messages are scoped to the
current image and pipeline revision.

The active and restorable graph is audited against a separate 53-node
direct-input contract rather than testing only the connectors that happen to
exist. Dish-region, corrected-image, physical-scale, seed-diameter, mask,
proposal, and reference inputs are therefore shown on every calculation that
directly reads them. For example, **Edge gradients** is connected to **Layout
detection**, and **Oriented edge traces** is connected to **Seed scale
estimate**. A seed-scale change invalidates the trace calculation but preserves
its reusable ridge field. Generated products such as sensor noise and local
shadow use one consistently typed output socket instead of duplicate aliases.

Full-image CUDA jobs are serialized through a dedicated one-worker queue and
new requests are coalesced while a job is running. Completed node caches use a
least-recently-used limit of three images and a 2 GiB reachable-CUDA budget;
failed and evicted jobs explicitly release lazy CPU mirrors. Overlay downloads
are retained only until the next overlay is rendered. Applied annotations are
immutable analysis inputs, while copy-on-write drafts allocate only the layer
currently being edited.

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
   preserves both configured distances. An undimmed swatch and hexadecimal
   label show the median starting background colour selected from that band.
3. **Seed scale estimate** examines the fixed isolated-reference region above
   the ruler in CIE Lab. Colour-difference components are morphologically
   cleaned and filtered by edge contact, area, aspect ratio, and size relative
   to the dish. The corrected median equivalent diameter of up to three accepted
   components becomes the global seed diameter; if none survive, the fallback
   is 16% of dish radius. Pixels belonging to the accepted isolated components
   also provide the independent automatic foreground colour samples.
4. **Foreground colour probability** retains individual Lab samples from its compact
   rim-adjacent prior and fits a multimodal background distribution without
   allowing in-dish seed colours to redefine it. Weighted Lab distance supplies
   the initial foreground strength. The isolated reference-seed samples are
   quantized into individual colour-frequency modes rather than averaged, then
   compared with the background distribution using normalized relative
   membership. Seed-scale local lightness favours coat surfaces over dark gaps,
   while unsupported colours are reduced in crowded dishes. Painted foreground
   references replace the automatic colour start. The applied threshold is
   `max(8, Otsu threshold × Foreground threshold)`, followed by seed-scale
   elliptical opening and closing.
   When painted samples occupy more quantized cells than the configured mode
   budget, retained modes are selected for Lab-space coverage rather than raw
   frequency alone. Adding a large, varied affirmative region therefore cannot
   evict an earlier colour merely because that earlier patch is smaller. The
   full-pane HSV diagnostic evaluates one exact Value slice at a time,
   confines visually neutral membership to a separate swatch, and labels only
   the leading frequencies; near-neutral dark evidence therefore cannot appear
   as an unrelated all-hue band.
   The **Background colour probability** diagnostic consumes the separately
   controlled perimeter reference, evaluates the same fitted colour model over
   that exact buffered annulus, and shows the resulting probability beside the
   dish-resident raster. The annulus remains outlined without an opaque colour
   tint (or uses the inside-rim fallback when necessary).
   Its node-owned full-pane overlay shows 25%, 50%, 75%, and 90% fitted
   membership contours over the exact HSV hue/saturation slice selected by the
   toolbar Value control, with learned mode frequencies and a neutral swatch.
   The **Background noise probability** applies its learned three-band texture
   classifier to that same surrounding annulus and displays the resulting noise
   likelihood alongside the dish-resident refined map.

   The background colour and noise nodes also expose **Other colour probability**
   and **Other noise probability** diagnostics learned from painted Other material.
   Both use direct brightness (white is stronger Other evidence), do not force
   painted coordinates to probability one, and remain blank when no Other
   reference has been painted. Other noise uses a base score of 72% learned
   three-band texture probability plus 28% Other-colour probability before
   directional ray integration. These class-specific diagnostics are distinct
   from **Reference Other-material probability**, which combines colour,
   texture, residual, edge, and ridge prototype features.

   The foreground and background colour maps are related evidence scores, not
   complementary or mutually calibrated posteriors. Background colour uses a
   compact robust multimodal Lab fit anchored by painted background examples or
   the adjustable perimeter annulus. Foreground colour instead combines distance
   from an independently fitted background model, an Otsu-derived cutoff,
   seed-scale lightness support, and a higher-capacity Lab frequency table for
   painted or isolated-reference-seed colours. This extra capacity is intentional
   for patterned seed coats, but it means equal numeric values in the two maps do
   not imply equal confidence. The automatic foreground path also retains its
   older compact rim prior rather than consuming the adjustable perimeter band;
   unifying and recalibrating those starts requires fixture-level retuning.
5. **Foreground noise probability** learns a matching three-band profile from
   foreground-colour pseudo-labels. It continues texture evidence across
   patterned coats without hard-forcing painted reference pixels.
6. **Edge gradients**, **Thinned edge ridges**, **Thinned reference edge
   ridge**, and **Oriented edge traces**
   retain the strongest current evidence for visible seed boundaries without
   depending on an unvalidated centre proposal. The reference ridge is a
   separately cached GPU node fed by broad physical-edge probability and the
   continuous edge normals; it preserves peak probability after NMS and exposes
   its own step, low/high threshold, hysteresis-reach, and working-size controls.
   The **Instance-derived edge probabilities** node also owns two comparison
   diagnostics. **Physical blue / non-physical red** places the existing
   physical score in blue and the non-physical score in red, with magenta
   showing simultaneous support. **Net physical-edge probability** displays
   `max(physical - k × non-physical, 0)` in blue and clamps ties or
   non-physical-dominant pixels to black. Its node-owned **Internal-edge
   subtraction** coefficient `k` defaults to 0.5 and is adjustable from 0 to 2,
   allowing overlap between the two learned classes without automatically
   punching holes through a physical trace. It changes only this diagnostic;
   the two source probability rasters remain unchanged. These are evidence-score
   comparisons rather than calibrated complementary probabilities.
   Non-adjacent trace-gap links
   require a tight chord/tangent match, consistent directed gradient side, and
   an open directional endpoint. An additional **Convexity bias** defaults to
   **prefer**: it compares the tangent-to-chord bend at both endpoints and
   rejects confident S-shaped bridges while retaining either winding of a
   convex C-shaped continuation. **Require** applies the editable angular
   ambiguity tolerance strictly, while **off** preserves tangent-only linking.
   This is an honest local rule for initial gap assignments; the later
   circle/ellipse model remains responsible for global convexity. Raising the
   gap can therefore reconnect a broken rim without linking the interiors of
   adjacent parallel seed rims.
7. **Procedural seed separation** fuses foreground colour/noise, the optional
   many-prototype seed-surface likelihood, and inverse background into a material
   likelihood, fills only enclosed seed-sized coat
   holes, and excludes a seed-relative dish margin. Edge magnitude and thinned
   ridges form the generic boundary candidate field. The instance-derived
   physical-minus-non-physical semantic margin gates that whole field (and is
   neutral when no annotations exist), while the separately thinned physical
   ridge and coherent convex oriented traces add narrow, continuity-aware
   support. Sensor/noise, shadow, and internal-boundary peaks are not direct
   boundary or centre evidence. Smoothed material geometry and seed-interior
   depth produce automatic markers; distinct painted instance IDs replace
   nearby automatic markers. OpenCV marker-controlled watershed performs the
   bounded topological partition, and marker/boundary/area evidence supplies a
   per-instance confidence. Six owned overlays expose every stage. GPU inputs
   are resized before this topology-only CPU transfer. Labels and diagnostic
   rasters remain at that bounded working resolution and are scaled by Qt only
   for display, then the compact result is cached at the node.
   Applied instance annotations can also supervise a bounded, image-local
   parameter search from the node inspector through **Fit settings to applied
   annotations…**. Each trial deliberately withholds the annotated IDs from
   watershed marker generation, then scores its independent prediction against
   the reviewed masks with one-to-one instance matching. Overreach is distance
   weighted rather than uniform: the cost is zero on the reviewed seed, small
   immediately outside it, equals the editable overreach amplitude at the
   default 0.50-seed-diameter distance scale, and then rises exponentially
   (with a finite safety cap). Matched predictions are measured from their own
   target, so merging into a neighbouring annotated seed cannot exploit the
   distance to the annotation union. Missing seed pixels retain a fixed 1.0
   cost. The search returns an unapplied proposal, reports raw precision/recall
   and FP/FN counts plus distance-weighted FP equivalents and the effective
   pixel scale, and changes the graph only after explicit acceptance.
   **Annotations cover whole dish** is off by
   default: partial-review scoring ignores predictions wholly disjoint from an
   annotated seed and labels its metrics accordingly. Turn it on only after
   every seed in the detected dish has a complete instance mask; whole-dish
   scoring then penalizes every predicted object, including isolated background
   false positives. The fit is measured on one image, but accepting its proposal
   changes the procedural node settings globally for every image. This is useful
   in-sample adjustment, not evidence that settings generalize; publishable
   validation still requires independent, complete reviewed masks.
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
separate editable weights. Its dish geometry and the distance-candidate density
that adjusts its threshold are explicit inputs; it no longer hides a private grayscale-edge
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
  directed-ray geometry, direction integration (including foreground's default
  exact first tertile), and GPU working resolution;
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
- ridge NMS/hysteresis, oriented trace linking, gap and local-convexity
  controls, dense radius
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
Changing only the trace convexity policy or its angular ambiguity likewise
rebuilds trace assignments and their dependents without recalculating ridge
NMS/hysteresis.

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
