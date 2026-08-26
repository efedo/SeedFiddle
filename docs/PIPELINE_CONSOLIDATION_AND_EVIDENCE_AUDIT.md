# Pipeline consolidation and evidence-chain audit

Date: 2026-08-24

## Purpose

This redesign makes the visible graph describe calculations rather than legacy
implementation stages. It also separates semantic class evidence from signal
strength, establishes an explicit no-target-leakage rule for fitting and
inference, and preserves node-local cache invalidation.

The implementation followed this order:

1. Inventory every visible node, overlay owner, typed connector, runtime cache
   product, progress stage, and settings-profile record.
2. Select the surviving calculation owner for each wrapper pair and move all
   computational parameters, overlays, timings, and status reporting to it.
3. Rebuild connections from actual runtime reads, then remove obsolete ports
   only after every extant overlay had a surviving owner.
4. Audit evidence equations for accidental class normalization, hard masking,
   signal-strength leakage, and annotation resubstitution.
5. Add graph, cache, descriptor, fitting, and Qt overlay-contract regressions.

## Consolidated graph

- **Species and metadata** and **Raw images** are independent first-tier inputs.
  Metadata no longer receives an invented image-data dependency.
- **Deskew and colour balance** owns colour-card detection, the detected-swatch
  overlay, projective correction, and colour balance.
- **Ruler detection and scale** owns ruler evidence, geometry, tick families,
  unit interpretation, pixels/mm, and the calibrated-image overlay.
- **Edge gradients** owns original/wavelet source fusion, derivative selection,
  magnitude, both tangent encodings, and thinned generic ridges.
- **Reference edges** owns the physical-minus-non-physical products, local
  normalization, and all three thinned semantic ridge products. The upstream
  prototype node still owns the two raw semantic prototype-class overlays.
- The former visible Seed-interior wrapper was removed. Its only useful
  downstream input is the authoritative **Resolved Seed probability** from
  Material evidence decision. A private compatibility raster may still be
  calculated for dormant advanced diagnostics; it is not a node, overlay, or
  procedural input.
- The Procedural seed separation `Seed-material likelihood` duplicate was
  removed. Procedural display starts with its own material mask and distinct
  centre, boundary, candidate, instance, confidence, and concavity products.

Legacy settings versions migrate parameters and connection state from the
retired wrappers. Unknown nodes, parameters, and wires remain hard errors; a
profile cannot be made to appear valid by silently dropping an unrecognized
endpoint.

## Evidence-chain findings

### Material colour

Foreground, Background, and Other colour rasters remain independent raw scores;
they are not complements and do not sum to one. Painted coordinates train the
models but never overwrite output pixels. Foreground requires painted evidence
or the explicitly enabled safely inset annotated-instance source. Background
uses painted regions and/or the retained perimeter source. Other is its own
positive class, not a special Foreground veto.

### Material texture/noise

The previous implementation did have an undesirable dependency: after fitting
a texture classifier it blended the texture probability back with the colour
probability. It also used colour thresholds to invent positive/negative texture
samples when direct examples were absent. This made a purported raw texture
overlay partly duplicate colour evidence and caused double-counting in Material
evidence decision.

The corrected node is texture-only:

- direct painted, perimeter, and safely inset annotated-instance regions define
  target texture samples;
- painted counterclasses define contrastive samples;
- if counterexamples are missing, they are selected by standardized texture
  distance from the fitted target rather than low colour probability;
- Background, Foreground, and Other output probabilities use only their fitted
  multiscale texture distributions and directional continuation;
- disabling Background colour no longer disables Background texture analysis;
- the retired colour-pseudo-label threshold controls were removed.

The only remaining graph edge from Material colour probabilities to Material
noise probabilities carries the *derived annotated-Foreground reference
region*, not a colour probability. It exists because that safely inset region
is option-controlled and must participate honestly in cache invalidation.

### Reference prototypes and reference edges

The concern about the edge-class overlays was correct. The former descriptor
embedded absolute edge/ridge amplitude, and the upstream class output multiplied
prototype similarity by edge support. Therefore a value labelled “prototype
probability” was actually a strength-dependent amalgamation.

The corrected tangent-aligned descriptor contains only semantic appearance:
interior, edge-centre, and exterior Lab strips; local L/chroma residuals; signed
cross-edge Lab contrast; axial tangent coherence; and valid support. Absolute
edge and ridge magnitudes are excluded. Physical and non-physical outputs are
conditional class probabilities derived from their competing prototype
similarities. Reference edges combines those semantics with the independent
selected gradient strength only when constructing raw/net/normalized ridge or
barrier products.

Local net-edge normalization now normalizes the independent image-gradient
envelope, applies a bounded gain, retains an absolute signal floor, and then
weights it by semantic margin/confidence. A confident semantic match in a flat
region cannot become a boundary.

### Edge gradients, ridges, and oriented traces

Every derivative method now reaches the shared magnitude calculation, and a
method-only regression proves that the displayed magnitude changes. Wavelet
source selection remains an explicit upstream dependency; detail layers plus
the residual exactly reconstruct the corrected image.

The doubled normalized-reference traces came from treating every nonzero pixel
after bilinear resize/restoration as a ridge. A one-pixel source consequently
became two or three parallel positive pixels. Semantic overrides are now
re-thinned along the live continuous normal with deterministic tie-breaking
before oriented component linking. A regression supplies a three-pixel
interpolation halo and requires a single traced column.

### Material decision

Material evidence decision is the only place that resolves Seed versus
Non-seed. It combines independent colour, texture, and material-prototype
evidence with ambiguity and unknown mass. Background and Other contribute
symmetrically to Non-seed and are resolved as conditional subtypes only after
the top-level decision. No upstream raw probability raster directly masks
another class.

### Procedural separation and learned inference

Applied instance IDs are valid reference/training evidence, but they are not
automatic inference results. The enforced boundaries are:

- normal procedural inference receives `seed_instance_annotations=None`;
- provisional instance masks have no annotation wire;
- U-Net and StarDist decoders receive `painted_instances=None` and their decoder
  cache identity contains no annotation hash;
- procedural parameter fitting receives neither annotation-derived material
  probability, semantic edge probability/ridges, oriented traces, nor reference
  surface probability; annotations are used only as the scoring target;
- reference-edge parameter optimization trains prototypes on alternating
  instance IDs and scores a disjoint held-out set of IDs.

Manual seed centres remain an explicit user-authored inference constraint; they
are not silently derived from the masks being evaluated. Low-level procedural
APIs retain annotation-marker support for deliberate annotation-tool workflows,
but the desktop automatic and fitting paths do not invoke it.

These rules prevent exact painted outlines from entering automatic outputs by
construction. They do not claim cross-image generalization: fit quality on one
project still requires independent images or held-out annotations for scientific
validation.

## Cache and overlay contracts

Changing a parameter invalidates its owner and graph descendants only. In
particular, changing Edge gradients does not invalidate Raw images, metadata,
deskew/colour, ruler, layout, wavelets, material colour, or material noise.
Internal legacy timing labels are aggregated onto their visible owner and do
not create graph nodes or upstream invalidation.

Every selectable overlay has exactly one visible owner whose title matches the
toolbar group and inspector. When an overlay is also a real data product, its
identically labelled connector is that typed product connector rather than an
extra viewer-only socket. Viewer-only diagnostic renderings receive a socket
only when no reusable calculation output exists. Duplicate typed sockets left
by old names are rejected by graph-contract tests.

## Regression coverage

The focused suite covers:

- exact active/toolbox node and connection catalogues;
- audited direct-input contracts and acyclicity;
- settings migration and strict unknown-endpoint rejection;
- node/inspector/overlay-title and overlay/output-connector equality;
- method-only shared-gradient changes and downstream-only invalidation;
- texture-probability invariance to the supplied colour raster under direct
  supervision;
- semantic-strip invariance to absolute edge/ridge amplitude;
- disjoint instance-ID prototype fitting/evaluation;
- annotation-free procedural and learned desktop inference paths;
- single-ridge oriented tracing after interpolation halos.
