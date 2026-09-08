# Comprehensive node-calculation forensic audit

Date: 2026-08-27

## Scope and standard

This is a read-only forensic audit of the calculations behind every active and
toolbox node in the Seed Fiddle graph. It does not treat a passing UI/port test
as proof that the represented calculation is scientifically or numerically
correct. The audit checked:

- graph-declared inputs, outputs, parameter ownership, and invalidation;
- runtime data flow, including inputs passed outside graph connections;
- cache keys, recomputation boundaries, and progress/timing attribution;
- numerical definitions, units, dtype/range handling, resizing, and topology;
- annotation use at fitting, inference, and evaluation time;
- probability calibration and semantic meaning;
- active, disabled, shelved, and not-yet-implemented branches;
- focused and full automated-test results.

The source tree was not altered except for this report. Existing uncommitted
work was preserved.

## Executive result

The graph is structurally much stronger than an ordinary experimental CV
application: node/overlay/connector contracts, exposed controls, persistence,
wavelet reconstruction, material-mass conservation, and targeted edge tests
are covered well. The focused audit suite passed 216/216 tests.

That structural coverage does not catch several latent calculation defects.
The audit found six high-priority defects in the active or restorable analysis
paths, thirteen medium-priority calculation or scientific-validity defects, and
several lower-priority diagnostics/test-infrastructure defects. No evidence was
found that the active Procedural, U-Net, or StarDist inference call directly
receives painted instance shapes. A dormant legacy instance node does do so,
despite explicitly claiming that it does not.

Severity used below:

- **P1**: can materially change an inference result, defeat a hard control, or
  violate the graph/cache/annotation contract.
- **P2**: scientifically misleading, unreliable in an important edge case, or
  capable of substantial resource/correctness impact.
- **P3**: diagnostic, provenance, API, or maintenance problem with limited
  immediate effect on the default desktop graph.

## Confirmed findings

### NC-01 — A restored Instance colour masks node injects and hard-writes annotations (P1, dormant)

The node's own description says applied instance annotations are deliberately
excluded from inference. Runtime does the opposite:

1. `_merge_annotated_instance_seeds` prepends every annotation centroid and
   suppresses nearby automatic proposals.
2. `instance_voronoi` writes the painted instance identifiers back into the
   output label raster at every annotated pixel.
3. The node has no declared Project/Annotations input, so this is also a hidden
   dependency outside the graph.

This is exact target leakage and a cache/provenance defect if the shelved node
is restored. The default graph is protected only because the node is currently
disabled and shelved. Remove the annotation arguments from this calculation,
or make a separately named annotation-assisted review node with an explicit
Project connection. Add a test that perturbing instance labels cannot alter
automatic instance output when only inference inputs are held constant.

Evidence: `seedvision/cuda/layers.py` around `_merge_annotated_instance_seeds`,
`instance_voronoi`, and the `instance_masks` branch; `seedvision/pipeline/model.py`
in the Instance colour masks details and connection table.

### NC-02 — Procedural hard shape limits use incorrect geometry (P1, active)

There are two independent defects in `_candidate_from_component`:

- `area` is a count of raster pixels, while `hull_area` is OpenCV's continuous
  polygon area. For a filled 10x10 pixel square these are 100 and 81. The
  reported solidity is therefore 1.2346 and concavity is clamped to zero.
  Concavity is systematically understated and solidity overstated, especially
  for small candidates.
- “Maximum width” is `max(width, height)` from `cv2.minAreaRect`, not maximum
  side-to-side/Feret width. For that same square it reports 9 pixels while the
  maximum Feret span is 12.73 pixels: a 29.3% underestimate. Consequently a
  seed can visibly exceed the hard maximum and still pass it.

Use one area convention for both region and hull (prefer filled raster hull
counts at the working resolution), and calculate actual maximum Feret width
from the convex hull. Re-test every hard/soft width, concavity, and solidity
boundary at several working scales.

Evidence: `seedvision/segmentation/procedural.py` lines around 748-808.

### NC-03 — Reference-edge fitting controls are owned by the wrong graph node (P1, active)

The right pane assigns edge prototype count, support, fit iterations,
similarity scale, class contrast, edge working resolution, strip geometry,
ridge weight, and instance-interior buffer to **Reference edges**. Those values
are actually consumed while computing the upstream **Reference texture
prototypes** product and are included in its private runtime signature.

Changing `reference_edge_similarity_scale`, for example, graph-invalidates
Reference edges and its descendants only. At runtime the changed private
signature silently recomputes Reference texture prototypes anyway. The graph
can therefore report the upstream node as reused while it actually performs
the expensive calculation; Run-to-node, progress attribution, cache reasoning,
and parameter ownership are all false.

Move semantic edge-bank fitting/matching completely into Reference edges, with
Reference texture prototypes restricted to material products, or move the
controls and ownership upstream and model the dependency honestly. A test must
assert both numerical output and exact computed/reused node lists for every
control.

Evidence: `seedvision/pipeline/model.py` in
`reference_edge_probability_parameters`; `seedvision/cuda/layers.py` around
`reference_texture_signature` and `reference_texture_probabilities`.

### NC-04 — Automatic background authority is defeated in material-prototype fitting (P1, active)

When painted Background exists, the automatic perimeter source is spatially
thinned according to `background_automatic_authority`. However,
`_fit_feature_prototype_bank(..., priority_mask=painted_background_mask)` then
allocates exactly half of its fitting samples to painted pixels and half to the
remaining automatic pixels whenever both are nonempty. If the thinned ring
still contains enough pixels—as it normally does—the configured authority no
longer controls its prototype-bank influence.

This means weak automatic background evidence is not necessarily rapidly
suppressed relative to painted evidence in the Reference texture prototypes
path, even though the colour model honors that policy. Allocate sample/support
weight according to the actual authority, retain source provenance per
prototype, and regression-test outputs while sweeping authority from zero to
one.

Evidence: `seedvision/cuda/layers.py` around 4671-4686, 3708-3762, and
4825-4838.

### NC-05 — Procedural fallback ignored the configured non-physical subtraction weight (P1, resolved)

The raw weighted margin is now explicitly diagnostic. The Reference edges node
publishes a separate authoritative field:
`thinned_true_edge_support * max(Pphysical - weight * Pnonphysical, 0)`.
Procedural preparation receives that cached field directly. It uses the
normalized supported derivative when available and the exact authoritative
field otherwise; it no longer reconstructs `physical - nonphysical` with an
implicit weight of 1.0. Separately supplied Physical and Non-physical channels
are also multiplied by true-edge support before entering procedural, curve, or
learned consumers.

Regression coverage requires the authoritative fallback to be bit-identical to
the same-valued normalized input and requires the full-resolution supported
field to be zero wherever the generic thinned ridge is zero.

### NC-06 — Ruler changes recompute and replace the upstream deskew calculation (P1/P2, active)

The graph says Ruler detection and scale is downstream of Deskew and colour
balance. Runtime stores both in one `calibration` cache object and calls the
entire `calibrate_image` routine when either node is dirty. A ruler-only
parameter update therefore recalculates deskew/colour correction and overwrites
its cached upstream result, although the graph invalidates and attributes only
the ruler node.

Split colour-card detection/deskew from corrected-image ruler detection and
give each stage its own signature/cache product. This is primarily a node-local
caching and provenance violation, but it can become numerical if stochastic or
threshold-sensitive upstream detection changes on a downstream-only edit.

Evidence: `seedvision/segmentation/baseline.py` around 590-603 and
`seedvision/calibration/image.py` in `calibrate_image`.

### NC-07 — A reliable imperial scale cannot rescue an unreliable metric scale (P2, active)

Metric and imperial pixel/mm estimates and reliability flags are calculated
independently, but `RulerDetection.scale_reliable` is set only from
`metric_reliable`. Absolute calibration then refuses scale unless that flag is
true and always prefers the metric estimate. A clean imperial scale is
discarded if the metric row is damaged or cropped.

Use reliable metric as the preferred source, reliable imperial as a clearly
reported fallback, and their robust combination only when both pass hierarchy
and cross-scale sanity checks.

Evidence: `seedvision/calibration/image.py` around 971-988, 1031-1040, and
294-313.

### NC-08 — “Thinned” full-resolution ridges are bilinear-expanded rasters (P2, active)

Non-maximum suppression is performed at a bounded working resolution. The
binary/continuous ridge is then restored with bilinear interpolation. A
one-pixel impulse at 4x4 restored to 20x20 produces 81 nonzero pixels (13 above
0.5), so the advertised full-resolution ridge is no longer one pixel wide.
The trace override path re-thins selected sources, but overlays and some
procedural consumers still receive broadened ridges.

Restore continuous evidence and rerun NMS at source resolution, or use a
centreline-preserving coordinate/vector representation. Nearest interpolation
alone preserves discreteness but still produces a thick scale block.

Evidence: `seedvision/cuda/layers.py` in `reference_probability_ridges` around
5560-5569 and `seed_boundary_tracing.full_u8`/ridge return paths.

### NC-09 — Material source authority is calibrated on its own fitting pixels (P2, active)

`_source_reliability` explicitly computes “held-in discrimination.” Painted
pixels used to fit a colour/noise/prototype source are reused to decide that
source's authority. This is not a coordinate overwrite, but it is
resubstitution bias: flexible per-image models can receive high authority for
memorizing their samples. The edge-settings fitter now withholds instances,
but the material decision has no analogous out-of-fold calibration.

Estimate reliability by held-out painted components/instances or deterministic
spatial folds. When evidence is too sparse, report it as uncalibrated and use a
conservative prior rather than silently assigning full reliability.

Evidence: `seedvision/cuda/material.py` in `_source_reliability` and its nine
calls in `hierarchical_material_evidence`.

### NC-10 — Procedural reports pre-truncation geometry after overlap resolution (P2/P3, active)

Selected candidates may overlap by the configured fraction. Final labels give
overlapping pixels to the earlier label, but area, width, concavity,
protrusion, solidity, axis ratio, and confidence are copied from each original
candidate. The selected mask and its reported statistics can therefore differ;
concavity visualization also derives from the pre-assignment shape.

Recompute all final geometry/statistics from the actual assigned labels and
apply hard limits once more after assignment.

Evidence: `seedvision/segmentation/procedural.py` around 1947-1978 and
2078-2099.

### NC-11 — The centre-likelihood weights can normalize to an all-zero source set (P2, active)

Settings validation requires the configured centre weights to sum above zero.
At runtime the flattened-grayscale weight is forcibly set to zero if the
illumination product is unavailable. Thus a valid configuration with material
and distance weights at zero and flattened weight above zero becomes an
effective all-zero vector; division uses an epsilon and the complete centre
field is black instead of producing a validation error or fallback.

Validate effective, available inputs at execution time and either stop with a
specific error or fall back to a declared source.

Evidence: `seedvision/segmentation/procedural.py` around 1478-1511.

### NC-12 — A single available semantic prototype class blanks all class outputs (P2, active edge case)

`_prototype_class_probabilities` returns an empty mapping unless at least two
prototype banks exist. For edge annotations with strong physical contours but
no qualifying internal patterned edges, the physical bank can be valid while
the non-physical bank is absent; both displayed probabilities then become
black. The UI does not distinguish “unavailable competition” from genuine zero
probability.

Retain a known-class similarity/unknown mass output, or declare the probability
unavailable with an explicit status overlay. Do not encode unavailable as
probability zero.

Evidence: `seedvision/cuda/layers.py` in
`_prototype_class_probabilities` and the edge probability fallback around
5169-5215.

### NC-13 — Image-quality “focus” and “sensor noise” conflate scene content with defects (P2, active diagnostic)

Local focus is merely globally normalized gradient magnitude. Every smooth but
sharp seed interior is consequently classified as poor focus. Sensor/noise is
absolute residual from a Gaussian blur, so genuine seed texture and edges are
classified as sensor noise. Quality risk takes the maximum of these fields,
making the resulting map unsuitable as a calibrated quality diagnostic. The
sensor-noise field also feeds optional learned/circle paths.

Estimate focus from local high-frequency energy relative to local structure,
and noise from flat-region residual statistics or a signal-dependent camera
noise model. Separate “texture” from “sensor noise.”

Evidence: `seedvision/visualization/advanced.py` around 799-833 and
`sensor_noise_likelihood_tensor`.

### NC-14 — Calibration residuals select the nearest anchor before considering anchor risk (P2/P3, toolbox)

The calculation chooses the spatially nearest anchor, then adds that anchor's
risk. It should minimize the combined candidate cost (anchor risk plus spatial
distance penalty). A slightly farther, high-confidence anchor can correctly be
lower risk than the nearest poor anchor, but can never win under the current
equation.

Evidence: `seedvision/visualization/advanced.py` around 1032-1065.

### NC-15 — Learned-model cache can retain unlimited GPU checkpoints (P2, optional active branch)

`_MODEL_CACHE` evicts older revisions only for the same path. Loading many
distinct checkpoint paths/devices retains every model and payload indefinitely.
On the target 8 GiB GPU this can exhaust memory during model comparison,
training/refinement, or profile switching.

Use a small LRU keyed by path/fingerprint/device, explicit cache release, and
CUDA memory cleanup after eviction.

Evidence: `seedvision/learning/pipeline.py` in `cached_checkpoint`.

### NC-16 — Raw-image cache identity omits the file fingerprint (P2/P3, active)

The Project raw-image cache is keyed only by case-folded resolved path. If an
image is replaced or edited externally while the project remains open, a rerun
can reuse stale pixels indefinitely unless `project_image` is explicitly dirty.

Include size and high-resolution modification time (and optionally a cheap
content fingerprint) in the raw cache signature.

Evidence: `seedvision/segmentation/baseline.py` around 448-480.

### NC-17 — Width/concavity filters still lack a protrusion-length invariant (P2, active design gap)

`maximum_protrusion_area_fraction` measures pixels removed by a fixed
morphological opening. An arbitrarily long, sufficiently thin tail can have
small area and evade this limit; the incorrect width calculation in NC-02
makes that easier. Add a skeleton/geodesic protrusion-length-to-local-thickness
limit or a convex-hull radial excursion limit, then combine it with true Feret
width and corrected concavity.

Evidence: `seedvision/segmentation/procedural.py` around 788-805.

### NC-18 — API default execution does not mean “default graph” (P2/P3, API)

`analyze_image(..., enabled_nodes=None)` treats every layer/advanced node as
enabled, including dormant legacy branches. Procedural and learned nodes use a
different convention and remain opt-in. Thus the public default API is neither
the desktop's default graph nor a consistent “all nodes” mode.

Make the default explicit: resolve the default graph's active/enabled set, and
provide a separately named all-experimental mode.

Evidence: `seedvision/segmentation/baseline.py` around 580-588 and 1835-1839;
`seedvision/visualization/advanced.py` around 461-469.

### NC-19 — Disabled placeholder products are counted as computed nodes (P3, active bookkeeping)

The layer cache creates zero placeholders for disabled branches. The later
`layer_cache_keys` loop records a node as computed whenever its cache key was
missing/dirty without checking whether the node was enabled. Timings may remain
empty, but `last_computed_nodes` can claim work for disabled/shelved aliases.

Filter bookkeeping through the actual enabled set and distinguish “placeholder
installed” from “calculation executed.”

Evidence: `seedvision/segmentation/baseline.py` around 1409-1435.

### NC-20 — The material-evidence audit tool is stale and mislabels current evidence (P3, tooling)

`tools/audit_material_evidence.py` requests several retired node IDs and calls
the current sole reference-derived foreground probability “automatic
foreground,” while the “reviewed foreground” product is now normally absent.
The tool can therefore produce a plausible but semantically wrong forensic
report.

Generate its node set from the graph, use current product names, and add a test
that each reported raster has a live owning node/port.

### NC-21 — The committed learning manifest references a missing fixture (P3, repository integrity)

The full suite's one known failure is not a calculation assertion: the checked
manifest references `images/IMG_9689c.JPG`, but the repository contains
`images/IMG_9689.JPG`. This prevents a green full-suite baseline and can hide a
new regression among expected noise.

Repair the manifest/fixture association deliberately; do not synthesize or
rename scientific fixture data without confirming the intended image.

### NC-22 — `RulerDetection` defaults an unverified scale to reliable (P3, API invariant)

The dataclass default is `scale_reliable=True` even though default tick pitch,
metric px/mm, hierarchy consistency, and both hierarchy flags are zero/false.
Internal detection sets the field explicitly, but manually constructed/test
objects can represent an impossible reliable scale. Default it to false and
validate consistency in `__post_init__`.

### NC-23 — Annotation-derived seed scale is valid assistance but invalidates naive holdout claims (P3, scientific provenance)

The active Seed scale estimate intentionally replaces the automatic reference
diameter with the mean of selected annotated-instance Feret widths. That was a
requested feature and is not shape injection. However, Procedural and learned
results on that same image are not wholly annotation-independent because their
scale, material prototypes, and edge prototypes can all derive from those
annotations. Performance reports must distinguish:

- annotation-assisted inference on the same image;
- held-out-instance evaluation with prototypes fitted on other instances; and
- fully annotation-free inference.

The edge parameter fitter withholds prototype pixels by alternating instance
IDs, but repeatedly tunes parameters on that same fixed holdout fold; it is a
validation fold, not an unbiased final test.

### NC-24 — The future Measurements node repeats the pixel/polygon area mismatch (P2/P3, toolbox)

`measure_mask` counts raster pixels for seed area but computes convex-hull area
from boundary-pixel centre coordinates. It then takes `max(pixel_area,
polygon_hull_area)`, which can force solidity to exactly 1.0 for genuinely
concave small objects. In a synthetic 10x10 L-shaped region, pixel area is 75,
polygon hull area is 68.5, and reported solidity is therefore 1.0. Feret lengths
are likewise measured between pixel centres while area represents pixel cells,
creating a one-pixel convention discrepancy.

Use a filled hull raster for pixel-domain solidity or consistently apply a
pixel-cell contour convention to area, hull, perimeter, and Feret dimensions.
Add analytic rectangle, ellipse, L-shape, and border-contact tests before the
Measurements node becomes reachable.

Evidence: `seedvision/measurement/geometry.py` in `measure_mask`.

## Node-by-node disposition

“No new defect” means the calculation and its declared controls/data flow were
inspected and no additional concrete flaw was identified; it is not a proof of
scientific accuracy on unseen imagery.

### Active graph

| Node | Audit disposition |
|---|---|
| Project | Raw-image cache defect NC-16; otherwise source/annotation ownership is explicit. |
| Species and metadata | No numerical calculation; project dependency is explicit. |
| Deskew and colour balance | Coupled cache/ownership defect NC-06. |
| Ruler detection and scale | NC-06, NC-07, and API invariant NC-22. Tick-family hierarchy and cross-scale error otherwise have direct calculations. |
| Layout detection | No new arithmetic defect found in circular/ellipse/rim-pair calculations; still empirical and fixture-dependent. |
| Hue only | No new defect; deterministic display transform. |
| Wavelet decomposition | Exact additive reconstruction is tested; no new defect. |
| Seed scale estimate | Calculation uses correct annotated maximum-Feret widths; provenance caveat NC-23. |
| Material colour probabilities | No coordinate overwrite found. Source models are independent; audit-validity concern NC-09 applies downstream. |
| Material noise probabilities | No coordinate overwrite found. Noise remains per-image texture classification, not a physical sensor-noise measure. |
| Edge gradients | Method/wavelet selection is connected and consumed; no new control defect. Ridge restoration contributes to NC-08. |
| Directional surface darkness gradients | Independent surface-slope information is calculated; no new defect found. |
| Multiscale darkness and colour noise | Six distinct band/colour products are calculated; no new defect found. |
| Reference texture prototypes | NC-03, NC-04, NC-09, NC-12. |
| Material evidence decision | Four top-level masses sum to one on valid pixels; Background and Other are parallel Non-seed evidence. Held-in authority defect NC-09. |
| Reference edges | NC-03, NC-05, NC-08, NC-12. Net subtraction itself is correctly weighted in the owning product. |
| Oriented edge traces | Selected reference sources are re-thinned before tracing; no new double-edge defect found. Upstream ridge representation NC-08 remains. |
| Seed-boundary confirmation | Oval voting/centre calculation is bounded and explicit; no new concrete defect found. |
| Boundary confidence and normals | Disabled by default; its outputs are placeholders when bypassed. The enabled heuristic is not calibrated boundary confidence. |
| Grayscale and local lighting | No new defect found; products are heuristic local illumination/reflectance, not intrinsic-image ground truth. |
| Image-quality diagnostics | NC-13. |
| Procedural seed separation | NC-02, NC-05, NC-10, NC-11, NC-17, plus provenance caveat NC-23. |
| U-Net + watershed instances | No direct painted mask reaches inference. NC-15 and NC-23 apply; no reviewed production checkpoint/manifest baseline is available. |
| StarDist seed instances | No direct painted mask reaches inference. NC-15 and NC-23 apply; no reviewed production checkpoint/manifest baseline is available. |
| Wrinkling likelihood | Disabled by default; heuristic only, and depends on the dormant boundary node if enabled. |
| Coat-pattern decomposition | Heuristic class softmax gated by resolved Seed mass; output channels are joint seed/class mass, not unconditional classes summing to one everywhere. |
| Broad colour probabilities | Heuristic Lab class softmax gated by resolved Seed mass; same joint-mass naming caveat as pattern. |

### Toolbox/shelved graph

| Node | Audit disposition |
|---|---|
| Circle candidates | Disabled; relies on NC-13 sensor-noise evidence if restored. |
| Lightening derivative upper cutoff | Disabled threshold diagnostic; no new defect. |
| Darkening derivative upper cutoff | Disabled threshold diagnostic; no new defect. |
| Distance-peak candidates | Disabled legacy proposal branch; no new defect. |
| Seed identification | Disabled fusion of legacy candidates; no new defect. |
| Touching-seed split likelihood | Disabled heuristic; depends on dormant boundary confidence. |
| Instance colour masks | Exact annotation leakage and hidden dependency NC-01. |
| Multiscale ellipse likelihood | Disabled heuristic over provisional instances; inherits NC-01 if restored together. |
| Instance-assignment confidence | Disabled heuristic; inherits provisional-label and dormant-boundary limitations. |
| Per-seed radial profiles | Disabled; inherits provisional-label limitations. |
| Proposal disagreement | Disabled; no new arithmetic defect beyond input limitations. |
| Occlusion/contact graph | Disabled; proximity graph is not evidence of physical occlusion. |
| Seed-coat damage likelihood | Disabled heuristic; no independently calibrated damage model. |
| Human review | Not implemented. |
| Measurements | Marked implemented but unreachable behind Human review; its latent geometry defect is NC-24. |
| Coat and condition | Not implemented. |
| Lot aggregation | Not implemented. |
| Final output | Not implemented. |
| Calibration residual risk | NC-14. |

## What the existing tests prove—and what they do not

Strong current coverage includes:

- exact active/toolbox node inventory and connection contracts;
- overlay-to-node and overlay-to-output-port identity;
- parameter section completeness and persistence;
- at least one syntactic consumer for each exposed control;
- graph downstream invalidation behavior;
- material probability mass conservation and non-overwrite behavior;
- reference-edge probability calibration and subtraction controls;
- procedural fitting loss behavior and learned inference target isolation;
- exact wavelet reconstruction.

Important gaps exposed by this audit:

- parameter-consumer tests search by attribute name, not calculation owner;
- input contract tests validate authored wires, not hidden runtime arguments;
- geometry tests do not compare width/area definitions against ground truth;
- cache tests do not verify that runtime computed/reused stages match graph
  invalidation for every individual setting;
- no metamorphic tests assert that annotation ID/shape perturbations leave
  inference-only nodes unchanged;
- no scale-family fallback test covers metric failure plus reliable imperial;
- no ridge-width invariant exists after working-to-source restoration;
- material reliability has no out-of-fold calibration test;
- optional learned branches lack committed reviewed-image checkpoint coverage.

## Recommended remediation order

1. Fix NC-01 and add global annotation-leakage/metamorphic tests before any
   dormant instance branch can be restored.
2. Fix NC-02, NC-17, and NC-24; recompute final statistics (NC-10), and add
   exact synthetic geometry boundary tests. These directly explain implausible
   procedural shapes passing hard controls and prevent the same error from
   reaching final measurements.
3. Repair graph ownership/cache boundaries NC-03 and NC-06; assert exact
   computed/reused node identities for every node parameter.
4. Make net-edge use authoritative everywhere (NC-05) and preserve true
   full-resolution ridge topology (NC-08).
5. Honor automatic-background authority inside prototype fitting (NC-04), then
   replace held-in material reliability with out-of-fold calibration (NC-09).
6. Add imperial fallback NC-07 and explicit unavailable-prototype semantics
   NC-12.
7. Fix effective centre validation NC-11, learned GPU cache NC-15, and raw
   source fingerprinting NC-16.
8. Redesign image-quality and calibration-risk diagnostics (NC-13/NC-14) before
   giving those fields authority in learned or proposal calculations.
9. Repair the audit tool, fixture manifest, API defaults, and bookkeeping so a
   clean test/audit baseline is trustworthy.

## Verification record

- Focused node/calibration/material/edge/procedural suite: **216 tests passed**.
- Full `unittest` discovery: **527 tests run in 863.456 s; 524 passed, 2
  skipped, 1 failed**. The sole failure is NC-21: the manifest references the
  absent `images/IMG_9689c.JPG`. No calculation assertion failed.
