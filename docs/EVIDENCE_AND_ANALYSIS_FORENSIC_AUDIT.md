# Evidence and analytical-chain forensic audit

Date: 2026-08-23

## Scope and method

This audit traces every default graph node and every toolbox node from its
authored ports to the runtime consumer. It checks class semantics, reference
provenance, normalization, missing-class behavior, GPU/CPU boundaries, cache
ownership, displayed legends, and whether an exposed setting changes the
calculation it describes. The material chain was additionally exercised on all
six saved reference-region sidecars currently present in `projects/`.

The audit distinguishes three meanings that were previously conflated:

- **raw evidence/likelihood**: an independent model response in `[0, 1]`; raw
  class maps neither complement nor sum to one;
- **resolved probability mass**: one output of an explicit normalized decision;
- **binary proposal**: a thresholded and morphologically cleaned consumer of a
  resolved probability, never a second hidden classifier.

## Critical findings and implemented corrections

| Finding | Failure mode | Correction |
|---|---|---|
| No authoritative material decision | Foreground colour owned one mask while procedural inference used a different `max(Foreground)`/`min(Background)` formula. | Added one hierarchical Seed-versus-Non-seed decision with explicit ambiguity and unknown mass. |
| Automatic and reviewed Foreground were inseparable | A poor automatic prior could suppress an excellent painted or annotation-derived match. | Published automatic and reviewed maps separately. Automatic authority decays exponentially with reviewed coverage; reviewed evidence is fused independently. |
| Automatic authority was initially applied as a low-valued observation | A low-authority source could behave like negative evidence or be renormalized back to full authority. | Authority now scales contribution against the source's nominal weight, leaving unknown mass instead of inventing negative support. |
| Broad sources cross-matched reviewed non-targets | On `IMG_9533.JPG`, raw Other maps strongly matched annotated seeds and initially moved about 73% of their decision mass into ambiguity. | Each source now earns reliability from target median minus the 90th-percentile reviewed non-target response. Reliability changes influence only; raw maps remain visible. Background and Other calibrate against Seed, never against one another. |
| Other was a hidden contrastive subtraction in Foreground, Background, and the colour inspector | Depending on the path, painting Other could erase Background, erase Foreground, or make procedural occupancy more permissive. | Removed every runtime and viewer subtraction. Other is positive Non-seed evidence and an independent conditional subtype. The obsolete public contrastive helper was removed. |
| Background/Other overlap was forced to a 50/50 subtype split | Glass compatible with both appeared resolved even though it was genuinely ambiguous. | Conditional Non-seed classification now has resolved Background, resolved Other, subtype ambiguity, and subtype unknown masses. |
| One-class prototype similarities were published as probabilities | Annotation-only Foreground prototypes could produce high “probability” without any contrast class. | Material prototype rasters require at least two valid material banks. Physical/non-physical prototype compatibility requires both edge banks. One-bank medoids remain visible only as provenance. |
| Distance peaks consumed the raw colour mask despite a graph wire from the resolved material mask | The displayed graph and runtime disagreed; restored toolbox branches could reproduce old overshoot. | Material evidence is evaluated first; distance/circle/fusion and candidate-dependent diagnostics are refreshed afterward without recomputing evidence producers. |
| Seed-interior probability recombined final Seed mass with raw Foreground noise, prototype, and Background maps | Sources were double-counted and ambiguity could be silently undone. | Seed interior is now only a seed-scale smoothing of resolved Seed mass; the obsolete weights and graph wires were removed. |
| Provisional instance assignment used `min(background colour, background noise)` | One weak Background modality removed the assignment barrier. | Restored candidate assignment consumes resolved Non-seed probability. |
| Edge-curve semantic sides used final Seed but raw Background | The two sides came from different decision systems. | Curve confirmation consumes resolved Seed and resolved Non-seed probabilities. |
| Procedural fitting reused annotation-trained evidence while scoring those annotations | The target masks could leak into fitted boundary evidence. | Fitting deliberately excludes instance-trained material/reference-edge evidence and scores only annotation-independent background, generic gradients, and illumination. Inference separately receives authoritative supported reference-edge evidence. |

## Material chain after redesign

### Source models

1. **Automatic Foreground colour** is the legacy perimeter-relative Lab,
   shadow-rejection, and isolated-reference-seed heuristic. It remains visible
   but is treated as a prior.
2. **Reviewed Foreground colour** is the multimodal Lab-frequency response fit
   from painted Foreground and safely inset applied seed-instance interiors.
   Reference coordinates are examples, not hard-coded output pixels.
3. **Foreground texture** is a directional, orientation-aware multiscale
   residual compatibility model fitted only from painted/annotated Foreground.
   Background and Other do not enter its calibration or pixel score.
4. **Background colour** uses painted Background plus the accepted,
   colour-filtered exterior annulus. The annulus is never recruited from inside
   the dish. Its fit authority decays with painted Background coverage.
5. **Background texture** uses only direct painted/annulus target texture.
   Foreground and Other annotations can exclude mislabeled source pixels but
   neither class enters Background compatibility calibration or scoring.
6. **Other colour and texture** are independent positive models. Other texture
   is fitted and calibrated only from Other examples. Foreground and Background
   cannot suppress it, so either reference layer can change arbitrarily without
   changing the Other raster.
7. **Reference material prototypes** use the common 14-channel descriptor and
   publish calibrated competing probabilities only when two or more material
   classes have valid banks. Gaussian-kernel similarity is separated into an
   unsharpened known-material confidence and an exposed relative class-contrast
   term; the calibration receives no annotation masks or coordinates.

### Reliability and authority

Automatic coverage authority is

```
floor + (1 - floor) * 2 ** (-reviewed_seed_areas / half_life)
```

For every available colour, texture, or prototype source, Material evidence
decision—not the raw source model—calculates target-vs-non-target reliability as

```
clip((median(target) - percentile90(non_target)) / 0.20, 0, 1)
```

Foreground uses Background plus Other as non-target. Background and Other each
use only Foreground as their top-level non-target, so Background/Other overlap
does not reduce either Non-seed subtype's authority.

### Normalized decisions

Let `s` be calibrated Seed support and `n` the probabilistic union of calibrated
Background and Other support. Before temperature and normalization:

```
Seed       = s (1 - n)
Non-seed   = n (1 - s)
Ambiguous  = 2 s n
Unknown    = u (1 - s) (1 - n)
```

These four rasters sum to one over the proposal-valid region. Background versus
Other uses the same four-mass construction conditionally, with explicit subtype
ambiguity and subtype unknown. The binary proposal uses resolved Seed `>= 0.68`
and one seed-relative opening/closing kernel.

## Saved-reference validation

The read-only `tools/audit_material_evidence.py` runner validates mass
conservation, authority, source reliability, and distributions within corrected
paint/annotation coordinates. Current results are:

| Image | Reviewed region | Mean resolved Seed | Mean resolved Non-seed | Mean ambiguity |
|---|---:|---:|---:|---:|
| IMG_0002c | painted Foreground | 0.719 | 0.009 | 0.256 |
| IMG_0002c | annotated seeds | 0.697 | 0.010 | 0.276 |
| IMG_9405 | painted Foreground | 0.710 | 0.010 | 0.261 |
| IMG_9405 | annotated seeds | 0.710 | 0.010 | 0.261 |
| IMG_9533 | painted Foreground | 0.741 | 0.012 | 0.218 |
| IMG_9533 | annotated seeds | 0.737 | 0.012 | 0.223 |
| IMG_9546 | painted Foreground | 0.827 | 0.007 | 0.140 |
| IMG_9546 | annotated seeds | 0.823 | 0.007 | 0.144 |
| IMG_9551 | annotation-only seeds | 0.923 | 0.003 | 0.044 |
| IMG_9632c | annotated seeds | 0.917 | 0.003 | 0.037 |

Reviewed Background means remain below the Seed threshold in every applicable
fixture. For example, `IMG_9533` painted Background averages 0.083 Seed and
0.447 resolved Non-seed; `IMG_9546` averages 0.014 Seed and 0.810 Non-seed.
The four top-level masses conserve 255-valued quantized mass to within one unit.

The diagnostic also established that a 0.50 binary threshold admitted 92.8% of
the dense unreviewed `IMG_0002c` proposal region. The audited 0.68 default is
more conservative, retains the 16-object sparse-pilot regression, and leaves
the saved reviewed-seed means above the threshold.

## Calibration and geometric evidence audit

| Chain | Audit result |
|---|---|
| Raw image and metadata | Read-only image decode is cache-keyed by resolved path. Metadata does not influence pixel classification except species/model selection. |
| Colour-card detection and deskew | Card geometry supplies rectification and neutral gains; gamut-safe corrected BGR remains the common source tensor. Calibration residuals are a separate risk diagnostic, not class evidence. |
| Ruler detection and scale | Tick lengths are measured before semantic assignment; their independently clustered ranks must put the longest marks on major increments, and a coherent phase can realign the lattice. Both hierarchy scores are shown. The plastic outline independently fits four close exterior lines as a mildly projective quadrilateral. Each tick family produces an independent pixels/mm estimate; their reported symmetric disagreement is a second sanity check, while only reliable metric scale drives calibration. Failed detection does not fabricate a physical scale. |
| Dish layout | Mild ellipse support is retained for non-fronto-parallel vessels. The accepted vessel defines crop/valid geometry; the material proposal inset is independently explicit. |
| Seed scale | Shadow-resistant maximum-width fits of isolated ruler-reference seeds supply the initial diameter and optional automatic Foreground samples. When sufficient applied annotations exist, the largest configurable fraction of complete annotated maximum widths overrides that estimate; cutoff/disconnected instances remain visible but excluded. Their pixels do not become hard labels. |
| Perimeter Background | Colour and texture consume the same accepted, median-filtered exterior annulus. Texture features are extracted directly in the annulus coordinate frame. Colour and texture remain separate raw evidence inputs to the material decision. The displayed source mask is exactly the fitted annulus; no inside-plate lookalikes are recruited. |

## Edge and boundary evidence audit

| Chain | Audit result |
|---|---|
| Multichannel gradients | Shared Lab/Scharr strength is generic edge evidence only; directed and axial tangent views are alternate encodings of the same field. |
| Surface gradients | Lightening/darkening direction and magnitude are independent seed-scale one-sided slopes. Procedural inference uses surface darkening only through its explicit boundary weight. Ceiling nodes remain optional transforms. |
| Frequency bands and wavelets | Frequency masks are local residual-energy evidence. The undecimated B3-spline wavelet details plus residual reconstruct the source exactly; wavelets are diagnostic and do not silently enter material classification. |
| Generic ridges and traces | NMS/hysteresis localizes generic gradients. Trace continuity measures tangent/curvature-compatible continuation and reports its selected source. It is geometric support, not physical-edge probability. |
| Instance-derived edge classes | Training uses annotation geometry, narrow tangent-aligned interior/edge/exterior strips, signed cross-edge Lab contrast, and adaptive high resolution. Both physical and internal-edge banks are mandatory. Gaussian-kernel scores are calibrated into unsharpened known-edge confidence and exposed relative class contrast; the transform cannot see annotation targets or coordinates. |
| Reference-edge compatibility and support | Raw Physical, Non-physical, and weighted Net outputs are broad prototype-compatibility diagnostics only. Authoritative Reference-edge probability is full-resolution continuous image-gradient magnitude times Physical probability, retaining unknown confidence and without a second class subtraction. Local normalization acts only on gradient support, reapplies Physical probability, and remasks to the exact gradient footprint after restoration. Reference-specific NMS/hysteresis is applied afterwards, not to already thinned generic ridges. The supported weighted margin survives separately as optional Conservative net physical-edge evidence and its own ridge. Changing its weight cannot change authoritative or normalized probability. |
| Procedural boundary cost | Generic edge/ridge/surface/trace evidence nominates boundaries. When semantic classes exist, normalized supported Reference-edge evidence and separately edge-supported class channels gate/discount candidates. Raw prototype compatibility never enters the boundary path. Missing semantic classification remains neutral rather than crushing generic edges. |

## Instance and analytical consumer audit

| Consumer | Material input and status |
|---|---|
| Binary proposal | Thresholded resolved Seed only. |
| Distance/circle/identification toolbox | Resolved binary mask; post-decision staged recomputation keeps runtime equal to graph. |
| Provisional Voronoi masks | Resolved Non-seed assignment barrier plus reviewed instance labels. This toolbox branch is not the default final caller. |
| Seed interior/boundary normals | Seed-scale-smoothed resolved Seed, followed by morphology-gradient and image-gradient boundary evidence. No raw-material double fusion remains. |
| Procedural instances | Resolved Seed is the sole occupancy likelihood. Many candidate shapes are scored for material, boundary, size, width, area, concavity, protrusion, and overlap before global non-overlap selection. Manual centres modify markers explicitly. |
| U-Net and StarDist branches | Raw named channels remain intentionally unchanged because checkpoint schemas bind those channel contracts. They do not receive the resolved posterior unless a future checkpoint version explicitly declares it. |
| Traits | Wrinkling, pattern, and colour operate on reviewed/selected interiors and report per-class diagnostics. Coat damage, radial profile, and several assignment diagnostics remain toolbox nodes and cannot affect default material or procedural decisions. |
| Review, measurement, classification, aggregation, output | These remain authored downstream workflow/toolbox nodes. They are not evidence producers and cannot feed back into segmentation. |

## Caching, persistence, and failure behavior

- All full-resolution material rasters and reusable feature tensors remain on
  the selected PyTorch device; the audit runner downloads only explicit summary
  samples.
- The new decision owns `layer.material_evidence`. Source changes invalidate it
  and downstream consumers; threshold/temperature changes reuse colour,
  texture, gradient, and prototype caches.
- Candidate staging refreshes only distance/circle/fusion and candidate-derived
  instance/curve products. It does not rerun evidence producers.
- Reference sidecars are bound to image identity and SHA-256, schema-checked,
  and loaded in corrected-image coordinates. Invalid archives can be backed up
  and atomically replaced; the source archive is preserved first.
- Material and annotated-instance association remains project-local. Applying
  an annotation updates the in-memory/project sidecar path; the project master
  is still the authoritative bundle when explicitly saved.
- Node timings and cancellation cover the expensive evidence nodes so a failed
  image cannot leave later queued images waiting on an unreported active node.

## Remaining limitations and next validation

1. The raw maps are image-local likelihoods, not scientifically calibrated
   probabilities. Source reliability is an in-project reviewed-mask calibration,
   not an estimate from independent held-out images.
2. A class with no reviewed target/non-target pair retains unit reliability;
   missing class rasters still contribute nothing. More projects should supply
   Background and Other examples inside the analysis region for stronger
   calibration.
3. The public low-level procedural function retains a compatibility fallback
   for callers that omit resolved material. The desktop graph and fitting path
   never use it. Removing that fallback is a future API-breaking cleanup.
4. Learned checkpoints intentionally keep their raw-channel contracts. A new
   checkpoint schema is required before resolved/ambiguity channels can be
   added safely.
5. Reliability weights and the 0.68 proposal threshold should ultimately be
   fit on held-out reviewed masks across species, lighting, vessels, and seed
   densities. The current defaults are supported by the six available saved
   reference fixtures, not claimed as universal calibration.
