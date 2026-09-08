# Hierarchical material-evidence redesign

## Problem statement

Seed Fiddle previously exposed several useful but incompatible evidence maps:
foreground colour, foreground texture, background colour, background texture,
Other colour, Other texture, and reference-prototype similarities.  There was
no single material decision.  The foreground colour node owned one binary mask,
while procedural separation assembled a different permissive combination from
the same inputs.  `Other` was implemented as a contrastive subtraction inside
both foreground and background models, even though glass rims may legitimately
look like background.  A one-class reference-prototype bank was also normalized
as though its similarity were a multiclass probability.

Those choices made the semantics hard to inspect and created asymmetric failure
modes.  In particular, procedural occupancy used the maximum of two foreground
maps and the minimum of two background maps.  Weakness in any background map
therefore removed non-seed evidence, while strength in either foreground map
admitted a pixel.  Suppressing background with `Other` could consequently make
an Other region *more* eligible for procedural seed material.

## Decision hierarchy

The redesigned chain has one authoritative **Material evidence decision** node.
It makes two decisions in order:

1. **Seed versus Non-seed.** Background and Other are both positive sources of
   Non-seed evidence.  Neither is defined as the absence of the other.
2. **Background versus Other**, conditional on Non-seed. A second four-way
   normalization publishes resolved Background, resolved Other, explicit
   Background/Other ambiguity, and unknown subtype. A glass rim compatible
   with both classes therefore stays visibly ambiguous.

Raw evidence maps remain independent and do not need to sum to one.  The final
decision publishes four mutually exclusive masses that do sum to one at every
valid pixel:

- resolved Seed;
- resolved Non-seed;
- conflicting/ambiguous evidence;
- insufficient/unknown evidence.

Let `s` be aggregated Seed support and `n` the union of Background and Other
support.  Before normalization the four masses are:

```
Seed       = s (1 - n)
Non-seed   = n (1 - s)
Ambiguous  = 2 s n
Unknown    = u (1 - s) (1 - n)
```

where `u` is an exposed unknown-evidence weight.  The factor of two makes a
strong contradiction visibly ambiguous instead of allowing a small arbitrary
difference to become a confident class.  Temperature is applied to the four
positive masses before their final normalization.  Invalid pixels are zero.

Within each semantic class, colour, directional texture, and valid
multifeature prototypes are fused as reliability-weighted support. Zero means
that a source supplies no support for its class and one means maximum support;
these raw maps are not complements and are not required to sum to one. A
missing source contributes nothing. This
avoids both the old permissive `max` and the opposite failure in which one
missing/weak optional modality vetoes all other evidence.

Colour and directional-texture sources are target-only compatibility models.
Each is fitted and calibrated solely from its own class examples; changing a
non-overlapping sibling reference class cannot change the raw raster. Texture
uses the 95th-percentile robust target distance as its half-support distance.
Semantic counterclasses participate only in the reliability and decision steps
below, never in source-model pixel scores.

Each available source also earns a scalar reliability from the reviewed masks:
the median response on its target class must exceed the 90th-percentile
response on reviewed top-level non-target pixels. A 0.20 margin receives full
authority and a zero or reversed margin receives none. Foreground calibrates
against reviewed Background and Other; Background and Other each calibrate
against reviewed Foreground, never against one another. Thus a broad Other
model that cross-matches many annotated seed interiors cannot veto those seeds,
while legitimate Background/Other glass overlap remains intact. Reliability
scales source contribution and leaves unknown mass; it never changes the raw
diagnostic raster.

## Reviewed versus automatic authority

Painted material regions and safely inset applied instance annotations are
reviewed evidence.  Automatic foreground heuristics and the perimeter ring are
priors, not peers.  Whenever reviewed samples exist, each automatic source's
fit authority decays by reviewed coverage measured in nominal seed areas:

```
authority = floor + (1 - floor) * 2 ** (-reviewed_seed_areas / half_life)
```

The floor and half-life are exposed on the source node.  With the defaults,
one seed-area of reviewed material leaves the automatic branch close to its
floor.  Reviewed foreground membership is also supplied separately to the
decision node: a strong reviewed foreground match discounts generic Background
support, so a poor automatic background prior cannot veto an excellent painted
or annotated seed-colour match.  Other remains positive Non-seed evidence and
can still create honest ambiguity when the same appearance was painted as both
Seed and Other.

The perimeter colour and texture sources use the same accepted, colour-filtered
annulus and the same authority rule.  Foreground automatic calibration uses
that perimeter source rather than a second legacy ring.

## Class semantics

- **Foreground/Seed** is positive seed-material evidence.
- **Background** is positive non-seed evidence and a possible non-seed subtype.
- **Other** is also positive non-seed evidence and a possible subtype.  It is
  not subtracted from Background, so legitimate Background/Other overlap is
  preserved and shown as subtype uncertainty.
- Other examples remain negative supervision for the top-level Seed decision.
- Safely inset annotated seed interiors are negative supervision for the
  Background texture classifier as well as positive supervision for Seed.

## Prototype validity

Reference material descriptors are only published as probabilities when at
least two semantic material classes have valid banks.  A single bank is a
similarity model without a contrast class; its medoids remain visible in the
prototype collage, but it publishes no probability raster and cannot influence
the material decision or procedural separation.  This prevents a one-class
annotated-seed bank from assigning high seed probability across most of an
image.

The multivariate prototype matcher first produces Gaussian-kernel similarities,
not posterior probabilities. A typical valid in-class descriptor one robust
scale from a medoid has similarity near `exp(-0.5)`, so directly dividing those
scores by their sum plus unknown mass artificially made even good examples look
middling. Calibration now keeps two questions separate:

```
known confidence = strongest raw similarity / (strongest raw similarity + unknown mass)
conditional class = raw similarity ** class contrast / sum(raw similarity ** class contrast)
published class probability = known confidence * conditional class
```

The exposed class-contrast setting only sharpens relative Background,
Foreground, and Other competition. It cannot turn uniformly weak matches into
known material because known confidence uses the unsharpened strongest score.
Equal Background/Other matches remain equal. This transform receives only the
three score rasters and the valid-image mask: annotation masks and reference
coordinates are not available, and reviewed pixels are never overwritten.

Physical-edge and Non-physical-edge prototype scores use the same two-stage
calibration, with their own exposed edge-class contrast and a 0.10 unknown-edge
reserve. This replaces the former `physical / (physical + nonphysical + 0.10)`
calculation, which imposed the same artificial ceiling on ordinary Gaussian-
kernel matches. The edge transform likewise receives only score rasters and
the descriptor-valid mask, never annotation targets or coordinates. The
leakage-safe held-out-instance edge fitter optimizes edge contrast along with
descriptor tolerance and geometry.

## Consumers and caching

The final Seed mass and its binary proposal mask replace ad-hoc material fusion
in procedural separation and own the user-facing seed-material overlays.  Raw
colour/noise/prototype maps remain available for forensic inspection and for
learned-model feature contracts whose checkpoint schemas explicitly name those
raw channels.

The new decision is node-cached.  A change to a raw evidence producer
invalidates the decision and its dependents; a decision-only threshold or
calibration change does not recompute colour, texture, edge, or prototype
features.  Full-resolution rasters remain on the selected PyTorch device.

The default binary threshold is 0.68 resolved Seed mass. Saved-fixture
diagnostics showed that 0.50 admitted most of a densely filled dish whenever
little visible Background remained, whereas 0.68 retained the sparse-pilot
count regression and restored a conservative automatic proposal area.

## Validation and forensic audit

Validation covers:

- mass conservation and invalid-pixel behavior;
- reviewed-authority decay and reviewed foreground override;
- Background/Other overlap without cross-suppression;
- one-class prototype refusal;
- graph ports, dependencies, settings persistence, cache scope, and legends;
- procedural use of only the final material decision;
- current saved-reference sidecars, including foreground, background, Other,
  and annotation-only cases;
- the full unit-test suite.

This is a calibrated evidence architecture, not a claim that the image-local
models are scientifically calibrated probabilities.  Held-out reviewed masks
are still required to fit reliability weights and probability calibration for
production measurement.
