# Edge-supported reference-edge evidence and normalization

## Problem

The instance-derived edge classifier publishes two spatially broad descriptor
compatibility fields that intentionally exclude absolute gradient strength:

```text
Pphysical = compatibility with reviewed physical-edge strip prototypes
Pnonphysical = compatibility with reviewed non-physical-edge strip prototypes
```

The raw diagnostic margin is:

```text
prototype_margin = max(Pphysical - k * Pnonphysical, 0)
```

where `k` is **Non-physical subtraction weight**. This is useful for diagnosing
classification, but it answers only whether a neighbourhood looks like one
prototype class rather than the other. Descriptor context and interpolation
make the field wider than a true edge, so it must not establish boundary
support by itself.

1. Does the local tangent-strip descriptor look more like a reviewed physical
   boundary than a reviewed internal edge?
2. Is the pixel actually on a thinned, image-derived edge ridge?

The authoritative field separates those questions by multiplying Physical
probability by the existing full-resolution thinned true-edge support. A descriptor
halo therefore becomes exactly zero away from a real ridge.

The Physical class output already incorporates competition with Non-physical
and the classifier's known-versus-unknown confidence. Subtracting Non-physical
again is an additional conservative policy, not a probability calibration.
Likewise, dividing Physical by Physical + Non-physical would discard unknown
confidence and could make two very weak matches look confidently Physical.
Keep the subtraction available separately, without applying it by default to
the authoritative probabilistic field.

Global min/max or percentile normalization is not suitable. A few very strong
edges elsewhere in the dish can suppress the seed under the cursor, and a
nearly flat image can have its noise promoted to a full-strength edge.

## Goals

- Preserve raw Physical, Non-physical, and Net prototype compatibility as
  diagnostics, labelled as compatibility rather than probability.
- Publish a separately cached authoritative Reference-edge probability.
- Guarantee that raw descriptor compatibility never directly reaches fill,
  curve, learned-instance, or procedural boundary consumers.
- Make moderate and strong versions of the same semantic boundary more alike
  without turning zero/near-zero evidence into a boundary.
- Preserve the Non-physical subtraction weight in a distinct, optional
  conservative evidence branch; keep authoritative probability invariant to it.
- Keep computation on the PyTorch device and restore only the final compact
  display raster at full crop resolution.
- Use seed-relative spatial scales so behavior is stable across calibrated
  photographs.
- Make every exposed control affect the calculation.
- Allow Smart fill, Shape fill, Trace edge, and procedural separation to use
  only the supported field or supported derivatives; raw choices are not
  exposed as operational edge sources.

## Chosen calculation

Let `P` and `I` be raw Physical and Non-physical prototype compatibility in
`[0, 1]`, and let `R` be the full-resolution thinned true-edge ridge support in
`[0, 1]`. Define:

```text
M = clamp(P - k * I, 0, 1)
E = R * P
C = R * M
```

`M` remains the broad **Net physical-edge prototype compatibility** diagnostic.
`E` is the authoritative **Reference-edge probability**. It is exactly zero
where `R` is zero, including every descriptor/restoration halo pixel.
`C` retains the previous supported subtraction as **Conservative net
physical-edge evidence**, an optional stricter selection of Physical boundaries,
not a calibrated probability. For non-negative `k`, `0 <= C <= E`; at `k = 0`
they are equal. Higher `k` can remove uncertain boundary sections, so this
does not promise a stronger wall or less leakage at the same fill threshold.

### Robust local support envelope

Local normalization acts on `R`, never on the broad descriptor field. The
window radius is an adjustable fraction of estimated seed diameter. A literal
sliding median or percentile would require a prohibitively large unfolded
full-raster tensor. Instead, the implementation uses two constant-memory
box-statistic passes on the GPU:

```text
E0 = sqrt(valid_box_mean(R^2))
C  = max(2.5 * E0, absolute_floor)
L  = sqrt(valid_box_mean(min(R, C)^2))
G  = min(L, max(R, absolute_floor))
```

The second pass is a winsorized RMS envelope. The gain denominator `G` is also
capped at the current pixel's supported response. This matters at a strong-to-
weak transition: the strong arc may lie inside the window, but it cannot raise
the denominator enough to prevent enhancement of its immediately adjacent weak
continuation. Box means use integral tensors and a valid-pixel denominator;
their cost does not grow quadratically with the window radius.

The local automatic gain is:

```text
gain = clamp(target_envelope / (G + epsilon), 1, maximum_gain)
U    = clamp(R * gain, 0, 1)
```

This gain is deliberately one-sided: it permits bounded amplification but does
not attenuate evidence. The separate absolute gate may still suppress
ultraquiet sub-floor noise. Above that floor, the former symmetric gain was the
reason a normalized ridge could appear weaker than its raw counterpart.

### Absolute evidence safeguard

Relative normalization alone would make isolated low-level noise look strong.
An absolute smooth gate prevents that:

```text
x = clamp(R / max(absolute_floor, epsilon), 0, 1)
A = x^2 * (3 - 2*x)
```

The final continuous field is:

```text
normalized_reference_edge = P * U * A
```

The exact full-resolution `(R > 0)` mask is reapplied after working-resolution
restoration. Zero ridge support therefore remains exactly zero even when
bilinear interpolation is used internally. Sub-floor ridge evidence is
smoothly reduced, and support reaches a full gate at the configured floor.

## Default parameters

- **Local radius / diameter:** `0.30`. This estimates a slowly varying local
  contrast envelope without using a completed seed contour.
- **Target local support:** `0.35`. Moderate and strong learned edge evidence is
  moved toward this common range before semantic gating.
- **Maximum local gain:** `2.5`. Enhancement is limited to `2.5x`; a value of
  `1` disables gain, and above the absolute floor evidence is never attenuated.
- **Absolute support floor:** `0.04`. The smooth absolute gate reaches full
  weight here; exactly flat locations remain zero.
- **Working maximum dimension:** the existing reference-ridge working limit.
  Normalization and thinning share one bounded GPU scale and both results are
  restored to the crop dimensions.

These parameters belong to the post-classification **Reference edges** node.
Changing them must recompute that node and its dependents, not the
prototype banks or the two source class probabilities.

## Outputs and consumers

The post-classification node publishes the following relevant products:

1. Raw Physical, Non-physical, and Net prototype compatibility diagnostics.
2. Authoritative **Reference-edge probability** `R × P` and its thinned ridge.
3. Edge-supported Physical and Non-physical compatibility channels for
   consumers that need class distinction without accepting descriptor halos.
4. Normalized Reference-edge probability and its thinned hysteresis ridge.
5. Optional **Conservative net physical-edge evidence** `R × M` and its own
   thinned hysteresis ridge. Only this branch and raw `M` depend on `k`.

The image pane keeps the three raw compatibility overlays for diagnosis and
exposes supported and normalized products in the same blue scale for direct
comparison. Assisted tools expose Reference-edge probability, supported
ridges, normalized supported evidence, generic ridges, and traces. They do not
offer raw Physical compatibility as a barrier. Compatibility fallbacks for old
compact results are multiplied by true-edge support before use.

Smart fill and Shape fill offer Conservative net physical-edge evidence as a
separate explicit choice; all assisted tools also offer its thinned ridge.
Oriented edge traces distinguish `reference_ridges` (authoritative) from
`net_reference_ridges` (conservative). Existing normalized storage/port IDs
containing `net` remain stable for saved graphs, but their displayed names and
calculations now unambiguously mean normalized authoritative probability.
The assisted-tool ID `net_physical` likewise remains a compatibility ID for
Reference-edge probability; the new conservative choice has its own
`conservative_net_physical` ID. No settings schema change or graph rewiring is
needed, and saved subtraction coefficients are preserved.

Procedural separation receives normalized supported probability, normalized
supported ridges, and edge-supported Physical/Non-physical class channels.
Its reference input therefore uses the authoritative probability without the
conservative coefficient. Its separately exposed Non-physical discount remains
part of the procedural boundary-cost decision, not the upstream probability.
Curve confirmation and learned-instance evidence likewise receive only
edge-supported class channels. Raw prototype compatibility is never spatial
boundary evidence outside visualization.

## Hysteresis and fill behavior

The existing normal-direction non-maximum suppression and high/low hysteresis
are applied after normalization. Strong normalized pixels seed a ridge; weak
connected pixels may continue it for the configured reach. This is safer than
lowering one absolute fill threshold globally because disconnected weak noise
does not become a barrier merely by exceeding the low threshold.

Smart fill and Shape fill continue to own their barrier thresholds and pressure
models. They receive a better-calibrated evidence field; the normalization does
not change categorical connectivity, gap sealing, distance falloff, or shape
limits.

## Deliberate non-goals

- Raw prototype compatibility remains available for diagnosis, but it is not
  presented or consumed as a boundary probability.
- No completed contour is assumed before filling.
- No angular sector is forced to contain an edge. Candidate-specific oval
  equalization can be evaluated later, but forcing a peak per sector risks
  inventing boundaries precisely where a seed is occluded or genuinely lacks
  image evidence.
- No normalization is performed on the CPU or on a downsampled Qt display
  image.

## Validation plan

Focused tests must establish that:

- two spatially separated physical arcs with the same semantic ratio but
  different absolute amplitudes become closer after local normalization;
- an internal-dominant arc remains limited by its low Physical probability,
  without a second class subtraction or a forced zero;
- zero-ridge evidence remains zero and sub-floor support stays smoothly
  suppressed in a locally quiet region;
- maximum gain is honored and no above-floor supported ridge sample is
  attenuated;
- a weak ridge continuation adjacent to a strong arc is enhanced and retained
  by hysteresis while a disconnected weak fragment remains rejected;
- changing only a normalization parameter reuses the prototype compatibility
  cache and rebuilds the normalized field/ridges and procedural dependent;
- all normalized overlays have the exact crop shape and remain lazy until
  selected;
- every assisted tool can select the normalized source, and Combined uses it;
- procedural prepared-input reuse includes both authoritative and normalized
  fields;
- raw compatibility overlays remain available but no raw compatibility source
  is offered to assisted or downstream operations;
- the authoritative full-resolution output equals `R × P` within uint8
  quantization and is zero everywhere `R == 0`;
- conservative evidence equals `R × max(P - k × I, 0)` and has the same
  zero-off-ridge guarantee;
- changing `k` changes only conservative/net products, leaves canonical and
  normalized Physical probability/ridges bit-identical, and reuses the raw
  prototype and gradient caches;
- cached and compact-fallback fill adapters keep the two sources separate;
- every procedural, curve, learned-instance, and fill path receives supported
  evidence rather than raw compatibility.

After focused tests, the complete `unittest` suite must run. Image-backed visual
validation should compare raw compatibility, Reference-edge probability,
normalized Reference-edge probability, and normalized ridge around reviewed
seeds in both sparse and crowded fixtures before defaults are tuned beyond the
conservative values above.
