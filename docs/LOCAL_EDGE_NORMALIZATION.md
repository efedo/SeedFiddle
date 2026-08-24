# Locally normalized physical-edge evidence

## Problem

The instance-derived edge classifier currently publishes two useful but
contrast-dependent fields:

```text
physical = edge_support * physical_similarity / normalization
internal = edge_support * internal_similarity / normalization
```

The displayed net field is then:

```text
raw_net = max(physical - k * internal, 0)
```

where `k` is **Internal-edge subtraction**. This is a valid diagnostic margin,
but it conflates two questions:

1. Does the local tangent-strip descriptor look more like a reviewed physical
   boundary than a reviewed internal edge?
2. How much absolute gradient/ridge support happens to be present at this
   location?

The second quantity varies around one otherwise coherent seed perimeter when
the exterior changes from tray to another seed, when coat lightness changes,
or when local lighting and shadow change. A fill threshold applied directly to
`raw_net` therefore has different practical meaning around the same seed.

Global min/max or percentile normalization is not suitable. A few very strong
edges elsewhere in the dish can suppress the seed under the cursor, and a
nearly flat image can have its noise promoted to a full-strength edge.

## Goals

- Preserve the raw Physical, Non-physical, and Net physical diagnostics.
- Add a separate, cached locally normalized net-physical field.
- Make moderate and strong versions of the same semantic boundary more alike
  without turning zero/near-zero evidence into a boundary.
- Preserve the effect of Internal-edge subtraction.
- Keep computation on the PyTorch device and restore only the final compact
  display raster at full crop resolution.
- Use seed-relative spatial scales so behavior is stable across calibrated
  photographs.
- Make every exposed control affect the calculation.
- Allow Smart fill, Shape fill, Trace edge, and procedural separation to use the
  normalized evidence while retaining explicit raw choices.

## Chosen calculation

Let `P` and `I` be the continuous physical and internal-edge fields in `[0, 1]`.
They contain the same multiplicative edge-support factor. Define total learned
edge evidence:

```text
T = P + I
```

and a support-independent positive semantic margin:

```text
M = clamp((P - k * I) / (T + epsilon), 0, 1)
```

Dividing by `T` cancels their shared absolute support while retaining the
physical-versus-internal decision. Locations where neither prototype class
matches remain protected because `T` is also used as the support and absolute
evidence gate below.

### Robust local support envelope

The local window radius is an adjustable fraction of the estimated seed
diameter. A literal sliding median or percentile would require a prohibitively
large unfolded full-raster tensor. Instead, the implementation uses two
constant-memory box-statistic passes on the GPU:

```text
E0 = sqrt(valid_box_mean(T^2))
C  = max(2.5 * E0, absolute_floor)
E  = sqrt(valid_box_mean(min(T, C)^2))
```

The second pass is a winsorized RMS envelope. Isolated extreme edges are clipped
relative to the preliminary local scale, so one strong neighbouring boundary
cannot dominate the normalization window. Box means use integral tensors and a
valid-pixel denominator; their cost does not grow quadratically with the window
radius.

The local automatic gain is:

```text
gain = clamp(target_envelope / (E + epsilon), 1 / maximum_gain, maximum_gain)
U    = clamp(T * gain, 0, 1)
```

This permits both attenuation and amplification, but bounds them symmetrically.
The default maximum gain is deliberately modest.

### Absolute evidence safeguard

Relative normalization alone would make isolated low-level noise look strong.
An absolute smooth gate prevents that:

```text
x = clamp((T - absolute_floor) / max(absolute_floor, epsilon), 0, 1)
A = x^2 * (3 - 2*x)
```

The final continuous field is:

```text
normalized_net = M * U * A
```

Zero evidence remains exactly zero. Evidence below the floor remains zero, and
the transition from the floor to twice the floor is smooth rather than a new
hard contour.

## Default parameters

- **Local radius / diameter:** `0.30`. This estimates a slowly varying local
  contrast envelope without using a completed seed contour.
- **Target local support:** `0.35`. Moderate and strong learned edge evidence is
  moved toward this common range before semantic gating.
- **Maximum local gain:** `2.5`. Gain and attenuation are limited to `2.5x` and
  `1/2.5x` respectively.
- **Absolute support floor:** `0.04`. Very weak total learned evidence cannot be
  promoted merely because its neighborhood is also weak.
- **Working maximum dimension:** the existing reference-ridge working limit.
  Normalization and thinning share one bounded GPU scale and both results are
  restored to the crop dimensions.

These parameters belong to the post-classification **Thinned reference edge
ridge** node. Changing them must recompute that node and its dependents, not the
prototype banks or the two source class probabilities.

## Outputs and consumers

The post-classification node publishes three distinct products:

1. Existing thinned Physical-edge ridge.
2. Existing thinned raw-net ridge.
3. New continuous **Locally normalized net physical edge** and its thinned
   hysteresis ridge.

The image pane keeps all raw overlays and adds overlays for the two normalized
products. Assisted annotation tools receive an explicit normalized source.
Their default **Combined (ridge priority)** source uses the normalized field as
its broad learned component, falling back to raw Physical probability only for
older/incomplete results. Generic ridges and oriented traces retain priority.

Procedural separation receives the normalized continuous field and normalized
ridge. The continuous field replaces the raw physical-minus-internal value in
the positive semantic gate, while raw Non-physical probability remains the
negative-evidence discount. This keeps reviewed internal edges suppressive and
lets coherent weak physical arcs contribute comparably to strong arcs.

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

- The raw Net physical overlay is not redefined or relabelled as a probability.
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
- an internal-dominant arc remains suppressed;
- evidence below the absolute floor remains zero in a locally quiet region;
- maximum gain and attenuation bounds are honored;
- changing only a normalization parameter reuses the prototype probability
  cache and rebuilds the normalized field/ridges and procedural dependent;
- all normalized overlays have the exact crop shape and remain lazy until
  selected;
- every assisted tool can select the normalized source, and Combined uses it;
- procedural prepared-input reuse includes the normalized field; and
- existing raw overlays and raw selectable sources remain unchanged.

After focused tests, the complete `unittest` suite must run. Image-backed visual
validation should compare raw net, normalized net, and normalized ridge around
reviewed seeds in both sparse and crowded fixtures before defaults are tuned
beyond the conservative values above.
