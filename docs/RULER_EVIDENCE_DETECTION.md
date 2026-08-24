# Ruler evidence detection

## Failure analysis

The former detector had two avoidable geometric errors:

1. It detected the ruler orientation at a bounded working size and also tried
   to assign individual 1 mm dashes there. On the 6,240-pixel fixtures, an
   approximately 18-pixel tick pitch became only 3--4 samples. A permissive
   per-tick threshold then omitted otherwise obvious dashes.
2. It used a broad quantile of every edge between the two long ruler sides as
   the longitudinal body extent. The adjacent colour-card edge could therefore
   become the left ruler edge. A symmetric terminal inset compounded the error:
   it prohibited the true tick train when its last dash was close to the ruler
   end and promoted the plastic edge to tick zero.
3. It did measure a transverse run at each expected tick, but assigned the
   semantic class from tick-index modulo first. Measurements that contradicted
   that assumption were replaced by synthetic class lengths, and a later
   clustering pass merely classified those rewritten lengths. This concealed a
   one-sixteenth imperial phase error on `IMG_9533.JPG`.
4. The two coarse parallel lines were reused as the final long ruler edges.
   They often represented the tick-root rows, while both end borders were
   forced perpendicular to them. Transparent plastic edges and mild residual
   projective distortion were therefore missed.

The old `IMG_9546.JPG` regression expectation encoded this bad phase: it
expected the metric span to start near corrected x=3,012 even though the first
visible metric dash is near x=3,076.

## Current method

Ruler evidence is now resolved in separate stages.

1. **Coarse body orientation.** Scharr evidence and a bounded angle accumulator
   find the two parallel long sides. This remains bounded because exact pixel
   phase is unnecessary here.
2. **Full-resolution strip.** Only the compact strip between and just beyond
   those sides is sampled from the full-resolution GPU grayscale tensor. The
   rest of the photograph is not copied to the CPU.
3. **Independent edge rows.** Narrow outer bands fit the metric and imperial
   families independently. The second family is required to occupy the
   opposite ruler edge, preventing a multiple of the metric pitch from being
   mislabeled imperial.
4. **Periodic pitch and preliminary phase.** Autocorrelation supplies a pitch prior. A
   localized lattice fit uses tick-versus-halfway contrast, coverage and the
   major-dash hierarchy. Terminal dashes may be close to the plastic edge; they
   are accepted through repeated support rather than an arbitrary symmetric
   inset.
5. **Every individual dash and hierarchy alignment.** Local maxima are followed
   across the complete expected train, permitting slow pitch drift from residual
   perspective. Each transverse dark run is measured before it has a semantic
   class. Robust one-dimensional clustering infers three metric or five
   imperial length ranks. Only then are those independent ranks compared with
   the 1/5/10 mm or 1/16, 1/8, 1/4, 1/2, 1 inch increment pattern. A coherent
   non-zero phase shifts the complete lattice and repeats measurement; on
   `IMG_9533.JPG` this moves the imperial family one sixteenth so full-inch
   marks occupy both endpoints. The reported hierarchy-consistency score checks
   rank agreement and, especially, that the longest measured ticks correspond
   to major increments. Metric or imperial scale is withheld when its hierarchy
   fails. Once validated, the periodic class supplies only a fallback length
   for an actually glare-obscured tick.
6. **Number and unit association.** Dark connected glyph evidence near each
   scale is grouped into printed-label boxes and associated with the nearest
   long divider. The tick lattice supplies the numeric sequence; glyph evidence
   corroborates which inferred numbers and unit labels are actually visible.
   This is geometric semantic recognition, not a fragile dependency on free-form
   OCR. The controlled dual ruler fixes metric labels left-to-right and the
   upside-down imperial labels right-to-left; branding density cannot reverse
   the numbers.
7. **Tick-constrained mildly projective outline.** An expanded full-resolution
   strip searches outward from each tick-root row for the transparent-plastic
   boundary. Top and bottom slopes are fitted independently. Left and right
   borders are independently fitted just outside the terminal ticks, with
   colour-card edges excluded by distance and whole-height continuity. The four
   line intersections form a mildly distorted quadrilateral rather than a
   forced rectangle.

Metric and imperial pixels-per-millimetre are calculated independently from
their own minor-tick pitches. Metric remains authoritative and imperial is
never averaged into it. Their symmetric percentage disagreement is reported in
the overlay and discounts confidence when it is large.

## Overlay semantics

- Green: the four fitted ruler edges.
- Red: all mapped metric dash segments, with inferred minor/intermediate/unit
  classes and emphasized 10 mm dividers.
- Orange: all mapped imperial dash segments, with inferred fractional/inch
  classes and emphasized inch dividers.
- Dark red/dark orange: recognized metric/imperial number and unit labels,
  associated with their inferred divider ticks.
- Summary text: separately calculated metric and imperial pixels/mm, their
  symmetric percentage disagreement, and both measured-length hierarchy scores.

## Regression contract

The synthetic suite covers rotation, a competing reverse scale, barcode
distractors and locally hidden minor dashes. The real `IMG_9546.JPG` fixture
must map all 151 metric ticks and all 97 imperial ticks. Its outline must enclose
both terminal trains while allowing small independent side slopes. The real
`IMG_9533.JPG` fixture additionally locks the corrected imperial phase, all five
monotonically increasing tick-length classes, full-inch dividers at indices
0/16/…/96, and right-to-left labels 6 through 0.
