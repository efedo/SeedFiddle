# Analysis reliability redesign (2026-08-22)

## Scope and evidence

This change set responds to the eleven numbered review findings concerning
foreground evidence, ruler calibration, vessel geometry, reference-seed
display, hue and wavelet diagnostics, learned-edge tracing, procedural shape
failures, probability interpretation, redundant image controls, and reference
persistence. The supplied screenshots are treated as observations of the
current application, not as executable instructions.

## Plan

1. Audit probability equations before changing thresholds. Determine which
   outputs are independent evidence models and which are classes of one shared
   model, then remove any mathematical coupling that can suppress reviewed
   foreground evidence.
2. Split ruler analysis into reviewable evidence stages. Retain the likely
   outline, metric tick family, imperial tick family, and likely scale-number
   components; derive metric scale only from a sufficiently regular metric tick
   family.
3. Preserve a mildly elliptical vessel fit after the circular search. Use the
   ellipse for the review overlay and its major semi-axis as the conservative
   legacy radius supplied to circular downstream masks.
4. Make the reference-seed display describe the components that were actually
   measured rather than drawing a synthetic example marker.
5. Add GPU-resident hue-only and stationary wavelet products, with host transfer
   only when the user selects an overlay. Require algebraic reconstruction from
   all enabled wavelet details plus the final residual.
6. Fix the locally normalized edge display conversion and make the ridge source
   for oriented traces an explicit computational setting.
7. Apply hard procedural geometry constraints to inferred candidates from every
   marker source. Preserve complete reviewed masks as authored geometry, while
   treating older tiny annotation blobs as centre markers. Expose per-instance
   measurements through click selection.
8. Jointly normalize Background, Foreground, and Other *reference material*
   prototype scores, retain a no-match mass, and standardize low/high colour
   keys in the overlay legend. Keep the separate broad colour/noise models
   independent because they are not outputs of one common classifier.
9. Remove the duplicate image-list button and retain the top-bar/File-menu Open
   images command.
10. Restore per-image reference buffers on every image switch and render them
    immediately even when the corrected analysis cache is absent.
11. Add focused regression tests for each equation or state transition, then run
    the complete `unittest` suite.

## Reasoning and design decisions

### Foreground evidence is not a three-way softmax

The broad foreground-colour, background-colour, and Other-colour nodes are
different image-local evidence models with different training sources and
feature equations. Forcing them to add to one would convert uncertainty or
shared pale colours into artificial competition. Their values therefore remain
independent and are not expected to sum to 1.

The previous foreground implementation added the reviewed prototype score to
the base score in log-odds space. If the background-distance/shadow model drove
the base probability close enough to zero, even a strong exact reference match
could remain low. Reviewed reference evidence is now fused by probabilistic
union:

`P(reference evidence) = 0.985 * (1 - exp(-4*w*P(reference match)))`

`P(foreground) = 1 - (1 - P(base)) * (1 - P(reference evidence))`.

This does not force painted coordinates to one. It gives every matching painted
or unpainted pixel the same evidence and prevents an unrelated low base score
from vetoing it. Safely inset annotated seed interiors participate by default;
the option remains editable.

The material-reference prototype node is different: Background, Foreground,
and Other are banks in one common descriptor and are evaluated by the same
equation. Those scores are jointly normalized with an explicit unknown mass.
They can sum to less than 1 when none fits, but one class can no longer look
strong merely because its separately normalized competitor was suppressed.

### Ruler calibration is tick-family calibration

A straight plastic outline, printed numbers, barcode, and the reverse imperial
scale all create high-contrast line evidence. Reducing all of that to one
one-dimensional profile made a visually obvious ruler unnecessarily ambiguous.
The detector now retains these separate intermediate products:

- green: likely ruler quadrilateral;
- red: the regular metric tick family;
- orange: a separately fitted imperial tick family;
- dark red/orange: compact likely number-glyph components on the corresponding
  sides.

Pixels per millimetre comes from the robust pitch of the accepted metric tick
family and the configured minor-tick spacing. The imperial family can
corroborate the outline but is never averaged into metric scale. An outline can
be shown with `scale_reliable = false`, which prevents a visually plausible but
periodically unsupported ruler from assigning absolute scale.

### Mild vessel ellipticity must not clip content

The high-recall circular bank remains a good way to locate the dish. Within a
narrow annulus around its outer rim, a bounded set of high-edge pixels is then
fit to an ellipse. Fits are accepted only near the circular radius, within a
15% axis ratio and a small centre displacement. The review overlay draws the
accepted rotated ellipse. Existing circular consumers receive the larger
semi-axis as their radius, so residual perspective can enlarge rather than
shrink the valid vessel region and cannot clip a seed at the major-axis ends.

### Hue and wavelets

Hue-only display converts corrected BGR to hue on the tensor device and fixes
display value/chroma at 62%; pixels with negligible chroma are neutral gray so
undefined hue does not create false colour.

The wavelet node is a stationary undecimated B3-spline à trous transform. At
each dyadic dilation, `detail = previous - smooth`; the residual is the final
smooth image. Consequently `sum(details) + residual == source` apart from
floating-point roundoff. Fixed unused outputs are exactly zero, keeping graph
ports stable when fewer levels are selected. Signed details display around
neutral gray; the underlying float tensors remain unscaled and reconstructive.

### Learned-edge tracing

The locally normalized net-edge calculation already produced a continuous
`[0, 1]` tensor, but its display path rounded that tensor directly to `uint8`,
leaving only values 0 and 1 and therefore an apparently black overlay. Display
conversion now multiplies by 255; downstream derivatives continue to use the
unquantized float field.

Oriented traces now expose `trace_edge_source` with four choices: generic
ridges, physical-reference ridges, net-reference ridges, and locally normalized
net-reference ridges. The selected ridge field controls trace membership. The
shared continuous image-gradient tensor still controls subpixel tangents, so
changing semantic ridge source does not replace orientation with a quantized
display image. The node status and overlay legend state the active source.

### Procedural geometry and inspection

The prior hard width, concavity, protrusion, solidity, axis-ratio, and maximum
area checks were bypassed whenever the marker originated from an annotation.
That made annotation-derived centres an unintended licence for watershed to
absorb neighbours into implausible shapes. All inferred candidates now obey
those hard upper/shape rules. A full reviewed annotation is retained as its
authored mask rather than expanded; an old tiny annotation blob remains a
centre marker and its grown candidate must pass the same geometry rules.

Each retained label stores source-resolution area and maximum width plus
diameter-relative width, concavity, protrusion fraction, solidity, axis ratio,
marker score, and final confidence. Clicking a coloured procedural instance
outlines it in yellow and reports those measurements in the procedural node
panel, making any future apparent threshold violation directly auditable.

### Probability display contract

Probability overlays now state their scale in the visible legend. The two
legacy background layers retain their intentionally inverted presentation
(`white = low`, `black = high`). Direct probability layers use `black = low`
and white or maximum semantic-colour brightness for high values. Net edge is an
evidence margin rather than a calibrated posterior and is described as such.

### Image controls and reference restoration

The old Add images button below the list called the exact same handler as Open
images in the toolbar and File menu; it is removed. Open images remains the one
batch-add/open command.

Loading a source image necessarily clears the graphics scene. The application
already retained draft and applied reference arrays per canonical image path,
but when no corrected result was cached it cleared analysis after restoring the
arrays and never repainted them. The no-analysis path now explicitly renders
the restored material and instance layers on the source scene. A later analysis
replaces that base with the corrected image and renders the same authoritative
arrays in corrected coordinates.

## Verification criteria

- A strong painted or annotated-seed colour match raises foreground probability
  even when the base foreground score is very low.
- Material-reference class probabilities are bounded, jointly normalized, and
  preserve unknown mass.
- Metric tick pitch, not a ruler endpoint guess, assigns px/mm; the diagnostic
  overlay exposes every retained evidence family.
- Mild ellipses produce conservative bounds and never shrink the legacy vessel
  radius below either fitted semi-axis.
- Hue and every wavelet overlay are selectable through their owning nodes; the
  float wavelet tensors reconstruct the input.
- A nonzero normalized net-edge tensor displays above black.
- Selecting each trace source changes the traced ridge membership.
- No inferred procedural candidate exceeds any enabled hard shape limit, and a
  clicked label reports the measurements used for that decision.
- Every probability legend states low/high presentation.
- Switching images restores visible reference layers without another disk load
  or an error dialog.
