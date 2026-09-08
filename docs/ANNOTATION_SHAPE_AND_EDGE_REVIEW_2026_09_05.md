# Annotation, ruler and shape review — 5 September 2026

## Changes and interpretation

- **Detected ruler:** the metric line is preserved. An orange imperial line uses
  the independently detected imperial tick roots, between the first and last
  whole-inch dividers (observed tick endpoints if dividers are unavailable).
  Its coordinate transform follows the ruler's own corrected/raw convention.
  Unreliable imperial calibration is dashed and explicitly unconfirmed; it never
  inherits metric scale. Ruler evidence tick lengths and geometry are unchanged.
- **Reviewed seed measurements:** yellow is the initial isolated-reference fit's
  centreline. Grey stroke halos were removed. Cyan endpoint circles and their
  chord mark an annotation's measured maximum span. A green body ellipse supplies
  ovality (major/minor body axes); it is not the maximum-Feret measurement.
  Per-seed labels give span and ovality with measurement sensitivity. The size
  histogram reports the full eligible distribution, mean, between-seed SD and
  uncertainty of the mean. The size–ovality chart has labelled axes, units,
  numeric grid values, pose colours, sample count and uncertainty bars.
- **Eligibility:** a complete outline or explicit **Full length visible** permits
  maximum-span measurement. No top-quartile selection remains. Full-length
  partial outlines do not train ovality, concavity, contour or pose families.
  Complete shape fitting still requires reviewed shape metadata and an explicit
  pose. Exclusions and meaningfully disconnected masks remain excluded. A
  mask touching an image edge needs explicit full-length visibility to contribute
  a span. Unmarked/unknown visibility is not silently inferred as complete.
- **Condition annotations:** **No defects** is mutually exclusive with all
  defect boxes. Any selected defect marks conditions reviewed; no selected boxes
  means unknown, not a negative training example. Persistence retains its
  explicit reviewed bit, preserving the distinction.
- **Annotation tool window:** labelled draggable top bar, bottom-right resize
  handle and retained user size. Updated 8 September: one four-digit numeric
  selector (`Seed:` / colour / number / `Next empty`), with red empty indicator
  and `Show selected only` beneath it. The redundant existing-ID dropdown and
  routine explanatory labels are removed. Hidden tool pages no longer
  determine the brush page's height. The physical-seed-ID editor is removed;
  existing IDs remain intact in archives for backward compatibility.
- **Hilum (updated 8 September):** Pick hilum places the landmark; dragging
  moves it. Direction is automatically derived from the painted seed-area
  centroid to the landmark, and is undefined if the two coincide. The read-only
  angle uses image coordinates (clockwise from right). Moving the point or
  changing/undoing the mask updates direction; Apply persists the derived unit
  vector in the existing metadata field. These gestures edit metadata rather
  than masks. Escape leaves landmark editing. Numeric position controls remain
  available for precision; independent direction controls are removed.
- **Annotation centres (8 September):** small white crosshairs with dark halos
  mark the area centroids of annotated IDs. Partial/disconnected painted areas
  are measured as painted, without inferring a complete seed. Crosshairs follow
  annotation visibility, selected-only mode and annotation opacity; they are
  display-only and are never used as detection markers or prediction targets.

## Actual defects found in shape calculations

`_robust_ellipse` used the reversed axis comparison when converting OpenCV's
ellipse angle to the major-axis angle. This rotated the reported body by 90°,
corrupted the residual used to choose robust support points, and polluted
non-ellipticity/contour signatures. Both the iterative fit and final conversion
are corrected; regression tests check the angle and residual, not merely that
ovality is rotation-invariant.

Opening/closing alone was an inadequate estimate of boundary uncertainty:
it scarcely changes a smooth ellipse. Sensitivity now also includes coherent
inward/outward boundary perturbations. These are **1σ-equivalent sensitivity
estimates**, not empirically calibrated confidence intervals. The uncertainty
overlay shows inner and outer ±2× sensitivity ellipses around the corrected
body, with an explicit explanation that these are measurements of annotations,
not automatic segmentation predictions. Measurement morphology works on compact
seed crops, preserving image-space coordinates.

Concavity now compares rasterized convex-hull area with raster mask area, avoiding
a continuous-contour/pixel-area mismatch. The reported fraction is
`(hull area - seed area) / seed area`. Means weight seeds equally.

For maximum span, reported uncertainty of the mean is
`sqrt(between-seed variance / n + mean(per-seed sensitivity²))`.
The second term is deliberately not divided by n: shared ruler error and
systematic annotation sensitivity must not disappear merely by painting more
seeds. Individual and population variability remain distinguishable.

## Boundary prior and caching

Each complete reviewed seed supplies an equal number of smoothed, equally
arc-spaced contour samples. Signed turning angles are winding-normalized:
positive is convex; negative is concave. Curvature multiplied by maximum span
is scale-invariant; rotation of the image is not learned as a biological prior.
The new **Reference boundary curvature distribution** overlay reports this
distribution, angular quantiles and mean internal concavity.

The measurement-summary output is explicitly wired to Oriented edge traces and
Seed-boundary confirmation. Its broad 10–90% absolute-curvature envelope can
widen the base trace-link angle allowance and softly weights trace continuity
and boundary radius hypotheses. This is suggestive, not a new hard rejection.
Mean concavity is calculated/reported for later procedural penalty control; no
new procedural penalty is silently enabled. No annotation coordinates or masks
are used to create predicted boundaries. Curvature summary values participate
in trace cache signatures; gradients and generic ridges are reused.

## Persistence and compatibility

Reference archives are version 5; versions 1–4 still load, with the new
`full_length_visible` bit defaulting to false. Complete outlines do not need this
bit. The bit participates in undo metadata and shape-only cache invalidation.
Analysis-settings format 18 removes the obsolete top-fraction parameter from
older profiles and supplies the new authored curvature connections.

Shape-summary and dimensions/shape-bank descriptors are version 3. Old shape
products cannot be repaired from their compact statistics, because the original
fit/support selection was wrong. Runtime disables those shape products with a
rebuild explanation; colour/texture products remain usable. Fork recovery does
not relabel obsolete shape observations as new-schema measurements. Rebuilding
shape banks requires the saved source annotations. No source archives or library
versions are overwritten by this code change.

## Why punctate edges can defeat the present classifier

This is a code-level diagnosis, not a claim that every error in IMG_9405.JPG has
been isolated through a new held-out benchmark. The current edge descriptor is
20-dimensional: three zones each containing Lab and lightness/chroma residuals,
three signed cross-edge colour contrasts, tangent coherence and valid support.
Each zone averages five tangent-aligned samples. The matcher uses nearest
diagonal-prototype distance, allowing reversed strip polarity. It already allows
256 edge prototypes per class by default; simply adding more prototypes cannot
recover spatial information that was averaged away.

In particular, repeated dots, a mottled patch and some ordinary boundary/shadow
transitions can have similar strip means and residual amplitudes. There is no
explicit dot radius, dot spacing, blob count, closed-loop support or two-dimensional
pattern descriptor. The default prototype working dimension is 960, so small dots
may also lose structure during reduction. This is a plausible second limitation,
not a measured claim about every speckle in the cited image.

### Recommended next classifier change

1. Preserve enough local sampling resolution to resolve the smallest meaningful
   dots. Extract bounded, edge-centred GPU patches instead of enlarging every
   full-frame descriptor tensor.
2. Add a separately inspectable multiscale texture channel: normalized
   Laplacian/Hessian blob response, blob scale/density, local contrast variance,
   and rotation-tolerant spatial-pattern statistics. Keep the existing aligned
   transition descriptor for physical boundary context; do not discard it.
3. Retain two-dimensional interior/exterior context rather than only strip means.
   A compact patch encoder is a later option if engineered descriptors plateau,
   not a substitute for fixing resolution and sampling first.
4. Train with hard negatives: punctate spots, pale coat transitions, seed shadows,
   and real seed-to-seed contacts. Weight seeds/images fairly and validate on
   entirely held-out images/accessions, with separate performance for punctate
   seeds. Never copy painted labels or reference outlines into predictions.
5. Compare classification at true edges before composing
   `true-edge support × physical compatibility`. Keep texture diagnostics,
   calibration and boundary support separate; do not hide errors with spatial
   overrides or stronger global subtraction.

The classifier itself is intentionally unchanged in this request: the user
asked for diagnosis and recommendations, not an unvalidated classifier redesign.

## Verification

- Full `unittest` discovery: 585 tests in 273.069 seconds; 582 passed, 2 skipped.
  The sole failure is the pre-existing missing `images/IMG_9689c.JPG` manifest
  fixture. No fixture or manifest was removed to conceal it.
- Focused regression coverage includes angle correctness, visibility gates,
  positive boundary uncertainty, source-balanced curvature and node-local cache
  reuse, persistence/settings migrations, schema compatibility, annotation
  selection/conditions, non-paint hilum gestures and imperial transforms.
- Offscreen Qt renders of the resized controls and labelled distribution plot
  were visually inspected. Generated images remain in ignored `artifacts/`.
- Python compilation and scoped `git diff --check` passed.
