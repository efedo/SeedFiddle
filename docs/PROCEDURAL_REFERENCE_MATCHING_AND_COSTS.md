# Procedural reference matching and costs — 2026-09-05

## Findings

The comparison and fitter previously sorted pairs greedily by **raw intersection
area**, with IoU only as a tie-breaker. Any positive overlap was sufficient. A
large neighbouring prediction could claim the wrong reference, prevent a better
one-to-one assignment, and then receive a large exponential overreach penalty
against the wrong seed. Incidental contact was also sufficient to evaluate an
entire otherwise unreviewed neighbour as a false object.

The fitter already counted missed references, but the overlay only rendered
matched pairs. Furthermore, the old `error / (TP + error)` normalization made
every empty prediction have pixel loss 1, regardless of the missing-pixel
weight. Simply adding a lower missed-seed weight would not have fixed that.
The overlay had its own overreach weights, independent of the fitting controls,
and overlapping error rasters could overwrite one another in iteration order.

## Assignment

1. Measure intersections, reference areas and candidate areas on the same
   bounded categorical grid, with nearest-neighbour reference resampling.
2. Compute pair IoU: intersection / union. Pairs must exceed **Minimum matching
   overlap (IoU)**, default 0.20. A touching edge alone is not a correspondence.
3. Globally maximize the sum of `IoU - minimum_match_iou`, with independent
   unmatched choices, under a one-reference/one-candidate constraint. This is
   a Hungarian assignment on compact metadata. Independent overlap components
   are solved separately rather than constructing a whole-image dense matrix.
4. Do not force a marginal pair just to increase the number of matches. The
   threshold is a geometric identity gate, not an adjustable scoring weight the
   optimizer is allowed to minimize. Exact relabelled shapes still match.

The default threshold is an explicit conservative starting point, not a claim
of validated seed identity. Highly incomplete or displaced predictions can
remain unmatched. The control lets the user inspect that trade-off.

In partial-review mode, predictions with no eligible reference pair are outside
the evaluated subset. This includes unrelated seeds and incidental contacts.
Substantial competing fragments remain evaluated even if they lose the global
assignment. **Annotations cover whole dish** evaluates every prediction instead.
Every reference is evaluated in either mode, including references with no match.
Unannotated seeds must not silently become background negatives.

## Shared cost calculation

Both fitting and the overlay call one implementation. In default pixel units:

| Error | Pixel cost | Overlay |
| --- | --- | --- |
| Matched reference pixel omitted by its candidate | 1 | Blue |
| Candidate pixel outside its matched reference | `2 * (2^min(d / s, 8) - 1)` | Red |
| Reference with no matched candidate | 0.5 for each reference pixel | Amber |
| Incorrect pixel in a candidate perimeter-concavity pocket | Additional 2 | Magenta |

Here `d` is distance to that candidate's **own** reference, not the annotation
union. `s` defaults to half the estimated seed diameter, floored at one working
pixel. Merging into another annotated seed therefore remains an error. Evaluated
unmatched predictions use distance to the annotation union and retain a separate
false-instance count penalty.

An internal-concavity pocket means the exterior-connected gap between a
candidate's perimeter and its convex hull, separately for each connected
component. The surcharge is the intersection of that pocket with **incorrectly
omitted target pixels**. By definition these pockets are outside the candidate,
so this is an underreach surcharge, not a blanket penalty on all overreach near
a jagged outline. Correctly reproduced natural hilum indentations, enclosed
holes, and gaps between disconnected pieces receive no concavity surcharge.
Enclosed holes still incur ordinary underreach. Overreach retains its distance
cost and inference retains its existing protrusion/concavity safeguards.

Costs at the same image pixel **add**, including errors against two different
references. Pixel loss is total pixel cost / fixed total reviewed seed area.
The small instance-count term is still separate:

`0.1 * (overreach_weight * extra_instances + missed_seed_weight * missed_instances) / reference_count`

Thus an entirely empty prediction defaults to loss **0.55**, not zero and not
an invariant 1.1. A correct prediction has zero loss, useful approximate shapes
can improve on omission, and seriously wrong outlines can cost more than simply
missing a seed. Loss is not artificially capped at 1. The missed-seed cost,
overreach amplitude/distance scale, and concavity surcharge are user controls;
they are fixed during optimization and cannot be tuned away by the fitter.

Overlay alpha is total pixel cost / **4**, capped at 1, followed by the toolbar
opacity. This fixed scale is not renormalized to an image's maximum or to the
chosen penalty weight. Colour identifies the strongest cost term, with magenta
marking a nonzero concavity surcharge. Exact unsaturated pixel costs remain in
the comparison result and are tested against the fitter's totals. Counts of
matches, misses, and incorrect concavity pixels appear in the overlay legend;
fit reports also include missed-seed and concavity costs. The count-only term
is not represented as an invented spatial mask.

## Scope, UI and compatibility

No annotation coordinates enter automatic marker discovery, watershed,
candidate geometry, confidence, or selection through this comparison. Cost
settings do not change predictions; they affect the objective when searching
for procedural parameters and the post-inference diagnostic. The existing
annotation-independent fitting input preparation remains unchanged.

The procedural inspector groups these controls under **Reference matching and
fitting costs**. Its existing fit-action controls are synchronized aliases of
the same node values, including whole-dish coverage, rather than a second set
of unpersisted settings. Analysis settings version 19 restores version 18 and
older with the new defaults and preserves the previous overreach controls.
Invalidation remains confined to the procedural node and its dependents.

## Verification

Regression cases cover the greedy assignment failure, exhaustive assignment
optimality with skip choices, incidental neighbouring contacts, complete versus
partial review, adjustable and visible misses, omission versus poor outlines,
incorrect notches versus correct natural concavities and holes, exact summed
overlay/fitting costs, source-label preservation, fit-option propagation, UI
alias synchronization, persistence migration, and unchanged automatic labels,
centres, confidence and boundary costs when comparison controls change.

The 135 focused tests pass. Full unittest discovery ran **595 tests in 268.691
seconds**: 592 passed, 2 skipped, and the sole failure remains the pre-existing
missing `images/IMG_9689c.JPG` fixture named by the committed reference manifest.
The test and manifest remain intact. Compilation, whitespace checks and CUDA
runtime diagnostics pass. A synthetic five-case overlay contact sheet was
visually checked; generated QA artifacts remain under ignored `artifacts/`.
