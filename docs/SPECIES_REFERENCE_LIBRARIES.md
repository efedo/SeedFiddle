# Species reference libraries: implementation plan

Status: implemented on 2026-09-01. The explicitly deferred research extensions
listed near the end of this document remain outside the implemented contract.

## Implemented result

The application now has one shared species-library system for foreground
colour/noise profiles, material and aligned physical/non-physical edge
prototypes, reviewed seed traits, diagnostic shape observations, and the joint
dimensions/shape bank. The implementation includes:

- strict immutable manifests, content hashes, descriptor-schema hashes,
  bounded JSON/NPZ loading, atomic publication, catalogue retirement,
  import/export, forking, and exact-version project pins;
- source-image and annotation provenance, per-source compact contributions,
  deterministic class/source quotas, exact current-image exclusion, grouped
  validation, duplication audits, and product-specific validation tiers;
- production-compatible CUDA descriptor extraction for colour, directional
  multiscale texture, rich material descriptors, tangent/normal edge strips,
  and seed traits, with only bounded selected banks transferred to CPU;
- a project-facing library manager for building, validating, publishing,
  importing, exporting, retiring, forking, and pinning versions, plus explicit
  recovery when a pinned bundle is missing;
- a typed **Species reference library** graph node, provenance/status reporting,
  cache signatures, and current/library/combined controls on every supported
  consumer; and
- the companion plan's reviewed measurement, pose, hierarchy, uncertainty, and
  source-excluded dimensions/shape products in the same artifact and workflow.

Background and Other remain image/capture evidence. Foreground library products
do not alter those maps, and raw colour/noise evidence remains target-only and
independent; semantic contrast still occurs only in **Material evidence
decision**. Reviewed coordinates select training descriptors but never receive
hard-coded output values. The 2-D dimensions/shape implementation is described
in the companion document.

The shape-specific companion plan is
[`REFERENCE_SEED_DIMENSIONS_AND_SHAPE.md`](REFERENCE_SEED_DIMENSIONS_AND_SHAPE.md).
This document owns the common library, publication, provenance, pinning, and
runtime-resolution framework. The companion document owns shape measurements,
uncertainty, pose, hierarchical statistical modelling, and shape consumers.
Neither plan may introduce a parallel persistence or publication path.

## Cross-plan contract

The two plans share these fixed decisions:

1. One immutable species-library version may contain multiple accession-,
   lineage-, and lot-aware shape components, but it is still one exact pin for
   the species. A new accession falls back to the broader compatible component;
   the application never silently selects another library version.
2. `capture_group_id` identifies an observational context such as a camera,
   session, or imaging protocol. `seed_lot_id` is biological/material metadata.
   They are separate fields and may not be overloaded.
3. Generic colour/noise/prototype products use per-component/per-image quotas.
   Shape products instead use one biological contribution per physical seed,
   retain repeated views as correlated observations, and model image/capture
   effects explicitly. The generic `Current-image reference weight` does not
   apply to shape inference.
4. Shape observations require the shape plan's explicit review, visibility,
   pose, exclusion, and physical-seed provenance. Existing connectivity/image-
   edge checks do not imply that a mask is reviewed, complete, or flat.
5. `SpeciesShapeSummary` is a diagnostic product only.
   `SpeciesDimensionsShapeBank` is the uncertainty-aware product defined
   by the shape plan. A summary cannot be silently interpreted as the richer
   prior.
6. Shape publishing and validation use the same immutable artifact, explicit
   promotion, source exclusion, schema hashing, and held-out infrastructure as
   every other species-library product.

## Outcome

Seed Fiddle can turn deliberately selected, reviewed annotations
from several images into a reusable reference library for one species. A later
image of that species can use the frozen library without repainting material,
edge, and seed-trait references. Results remain reproducible because a project
pins one exact immutable library version rather than silently following the
latest library.

The implementation reuses the application's existing, interpretable
classifiers. It does not introduce a server or a neural training pipeline.

## Decisions fixed by this plan

1. **Reviewed image sidecars remain authoritative.** A library is a rebuildable
   derived artifact. Applying or saving an image annotation never mutates a
   library automatically.
2. **Promotion is explicit.** The user chooses reviewed images and publishes a
   new library version from the Learning menu.
3. **Published versions are immutable and content-addressed.** Updating a
   library creates another version. Existing projects keep their pinned result.
4. **Species identity is strict.** Stable IDs from `config/traits.json`, not UI
   display strings, identify libraries. The trait-vocabulary hash is part of the
   compatibility contract.
5. **Foreground, physical/non-physical edge, and seed-trait knowledge are
   species-level.** Background and Other remain image/project/capture-context
   evidence in version one because dishes, cards, surfaces, and lighting are not
   biological properties of a species.
6. **Material colour and noise outputs remain independent target evidence.** A
   library supplies additional target profiles; it does not add semantic
   suppression or make Foreground/Background/Other sum to one. Semantic contrast
   remains exclusively in `Material evidence decision`.
7. **Existing reference-prototype mechanics remain recognizable.** Library
   material and edge banks use the same descriptor schemas as the current
   image-local `Reference texture prototypes` path. They are reusable banks, not
   a second competing classifier.
8. **No annotation-coordinate shortcuts are permitted.** Reviewed masks select
   training samples only. They are never copied into output maps, used as
   position priors, or written over predictions.
9. **Self-evidence is removable.** Every stored contribution retains its source
   image SHA-256 and annotation SHA-256. When analysing an image, library
   contributions from that same image are excluded before fitting/evaluation.
10. **One project pins one library version per species.** A version may contain
    hierarchical shape components for several lineages/accessions/lots, but
    there is no silent fallback to a newer version, and a missing or incompatible
    pin is a visible degraded/error state.

## Scope of the first usable release and aligned extension

All nine rows are implemented products. `SpeciesShapeSummary` remains a
diagnostic type and is never accepted where the richer dimensions/shape bank is
required:

| Product | Source annotations | Consumer | Version-one behaviour |
| --- | --- | --- | --- |
| Foreground colour profiles | Painted Foreground and safely inset annotated seed interiors | Material colour probabilities | Corrected-Lab, source-balanced target profiles |
| Foreground noise profiles | Same Foreground sources | Material noise probabilities | Target-only directional/multiscale profiles |
| Foreground material prototypes | Same Foreground sources | Reference texture prototypes | Existing rich material descriptor and medoid matching |
| Physical edge prototypes | Contours of complete annotated seeds | Reference texture prototypes / Reference edges | Existing tangent-normal strip descriptor, including suggestive interior direction |
| Non-physical edge prototypes | Prominent internal edges safely inset from complete annotated seeds | Reference texture prototypes / Reference edges | Existing edge-aligned internal-edge sampling |
| Coat-pattern prototypes | Reviewed, species-valid coat labels | Reference seed traits | Mutually exclusive conditional probabilities |
| Condition prototypes | Reviewed present/absent condition labels | Reference seed traits | Independent per-condition probabilities |
| Shape summary | Explicitly shape-reviewed, visibility-qualified seed masks | Seed scale estimate | Audited per-seed calibration-qualified maximum-Feret/area plus dimensionless aspect, solidity, and concavity distributions only; no automatic prior |
| Dimensions/shape bank | Shape-reviewed masks plus pose, visibility, physical-seed, and biological-context metadata | Reference seed dimensions and shape | Joint uncertainty-aware hierarchy and pose families defined by the companion plan |

The dimensions/shape-bank row follows the contract in the companion plan.
Uncalibrated pixel dimensions remain
source-local diagnostics and are never pooled as though they were physical
measurements.

Background and Other libraries, learned procedural settings, U-Net/StarDist
weights, and automatic cross-species transfer are explicit non-goals for this
release. A later *capture-context library* may reuse Background and Other
evidence, but it must be separate from the biological species library.

## Ownership and data flow

Library construction is deliberately outside the per-image analysis graph. The
graph only loads and consumes a frozen artifact.

```text
reviewed image + compatible reference sidecar + species vocabulary
                    |
          explicit Promote / Rebuild
                    |
        per-image source contributions
                    |
       validate, balance, aggregate, publish
                    |
       immutable species-library version
                    |
      project pin (ID + version + SHA-256)
                    |
        Species reference library node
          /        |         |       \
     colour      noise    prototypes  traits/shape
```

The graph node consumes the project pin and `Species and metadata` output. It
must not consume the current image's annotation raster. This makes the frozen
library boundary visible in both code and the pipeline editor.

The node outputs compact typed data, not full-resolution rasters:

- `SpeciesForegroundColourBank`
- `SpeciesForegroundNoiseBank`
- `SpeciesMaterialPrototypeBank`
- `SpeciesEdgePrototypeBank`
- `SpeciesSeedTraitBank`
- `SpeciesShapeSummary`
- `SpeciesDimensionsShapeBank` (separate schema and compatibility result)
- `SpeciesLibraryProvenance`

Existing consumer nodes still calculate the full-resolution probability maps on
CUDA. The new node only resolves compatibility, filters sources, and uploads the
compact selected banks to the active device.

## Reference-source policy

Each colour/noise/prototype/trait consumer that can use reusable references gets
an honest `Reference source` control with three values:

- `Current image only`
- `Species library only`
- `Species library + current image` (default when a compatible library is pinned)

Combined mode is a union of source-balanced raw profiles/prototypes followed by
one normal evaluation/calibration pass. It is not an average of already
calibrated probability maps. If one source has no valid examples for a class,
that class uses the other source. If neither has support, the output is marked
unavailable rather than inventing a probability.

The combined mode also exposes `Current-image reference weight`, initially
`1.0`. The weight operates at the source-contribution level. It must not be an
overlay-only control or directly alter reviewed pixels.

Shape uses analogous but statistically distinct modes:

- `Current image only`
- `Species library only`
- `Species-library prior + current reviewed observations`

Combined shape mode updates the selected hierarchical prior with local reviewed
measurements and their uncertainty. It is not a union of profiles, and it has no
generic current-image reference-weight control.
The pinned bank first excludes the current source; the explicit local
observations are then added exactly once. This in-sample local update is shown
in provenance and is excluded from every held-out performance claim.

For colour and noise, every source image contributes a separately fitted target
profile. At inference, compatible source profiles are evaluated independently
and pooled with source-balanced weights. This retains multimodality without
letting an image with a large painted area dominate. It also preserves the
target-only nature of texture evidence and the independence of the raw material
classes.

For material, edge, and trait prototypes, selection happens across the union of
eligible source-tagged medoids with a per-image cap. The current descriptor,
similarity, known/unknown, and class-competition equations remain downstream.

## On-disk design

Use `QStandardPaths.AppDataLocation` to choose the per-user root rather than
hard-coding a Windows path. The conceptual layout is:

```text
<application-data>/Seed Fiddle/species-libraries/
  <species-id>/
    index.json
    <library-id>/
      <version>/
        manifest.json
        banks.npz
        provenance.npz
```

An exported portable library is a zip bundle with the same manifest and arrays.
Import validates the complete bundle before atomically installing it. Numeric
archives are loaded with `allow_pickle=False`; JSON parsing uses the same size,
depth, duplicate-key, finite-number, and path-safety discipline as project
masters.

`index.json` is a convenience catalogue only. Draft and retirement state lives
there: retirement hides an immutable version from new selection without
altering its artifact. A project pin resolves by exact ID/version/content hash
and does not trust the index as authority.

### Manifest contract

`manifest.json` contains:

- format name and schema version;
- stable library ID, monotonic version, and content SHA-256 (calculated over the
  canonical manifest with that field omitted plus the exact array payloads);
- species ID, species display name at build time, and trait-vocabulary hash;
- Seed Fiddle version and creation timestamp;
- descriptor-schema IDs and hashes for colour, noise, material prototypes,
  edge strips, traits, and shape summaries;
- extraction settings and aggregation settings hashes;
- required calibration domain, including corrected-colour-space version;
- one record per source image: source SHA-256, annotation SHA-256, safe display
  label, capture-group ID when supplied, source dimensions, and review status;
- optional stable lineage/group, accession, and seed-lot IDs, kept distinct from
  capture group, plus the declared level at which each value applies;
- for shape products, compact per-seed records containing seed ID, optional
  cross-image physical-seed ID, shape-review state, visibility, pose, exclusion
  reason, calibration association, and measurement/uncertainty schema hash;
- per-product/per-class image, seed, raw-sample, retained-prototype, and effective
  weight counts;
- array names, dtypes, shapes, byte counts, and SHA-256 values in `banks.npz`;
- validation split definition, metrics, warnings, and publish eligibility; and
- immutable published status.

Absolute source paths are not portable metadata and will not be embedded in an
export. Optional source locations stay in the local catalogue only.

`banks.npz` contains inference data. `provenance.npz` contains compact source
associations, medoid source coordinates, and display thumbnails used for audit
views; inference must not depend on it.

### Source contributions, not one irreversible aggregate

The library must retain compact per-source contributions. A single final mean or
prototype bank would make exact self-image exclusion impossible. Each profile,
mixture component, prototype, and shape observation therefore carries a source
index and, where applicable, a seed ID. Shape observations also carry an
optional physical-seed ID so repeated views remain correlated. Runtime assembly
filters the current source SHA-256 and then applies product-specific
deterministic balancing or hierarchical assembly.

This design also permits a later library version to be rebuilt without the
original full-resolution images, provided all required compact contributions
were retained. The authoritative annotations are still needed to change the
extraction schema.

## Building a library version

### 1. Select and validate sources

The library manager lists project images whose applied reference archive:

- matches the current source image fingerprint and dimensions;
- declares the selected stable species ID;
- uses a compatible sidecar and trait-vocabulary schema;
- contains at least one supported reviewed evidence class; and
- is not a duplicate source image.

The user explicitly selects sources and may assign a capture-group label (for
example camera/session). Biological lineage/group, accession, and seed lot are
separate metadata. Invalid sources are reported individually and are never
silently skipped during publishing.

A new draft may start empty or fork an installed version. Forking retains its
compact source contributions, after which the user can add reviewed images from
the current project or remove prior sources before publishing a new immutable
version. The published parent is never edited.

### 2. Extract deterministic per-image contributions

Extraction invokes shared production descriptor functions rather than copying
their equations into a library module. It records the exact feature and settings
hashes. Full-resolution reusable intermediates remain on CUDA; only compact
profiles, medoids, statistics, and provenance are transferred to CPU for
persistence.

Sampling is bounded at three levels:

1. an equal initial budget per reviewed seed or painted region component;
2. an equal effective budget per source image within a class; and
3. a configurable maximum retained bank per class.

Farthest-first/medoid selection preserves descriptor diversity after those
quotas. Duplicating pixels, a seed mask, or an entire source record must not
increase that source's total class weight.

These quotas apply to material evidence and prototype banks. Shape extraction
does not pretend repeated views or many seeds from one image are independent:
one physical seed supplies one biological contribution, repeated views are
grouped, and image/capture calibration is retained as a shared observation
effect. Hierarchical accession/lot balance and representative-sampling audits
replace a generic per-image weight for shape.

### 3. Validate without leakage

Every candidate version runs grouped validation:

- leave-one-image-out for all evidence families;
- leave-one-capture-group-out when two or more groups exist;
- leave-one-physical-seed-out for repeated-view shape data;
- leave-one-accession-out for shape transfer when enough accessions exist;
- per-seed rather than random-pixel splits for trait and shape results;
- current production image-local evidence as a recorded baseline;
- class coverage, calibration, unknown rate, and disagreement summaries; and
- descriptor/domain outlier checks for each source.

No held-out image may contribute its profiles, prototype medoids, calibration
statistics, scale distribution, thresholds, or fitted procedural parameters.
The validator uses the same SHA-based exclusion seam as normal analysis.

A library with sparse support may be saved as a `draft`. Publishing is allowed
with conspicuous warnings, but the manager distinguishes:

- `provisional`: fewer than three independent source images for a product/class;
- `validated`: at least three images and successful leave-one-image-out checks;
- `multi-context validated`: successful held-out capture-group checks.

These labels communicate evidence strength without inventing an arbitrary claim
that a small library is unusable.

Validation tiers are reported per product. A library may be validated for
Foreground colour yet provisional or unavailable for dimensions/shape. The
shape product also reports effective physical-seed, accession, pose, and
visibility counts; three images alone do not validate a hierarchical shape
model.

### 4. Preview and publish

Before publishing, the manager shows source balance, class coverage, rejected
sources, descriptor outliers, validation metrics, and the exact products that
will be available. Publishing writes a temporary directory, verifies every
hash, atomically installs the immutable version, and updates the catalogue last.

## Pipeline integration

### New node

Add `Species reference library` immediately downstream of `Project` and
`Species and metadata`. `Project` gains a typed library-selection output; the
metadata input provides the stable species ID. The node reports:

- pinned library ID, version, and abbreviated content hash;
- selected species/lineage/accession/lot shape path and every broader fallback
  actually used;
- compatibility for each product;
- eligible and excluded source-image counts, including self-image exclusion;
- per-class source/seed/prototype counts;
- validation tier and warnings; and
- device/cache residency.

The node has no probability-map overlay. Its inspector provides a provenance
summary and an action that opens the library manager. Raster overlays remain
owned by the calculating consumer nodes, preserving the overlay/node connector
contract.

### Existing-node changes

1. `Material colour probabilities` accepts a species Foreground colour bank and
   adds the reference-source controls to its Foreground section. Background and
   Other calculations are unchanged.
2. `Material noise probabilities` accepts a species Foreground noise bank and
   adds equivalent controls to its Foreground section. All three outputs remain
   target-only.
3. `Reference texture prototypes` accepts species material and edge banks,
   source-balances them with any current-image banks, and retains all existing
   probability and provenance overlays.
4. `Reference edges` continues to consume evaluated physical/non-physical maps;
   no library-specific edge equation is introduced there.
5. `Reference seed traits` accepts the compatible species trait bank. Coat
   classes still normalize only across available coat classes; each condition
   still requires reviewed-present and reviewed-absent support.
6. **Reference seed dimensions and shape** displays a compatible
   `SpeciesShapeSummary` as a diagnostic and consumes
   `SpeciesDimensionsShapeBank` through the prior/local-update modes defined
   above. It may not use a basic summary as a procedural prior.

### Cache and cancellation contract

The cache signature of each direct consumer includes:

- library content hash;
- current-image exclusion hash;
- product descriptor-schema hash;
- selected biological-context path for hierarchical shape products;
- reference-source mode and local-source weight; and
- existing node parameters.

For shape, the cache uses the companion plan's local annotation/review,
physical-seed, pose, visibility, biological-context, calibration, measurement,
and contour-schema hashes; `local-source weight` is omitted because that control
does not exist for shape.

Changing a library pin invalidates only the library node, its direct consumers,
and their descendants. It must not recalculate calibration or other upstream
nodes. Compact banks are cached per content hash/device/dtype and uploaded once.
Image-switch cancellation applies during source filtering, bank assembly, and
GPU evaluation; a late result may not install into the new image.

## Project persistence and portability

Bump the project-master schema from version 1 to version 2 and add an optional
pin:

```json
{
  "species_library": {
    "library_id": "stable-uuid",
    "version": "1",
    "species_id": "lupinus_mutabilis",
    "sha256": "...",
    "resolution": "user_catalogue"
  },
  "biological_context": {
    "species_id": "lupinus_mutabilis",
    "lineage_group_id": null,
    "accession_id": null,
    "seed_lot_id": null
  },
  "capture_group_id": null
}
```

Version-one projects load unchanged with no pin or structured biological
context. Saving them writes version two. Project-level biological and capture
context supplies distinct defaults; an image record may override either.
Whether mixed images require a per-seed
accession/lot override remains a user decision in the companion plan and must be
resolved before the final schema is frozen. Reference-region archives remain
version three until the shape-annotation milestone introduces its separately
versioned migration.

When opening a project:

- an exact installed match is used;
- a missing version offers Locate/Import but never substitutes latest;
- a hash mismatch is an error, not a warning-only success;
- a species or vocabulary mismatch disables the incompatible products; and
- `Current image only` consumers can continue, while library-dependent consumers
  show why they cannot run.

Export Project may optionally embed the pinned portable bundle under a project
library directory. The project still records the same content hash, so moving a
project does not change its analysis identity.

## User interface

Add `Learning > Species reference libraries...` with these views:

1. **Libraries:** species, versions, status, products, validation tier, projects
   using the version, and create/import/export/retire actions.
2. **Sources:** eligible project images, annotation state, species, optional
   lineage/accession/lot, fingerprint, distinct capture group, included
   products, and validation warnings.
3. **Coverage:** image/seed/sample/prototype counts per class with source-balance
   histograms, plus physical-seed/accession/pose/visibility coverage for shape.
4. **Validation:** held-out metrics, per-source failures, domain outliers, and
   comparison with image-local evidence.
5. **Publish:** immutable-version summary and explicit confirmation.

The Project node and the existing `References and annotations` panel show the
active pin, selected biological/capture context, hierarchical shape fallback,
and whether the current image occurs in the library. On an image that is a
library source, the panel must say that its bank contribution was excluded and,
when combined mode is selected, that reviewed local observations were added once
as an explicitly in-sample update.

Consumer-node overlays add audit variants only where they materially help:

- `Current-image reference only`
- `Species-library reference only`
- `Combined reference`
- `Reference disagreement`

These variants are calculated through the same production path. They are not
UI approximations. Existing final overlay names remain stable where practical.

## Module-level implementation map

Create a small `seedvision/reference_library/` package:

- `contracts.py`: immutable manifests, pins, source records, typed product banks,
  compatibility results, and schema constants;
- `persistence.py`: strict JSON/NPZ parsing, hashing, atomic publish, catalogue,
  import/export, and immutable-version enforcement;
- `extraction.py`: calls shared production descriptor seams to create compact
  per-image contributions;
- `aggregation.py`: deterministic quotas, source balancing, medoid selection,
  and runtime self-source filtering;
- `validation.py`: grouped holdouts, metrics, duplication/leakage checks, and
  publish tiers; and
- `service.py`: UI-facing create/rebuild/publish/list/resolve operations with
  cancellation callbacks.

The companion shape implementation supplies shared measurement/model functions
and the `SpeciesDimensionsShapeBank` codec through this package's contracts and
persistence seams. `reference_library/aggregation.py` must not reimplement its
statistics.

Integrate with existing modules as follows:

- `seedvision/persistence/project_analysis.py`: version-two optional pin and
  portable resolution;
- `seedvision/pipeline/model.py`: node, typed ports, consumer dependencies,
  settings sections, and cache-local invalidation;
- `seedvision/cuda/layers.py` and `seedvision/segmentation/baseline.py`: shared
  descriptor extraction/evaluation seams and compact-bank device caching;
- `seedvision/segmentation/reference_edge_fit.py`: source-tagged edge banks and
  self-source filtering without changing physical-edge equations;
- `seedvision/annotation/seed_traits.py`: vocabulary hashing and compatibility;
- `seedvision/ui/main_window.py`: manager workflow, project pinning, error state,
  cancellation, and menu actions;
- `seedvision/ui/pipeline_inspector.py`: library/provenance inspector and grouped
  consumer controls; and
- `seedvision/visualization/`: only the consumer-owned audit overlays described
  above.

Descriptor implementations must be factored into shared pure/tensor functions
before the library extractor calls them. The library package must not import Qt
except in its UI adapter, and it must not fork a second implementation of an
existing classifier.

## Delivery sequence and gates

### Milestone 1: contracts and persistence

- Implement manifest, pin, product-bank, and compatibility types.
- Implement strict, bounded, atomic local persistence plus import/export.
- Add project-master version-two migration and exact pin resolution.
- Add library catalogue and immutable published versions.

Gate: round trips are deterministic; corruption, duplicate JSON keys, unsafe
paths, unsupported schemas, hash mismatches, and species/vocabulary mismatches
are rejected without altering an installed library.

### Milestone 2: extraction and library construction

- Factor shared colour, noise, material, edge, and trait descriptor seams.
- Extract compact source-tagged contributions from valid applied sidecars.
- Implement per-seed/per-image quotas and deterministic aggregation.
- Implement grouped validation, draft/publish tiers, and audit reports.

Gate: rebuilding the same inputs in a different order produces byte-equivalent
inference banks; duplicated pixels/seeds/images do not change source weight; a
held-out source cannot be found in its runtime bank.

The basic `SpeciesShapeSummary` and richer `SpeciesDimensionsShapeBank` use the
reviewed eligibility and canonical measurements implemented by the companion
shape plan. Non-shape products remain independently available when shape review
is absent.

### Milestone 3: graph node and foreground colour/noise

- Add the `Species reference library` node and project pin port.
- Add current/library/combined modes to the Foreground sections of the colour
  and noise nodes.
- Add compact-bank CUDA cache, correct downstream-only invalidation, status, and
  cancellation.

Gate: library-only outputs are invariant to current-image annotation edits;
Background and Other outputs are bitwise unchanged; texture outputs remain
target-only; changing a pin does not rerun upstream calibration nodes.

### Milestone 4: material and edge prototypes

- Feed compatible material and edge banks through the existing prototype path.
- Preserve edge-strip alignment, physical/non-physical competition, interior
  direction, net subtraction, normalization, and ridge products.
- Add source-specific and disagreement audit overlays.

Gate: no painted coordinate is overwritten; prototype equations are identical
for authored and unauthored pixels; current-source exclusion works; physical and
non-physical classes retain their expected edge alignment and class balance.

### Milestone 5: seed traits and shape summaries

- Feed compatible trait banks through `Reference seed traits`.
- Publish and display `SpeciesShapeSummary` calibration-qualified maximum-Feret
  width/area and dimensionless aspect, solidity, and concavity distributions
  only from explicitly shape-reviewed, visibility-qualified observations.
- Keep procedural use of shape priors disabled until held-out tests demonstrate
  improvement.

Gate: coat outputs sum to one only over supported coat classes inside the output
domain; condition outputs remain independent; unknown/unreviewed conditions are
not treated as negatives; shape statistics exclude unreviewed,
uncertain-visibility, cutoff, partial, and explicitly excluded seeds. The
current upper-quartile processing-scale heuristic never enters the summary's
population distribution.

### Milestone 6: complete UI and portable workflow

- Finish manager views, explicit promotion/publish, pin selection, import/export,
  missing-pin recovery, and provenance presentation.
- Add end-to-end documentation and an example workflow.
- Run the complete regression, performance, and GPU-memory suites.

Gate: a clean installation can import a bundle, open a moved project, resolve
the exact pin, analyse an unannotated same-species image, and reproduce the
recorded result without access to the source projects.

## Required tests

### Persistence and security

- schema round-trip, migration, deterministic serialization, and content hashes;
- archive size/dtype/shape/count limits and `allow_pickle=False`;
- duplicate keys, NaN/infinity, zip bombs, path traversal, and truncated arrays;
- atomic-write failure recovery and immutable published-version enforcement;
- missing, corrupt, altered, wrong-species, and wrong-vocabulary libraries; and
- portable import/export with no absolute-path dependency.

### Anti-cheating and statistical invariants

- exact current-image SHA exclusion from every product and calibration statistic;
- grouped holdout by image, seed, and capture group;
- grouped shape holdout by physical seed and accession where supported;
- library-only invariance when current masks, seed IDs, or manual centres change;
- coordinate permutation: identical descriptors at different positions receive
  identical predictions;
- reviewed-pixel values are produced by the same equation as all other pixels;
- duplicating Background data cannot change a Foreground target-only profile,
  and duplicating any one source cannot increase its effective class weight;
- duplicating a shape view cannot increase biological sample weight, and a seed
  lot cannot be treated as a capture group;
- unavailable classes remain unavailable/unknown rather than receiving a
  uniform or synthetic probability; and
- no seed trait, shape label, or instance outline feeds material or edge outputs
  except through its documented training-sample role.

### Pipeline and cache behaviour

- node/overlay/connector ownership contract for every new audit overlay;
- typed graph dependencies and topological order;
- source-mode and library-hash cache signatures;
- pin changes invalidate only consumers and descendants;
- switching images aborts library assembly/evaluation and rejects late installs;
- CUDA compact-bank reuse does not retain stale device/dtype data; and
- full-resolution tensors remain on device except compact persistence/display
  metadata.

### Evidence regression

- Background and Other colour/noise maps are unchanged by a Foreground library;
- raw colour/noise probabilities remain independent and need not sum to one;
- semantic contrast remains only in `Material evidence decision`;
- material/edge library-only results match an equivalent image-local bank within
  tolerance;
- net physical edge still subtracts the configured non-physical weight;
- trait mutual-exclusion and independent-condition contracts remain intact; and
- locked multi-image fixtures compare current-only, library-only, and combined
  modes, including an out-of-domain image.

### UI and workflow

- explicit promotion is required; Apply/Save Reference Regions never publishes;
- draft/published/retired states and validation warnings are visible;
- an exact project pin survives save/open and a project-tree move;
- no latest-version substitution occurs;
- the current source's exclusion is visible in Project and library inspectors;
  and
- controls are disabled or explanatory when their required product is absent.

## Definition of done

Species libraries are complete when all of the following are true:

1. A user can deliberately build and publish an immutable library from several
   reviewed images of one species.
2. Another project can import and pin that exact version.
3. An unannotated image of the same species can consume the supported library
   products at full resolution through the existing CUDA analysis paths.
4. The UI exposes source counts, compatibility, validation tier, exclusions,
   and provenance clearly enough to explain every reused evidence family.
5. The same image never contributes to its own library-only result, and the
   anti-cheating/duplication tests prove that contract.
6. Background/Other semantics and all current material/edge/trait probability
   contracts remain intact.
7. Node-local caching, cancellation, project portability, and exact-version
   reproducibility pass the full test suite.

The full uncertainty-aware shape model has its own definition of done in the
companion plan. Species-library completion requires only that its common
framework can publish, resolve, validate, and audit any supported shape product
without weakening that product's stricter gates.

## Deliberately deferred extensions

These should be decided from held-out evidence after the first release rather
than assumed now:

- whether a separate capture-context library is worthwhile for Background and
  Other;
- whether colour profiles need camera/session-specific sub-banks beyond the
  corrected-colour compatibility signature;
- whether shape summaries improve procedural constraints without suppressing
  legitimate species variation (the companion plan requires an explicit,
  held-out, opt-in milestone before any such use);
- mixed-accession/lot images and per-seed biological-context overrides; the
  implemented contract assigns one biological context to an image, with an
  image-level override of the project default, so mixed material must currently
  be split into separately identified images;
- whether library channels improve U-Net/StarDist training; and
- whether cross-species initialization is safe enough to expose.
