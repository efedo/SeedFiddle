# Frozen recipe profile coverage

All **223 frozen numerical profiles remain in acceptance**. This mapping reconciles the current `canonical_profiles(validate=False)` order and identities against the immutable V2 baseline; selected fixture recipes differ from their canonical contracts only in parameter values. No graph, CUDA calculation or numerical comparison was executed to produce it.

The complete per-profile mapping is [audit-recipe-profile-map-v1.json](../../artifacts/audit-recipe-profile-map-v1.json). It records each identity/version/provenance and fixture family, every selected control and port/dependency contract, source and frame, array shape/dtype/hash, complete recorded-node hash, lifecycle checks, replay representative and shared prefixes. The artifact is ignored local audit evidence; this reviewed summary is the maintained repository document.

| Profile boundary | Count | Distinction |
| --- | ---: | --- |
| Canonical | 69 | Original/native seed identities, calibrated variants, ROI/diagnostic compositions and standalone adapters. |
| Median ROI | 56 | Historical version 2.0.0 with explicit local median bounds, retaining preceding composition contracts. |
| Reader | 48 | Version 3.0.0: 24 uncalibrated identities in each of color-derived and codec-gray modes. |
| Reader with median ROI | 48 | Version 3.1.0: the same 24 identities and two modes with explicit median ROI. |
| Four-input historical/native | 2 | Same recipe ID/version with distinct original-reference/native provenance and four bound image/mask inputs. |
| **Total** | **223** | **No numerical expectation retired.** |

| Fixture family | Profiles |
| --- | ---: |
| seeds | 147 |
| calibrated | 64 |
| bayes | 1 |
| kmeans | 1 |
| card | 1 |
| recovery | 1 |
| paper | 1 |
| extended | 5 |
| four-input | 2 |

Every profile retains a cold run, a second warm run with all nodes reused, exact cold/warm node products, stable producer/input keys, a verified V5 archive round trip and matching archived keys. The final acceptance assertion compares the complete capture JSON against frozen expectations after explicitly reviewed code/environment identity transitions; array hashes, scalar values, ordering and metadata remain exact. The map contains 3796 recorded array expectations across all stage outputs.

Replay is already consolidated across this matrix: **70 distinct operation plus serialized-port contracts** each have one real source-absent replay owner. Source images and model assets are hidden during replay, and matching status plus zero mismatched elements are required. Other profiles link to the exact owning case/stage; they do not each execute every replay. Unsupported operations stay explicitly listed. A separately archived partial mean boundary is excluded from the 223 complete profiles.

Frame and control boundaries prevent treating similarly named workflows as duplicates. The seed fixture uses a 96 × 128 decoded frame. Calibrated workflows begin at 620 × 900 and record a 96 × 128 local crop, bounds `[600,260,728,356]`, local-to-parent translation `[600,260]`, transformed frame identity and millimetre scale. Median/fill/erode/dilate ROI profiles select `[4,4,124,92]` inside that local frame. The mapping preserves the actual per-stage scale/frame facts rather than inferring them from recipe names. Four-input correction binds separate source/target RGB and label-mask identities. Classifier profiles retain model-source values and input contract distinctions.

Exact source/control/frame and recorded-output comparison identifies **269 repeated prefixes**, of which **182 are maximal for their same member set**. A group retains every selected stage contract and incoming edge through its endpoint; changing ROI, reader mode, ports, source, preparation, runtime, frame or recorded products breaks equality. Whole-recipe wrappers and downstream stages are deliberately outside a prefix key. This is an execution-planning map, not proof that two complete acceptance profiles are interchangeable.

| Representative case | Shared profiles | Prefix stages | Last operation |
| --- | ---: | ---: | --- |
| 6 | 32 | 10 | `calibrated_seed.crop` |
| 13 | 32 | 10 | `calibrated_seed.crop` |
| 0 | 9 | 1 | `external.image_inputs` |
| 1 | 9 | 1 | `external.image_inputs` |
| 2 | 9 | 1 | `external.image_inputs` |
| 9 | 8 | 11 | `threshold.custom_range` |
| 16 | 8 | 11 | `threshold.custom_range` |
| 10 | 8 | 1 | `external.image_inputs` |
| 11 | 8 | 1 | `external.image_inputs` |
| 12 | 8 | 1 | `external.image_inputs` |
| 125 | 8 | 1 | `external.image_inputs` |
| 126 | 8 | 1 | `external.image_inputs` |

A future reduction must map each affected historical version, selected control, provenance/admission path, frame, model/multi-input boundary, cache key and archive/replay contract to an independent retained witness. Common operator names, equal pixels on this one fixture or a common prefix are insufficient grounds to delete a frozen expectation. Prefix sharing may be evaluated while retaining all 223 expected profile records and their complete comparisons.

The baseline remains SHA256 `82861ec3480b7ac9bcaa65be1d27bbc9dc15d928d3e813210e0fad29c5bf6eb1` (143141367 bytes). The mapping artifact is SHA256 `c272d9e5b78fcb1dc050871191e586eb7e4c7519a4b21aa7dc80263c1c11d0a2`. Source hashes and fingerprint definitions are embedded in the artifact. This is a source/data audit of finite lifecycle regression; it makes no new public-source conformance, biological validity or scientific acceptance claim.
