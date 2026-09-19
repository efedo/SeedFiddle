# Seed Fiddle: annotation-guided node optimization compliance review

> **Implementation update — 18 September 2026:** the findings below describe
> the historical reviewed revision. The shared node registry, project-wide
> sequential production optimizer, node actions, fixed reference objectives,
> diagnostic-target import, reversible application/history and regression tests
> are now implemented. See [Node optimization](NODE_OPTIMIZATION.md) for current
> coverage, usage, objective definitions and limitations. All 38 computational
> cards have an optimization route; ten cards have explicit exemptions. Runs
> are labelled in-sample adaptation, not independent accuracy validation.
>
> | Finding | Implemented response |
> |---|---|
> | O01 | Shared capability/parameter/exemption registry for all 48 cards. |
> | O02 | Analysis project command and node-local commands share a multi-image topological coordinator. |
> | O03 | Actual production inputs, isolated production caches, and missing/zero-evidence exclusions. |
> | O04 | All 35 eligible procedural controls; ineffective legacy fallback control disabled. |
> | O05 | Fixed contour/interior target buffer, including the compatibility fitter. |
> | O06 | Nineteen eligible edge controls scored through classifier, supported, normalized and ridge outputs. |
> | O07 | Complete/connected/reviewed instance filtering, per-image coverage, immutable targets and explicit in-sample scope. |
> | O08 | Learned decoder tuning through the same production settings and node UI; model-weight training remains separate. |
> | O09 | Shared plan, budget, eligibility, result comparison/overlays, cancel/reject/apply and history dialogs. |
> | O10 | True-producer invalidation, provenance journals, source/staleness checks, rollback and Save As companions. |
> | O11 | Contract, search, Qt, fixed-target and two-image production/application reproducibility tests. |
>
> Conditional diagnostics require reviewed targets; independent geometry is not
> inferred from seed masks. Real-image held-out accuracy remains unestablished.

**Review date:** 9 September 2026  
**Reviewed application revision:** `60a4b8bb3af1c34f54c4945231699fafa496a694`  
**Requirement:** every node with features that can be optimized from reference annotations must offer that option; a project-wide procedural command must sequentially optimize all eligible nodes.  
**Disposition:** additional review only. No optimizer, node, or application behavior was changed.

**The application does not meet this requirement.** The actual Qt inspector offers annotation-guided parameter fitting on only **two cards: Reference edges and Procedural seed separation**. Both have substantial coverage or objective defects. Other annotation-trained nodes automatically estimate models from examples, but do not offer optimization of their exposed settings. The Analysis menu contains only **Run pipeline**; there is no project-wide optimization coordinator. Learned-model training and command-line decoder searches provide useful separate capabilities, but do not fill these gaps.

This addendum extends the [critical application review](D:/Programming/SeedFiddle/docs/CRITICAL_APPLICATION_REVIEW_2026_09_08.md). Its conclusions are based on current source, a fresh inventory of every card and actual Qt inspector, focused regression tests, the full test suite, and small synthetic counterexamples. Proposed objectives below are implementation recommendations, not measured evidence that tuning a particular node will improve real-image accuracy.

## 1. Compliance standard and scope

A compliant optimization option needs a declared target, a bounded set of meaningful parameters, an evaluator that exercises the corresponding production calculation, and a way to review and persist the result. It must explain unavailable supervision or inactive inputs. A button that searches ineffective settings, or a model that simply rebuilds its prototypes using unchanged settings, is insufficient.

Three distinctions matter:

- **Model fitting versus parameter optimization.** Building colour distributions, texture prototypes, a shape prior, or trait prototypes from references is already fitting. It does not optimize prototype capacity, feature scales, fusion weights, rejection thresholds, or other exposed choices. A node can have both operations under one coherent fitting interface, but their scope and evaluation must be explicit.
- **Direct versus downstream supervision.** A gradient, wavelet, or illumination node need not consume annotations directly to qualify. Its parameters can be varied while its dependent classifier or separator is scored against references. The downstream objective and every recomputed dependency must be declared. Comparing a gradient magnitude directly to a binary seed mask is generally the wrong objective.
- **Capability versus present eligibility.** An optimizable card should expose the option even when its current project has too few examples, with a specific disabled-state explanation. Disabled or toolbox cards need the capability when activated; a project command should not silently activate them. Cards with no meaningful target or adjustable calculation need an explicit exemption, not a dummy search.

Do not optimize physical facts, reference identities, annotation completeness declarations, coordinate transforms used to define truth, or the penalties defining success. Runtime and resolution limits should ordinarily be fixed for a run. Discrete algorithm choices and model capacity are legitimate candidates where a fixed evaluation and resource budget permit them. A quick default search may cover a subset, provided the remaining eligible features have an accessible optimization route.

### Inventory and observed entry points

| Observation | Result |
|---|---|
| Default active graph cards | 29, including 4 disabled cards |
| Unused/toolbox cards | 19; all disabled, 4 not implemented |
| Parameter entries | 276 active + 33 unused = 309 |
| Cards with a node-local annotation-fit action | 2 |
| Procedural default search | 16 parameters; custom API whitelist contains 28 |
| Reference-edge default search | 6 parameters |
| Project-wide sequential optimizer | Absent |
| Analysis menu, instantiated Qt window | `Run pipeline` only |
| Separate learning facilities | GUI train/refine U-Net or StarDist; CLI validation decoder searches |

Counts include selectors, resource controls, and scoring-policy fields; **2/48 is not an appropriate compliance percentage**, because some cards have no fit-worthy calculation. The older `NODE_CONTROL_AUDIT.md` checks control exposure and wiring, not optimization compliance, and its catalogue predates this inventory.

The fit actions are hard-coded in [PipelineInspector](D:/Programming/SeedFiddle/seedvision/ui/pipeline_inspector.py:911) and dispatched by node ID in [MainWindow](D:/Programming/SeedFiddle/seedvision/ui/main_window.py:10419). [PipelineNode](D:/Programming/SeedFiddle/seedvision/pipeline/model.py:107) has no optimizer capability, target, or exclusion metadata. The [menu construction](D:/Programming/SeedFiddle/seedvision/ui/main_window.py:2964) confirms the missing project command. “Fit workspace” and the canvas “Fit” control only change the view.

## 2. Node-by-node assessment

**Status key:** **Missing** = annotation optimization is justified but no card action exists; **Partial** = card action exists with gaps described below; **Conditional** = a target mapping, additional reviewed labels, or a suitable downstream consumer is required; **Exempt** = no current fit-worthy control; **Planned** = unimplemented, so future compliance must be specified. Conditional is not a permanent exemption.

**Priorities:** P1 = required before claiming compliance or trusting the affected optimization result; P2 = required for the affected secondary/optional workflow, after the core fitting foundation; P3 = improvement rather than a compliance blocker. Priority does not remove a node from the requirement.

### Active graph: all 29 cards

| Card / ID | Controls | Status | Appropriate optimization scope and reference target | Priority |
|---|---:|---|---|---|
| Project / `project` | 0 | Exempt | Owns images, reference editing, and project state. Put the project command here or in Analysis; do not manufacture numerical parameters. | P1 command |
| Species and metadata / `metadata` | 0 | Exempt | Species and specimen facts are supplied metadata, not values to change until segmentation improves. | — |
| Species reference library / `species_reference_library` | 0 | Exempt at this card | Library construction/management exists. Optimize consuming nodes' blend/model settings; freeze library versions and training membership. Reassess if selection or adaptation controls are added to this card. | P1 provenance |
| Ruler detection and scale / `ruler_detection` | 2 | Exempt for exposed controls | `ruler_length_mm` and `minor_tick_mm` are physical facts. Future detector thresholds could be optimized against independently reviewed ruler/tick geometry. Never change the ruler units to improve seed overlap. | — |
| Deskew and colour balance / `deskew_colour` | 4 | Missing / conditional geometry | Evaluate the colour-balance choice against material/trait reference performance. Deskew/perspective limits require reviewed calibration geometry and correctly transformed targets; freeze geometry during ordinary segmentation fits. | P2 |
| Layout detection / `layout_detection` | 16 | Missing | Fit detector thresholds, radius/centre search priors, rim support, and background-band sampling using reviewed dish/rim geometry or a fixed-frame downstream material/instance objective. Fix known outer diameter and scoring coverage. | P1 |
| Hue only / `hue_only` | 0 | Exempt | Fixed conversion/display; no exposed fit-worthy parameter. | — |
| Wavelet decomposition / `wavelet_decomposition` | 1 | Missing | Search level count as a discrete feature choice against dependent edge/material/instance performance. Recompute the selected wavelet consumers. | P2 |
| Reference seed dimensions and shape / `seed_scale_estimation` | 18 | Missing | Model estimation exists. Optimize extraction thresholds and ROI selection where legitimate, empirical scale correction, shrinkage, component complexity, contour modes, and prior use against held-out contours/dimensions and downstream separation. Treat calibration uncertainty as measured input, not a score-reduction knob. | P1 |
| Material colour probabilities / `background_likelihood` | 24 | Missing | Fit colour model capacity, sampling, scale/softness, frequency weighting, chroma contribution, and permitted current/library blend using held-out Foreground/Background/Other examples and valid instance interiors. Automatic distribution fitting alone is insufficient. | P1 |
| Material noise probabilities / `refined_background_likelihood` | 20 | Missing | Fit directional, scale, length, decay, integration, and blend choices with class-labelled material references; assess both foreground and background channels. Freeze working dimensions for a comparable search. | P1 |
| Edge gradients / `edge_gradients` | 18 | Missing | Search method/source fusion, wavelet participation, blur, chroma, normalization, gamma, and ridge thresholds. Score contour localization and internal-pattern rejection, or dependent boundary/instance performance. | P1 |
| Directional surface darkness gradients / `surface_darkness_gradients` | 7 | Missing | Fit smoothing, radius, angular/sample resolution, normalization and gamma against downstream physical-boundary/instance quality; seed outlines alone do not label intrinsic surface darkness. | P2 |
| Multiscale darkness and colour noise / `frequency_noise_masks` | 7 | Missing | Fit fine/medium/coarse/context scales and response calibration through material or edge classification on held-out reference regions. | P1 |
| Reference texture prototypes / `reference_texture_prototypes` | 10 | Missing | Fit prototype capacity, sample minimum, descriptor context/patch scales, similarity and contrast, and permitted library blending against held-out material classes. Rebuilding medoids with fixed settings is not this search. | P1 |
| Material evidence decision / `material_evidence_decision` | 7 | Missing | Fit colour/noise/prototype/unknown weights, temperature, seed threshold and morphology against material labels and complete-instance silhouettes, respecting unreviewed pixels. | P1 |
| Reference seed traits / `reference_seed_traits` | 10 | Missing | Fit capacity, feature context, interior sampling, similarity, contrast and blend against reviewed coat/condition labels on held-out instances. Preserve species vocabulary and distinguish absent labels from reviewed negatives. | P1 |
| Reference edges / `reference_edge_probability` | 23 | Partial | Six classifier/strip parameters are searched. Final evidence subtraction, support normalization, ridge extraction and several bank/blend controls are uncovered; the evaluation target also moves during search. See O05–O06. | P1 |
| Oriented edge traces / `edge_traces` | 10 | Missing | Fit source, diameter, tangent/curvature tolerances, gap, window, sample count, minimum length and junction limit. Match complete instance contours with fixed tolerance, penalizing internal false traces; verify downstream separation. | P1 |
| Seed-boundary confirmation / `seed_edge_curves` | 21 | Missing | Fit radius/arc search, residual and orientation tolerances, vote smoothing/weights, semantic/negative-edge influence and confidence. Use contour and centre detection metrics, including unmatched detections, with explicit treatment of occluded seeds. | P1 |
| Boundary confidence and normals / `boundary_normals` (disabled) | 1 | Missing | Fit boundary width against complete contour location/normal targets derived at a fixed resolution; use a defined tolerance rather than treating every near-edge pixel as exact truth. | P2 |
| Grayscale and local lighting / `illumination_decomposition` | 10 | Missing | Fit illumination, contrast and despeckling parameters through downstream material/edge/instance objectives. Shadow/highlight-specific controls additionally need suitable labels or an actual consumer affected by those outputs. | P1 |
| Image-quality diagnostics / `image_quality` | 1 | Conditional | Fit noise scale against reviewed artifact/quality labels, or downstream proposal quality when a consumer of sensor-noise evidence is enabled. Seed masks alone are not true noise labels. | P2 |
| Procedural seed separation / `procedural_instances` | 43 | Partial | Fit topology, boundary/centre fusion, candidate generation, overlap and shape rejection using one-to-one instance matching on a fixed reviewed domain. Existing search has incomplete coverage and a different input path from live inference. See O03–O04. | P1 |
| U-Net + watershed instances / `unet_instances` (disabled) | 12 | Missing on card; separate backend | GUI training and CLI tuning exist. Expose decoder threshold, separation, topography/boundary/pattern/uncertainty, erosion and area optimization against validation instances, using frozen model outputs. Distinguish decoder tuning from weight training. | P2 |
| StarDist seed instances / `stardist_instances` (disabled) | 8 | Missing on card; separate backend | Expose object/NMS threshold, peak radius, area and candidate-limit optimization against validation instances. Keep checkpoint choice and resource budget explicit. | P2 |
| Wrinkling likelihood / `wrinkling` (disabled) | 1 | Missing with semantic eligibility | Fit scale using reviewed wrinkled/not-wrinkled instance labels as weak supervision; use localization targets if available. The existing `wrinkled` condition provides a possible semantic link, but omission must not automatically mean negative. | P2 |
| Coat-pattern decomposition / `pattern_decomposition` | 1 | Conditional | Fit pattern scale after mapping reviewed trait labels to the diagnostic's pattern classes. Whole-seed category labels can supervise aggregated predictions, but are not pixel-level pattern truth. | P2 |
| Broad colour probabilities / `colour_probabilities` | 1 | Conditional | Fit temperature against explicitly reviewed, compatible colour classes using probability loss. Material Foreground paint does not establish a broad colour category; trait/schema vocabulary alone is not a set of applied class labels. | P2 |

### Unused/toolbox: all 19 cards

These cards should be explicitly skipped by the project command while inactive. Restoring an implemented, optimizable card must make its fitting option available when its inputs and targets are ready.

| Card / ID | Controls | Status | Appropriate optimization scope and reference target | Priority |
|---|---:|---|---|---|
| Circle candidates / `circle_candidates` | 15 | Missing | Fit detection thresholds, radius/separation, evidence weights, merge distance and confidence against reference centres/instances; score recall and duplicate/background proposals. Fix execution resolution. | P2 |
| Lightening derivative upper cutoff / `lightening_gradient_ceiling` | 1 | Missing | Fit maximum slope through boundary/instance performance where this branch is connected. Retain physical-edge recall while suppressing internal pattern responses. | P2 |
| Darkening derivative upper cutoff / `darkening_gradient_ceiling` | 1 | Missing | Same contract for darkening slope; do not assume its optimum equals the lightening cutoff. | P2 |
| Distance-peak candidates / `distance_candidates` | 4 | Missing | Fit blur, neighbourhood, minimum depth and proposal radius against annotated seed centres or one-to-one instance detections. | P2 |
| Seed identification / `identification` | 1 | Missing | Fit distance-proposal confidence against matched/unmatched reference objects or downstream instance error; changing a displayed confidence without a defined loss is insufficient. | P2 |
| Touching-seed split likelihood / `touching_split` | 1 | Missing | Fit neck fraction using touching-instance examples and split/merge error, not merely foreground union overlap. | P2 |
| Instance colour masks / `instance_masks` | 3 | Missing | Fit minimum/maximum extent and radius multiplier against complete reference masks with one-to-one matching. | P2 |
| Multiscale ellipse likelihood / `ellipse_likelihood` | 1 | Missing | Fit radial tolerance against complete seed contours or dependent detection performance; account for non-elliptical reference shapes. | P2 |
| Instance-assignment confidence / `assignment_confidence` | 1 | Missing | Fit boundary penalty to labelled assignment correctness with a probability/calibration objective. Reference masks can distinguish correctly and incorrectly assigned pixels. | P2 |
| Per-seed radial profiles / `radial_profile` | 1 | Conditional | Optimize bin count if profiles feed a supervised trait/damage objective; otherwise expose it as a diagnostic resolution choice and explain the lack of a predictive target. Reconstruction of the same image profile is not independent seed truth. | P2 |
| Proposal disagreement / `proposal_disagreement` | 1 | Missing | Fit disagreement scale against actual candidate/instance errors from references, using a defined error-risk calibration objective. | P2 |
| Occlusion/contact graph / `contact_graph` | 1 | Conditional | Fit distance multiplier with reviewed pairwise contacts or conservatively derived visible contact labels. Instance masks can support adjacency; they do not establish hidden occlusion order. | P2 |
| Seed-coat damage likelihood / `coat_damage` | 1 | Conditional | Fit anomaly scale using explicit damage labels/regions or an agreed mapping from reviewed conditions. A generic stained/split label must not silently stand for every type of coat damage. | P2 |
| Human review / `review` | 0 | Planned | No implemented optimizer obligation today. Future edits create targets; do not optimize away human decisions. | P2 workflow |
| Measurements / `measurements` | 0 | Exempt | Current geometry computation has no node controls to fit. Add validation against known measurements; any later estimator/calibration parameters need a new eligibility review. | — |
| Coat and condition / `classification` | 0 | Planned | Future classifier must expose training/tuning against compatible reviewed labels. | P2 workflow |
| Lot aggregation / `aggregation` | 0 | Planned | Fixed sums/statistics do not need arbitrary tuning; learned grading thresholds would. | P2 workflow |
| Final output / `output` | 0 | Planned | Formatting/export itself does not call for optimization. Future computational grading needs a declared target. | P2 workflow |
| Calibration residual risk / `calibration_residuals` | 1 | Conditional | Fit gain only against independently observed calibration/measurement errors. Minimizing the displayed risk would trivially reward understating uncertainty. | P2 |

The graph catalogue is in [model.py](D:/Programming/SeedFiddle/seedvision/pipeline/model.py:999). Diagnostic calculations supporting the conditional judgments are in [advanced.py](D:/Programming/SeedFiddle/seedvision/visualization/advanced.py:1038), including radial profiles, wrinkling, damage, patterns and colour probabilities; residual-risk scaling is at [line 1264](D:/Programming/SeedFiddle/seedvision/visualization/advanced.py:1264). The current applied semantic vocabulary is loaded by [seed_traits.py](D:/Programming/SeedFiddle/seedvision/annotation/seed_traits.py:57) from [traits.json](D:/Programming/SeedFiddle/config/traits.json).

## 3. Required improvements, with evidence and acceptance criteria

### O01 — P1: make optimization a declared node capability

**Finding.** Only two node IDs receive fit controls. Directly supervised colour, noise, material/edge/trait prototypes and shape models have no parameter-search action. Indirectly supervised feature and geometry nodes have none either. There is no common contract through which the UI or a project coordinator could discover these capabilities.

**Required change.** Add a node optimization registry/contract recording target requirements, objective, candidate controls and types, dependencies, actual calculation owner, scope, resource budget, and explicit exclusions. Generate node actions and project eligibility from this contract. Existing optimizer implementations should register with it. Do not add independent hard-coded dialogs for every node.

**Acceptance.** Every implemented card in the inventory declares either an actionable optimizer or a justified exemption/conditional state. Every computational control is classified as searchable, fixed for a documented reason, or inactive under a specific input configuration. A node cannot silently lose coverage when a new parameter is added. See [PipelineNode](D:/Programming/SeedFiddle/seedvision/pipeline/model.py:107) and [inspector dispatch](D:/Programming/SeedFiddle/seedvision/ui/pipeline_inspector.py:1641).

### O02 — P1: add the project-wide sequential command

**Finding.** Neither the menu nor controller supplies this workflow. Both current fit workers operate on the selected image, while accepted parameters affect every image in the project. Running each current fitter on successive images would simply overwrite shared settings with the last accepted image's values.

**Required change.** Provide an **Optimize eligible nodes from project references…** command that builds a plan from the actual active graph, fits each eligible node against the selected project reference set, adopts accepted upstream values within the run, and then fits downstream nodes using those values. Aggregate losses across selected images/groups with declared weighting. Report skipped cards and reasons.

**Acceptance.** A project with two different annotated images yields one shared proposal per node scored on both images, with per-image results. A missing target on one image does not make it a negative example. One-image operation remains available and is labelled accordingly. The command requires no manual node-by-node intervention after the user selects the run scope and application policy. Full contract: section 4. Current scope is visible in [edge fit start](D:/Programming/SeedFiddle/seedvision/ui/main_window.py:10572) and [procedural application](D:/Programming/SeedFiddle/seedvision/ui/main_window.py:11176).

### O03 — P1: fit the production procedural path and remove ineffective trials

**Finding.** The procedural worker deliberately avoids reusing evidence trained on the scoring masks, which is a valuable leakage precaution. However, it constructs a different pipeline: resolved material is removed, foreground becomes inverse background, foreground noise is zeroed, physical/nonphysical/reference edges and traces are absent, validated oval centres are omitted, and no shape model is passed. Normal inference supplies these inputs.

As a consequence, **four of the 16 default search parameters are structurally ineffective in this worker**:

| Parameter | Why it cannot affect this fit |
|---|---|
| `boundary_semantic_floor` | All semantic edge inputs are absent, so the semantic-gating branch is skipped. |
| `boundary_nonphysical_discount` | Same branch is skipped; nonphysical evidence is zero. |
| `boundary_physical_ridge_weight` | The reference-ridge raster is zero. |
| `boundary_trace_weight` | Trace labels and continuity are zero, so trace support is zero. |

A synthetic, nonempty two-seed reproduction set each parameter to both 0 and 1: **labels, boundary cost and centre likelihood were identical in all eight trials**. This is supported by the branches themselves, not just an insensitive fixture. The worker can spend search budget on parameters it cannot optimize. Changes it does find are evaluated on the simplified input path, so improvement there does not establish improvement in the displayed pipeline.

**Required change.** Evaluate the production path with upstream learned evidence rebuilt from the permitted training fold, then score held-out references. An explicitly labelled in-sample adaptation mode can also be supported; it must still use the intended production inputs. Do not solve leakage by deleting features and presenting their search as meaningful. Exclude unavailable-input parameters from a run with visible reasons.

**Acceptance.** Each included search parameter can affect its production calculation under the run's input configuration; candidate results are scored with production-equivalent upstream evidence. Predictions and scores for the selected proposal can be reproduced after application. Reference masks remain excluded as automatic watershed markers. Sources: [worker](D:/Programming/SeedFiddle/seedvision/ui/main_window.py:1076), [input preparation](D:/Programming/SeedFiddle/seedvision/segmentation/procedural.py:1193), [boundary calculation](D:/Programming/SeedFiddle/seedvision/segmentation/procedural.py:1506), [normal inference](D:/Programming/SeedFiddle/seedvision/segmentation/baseline.py:2068).

### O04 — P1: complete procedural parameter coverage and resolve the legacy control

**Finding.** The node exposes 43 entries, while the default search covers 16 and the custom API allows 28. Six `reference_error_*` entries define scoring policy/completeness and should remain fixed; the working-dimension limit should also be fixed by default. The uncovered computation is much larger than those legitimate exclusions.

| Coverage gap | Exact controls |
|---|---|
| In the custom whitelist, absent from the GUI default search | `occupancy_hole_area_fraction`, `dish_margin_fraction`, `boundary_edge_weight`, `boundary_ridge_weight`, `trace_minimum_length_fraction`, `trace_convexity_weight`, `centre_geometry_smoothing_fraction`, `centre_material_weight`, `centre_distance_weight`, `maximum_instance_area_fraction`, `minimum_instance_solidity`, `maximum_instance_axis_ratio` |
| Computational controls excluded even from the custom whitelist | `boundary_surface_darkening_weight`, `centre_flattened_grayscale_weight`, `centre_validated_oval_weight`, `sparse_seed_area_fraction`, `packed_seed_cell_fraction`, `candidate_hypotheses_per_marker`, `candidate_overlap_fraction` |
| Exposed legacy/fallback control requiring resolution | `reference_texture_weight` |

`reference_texture_weight` is used only when `material_probability` is absent. The normal baseline passes the resolved material probability, bypassing that mixture. A second probe with nonzero material and reference-surface inputs confirmed that changing its weight from 0 to 1 changes neither occupancy nor labels. It should be conditionally hidden/disabled with a reason, removed from the active node if obsolete, or given a legitimate calculation role before optimization is promised.

**Required change.** Expose a bounded advanced search for all remaining eligible controls, including integer candidate hypotheses and discrete choices where relevant. Group searches by computation, support constrained combinations, and show excluded controls. Preserve the fixed cost/coverage policy throughout candidate evaluation.

**Acceptance.** Every meaningful procedural control has an accessible search path; default subsets are labelled. Inactive legacy controls are handled honestly. The 28-name whitelist is not mistaken for GUI coverage. Sources: [default search and whitelist](D:/Programming/SeedFiddle/seedvision/segmentation/procedural_fit.py:49), [material branch](D:/Programming/SeedFiddle/seedvision/segmentation/procedural.py:1441).

### O05 — P1: freeze the edge evaluation targets before searching

**Finding.** `reference_texture_instance_interior_buffer_fraction` is both a searched parameter and an input to the held-out target generator. Changing it changes which internal-edge pixels contribute to loss. The search can improve its score by selecting a different, easier evaluation subset even when predictions do not change.

The reproduction uses the real edge evaluator and target generator, with only classifier outputs replaced by fixed synthetic probability fields:

| Interior buffer fraction | Physical target pixels | Internal-edge target pixels | Balanced loss, identical predictions |
|---:|---:|---:|---:|
| 0.03 | 196 | 1,581 | 0.488642 |
| 0.24 | 196 | 465 | 0.368601 |

This is a counterexample to score comparability, not a measured effect size on real photographs. The apparent improvement is entirely from changing the scored domain. Also, the current internal-edge targets are generated from gradients/ridges, so future optimization of those upstream nodes must not regenerate their own scoring truth from each proposal.

**Required change.** Freeze reviewed target masks, validity regions, tolerances and fold membership before optimization. A training-sample interior buffer may remain searchable, but the evaluation buffer must be independent and fixed. When pseudo-labels are necessary, record how they were produced and keep them fixed for candidate comparisons.

**Acceptance.** Fixed predictions have identical scores for every proposal, regardless of training-sampling settings. Test target identity across buffer, gradient, ridge, crop and scale proposals. Source: [edge evaluator](D:/Programming/SeedFiddle/seedvision/segmentation/reference_edge_fit.py:129), especially [target generation](D:/Programming/SeedFiddle/seedvision/segmentation/reference_edge_fit.py:226), and [internal-edge selection](D:/Programming/SeedFiddle/seedvision/annotation/instance_references.py:165).

### O06 — P1: optimize all Reference edges operations against the intended outputs

**Finding.** The six searched controls are similarity scale, class contrast, normal strip offset, tangent half-length, ridge blend, and instance interior buffer. The evaluator scores raw `physical_edge_field` and `non_edge_field`; it does not score the final supported/net, locally normalized, or thinned reference-edge outputs. It also supplies no species library to prototype construction, even though the node exposes source and blend controls.

Missing search areas include bank capacity/minimum samples, allowed library blend, internal-edge subtraction, local normalization radius/support/gain/floor, and ridge NMS/threshold/hysteresis. Resolution and iteration budgets deserve explicit fixed-versus-searchable decisions. Appending all these names to the existing search would not work: final-output controls cannot influence the raw probability fields it evaluates, and library controls need actual permitted library inputs.

**Required change.** Define separate classifier and final-edge objectives, with a documented downstream instance check. Invoke the full node calculations for affected candidates and include the frozen library context. Cover each eligible operation; fixed source provenance must not become a way to select evaluation labels into training.

**Acceptance.** Every included parameter affects an output contributing to its objective. The reported improvement is reproducible for the selected output after application. The search can tune final normalization/ridges without silently evaluating only raw classifier membership. Sources: [six search parameters](D:/Programming/SeedFiddle/seedvision/segmentation/reference_edge_fit.py:21), [evaluated fields](D:/Programming/SeedFiddle/seedvision/segmentation/reference_edge_fit.py:178), [library-capable producer](D:/Programming/SeedFiddle/seedvision/cuda/layers.py:5428).

### O07 — P1: share supervision eligibility and evaluation rules across optimizers

**Finding.** The current controllers check project existence, applied/nonempty masks and connected IDs. They do not enforce a shared per-instance completeness/review eligibility policy. The edge evaluator additionally needs at least two IDs and both contour and internal-edge samples, but these requirements are discovered after starting work. A complete contiguous painted patch can still be an incomplete seed. This is the annotation-quality problem documented as R03 in the original review.

The edge evaluator correctly separates prototype-training IDs from scoring IDs. That is a useful safeguard, but its one deterministic within-image split is reused for selection; it is not a final independent project validation set. The procedural worker correctly refuses to use target masks as watershed markers, but its substitutions introduce the O03 mismatch. Both mechanisms should be preserved in intent while being replaced by a shared, production-consistent contract.

**Required change.** Filter complete/eligible instances by persisted metadata, validate semantic classes, use reviewed-region masks for partial coverage, and freeze source/annotation hashes and coordinate mappings. Keep unreviewed regions unknown. For predictive validation, rebuild every annotation-trained ancestor without evaluation examples, including colour, texture, shape and library inputs. Record project/image/group split and distinguish fitting, validation used for selection, and independent final assessment.

**Acceptance.** Partial, draft, stale, or ineligible masks cannot silently become complete-seed targets. Each image carries its own coverage declaration; a single whole-dish checkbox must not automatically declare every project image exhaustively annotated. Calibration/crop changes cannot alter the truth domain to reduce error. Scientific claims remain limited to the actual split/provenance. Sources: [procedural eligibility](D:/Programming/SeedFiddle/seedvision/ui/main_window.py:10863), [edge preflight](D:/Programming/SeedFiddle/seedvision/ui/main_window.py:10572), [instance fold split](D:/Programming/SeedFiddle/seedvision/segmentation/reference_edge_fit.py:160), original review R03/R08/R12.

### O08 — P2: expose learned-node tuning through the same node interface

**Finding.** The Learning menu can train/refine U-Net or StarDist. `seed_vision.py --optimize-decoder` searches validation decoder parameters, and hybrid tuning also exists. These are real optimization capabilities, but neither learned card provides the requested option or participates in a project node sequence.

The U-Net search covers five settings: interior threshold, centre threshold, separation in pixels, pattern-boundary discount and erosion in pixels. The StarDist search covers object threshold, NMS IoU and local-maximum radius. Other exposed decoder controls are omitted. GUI node separation/erosion settings use seed-relative fractions, so CLI settings cannot simply be copied into cards without conversion and image-scale semantics.

**Required change.** Reuse the backend search in node-local actions and the coordinator where learned nodes are active and compatible checkpoints exist. Complete decoder coverage with fixed predictions and consistent units. Provide model-weight training as a separate, explicit scope; the project procedural command should not unexpectedly launch long training runs.

**Acceptance.** Optimized values have the same meaning in offline evaluation and live node inference, persist in the project, and produce identical decoded instances from the same outputs. Training and decoder tuning are clearly distinguished. Sources: [decoder grids](D:/Programming/SeedFiddle/seedvision/learning/evaluation.py:138), [CLI options](D:/Programming/SeedFiddle/seed_vision.py:129), [GUI training action](D:/Programming/SeedFiddle/seedvision/ui/main_window.py:2927).

### O09 — P2: provide a consistent, informative fitting experience

**Finding.** Procedural fitting presents eligibility, busy state and a result summary, then asks whether to apply the proposal. Edge fitting has only an enabled/implemented button gate in the inspector, confirms before searching, and applies an improving result immediately afterward. The fresh inspector inventory found the edge button enabled without any image. The controller prevents that invalid start, but the user learns requirements through a dialog rather than the card.

**Required change.** Show each eligible card's annotation counts, target type, fit scope, included/excluded parameter groups, search budget, and unavailable reasons. Use consistent result comparisons and application policy for individual and project runs. Let users inspect before/after overlays and per-image regressions; report cancellation, no improvement and budget exhaustion distinctly. A visible score must identify its output, metric and evaluation set.

**Acceptance.** No fit opens only to discover a predictable missing prerequisite. A project run does not prompt repeatedly at each node. Existing cancellation and stale-result guards remain effective. Sources: [procedural state](D:/Programming/SeedFiddle/seedvision/ui/pipeline_inspector.py:1641), [edge description/button](D:/Programming/SeedFiddle/seedvision/ui/pipeline_inspector.py:1275), [edge result handling](D:/Programming/SeedFiddle/seedvision/ui/main_window.py:10690), [procedural result handling](D:/Programming/SeedFiddle/seedvision/ui/main_window.py:11096).

### O10 — P1: make ownership, caching and reproducibility part of the coordinator

**Finding.** The graph already offers topological ordering, parameter validation and downstream invalidation. Existing fitters guard against image/revision/annotation changes and persist accepted settings as project-local parameters. These are good foundations. However, there is no shared fit plan or persisted run record. Reference-edge controls also affect a producer owned by Reference texture prototypes: the current application handler explicitly invalidates that producer and its descendants after an edge fit. A generic “changed card plus graph descendants” implementation would miss this dependency unless ownership is declared.

**Required change.** Resolve true computation ownership in the optimization contract, ideally aligning it with the graph. Candidate evaluations must use isolated settings/cache namespaces, preserve reusable upstream CUDA tensors, and invalidate the changed computation and affected consumers for each selected image. Persist the run's inputs, settings, objective version, splits, trials/summaries, final decision and changed nodes. Keep an atomic proposal or a reversible checkpoint for the sequence.

**Acceptance.** Unrelated cached nodes are reused; every affected producer is recomputed. Cancellation/failure cannot leave a partly installed candidate. Saved/reopened projects reproduce accepted parameters and expose the run's provenance. A final full-pipeline comparison detects cross-node regressions. Sources: [topological order](D:/Programming/SeedFiddle/seedvision/pipeline/model.py:761), [parameter updates](D:/Programming/SeedFiddle/seedvision/pipeline/model.py:783), [edge producer invalidation](D:/Programming/SeedFiddle/seedvision/ui/main_window.py:10761). This is a required design constraint for the missing coordinator, not a claim that the current explicit edge invalidation is absent.

### O11 — P1: test compliance and objective validity, not just successful search

**Finding.** The focused tests verify useful properties of the existing two fitters, including contextual actions, scoring, cancellation/application and disjoint edge IDs. They do not establish coverage of the catalogue, production-equivalent candidate evaluation, fixed targets, or project sequencing. The synthetic counterexamples pass through paths that the current suite permits.

**Required change.** Add contract-driven coverage tests and meaningful integration tests as the optimizers are implemented. Include continuous, integer and categorical search; fixed-parameter enforcement; missing-input exclusions; annotation eligibility; invariant target sets; project aggregation; true-owner invalidation; stale/cancelled runs; and application/reload equivalence. Use purpose-designed fixtures where changing a candidate parameter changes relevant evidence; do not demand that every perturbation changes discrete masks.

**Acceptance.** Adding an eligible node/control without an optimization disposition fails a coverage test. The O03/O05 counterexamples fail before their fixes and pass afterward. At least one multi-image end-to-end optimization test verifies sequential dependence and one shared project proposal. Real annotated image benchmarks are required before claiming accuracy gains; synthetic correctness tests alone cannot supply them.

## 4. Required project command behavior

The intended command can remain entirely within the existing PySide6 desktop application and single `seed_vision.py` launch point. No server or alternative application architecture is necessary.

1. **Build and show a run plan.** Read the actual active graph and declared optimization capabilities. List enabled eligible nodes, conditional/missing targets, fixed parameters, selected project images/groups, metric definitions, resource limits and application policy. Do not include inactive toolbox branches automatically. Include learned decoder tuning only when applicable and selected; weight training is a separately requested operation.
2. **Freeze supervision and provenance.** Snapshot source identities, applied reference hashes and completeness, coordinate transforms, species/library versions, checkpoints, fold membership and reviewed domains. Fix metric weights and tolerances. If a current project has insufficient independent references, offer clearly labelled local adaptation or explain which target requirement is unmet.
3. **Establish the baseline.** Run the relevant production pipeline for the selected images and record both the node objectives and final output metrics. Preserve per-image and per-group results so averaging cannot conceal a failed image or minority class.
4. **Optimize in dependency order.** Use the current graph's topological order, adjusted for true calculation ownership. For each node, keep accepted ancestors fixed, vary its eligible parameters within a declared budget, and recompute the affected path to its objective. If downstream learned evidence changes with the candidate, refit that evidence using training references only. Reuse other intermediates.
5. **Adopt a node proposal inside the run.** Keep an improving, valid candidate for subsequent downstream optimization; preserve original settings and record unchanged/no-improvement/skipped states. Node-specific metrics need not be numerically comparable across nodes. The final pipeline objective provides a separate regression check. A single upstream-to-downstream pass is a sensible minimum; optional repeat passes can stop at a global budget or lack of material improvement. Sequential search is not a guarantee of a global optimum.
6. **Review and apply the project result.** Present node/parameter changes, before/after overlays, per-image results and regression warnings supported by actual measurements. Apply the accepted set as one coherent project change or use an explicitly selected incremental policy with recovery points. One initial user-selected run policy can authorize the sequence; no chain of modal confirmations is necessary.
7. **Persist and support recovery.** Save final project settings and an inspectable optimization record. Cancel safely between evaluations, discard stale candidates, preserve a failed-run summary, and provide rollback to the pre-run settings. Reopen and reproduce the accepted configuration.

For upstream feature nodes, a cheaper local objective may be useful, but the plan must name it and validate the final pipeline afterward. Conversely, running the entire project pipeline for every trivial parameter proposal would discard the application's existing caching advantage. The registry should identify the smallest affected path that supplies the chosen objective. Full-resolution reusable rasters remain on CUDA; bounded CPU topology and compact scores can retain their existing roles.

## 5. Suggested delivery order

| Stage | Required outcome | Relevant findings |
|---|---|---|
| 1 — Correctness foundation | Shared supervision/target contract; fixed edge evaluation masks; production-consistent procedural evaluation; resolve ineffective and legacy controls. | O03–O07 |
| 2 — Shared capability and project runner | Node optimizer registry, explicit exclusions, sequential multi-image plan, ownership-aware caching, safe application and provenance. Migrate both existing fitters first. | O01–O02, O09–O11 |
| 3 — Complete the default scientific path | Add material colour/noise/texture/decision, dimensions/shape, trait, edge/trace/oval, illumination and layout optimization; cover all remaining eligible controls of the existing fit nodes. | Active P1 rows, O04/O06 |
| 4 — Complete optional and conditional branches | Learned decoder cards, remaining feature/diagnostic cards, restored proposal/instance toolbox branches, and compatible semantic/contact/quality targets. Explain unavailable target types in the UI. | Active/toolbox P2 rows, O08 |
| 5 — Demonstrate usefulness | Grouped real-image benchmarks, parameter sensitivity/ablation, repeatability, per-class and per-image regression analysis; refine search budgets from measured benefit. | O07/O11 |

Stages 1–2 make the optimization infrastructure credible; they **do not** by themselves satisfy the “all eligible nodes” requirement. That claim requires stage 3 and the applicable parts of stage 4, with explicit, defensible exemptions for the remainder. More sophisticated search algorithms, multi-start searches and automatic budget allocation are P3 enhancements once bounded coordinate/discrete searches are correct and complete.

## 6. Verification and limitations

| Check | Fresh result |
|---|---|
| Actual graph + Qt inspector inventory | All 48 cards inspected; exactly two fit containers visible on their owning cards |
| Actual MainWindow menu inventory | Analysis contains only Run pipeline; no project optimizer |
| Procedural parameter counterexample | Four default search parameters have identical labels, boundary cost and centre likelihood at both extreme values; baseline contains two instances |
| Edge target counterexample | Identical classifier outputs score 0.488642 vs 0.368601 solely because the interior-buffer candidate changes the evaluation set |
| Legacy procedural control counterexample | With resolved material input, reference texture weight 0 vs 1 leaves occupancy and labels unchanged |
| Focused existing tests | **43 passed in 5.688 seconds**: procedural fit, procedural fit UI, reference edge fit, pipeline inspector actions |
| Full `unittest` suite | **611 run: 608 passed, 2 skipped, 1 failed; 315.771 seconds** |
| Full-suite failure | Existing manifest references absent `images/IMG_9689c.JPG`; same failure as the original review |

The two skips concern the unavailable complete eleven-image ruler batch and a case-sensitive-filesystem check inapplicable on this host. Application code was unchanged, and no missing fixture was fabricated.

Reproduction commands from the repository root:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe artifacts\node_optimization_review.py
.\.venv\Scripts\python.exe artifacts\node_optimization_probes.py
.\.venv\Scripts\python.exe -m unittest tests.test_procedural_fit tests.test_procedural_fit_ui tests.test_reference_edge_fit tests.test_pipeline_inspector_actions -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Local evidence is available in [inventory.json](D:/Programming/SeedFiddle/artifacts/node-optimization-review-2026-09-09/inventory.json), [probes.json](D:/Programming/SeedFiddle/artifacts/node-optimization-review-2026-09-09/probes.json), the [inventory harness](D:/Programming/SeedFiddle/artifacts/node_optimization_review.py), [counterexample harness](D:/Programming/SeedFiddle/artifacts/node_optimization_probes.py), [focused test log](D:/Programming/SeedFiddle/artifacts/node-optimization-review-focused-suite-2026-09-09.log), and [full test log](D:/Programming/SeedFiddle/artifacts/node-optimization-review-full-suite-2026-09-09.log). These generated review artifacts remain ignored under `artifacts/`; the tables and counterexample results above retain the essential evidence in the report.

This review does not claim exhaustive numerical sensitivity testing of all 309 parameter entries or a measured real-world gain from every proposed optimization objective. It establishes present UI/backend coverage, identifies specific objective and input-path defects, and defines a complete node-level compliance plan with testable acceptance criteria.
