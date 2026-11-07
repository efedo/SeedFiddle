# Complete test inventory

This is a source-derived inventory of every discovered test method, audited on 2026-11-06. Functionality descriptions use method docstrings or scenario names, with assertion and call evidence in [the JSON inventory](test_inventory.json). These are intended coverage claims, **not measured statement or branch coverage**.

Markdown shows at most the first two assertion expressions, shortened to 260 characters. JSON retains every direct assertion expression, lexical production-call candidates, interface calls, helper references and their shared assertion evidence, loops, subtest sites, decorators and fixture calls. Dynamic dispatch and inherited fixture behavior require source inspection; this is not a manually proven call graph or an exhaustive enumeration of runtime subtest cases.

Total: **317 methods in 116 test files**. A method can contain many subtests, input cases or expensive executions. Those are not separate unittest methods.

Discovery reconciliation: exact match, 317 canonical unittest IDs; zero discovery errors.

No historical run receipt was used, so status and seconds are blank. Class setup/teardown, discovery and interpreter shutdown are not allocated to methods. A dash means no method duration was recorded, not zero cost. Runtime skips can differ by environment; their exact historical reasons are in JSON.

Regenerate: `.venv\Scripts\python.exe tools/audit_test_inventory.py --collect`. This imports modules for discovery but never runs tests or setup fixtures. Without the ignored historical receipts, use `--without-history`; historical status/timing fields are then unrecorded.

See [the multi-pass audit](../archive/TEST_INFRASTRUCTURE_AUDIT.md) for overlap, cost and infrastructure findings.

| Module | Tests | Historical method seconds | Scope |
| --- | ---: | ---: | --- |
| [tests.test_advanced_analysis](#teststest_advanced_analysis) | 1 | 0.000 | Advanced image layers and CUDA analysis products |
| [tests.test_analysis_settings_persistence](#teststest_analysis_settings_persistence) | 2 | 0.000 | Analysis settings serialization, normalization and private workspace state |
| [tests.test_annotation_centres](#teststest_annotation_centres) | 4 | 0.000 | Annotation centroids and radial hilum metadata must never become predictions. |
| [tests.test_annotation_edge_sources](#teststest_annotation_edge_sources) | 2 | 0.000 | Annotation tools select the intended edge evidence and clear obsolete sources |
| [tests.test_annotation_generation_binding](#teststest_annotation_generation_binding) | 2 | 0.000 | Native label producers bind to the displayed corrected source, including history. |
| [tests.test_annotation_shape_revision](#teststest_annotation_shape_revision) | 4 | 0.000 | Shape review revisions, visibility and reference-library updates |
| [tests.test_annotation_tools](#teststest_annotation_tools) | 2 | 0.000 | Annotation drawing, brush geometry, region operations and instance tools |
| [tests.test_annotation_transfer](#teststest_annotation_transfer) | 5 | 0.000 | Project annotation folders, unsaved drafts and annotation bundle export, restore and manifest validation |
| [tests.test_assignment_budget](#teststest_assignment_budget) | 2 | 0.000 | Assignment metrics and segmentation matching respect explicit work budgets |
| [tests.test_authoritative_parity](#teststest_authoritative_parity) | 2 | 0.000 | Real native CUDA analysis, desktop/batch identity and exact review portability. |
| [tests.test_automatic_material_references](#teststest_automatic_material_references) | 2 | 0.000 | Initial Foreground, Background and Other references without painting. |
| [tests.test_backend_outcomes](#teststest_backend_outcomes) | 1 | 0.000 | Actual backend outcomes, cache recovery, Qt/batch parity and immutable review binding. |
| [tests.test_benchmark_cli](#teststest_benchmark_cli) | 2 | 0.000 | Single-entry CLI routing and bounded adapters; GPU producers tested separately. |
| [tests.test_benchmark_instances](#teststest_benchmark_instances) | 1 | 0.000 | The instance benchmark harness scores source-frame masks and states the reference caveat. |
| [tests.test_benchmark_manifest](#teststest_benchmark_manifest) | 2 | 0.000 | Benchmark manifests, source/environment identity and evidence contracts |
| [tests.test_bootstrap](#teststest_bootstrap) | 13 | 0.000 | Python/runtime dependency admission and single-entry startup |
| [tests.test_calibration_result_revision](#teststest_calibration_result_revision) | 2 | 0.000 | Actual calibration state must travel with authoritative review revisions. |
| [tests.test_catalogue_nodes](#teststest_catalogue_nodes) | 2 | 0.000 | Independent catalogue instances, typed projections and portable project state. |
| [tests.test_catalogue_nodes_ui](#teststest_catalogue_nodes_ui) | 2 | 0.000 | CPU/Qt ownership, routing and stale-publication checks for added nodes. |
| [tests.test_cellpose_sam](#teststest_cellpose_sam) | 2 | 0.000 | Optional Cellpose-SAM branch: honest settings, flow decoding and engine caching. |
| [tests.test_chart_colour_correction](#teststest_chart_colour_correction) | 3 | 0.000 | Colour correction against a chart's published reference colours. |
| [tests.test_colour_correction](#teststest_colour_correction) | 2 | 0.000 | Chart-fitted colour-correction matrices, CUDA application and fit diagnostics |
| [tests.test_conditioned_shape_pooling](#teststest_conditioned_shape_pooling) | 2 | 0.000 | Conditioned pooling uses the same Gaussian model without dropping directions. |
| [tests.test_critical_review_fixes](#teststest_critical_review_fixes) | 2 | 0.000 | Earlier regression fixes spanning annotation coordinates, learning and export |
| [tests.test_default_pipeline_copies](#teststest_default_pipeline_copies) | 2 | 0.000 | Default pipelines stay locked; their edit controls offer a named, editable copy. |
| [tests.test_deferred_callbacks](#teststest_deferred_callbacks) | 6 | 0.000 | Deferred callbacks die with their owner and report what they raise. |
| [tests.test_desktop_terminal_history](#teststest_desktop_terminal_history) | 4 | 0.000 | Earlier desktop rejection boundaries and bounded reading of saved run records. |
| [tests.test_diagnostics](#teststest_diagnostics) | 1 | 0.000 | Runtime diagnostic facts |
| [tests.test_edge_trace_commit](#teststest_edge_trace_commit) | 1 | 0.000 | Edge trace tool commits and desktop annotation updates |
| [tests.test_entrypoint](#teststest_entrypoint) | 3 | 0.000 | Native desktop application launch and command routing |
| [tests.test_graph_default_pipelines](#teststest_graph_default_pipelines) | 2 | 0.000 | Procedural and Cellpose seed analysis are one catalogue with different active nodes. |
| [tests.test_grouped_comparison](#teststest_grouped_comparison) | 2 | 0.000 | Compact synthetic engineering witnesses, not biological validation. |
| [tests.test_gui_layout](#teststest_gui_layout) | 1 | 0.000 | Desktop layout contracts from the GUI review: menus, toolbar, panels, canvas. |
| [tests.test_icon_asset](#teststest_icon_asset) | 1 | 0.000 | Application icon assets |
| [tests.test_image_calibration](#teststest_image_calibration) | 5 | 0.000 | Image calibration, geometry, chart/ruler detection and real-image fixtures |
| [tests.test_image_list_controls](#teststest_image_list_controls) | 4 | 0.000 | Native image-list actions, thumbnails, file preservation and edit ownership. |
| [tests.test_instance_annotation_tiling](#teststest_instance_annotation_tiling) | 2 | 0.000 | Annotation tile rendering, invalidation and image-view state |
| [tests.test_instance_continuity](#teststest_instance_continuity) | 2 | 0.000 | Instance continuity during annotation updates |
| [tests.test_instance_continuity_ui](#teststest_instance_continuity_ui) | 2 | 0.000 | Desktop selection follows instance continuity |
| [tests.test_instance_mask_import](#teststest_instance_mask_import) | 2 | 0.000 | Instance mask loading, source identity and validation |
| [tests.test_instance_mask_import_ui](#teststest_instance_mask_import_ui) | 2 | 0.000 | Instance mask import dialogs and desktop application |
| [tests.test_instance_reference_evidence](#teststest_instance_reference_evidence) | 2 | 0.000 | Annotated-instance reference evidence and visualization |
| [tests.test_instance_reference_repository](#teststest_instance_reference_repository) | 1 | 0.000 | Committed reference fixture repository integrity |
| [tests.test_learned_segmentation](#teststest_learned_segmentation) | 6 | 0.000 | Learned segmentation features, targets, model, loss, decoding and metrics |
| [tests.test_learning_workflow](#teststest_learning_workflow) | 1 | 0.000 | Training workflow, evidence export and desktop controls |
| [tests.test_local_shape_units](#teststest_local_shape_units) | 2 | 0.000 | Current-image shape fits must keep calibrated and normalized blocks apart. |
| [tests.test_manual_seed_centre_persistence](#teststest_manual_seed_centre_persistence) | 2 | 0.000 | Manual seed centre sidecar persistence and source binding |
| [tests.test_manual_seed_centre_ui](#teststest_manual_seed_centre_ui) | 2 | 0.000 | Manual centre drawing, selection and desktop persistence |
| [tests.test_material_evidence](#teststest_material_evidence) | 2 | 0.000 | CUDA material likelihoods, references and evidence decisions |
| [tests.test_measurement_description_ui](#teststest_measurement_description_ui) | 2 | 0.000 | Reviewed overlay facts use the same measurement records as their values. |
| [tests.test_measurement_descriptions](#teststest_measurement_descriptions) | 2 | 0.000 | Value descriptions preserve calculation facts without inventing lineage. |
| [tests.test_measurement_scale_safety](#teststest_measurement_scale_safety) | 2 | 0.000 | Numeric scale availability is separate from biological/physical validity. |
| [tests.test_measurements](#teststest_measurements) | 2 | 0.000 | Seed measurement geometry |
| [tests.test_neutral_calibration](#teststest_neutral_calibration) | 3 | 0.000 | Selected grid references, CUDA sampling, cache boundaries and saved facts. |
| [tests.test_node_optimization](#teststest_node_optimization) | 4 | 0.000 | Node parameter search, optimizer state and cache reuse |
| [tests.test_node_optimization_production](#teststest_node_optimization_production) | 1 | 0.000 | Production integration on two generated images; no accuracy claims. |
| [tests.test_node_optimization_ui](#teststest_node_optimization_ui) | 2 | 0.000 | Node optimizer controls and desktop result application |
| [tests.test_node_vocabulary](#teststest_node_vocabulary) | 3 | 0.000 | The engine's node vocabulary must stay bound to the pipeline graph. |
| [tests.test_oriented_edge_components](#teststest_oriented_edge_components) | 2 | 0.000 | Oriented edge components, connectivity and CUDA outputs |
| [tests.test_physical_shape_fit_availability](#teststest_physical_shape_fit_availability) | 2 | 0.000 | Finite calibrated evidence may exceed the physical fitter's numeric domain. |
| [tests.test_pilot_analysis](#teststest_pilot_analysis) | 5 | 0.000 | Committed pilot-image segmentation, calibration and end-to-end analysis |
| [tests.test_pipeline](#teststest_pipeline) | 10 | 0.000 | Pipeline graph topology, controls, settings, state and invalidation |
| [tests.test_pipeline_cable_routing](#teststest_pipeline_cable_routing) | 1 | 0.000 | Pipeline cable geometry and routing |
| [tests.test_pipeline_documents](#teststest_pipeline_documents) | 2 | 0.000 | Standalone graph/layout transport, factory protection and frozen file loads. |
| [tests.test_pipeline_inspector_actions](#teststest_pipeline_inspector_actions) | 1 | 0.000 | Pipeline inspector parameter editing and actions |
| [tests.test_pipeline_ui](#teststest_pipeline_ui) | 15 | 0.000 | Pipeline desktop controls, graph state, navigation, workers and status |
| [tests.test_pipeline_workflow_ui](#teststest_pipeline_workflow_ui) | 1 | 0.000 | Main-workspace routing between the default pipelines and the shared project nodes. |
| [tests.test_pipeline_workspace_files](#teststest_pipeline_workspace_files) | 2 | 0.000 | Desktop default-pipeline lock, named copies and complete graph/layout file integration. |
| [tests.test_polynomial_correction_integrity](#teststest_polynomial_correction_integrity) | 2 | 0.000 | Whole-call input and retained-matrix integrity for polynomial correction. |
| [tests.test_portable_result_identity](#teststest_portable_result_identity) | 3 | 0.000 | Actual relocated native results, artifact facts and private review history. |
| [tests.test_preflight_outcomes](#teststest_preflight_outcomes) | 3 | 0.000 | Incomplete request evidence through real batch and Qt preparation boundaries. |
| [tests.test_procedural_fit](#teststest_procedural_fit) | 5 | 0.000 | Procedural seed fitting, search behavior and geometric evidence |
| [tests.test_procedural_fit_ui](#teststest_procedural_fit_ui) | 2 | 0.000 | Procedural fit controls and result application |
| [tests.test_procedural_instances](#teststest_procedural_instances) | 2 | 0.000 | CUDA procedural instance splitting, ownership and analysis integration |
| [tests.test_project_analysis_persistence](#teststest_project_analysis_persistence) | 3 | 0.000 | Project analysis settings, image locators and atomic persistence |
| [tests.test_project_analysis_ui](#teststest_project_analysis_ui) | 3 | 0.000 | Project analysis save/reopen, dirty state, selections and desktop guards |
| [tests.test_reference_edge_fit](#teststest_reference_edge_fit) | 2 | 0.000 | Reference-driven edge fitting and rendered evidence |
| [tests.test_reference_edge_strip_descriptor](#teststest_reference_edge_strip_descriptor) | 2 | 0.000 | Reference edge strip extraction, descriptors and overlays |
| [tests.test_reference_edge_toolbar_revision](#teststest_reference_edge_toolbar_revision) | 1 | 0.000 | Edge reference toolbar controls, graph revisions and image-view state |
| [tests.test_reference_graph_readonly](#teststest_reference_graph_readonly) | 1 | 0.000 | Read-only public reference graph presentation and edit prevention |
| [tests.test_reference_history](#teststest_reference_history) | 2 | 0.000 | Reference history revision selection and UI presentation |
| [tests.test_reference_library](#teststest_reference_library) | 2 | 0.000 | Reference library aggregation, extraction, contracts and persistence |
| [tests.test_reference_persistence](#teststest_reference_persistence) | 4 | 0.000 | Reference region sidecars, source binding and desktop lifecycle |
| [tests.test_reference_source_identity](#teststest_reference_source_identity) | 5 | 0.000 | Source association is preserved separately from annotation generation. |
| [tests.test_reference_texture_collage](#teststest_reference_texture_collage) | 1 | 0.000 | Reference texture collage layout and image rendering |
| [tests.test_reference_undo](#teststest_reference_undo) | 2 | 0.000 | Reference annotation undo/redo and independent history |
| [tests.test_resources](#teststest_resources) | 1 | 0.000 | CUDA resource estimation and runtime admission |
| [tests.test_result_prediction_revision](#teststest_result_prediction_revision) | 4 | 0.000 | Exact prediction facts must travel with review and exported revisions. |
| [tests.test_review_workflows](#teststest_review_workflows) | 2 | 0.000 | Review export, portable snapshots and desktop workflow |
| [tests.test_reviewed_measurement_lineage](#teststest_reviewed_measurement_lineage) | 3 | 0.000 | Actual review snapshots, producer cache boundaries and export identity. |
| [tests.test_run_outcomes](#teststest_run_outcomes) | 3 | 0.000 | Whole-run terminal evidence, actual Qt/batch parity and publication guards. |
| [tests.test_run_to_node_cache](#teststest_run_to_node_cache) | 2 | 0.000 | A run to one node recomputes that node alone and leaves other results standing. |
| [tests.test_scene_analysis_region](#teststest_scene_analysis_region) | 3 | 0.000 | Photographs without a vessel are analysed around their confident seeds. |
| [tests.test_seed_crop_artifacts](#teststest_seed_crop_artifacts) | 2 | 0.000 | Seed crop artifacts, export identity and portable review |
| [tests.test_seed_crops](#teststest_seed_crops) | 8 | 0.000 | Focused tests for explicit CUDA per-seed crop artifacts. |
| [tests.test_seed_shape_model](#teststest_seed_shape_model) | 1 | 0.000 | Seed shape model fitting, reference aggregation and scoring |
| [tests.test_selected_instance_view](#teststest_selected_instance_view) | 3 | 0.000 | Selected instance overlay and desktop view state |
| [tests.test_shape_guided_fill](#teststest_shape_guided_fill) | 2 | 0.000 | Shape-guided annotation filling and selection behavior |
| [tests.test_shape_guided_ui](#teststest_shape_guided_ui) | 2 | 0.000 | Shape-guided annotation desktop controls |
| [tests.test_shape_model_export](#teststest_shape_model_export) | 2 | 0.000 | Actual shape-update outcomes persist and bind the authoritative result revision. |
| [tests.test_shape_pooling_overflow](#teststest_shape_pooling_overflow) | 2 | 0.000 | High-precision Gaussian oracles for representable intermediate-overflow cases. |
| [tests.test_shape_scoring_availability](#teststest_shape_scoring_availability) | 2 | 0.000 | Candidate scoring distinguishes computational abstention from incompatibility. |
| [tests.test_shape_update_availability](#teststest_shape_update_availability) | 2 | 0.000 | Precision pooling availability is separate from fit/scoring availability. |
| [tests.test_shape_view_normalization](#teststest_shape_view_normalization) | 2 | 0.000 | Repeated biological views must be normalized before their units are mixed. |
| [tests.test_species_library_dialog](#teststest_species_library_dialog) | 5 | 0.000 | Species reference library desktop editing |
| [tests.test_terminal_snapshot](#teststest_terminal_snapshot) | 2 | 0.000 | Real terminal/acknowledgment transport and private Qt historical consumers. |
| [tests.test_test_runner](#teststest_test_runner) | 11 | 0.000 | Runner failure reporting, exhaustive tier partition and fixture lifecycle. |
| [tests.test_training_experiment](#teststest_training_experiment) | 1 | 0.000 | The three-arm training experiment: radial weighting, shared network, budget and reports. |
| [tests.test_trait_classifier](#teststest_trait_classifier) | 2 | 0.000 | Software oracles for trait classification; all truth is explicitly synthetic. |
| [tests.test_trait_evidence](#teststest_trait_evidence) | 4 | 0.000 | Trait training from a project's own labels, and models kept with the project. |
| [tests.test_trait_pipeline](#teststest_trait_pipeline) | 2 | 0.000 | Native selection/cache/report witnesses for the frozen trait classifier. |
| [tests.test_trait_review_ui](#teststest_trait_review_ui) | 2 | 0.000 | Actual frozen CUDA trait predictions must reach the native review widgets. |
| [tests.test_trait_snapshot](#teststest_trait_snapshot) | 2 | 0.000 | Portable trait evidence and actual relocated CUDA/desktop/batch consumers. |
| [tests.test_trait_training](#teststest_trait_training) | 4 | 0.000 | Synthetic native evidence, CUDA extraction, stop and publication witnesses. |
| [tests.test_ui_text_limits](#teststest_ui_text_limits) | 3 | 0.000 | Keep command labels short and tooltips readable across the desktop UI. |
| [tests.test_visualization_layers](#teststest_visualization_layers) | 2 | 0.000 | Analysis visualization layers, overlays, crops and display contracts |

## tests.test_advanced_analysis

Advanced image layers and CUDA analysis products

Source: [tests/test_advanced_analysis.py](../../tests/test_advanced_analysis.py). Category: Image analysis, calibration and rendering.

### AdvancedAnalysisTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_advanced_analysis.py#L328) | 6 related AdvancedAnalysisTests scenarios, each with private fixture state. | `helper: tests.test_advanced_analysis.AdvancedAnalysisTests.check_test_all_fourteen_products_and_intermediates_are_renderable: self.assertEqual(layers.backend.used, 'cpu')` | unrecorded / — |

## tests.test_analysis_settings_persistence

Analysis settings serialization, normalization and private workspace state

Source: [tests/test_analysis_settings_persistence.py](../../tests/test_analysis_settings_persistence.py). Category: Pipeline workspace and controls.

### AnalysisSettingsPersistenceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_analysis_settings_persistence.py#L427) | 4 related AnalysisSettingsPersistenceTests scenarios, each with private fixture state. | `helper: tests.test_analysis_settings_persistence.AnalysisSettingsPersistenceTests.check_test_json_round_trip_is_human_readable_and_analysis_only: self.assertEqual(analysis_settings_profile_to_payload(loaded), analysis_settings_profile_to_payload(analysis_se…` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_analysis_settings_persistence.py#L434) | 7 related AnalysisSettingsPersistenceTests scenarios, each with private fixture state. | `helper: tests.test_analysis_settings_persistence.AnalysisSettingsPersistenceTests.check_test_graph_compatibility_is_fully_validated_before_mutation: self.assertRaisesRegex(IncompatibleAnalysisSettingsProfile, 'minimum_marker_score')` | unrecorded / — |

## tests.test_annotation_centres

Annotation centroids and radial hilum metadata must never become predictions.

Source: [tests/test_annotation_centres.py](../../tests/test_annotation_centres.py). Category: Annotation and interactive image review.

### AnnotationGeometryTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_annotation_centres.py#L67) | 3 related AnnotationGeometryTests scenarios, each with private fixture state. | `helper: tests.test_annotation_centres.AnnotationGeometryTests.check_test_median_length_is_the_long_axis_at_any_rotation: self.assertAlmostEqual(annotation_median_length(labels), 80.0, delta=1.0)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_annotation_centres.py#L73) | 1 related AnnotationGeometryTests scenarios, each with private fixture state. | `helper: tests.test_annotation_centres.AnnotationGeometryTests.check_test_empty_invalid_and_degenerate_geometry: self.assertEqual(annotation_centres(labels), {})` | unrecorded / — |
### AnnotationCentresUiTests

Fixtures: `setUpClass` at line 81.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_annotation_centres.py#L546) | 12 related AnnotationCentresUiTests scenarios, each with private fixture state. | `helper: tests.test_annotation_centres.AnnotationCentresUiTests.check_test_crosshairs_span_a_quarter_of_the_median_seed_at_every_zoom: self.assertAlmostEqual(columns[-1] - columns[0] + 1, 30.0 * zoom, delta=5.0)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_annotation_centres.py#L561) | 1 related AnnotationCentresUiTests scenarios, each with private fixture state. | `helper: tests.test_annotation_centres.AnnotationCentresUiTests.check_test_next_unannotated_cycles_painted_seeds_missing_condition_or_shape: self.assertIsNotNone(key)` | unrecorded / — |

## tests.test_annotation_edge_sources

Annotation tools select the intended edge evidence and clear obsolete sources

Source: [tests/test_annotation_edge_sources.py](../../tests/test_annotation_edge_sources.py). Category: Annotation and interactive image review.

### AnnotationEdgeSourceTests

Fixtures: `setUpClass` at line 18.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_annotation_edge_sources.py#L438) | 2 related AnnotationEdgeSourceTests scenarios, each with private fixture state. | `helper: tests.test_annotation_edge_sources.AnnotationEdgeSourceTests.check_test_trace_and_fill_selectors_expose_only_applicable_edge_sources: self.assertTrue(image.save(str(image_path)))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_annotation_edge_sources.py#L443) | 1 related AnnotationEdgeSourceTests scenarios, each with private fixture state. | `helper: tests.test_annotation_edge_sources.AnnotationEdgeSourceTests.check_test_image_view_adapts_each_edge_source_and_invalidates_cached_values: self.assertTrue(image.save(str(image_path)))` | unrecorded / — |

## tests.test_annotation_generation_binding

Native label producers bind to the displayed corrected source, including history.

Source: [tests/test_annotation_generation_binding.py](../../tests/test_annotation_generation_binding.py). Category: Annotation and interactive image review.

### AnnotationGenerationBindingTests

Fixtures: `setUpClass` at line 23, `setUp` at line 27.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_annotation_generation_binding.py#L296) | 12 related AnnotationGenerationBindingTests scenarios, each with private fixture state. | `helper: tests.test_annotation_generation_binding.AnnotationGenerationBindingTests.check_test_actual_mouse_brush_binds_source_and_retains_legacy_signal: self.assertTrue(np.any(self.current_labels()))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_annotation_generation_binding.py#L311) | 5 related AnnotationGenerationBindingTests scenarios, each with private fixture state. | `helper: tests.test_annotation_generation_binding.AnnotationGenerationBindingTests.check_test_queued_old_result_and_empty_events_are_rejected: np.testing.assert_array_equal(self.current_labels(), before)` | unrecorded / — |

## tests.test_annotation_shape_revision

Shape review revisions, visibility and reference-library updates

Source: [tests/test_annotation_shape_revision.py](../../tests/test_annotation_shape_revision.py). Category: Annotation and interactive image review.

### ShapeRevisionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_curvature_prior_changes_only_trace_and_downstream_cache`](../../tests/test_annotation_shape_revision.py#L147) | curvature prior changes only trace and downstream cache | `helper: tests.test_annotation_shape_revision.ShapeRevisionTests.check_test_curvature_prior_changes_only_trace_and_downstream_cache: self.assertIs(cache['layer.edge_gradients'], gradients)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_annotation_shape_revision.py#L150) | 5 related ShapeRevisionTests scenarios, each with private fixture state. | `helper: tests.test_annotation_shape_revision.ShapeRevisionTests.check_test_old_shape_libraries_cannot_be_relabelled_after_source_exclusion: self.assertIsNone(resolved.artifact.dimensions_shape)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_annotation_shape_revision.py#L158) | 1 related ShapeRevisionTests scenarios, each with private fixture state. | `helper: tests.test_annotation_shape_revision.ShapeRevisionTests.check_test_boundary_distribution_is_seed_balanced_and_rotation_invariant: self.assertAlmostEqual(sum(summaries[0].boundary_turn_degrees), 360.0, delta=1.0)` | unrecorded / — |
### AnnotationWidgetTests

Fixtures: `setUpClass` at line 166.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_annotation_shape_revision.py#L477) | 6 related AnnotationWidgetTests scenarios, each with private fixture state. | `helper: tests.test_annotation_shape_revision.AnnotationWidgetTests.check_test_distribution_has_axes_units_pose_key_and_error_bars: self.assertIn('Maximum span (mm)', text)` | unrecorded / — |

## tests.test_annotation_tools

Annotation drawing, brush geometry, region operations and instance tools

Source: [tests/test_annotation_tools.py](../../tests/test_annotation_tools.py). Category: Annotation and interactive image review.

### AnnotationToolTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_annotation_tools.py#L453) | 12 related AnnotationToolTests scenarios, each with private fixture state. | `helper: tests.test_annotation_tools.AnnotationToolTests.check_test_edge_cursor_prefers_nearby_ridge_to_stronger_parallel_ridge: self.assertTupleEqual(snap_edge_point((30, 20), edge, 12), (30, 20))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_annotation_tools.py#L468) | 4 related AnnotationToolTests scenarios, each with private fixture state. | `helper: tests.test_annotation_tools.AnnotationToolTests.check_test_smart_fill_adapts_to_touching_colours_and_stops_at_edge: self.assertGreater(added, 900)` | unrecorded / — |

## tests.test_annotation_transfer

Project annotation folders, unsaved drafts and annotation bundle export, restore and manifest validation

Source: [tests/test_annotation_transfer.py](../../tests/test_annotation_transfer.py). Category: Annotation and interactive image review.

### ProjectAnnotationFolderTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_folder_sits_beside_the_project_file`](../../tests/test_annotation_transfer.py#L42) | folder sits beside the project file | `self.assertEqual(directory, Path('D:/work/analysis2.seedfiddle-annotations')); self.assertEqual(annotation_folder(directory, MANUAL_SEED_CENTRES_SIDECAR, unsaved=True), directory / 'unsaved' / 'manual-seed-centres')` | unrecorded / — |
| [`test_unsaved_files_are_listed_and_discarded`](../../tests/test_annotation_transfer.py#L55) | unsaved files are listed and discarded | `self.assertEqual(unsaved_annotation_files(directory), (draft,)); self.assertEqual(saved.read_bytes(), draft.read_bytes())` | unrecorded / — |
### AnnotationBundleTests

Fixtures: `setUp` at line 79, `tearDown` at line 100.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_round_trip_restores_both_kinds_for_the_same_images`](../../tests/test_annotation_transfer.py#L123) | round trip restores both kinds for the same images | `self.assertEqual([(entry.image_name, entry.kind) for entry in entries], [('a.png', MANUAL_SEED_CENTRES_SIDECAR), ('b.png', REFERENCE_REGIONS_SIDECAR)]); self.assertEqual(entries[0].image_sha256, file_sha256(self.first))` | unrecorded / — |
| [`test_export_rejects_an_archive_drawn_on_another_image`](../../tests/test_annotation_transfer.py#L143) | export rejects an archive drawn on another image | `self.assertRaises(ManualSeedCentreFingerprintMismatch); self.assertFalse(self.bundle.exists())` | unrecorded / — |
| [`test_reading_rejects_unsafe_or_ambiguous_manifests`](../../tests/test_annotation_transfer.py#L152) | reading rejects unsafe or ambiguous manifests | `self.assertRaises(AnnotationTransferError); self.assertRaises(AnnotationTransferError)` | unrecorded / — |

## tests.test_assignment_budget

Assignment metrics and segmentation matching respect explicit work budgets

Source: [tests/test_assignment_budget.py](../../tests/test_assignment_budget.py). Category: Learning and trait analysis.

### AssignmentBudgetTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_assignment_budget.py#L98) | 2 related AssignmentBudgetTests scenarios, each with private fixture state. | `helper: tests.test_assignment_budget.AssignmentBudgetTests.check_test_large_one_column_evaluator_component_is_rejected_before_matrix_allocation: self.assertRaisesRegex(MemoryError, 'assignment budget')` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_assignment_budget.py#L103) | 3 related AssignmentBudgetTests scenarios, each with private fixture state. | `helper: tests.test_assignment_budget.AssignmentBudgetTests.check_test_direct_solver_preflights_dummy_columns_before_cost_allocation: self.assertRaisesRegex(MemoryError, 'assignment budget')` | unrecorded / — |

## tests.test_authoritative_parity

Real native CUDA analysis, desktop/batch identity and exact review portability.

Source: [tests/test_authoritative_parity.py](../../tests/test_authoritative_parity.py). Category: Image analysis, calibration and rendering.

### AuthoritativeParityTests

Fixtures: `setUpClass` at line 29, `setUp` at line 36.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_missing_model_and_incompatible_review_are_explicit_on_both_routes`](../../tests/test_authoritative_parity.py#L172) | missing model and incompatible review are explicit on both routes | `self.assertTrue(all((row['trait_prediction']['abstention_reason'] == 'model_file_missing' for row in desktop['instances']))); self.assertEqual(batch['review_lookup'], 'incompatible')` | unrecorded / — |
### ReviewLookupTests

Fixtures: `setUpClass` at line 191.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_invalid_duplicate_oversized_and_stale_reviews_remain_distinct_and_unchanged`](../../tests/test_authoritative_parity.py#L197) | invalid duplicate oversized and stale reviews remain distinct and unchanged | `self.assertEqual(lookup_result_review(root, revision), (None, 'not_recorded')); self.assertEqual(lookup_result_review(root, revision), (None, 'malformed'))` | unrecorded / — |

## tests.test_automatic_material_references

Initial Foreground, Background and Other references without painting.

Source: [tests/test_automatic_material_references.py](../../tests/test_automatic_material_references.py). Category: Image analysis, calibration and rendering.

### AutomaticMaterialReferenceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_sources`](../../tests/test_automatic_material_references.py#L371) | 7 related AutomaticMaterialReferenceTests scenarios, each with private fixture state. | `helper: tests.test_automatic_material_references.AutomaticMaterialReferenceTests.check_test_rim_support_separates_a_visible_vessel_from_a_fitted_circle: self.assertGreater(ar.vessel_rim_support(lab_of(image), dish), 0.95)` | unrecorded / — |
| [`test_weighting`](../../tests/test_automatic_material_references.py#L382) | 4 related AutomaticMaterialReferenceTests scenarios, each with private fixture state. | `helper: tests.test_automatic_material_references.AutomaticMaterialReferenceTests.check_test_controls_switch_classes_and_shape_the_areas: self.assertIsNone(off.foreground)` | unrecorded / — |

## tests.test_backend_outcomes

Actual backend outcomes, cache recovery, Qt/batch parity and immutable review binding.

Source: [tests/test_backend_outcomes.py](../../tests/test_backend_outcomes.py). Category: Persistence, export, identity and outcomes.

### BackendOutcomeTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_backend_outcomes.py#L149) | 6 related BackendOutcomeTests scenarios, each with private fixture state. | `helper: tests.test_backend_outcomes.BackendOutcomeTests.check_test_available_empty_missing_and_disabled_are_distinct_and_round_trip: self.assertEqual(records['procedural_instances']['state'], 'available')` | unrecorded / — |

## tests.test_benchmark_cli

Single-entry CLI routing and bounded adapters; GPU producers tested separately.

Source: [tests/test_benchmark_cli.py](../../tests/test_benchmark_cli.py). Category: Benchmarks, added nodes and reference views.

### BenchmarkCLITests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_benchmark_cli.py#L90) | 5 related BenchmarkCLITests scenarios, each with private fixture state. | `helper: tests.test_benchmark_cli.BenchmarkCLITests.check_test_benchmark_actions_are_exclusive_with_each_other_and_learning: self.assertRaises(SystemExit)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_benchmark_cli.py#L98) | 1 related BenchmarkCLITests scenarios, each with private fixture state. | `helper: tests.test_benchmark_cli.BenchmarkCLITests.check_test_output_and_evidence_arguments_fail_before_runtime_bootstrap: self.assertEqual(launch.main(args), 2)` | unrecorded / — |

## tests.test_benchmark_instances

The instance benchmark harness scores source-frame masks and states the reference caveat.

Source: [tests/test_benchmark_instances.py](../../tests/test_benchmark_instances.py). Category: Benchmarks, added nodes and reference views.

### BenchmarkInstanceHarnessTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_benchmark_harness`](../../tests/test_benchmark_instances.py#L32) | benchmark harness | `helper: tests.test_benchmark_instances.BenchmarkInstanceHarnessTests.check_scores_methods_and_restricts_to_dish_region: self.assertIn('reviewed: false', report['caveat'])` | unrecorded / — |

## tests.test_benchmark_manifest

Benchmark manifests, source/environment identity and evidence contracts

Source: [tests/test_benchmark_manifest.py](../../tests/test_benchmark_manifest.py). Category: Benchmarks, added nodes and reference views.

### BenchmarkManifestTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_benchmark_manifest.py#L586) | 9 related BenchmarkManifestTests scenarios, each with private fixture state. | `helper: tests.test_benchmark_manifest.BenchmarkManifestTests.check_test_round_trip_is_immutable_strict_and_does_not_mutate_payload: self.assertEqual(payload, original)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_benchmark_manifest.py#L598) | 6 related BenchmarkManifestTests scenarios, each with private fixture state. | `helper: tests.test_benchmark_manifest.BenchmarkManifestTests.check_test_missing_groups_truth_or_access_log_blocks_evaluation_without_guessing: self.assertFalse(audit.evaluation_eligible)` | unrecorded / — |

## tests.test_bootstrap

Python/runtime dependency admission and single-entry startup

Source: [tests/test_bootstrap.py](../../tests/test_bootstrap.py). Category: Startup, resources and diagnostics.

### VersionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_bootstrap.py#L63) | 2 related VersionTests scenarios, each with private fixture state. | `helper: tests.test_bootstrap.VersionTests.check_test_version_key_accepts_common_suffixes: self.assertEqual(version_key('2.13.0+cu130'), (2, 13, 0))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_bootstrap.py#L68) | 1 related VersionTests scenarios, each with private fixture state. | `helper: tests.test_bootstrap.VersionTests.check_test_invalid_version_is_rejected: self.assertRaises(ValueError)` | unrecorded / — |
### DependencyInspectionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_bootstrap.py#L115) | 1 related DependencyInspectionTests scenarios, each with private fixture state. | `helper: tests.test_bootstrap.DependencyInspectionTests.check_test_missing_import_requires_installation: self.assertFalse(check.installed)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_bootstrap.py#L119) | 3 related DependencyInspectionTests scenarios, each with private fixture state. | `helper: tests.test_bootstrap.DependencyInspectionTests.check_test_compatible_supplied_package_is_kept: self.assertTrue(check.installed)` | unrecorded / — |
### PipCommandTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_bootstrap.py#L233) | 6 related PipCommandTests scenarios, each with private fixture state. | `helper: tests.test_bootstrap.PipCommandTests.check_test_online_command_can_prefer_local_wheels: self.assertIn('--find-links', command)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_bootstrap.py#L242) | 1 related PipCommandTests scenarios, each with private fixture state. | `helper: tests.test_bootstrap.PipCommandTests.check_test_offline_command_rejects_an_empty_wheelhouse: self.assertRaises(BootstrapError)` | unrecorded / — |
### ManifestTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_repository_manifest_loads`](../../tests/test_bootstrap.py#L248) | repository manifest loads | `self.assertEqual(distributions, {'PySide6', 'numpy', 'opencv-python', 'matplotlib', 'torch', 'cellpose'}); self.assertFalse(cellpose.required)` | unrecorded / — |
| [`test_trait_manifest_contains_agreed_species_and_damage_fields`](../../tests/test_bootstrap.py#L397) | trait manifest contains agreed species and damage fields | `self.assertEqual(species, {'soybean', 'lupinus_mutabilis', 'lupinus_polyphyllus', 'lupinus_mexicanus'}); self.assertEqual(set(traits['damage_flags']), {'cracking', 'missing_or_peeling_coat', 'chipping_or_breakage', 'discoloration'})` | unrecorded / — |
### CellposeInstallTests

Fixtures: `setUpClass` at line 437.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_bootstrap.py#L524) | 4 Cellpose-SAM installation scenarios, each with private fixture state. | `helper: tests.test_bootstrap.CellposeInstallTests.check_test_install_keeps_pytorch_and_gui_opencv: self.assertIn('torchvision', torchvision)` | unrecorded / — |
### CudaRuntimeTests

Fixtures: `setUpClass` at line 534.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_build_and_device_readiness_are_classified_separately`](../../tests/test_bootstrap.py#L576) | build and device readiness are classified separately | `self.assertTrue(working.ready); self.assertTrue(working.cuda_ready)` | unrecorded / — |
| [`test_probe_handles_timeout_crash_import_and_malformed_output`](../../tests/test_bootstrap.py#L633) | probe handles timeout crash import and malformed output | `self.assertEqual(timed_out.state, 'timeout'); self.assertIn('timeout', timed_out.failure_detail)` | unrecorded / — |
| [`test_probe_parses_real_calculation_status_from_selected_interpreter`](../../tests/test_bootstrap.py#L675) | probe parses real calculation status from selected interpreter | `self.assertEqual(result.python_executable, 'selected-python.exe'); self.assertEqual(result.state, 'completed')` | unrecorded / — |
| [`test_diagnostics_separate_build_and_real_device_execution`](../../tests/test_bootstrap.py#L705) | diagnostics separate build and real device execution | `self.assertIn('PyTorch build: NVIDIA CUDA build', output); self.assertIn('CUDA device execution: FAIL', output)` | unrecorded / — |

## tests.test_calibration_result_revision

Actual calibration state must travel with authoritative review revisions.

Source: [tests/test_calibration_result_revision.py](../../tests/test_calibration_result_revision.py). Category: Persistence, export, identity and outcomes.

### CalibrationResultRevisionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_calibration_result_revision.py#L85) | 4 related CalibrationResultRevisionTests scenarios, each with private fixture state. | `helper: tests.test_calibration_result_revision.CalibrationResultRevisionTests.check_test_neutral_request_and_witness_changes_bind_revision: self.assertNotEqual(old['result_revision'], frame_revision)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_calibration_result_revision.py#L92) | 1 related CalibrationResultRevisionTests scenarios, each with private fixture state. | `helper: tests.test_calibration_result_revision.CalibrationResultRevisionTests.check_test_invalid_calibration_numbers_or_inconsistent_neutral_facts_are_unreviewable: self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_catalogue_nodes

Independent catalogue instances, typed projections and portable project state.

Source: [tests/test_catalogue_nodes.py](../../tests/test_catalogue_nodes.py). Category: Benchmarks, added nodes and reference views.

### CatalogueNodeTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_catalogue_nodes.py#L128) | 3 related CatalogueNodeTests scenarios, each with private fixture state. | `helper: tests.test_catalogue_nodes.CatalogueNodeTests.check_test_catalogue_lists_each_default_node_once: self.assertEqual(len(entries), 52)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_catalogue_nodes.py#L134) | 1 related CatalogueNodeTests scenarios, each with private fixture state. | `helper: tests.test_catalogue_nodes.CatalogueNodeTests.check_test_cross_instance_edges_and_duplicate_or_unregistered_descriptors_reject: self.assertRaisesRegex(ValueError, 'explicit adapter')` | unrecorded / — |

## tests.test_catalogue_nodes_ui

CPU/Qt ownership, routing and stale-publication checks for added nodes.

Source: [tests/test_catalogue_nodes_ui.py](../../tests/test_catalogue_nodes_ui.py). Category: Benchmarks, added nodes and reference views.

### CatalogueNodesControllerTests

Fixtures: `setUpClass` at line 55, `setUp` at line 58, `tearDown` at line 64.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_catalogue_nodes_ui.py#L157) | 3 related CatalogueNodesControllerTests scenarios, each with private fixture state. | `helper: tests.test_catalogue_nodes_ui.CatalogueNodesControllerTests.check_test_duplicate_default_instances_own_controls_cache_and_prerequisite_presence: self.assertNotEqual(first, second)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_catalogue_nodes_ui.py#L163) | 2 related CatalogueNodesControllerTests scenarios, each with private fixture state. | `helper: tests.test_catalogue_nodes_ui.CatalogueNodesControllerTests.check_test_queue_waits_for_primary_and_stop_withholds_old_result: self.assertIsNone(component.result)` | unrecorded / — |

## tests.test_cellpose_sam

Optional Cellpose-SAM branch: honest settings, flow decoding and engine caching.

Source: [tests/test_cellpose_sam.py](../../tests/test_cellpose_sam.py). Category: Image analysis, calibration and rendering.

### CellposeSamTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_branch_contract`](../../tests/test_cellpose_sam.py#L164) | 5 related CellposeSamTests scenarios, each with private fixture state. | `helper: tests.test_cellpose_sam.CellposeSamTests.check_test_settings_reject_values_no_calculation_could_use: self.assertRaises(ValueError)` | unrecorded / — |
### CellposeSamModelTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_pretrained_model_separates_dark_seeds_on_a_light_dish`](../../tests/test_cellpose_sam.py#L177) | pretrained model separates dark seeds on a light dish | `self.assertIn('#sha256=', producer); self.assertEqual(int(result.labels.max()), 5)` | unrecorded / — |

## tests.test_chart_colour_correction

Colour correction against a chart's published reference colours.

Source: [tests/test_chart_colour_correction.py](../../tests/test_chart_colour_correction.py). Category: Image analysis, calibration and rendering.

### ChartColourCorrectionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_reference_colours`](../../tests/test_chart_colour_correction.py#L210) | 3 related ChartColourCorrectionTests scenarios, each with private fixture state. | `helper: tests.test_chart_colour_correction.ChartColourCorrectionTests.check_test_published_lab_values_reproduce_the_published_srgb_table: np.testing.assert_allclose(srgb[index], expected, atol=3.0)` | unrecorded / — |
| [`test_fitting`](../../tests/test_chart_colour_correction.py#L216) | 4 related ChartColourCorrectionTests scenarios, each with private fixture state. | `helper: tests.test_chart_colour_correction.ChartColourCorrectionTests.check_test_identified_chart_fit_recovers_the_camera: self.assertEqual(facts.model, 'chart_reference_fit')` | unrecorded / — |
| [`test_calibration`](../../tests/test_chart_colour_correction.py#L224) | 3 related ChartColourCorrectionTests scenarios, each with private fixture state. | `helper: tests.test_chart_colour_correction.ChartColourCorrectionTests.check_test_gpu_application_matches_the_numpy_transform: self.assertEqual(result.device.type, tensor.device.type)` | unrecorded / — |

## tests.test_colour_correction

Chart-fitted colour-correction matrices, CUDA application and fit diagnostics

Source: [tests/test_colour_correction.py](../../tests/test_colour_correction.py). Category: Image analysis, calibration and rendering.

### ColourCorrectionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_colour_correction.py#L127) | 3 related ColourCorrectionTests scenarios, each with private fixture state. | `helper: tests.test_colour_correction.ColourCorrectionTests.check_test_chip_rgb_means_and_headers_match_numpy: self.assertEqual(headers, ('chip_number', 'r_avg', 'g_avg', 'b_avg'))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_colour_correction.py#L133) | 3 related ColourCorrectionTests scenarios, each with private fixture state. | `helper: tests.test_colour_correction.ColourCorrectionTests.check_test_tiled_chip_reduction_and_affine_cancellation_preserve_input: np.testing.assert_array_equal(colour_matrix_cuda(image, mask)[1], expected)` | unrecorded / — |

## tests.test_conditioned_shape_pooling

Conditioned pooling uses the same Gaussian model without dropping directions.

Source: [tests/test_conditioned_shape_pooling.py](../../tests/test_conditioned_shape_pooling.py). Category: Measurements and shape models.

### ConditionedShapePoolingTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_conditioned_shape_pooling.py#L171) | 8 related ConditionedShapePoolingTests scenarios, each with private fixture state. | `helper: tests.test_conditioned_shape_pooling.ConditionedShapePoolingTests.check_test_anisotropic_diagonal_pooling_matches_independent_scalar_gaussian_oracle: self.assertEqual(result.state, 'applied')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_conditioned_shape_pooling.py#L182) | 2 related ConditionedShapePoolingTests scenarios, each with private fixture state. | `helper: tests.test_conditioned_shape_pooling.ConditionedShapePoolingTests.check_test_conditioned_signed_mean_cancellation_is_valid: self.assertEqual(result.state, 'applied')` | unrecorded / — |

## tests.test_critical_review_fixes

Earlier regression fixes spanning annotation coordinates, learning and export

Source: [tests/test_critical_review_fixes.py](../../tests/test_critical_review_fixes.py). Category: Image analysis, calibration and rendering.

### CriticalIntegrityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_critical_review_fixes.py#L274) | 10 related CriticalIntegrityTests scenarios, each with private fixture state. | `helper: tests.test_critical_review_fixes.CriticalIntegrityTests.check_test_checkpoint_cache_uses_content_and_result_rejects_replaced_weights: self.assertEqual(loader.call_count, 2)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_critical_review_fixes.py#L287) | 9 related CriticalIntegrityTests scenarios, each with private fixture state. | `helper: tests.test_critical_review_fixes.CriticalIntegrityTests.check_test_streamed_evaluation_both_families_and_hybrid: self.assertFalse(report['scientifically_validated'])` | unrecorded / — |

## tests.test_default_pipeline_copies

Default pipelines stay locked; their edit controls offer a named, editable copy.

Source: [tests/test_default_pipeline_copies.py](../../tests/test_default_pipeline_copies.py). Category: Image analysis, calibration and rendering.

### DefaultPipelineCopyTests

Fixtures: `setUpClass` at line 45, `setUp` at line 48, `tearDown` at line 57.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_default_pipeline_copies.py#L470) | 9 related DefaultPipelineCopyTests scenarios, each with private fixture state. | `helper: tests.test_default_pipeline_copies.DefaultPipelineCopyTests.check_test_list_offers_the_default_pipelines_then_the_project_copies: self.assertNotIn('Settings', analysis)` | unrecorded / — |
| [`test_refusal_and_recovery`](../../tests/test_default_pipeline_copies.py#L482) | 6 related DefaultPipelineCopyTests scenarios, each with private fixture state. | `helper: tests.test_default_pipeline_copies.DefaultPipelineCopyTests.check_test_declining_keeps_the_default_and_never_asks_for_a_name: self.assert_locked_default()` | unrecorded / — |

## tests.test_deferred_callbacks

Deferred callbacks die with their owner and report what they raise.

Source: [tests/test_deferred_callbacks.py](../../tests/test_deferred_callbacks.py). Category: Image analysis, calibration and rendering.

### DeferredCallbackTests

Fixtures: `setUpClass` at line 56.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_callback_runs_in_one_event_pass_in_order_and_leaves_no_timer`](../../tests/test_deferred_callbacks.py#L81) | callback runs in one event pass in order and leaves no timer | `self.assertEqual(calls, []); self.assertEqual(calls, ['first', 'second'])` | unrecorded / — |
| [`test_deleting_the_owner_cancels_the_callback`](../../tests/test_deferred_callbacks.py#L90) | deleting the owner cancels the callback | `self.assertEqual(calls, [])` | unrecorded / — |
| [`test_raising_callback_is_reported_and_new_cards_paint_during_process_events`](../../tests/test_deferred_callbacks.py#L98) | raising callback is reported and new cards paint during process events | `self.assertEqual(errors, ['raised by a deferred callback'])` | unrecorded / — |
| [`test_raising_callback_is_reported_and_new_cards_paint_during_a_static_dialog`](../../tests/test_deferred_callbacks.py#L105) | raising callback is reported and new cards paint during a static dialog | `self.assertEqual(errors, ['raised by a deferred callback']); self.assertTrue(all((card.paints for card in cards)))` | unrecorded / — |
| [`test_deleted_window_with_a_pending_workspace_restore_cannot_harm_a_new_canvas`](../../tests/test_deferred_callbacks.py#L117) | deleted window with a pending workspace restore cannot harm a new canvas | `self.assertEqual(errors, [])` | unrecorded / — |
| [`test_application_code_defers_only_through_owned_timers`](../../tests/test_deferred_callbacks.py#L141) | application code defers only through owned timers | `self.assertEqual(users, [], 'Use seedvision.ui.deferred.run_soon, not QTimer.singleShot.')` | unrecorded / — |

## tests.test_desktop_terminal_history

Earlier desktop rejection boundaries and bounded reading of saved run records.

Source: [tests/test_desktop_terminal_history.py](../../tests/test_desktop_terminal_history.py). Category: Persistence, export, identity and outcomes.

### TerminalHistoryTests

Fixtures: `setUp` at line 24.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_desktop_terminal_history.py#L126) | 3 related TerminalHistoryTests scenarios, each with private fixture state. | `helper: tests.test_desktop_terminal_history.TerminalHistoryTests.check_test_loading_old_native_and_preflight_records_does_not_probe_deleted_source_or_implementation: self.assertLogs('seedvision.export.run_outcomes', level='ERROR')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_desktop_terminal_history.py#L132) | 4 related TerminalHistoryTests scenarios, each with private fixture state. | `helper: tests.test_desktop_terminal_history.TerminalHistoryTests.check_test_identical_copies_deduplicate_but_conflicting_diagnostics_withhold_all_copies: self.assertEqual(len(history.load_run_outcome_history(self.directory).outcomes), 1)` | unrecorded / — |
### DesktopTerminalHistoryTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_desktop_terminal_history.py#L228) | 2 related DesktopTerminalHistoryTests scenarios, each with private fixture state. | `helper: tests.test_desktop_terminal_history.DesktopTerminalHistoryTests.check_test_project_apply_failure_keeps_existing_image_and_drafts_before_session_clear: self.assertFalse(self.window._open_project(self.master))` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_desktop_terminal_history.py#L233) | 1 related DesktopTerminalHistoryTests scenarios, each with private fixture state. | `helper: tests.test_desktop_terminal_history.DesktopTerminalHistoryTests.check_results_says_why_the_open_images_run_did_not_complete: opened.assert_not_called()` | unrecorded / — |

## tests.test_diagnostics

Runtime diagnostic facts

Source: [tests/test_diagnostics.py](../../tests/test_diagnostics.py). Category: Startup, resources and diagnostics.

### DiagnosticLoggingTests

Fixtures: `tearDown` at line 20.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_rotating_log_retains_full_exception_traceback`](../../tests/test_diagnostics.py#L23) | rotating log retains full exception traceback | `self.assertIn('Could not install completed analysis', text); self.assertIn('Traceback (most recent call last):', text)` | unrecorded / — |

## tests.test_edge_trace_commit

Edge trace tool commits and desktop annotation updates

Source: [tests/test_edge_trace_commit.py](../../tests/test_edge_trace_commit.py). Category: Annotation and interactive image review.

### EdgeTraceCommitTests

Fixtures: `setUpClass` at line 17.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_edge_trace_commit.py#L177) | 4 related EdgeTraceCommitTests scenarios, each with private fixture state. | `helper: tests.test_edge_trace_commit.EdgeTraceCommitTests.check_test_open_trace_is_exactly_one_pixel_independent_of_brush_radius: self.assertEqual(changed, 49)` | unrecorded / — |

## tests.test_entrypoint

Native desktop application launch and command routing

Source: [tests/test_entrypoint.py](../../tests/test_entrypoint.py). Category: Startup, resources and diagnostics.

### EntrypointTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_entrypoint.py#L318) | 3 related EntrypointTests scenarios, each with private fixture state. | `helper: tests.test_entrypoint.EntrypointTests.check_test_importing_entrypoint_does_not_import_gui_or_ml_packages: self.assertEqual(completed.stdout.strip(), '')` | unrecorded / — |
| [`test_cellpose_installation`](../../tests/test_entrypoint.py#L324) | 3 Cellpose-SAM launch scenarios, each with isolated bootstrap mocks. | `helper: tests.test_entrypoint.EntrypointTests.check_test_ready_launch_adds_cellpose_unless_told_not_to: self.assertEqual(statuses, [0, 0, 0, 0])` | unrecorded / — |
| [`test_cuda_readiness_enforcement`](../../tests/test_entrypoint.py#L330) | 6 readiness scenarios, each with isolated bootstrap mocks. | `helper: tests.test_entrypoint.EntrypointTests.check_test_failed_cuda_readiness_blocks_launch_and_diagnostics_returns_nonzero: self.assertEqual(diagnostics_status, 2)` | unrecorded / — |

## tests.test_graph_default_pipelines

Procedural and Cellpose seed analysis are one catalogue with different active nodes.

Source: [tests/test_graph_default_pipelines.py](../../tests/test_graph_default_pipelines.py). Category: Image analysis, calibration and rendering.

### GraphDefaultPipelineTests

Fixtures: `setUpClass` at line 34, `setUp` at line 37, `tearDown` at line 45.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_the_two_graphs`](../../tests/test_graph_default_pipelines.py#L181) | 3 related GraphDefaultPipelineTests scenarios, each with private fixture state. | `helper: tests.test_graph_default_pipelines.GraphDefaultPipelineTests.check_test_procedural_excludes_only_cellpose: self.assertEqual(set(graph.nodes), set(whole.nodes) - {CELLPOSE_NODE})` | unrecorded / — |
| [`test_choosing_between_them`](../../tests/test_graph_default_pipelines.py#L187) | 9 related GraphDefaultPipelineTests scenarios, each with private fixture state. | `helper: tests.test_graph_default_pipelines.GraphDefaultPipelineTests.check_test_list_changes_the_graph_and_back: self.assert_on(PROCEDURAL_SEED_ANALYSIS, procedural)` | unrecorded / — |

## tests.test_grouped_comparison

Compact synthetic engineering witnesses, not biological validation.

Source: [tests/test_grouped_comparison.py](../../tests/test_grouped_comparison.py). Category: Benchmarks, added nodes and reference views.

### GroupedComparisonTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_grouped_comparison.py#L278) | 6 related GroupedComparisonTests scenarios, each with private fixture state. | `helper: tests.test_grouped_comparison.GroupedComparisonTests.check_test_constant_paired_shift_has_exact_bounds_and_never_certifies_science: self.assertEqual(report['state'], 'pass')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_grouped_comparison.py#L287) | 8 related GroupedComparisonTests scenarios, each with private fixture state. | `helper: tests.test_grouped_comparison.GroupedComparisonTests.check_test_lower_error_uses_upper_bound_and_absolute_failure_is_not_equivalence: self.assertAlmostEqual(metric['difference_one_sided_bound'], 0.1)` | unrecorded / — |

## tests.test_gui_layout

Desktop layout contracts from the GUI review: menus, toolbar, panels, canvas.

Source: [tests/test_gui_layout.py](../../tests/test_gui_layout.py). Category: Image analysis, calibration and rendering.

### GuiLayoutTests

Fixtures: `setUpClass` at line 19, `setUp` at line 24.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_gui_layout.py#L411) | 18 related GuiLayoutTests scenarios, each with private fixture state. | `helper: tests.test_gui_layout.GuiLayoutTests.check_test_every_shortcut_is_unique: self.assertEqual({key: count for key, count in shortcuts.items() if count > 1}, {})` | unrecorded / — |

## tests.test_icon_asset

Application icon assets

Source: [tests/test_icon_asset.py](../../tests/test_icon_asset.py). Category: Startup, resources and diagnostics.

### IconAssetTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_taskbar_icon_uses_the_square_canvas_at_small_sizes`](../../tests/test_icon_asset.py#L11) | taskbar icon uses the square canvas at small sizes | `self.assertIsNotNone(image); self.assertEqual(image.ndim, 3)` | unrecorded / — |

## tests.test_image_calibration

Image calibration, geometry, chart/ruler detection and real-image fixtures

Source: [tests/test_image_calibration.py](../../tests/test_image_calibration.py). Category: Image analysis, calibration and rendering.

### ImageCalibrationTests

Fixtures: `setUpClass` at line 16.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_fixture_rims_match_the_calibrated_96_mm_dish`](../../tests/test_image_calibration.py#L633) | fixture rims match the calibrated 96 mm dish | `helper: tests.test_image_calibration.ImageCalibrationTests.check_test_fixture_rims_match_the_calibrated_96_mm_dish: self.assertIsNotNone(pixels_per_mm)` | unrecorded / — |
| [`test_low_quality_reference_photographs`](../../tests/test_image_calibration.py#L636) | low quality reference photographs | `helper: tests.test_image_calibration.ImageCalibrationTests.check_test_low_quality_photographs_take_scale_from_partial_rulers: self.assertIsNone(result.ruler)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_image_calibration.py#L642) | 3 related ImageCalibrationTests scenarios, each with private fixture state. | `helper: tests.test_image_calibration.ImageCalibrationTests.check_test_scale_is_rejected_when_longest_ticks_are_not_major_increments: self.assertEqual(len(ruler.metric_tick_segments), 151)` | unrecorded / — |
| [`test_reference_orientation`](../../tests/test_image_calibration.py#L648) | 5 ImageCalibrationTests scenarios: charts and rulers at any angle or in part. | `helper: tests.test_image_calibration.ImageCalibrationTests.check_test_colorchecker_is_identified_and_numbered_at_any_angle: self.assertEqual(card.chart, COLORCHECKER_CLASSIC_24.name)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_image_calibration.py#L658) | 5 related ImageCalibrationTests scenarios, each with private fixture state. | `helper: tests.test_image_calibration.ImageCalibrationTests.check_test_swatch_grid_snaps_to_shared_printed_edges: self.assertLess(float(np.max(np.abs(detected - expected))), 2.5)` | unrecorded / — |

## tests.test_image_list_controls

Native image-list actions, thumbnails, file preservation and edit ownership.

Source: [tests/test_image_list_controls.py](../../tests/test_image_list_controls.py). Category: Image analysis, calibration and rendering.

### ImageListControlsTests

Fixtures: `setUpClass` at line 78.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_header_open_and_row_removal_preserve_files_and_unresolved_project_records`](../../tests/test_image_list_controls.py#L81) | header open and row removal preserve files and unresolved project records | `self.assertTrue(window._open_project(project)); self.assertNotIn(window.open_action, window.workflow_toolbar.actions())` | unrecorded / — |
| [`test_removal_protects_edits_and_busy_work_then_clears_the_selection`](../../tests/test_image_list_controls.py#L159) | removal protects edits and busy work then clears the selection | `self.assertFalse(window._remove_image(first)); busy.assert_called_once()` | unrecorded / — |
### ImageThumbnailLoaderTests

Fixtures: `setUpClass` at line 218, `setUp` at line 221.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_background_decoding_and_cache`](../../tests/test_image_list_controls.py#L390) | background decoding and cache | `helper: tests.test_image_list_controls.ImageThumbnailLoaderTests.check_reader_fits_the_box_and_reports_what_it_cannot_read: self.assertEqual(error, '')` | unrecorded / — |
### ImageListRowTests

Fixtures: `setUpClass` at line 406, `setUp` at line 409.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_rows_show_thumbnails_status_and_the_open_image`](../../tests/test_image_list_controls.py#L942) | rows show thumbnails status and the open image | `helper: tests.test_image_list_controls.ImageListRowTests.check_every_row_shows_its_own_image_replacing_the_blank_document: self.assertEqual(self.list.count(), 4)` | unrecorded / — |

## tests.test_instance_annotation_tiling

Annotation tile rendering, invalidation and image-view state

Source: [tests/test_instance_annotation_tiling.py](../../tests/test_instance_annotation_tiling.py). Category: Annotation and interactive image review.

### InstanceAnnotationTilingTests

Fixtures: `setUpClass` at line 16.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_instance_annotation_tiling.py#L327) | 5 related InstanceAnnotationTilingTests scenarios, each with private fixture state. | `helper: tests.test_instance_annotation_tiling.InstanceAnnotationTilingTests.check_test_6240_by_4160_tiles_preserve_exact_one_pixel_marks_and_ridges: self.assertEqual((bounds.x(), bounds.y()), (0.0, 0.0))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_instance_annotation_tiling.py#L335) | 2 related InstanceAnnotationTilingTests scenarios, each with private fixture state. | `helper: tests.test_instance_annotation_tiling.InstanceAnnotationTilingTests.check_test_selected_filter_and_in_place_edit_invalidate_cached_pixels: self.assertIsInstance(item, _InstanceAnnotationTileItem)` | unrecorded / — |

## tests.test_instance_continuity

Instance continuity during annotation updates

Source: [tests/test_instance_continuity.py](../../tests/test_instance_continuity.py). Category: Annotation and interactive image review.

### InstanceContinuityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_instance_continuity.py#L130) | 7 related InstanceContinuityTests scenarios, each with private fixture state. | `helper: tests.test_instance_continuity.InstanceContinuityTests.check_test_empty_raster_has_an_empty_summary: self.assertEqual(summary.identifiers, ())` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_instance_continuity.py#L140) | 1 related InstanceContinuityTests scenarios, each with private fixture state. | `helper: tests.test_instance_continuity.InstanceContinuityTests.check_test_rejects_non_integer_or_out_of_range_labels: self.assertRaisesRegex(ValueError, 'two-dimensional')` | unrecorded / — |

## tests.test_instance_continuity_ui

Desktop selection follows instance continuity

Source: [tests/test_instance_continuity_ui.py](../../tests/test_instance_continuity_ui.py). Category: Annotation and interactive image review.

### InstanceContinuityUiTests

Fixtures: `setUpClass` at line 17, `setUp` at line 25.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_selected_seed_warning_is_exact_and_does_not_block_apply`](../../tests/test_instance_continuity_ui.py#L54) | selected seed warning is exact and does not block apply | `self.assertFalse(warning.isHidden()); self.assertEqual(warning.text(), '⚠ Seed 7 has 2 disconnected areas.')` | unrecorded / — |
| [`test_global_warning_and_identity_cache_update_for_a_new_draft`](../../tests/test_instance_continuity_ui.py#L100) | global warning and identity cache update for a new draft | `self.assertIn('1 seed ID has disconnected areas: 2', warning.text()); self.assertEqual(banner.text(), warning.text())` | unrecorded / — |

## tests.test_instance_mask_import

Instance mask loading, source identity and validation

Source: [tests/test_instance_mask_import.py](../../tests/test_instance_mask_import.py). Category: Annotation and interactive image review.

### InstanceMaskImportTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_instance_mask_import.py#L372) | 8 related InstanceMaskImportTests scenarios, each with private fixture state. | `helper: tests.test_instance_mask_import.InstanceMaskImportTests.check_test_corrected_png_tiff_and_npz_preserve_sparse_ids_exactly: self.assertTrue(cv2.imwrite(str(paths[0]), labels))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_instance_mask_import.py#L383) | 2 related InstanceMaskImportTests scenarios, each with private fixture state. | `helper: tests.test_instance_mask_import.InstanceMaskImportTests.check_test_corrected_import_rejects_shape_type_range_and_corruption: self.assertTrue(cv2.imwrite(str(colour), np.zeros((5, 6, 3), np.uint8)))` | unrecorded / — |

## tests.test_instance_mask_import_ui

Instance mask import dialogs and desktop application

Source: [tests/test_instance_mask_import_ui.py](../../tests/test_instance_mask_import_ui.py). Category: Annotation and interactive image review.

### InstanceMaskImportUiTests

Fixtures: `setUpClass` at line 21, `setUp` at line 29.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_instance_mask_import_ui.py#L608) | 13 related InstanceMaskImportUiTests scenarios, each with private fixture state. | `helper: tests.test_instance_mask_import_ui.InstanceMaskImportUiTests.check_test_panel_import_is_draft_with_provenance_warning_and_undo: chooser.assert_not_called()` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_instance_mask_import_ui.py#L624) | 2 related InstanceMaskImportUiTests scenarios, each with private fixture state. | `helper: tests.test_instance_mask_import_ui.InstanceMaskImportUiTests.check_test_validation_failure_warns_without_mutating_draft_or_history: warning.assert_called_once()` | unrecorded / — |

## tests.test_instance_reference_evidence

Annotated-instance reference evidence and visualization

Source: [tests/test_instance_reference_evidence.py](../../tests/test_instance_reference_evidence.py). Category: Annotation and interactive image review.

### InstanceReferenceEvidenceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_instance_reference_evidence.py#L170) | 5 related InstanceReferenceEvidenceTests scenarios, each with private fixture state. | `helper: tests.test_instance_reference_evidence.InstanceReferenceEvidenceTests.check_test_distinct_instance_contact_is_physical_and_interiors_are_inset: self.assertTrue(np.all(references.physical_edge[13:51, 35:37]))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_instance_reference_evidence.py#L178) | 2 related InstanceReferenceEvidenceTests scenarios, each with private fixture state. | `helper: tests.test_instance_reference_evidence.InstanceReferenceEvidenceTests.check_test_rejects_non_integer_and_negative_instance_ids: self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_instance_reference_repository

Committed reference fixture repository integrity

Source: [tests/test_instance_reference_repository.py](../../tests/test_instance_reference_repository.py). Category: Annotation and interactive image review.

### InstanceReferenceRepositoryTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_manifest_and_every_reference_mask_are_self_consistent`](../../tests/test_instance_reference_repository.py#L137) | manifest and every reference mask are self consistent | `self.assertEqual(payload.get('version'), 1); self.assertIsInstance(entries, list)` | unrecorded / — |

## tests.test_learned_segmentation

Learned segmentation features, targets, model, loss, decoding and metrics

Source: [tests/test_learned_segmentation.py](../../tests/test_learned_segmentation.py). Category: Learning and trait analysis.

### LearnedTargetTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_learned_segmentation.py#L114) | 1 related LearnedTargetTests scenarios, each with private fixture state. | `helper: tests.test_learned_segmentation.LearnedTargetTests.check_test_dense_targets_keep_missing_pattern_annotations_unknown: self.assertEqual(targets.interior.shape, labels.shape)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_learned_segmentation.py#L118) | 2 related LearnedTargetTests scenarios, each with private fixture state. | `helper: tests.test_learned_segmentation.LearnedTargetTests.check_test_stardist_targets_have_radial_geometry_at_seed_centres: self.assertTrue(np.all(targets.radial_valid[:, 32, 32] > 0))` | unrecorded / — |
### LearningDataTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_exported_human_sample_round_trips_and_builds_cached_targets`](../../tests/test_learned_segmentation.py#L125) | exported human sample round trips and builds cached targets | `self.assertEqual(len(manifest.samples), 2); self.assertEqual(Path(manifest.samples[0].physical_boundary).name, 'capture_01.physical.png')` | unrecorded / — |
### LearnedModelTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_models_emit_expected_full_resolution_heads_and_backpropagate`](../../tests/test_learned_segmentation.py#L197) | models emit expected full resolution heads and backpropagate | `self.assertEqual(outputs['interior_logits'].shape, (2, 1, 64, 80)); self.assertEqual(outputs['uncertainty_logits'].shape, (2, 5, 64, 80))` | unrecorded / — |
### LearnedDecoderTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_learned_segmentation.py#L367) | 4 related LearnedDecoderTests scenarios, each with private fixture state. | `helper: tests.test_learned_segmentation.LearnedDecoderTests.check_test_pattern_aware_watershed_decodes_ideal_touching_instances: self.assertEqual(result.count, 3)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_learned_segmentation.py#L374) | 1 related LearnedDecoderTests scenarios, each with private fixture state. | `helper: tests.test_learned_segmentation.LearnedDecoderTests.check_test_hybrid_rejects_stardist_candidates_outside_unet_interior: self.assertEqual(result.count, 3)` | unrecorded / — |

## tests.test_learning_workflow

Training workflow, evidence export and desktop controls

Source: [tests/test_learning_workflow.py](../../tests/test_learning_workflow.py). Category: Learning and trait analysis.

### LearningWorkflowTests

Fixtures: `setUpClass` at line 19.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_learning_workflow.py#L431) | 8 related LearningWorkflowTests scenarios, each with private fixture state. | `helper: tests.test_learning_workflow.LearningWorkflowTests.check_test_default_splits_hold_out_about_a_fifth_per_species: self.assertEqual(default_training_splits([]), {})` | unrecorded / — |

## tests.test_local_shape_units

Current-image shape fits must keep calibrated and normalized blocks apart.

Source: [tests/test_local_shape_units.py](../../tests/test_local_shape_units.py). Category: Measurements and shape models.

### LocalShapeUnitTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_local_shape_units.py#L150) | 9 related LocalShapeUnitTests scenarios, each with private fixture state. | `helper: tests.test_local_shape_units.LocalShapeUnitTests.check_test_local_pixels_are_dimensionless_and_mm_are_physical: self.assertEqual(component.physical_dimensions_available, scale is not None)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_local_shape_units.py#L162) | 1 related LocalShapeUnitTests scenarios, each with private fixture state. | `helper: tests.test_local_shape_units.LocalShapeUnitTests.check_test_invalid_mixed_shape_covariance_is_rejected_explicitly: self.assertRaisesRegex(ValueError, 'symmetric and positive semidefinite')` | unrecorded / — |

## tests.test_manual_seed_centre_persistence

Manual seed centre sidecar persistence and source binding

Source: [tests/test_manual_seed_centre_persistence.py](../../tests/test_manual_seed_centre_persistence.py). Category: Annotation and interactive image review.

### ManualSeedCentreStoreTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_manual_seed_centre_persistence.py#L181) | 3 related ManualSeedCentreStoreTests scenarios, each with private fixture state. | `helper: tests.test_manual_seed_centre_persistence.ManualSeedCentreStoreTests.check_test_round_trip_preserves_exact_source_floats_mode_and_empty_replace: self.assertIsNotNone(loaded)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_manual_seed_centre_persistence.py#L187) | 2 related ManualSeedCentreStoreTests scenarios, each with private fixture state. | `helper: tests.test_manual_seed_centre_persistence.ManualSeedCentreStoreTests.check_test_changed_source_bytes_are_rejected_without_mutating_sidecar: self.assertRaises(ManualSeedCentreFingerprintMismatch)` | unrecorded / — |

## tests.test_manual_seed_centre_ui

Manual centre drawing, selection and desktop persistence

Source: [tests/test_manual_seed_centre_ui.py](../../tests/test_manual_seed_centre_ui.py). Category: Annotation and interactive image review.

### ManualSeedCentreUiTests

Fixtures: `setUpClass` at line 19.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_manual_seed_centre_ui.py#L406) | 2 related ManualSeedCentreUiTests scenarios, each with private fixture state. | `helper: tests.test_manual_seed_centre_ui.ManualSeedCentreUiTests.check_test_qt_add_manual_drag_auto_drag_and_right_delete_use_marker_semantics: self.assertTrue(succeeded, error)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_manual_seed_centre_ui.py#L411) | 3 related ManualSeedCentreUiTests scenarios, each with private fixture state. | `helper: tests.test_manual_seed_centre_ui.ManualSeedCentreUiTests.check_test_completed_edit_inverse_projects_autosaves_and_invalidates_only_dependents: self.assertIsNotNone(key)` | unrecorded / — |

## tests.test_material_evidence

CUDA material likelihoods, references and evidence decisions

Source: [tests/test_material_evidence.py](../../tests/test_material_evidence.py). Category: Image analysis, calibration and rendering.

### MaterialEvidenceTests

Fixtures: `setUp` at line 14.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_material_evidence.py#L195) | 7 related MaterialEvidenceTests scenarios, each with private fixture state. | `helper: tests.test_material_evidence.MaterialEvidenceTests.check_test_four_decision_masses_sum_to_one: np.testing.assert_allclose(total, 255, atol=2)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_material_evidence.py#L205) | 1 related MaterialEvidenceTests scenarios, each with private fixture state. | `helper: tests.test_material_evidence.MaterialEvidenceTests.check_test_invalid_pixels_have_no_material_mass: self.assertFalse(np.any(np.asarray(raster)[:3]))` | unrecorded / — |

## tests.test_measurement_description_ui

Reviewed overlay facts use the same measurement records as their values.

Source: [tests/test_measurement_description_ui.py](../../tests/test_measurement_description_ui.py). Category: Measurements and shape models.

### MeasurementDescriptionUITests

Fixtures: `setUpClass` at line 24, `setUp` at line 27, `tearDown` at line 41.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_measurement_description_ui.py#L172) | 5 related MeasurementDescriptionUITests scenarios, each with private fixture state. | `helper: tests.test_measurement_description_ui.MeasurementDescriptionUITests.check_test_body_span_and_point_tooltips_bind_definitions_units_and_frame_without_refitting: self.assertIn('body length:', tooltips)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_measurement_description_ui.py#L180) | 2 related MeasurementDescriptionUITests scenarios, each with private fixture state. | `helper: tests.test_measurement_description_ui.MeasurementDescriptionUITests.check_test_unavailable_legacy_and_boundary_errors_omit_bars_but_valid_zero_is_available: self.assertIn('Definition: unknown', tooltips)` | unrecorded / — |

## tests.test_measurement_descriptions

Value descriptions preserve calculation facts without inventing lineage.

Source: [tests/test_measurement_descriptions.py](../../tests/test_measurement_descriptions.py). Category: Measurements and shape models.

### MeasurementDescriptionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_measurement_descriptions.py#L166) | 7 related MeasurementDescriptionTests scenarios, each with private fixture state. | `helper: tests.test_measurement_descriptions.MeasurementDescriptionTests.check_test_vector_descriptions_bind_current_values_definitions_and_uncertainties: self.assertEqual(tuple((r.key for r in records[:19])), SHAPE_MEASUREMENT_NAMES)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_measurement_descriptions.py#L176) | 3 related MeasurementDescriptionTests scenarios, each with private fixture state. | `helper: tests.test_measurement_descriptions.MeasurementDescriptionTests.check_test_no_boundary_alternatives_withhold_sensitivity_instead_of_claiming_zero: self.assertEqual(item.provenance.boundary_alternative_count, 0)` | unrecorded / — |

## tests.test_measurement_scale_safety

Numeric scale availability is separate from biological/physical validity.

Source: [tests/test_measurement_scale_safety.py](../../tests/test_measurement_scale_safety.py). Category: Measurements and shape models.

### MeasurementScaleSafetyTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_measurement_scale_safety.py#L197) | 4 related MeasurementScaleSafetyTests scenarios, each with private fixture state. | `helper: tests.test_measurement_scale_safety.MeasurementScaleSafetyTests.check_test_invalid_scales_reject_before_raster_measurement: self.assertRaisesRegex(ValueError, 'finite positive real')` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_measurement_scale_safety.py#L204) | 8 related MeasurementScaleSafetyTests scenarios, each with private fixture state. | `helper: tests.test_measurement_scale_safety.MeasurementScaleSafetyTests.check_test_ordinary_shape_conversion_preserves_exact_dimensional_arithmetic: self.assertEqual(mm.maximum_span, px.maximum_span / 2.5)` | unrecorded / — |

## tests.test_measurements

Seed measurement geometry

Source: [tests/test_measurements.py](../../tests/test_measurements.py). Category: Measurements and shape models.

### MeasurementTests

Fixtures: `setUpClass` at line 9.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_circle_measurements_are_calibrated`](../../tests/test_measurements.py#L16) | circle measurements are calibrated | `self.assertAlmostEqual(result.maximum_feret, 10.0, delta=0.15); self.assertAlmostEqual(result.minimum_feret, 10.0, delta=0.15)` | unrecorded / — |
| [`test_empty_mask_is_rejected`](../../tests/test_measurements.py#L33) | empty mask is rejected | `self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_neutral_calibration

Selected grid references, CUDA sampling, cache boundaries and saved facts.

Source: [tests/test_neutral_calibration.py](../../tests/test_neutral_calibration.py). Category: Image analysis, calibration and rendering.

### NeutralCalibrationTests

Fixtures: `setUp` at line 40.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_neutral_calibration.py#L224) | 8 related NeutralCalibrationTests scenarios, each with private fixture state. | `helper: tests.test_neutral_calibration.NeutralCalibrationTests.check_test_default_source_mode_preserves_legacy_gain_and_pixel_arithmetic: np.testing.assert_array_equal(result.channel_gains_bgr, tuple((float(x) for x in gains)))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_neutral_calibration.py#L235) | 7 related NeutralCalibrationTests scenarios, each with private fixture state. | `helper: tests.test_neutral_calibration.NeutralCalibrationTests.check_test_missing_selection_has_no_partial_or_automatic_fallback: self.assertEqual(result.neutral_balance.state, 'unavailable')` | unrecorded / — |
### NeutralSettingsTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_neutral_calibration.py#L323) | 5 related NeutralSettingsTests scenarios, each with private fixture state. | `helper: tests.test_neutral_calibration.NeutralSettingsTests.check_test_selection_parser_is_shared_canonical_and_strict: self.assertEqual(normalize_neutral_patch_selection(raw), expected)` | unrecorded / — |

## tests.test_node_optimization

Node parameter search, optimizer state and cache reuse

Source: [tests/test_node_optimization.py](../../tests/test_node_optimization.py). Category: Pipeline workspace and controls.

### OptimizationContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_node_optimization.py#L210) | 7 related OptimizationContractTests scenarios, each with private fixture state. | `helper: tests.test_node_optimization.OptimizationContractTests.check_test_every_card_and_control_has_a_disposition: self.assertEqual(set(nodes), set(OBJECTIVES) \| set(EXEMPT))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_node_optimization.py#L221) | 3 related OptimizationContractTests scenarios, each with private fixture state. | `helper: tests.test_node_optimization.OptimizationContractTests.check_test_edge_controls_invalidate_true_prototype_owner: self.assertIn('reference_texture_prototypes', affected)` | unrecorded / — |
### OptimizationReferenceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_node_optimization.py#L376) | 10 related OptimizationReferenceTests scenarios, each with private fixture state. | `helper: tests.test_node_optimization.OptimizationReferenceTests.check_test_only_complete_reviewed_connected_shapes_are_eligible: self.assertEqual(set(np.unique(result)), {0, 1})` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_node_optimization.py#L389) | 1 related OptimizationReferenceTests scenarios, each with private fixture state. | `helper: tests.test_node_optimization.OptimizationReferenceTests.check_test_failed_candidate_cache_is_discarded: self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_node_optimization_production

Production integration on two generated images; no accuracy claims.

Source: [tests/test_node_optimization_production.py](../../tests/test_node_optimization_production.py). Category: Pipeline workspace and controls.

### ProductionOptimizationTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_ui_samples_fit_local_references_without_a_library_pin`](../../tests/test_node_optimization_production.py#L20) | One bounded real UI -> frozen samples -> production fit witness. | `self.assertTrue(cv2.imwrite(str(path), image)); self.assertEqual(len(samples), 2)` | unrecorded / — |

## tests.test_node_optimization_ui

Node optimizer controls and desktop result application

Source: [tests/test_node_optimization_ui.py](../../tests/test_node_optimization_ui.py). Category: Pipeline workspace and controls.

### NodeOptimizationUITests

Fixtures: `setUpClass` at line 21.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_node_optimization_ui.py#L261) | 6 related NodeOptimizationUITests scenarios, each with private fixture state. | `helper: tests.test_node_optimization_ui.NodeOptimizationUITests.check_test_all_computational_cards_offer_shared_action: self.assertEqual(not inspector.optimization_button.isHidden(), bool(capability(node).searchable))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_node_optimization_ui.py#L270) | 2 related NodeOptimizationUITests scenarios, each with private fixture state. | `helper: tests.test_node_optimization_ui.NodeOptimizationUITests.check_test_completed_progress_does_not_cancel_success_and_apply_is_atomic: self.assertFalse(task.stop.is_set())` | unrecorded / — |

## tests.test_node_vocabulary

The engine's node vocabulary must stay bound to the pipeline graph.

Source: [tests/test_node_vocabulary.py](../../tests/test_node_vocabulary.py). Category: Image analysis, calibration and rendering.

### NodeVocabularyTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_engine_literals_are_graph_nodes_or_declared_aliases`](../../tests/test_node_vocabulary.py#L68) | engine literals are graph nodes or declared aliases | `self.assertGreater(len(literals), 20, 'the engine scan found suspiciously few names'); self.assertEqual(unknown, {}, 'baseline.py names stages that are neither graph nodes nor declared in seedvision.pipeline.node_aliases')` | unrecorded / — |
| [`test_alias_targets_and_markers_are_consistent`](../../tests/test_node_vocabulary.py#L82) | alias targets and markers are consistent | `self.assertTrue(set(ENGINE_NODE_ALIASES.values()) <= node_ids); self.assertFalse(set(ENGINE_NODE_ALIASES) & node_ids, 'an alias shadows a node id')` | unrecorded / — |
| [`test_window_overlay_owners_are_graph_nodes`](../../tests/test_node_vocabulary.py#L89) | window overlay owners are graph nodes | `self.assertEqual(unknown, [])` | unrecorded / — |

## tests.test_oriented_edge_components

Oriented edge components, connectivity and CUDA outputs

Source: [tests/test_oriented_edge_components.py](../../tests/test_oriented_edge_components.py). Category: Image analysis, calibration and rendering.

### OrientedEdgeComponentTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_oriented_edge_components.py#L264) | 7 related OrientedEdgeComponentTests scenarios, each with private fixture state. | `helper: tests.test_oriented_edge_components.OrientedEdgeComponentTests.check_test_gap_four_reconnects_facing_fragments_without_gap_two: self.assertNotEqual(int(short[4, 2]), int(short[4, 8]))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_oriented_edge_components.py#L274) | 3 related OrientedEdgeComponentTests scenarios, each with private fixture state. | `helper: tests.test_oriented_edge_components.OrientedEdgeComponentTests.check_test_curvature_policy_rejects_s_bridge_but_off_preserves_it: self.assertEqual(int(legacy[4, 2]), int(legacy[5, 8]))` | unrecorded / — |

## tests.test_physical_shape_fit_availability

Finite calibrated evidence may exceed the physical fitter's numeric domain.

Source: [tests/test_physical_shape_fit_availability.py](../../tests/test_physical_shape_fit_availability.py). Category: Measurements and shape models.

### PhysicalShapeFitAvailabilityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_physical_shape_fit_availability.py#L247) | 13 related PhysicalShapeFitAvailabilityTests scenarios, each with private fixture state. | `helper: tests.test_physical_shape_fit_availability.PhysicalShapeFitAvailabilityTests.check_test_real_calibrated_area_covariance_overflow_retains_all_dimensionless_groups: self.assertTrue(all((np.isfinite(m.vector()).all() for m in values)))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_physical_shape_fit_availability.py#L263) | 3 related PhysicalShapeFitAvailabilityTests scenarios, each with private fixture state. | `helper: tests.test_physical_shape_fit_availability.PhysicalShapeFitAvailabilityTests.check_test_parent_predictive_sum_overflow_has_dedicated_numeric_failure: self.assertRaises(model._PhysicalFitNumericUnavailable)` | unrecorded / — |

## tests.test_pilot_analysis

Committed pilot-image segmentation, calibration and end-to-end analysis

Source: [tests/test_pilot_analysis.py](../../tests/test_pilot_analysis.py). Category: Image analysis, calibration and rendering.

### PilotAnalysisTests

Fixtures: `setUpClass` at line 15.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_sparse_pilot_starts_from_automatic_material_references`](../../tests/test_pilot_analysis.py#L891) | sparse pilot starts from automatic material references | `helper: tests.test_pilot_analysis.PilotAnalysisTests.check_test_sparse_pilot_starts_from_automatic_material_references: self.assertGreater(result.count, 0)` | unrecorded / — |
| [`test_dense_lupin_starts_from_reference_seeds_and_paint_refines_it`](../../tests/test_pilot_analysis.py#L894) | dense lupin starts from reference seeds and paint refines it | `helper: tests.test_pilot_analysis.PilotAnalysisTests.check_test_dense_lupin_starts_from_reference_seeds_and_paint_refines_it: self.assertIsNotNone(automatic.perimeter_background_lab)` | unrecorded / — |
| [`test_node_cache_reuses_upstream_stages_for_edge_change`](../../tests/test_pilot_analysis.py#L897) | node cache reuses upstream stages for edge change | `helper: tests.test_pilot_analysis.PilotAnalysisTests.check_test_node_cache_reuses_upstream_stages_for_edge_change: self.assertIs(second.calibration, calibration)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_pilot_analysis.py#L900) | 12 related PilotAnalysisTests scenarios, each with private fixture state. | `helper: tests.test_pilot_analysis.PilotAnalysisTests.check_test_annotated_seed_scale_uses_all_explicitly_complete_masks: self.assertEqual(len(selected), 8)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_pilot_analysis.py#L915) | 1 related PilotAnalysisTests scenarios, each with private fixture state. | `helper: tests.test_pilot_analysis.PilotAnalysisTests.check_test_perimeter_background_band_rejects_colour_card_contamination: self.assertIsNotNone(band.accepted_sample_mask)` | unrecorded / — |

## tests.test_pipeline

Pipeline graph topology, controls, settings, state and invalidation

Source: [tests/test_pipeline.py](../../tests/test_pipeline.py). Category: Pipeline workspace and controls.

### PipelineModelTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_pipeline.py#L1384) | 30 related PipelineModelTests scenarios, each with private fixture state. | `helper: tests.test_pipeline.PipelineModelTests.check_test_foreground_requires_authored_evidence_and_has_no_refinement_controls: self.assertFalse(any(('automatic' in key for key in foreground_keys)))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_pipeline.py#L1417) | 7 related PipelineModelTests scenarios, each with private fixture state. | `helper: tests.test_pipeline.PipelineModelTests.check_test_reconnection_rejects_unimplemented_typed_substitutions: self.assertRaisesRegex(ValueError, 'different data types\|requires\|not an authored calculation')` | unrecorded / — |
### NodeTimingRecorderTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_cpu_and_explicit_spans_are_aggregated_by_node`](../../tests/test_pipeline.py#L1429) | cpu and explicit spans are aggregated by node | `self.assertEqual(result['decode'], 0.125); self.assertGreater(result['analysis'], 0.0)` | unrecorded / — |
| [`test_cancellation_stops_at_the_next_timed_node_boundary`](../../tests/test_pipeline.py#L1456) | cancellation stops at the next timed node boundary | `self.assertRaises(AnalysisCancelled); self.assertRaises(AnalysisCancelled)` | unrecorded / — |
| [`test_cancellation_stops_before_the_next_torch_call_inside_a_node`](../../tests/test_pipeline.py#L1470) | cancellation stops before the next torch call inside a node | `self.assertEqual(float(torch.sum(values)), 3.0); self.assertRaises(AnalysisCancelled)` | unrecorded / — |
| [`test_cancellation_checks_stay_on_the_analysis_thread`](../../tests/test_pipeline.py#L1494) | cancellation checks stay on the analysis thread | `self.assertEqual(results, [2.0])` | unrecorded / — |
| [`test_analysis_without_a_cancellation_callback_has_no_checks`](../../tests/test_pipeline.py#L1511) | analysis without a cancellation callback has no checks | `self.assertIsNone(_get_current_function_mode()); self.assertEqual(analysis(1), (1, None, None))` | unrecorded / — |
| [`test_stop_ends_a_real_analysis_inside_the_running_node`](../../tests/test_pipeline.py#L1529) | stop ends a real analysis inside the running node | `self.assertRaises(AnalysisCancelled); self.assertIn(('edge_gradients', 'started'), events)` | unrecorded / — |
### BaselineSettingsTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_out_of_range_setting_is_rejected`](../../tests/test_pipeline.py#L1870) | out of range setting is rejected | `helper: tests.test_pipeline.BaselineSettingsTests.check_test_out_of_range_setting_is_rejected: self.assertRaises(ValueError)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_pipeline.py#L1873) | 6 related BaselineSettingsTests scenarios, each with private fixture state. | `helper: tests.test_pipeline.BaselineSettingsTests.check_test_composite_dish_and_layer_settings_are_validated: self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_pipeline_cable_routing

Pipeline cable geometry and routing

Source: [tests/test_pipeline_cable_routing.py](../../tests/test_pipeline_cable_routing.py). Category: Pipeline workspace and controls.

### PipelineCableRoutingTests

Fixtures: `setUpClass` at line 147.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_cable_routing.py#L515) | 8 related PipelineCableRoutingTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_cable_routing.PipelineCableRoutingTests.check_test_default_mode_preserves_the_established_cubic_paths: self.assertFalse(canvas.cable_bundling_enabled)` | unrecorded / — |

## tests.test_pipeline_documents

Standalone graph/layout transport, factory protection and frozen file loads.

Source: [tests/test_pipeline_documents.py](../../tests/test_pipeline_documents.py). Category: Pipeline workspace and controls.

### PipelineDocumentTests

Fixtures: `setUpClass` at line 32.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_documents.py#L170) | 1 related PipelineDocumentTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_documents.PipelineDocumentTests.check_test_full_roundtrip_preserves_settings_instances_and_layout: self.assertEqual(reopened.to_dict(), document.to_dict())` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_pipeline_documents.py#L174) | 2 related PipelineDocumentTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_documents.PipelineDocumentTests.check_test_invalid_files_atomic_failure_and_factory_names_cannot_replace_existing_bytes: self.assertRaises(PipelineDocumentError)` | unrecorded / — |

## tests.test_pipeline_inspector_actions

Pipeline inspector parameter editing and actions

Source: [tests/test_pipeline_inspector_actions.py](../../tests/test_pipeline_inspector_actions.py). Category: Pipeline workspace and controls.

### PipelineInspectorActionTests

Fixtures: `setUpClass` at line 13.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_inspector_actions.py#L171) | 4 related PipelineInspectorActionTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_inspector_actions.PipelineInspectorActionTests.check_test_procedural_fit_action_is_contextual_and_emits_loss_weights: self.assertTrue(inspector.procedural_fit_container.isHidden())` | unrecorded / — |

## tests.test_pipeline_ui

Pipeline desktop controls, graph state, navigation, workers and status

Source: [tests/test_pipeline_ui.py](../../tests/test_pipeline_ui.py). Category: Pipeline workspace and controls.

### PipelineCanvasTests

Fixtures: `setUpClass` at line 23.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_approved_reference_edge_fit_updates_live_node_parameters`](../../tests/test_pipeline_ui.py#L4572) | approved reference edge fit updates live node parameters | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_approved_reference_edge_fit_updates_live_node_parameters: self.assertAlmostEqual(node.parameters['reference_edge_similarity_scale'], 1.3)` | unrecorded / — |
| [`test_switching_images_cancels_all_previous_image_work_before_load`](../../tests/test_pipeline_ui.py#L4575) | switching images cancels all previous image work before load | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_switching_images_cancels_all_previous_image_work_before_load: self.assertTrue(analysis_task.cancelled)` | unrecorded / — |
| [`test_cancelled_previous_image_completion_cannot_install_stale_result`](../../tests/test_pipeline_ui.py#L4578) | cancelled previous image completion cannot install stale result | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_cancelled_previous_image_completion_cannot_install_stale_result: mark_complete.assert_not_called()` | unrecorded / — |
| [`test_failed_analysis_purges_its_partial_cache`](../../tests/test_pipeline_ui.py#L4581) | failed analysis purges its partial cache | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_failed_analysis_purges_its_partial_cache: self.assertNotIn(key, window._analysis_caches)` | unrecorded / — |
| [`test_superseded_analysis_is_cancelled_and_remains_visible`](../../tests/test_pipeline_ui.py#L4584) | superseded analysis is cancelled and remains visible | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_superseded_analysis_is_cancelled_and_remains_visible: self.assertTrue(task.cancelled)` | unrecorded / — |
| [`test_worker_progress_turns_only_the_current_revision_node_green`](../../tests/test_pipeline_ui.py#L4587) | worker progress turns only the current revision node green | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_worker_progress_turns_only_the_current_revision_node_green: self.assertTrue(succeeded, error)` | unrecorded / — |
| [`test_cleared_reference_layer_returns_to_paint_and_accepts_dabs`](../../tests/test_pipeline_ui.py#L4590) | cleared reference layer returns to paint and accepts dabs | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_cleared_reference_layer_returns_to_paint_and_accepts_dabs: self.assertTrue(window.image_view._reference_erase_enabled)` | unrecorded / — |
| [`test_perimeter_source_checkbox_invalidates_every_background_consumer`](../../tests/test_pipeline_ui.py#L4593) | perimeter source checkbox invalidates every background consumer | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_perimeter_source_checkbox_invalidates_every_background_consumer: self.assertTrue(window.keep_perimeter_background_reference_checkbox.isChecked())` | unrecorded / — |
| [`test_annotated_seed_foreground_checkbox_defaults_on_and_invalidates_consumers`](../../tests/test_pipeline_ui.py#L4596) | annotated seed foreground checkbox defaults on and invalidates consumers | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_annotated_seed_foreground_checkbox_defaults_on_and_invalidates_consumers: self.assertTrue(checkbox.isChecked())` | unrecorded / — |
| [`test_new_seed_preserves_the_selected_instance_tool`](../../tests/test_pipeline_ui.py#L4599) | new seed preserves the selected instance tool | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_new_seed_preserves_the_selected_instance_tool: self.assertTrue(image.save(str(image_path)))` | unrecorded / — |
| [`test_reference_masks_run_analysis_only_after_explicit_apply`](../../tests/test_pipeline_ui.py#L4602) | reference masks run analysis only after explicit apply | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_reference_masks_run_analysis_only_after_explicit_apply: self.assertEqual(analyses, [])` | unrecorded / — |
| [`test_instance_annotations_apply_separately_from_reference_masks`](../../tests/test_pipeline_ui.py#L4605) | instance annotations apply separately from reference masks | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_instance_annotations_apply_separately_from_reference_masks: self.assertIn(key, window._instance_annotations_dirty)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_ui.py#L4608) | 58 related PipelineCanvasTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_canvas_contains_default_nodes_and_live_edges: self.assertGreater(canvas.horizontalScrollBar().maximum(), 0)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_pipeline_ui.py#L4672) | 1 related PipelineCanvasTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_setting_change_invalidates_only_node_and_graph_dependents: self.assertEqual(window.pipeline.node('edge_gradients').status, NodeStatus.IDLE)` | unrecorded / — |
| [`test_calibration_nodes_render_references_and_metric_scale`](../../tests/test_pipeline_ui.py#L5103) | calibration nodes render references and metric scale | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_calibration_nodes_render_references_and_metric_scale: self.assertTrue(succeeded, error)` | unrecorded / — |

## tests.test_pipeline_workflow_ui

Main-workspace routing between the default pipelines and the shared project nodes.

Source: [tests/test_pipeline_workflow_ui.py](../../tests/test_pipeline_workflow_ui.py). Category: Pipeline workspace and controls.

### PipelineWorkflowUiTests

Fixtures: `setUpClass` at line 25, `setUp` at line 28, `tearDown` at line 35.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_workflow_ui.py#L204) | 5 related PipelineWorkflowUiTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_workflow_ui.PipelineWorkflowUiTests.check_test_shared_project_nodes_and_two_way_image_selection: self.assertIs(self.controller.graph_pipeline, pipeline)` | unrecorded / — |

## tests.test_pipeline_workspace_files

Desktop default-pipeline lock, named copies and complete graph/layout file integration.

Source: [tests/test_pipeline_workspace_files.py](../../tests/test_pipeline_workspace_files.py). Category: Pipeline workspace and controls.

### PipelineWorkspaceFileTests

Fixtures: `setUpClass` at line 26, `setUp` at line 29, `tearDown` at line 36.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_workspace_files.py#L179) | 3 related PipelineWorkspaceFileTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_workspace_files.PipelineWorkspaceFileTests.check_test_lock_guards_edits_but_allows_layout_and_run_to_node: self.assertTrue(self.files.locked)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_pipeline_workspace_files.py#L185) | 1 related PipelineWorkspaceFileTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_workspace_files.PipelineWorkspaceFileTests.check_test_refused_pipeline_and_project_files_leave_the_working_graph: self.assertFalse(self.files.load_pipeline(bad))` | unrecorded / — |

## tests.test_polynomial_correction_integrity

Whole-call input and retained-matrix integrity for polynomial correction.

Source: [tests/test_polynomial_correction_integrity.py](../../tests/test_polynomial_correction_integrity.py). Category: Image analysis, calibration and rendering.

### PolynomialCorrectionIntegrityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_polynomial_correction_integrity.py#L175) | 4 related PolynomialCorrectionIntegrityTests scenarios, each with private fixture state. | `helper: tests.test_polynomial_correction_integrity.PolynomialCorrectionIntegrityTests.check_test_target_inputs_are_guarded_during_source_extraction: self.assertRaises(correction.CorrectionMutation)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_polynomial_correction_integrity.py#L182) | 3 related PolynomialCorrectionIntegrityTests scenarios, each with private fixture state. | `helper: tests.test_polynomial_correction_integrity.PolynomialCorrectionIntegrityTests.check_test_cross_phase_mutation_takes_precedence_over_callback_exception: self.assertRaises(correction.CorrectionMutation)` | unrecorded / — |

## tests.test_portable_result_identity

Actual relocated native results, artifact facts and private review history.

Source: [tests/test_portable_result_identity.py](../../tests/test_portable_result_identity.py). Category: Persistence, export, identity and outcomes.

### ArtifactIdentityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_portable_result_identity.py#L138) | 3 related ArtifactIdentityTests scenarios, each with private fixture state. | `helper: tests.test_portable_result_identity.ArtifactIdentityTests.check_test_equivalent_verified_locations_keep_paths_but_share_content_identity: self.assertEqual(reports[0]['identity'], reports[1]['identity'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_portable_result_identity.py#L144) | 2 related ArtifactIdentityTests scenarios, each with private fixture state. | `helper: tests.test_portable_result_identity.ArtifactIdentityTests.check_test_changed_disappeared_and_newly_available_selections_reject_stale_capture: self.assertRaisesRegex(ValueError, 'changed after analysis')` | unrecorded / — |
### PrivateReviewLookupTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_only_absent_primary_uses_project_private_review_without_writing_shared_history`](../../tests/test_portable_result_identity.py#L151) | only absent primary uses project private review without writing shared history | `self.assertEqual(lookup_result_review(root, revision, project_path=project)[1], 'matched'); self.assertFalse((root / 'result-reviews').exists())` | unrecorded / — |

## tests.test_preflight_outcomes

Incomplete request evidence through real batch and Qt preparation boundaries.

Source: [tests/test_preflight_outcomes.py](../../tests/test_preflight_outcomes.py). Category: Persistence, export, identity and outcomes.

### PreflightOutcomeTests

Fixtures: `setUp` at line 25.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_preflight_outcomes.py#L198) | 8 related PreflightOutcomeTests scenarios, each with private fixture state. | `helper: tests.test_preflight_outcomes.PreflightOutcomeTests.check_test_invalid_settings_and_duplicate_positional_arguments_never_execute_analysis: runner.assert_not_called()` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_preflight_outcomes.py#L209) | 2 related PreflightOutcomeTests scenarios, each with private fixture state. | `helper: tests.test_preflight_outcomes.PreflightOutcomeTests.check_test_capture_changes_preserve_actual_before_and_after_without_inventing_request: self.assertEqual(facts['subject_at_request']['sha256'], initial)` | unrecorded / — |
### NativePreflightOutcomeTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_project_subject_and_stale_pretask_attempt_preserve_history_without_image_cleanup`](../../tests/test_preflight_outcomes.py#L226) | project subject and stale pretask attempt preserve history without image cleanup | `cleanup.assert_not_called(); self.assertIs(self.window._active_tasks[self.key], replacement)` | unrecorded / — |

## tests.test_procedural_fit

Procedural seed fitting, search behavior and geometric evidence

Source: [tests/test_procedural_fit.py](../../tests/test_procedural_fit.py). Category: Image analysis, calibration and rendering.

### ProceduralFitScoreTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_procedural_fit.py#L122) | 5 related ProceduralFitScoreTests scenarios, each with private fixture state. | `helper: tests.test_procedural_fit.ProceduralFitScoreTests.check_test_perfect_relabelled_prediction_has_zero_loss: self.assertTrue(score.perfect_fit)` | unrecorded / — |
### ProceduralFitDistanceWeightedOverreachTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_procedural_fit.py#L376) | 9 related ProceduralFitDistanceWeightedOverreachTests scenarios, each with private fixture state. | `helper: tests.test_procedural_fit.ProceduralFitDistanceWeightedOverreachTests.check_test_equal_raw_overreach_costs_more_when_farther_from_target: self.assertEqual(adjacent.false_positive_pixels, 1)` | unrecorded / — |
### ProceduralSettingsFitTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_procedural_fit.py#L485) | 3 related ProceduralSettingsFitTests scenarios, each with private fixture state. | `helper: tests.test_procedural_fit.ProceduralSettingsFitTests.check_test_annotation_distance_context_is_prepared_once_per_fit: self.assertEqual(result.evaluations, 4)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_procedural_fit.py#L491) | 1 related ProceduralSettingsFitTests scenarios, each with private fixture state. | `helper: tests.test_procedural_fit.ProceduralSettingsFitTests.check_test_fit_is_deterministic_and_honours_evaluation_budget: self.assertLessEqual(first.evaluations, 5)` | unrecorded / — |
### ProceduralReferenceAssignmentTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_procedural_fit.py#L613) | 7 related ProceduralReferenceAssignmentTests scenarios, each with private fixture state. | `helper: tests.test_procedural_fit.ProceduralReferenceAssignmentTests.check_test_global_assignment_recovers_pair_stolen_by_greedy_overlap: self.assertEqual(set(comparison.matched_pairs), {(3, 44), (7, 91)})` | unrecorded / — |

## tests.test_procedural_fit_ui

Procedural fit controls and result application

Source: [tests/test_procedural_fit_ui.py](../../tests/test_procedural_fit_ui.py). Category: Image analysis, calibration and rendering.

### ProceduralFitMainWindowTests

Fixtures: `setUpClass` at line 19.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_fit_task_passes_complete_coverage_to_core_options`](../../tests/test_procedural_fit_ui.py#L128) | fit task passes complete coverage to core options | `self.assertTrue(options.annotations_are_complete); self.assertEqual(options.false_positive_weight, 2.5)` | unrecorded / — |
| [`test_completed_whole_dish_fit_labels_reported_metrics`](../../tests/test_procedural_fit_ui.py#L220) | completed whole dish fit labels reported metrics | `self.assertIn('Whole-dish metrics', window.pipeline_inspector.procedural_fit_status_label.text())` | unrecorded / — |

## tests.test_procedural_instances

CUDA procedural instance splitting, ownership and analysis integration

Source: [tests/test_procedural_instances.py](../../tests/test_procedural_instances.py). Category: Image analysis, calibration and rendering.

### ProceduralInstanceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_procedural_instances.py#L1390) | 22 related ProceduralInstanceTests scenarios, each with private fixture state. | `helper: tests.test_procedural_instances.ProceduralInstanceTests.check_test_reference_cost_settings_never_change_automatic_predictions: self.assertGreater(first.count, 0)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_procedural_instances.py#L1415) | 11 related ProceduralInstanceTests scenarios, each with private fixture state. | `helper: tests.test_procedural_instances.ProceduralInstanceTests.check_test_generated_frankenstein_region_is_hard_rejected_by_axis_ratio: self.assertEqual(result.count, 0)` | unrecorded / — |

## tests.test_project_analysis_persistence

Project analysis settings, image locators and atomic persistence

Source: [tests/test_project_analysis_persistence.py](../../tests/test_project_analysis_persistence.py). Category: Persistence, export, identity and outcomes.

### ProjectAnalysisPersistenceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_posix_project_can_store_source_locators_that_differ_only_by_case`](../../tests/test_project_analysis_persistence.py#L729) | posix project can store source locators that differ only by case | `helper: tests.test_project_analysis_persistence.ProjectAnalysisPersistenceTests.check_test_posix_project_can_store_source_locators_that_differ_only_by_case: self.assertEqual(loaded.issues, ())` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_project_analysis_persistence.py#L732) | 7 related ProjectAnalysisPersistenceTests scenarios, each with private fixture state. | `helper: tests.test_project_analysis_persistence.ProjectAnalysisPersistenceTests.check_test_round_trip_is_compact_ordered_and_resolves_relative_and_external_images: self.assertTrue(destination.name.endswith(PROJECT_ANALYSIS_EXTENSION))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_project_analysis_persistence.py#L742) | 8 related ProjectAnalysisPersistenceTests scenarios, each with private fixture state. | `helper: tests.test_project_analysis_persistence.ProjectAnalysisPersistenceTests.check_test_missing_and_changed_files_are_structured_issues_not_schema_failures: self.assertIn(('source_image', ProjectFileStatus.MISSING), issue_pairs)` | unrecorded / — |

## tests.test_project_analysis_ui

Project analysis save/reopen, dirty state, selections and desktop guards

Source: [tests/test_project_analysis_ui.py](../../tests/test_project_analysis_ui.py). Category: Persistence, export, identity and outcomes.

### ProjectAnalysisUiTests

Fixtures: `setUpClass` at line 22.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_project_analysis_ui.py#L2096) | 14 related ProjectAnalysisUiTests scenarios, each with private fixture state. | `helper: tests.test_project_analysis_ui.ProjectAnalysisUiTests.check_test_actions_and_profile_load_replace_only_portable_analysis_state: self.assertEqual(window.open_action.shortcut().toString(), QKeySequence(QKeySequence.StandardKey.Open).toString())` | unrecorded / — |
| [`test_project_owned_annotations`](../../tests/test_project_analysis_ui.py#L2113) | 3 related ProjectAnalysisUiTests scenarios, each with private fixture state. | `helper: tests.test_project_analysis_ui.ProjectAnalysisUiTests.check_test_each_project_keeps_its_own_annotations_until_save: self.assertTrue(window._write_project(first))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_project_analysis_ui.py#L2119) | 9 related ProjectAnalysisUiTests scenarios, each with private fixture state. | `helper: tests.test_project_analysis_ui.ProjectAnalysisUiTests.check_test_incompatible_profile_is_rejected_before_confirmation_and_is_atomic: self.assertFalse(loaded)` | unrecorded / — |

## tests.test_reference_edge_fit

Reference-driven edge fitting and rendered evidence

Source: [tests/test_reference_edge_fit.py](../../tests/test_reference_edge_fit.py). Category: Annotation and interactive image review.

### ReferenceEdgeFitTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_edge_fit.py#L248) | 5 related ReferenceEdgeFitTests scenarios, each with private fixture state. | `helper: tests.test_reference_edge_fit.ReferenceEdgeFitTests.check_test_internal_prototype_sources_are_prominent_normal_maxima: self.assertTrue(bool(torch.all(selected[:, 6]).item()))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_edge_fit.py#L256) | 2 related ReferenceEdgeFitTests scenarios, each with private fixture state. | `helper: tests.test_reference_edge_fit.ReferenceEdgeFitTests.check_test_balanced_score_rewards_correct_support_and_rejects_cross_matches: self.assertLess(good.loss, crossed.loss)` | unrecorded / — |

## tests.test_reference_edge_strip_descriptor

Reference edge strip extraction, descriptors and overlays

Source: [tests/test_reference_edge_strip_descriptor.py](../../tests/test_reference_edge_strip_descriptor.py). Category: Annotation and interactive image review.

### ReferenceEdgeStripDescriptorTests

Fixtures: `setUpClass` at line 16.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_edge_strip_descriptor.py#L760) | 12 related ReferenceEdgeStripDescriptorTests scenarios, each with private fixture state. | `helper: tests.test_reference_edge_strip_descriptor.ReferenceEdgeStripDescriptorTests.check_test_adaptive_edge_shape_resolves_small_seeds_without_exceeding_cap: self.assertAlmostEqual(scale, 0.64)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_edge_strip_descriptor.py#L775) | 2 related ReferenceEdgeStripDescriptorTests scenarios, each with private fixture state. | `helper: tests.test_reference_edge_strip_descriptor.ReferenceEdgeStripDescriptorTests.check_test_physical_training_strip_uses_annotation_axis_not_wrong_image_tangent: self.assertEqual(len(observed), 2)` | unrecorded / — |

## tests.test_reference_edge_toolbar_revision

Edge reference toolbar controls, graph revisions and image-view state

Source: [tests/test_reference_edge_toolbar_revision.py](../../tests/test_reference_edge_toolbar_revision.py). Category: Annotation and interactive image review.

### ReferenceEdgeToolbarRevisionTests

Fixtures: `setUpClass` at line 20, `tearDownClass` at line 33.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_edge_toolbar_revision.py#L243) | 6 related ReferenceEdgeToolbarRevisionTests scenarios, each with private fixture state. | `helper: tests.test_reference_edge_toolbar_revision.ReferenceEdgeToolbarRevisionTests.check_test_toolbar_toggles_and_opacity_are_display_only: self.assertEqual(window.analyze_action.text(), 'Run pipeline')` | unrecorded / — |

## tests.test_reference_graph_readonly

Read-only public reference graph presentation and edit prevention

Source: [tests/test_reference_graph_readonly.py](../../tests/test_reference_graph_readonly.py). Category: Benchmarks, added nodes and reference views.

### ReferenceGraphReadOnlyTests

Fixtures: `setUpClass` at line 13.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_graph_readonly.py#L374) | 5 related ReferenceGraphReadOnlyTests scenarios, each with private fixture state. | `helper: tests.test_reference_graph_readonly.ReferenceGraphReadOnlyTests.check_test_canvas_blocks_direct_and_real_qt_edit_paths: self.assertTrue(canvas.read_only)` | unrecorded / — |

## tests.test_reference_history

Reference history revision selection and UI presentation

Source: [tests/test_reference_history.py](../../tests/test_reference_history.py). Category: Annotation and interactive image review.

### RasterUndoHistoryTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_history.py#L113) | 4 related RasterUndoHistoryTests scenarios, each with private fixture state. | `helper: tests.test_reference_history.RasterUndoHistoryTests.check_test_grouped_reference_edit_is_restored_atomically: self.assertTrue(history.record('background stroke', 'material', before, after))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_history.py#L120) | 1 related RasterUndoHistoryTests scenarios, each with private fixture state. | `helper: tests.test_reference_history.RasterUndoHistoryTests.check_test_limit_must_retain_at_least_five_commands: self.assertRaisesRegex(ValueError, 'at least five')` | unrecorded / — |

## tests.test_reference_library

Reference library aggregation, extraction, contracts and persistence

Source: [tests/test_reference_library.py](../../tests/test_reference_library.py). Category: Measurements and shape models.

### SpeciesReferenceLibraryTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_library.py#L340) | 6 related SpeciesReferenceLibraryTests scenarios, each with private fixture state. | `helper: tests.test_reference_library.SpeciesReferenceLibraryTests.check_test_extraction_uses_production_descriptor_schemas: self.assertTrue(cv2.imwrite(str(source_path), image))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_library.py#L349) | 1 related SpeciesReferenceLibraryTests scenarios, each with private fixture state. | `helper: tests.test_reference_library.SpeciesReferenceLibraryTests.check_test_publish_load_export_import_and_tamper_detection: self.assertEqual(loaded.manifest.content_sha256, pin.sha256)` | unrecorded / — |

## tests.test_reference_persistence

Reference region sidecars, source binding and desktop lifecycle

Source: [tests/test_reference_persistence.py](../../tests/test_reference_persistence.py). Category: Persistence, export, identity and outcomes.

### ReferenceRegionStoreTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_reference_persistence.py#L282) | 5 related ReferenceRegionStoreTests scenarios, each with private fixture state. | `helper: tests.test_reference_persistence.ReferenceRegionStoreTests.check_test_invalid_shape_review_names_seed_and_required_repair: self.assertRaises(ReferenceRegionError)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_reference_persistence.py#L290) | 4 related ReferenceRegionStoreTests scenarios, each with private fixture state. | `helper: tests.test_reference_persistence.ReferenceRegionStoreTests.check_test_round_trip_preserves_material_and_instance_outputs_only: self.assertIsNotNone(loaded)` | unrecorded / — |
### ReferenceRegionMainWindowTests

Fixtures: `setUpClass` at line 301.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_persistence.py#L719) | 7 related ReferenceRegionMainWindowTests scenarios, each with private fixture state. | `helper: tests.test_reference_persistence.ReferenceRegionMainWindowTests.check_test_saved_snapshot_automatically_loads_as_immutable_applied_state: self.assertIsNotNone(key)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_persistence.py#L729) | 2 related ReferenceRegionMainWindowTests scenarios, each with private fixture state. | `helper: tests.test_reference_persistence.ReferenceRegionMainWindowTests.check_test_loose_corrected_shape_is_deferred_instead_of_rejected_against_raw_image: self.assertIn(key, window._pending_unbound_reference_bundles)` | unrecorded / — |

## tests.test_reference_source_identity

Source association is preserved separately from annotation generation.

Source: [tests/test_reference_source_identity.py](../../tests/test_reference_source_identity.py). Category: Persistence, export, identity and outcomes.

### ReferenceSourceArchiveTests

Fixtures: `setUp` at line 26.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_source_identity.py#L118) | 3 related ReferenceSourceArchiveTests scenarios, each with private fixture state. | `helper: tests.test_reference_source_identity.ReferenceSourceArchiveTests.check_test_v7_round_trip_preserves_known_generation_and_unknown_history: self.assertEqual(loaded.archive_source_sha256, self.digest)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_source_identity.py#L124) | 3 related ReferenceSourceArchiveTests scenarios, each with private fixture state. | `helper: tests.test_reference_source_identity.ReferenceSourceArchiveTests.check_test_save_rejects_bad_identity_before_rasters_or_destination_creation: self.assertRaises(ReferenceRegionError)` | unrecorded / — |
### ReferenceSourceAnalysisTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_source_identity.py#L273) | 4 related ReferenceSourceAnalysisTests scenarios, each with private fixture state. | `helper: tests.test_reference_source_identity.ReferenceSourceAnalysisTests.check_test_saved_project_batch_forwards_archive_and_generation_bindings: self.assertEqual(len(run_project(root, master, root / 'output')), 1)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_source_identity.py#L280) | 2 related ReferenceSourceAnalysisTests scenarios, each with private fixture state. | `helper: tests.test_reference_source_identity.ReferenceSourceAnalysisTests.check_test_analyze_path_rejects_mismatch_before_decoding_and_forwards_verified_facts: self.assertRaises(ValueError)` | unrecorded / — |
### ReferenceSourceDesktopTests

Fixtures: `setUpClass` at line 288.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_worker_detaches_and_forwards_reference_identity_metadata`](../../tests/test_reference_source_identity.py#L293) | worker detaches and forwards reference identity metadata | `self.assertEqual(analyzer.call_args.kwargs['reference_archive_source_sha256'], 'a' * 64); self.assertEqual(analyzer.call_args.kwargs['reference_annotation_snapshot_source_sha256'], 'b' * 64)` | unrecorded / — |

## tests.test_reference_texture_collage

Reference texture collage layout and image rendering

Source: [tests/test_reference_texture_collage.py](../../tests/test_reference_texture_collage.py). Category: Annotation and interactive image review.

### ReferenceTextureCollageTests

Fixtures: `setUpClass` at line 18.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_texture_collage.py#L157) | 3 related ReferenceTextureCollageTests scenarios, each with private fixture state. | `helper: tests.test_reference_texture_collage.ReferenceTextureCollageTests.check_test_edge_strip_markers_use_source_pixel_offsets_and_lengths: self.assertAlmostEqual(centre.x1(), 18.6)` | unrecorded / — |

## tests.test_reference_undo

Reference annotation undo/redo and independent history

Source: [tests/test_reference_undo.py](../../tests/test_reference_undo.py). Category: Annotation and interactive image review.

### ReferenceUndoUiTests

Fixtures: `setUpClass` at line 22, `setUp` at line 40.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_undo.py#L470) | 7 related ReferenceUndoUiTests scenarios, each with private fixture state. | `helper: tests.test_reference_undo.ReferenceUndoUiTests.check_test_material_exclusivity_multi_undo_returns_to_clean_baseline: self.assertFalse(self.window.reference_undo_button.isEnabled())` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_undo.py#L480) | 1 related ReferenceUndoUiTests scenarios, each with private fixture state. | `helper: tests.test_reference_undo.ReferenceUndoUiTests.check_test_retired_boundary_reference_edits_are_ignored: self.assertNotIn(key, self.window._reference_masks_dirty)` | unrecorded / — |

## tests.test_resources

CUDA resource estimation and runtime admission

Source: [tests/test_resources.py](../../tests/test_resources.py). Category: Startup, resources and diagnostics.

### ResourceOwnershipTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_counts_unique_lazy_storage_and_releases_host_mirrors`](../../tests/test_resources.py#L12) | counts unique lazy storage and releases host mirrors | `self.assertEqual(before.cpu_tensor, 64); self.assertEqual(before.numpy, 0)` | unrecorded / — |

## tests.test_result_prediction_revision

Exact prediction facts must travel with review and exported revisions.

Source: [tests/test_result_prediction_revision.py](../../tests/test_result_prediction_revision.py). Category: Persistence, export, identity and outcomes.

### ResultPredictionRevisionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_result_prediction_revision.py#L168) | 2 related ResultPredictionRevisionTests scenarios, each with private fixture state. | `helper: tests.test_result_prediction_revision.ResultPredictionRevisionTests.check_test_every_reported_prediction_fact_stales_old_review_with_unchanged_labels: self.assertEqual(result_report(analysis, review=review)['accepted_count'], 1)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_result_prediction_revision.py#L173) | 5 related ResultPredictionRevisionTests scenarios, each with private fixture state. | `helper: tests.test_result_prediction_revision.ResultPredictionRevisionTests.check_test_exact_dtype_and_values_bind_revision_but_array_layout_does_not: self.assertEqual(result_report(analysis)['result_revision'], first['result_revision'])` | unrecorded / — |
### PredictionRevisionCropTests

Fixtures: `setUpClass` at line 184.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_confidence_only_stale_crop_export_rejects_before_output`](../../tests/test_result_prediction_revision.py#L189) | confidence only stale crop export rejects before output | `self.assertRaisesRegex(ValueError, 'Result changed'); self.assertFalse(output.exists())` | unrecorded / — |
| [`test_confidence_change_during_encoding_prevents_crop_publication`](../../tests/test_result_prediction_revision.py#L202) | confidence change during encoding prevents crop publication | `self.assertRaisesRegex(ValueError, 'Result changed during'); self.assertFalse(list(Path(directory).glob('seed-crops-*')))` | unrecorded / — |

## tests.test_review_workflows

Review export, portable snapshots and desktop workflow

Source: [tests/test_review_workflows.py](../../tests/test_review_workflows.py). Category: Persistence, export, identity and outcomes.

### ReviewWorkflowTests

Fixtures: `setUpClass` at line 38.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_review_workflows.py#L118) | 5 related ReviewWorkflowTests scenarios, each with private fixture state. | `helper: tests.test_review_workflows.ReviewWorkflowTests.check_test_batch_project_does_not_require_an_optional_species_library: self.assertEqual(run_project(root, master, root / 'results'), [root / 'result'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_review_workflows.py#L126) | 1 related ReviewWorkflowTests scenarios, each with private fixture state. | `helper: tests.test_review_workflows.ReviewWorkflowTests.check_test_annotation_redo_and_byte_budget: np.testing.assert_array_equal(undo.rasters[0], before)` | unrecorded / — |

## tests.test_reviewed_measurement_lineage

Actual review snapshots, producer cache boundaries and export identity.

Source: [tests/test_reviewed_measurement_lineage.py](../../tests/test_reviewed_measurement_lineage.py). Category: Persistence, export, identity and outcomes.

### ReviewedMeasurementLineageTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reviewed_measurement_lineage.py#L133) | 5 related ReviewedMeasurementLineageTests scenarios, each with private fixture state. | `helper: tests.test_reviewed_measurement_lineage.ReviewedMeasurementLineageTests.check_test_prepared_inputs_detach_normalize_and_hash_consumed_last_wins_defaults: self.assertEqual([item.seed_id for item in prepared.annotations], [1, 2])` | unrecorded / — |
### ReviewedMeasurementExportTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reviewed_measurement_lineage.py#L231) | 4 related ReviewedMeasurementExportTests scenarios, each with private fixture state. | `helper: tests.test_reviewed_measurement_lineage.ReviewedMeasurementExportTests.check_test_actual_bundles_units_definitions_and_lineage_export_without_physical_prediction_claims: self.assertEqual(report['schema_version'], 3)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reviewed_measurement_lineage.py#L238) | 2 related ReviewedMeasurementExportTests scenarios, each with private fixture state. | `helper: tests.test_reviewed_measurement_lineage.ReviewedMeasurementExportTests.check_test_stored_known_source_image_transform_calibration_or_annotation_mismatch_rejects: self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_run_outcomes

Whole-run terminal evidence, actual Qt/batch parity and publication guards.

Source: [tests/test_run_outcomes.py](../../tests/test_run_outcomes.py). Category: Persistence, export, identity and outcomes.

### RunOutcomeTests

Fixtures: `setUp` at line 24.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_run_outcomes.py#L178) | 6 related RunOutcomeTests scenarios, each with private fixture state. | `helper: tests.test_run_outcomes.RunOutcomeTests.check_test_failed_request_is_detached_unknown_outputs_and_diagnostic_paths_do_not_hash: self.assertEqual(first.payload['identity'], second.payload['identity'])` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_run_outcomes.py#L187) | 1 related RunOutcomeTests scenarios, each with private fixture state. | `helper: tests.test_run_outcomes.RunOutcomeTests.check_test_atomic_publication_and_acknowledgment_never_create_instance_approvals: self.assertNotEqual(first, second)` | unrecorded / — |
### NativeRunOutcomeTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_batch_rejected_cancelled_and_device_failure_publish_terminal_without_companions`](../../tests/test_run_outcomes.py#L212) | batch rejected cancelled and device failure publish terminal without companions | `self.assertLogs('seedvision.export.run_outcomes', level='ERROR'); self.assertRaises(raised)` | unrecorded / — |

## tests.test_run_to_node_cache

A run to one node recomputes that node alone and leaves other results standing.

Source: [tests/test_run_to_node_cache.py](../../tests/test_run_to_node_cache.py). Category: Image analysis, calibration and rendering.

### RunToNodeCacheTests

Fixtures: `setUp` at line 31.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_layer_builder`](../../tests/test_run_to_node_cache.py#L226) | How a run to one node treats the layer products it leaves out. | `helper: tests.test_run_to_node_cache.RunToNodeCacheTests.check_retained_layers_keep_their_results_when_their_inputs_change: self.assertIsNot(cache['layer.edge_gradients'], gradients)` | unrecorded / — |
| [`test_analysis`](../../tests/test_run_to_node_cache.py#L233) | What a run to one node executes and keeps across the whole analysis. | `helper: tests.test_run_to_node_cache.RunToNodeCacheTests.check_analysis_runs_only_the_target_and_keeps_every_other_result: self.assertIn('reference_texture_prototypes', left_out)` | unrecorded / — |

## tests.test_scene_analysis_region

Photographs without a vessel are analysed around their confident seeds.

Source: [tests/test_scene_analysis_region.py](../../tests/test_scene_analysis_region.py). Category: Image analysis, calibration and rendering.

### SceneAnalysisRegionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_region`](../../tests/test_scene_analysis_region.py#L168) | 4 related SceneAnalysisRegionTests scenarios, each with private fixture state. | `helper: tests.test_scene_analysis_region.SceneAnalysisRegionTests.check_test_region_spans_the_confident_seeds_and_leaves_out_the_chart: self.assertTrue(region[y, x], (x, y))` | unrecorded / — |
| [`test_large_frames`](../../tests/test_scene_analysis_region.py#L176) | 2 related SceneAnalysisRegionTests scenarios, each with private fixture state. | `helper: tests.test_scene_analysis_region.SceneAnalysisRegionTests.check_test_distance_transform_handles_elongated_frames: self.assertLess(float(np.abs(distance[0, 0].cpu().numpy() - ndimage.distance_transform_edt(mask)).max()), 0.5)` | unrecorded / — |
| [`test_dishless_fixture`](../../tests/test_scene_analysis_region.py#L182) | dishless fixture | `helper: tests.test_scene_analysis_region.SceneAnalysisRegionTests.check_test_dishless_fixture_is_analysed_around_its_seeds: self.assertIsNotNone(cache.values['crop.scene'])` | unrecorded / — |

## tests.test_seed_crop_artifacts

Seed crop artifacts, export identity and portable review

Source: [tests/test_seed_crop_artifacts.py](../../tests/test_seed_crop_artifacts.py). Category: Persistence, export, identity and outcomes.

### SeedCropArtifactTests

Fixtures: `setUpClass` at line 47.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_seed_crop_artifacts.py#L236) | 4 related SeedCropArtifactTests scenarios, each with private fixture state. | `helper: tests.test_seed_crop_artifacts.SeedCropArtifactTests.check_test_exported_pixels_masks_coordinates_and_hashes_are_bound_to_result: self.assertEqual(payload['result_revision'], report['result_revision'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_seed_crop_artifacts.py#L243) | 5 related SeedCropArtifactTests scenarios, each with private fixture state. | `helper: tests.test_seed_crop_artifacts.SeedCropArtifactTests.check_test_changed_labels_and_source_are_rejected_before_output: self.assertRaisesRegex(ValueError, 'Result changed')` | unrecorded / — |

## tests.test_seed_crops

Focused tests for explicit CUDA per-seed crop artifacts.

Source: [tests/test_seed_crops.py](../../tests/test_seed_crops.py). Category: Image analysis, calibration and rendering.

### SeedCropTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_actual_cuda_crop_is_aligned_isolated_and_transformable`](../../tests/test_seed_crops.py#L426) | actual cuda crop is aligned isolated and transformable | `helper: tests.test_seed_crops.SeedCropTests.check_test_actual_cuda_crop_is_aligned_isolated_and_transformable: self.assertEqual(batch.state, 'complete')` | unrecorded / — |
| [`test_procedural_scale_uses_actual_per_axis_rounding`](../../tests/test_seed_crops.py#L429) | procedural scale uses actual per axis rounding | `helper: tests.test_seed_crops.SeedCropTests.check_test_procedural_scale_uses_actual_per_axis_rounding: self.assertEqual(crop.status, 'available')` | unrecorded / — |
| [`test_padding_identity_clipping_and_missing_ids_are_explicit`](../../tests/test_seed_crops.py#L432) | padding identity clipping and missing ids are explicit | `helper: tests.test_seed_crops.SeedCropTests.check_test_padding_identity_clipping_and_missing_ids_are_explicit: self.assertEqual(first.state, 'complete')` | unrecorded / — |
| [`test_cuda_tensor_must_match_corrected_cpu_frame_dimensions`](../../tests/test_seed_crops.py#L435) | cuda tensor must match corrected cpu frame dimensions | `helper: tests.test_seed_crops.SeedCropTests.check_test_cuda_tensor_must_match_corrected_cpu_frame_dimensions: self.assertEqual(batch.state, 'unavailable')` | unrecorded / — |
| [`test_cancellation_marks_unfinished_ids_and_never_mutates_inputs`](../../tests/test_seed_crops.py#L438) | cancellation marks unfinished ids and never mutates inputs | `helper: tests.test_seed_crops.SeedCropTests.check_test_cancellation_marks_unfinished_ids_and_never_mutates_inputs: self.assertEqual(batch.state, 'cancelled')` | unrecorded / — |
| [`test_large_padding_is_rejected_before_any_cuda_crop_allocation`](../../tests/test_seed_crops.py#L441) | large padding is rejected before any cuda crop allocation | `helper: tests.test_seed_crops.SeedCropTests.check_test_large_padding_is_rejected_before_any_cuda_crop_allocation: self.assertEqual(batch.state, 'failed')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_seed_crops.py#L444) | 2 related SeedCropTests scenarios, each with private fixture state. | `helper: tests.test_seed_crops.SeedCropTests.check_test_cuda_absence_is_unavailable_without_cpu_fallback: self.assertEqual(batch.state, 'unavailable')` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_seed_crops.py#L449) | 2 related SeedCropTests scenarios, each with private fixture state. | `helper: tests.test_seed_crops.SeedCropTests.check_test_result_revision_and_integer_geometry_inputs_are_strict: self.assertRaisesRegex(ValueError, 'SHA-256')` | unrecorded / — |

## tests.test_seed_shape_model

Seed shape model fitting, reference aggregation and scoring

Source: [tests/test_seed_shape_model.py](../../tests/test_seed_shape_model.py). Category: Measurements and shape models.

### SeedDimensionsShapeTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_seed_shape_model.py#L219) | 8 related SeedDimensionsShapeTests scenarios, each with private fixture state. | `helper: tests.test_seed_shape_model.SeedDimensionsShapeTests.check_test_synthetic_ellipse_is_rotation_invariant: self.assertAlmostEqual(first.maximum_span, rotated.maximum_span, delta=2.5)` | unrecorded / — |

## tests.test_selected_instance_view

Selected instance overlay and desktop view state

Source: [tests/test_selected_instance_view.py](../../tests/test_selected_instance_view.py). Category: Annotation and interactive image review.

### SelectedInstanceImageViewTests

Fixtures: `setUpClass` at line 18, `setUp` at line 26.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_selected_instance_view.py#L173) | 4 related SelectedInstanceImageViewTests scenarios, each with private fixture state. | `helper: tests.test_selected_instance_view.SelectedInstanceImageViewTests.check_test_selected_only_overlay_is_cropped_and_excludes_other_ids: self.assertIsNotNone(item)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_selected_instance_view.py#L180) | 1 related SelectedInstanceImageViewTests scenarios, each with private fixture state. | `helper: tests.test_selected_instance_view.SelectedInstanceImageViewTests.check_test_label_replacement_and_committed_edit_invalidate_bounds_cache: self.assertEqual(self.view._instance_bounds(1), (120, 100, 130, 110))` | unrecorded / — |
### SelectedInstanceMainWindowTests

Fixtures: `setUpClass` at line 187.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_show_dropdown_plumbs_to_view_and_tracks_analysis_availability`](../../tests/test_selected_instance_view.py#L195) | show dropdown plumbs to view and tracks analysis availability | `self.assertTrue(image.save(str(image_path))); self.assertEqual(selector.currentData(), 'all')` | unrecorded / — |

## tests.test_shape_guided_fill

Shape-guided annotation filling and selection behavior

Source: [tests/test_shape_guided_fill.py](../../tests/test_shape_guided_fill.py). Category: Annotation and interactive image review.

### ShapeGuidedFillTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_shape_guided_fill.py#L412) | 10 related ShapeGuidedFillTests scenarios, each with private fixture state. | `helper: tests.test_shape_guided_fill.ShapeGuidedFillTests.check_test_shape_options_expose_only_shape_specific_growth_controls: self.assertEqual(options.outward_penalty_half_life_fraction, 0.05)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_shape_guided_fill.py#L425) | 2 related ShapeGuidedFillTests scenarios, each with private fixture state. | `helper: tests.test_shape_guided_fill.ShapeGuidedFillTests.check_test_boundary_search_can_reach_a_real_edge_deep_inside_the_prior: self.assertTrue(region.accepted, region.reason)` | unrecorded / — |

## tests.test_shape_guided_ui

Shape-guided annotation desktop controls

Source: [tests/test_shape_guided_ui.py](../../tests/test_shape_guided_ui.py). Category: Annotation and interactive image review.

### ShapeGuidedImageViewTests

Fixtures: `setUpClass` at line 18, `setUp` at line 26.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_shape_guided_ui.py#L148) | 1 related ShapeGuidedImageViewTests scenarios, each with private fixture state. | `helper: tests.test_shape_guided_ui.ShapeGuidedImageViewTests.check_test_preview_shows_prior_refined_boundary_and_commits_cached_mask: self.assertIsNotNone(region)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_shape_guided_ui.py#L152) | 3 related ShapeGuidedImageViewTests scenarios, each with private fixture state. | `helper: tests.test_shape_guided_ui.ShapeGuidedImageViewTests.check_test_empty_evidence_refuses_without_changing_labels: self.assertIsNotNone(region)` | unrecorded / — |

## tests.test_shape_model_export

Actual shape-update outcomes persist and bind the authoritative result revision.

Source: [tests/test_shape_model_export.py](../../tests/test_shape_model_export.py). Category: Measurements and shape models.

### ShapeModelExportTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_shape_model_export.py#L266) | 7 related ShapeModelExportTests scenarios, each with private fixture state. | `helper: tests.test_shape_model_export.ShapeModelExportTests.check_test_actual_file_analysis_and_cached_model_export_without_cuda_transfer: self.assertTrue(cv2.imwrite(str(source), image))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_shape_model_export.py#L276) | 7 related ShapeModelExportTests scenarios, each with private fixture state. | `helper: tests.test_shape_model_export.ShapeModelExportTests.check_test_ordinary_mixed_local_only_and_unavailable_actual_outcomes_roundtrip: self.assertEqual([x.state for x in outcomes], ['applied', 'applied', 'local_only', 'unavailable'])` | unrecorded / — |

## tests.test_shape_pooling_overflow

High-precision Gaussian oracles for representable intermediate-overflow cases.

Source: [tests/test_shape_pooling_overflow.py](../../tests/test_shape_pooling_overflow.py). Category: Measurements and shape models.

### ShapePoolingOverflowTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_shape_pooling_overflow.py#L145) | 6 related ShapePoolingOverflowTests scenarios, each with private fixture state. | `helper: tests.test_shape_pooling_overflow.ShapePoolingOverflowTests.check_test_preventable_overflow_matches_decimal_gaussian_outputs_and_support: self.assertEqual(result.state, 'applied')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_shape_pooling_overflow.py#L154) | 1 related ShapePoolingOverflowTests scenarios, each with private fixture state. | `helper: tests.test_shape_pooling_overflow.ShapePoolingOverflowTests.check_test_variance_boundary_continuity_and_positive_subnormal_output: self.assertEqual(result.state, 'applied')` | unrecorded / — |

## tests.test_shape_scoring_availability

Candidate scoring distinguishes computational abstention from incompatibility.

Source: [tests/test_shape_scoring_availability.py](../../tests/test_shape_scoring_availability.py). Category: Measurements and shape models.

### ShapeScoringAvailabilityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_shape_scoring_availability.py#L157) | 7 related ShapeScoringAvailabilityTests scenarios, each with private fixture state. | `helper: tests.test_shape_scoring_availability.ShapeScoringAvailabilityTests.check_test_ordinary_diagonal_distance_has_known_physical_and_dimensionless_values: self.assertTrue(assessment.scoring_available)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_shape_scoring_availability.py#L167) | 5 related ShapeScoringAvailabilityTests scenarios, each with private fixture state. | `helper: tests.test_shape_scoring_availability.ShapeScoringAvailabilityTests.check_test_finite_psd_covariance_trace_overflow_is_unavailable: self.assert_unavailable(vector, diagonal_component(component, 1e+308))` | unrecorded / — |

## tests.test_shape_update_availability

Precision pooling availability is separate from fit/scoring availability.

Source: [tests/test_shape_update_availability.py](../../tests/test_shape_update_availability.py). Category: Measurements and shape models.

### ShapeUpdateAvailabilityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_shape_update_availability.py#L213) | 6 related ShapeUpdateAvailabilityTests scenarios, each with private fixture state. | `helper: tests.test_shape_update_availability.ShapeUpdateAvailabilityTests.check_test_ordinary_same_representation_pooling_preserves_exact_existing_arithmetic: self.assertEqual(result.state, 'applied')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_shape_update_availability.py#L222) | 7 related ShapeUpdateAvailabilityTests scenarios, each with private fixture state. | `helper: tests.test_shape_update_availability.ShapeUpdateAvailabilityTests.check_test_solver_failure_abstains_but_unrelated_programming_error_propagates: self.assertEqual(result.state, 'applied')` | unrecorded / — |

## tests.test_shape_view_normalization

Repeated biological views must be normalized before their units are mixed.

Source: [tests/test_shape_view_normalization.py](../../tests/test_shape_view_normalization.py). Category: Measurements and shape models.

### ShapeViewNormalizationTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_shape_view_normalization.py#L118) | 7 related ShapeViewNormalizationTests scenarios, each with private fixture state. | `helper: tests.test_shape_view_normalization.ShapeViewNormalizationTests.check_test_differing_views_are_normalized_before_biological_mean_and_rms: np.testing.assert_allclose(vectors[0, (0, 1, 2, 4)], (1, 0.65, 0.25, 0.3))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_shape_view_normalization.py#L128) | 1 related ShapeViewNormalizationTests scenarios, each with private fixture state. | `helper: tests.test_shape_view_normalization.ShapeViewNormalizationTests.check_test_invalid_span_or_unrepresentable_normalized_value_is_withheld: self.assertRaisesRegex(ValueError, 'positive spans')` | unrecorded / — |

## tests.test_species_library_dialog

Species reference library desktop editing

Source: [tests/test_species_library_dialog.py](../../tests/test_species_library_dialog.py). Category: Measurements and shape models.

### SpeciesLibraryDialogTests

Fixtures: `setUpClass` at line 52.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_species_without_a_library_leads_to_creating_one`](../../tests/test_species_library_dialog.py#L73) | species without a library leads to creating one | `self.assertEqual([dialog.tabs.tabText(index) for index in range(dialog.tabs.count())], ['Libraries', 'Create a library', 'Details']); self.assertEqual(dialog.species_combo.currentText(), 'Soybean')` | unrecorded / — |
| [`test_choosing_another_species_lists_its_images_and_builds_for_it`](../../tests/test_species_library_dialog.py#L124) | choosing another species lists its images and builds for it | `self.assertEqual(dialog.create_heading.text(), 'New Lupinus mutabilis library version'); self.assertEqual(dialog.source_list.count(), 1)` | unrecorded / — |
| [`test_unknown_species_must_be_chosen_before_creating`](../../tests/test_species_library_dialog.py#L133) | unknown species must be chosen before creating | `self.assertEqual(dialog.species_combo.currentText(), 'Choose a species…'); self.assertEqual(dialog.create_heading.text(), 'Choose a species first')` | unrecorded / — |
| [`test_existing_version_is_used_extended_and_only_pinned_for_its_species`](../../tests/test_species_library_dialog.py#L141) | existing version is used extended and only pinned for its species | `self.assertTrue(dialog.empty_state.isHidden()); self.assertEqual(dialog.version_table.rowCount(), 1)` | unrecorded / — |
| [`test_other_species_library_cannot_be_pinned_from_this_image`](../../tests/test_species_library_dialog.py#L183) | other species library cannot be pinned from this image | `self.assertEqual(dialog.version_table.rowCount(), 1); self.assertFalse(dialog.pin_button.isEnabled())` | unrecorded / — |

## tests.test_terminal_snapshot

Real terminal/acknowledgment transport and private Qt historical consumers.

Source: [tests/test_terminal_snapshot.py](../../tests/test_terminal_snapshot.py). Category: Persistence, export, identity and outcomes.

### TerminalSnapshotTests

Fixtures: `setUp` at line 28.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_terminal_snapshot.py#L275) | 8 related TerminalSnapshotTests scenarios, each with private fixture state. | `helper: tests.test_terminal_snapshot.TerminalSnapshotTests.check_test_actual_native_preflight_and_multiple_acknowledgments_survive_original_root_deletion: self.assertLogs('seedvision.export.run_outcomes', level='ERROR')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_terminal_snapshot.py#L286) | 8 related TerminalSnapshotTests scenarios, each with private fixture state. | `helper: tests.test_terminal_snapshot.TerminalSnapshotTests.check_test_conflicting_copies_reject_snapshot_without_selecting_an_arbitrary_diagnostic: self.assertEqual(loaded.outcomes, ())` | unrecorded / — |

## tests.test_test_runner

Runner failure reporting, exhaustive tier partition and fixture lifecycle.

Source: [tests/test_test_runner.py](../../tests/test_test_runner.py). Category: Startup, resources and diagnostics.

### TestRunnerTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_parallel_assignment_partitions_once_and_result_merge_cannot_hide_missing_or_failed_units`](../../tests/test_test_runner.py#L20) | parallel assignment partitions once and result merge cannot hide missing or failed units | `self.assertEqual(len(buckets), jobs); self.assertEqual({name for bucket in buckets for name in bucket}, expected)` | unrecorded / — |
| [`test_stall_watchdog_names_the_unit_a_worker_journal_left_open`](../../tests/test_test_runner.py#L54) | stall watchdog names the unit a worker journal left open | `self.assertIsNone(last_started_unit(journal)); self.assertIsNone(last_started_unit(journal))` | unrecorded / — |
| [`test_functional_units_own_scenario_state_and_report_failure_skip_and_cleanup`](../../tests/test_test_runner.py#L66) | functional units own scenario state and report failure skip and cleanup | `self.assertEqual(self.state, []); self.assertEqual(self.state, [])` | unrecorded / — |
| [`test_compilation_fixture_keys_exact_bytes_excludes_mutable_ast_and_restores`](../../tests/test_test_runner.py#L107) | compilation fixture keys exact bytes excludes mutable ast and restores | `self.assertRaisesRegex(RuntimeError, 'body witness'); self.assertIs(first, compile('value=1', 'fixture.py', 'exec', dont_inherit=True))` | unrecorded / — |
| [`test_local_observer_keeps_real_frames_returns_c_calls_and_restores_on_error`](../../tests/test_test_runner.py#L126) | local observer keeps real frames returns c calls and restores on error | `self.assertRaisesRegex(ValueError, 'body failure'); self.assertEqual(target([1, 2]), 2)` | unrecorded / — |
| [`test_invalid_module_exits_before_execution_and_reports_discovery_errors`](../../tests/test_test_runner.py#L156) | invalid module exits before execution and reports discovery errors | `self.assertEqual(process.returncode, 2, process.stderr); self.assertEqual(record['tests'], 0)` | unrecorded / — |
| [`test_worker_ends_with_its_code_and_flushed_output_but_no_interpreter_shutdown`](../../tests/test_test_runner.py#L186) | worker ends with its code and flushed output but no interpreter shutdown | `self.assertEqual(process.returncode, 7, process.stderr); self.assertEqual(process.stdout, 'flushed')` | unrecorded / — |
| [`test_memory_limit_fails_an_oversized_allocation_inside_its_own_test`](../../tests/test_test_runner.py#L195) | memory limit fails an oversized allocation inside its own test | `self.assertEqual(process.returncode, 0, process.stderr); self.assertEqual(process.stdout.split('\n')[:2], ['refused', 'next test True'])` | unrecorded / — |
| [`test_partition_is_complete_and_fixture_timings_preserve_lifecycle`](../../tests/test_test_runner.py#L210) | partition is complete and fixture timings preserve lifecycle | `self.assertEqual({k: len(v) for k, v in groups.items()}, {'cpu-qt': 1, 'cuda': 2}); self.assertEqual({t.id() for v in groups.values() for t in v}, {t.id() for t in items})` | unrecorded / — |
| [`test_windows_left_by_a_test_are_deleted_but_class_fixture_windows_last_the_class`](../../tests/test_test_runner.py#L240) | windows left by a test are deleted but class fixture windows last the class | `self.assertTrue(isValid(seen['class'])); self.assertFalse(isValid(seen['a']))` | unrecorded / — |
| [`test_prerequisites_require_cuda_only_for_full_and_cuda_runs`](../../tests/test_test_runner.py#L289) | prerequisites require cuda only for full and cuda runs | `self.assertTrue(check_prerequisites(mode)['passed'], mode); self.assertFalse(failed['passed'])` | unrecorded / — |

## tests.test_training_experiment

The three-arm training experiment: radial weighting, shared network, budget and reports.

Source: [tests/test_training_experiment.py](../../tests/test_training_experiment.py). Category: Image analysis, calibration and rendering.

### TrainingExperimentTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_training_experiment`](../../tests/test_training_experiment.py#L64) | Training experiment: 12 scenarios from the loss to the final report. | `helper: tests.test_training_experiment.TrainingExperimentTests.check_radial_error_counts_in_proportion_to_object_probability: self.assertAlmostEqual(radial_term(0) / radial_term(1), 10.0, places=4)` | unrecorded / — |

## tests.test_trait_classifier

Software oracles for trait classification; all truth is explicitly synthetic.

Source: [tests/test_trait_classifier.py](../../tests/test_trait_classifier.py). Category: Learning and trait analysis.

### TraitClassifierTests

Fixtures: `setUpClass` at line 26.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_trait_classifier.py#L286) | 8 related TraitClassifierTests scenarios, each with private fixture state. | `helper: tests.test_trait_classifier.TraitClassifierTests.check_test_synthetic_grouped_training_predicts_supplied_definition_ids: self.assertEqual([x['predicted_trait_id'] for x in result['predictions']], ['dark', 'dark', 'light', 'light'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_trait_classifier.py#L297) | 9 related TraitClassifierTests scenarios, each with private fixture state. | `helper: tests.test_trait_classifier.TraitClassifierTests.check_test_group_physical_and_example_overlap_reject_before_optimization: self.assertRaisesRegex(ValueError, f'overlap by {key}')` | unrecorded / — |

## tests.test_trait_evidence

Trait training from a project's own labels, and models kept with the project.

Source: [tests/test_trait_evidence.py](../../tests/test_trait_evidence.py). Category: Learning and trait analysis.

### TraitEvidenceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_project_evidence`](../../tests/test_trait_evidence.py#L201) | 9 related TraitEvidenceTests scenarios, each with private fixture state. | `helper: tests.test_trait_evidence.TraitEvidenceTests.check_seeds_are_counted_by_class_and_skip_reason: self.assertIsNone(image.problem)` | unrecorded / — |
### TraitEvidenceTrainingTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_training_on_project_labels`](../../tests/test_trait_evidence.py#L262) | 2 related TraitEvidenceTrainingTests scenarios, each with private fixture state. | `helper: tests.test_trait_evidence.TraitEvidenceTrainingTests.check_project_labels_train_a_model_that_loads: self.assertEqual(model.model_sha256, report['model_sha256'])` | unrecorded / — |
### TraitModelWindowTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_dialog`](../../tests/test_trait_evidence.py#L444) | 4 related TraitModelWindowTests scenarios, each with private fixture state. | `helper: tests.test_trait_evidence.TraitModelWindowTests.check_first_attempt_without_labels_explains_instead_of_failing: self.assertFalse(dialog.train_button.isEnabled())` | unrecorded / — |
| [`test_models_with_the_project`](../../tests/test_trait_evidence.py#L451) | 5 related TraitModelWindowTests scenarios, each with private fixture state. | `helper: tests.test_trait_evidence.TraitModelWindowTests.check_window_reads_labels_and_blocks_drafts: self.assertEqual(window._trait_model_directory(), self.root / 'projects' / 'trait-models')` | unrecorded / — |

## tests.test_trait_pipeline

Native selection/cache/report witnesses for the frozen trait classifier.

Source: [tests/test_trait_pipeline.py](../../tests/test_trait_pipeline.py). Category: Learning and trait analysis.

### TraitPipelineTests

Fixtures: `setUpClass` at line 28, `setUp` at line 32.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_trait_pipeline.py#L393) | 9 related TraitPipelineTests scenarios, each with private fixture state. | `helper: tests.test_trait_pipeline.TraitPipelineTests.check_test_selected_method_preference_is_shared_and_filters_disabled_branches: self.assertEqual(select_instances(**values, enabled=enabled), select_result(analysis, enabled))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_trait_pipeline.py#L405) | 8 related TraitPipelineTests scenarios, each with private fixture state. | `helper: tests.test_trait_pipeline.TraitPipelineTests.check_test_label_corrected_pixels_and_support_changes_invalidate_features: self.assertIsNot(old, self.cache['trait_classifier.features'])` | unrecorded / — |

## tests.test_trait_review_ui

Actual frozen CUDA trait predictions must reach the native review widgets.

Source: [tests/test_trait_review_ui.py](../../tests/test_trait_review_ui.py). Category: Learning and trait analysis.

### TraitReviewUiTests

Fixtures: `setUpClass` at line 25, `setUp` at line 32.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_trait_review_ui.py#L171) | 1 related TraitReviewUiTests scenarios, each with private fixture state. | `helper: tests.test_trait_review_ui.TraitReviewUiTests.check_test_actual_mixed_predictions_definitions_and_scores_reach_review_without_refitting: self.assertNotEqual(*confidences)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_trait_review_ui.py#L175) | 2 related TraitReviewUiTests scenarios, each with private fixture state. | `helper: tests.test_trait_review_ui.TraitReviewUiTests.check_test_actual_absent_invalid_and_outside_support_abstentions_stay_explanatory: self.assertEqual(set(observed['rows']), {1, 2})` | unrecorded / — |

## tests.test_trait_snapshot

Portable trait evidence and actual relocated CUDA/desktop/batch consumers.

Source: [tests/test_trait_snapshot.py](../../tests/test_trait_snapshot.py). Category: Persistence, export, identity and outcomes.

### TraitSnapshotTests

Fixtures: `setUpClass` at line 35, `setUp` at line 39.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_trait_snapshot.py#L181) | 1 related TraitSnapshotTests scenarios, each with private fixture state. | `helper: tests.test_trait_snapshot.TraitSnapshotTests.check_test_none_and_legacy_without_selected_model_preserve_absence: self.assertNotIn('trait_model_assets', self.index(snapshot))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_trait_snapshot.py#L185) | 5 related TraitSnapshotTests scenarios, each with private fixture state. | `helper: tests.test_trait_snapshot.TraitSnapshotTests.check_test_missing_invalid_or_oversized_selection_cannot_publish_snapshot: self.assertRaises((ValueError, OSError))` | unrecorded / — |

## tests.test_trait_training

Synthetic native evidence, CUDA extraction, stop and publication witnesses.

Source: [tests/test_trait_training.py](../../tests/test_trait_training.py). Category: Learning and trait analysis.

### TraitTrainingTests

Fixtures: `setUp` at line 49.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_trait_training.py#L321) | 7 related TraitTrainingTests scenarios, each with private fixture state. | `helper: tests.test_trait_training.TraitTrainingTests.check_test_compact_training_atomic_reload_and_actual_prediction: self.assertEqual(model.model_sha256, report['model_sha256'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_trait_training.py#L331) | 7 related TraitTrainingTests scenarios, each with private fixture state. | `helper: tests.test_trait_training.TraitTrainingTests.check_test_manifest_and_controls_reject_before_any_raster_or_optimizer: self.assertRaises(ValueError)` | unrecorded / — |
### TraitTrainingUITests

Fixtures: `setUpClass` at line 345, `setUp` at line 349.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_trait_training.py#L506) | 3 related TraitTrainingUITests scenarios, each with private fixture state. | `helper: tests.test_trait_training.TraitTrainingUITests.check_test_dialog_snapshots_manifest_controls_and_activation_choice: self.assertTrue(dialog.activate_checkbox.isChecked())` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_trait_training.py#L512) | 4 related TraitTrainingUITests scenarios, each with private fixture state. | `helper: tests.test_trait_training.TraitTrainingUITests.check_test_serial_busy_stop_and_stale_terminal_signals_preserve_current_job: started.assert_called_once_with(task)` | unrecorded / — |

## tests.test_ui_text_limits

Keep command labels short and tooltips readable across the desktop UI.

Source: [tests/test_ui_text_limits.py](../../tests/test_ui_text_limits.py). Category: Image analysis, calibration and rendering.

### UiTextLimitTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_command_labels_stay_short`](../../tests/test_ui_text_limits.py#L40) | command labels stay short | `self.assertEqual(long_labels, {}, f'command labels must be at most {COMMAND_LIMIT} characters')` | unrecorded / — |
| [`test_tooltips_stay_readable`](../../tests/test_ui_text_limits.py#L48) | tooltips stay readable | `self.assertEqual(long_tooltips, {}, f'tooltips must be at most {TOOLTIP_LIMIT} characters')` | unrecorded / — |
| [`test_scan_sees_the_interface`](../../tests/test_ui_text_limits.py#L56) | scan sees the interface | `self.assertGreater(sum((1 for _ in constant_strings(COMMAND_CALLS))), 100); self.assertGreater(sum((1 for _ in constant_strings({'setToolTip'}))), 100)` | unrecorded / — |

## tests.test_visualization_layers

Analysis visualization layers, overlays, crops and display contracts

Source: [tests/test_visualization_layers.py](../../tests/test_visualization_layers.py). Category: Image analysis, calibration and rendering.

### AnalysisLayerTests

Fixtures: `setUpClass` at line 1189.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_visualization_layers.py#L2518) | 40 related AnalysisLayerTests scenarios, each with private fixture state. | `helper: tests.test_visualization_layers.AnalysisLayerTests.check_test_reference_seed_traits_are_isolated_normalized_material_models: self.assertTrue(products.profile.coat_model_available)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_visualization_layers.py#L2562) | 2 related AnalysisLayerTests scenarios, each with private fixture state. | `helper: tests.test_visualization_layers.AnalysisLayerTests.check_test_missing_other_probabilities_render_black_not_an_unrelated_layer: self.assertFalse(np.any(rgba[..., :3]))` | unrecorded / — |
