# Complete test inventory

This is a source-derived inventory of every discovered test method, audited on 2026-10-23. Functionality descriptions use method docstrings or scenario names, with assertion and call evidence in [the JSON inventory](test_inventory.json). These are intended coverage claims, **not measured statement or branch coverage**.

Markdown shows at most the first two assertion expressions, shortened to 260 characters. JSON retains every direct assertion expression, lexical production-call candidates, interface calls, helper references and their shared assertion evidence, loops, subtest sites, decorators and fixture calls. Dynamic dispatch and inherited fixture behavior require source inspection; this is not a manually proven call graph or an exhaustive enumeration of runtime subtest cases.

Total: **649 methods in 252 test files**. A method can contain many subtests, input cases or expensive executions. Those are not separate unittest methods.

Discovery reconciliation: exact match, 649 canonical unittest IDs; zero discovery errors.

No historical run receipt was used, so status and seconds are blank. Class setup/teardown, discovery and interpreter shutdown are not allocated to methods. A dash means no method duration was recorded, not zero cost. Runtime skips can differ by environment; their exact historical reasons are in JSON.

Regenerate: `.venv\Scripts\python.exe tools/audit_test_inventory.py --collect`. This imports modules for discovery but never runs tests or setup fixtures. Without the ignored historical receipts, use `--without-history`; historical status/timing fields are then unrecorded.

See [the multi-pass audit](../archive/TEST_INFRASTRUCTURE_AUDIT.md) for overlap, cost and infrastructure findings.

| Module | Tests | Historical method seconds | Scope |
| --- | ---: | ---: | --- |
| [tests.test_adaptive_kernel_normalization](#teststest_adaptive_kernel_normalization) | 3 | 0.000 | Pinned public adaptive-kernel normalization, cache identity and saved replay. |
| [tests.test_advanced_analysis](#teststest_advanced_analysis) | 1 | 0.000 | Advanced image layers and CUDA analysis products |
| [tests.test_analysis_settings_persistence](#teststest_analysis_settings_persistence) | 2 | 0.000 | Analysis settings serialization, migration, normalization and private workspace state |
| [tests.test_annotation_centres](#teststest_annotation_centres) | 4 | 0.000 | Annotation centroids and radial hilum metadata must never become predictions. |
| [tests.test_annotation_edge_sources](#teststest_annotation_edge_sources) | 2 | 0.000 | Annotation tools select the intended edge evidence and clear obsolete sources |
| [tests.test_annotation_generation_binding](#teststest_annotation_generation_binding) | 2 | 0.000 | Native label producers bind to the displayed corrected source, including history. |
| [tests.test_annotation_shape_revision](#teststest_annotation_shape_revision) | 4 | 0.000 | Shape review revisions, visibility and reference-library updates |
| [tests.test_annotation_tools](#teststest_annotation_tools) | 2 | 0.000 | Annotation drawing, brush geometry, region operations and instance tools |
| [tests.test_assignment_budget](#teststest_assignment_budget) | 2 | 0.000 | Assignment metrics and segmentation matching respect explicit work budgets |
| [tests.test_authoritative_parity](#teststest_authoritative_parity) | 2 | 0.000 | Real native CUDA analysis, desktop/batch identity and exact review portability. |
| [tests.test_backend_outcomes](#teststest_backend_outcomes) | 1 | 0.000 | Actual backend outcomes, cache recovery, Qt/batch parity and immutable review binding. |
| [tests.test_benchmark_checkpoints](#teststest_benchmark_checkpoints) | 2 | 0.000 | Checkpoint identities, array limits, manifests, storage and corruption rejection |
| [tests.test_benchmark_cli](#teststest_benchmark_cli) | 2 | 0.000 | Single-entry CLI routing and bounded adapters; GPU producers tested separately. |
| [tests.test_benchmark_conformance](#teststest_benchmark_conformance) | 2 | 0.000 | Comparison metrics, tolerances and checkpoint conformance reports |
| [tests.test_benchmark_manifest](#teststest_benchmark_manifest) | 2 | 0.000 | Benchmark manifests, source/environment identity and evidence contracts |
| [tests.test_bootstrap](#teststest_bootstrap) | 8 | 0.000 | Python/runtime dependency admission and single-entry startup |
| [tests.test_calibrated_public_capture_dialog](#teststest_calibrated_public_capture_dialog) | 2 | 0.000 | Actual asynchronous calibrated capture and explicitly injected UI faults. |
| [tests.test_calibrated_public_capture_methods](#teststest_calibrated_public_capture_methods) | 0 | 0.000 | Real mean/Otsu desktop sessions; stored-result checks are not a new oracle. |
| [tests.test_calibration_result_revision](#teststest_calibration_result_revision) | 2 | 0.000 | Actual calibration state must travel with authoritative review revisions. |
| [tests.test_catalogue_nodes](#teststest_catalogue_nodes) | 2 | 0.000 | Independent catalogue instances, typed projections and portable project state. |
| [tests.test_catalogue_nodes_cuda](#teststest_catalogue_nodes_cuda) | 1 | 0.000 | One actual small CUDA witness for independent added-node execution/cache. |
| [tests.test_catalogue_nodes_ui](#teststest_catalogue_nodes_ui) | 2 | 0.000 | CPU/Qt ownership, routing and stale-publication checks for added nodes. |
| [tests.test_catalogue_source_fidelity](#teststest_catalogue_source_fidelity) | 2 | 0.000 | Canonical source fidelity, independent of a supplied graph's self-consistency. |
| [tests.test_checkpoint_bundle](#teststest_checkpoint_bundle) | 2 | 0.000 | Checkpoint bundle serialization, import/export and project references |
| [tests.test_classifier_evaluation](#teststest_classifier_evaluation) | 3 | 0.000 | CPU artifact/evidence binding; synthetic facts do not certify study conduct. |
| [tests.test_color_histogram](#teststest_color_histogram) | 2 | 0.000 | Color histogram calculation, comparison and desktop display |
| [tests.test_conditioned_shape_pooling](#teststest_conditioned_shape_pooling) | 2 | 0.000 | Conditioned pooling uses the same Gaussian model without dropping directions. |
| [tests.test_core_recovery](#teststest_core_recovery) | 2 | 0.000 | Seed core extraction, recovery, label ownership and CUDA bounds |
| [tests.test_correction_attempt_qc](#teststest_correction_attempt_qc) | 2 | 0.000 | Actual correction producers and their recorded terminal QC boundaries. |
| [tests.test_correction_project_dialog](#teststest_correction_project_dialog) | 3 | 0.000 | Exact-image correction Keep, worker handoff and reopened project pins. |
| [tests.test_critical_review_fixes](#teststest_critical_review_fixes) | 2 | 0.000 | Earlier regression fixes spanning annotation coordinates, learning and export |
| [tests.test_desktop_terminal_history](#teststest_desktop_terminal_history) | 4 | 0.000 | Earlier desktop rejection boundaries and bounded evidence-only history review. |
| [tests.test_diagnostics](#teststest_diagnostics) | 1 | 0.000 | Runtime diagnostic facts |
| [tests.test_edge_trace_commit](#teststest_edge_trace_commit) | 1 | 0.000 | Edge trace tool commits and desktop annotation updates |
| [tests.test_entrypoint](#teststest_entrypoint) | 1 | 0.000 | Native desktop application launch and command routing |
| [tests.test_fill_roi_recipe](#teststest_fill_roi_recipe) | 3 | 0.000 | Public small-object ROI equality and complete native working paths. |
| [tests.test_grouped_comparison](#teststest_grouped_comparison) | 2 | 0.000 | Compact synthetic engineering witnesses, not biological validation. |
| [tests.test_gui_layout](#teststest_gui_layout) | 1 | 0.000 | Desktop layout contracts from the GUI review: menus, toolbar, panels, canvas. |
| [tests.test_icon_asset](#teststest_icon_asset) | 1 | 0.000 | Application icon assets |
| [tests.test_image_calibration](#teststest_image_calibration) | 3 | 0.000 | Image calibration, geometry, chart/ruler detection and real-image fixtures |
| [tests.test_image_list_controls](#teststest_image_list_controls) | 2 | 0.000 | Native image-list actions, file preservation and edit ownership. |
| [tests.test_instance_annotation_tiling](#teststest_instance_annotation_tiling) | 2 | 0.000 | Annotation tile rendering, invalidation and image-view state |
| [tests.test_instance_continuity](#teststest_instance_continuity) | 2 | 0.000 | Instance continuity during annotation updates |
| [tests.test_instance_continuity_ui](#teststest_instance_continuity_ui) | 2 | 0.000 | Desktop selection follows instance continuity |
| [tests.test_instance_mask_import](#teststest_instance_mask_import) | 2 | 0.000 | Instance mask loading, source identity and validation |
| [tests.test_instance_mask_import_ui](#teststest_instance_mask_import_ui) | 2 | 0.000 | Instance mask import dialogs and desktop application |
| [tests.test_instance_reference_evidence](#teststest_instance_reference_evidence) | 2 | 0.000 | Annotated-instance reference evidence and visualization |
| [tests.test_instance_reference_repository](#teststest_instance_reference_repository) | 1 | 0.000 | Committed reference fixture repository integrity |
| [tests.test_kmeans_model_artifact](#teststest_kmeans_model_artifact) | 2 | 0.000 | Persistent model metadata, legacy compatibility and full-file identity. |
| [tests.test_kmeans_training_directory](#teststest_kmeans_training_directory) | 4 | 0.000 | Directory selection, bounded decoding and CUDA fits against public PlantCV. |
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
| [tests.test_median_roi_recipe](#teststest_median_roi_recipe) | 3 | 0.000 | Version compatibility, public median ROI values and native working paths. |
| [tests.test_method_qc](#teststest_method_qc) | 6 | 0.000 | Method quality-control facts, provenance and exported summaries |
| [tests.test_method_qc_snapshot](#teststest_method_qc_snapshot) | 3 | 0.000 | Explicit project ownership and actual portable QC consumers. |
| [tests.test_morphology_roi_recipe](#teststest_morphology_roi_recipe) | 2 | 0.000 | Public rectangle-helper equality and integrated native ROI morphology. |
| [tests.test_naive_bayes_plots](#teststest_naive_bayes_plots) | 2 | 0.000 | Compact default-style SVGs versus the actual pinned public plotting helper. |
| [tests.test_naive_bayes_training_directory](#teststest_naive_bayes_training_directory) | 2 | 0.000 | Real decoded directories against the pinned public binary trainer. |
| [tests.test_native_calibrated_range_recipe](#teststest_native_calibrated_range_recipe) | 3 | 0.000 | Complete corrected range-mask workflow with physical observations and replay. |
| [tests.test_native_calibrated_seed_recipe](#teststest_native_calibrated_seed_recipe) | 1 | 0.000 | Real composed card/seed execution, coordinate binding and desktop persistence. |
| [tests.test_native_card_recipe](#teststest_native_card_recipe) | 3 | 0.000 | Automatic-card working graph, CUDA cache boundaries and saved outputs. |
| [tests.test_native_card_replay](#teststest_native_card_replay) | 4 | 0.000 | Colour-card compact archive contracts and allocation-free replay admission. |
| [tests.test_native_card_stage_replay](#teststest_native_card_stage_replay) | 1 | 0.000 | Saved direct card inputs, mixed frames, every product, guards and actual Qt. |
| [tests.test_native_checkpoint_inspection](#teststest_native_checkpoint_inspection) | 2 | 0.000 | Native checkpoint inspection, evidence limits and retained outputs |
| [tests.test_native_color_replay](#teststest_native_color_replay) | 3 | 0.000 | Typed observations, public colour parity, isolated replay and Qt charts. |
| [tests.test_native_core_recovery_recipe](#teststest_native_core_recovery_recipe) | 2 | 0.000 | Native core/recovery graph contracts, CUDA execution and cache behavior |
| [tests.test_native_correction_dialog](#teststest_native_correction_dialog) | 2 | 0.000 | Color correction dialog controls, workers and imported inputs |
| [tests.test_native_correction_files](#teststest_native_correction_files) | 4 | 0.000 | Four-file correction identity, bounded preparation and actual output archives. |
| [tests.test_native_correction_recipe](#teststest_native_correction_recipe) | 1 | 0.000 | Native color correction recipe composition and execution |
| [tests.test_native_crop_replay](#teststest_native_crop_replay) | 3 | 0.000 | Actual dual-frame calibrated crop replay, admission, callback and Qt checks. |
| [tests.test_native_extended_threshold_controls](#teststest_native_extended_threshold_controls) | 1 | 0.000 | Pinned control witnesses and desktop routing for the four working thresholds. |
| [tests.test_native_extended_threshold_recipe](#teststest_native_extended_threshold_recipe) | 2 | 0.000 | Extended threshold recipe variants, contracts and native execution |
| [tests.test_native_extended_threshold_replay](#teststest_native_extended_threshold_replay) | 4 | 0.000 | Pinned public masks, actual archives, and bounded extended-threshold replay. |
| [tests.test_native_image_inputs](#teststest_native_image_inputs) | 3 | 0.000 | Pinned public reader semantics and shared native preparation boundaries. |
| [tests.test_native_kmeans_recipe](#teststest_native_kmeans_recipe) | 2 | 0.000 | Frozen-centre bindings, public predictions, native caches and Qt execution. |
| [tests.test_native_kmeans_training](#teststest_native_kmeans_training) | 2 | 0.000 | Real directory training, CUDA row caching and asynchronous desktop lifecycle. |
| [tests.test_native_local_stage_replay](#teststest_native_local_stage_replay) | 0 | 0.000 | Saved calibrated-ROI replay: exact frames, ancestors and bounded transfers. |
| [tests.test_native_mask_stage](#teststest_native_mask_stage) | 2 | 0.000 | Native mask-stage checkpoints and output identity |
| [tests.test_native_model_replay](#teststest_native_model_replay) | 7 | 0.000 | Archived model binding and shared native/replay resource admission. |
| [tests.test_native_model_stage_replay](#teststest_native_model_stage_replay) | 2 | 0.000 | Portable model/class-map replay, lineage, guards and actual classifier Qt. |
| [tests.test_native_naive_bayes_directory_training_dialog](#teststest_native_naive_bayes_directory_training_dialog) | 0 | 0.000 | Directory training, strict portable provenance, and actual Qt/project saves. |
| [tests.test_native_naive_bayes_plot_dialog](#teststest_native_naive_bayes_plot_dialog) | 2 | 0.000 | Actual async compact plots, model-first saves and full modal lifecycle. |
| [tests.test_native_naive_bayes_recipe](#teststest_native_naive_bayes_recipe) | 2 | 0.000 | Pinned class masks, asset bindings, cache invalidation and desktop execution. |
| [tests.test_native_naive_bayes_training](#teststest_native_naive_bayes_training) | 3 | 0.000 | Public sampled-RGB trainer, TSV handoff and frozen-model prediction gates. |
| [tests.test_native_naive_bayes_training_dialog](#teststest_native_naive_bayes_training_dialog) | 2 | 0.000 | Actual Bayes training UI, atomic TSV saves and project-held provenance. |
| [tests.test_native_paper_replay](#teststest_native_paper_replay) | 3 | 0.000 | Actual paper saved-input replay, primitive schema/admission and Qt workflow. |
| [tests.test_native_preprocessing](#teststest_native_preprocessing) | 3 | 0.000 | Native preprocessing chains, caching and public PlantCV parity |
| [tests.test_native_range_recipe](#teststest_native_range_recipe) | 3 | 0.000 | Public range outputs through real native, archive, Qt and batch paths. |
| [tests.test_native_reader_recipe](#teststest_native_reader_recipe) | 3 | 0.000 | Pinned codec grayscale and versioned desktop/project reader selection. |
| [tests.test_native_recipe](#teststest_native_recipe) | 2 | 0.000 | Native recipe execution, cache behavior, cancellation and results |
| [tests.test_native_recipe_dialog](#teststest_native_recipe_dialog) | 3 | 0.000 | Native recipe dialog selection, execution and result state |
| [tests.test_native_recovery_replay](#teststest_native_recovery_replay) | 2 | 0.000 | Pure recovery contracts, maximum inventory and allocation-free admission. |
| [tests.test_native_recovery_stage_replay](#teststest_native_recovery_stage_replay) | 1 | 0.000 | Actual saved recovery inputs, compact contracts, guards and desktop replay. |
| [tests.test_native_result_export](#teststest_native_result_export) | 4 | 0.000 | Actual native product persistence and saved-project batch bindings. |
| [tests.test_native_seed_working_recipe](#teststest_native_seed_working_recipe) | 2 | 0.000 | Complete native seed working workflows |
| [tests.test_native_size_replay](#teststest_native_size_replay) | 2 | 0.000 | Public size products, bounded typed replay, physical units and Qt state. |
| [tests.test_native_smpts_recipe](#teststest_native_smpts_recipe) | 2 | 0.000 | Native soybean paper workflow composition and outputs |
| [tests.test_native_stage_replay](#teststest_native_stage_replay) | 3 | 0.000 | Actual saved-input CUDA stage replay, lineage and unavailable outcomes. |
| [tests.test_native_threshold_stage](#teststest_native_threshold_stage) | 4 | 0.000 | Native threshold stage inputs, outputs, resources and provenance |
| [tests.test_native_watershed_diagnostic_recipe](#teststest_native_watershed_diagnostic_recipe) | 2 | 0.000 | Watershed diagnostic recipe variants and CUDA products |
| [tests.test_native_watershed_diagnostic_replay](#teststest_native_watershed_diagnostic_replay) | 3 | 0.000 | Saved watershed diagnostic stage replay and guards |
| [tests.test_native_watershed_replay_integration](#teststest_native_watershed_replay_integration) | 1 | 0.000 | Typed saved-stage resource, cancellation and actual asynchronous Qt checks. |
| [tests.test_native_watershed_stage](#teststest_native_watershed_stage) | 3 | 0.000 | Native watershed stage execution and checkpoints |
| [tests.test_native_watershed_stage_replay](#teststest_native_watershed_stage_replay) | 0 | 0.000 | Exact saved watershed products and typed source/local archive contracts. |
| [tests.test_neutral_calibration](#teststest_neutral_calibration) | 3 | 0.000 | Selected grid references, CUDA sampling, cache boundaries and saved facts. |
| [tests.test_node_optimization](#teststest_node_optimization) | 4 | 0.000 | Node parameter search, optimizer state and cache reuse |
| [tests.test_node_optimization_production](#teststest_node_optimization_production) | 1 | 0.000 | Production integration on two generated images; no accuracy claims. |
| [tests.test_node_optimization_ui](#teststest_node_optimization_ui) | 2 | 0.000 | Node optimizer controls and desktop result application |
| [tests.test_node_vocabulary](#teststest_node_vocabulary) | 3 | 0.000 | The engine's node vocabulary must stay bound to the pipeline graph. |
| [tests.test_oriented_edge_components](#teststest_oriented_edge_components) | 2 | 0.000 | Oriented edge components, connectivity and CUDA outputs |
| [tests.test_otsu_calculation_facts](#teststest_otsu_calculation_facts) | 3 | 0.000 | Exact binary domains and single-pass Otsu cutoff facts through native execution. |
| [tests.test_peer_feature_register](#teststest_peer_feature_register) | 1 | 0.000 | Structural checks for the PlantCV/SMPTS operation-level feature register. |
| [tests.test_peer_recipe_catalogue](#teststest_peer_recipe_catalogue) | 2 | 0.000 | Focused integrity checks for the audited, non-executed peer recipes. |
| [tests.test_performance_report](#teststest_performance_report) | 4 | 0.000 | Actual synchronized CUDA/resource/cache facts and explicit correction timing. |
| [tests.test_physical_shape_fit_availability](#teststest_physical_shape_fit_availability) | 2 | 0.000 | Finite calibrated evidence may exceed the physical fitter's numeric domain. |
| [tests.test_pilot_analysis](#teststest_pilot_analysis) | 5 | 0.000 | Committed pilot-image segmentation, calibration and end-to-end analysis |
| [tests.test_pipeline](#teststest_pipeline) | 6 | 0.000 | Pipeline graph topology, controls, settings, state and invalidation |
| [tests.test_pipeline_backend_adapters](#teststest_pipeline_backend_adapters) | 1 | 0.000 | Pinned public-call adapters beyond the seed template's elementary stages. |
| [tests.test_pipeline_backend_archive](#teststest_pipeline_backend_archive) | 0 | 0.000 | Actual public execution stays distinct through archive, QC and replay. |
| [tests.test_pipeline_backend_execution](#teststest_pipeline_backend_execution) | 4 | 0.000 | Actual backend choices, public CPU adapters and node-local cache contracts. |
| [tests.test_pipeline_cable_routing](#teststest_pipeline_cable_routing) | 1 | 0.000 | Pipeline cable geometry and routing |
| [tests.test_pipeline_documents](#teststest_pipeline_documents) | 2 | 0.000 | Standalone graph/layout transport, factory protection and frozen file loads. |
| [tests.test_pipeline_inspector_actions](#teststest_pipeline_inspector_actions) | 1 | 0.000 | Pipeline inspector parameter editing and actions |
| [tests.test_pipeline_ui](#teststest_pipeline_ui) | 15 | 0.000 | Pipeline desktop controls, graph state, navigation, workers and status |
| [tests.test_pipeline_workflow_batch](#teststest_pipeline_workflow_batch) | 0 | 0.000 | Saved workspace workflows execute through the ordinary project export route. |
| [tests.test_pipeline_workflow_ui](#teststest_pipeline_workflow_ui) | 1 | 0.000 | Main-workspace routing, editing and queued-delivery ownership. |
| [tests.test_pipeline_workflows](#teststest_pipeline_workflows) | 5 | 0.000 | Editable workspace contracts and real CUDA node-local execution. |
| [tests.test_pipeline_workspace_files](#teststest_pipeline_workspace_files) | 2 | 0.000 | Desktop freeze, factory copies and complete graph/layout file integration. |
| [tests.test_plantcv_card_detection](#teststest_plantcv_card_detection) | 2 | 0.000 | CUDA color card detection geometry and public-source parity |
| [tests.test_plantcv_card_preprocess](#teststest_plantcv_card_preprocess) | 2 | 0.000 | CUDA color card image preparation and source semantics |
| [tests.test_plantcv_color](#teststest_plantcv_color) | 4 | 0.000 | CUDA color-space conversion and PlantCV-compatible image analysis |
| [tests.test_plantcv_color_card](#teststest_plantcv_color_card) | 2 | 0.000 | CUDA color card chip sampling and correction facts |
| [tests.test_plantcv_contours](#teststest_plantcv_contours) | 2 | 0.000 | CUDA contour extraction, topology and public geometry |
| [tests.test_plantcv_correction](#teststest_plantcv_correction) | 2 | 0.000 | CUDA polynomial color correction and numerical compatibility |
| [tests.test_plantcv_distance](#teststest_plantcv_distance) | 3 | 0.000 | CUDA distance transform and public-source numerical parity |
| [tests.test_plantcv_extended_thresholds](#teststest_plantcv_extended_thresholds) | 2 | 0.000 | CUDA extended threshold algorithms and input boundaries |
| [tests.test_plantcv_fill](#teststest_plantcv_fill) | 6 | 0.000 | CUDA small-object and hole filling with public PlantCV parity |
| [tests.test_plantcv_kmeans](#teststest_plantcv_kmeans) | 2 | 0.000 | CUDA k-means analysis and public-model compatibility |
| [tests.test_plantcv_kmeans_classifier](#teststest_plantcv_kmeans_classifier) | 2 | 0.000 | CUDA k-means class assignment and model boundaries |
| [tests.test_plantcv_kmeans_control](#teststest_plantcv_kmeans_control) | 3 | 0.000 | Compact control decisions checked against actual pinned sklearn methods. |
| [tests.test_plantcv_kmeans_conversion](#teststest_plantcv_kmeans_conversion) | 6 | 0.000 | Offline source-model conversion and shared-codec import compatibility. |
| [tests.test_plantcv_kmeans_export](#teststest_plantcv_kmeans_export) | 2 | 0.000 | Compact fitted-centre export, identity and transfer/cancellation boundaries. |
| [tests.test_plantcv_kmeans_features](#teststest_plantcv_kmeans_features) | 2 | 0.000 | CUDA k-means features and input conversion |
| [tests.test_plantcv_kmeans_fit](#teststest_plantcv_kmeans_fit) | 2 | 0.000 | Complete native fitting compared to traced actual public MiniBatchKMeans.fit. |
| [tests.test_plantcv_kmeans_intervals](#teststest_plantcv_kmeans_intervals) | 2 | 0.000 | Certified compact numeric decisions and actual pinned scalar-state parity. |
| [tests.test_plantcv_kmeans_minibatch](#teststest_plantcv_kmeans_minibatch) | 2 | 0.000 | Exact pinned sklearn dense updates and CUDA row-order/resource contracts. |
| [tests.test_plantcv_kmeans_model](#teststest_plantcv_kmeans_model) | 2 | 0.000 | K-means model structure, identity and validation |
| [tests.test_plantcv_kmeans_plusplus](#teststest_plantcv_kmeans_plusplus) | 2 | 0.000 | Actual public/private pinned k-means++ IDs, centres and RNG state on CUDA. |
| [tests.test_plantcv_kmeans_predict](#teststest_plantcv_kmeans_predict) | 2 | 0.000 | CUDA k-means predictions and source-model agreement |
| [tests.test_plantcv_kmeans_preprocess](#teststest_plantcv_kmeans_preprocess) | 2 | 0.000 | CUDA k-means image preparation and feature order |
| [tests.test_plantcv_kmeans_random](#teststest_plantcv_kmeans_random) | 2 | 0.000 | Exact random-call outputs/state from actual pinned MiniBatchKMeans.fit. |
| [tests.test_plantcv_kmeans_sampled_data](#teststest_plantcv_kmeans_sampled_data) | 2 | 0.000 | Actual pinned public sampled patches, CUDA storage and source RNG order. |
| [tests.test_plantcv_kmeans_tolerance](#teststest_plantcv_kmeans_tolerance) | 2 | 0.000 | Pinned NumPy/sklearn tolerance values and bounded CUDA producers. |
| [tests.test_plantcv_kmeans_training](#teststest_plantcv_kmeans_training) | 2 | 0.000 | Ordered CUDA image training compared with pinned public PlantCV and sklearn. |
| [tests.test_plantcv_kmeans_training_data](#teststest_plantcv_kmeans_training_data) | 2 | 0.000 | Actual pinned public training patches and bounded native CUDA allocation. |
| [tests.test_plantcv_kmeans_training_score](#teststest_plantcv_kmeans_training_score) | 2 | 0.000 | Actual pinned Lloyd/inertia calls and CUDA training-score contracts. |
| [tests.test_plantcv_median](#teststest_plantcv_median) | 5 | 0.000 | CUDA median filtering and public PlantCV pixel parity |
| [tests.test_plantcv_minrect](#teststest_plantcv_minrect) | 2 | 0.000 | CUDA minimum-area rectangle geometry and degeneracies |
| [tests.test_plantcv_morphology](#teststest_plantcv_morphology) | 5 | 0.000 | CUDA erosion/dilation and public PlantCV pixel parity |
| [tests.test_plantcv_naive_bayes](#teststest_plantcv_naive_bayes) | 4 | 0.000 | CUDA Naive Bayes masks versus the pinned PlantCV public function. |
| [tests.test_plantcv_naive_bayes_sampling](#teststest_plantcv_naive_bayes_sampling) | 2 | 0.000 | Actual public binary Bayes sampler observations, RNG and CUDA boundaries. |
| [tests.test_plantcv_naive_bayes_training](#teststest_plantcv_naive_bayes_training) | 2 | 0.000 | Ordered CUDA KDE versus the actual pinned SciPy training dependency. |
| [tests.test_plantcv_peaks](#teststest_plantcv_peaks) | 8 | 0.000 | CUDA peak detection, spacing and source semantics |
| [tests.test_plantcv_range](#teststest_plantcv_range) | 6 | 0.000 | CUDA range thresholding and public pixel parity |
| [tests.test_plantcv_raster_spans](#teststest_plantcv_raster_spans) | 2 | 0.000 | CUDA contour rasterization, spans and output masks |
| [tests.test_plantcv_size](#teststest_plantcv_size) | 4 | 0.000 | CUDA PlantCV-compatible shape/size measurements |
| [tests.test_plantcv_size_diagnostic](#teststest_plantcv_size_diagnostic) | 4 | 0.000 | CUDA size diagnostic products and public compatibility |
| [tests.test_plantcv_thresholds](#teststest_plantcv_thresholds) | 4 | 0.000 | CUDA binary/mean/Otsu thresholds and isolated public-source comparison |
| [tests.test_plantcv_watershed](#teststest_plantcv_watershed) | 2 | 0.000 | CUDA watershed labeling and public-source numerical parity |
| [tests.test_plantcv_watershed_diagnostic](#teststest_plantcv_watershed_diagnostic) | 4 | 0.000 | CUDA watershed diagnostic labels, plots and bounds |
| [tests.test_polynomial_correction_integrity](#teststest_polynomial_correction_integrity) | 2 | 0.000 | Whole-call input and retained-matrix integrity for polynomial correction. |
| [tests.test_portable_result_identity](#teststest_portable_result_identity) | 3 | 0.000 | Actual relocated native results, artifact facts and private review history. |
| [tests.test_preflight_outcomes](#teststest_preflight_outcomes) | 3 | 0.000 | Incomplete request evidence through real batch and Qt preparation boundaries. |
| [tests.test_procedural_fit](#teststest_procedural_fit) | 5 | 0.000 | Procedural seed fitting, search behavior and geometric evidence |
| [tests.test_procedural_fit_ui](#teststest_procedural_fit_ui) | 2 | 0.000 | Procedural fit controls and result application |
| [tests.test_procedural_instances](#teststest_procedural_instances) | 2 | 0.000 | CUDA procedural instance splitting, ownership and analysis integration |
| [tests.test_project_analysis_persistence](#teststest_project_analysis_persistence) | 3 | 0.000 | Project analysis settings, image locators, migration and atomic persistence |
| [tests.test_project_analysis_ui](#teststest_project_analysis_ui) | 2 | 0.000 | Project analysis save/reopen, dirty state, selections and desktop guards |
| [tests.test_project_correction_inputs](#teststest_project_correction_inputs) | 4 | 0.000 | Per-image correction selections, byte-bound copies and portable projects. |
| [tests.test_project_correction_runner](#teststest_project_correction_runner) | 2 | 0.000 | Pinned correction execution, origin archives and actual batch companions. |
| [tests.test_project_native_recipes](#teststest_project_native_recipes) | 2 | 0.000 | Editable native project selections coexist with exact historical graph records. |
| [tests.test_project_reference_captures](#teststest_project_reference_captures) | 2 | 0.000 | Real public batch execution and reference-only publication/QC witnesses. |
| [tests.test_public_capture_layout](#teststest_public_capture_layout) | 2 | 0.000 | Preview geometry and clearing contracts, independent of source calculations. |
| [tests.test_public_reference_attempt_qc](#teststest_public_reference_attempt_qc) | 3 | 0.000 | Desktop boundary witnesses; no inferred outputs from thrown exceptions. |
| [tests.test_public_seed_acceptance](#teststest_public_seed_acceptance) | 3 | 0.000 | Finite actual public-chain acceptance; expected artifacts are never refreshed. |
| [tests.test_reader_acceptance](#teststest_reader_acceptance) | 2 | 0.000 | Frozen public/native reader pixels, outcomes and preparation-cache evidence. |
| [tests.test_recipe_acceptance](#teststest_recipe_acceptance) | 8 | 0.000 | Finite recipe lifecycle acceptance, separate from public/source conformance. |
| [tests.test_recipe_attempt_qc](#teststest_recipe_attempt_qc) | 2 | 0.000 | Actual recipe terminal boundaries, historical transport and Qt ownership. |
| [tests.test_recipe_batch_qc](#teststest_recipe_batch_qc) | 2 | 0.000 | Actual graph-ledger, interrupted publication and portable batch QC witnesses. |
| [tests.test_reference_calibrated_capture_session](#teststest_reference_calibrated_capture_session) | 3 | 0.000 | One actual capture; replayed results test session orchestration only. |
| [tests.test_reference_calibrated_seed](#teststest_reference_calibrated_seed) | 2 | 0.000 | Actual public composition checks, followed by explicitly injected faults. |
| [tests.test_reference_calibration_adapters](#teststest_reference_calibration_adapters) | 2 | 0.000 | Explicit public ROI and size-scale contexts for a composed reference workflow. |
| [tests.test_reference_capture](#teststest_reference_capture) | 3 | 0.000 | Public reference capture sessions, manifests and publication failures |
| [tests.test_reference_checkpoint_inspection](#teststest_reference_checkpoint_inspection) | 0 | 0.000 | Reference checkpoint inspection and graph-stage evidence |
| [tests.test_reference_edge_fit](#teststest_reference_edge_fit) | 2 | 0.000 | Reference-driven edge fitting and rendered evidence |
| [tests.test_reference_edge_strip_descriptor](#teststest_reference_edge_strip_descriptor) | 2 | 0.000 | Reference edge strip extraction, descriptors and overlays |
| [tests.test_reference_edge_toolbar_revision](#teststest_reference_edge_toolbar_revision) | 1 | 0.000 | Edge reference toolbar controls, graph revisions and image-view state |
| [tests.test_reference_graph_persistence](#teststest_reference_graph_persistence) | 2 | 0.000 | Reference graph project serialization and historical identity |
| [tests.test_reference_graph_readonly](#teststest_reference_graph_readonly) | 1 | 0.000 | Read-only public reference graph presentation and edit prevention |
| [tests.test_reference_history](#teststest_reference_history) | 2 | 0.000 | Reference history revision selection and UI presentation |
| [tests.test_reference_library](#teststest_reference_library) | 2 | 0.000 | Reference library aggregation, extraction, contracts and persistence |
| [tests.test_reference_mask_inspection](#teststest_reference_mask_inspection) | 1 | 0.000 | Saved public mask stage inspection |
| [tests.test_reference_mask_stage](#teststest_reference_mask_stage) | 2 | 0.000 | Public mask-stage capture and checkpoint identity |
| [tests.test_reference_model_assets](#teststest_reference_model_assets) | 2 | 0.000 | Real classifier bytes survive project moves and portable snapshot restore. |
| [tests.test_reference_persistence](#teststest_reference_persistence) | 4 | 0.000 | Reference region sidecars, migration, source binding and desktop lifecycle |
| [tests.test_reference_preprocessing](#teststest_reference_preprocessing) | 2 | 0.000 | Public preprocessing chains, capture, controls and cache identity |
| [tests.test_reference_recipes](#teststest_reference_recipes) | 2 | 0.000 | Contract tests for reference recipes; fixtures here are not real recipes. |
| [tests.test_reference_runner](#teststest_reference_runner) | 9 | 0.000 | Isolated PlantCV worker requests, limits, transport and failure outcomes |
| [tests.test_reference_seed_chain](#teststest_reference_seed_chain) | 2 | 0.000 | Actual public seed-chain composition, controls and cancellation |
| [tests.test_reference_session](#teststest_reference_session) | 1 | 0.000 | Shared reference imports retain fresh PlantCV state and actual calculations. |
| [tests.test_reference_setup](#teststest_reference_setup) | 3 | 0.000 | Pinned reference environment and source installation checks |
| [tests.test_reference_source_identity](#teststest_reference_source_identity) | 5 | 0.000 | Source association is preserved separately from annotation generation. |
| [tests.test_reference_stage](#teststest_reference_stage) | 2 | 0.000 | Public stage capture, graph adaptation and checkpoint identity |
| [tests.test_reference_texture_collage](#teststest_reference_texture_collage) | 1 | 0.000 | Reference texture collage layout and image rendering |
| [tests.test_reference_undo](#teststest_reference_undo) | 2 | 0.000 | Reference annotation undo/redo and independent history |
| [tests.test_reference_watershed_capture](#teststest_reference_watershed_capture) | 5 | 0.000 | Hook semantics, strict transport and actual pinned public watershed products. |
| [tests.test_resources](#teststest_resources) | 1 | 0.000 | CUDA resource estimation and runtime admission |
| [tests.test_result_prediction_revision](#teststest_result_prediction_revision) | 4 | 0.000 | Exact prediction facts must travel with review and exported revisions. |
| [tests.test_review_workflows](#teststest_review_workflows) | 2 | 0.000 | Review export, portable snapshots and desktop workflow |
| [tests.test_reviewed_measurement_lineage](#teststest_reviewed_measurement_lineage) | 3 | 0.000 | Actual review snapshots, producer cache boundaries and export identity. |
| [tests.test_run_outcomes](#teststest_run_outcomes) | 3 | 0.000 | Whole-run terminal evidence, actual Qt/batch parity and publication guards. |
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
| [tests.test_smpts_paper](#teststest_smpts_paper) | 5 | 0.000 | Soybean paper method calculations and explicitly supported boundaries |
| [tests.test_species_library_dialog](#teststest_species_library_dialog) | 2 | 0.000 | Species reference library desktop editing |
| [tests.test_terminal_snapshot](#teststest_terminal_snapshot) | 2 | 0.000 | Real terminal/acknowledgment transport and private Qt historical consumers. |
| [tests.test_test_runner](#teststest_test_runner) | 9 | 0.000 | Runner failure reporting, exhaustive tier partition and fixture lifecycle. |
| [tests.test_threshold_conformance](#teststest_threshold_conformance) | 4 | 0.000 | Public/native threshold checkpoint comparison and conformance |
| [tests.test_trait_classifier](#teststest_trait_classifier) | 2 | 0.000 | Software oracles for trait classification; all truth is explicitly synthetic. |
| [tests.test_trait_pipeline](#teststest_trait_pipeline) | 2 | 0.000 | Native selection/cache/report witnesses for the frozen trait classifier. |
| [tests.test_trait_review_ui](#teststest_trait_review_ui) | 2 | 0.000 | Actual frozen CUDA trait predictions must reach the native review widgets. |
| [tests.test_trait_snapshot](#teststest_trait_snapshot) | 2 | 0.000 | Portable trait evidence and actual relocated CUDA/desktop/batch consumers. |
| [tests.test_trait_training](#teststest_trait_training) | 4 | 0.000 | Synthetic native evidence, CUDA extraction, stop and publication witnesses. |
| [tests.test_ui_text_limits](#teststest_ui_text_limits) | 3 | 0.000 | Keep command labels short and tooltips readable across the desktop UI. |
| [tests.test_visualization_layers](#teststest_visualization_layers) | 2 | 0.000 | Analysis visualization layers, overlays, crops and display contracts |

## tests.test_adaptive_kernel_normalization

Pinned public adaptive-kernel normalization, cache identity and saved replay.

Source: [tests/test_adaptive_kernel_normalization.py](../../tests/test_adaptive_kernel_normalization.py). Category: Recipe, replay and reference infrastructure.

### AdaptiveKernelContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_integer_requests_normalize_before_effective_cap_and_keep_other_controls`](../../tests/test_adaptive_kernel_normalization.py#L52) | integer requests normalize before effective cap and keep other controls | `self.assertEqual(mean._validate_mean_parameters(requested, -1.25), (effective, -1.25)); self.assertRaises((TypeError, ValueError))` | unrecorded / — |
| [`test_replay_plans_use_effective_kernel_without_rewriting_request`](../../tests/test_adaptive_kernel_normalization.py#L62) | replay plans use effective kernel without rewriting request | `self.assertEqual(parameters['ksize'], requested); self.assertEqual(plan['work_details']['requested_kernel_size'], requested)` | unrecorded / — |
### AdaptiveKernelCudaTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_additional_numerical_regressions`](../../tests/test_adaptive_kernel_normalization.py#L203) | additional numerical regressions | `helper: tests.test_adaptive_kernel_normalization.AdaptiveKernelCudaTests.check_test_every_even_mean_kernel_matches_pinned_public_masks_on_structured_frame: self.assertEqual(len(cases), 756)` | unrecorded / — |

## tests.test_advanced_analysis

Advanced image layers and CUDA analysis products

Source: [tests/test_advanced_analysis.py](../../tests/test_advanced_analysis.py). Category: Image analysis, calibration and rendering.

### AdvancedAnalysisTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_advanced_analysis.py#L328) | 6 related AdvancedAnalysisTests scenarios, each with private fixture state. | `helper: tests.test_advanced_analysis.AdvancedAnalysisTests.check_test_all_fourteen_products_and_intermediates_are_renderable: self.assertEqual(layers.backend.used, 'cpu')` | unrecorded / — |

## tests.test_analysis_settings_persistence

Analysis settings serialization, migration, normalization and private workspace state

Source: [tests/test_analysis_settings_persistence.py](../../tests/test_analysis_settings_persistence.py). Category: Pipeline workspace and controls.

### AnalysisSettingsPersistenceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_analysis_settings_persistence.py#L1055) | workflow lifecycle | `helper: tests.test_analysis_settings_persistence.AnalysisSettingsPersistenceTests.check_test_version_twenty_two_inserts_image_selector_without_changing_calculations: self.assertEqual(analysis_settings_profile_from_graph(target), analysis_settings_profile_fr…` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_analysis_settings_persistence.py#L1076) | 7 related AnalysisSettingsPersistenceTests scenarios, each with private fixture state. | `helper: tests.test_analysis_settings_persistence.AnalysisSettingsPersistenceTests.check_test_graph_compatibility_is_fully_validated_before_mutation: self.assertRaisesRegex(IncompatibleAnalysisSettingsProfile, 'minimum_marker_score')` | unrecorded / — |

## tests.test_annotation_centres

Annotation centroids and radial hilum metadata must never become predictions.

Source: [tests/test_annotation_centres.py](../../tests/test_annotation_centres.py). Category: Annotation and interactive image review.

### AnnotationGeometryTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_annotation_centres.py#L49) | 2 related AnnotationGeometryTests scenarios, each with private fixture state. | `helper: tests.test_annotation_centres.AnnotationGeometryTests.check_test_all_ids_use_area_centroids_including_disconnected_and_partial_masks: self.assertEqual(set(centres), {1, 2, 65535})` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_annotation_centres.py#L54) | 1 related AnnotationGeometryTests scenarios, each with private fixture state. | `helper: tests.test_annotation_centres.AnnotationGeometryTests.check_test_empty_invalid_and_degenerate_geometry: self.assertEqual(annotation_centres(labels), {})` | unrecorded / — |
### AnnotationCentresUiTests

Fixtures: `setUpClass` at line 62.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_annotation_centres.py#L351) | 8 related AnnotationCentresUiTests scenarios, each with private fixture state. | `helper: tests.test_annotation_centres.AnnotationCentresUiTests.check_test_compact_seed_selector_and_no_routine_explanatory_labels: self.assertEqual(spin.prefix(), '')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_annotation_centres.py#L362) | 1 related AnnotationCentresUiTests scenarios, each with private fixture state. | `helper: tests.test_annotation_centres.AnnotationCentresUiTests.check_test_next_unannotated_cycles_painted_seeds_missing_condition_or_shape: self.assertIsNotNone(key)` | unrecorded / — |

## tests.test_annotation_edge_sources

Annotation tools select the intended edge evidence and clear obsolete sources

Source: [tests/test_annotation_edge_sources.py](../../tests/test_annotation_edge_sources.py). Category: Annotation and interactive image review.

### AnnotationEdgeSourceTests

Fixtures: `setUpClass` at line 17.

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
| [`test_curvature_prior_changes_only_trace_and_downstream_cache`](../../tests/test_annotation_shape_revision.py#L161) | curvature prior changes only trace and downstream cache | `helper: tests.test_annotation_shape_revision.ShapeRevisionTests.check_test_curvature_prior_changes_only_trace_and_downstream_cache: self.assertIs(cache['layer.edge_gradients'], gradients)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_annotation_shape_revision.py#L164) | 6 related ShapeRevisionTests scenarios, each with private fixture state. | `helper: tests.test_annotation_shape_revision.ShapeRevisionTests.check_test_old_shape_libraries_cannot_be_relabelled_after_source_exclusion: self.assertIsNone(resolved.artifact.dimensions_shape)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_annotation_shape_revision.py#L173) | 1 related ShapeRevisionTests scenarios, each with private fixture state. | `helper: tests.test_annotation_shape_revision.ShapeRevisionTests.check_test_boundary_distribution_is_seed_balanced_and_rotation_invariant: self.assertAlmostEqual(sum(summaries[0].boundary_turn_degrees), 360.0, delta=1.0)` | unrecorded / — |
### AnnotationWidgetTests

Fixtures: `setUpClass` at line 181.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_annotation_shape_revision.py#L402) | 5 related AnnotationWidgetTests scenarios, each with private fixture state. | `helper: tests.test_annotation_shape_revision.AnnotationWidgetTests.check_test_distribution_has_axes_units_pose_key_and_error_bars: self.assertIn('Maximum span (mm)', text)` | unrecorded / — |

## tests.test_annotation_tools

Annotation drawing, brush geometry, region operations and instance tools

Source: [tests/test_annotation_tools.py](../../tests/test_annotation_tools.py). Category: Annotation and interactive image review.

### AnnotationToolTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_annotation_tools.py#L453) | 12 related AnnotationToolTests scenarios, each with private fixture state. | `helper: tests.test_annotation_tools.AnnotationToolTests.check_test_edge_cursor_prefers_nearby_ridge_to_stronger_parallel_ridge: self.assertTupleEqual(snap_edge_point((30, 20), edge, 12), (30, 20))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_annotation_tools.py#L468) | 4 related AnnotationToolTests scenarios, each with private fixture state. | `helper: tests.test_annotation_tools.AnnotationToolTests.check_test_smart_fill_adapts_to_touching_colours_and_stops_at_edge: self.assertGreater(added, 900)` | unrecorded / — |

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

Fixtures: `setUpClass` at line 28, `setUp` at line 35.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_missing_model_and_incompatible_review_are_explicit_on_both_routes`](../../tests/test_authoritative_parity.py#L172) | missing model and incompatible review are explicit on both routes | `self.assertTrue(all((row['trait_prediction']['abstention_reason'] == 'model_file_missing' for row in desktop['instances']))); self.assertEqual(batch['review_lookup'], 'incompatible')` | unrecorded / — |
### ReviewLookupTests

Fixtures: `setUpClass` at line 191.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_invalid_duplicate_oversized_and_stale_reviews_remain_distinct_and_unchanged`](../../tests/test_authoritative_parity.py#L197) | invalid duplicate oversized and stale reviews remain distinct and unchanged | `self.assertEqual(lookup_result_review(root, revision), (None, 'not_recorded')); self.assertEqual(lookup_result_review(root, revision), (None, 'malformed'))` | unrecorded / — |

## tests.test_backend_outcomes

Actual backend outcomes, cache recovery, Qt/batch parity and immutable review binding.

Source: [tests/test_backend_outcomes.py](../../tests/test_backend_outcomes.py). Category: Persistence, export, identity and outcomes.

### BackendOutcomeTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_backend_outcomes.py#L129) | 5 related BackendOutcomeTests scenarios, each with private fixture state. | `helper: tests.test_backend_outcomes.BackendOutcomeTests.check_test_available_empty_missing_and_disabled_are_distinct_and_round_trip: self.assertEqual(records['procedural_instances']['state'], 'available')` | unrecorded / — |

## tests.test_benchmark_checkpoints

Checkpoint identities, array limits, manifests, storage and corruption rejection

Source: [tests/test_benchmark_checkpoints.py](../../tests/test_benchmark_checkpoints.py). Category: Recipe, replay and reference infrastructure.

### BenchmarkCheckpointTests

Fixtures: `setUp` at line 131, `tearDown` at line 135.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_benchmark_checkpoints.py#L818) | 16 related BenchmarkCheckpointTests scenarios, each with private fixture state. | `helper: tests.test_benchmark_checkpoints.BenchmarkCheckpointTests.check_test_real_npz_round_trip_preserves_identity_semantics_and_dtype: self.assertEqual(loaded.checkpoint_sha256, reference.checkpoint_sha256)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_benchmark_checkpoints.py#L837) | 17 related BenchmarkCheckpointTests scenarios, each with private fixture state. | `helper: tests.test_benchmark_checkpoints.BenchmarkCheckpointTests.check_test_selected_loading_rejects_unknown_names_and_keeps_aggregate_admission: self.assertRaises(InvalidCheckpoint)` | unrecorded / — |

## tests.test_benchmark_cli

Single-entry CLI routing and bounded adapters; GPU producers tested separately.

Source: [tests/test_benchmark_cli.py](../../tests/test_benchmark_cli.py). Category: Recipe, replay and reference infrastructure.

### BenchmarkCLITests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_benchmark_cli.py#L210) | 9 related BenchmarkCLITests scenarios, each with private fixture state. | `helper: tests.test_benchmark_cli.BenchmarkCLITests.check_test_benchmark_actions_are_exclusive_with_each_other_and_learning: self.assertRaises(SystemExit)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_benchmark_cli.py#L222) | 4 related BenchmarkCLITests scenarios, each with private fixture state. | `helper: tests.test_benchmark_cli.BenchmarkCLITests.check_test_output_and_evidence_arguments_fail_before_runtime_bootstrap: self.assertEqual(launch.main(args), 2)` | unrecorded / — |

## tests.test_benchmark_conformance

Comparison metrics, tolerances and checkpoint conformance reports

Source: [tests/test_benchmark_conformance.py](../../tests/test_benchmark_conformance.py). Category: Recipe, replay and reference infrastructure.

### BenchmarkConformanceTests

Fixtures: `setUp` at line 127, `tearDown` at line 131.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_benchmark_conformance.py#L441) | 1 related BenchmarkConformanceTests scenarios, each with private fixture state. | `helper: tests.test_benchmark_conformance.BenchmarkConformanceTests.check_test_threshold_border_correspondence_is_explicit_and_exact: self.assertRaisesRegex(ConformanceError, 'Mask representation semantics differ')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_benchmark_conformance.py#L445) | 2 related BenchmarkConformanceTests scenarios, each with private fixture state. | `helper: tests.test_benchmark_conformance.BenchmarkConformanceTests.check_test_primitive_path_rejects_raster_representation_semantic_mismatch: self.assertRaisesRegex(ConformanceError, 'Mask raster representation semantics')` | unrecorded / — |

## tests.test_benchmark_manifest

Benchmark manifests, source/environment identity and evidence contracts

Source: [tests/test_benchmark_manifest.py](../../tests/test_benchmark_manifest.py). Category: Recipe, replay and reference infrastructure.

### BenchmarkManifestTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_benchmark_manifest.py#L578) | 9 related BenchmarkManifestTests scenarios, each with private fixture state. | `helper: tests.test_benchmark_manifest.BenchmarkManifestTests.check_test_round_trip_is_immutable_strict_and_does_not_mutate_payload: self.assertEqual(payload, original)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_benchmark_manifest.py#L590) | 6 related BenchmarkManifestTests scenarios, each with private fixture state. | `helper: tests.test_benchmark_manifest.BenchmarkManifestTests.check_test_missing_groups_truth_or_access_log_blocks_evaluation_without_guessing: self.assertFalse(audit.evaluation_eligible)` | unrecorded / — |

## tests.test_bootstrap

Python/runtime dependency admission and single-entry startup

Source: [tests/test_bootstrap.py](../../tests/test_bootstrap.py). Category: Startup, resources and diagnostics.

### VersionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_bootstrap.py#L51) | 2 related VersionTests scenarios, each with private fixture state. | `helper: tests.test_bootstrap.VersionTests.check_test_version_key_accepts_common_suffixes: self.assertEqual(version_key('2.13.0+cu130'), (2, 13, 0))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_bootstrap.py#L56) | 1 related VersionTests scenarios, each with private fixture state. | `helper: tests.test_bootstrap.VersionTests.check_test_invalid_version_is_rejected: self.assertRaises(ValueError)` | unrecorded / — |
### DependencyInspectionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_bootstrap.py#L103) | 1 related DependencyInspectionTests scenarios, each with private fixture state. | `helper: tests.test_bootstrap.DependencyInspectionTests.check_test_missing_import_requires_installation: self.assertFalse(check.installed)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_bootstrap.py#L107) | 3 related DependencyInspectionTests scenarios, each with private fixture state. | `helper: tests.test_bootstrap.DependencyInspectionTests.check_test_compatible_supplied_package_is_kept: self.assertTrue(check.installed)` | unrecorded / — |
### PipCommandTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_bootstrap.py#L195) | 4 related PipCommandTests scenarios, each with private fixture state. | `helper: tests.test_bootstrap.PipCommandTests.check_test_online_command_can_prefer_local_wheels: self.assertIn('--find-links', command)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_bootstrap.py#L202) | 1 related PipCommandTests scenarios, each with private fixture state. | `helper: tests.test_bootstrap.PipCommandTests.check_test_offline_command_rejects_an_empty_wheelhouse: self.assertRaises(BootstrapError)` | unrecorded / — |
### ManifestTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_repository_manifest_loads`](../../tests/test_bootstrap.py#L208) | repository manifest loads | `self.assertEqual(distributions, {'PySide6', 'numpy', 'opencv-python', 'matplotlib', 'torch'}); self.assertEqual(configuration.python.recommended, '3.12')` | unrecorded / — |
| [`test_trait_manifest_contains_agreed_species_and_damage_fields`](../../tests/test_bootstrap.py#L232) | trait manifest contains agreed species and damage fields | `self.assertEqual(species, {'soybean', 'lupinus_mutabilis', 'lupinus_polyphyllus', 'lupinus_mexicanus'}); self.assertEqual(set(traits['damage_flags']), {'cracking', 'missing_or_peeling_coat', 'chipping_or_breakage', 'discoloration'})` | unrecorded / — |

## tests.test_calibrated_public_capture_dialog

Actual asynchronous calibrated capture and explicitly injected UI faults.

Source: [tests/test_calibrated_public_capture_dialog.py](../../tests/test_calibrated_public_capture_dialog.py). Category: Recipe, replay and reference infrastructure.

### CalibratedPublicDesktopContractsTests

Fixtures: `setUpClass` at line 120, `setUp` at line 123.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_source_replacement_during_worker_discards_failure_and_previews`](../../tests/test_calibrated_public_capture_dialog.py#L142) | source replacement during worker discards failure and previews | `self.assertTrue(entered.wait(2)); self.assertIn('Source image changed', dialog.status_label.text())` | unrecorded / — |
| [`test_request_binding_rejects_wrong_spec_or_source_before_checkpoint_reads`](../../tests/test_calibrated_public_capture_dialog.py#L161) | request binding rejects wrong spec or source before checkpoint reads | `self.assertRaisesRegex(ValueError, 'selection'); self.assertRaisesRegex(ValueError, 'byte identity')` | unrecorded / — |

## tests.test_calibrated_public_capture_methods

Real mean/Otsu desktop sessions; stored-result checks are not a new oracle.

Source: [tests/test_calibrated_public_capture_methods.py](../../tests/test_calibrated_public_capture_methods.py). Category: Recipe, replay and reference infrastructure.


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

Source: [tests/test_catalogue_nodes.py](../../tests/test_catalogue_nodes.py). Category: Recipe, replay and reference infrastructure.

### CatalogueNodeTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_catalogue_nodes.py#L206) | 4 related CatalogueNodeTests scenarios, each with private fixture state. | `helper: tests.test_catalogue_nodes.CatalogueNodeTests.check_test_catalogue_includes_default_smpts_models_and_four_input_correction: self.assertEqual(sum((entry.backend == 'default' for entry in entries)), 50)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_catalogue_nodes.py#L213) | 1 related CatalogueNodeTests scenarios, each with private fixture state. | `helper: tests.test_catalogue_nodes.CatalogueNodeTests.check_test_cross_instance_edges_and_duplicate_or_unregistered_descriptors_reject: self.assertRaisesRegex(ValueError, 'explicit adapter')` | unrecorded / — |

## tests.test_catalogue_nodes_cuda

One actual small CUDA witness for independent added-node execution/cache.

Source: [tests/test_catalogue_nodes_cuda.py](../../tests/test_catalogue_nodes_cuda.py). Category: Recipe, replay and reference infrastructure.

### CatalogueNodesCUDAIsolationTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_duplicate_thresholds_keep_independent_pixels_cache_and_saved_controls`](../../tests/test_catalogue_nodes_cuda.py#L27) | duplicate thresholds keep independent pixels cache and saved controls | `self.assertTrue(cv2.imwrite(str(path), np.repeat(gray[..., None], 3, axis=2))); self.assertIsNot(a.cache, b.cache)` | unrecorded / — |

## tests.test_catalogue_nodes_ui

CPU/Qt ownership, routing and stale-publication checks for added nodes.

Source: [tests/test_catalogue_nodes_ui.py](../../tests/test_catalogue_nodes_ui.py). Category: Recipe, replay and reference infrastructure.

### CatalogueNodesControllerTests

Fixtures: `setUpClass` at line 62, `setUp` at line 67, `tearDown` at line 73.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_catalogue_nodes_ui.py#L191) | 4 related CatalogueNodesControllerTests scenarios, each with private fixture state. | `helper: tests.test_catalogue_nodes_ui.CatalogueNodesControllerTests.check_test_duplicate_default_instances_own_controls_cache_and_prerequisite_presence: self.assertNotEqual(first, second)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_catalogue_nodes_ui.py#L198) | 2 related CatalogueNodesControllerTests scenarios, each with private fixture state. | `helper: tests.test_catalogue_nodes_ui.CatalogueNodesControllerTests.check_test_queue_waits_for_primary_and_stop_withholds_old_result: self.assertIsNone(component.result)` | unrecorded / — |

## tests.test_catalogue_source_fidelity

Canonical source fidelity, independent of a supplied graph's self-consistency.

Source: [tests/test_catalogue_source_fidelity.py](../../tests/test_catalogue_source_fidelity.py). Category: Recipe, replay and reference infrastructure.

### CatalogueSourceFidelityTests

Fixtures: `setUpClass` at line 21.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_catalogue_source_fidelity.py#L135) | 3 related CatalogueSourceFidelityTests scenarios, each with private fixture state. | `helper: tests.test_catalogue_source_fidelity.CatalogueSourceFidelityTests.check_test_changed_operation_source_symbol_and_port_semantics_reject_despite_valid_pins: self.assertEqual(forged.provenance, self.recipe.provenance)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_catalogue_source_fidelity.py#L141) | 4 related CatalogueSourceFidelityTests scenarios, each with private fixture state. | `helper: tests.test_catalogue_source_fidelity.CatalogueSourceFidelityTests.check_test_canonical_bytes_roundtrip_and_layout_only_changes_remain_valid: self.assertEqual(RecipeCatalogue.from_json(original).to_json(), original)` | unrecorded / — |

## tests.test_checkpoint_bundle

Checkpoint bundle serialization, import/export and project references

Source: [tests/test_checkpoint_bundle.py](../../tests/test_checkpoint_bundle.py). Category: Recipe, replay and reference infrastructure.

### CheckpointBundleTests

Fixtures: `setUp` at line 31, `tearDown` at line 36.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_checkpoint_bundle.py#L149) | 4 related CheckpointBundleTests scenarios, each with private fixture state. | `helper: tests.test_checkpoint_bundle.CheckpointBundleTests.check_test_missing_dependency_and_small_budgets_never_publish: self.assertRaises(InvalidCheckpoint)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_checkpoint_bundle.py#L156) | 2 related CheckpointBundleTests scenarios, each with private fixture state. | `helper: tests.test_checkpoint_bundle.CheckpointBundleTests.check_test_source_changed_after_copy_is_not_published: self.assertRaisesRegex(InvalidCheckpoint, 'changed while bundling')` | unrecorded / — |

## tests.test_classifier_evaluation

CPU artifact/evidence binding; synthetic facts do not certify study conduct.

Source: [tests/test_classifier_evaluation.py](../../tests/test_classifier_evaluation.py). Category: Recipe, replay and reference infrastructure.

### ClassifierEvaluationTests

Fixtures: `setUp` at line 68.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_actual_native_cuda_training_artifact_binds_consumed_source`](../../tests/test_classifier_evaluation.py#L260) | actual native cuda training artifact binds consumed source | `helper: tests.test_classifier_evaluation.ClassifierEvaluationTests.check_test_actual_native_cuda_training_artifact_binds_consumed_source: self.assertTrue(cv2.imwrite(str(image), np.arange(64, dtype=np.uint8).reshape(8, 8)))` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_classifier_evaluation.py#L263) | 9 related ClassifierEvaluationTests scenarios, each with private fixture state. | `helper: tests.test_classifier_evaluation.ClassifierEvaluationTests.check_test_native_membership_binds_actual_full_artifact_and_training_bytes: self.assertEqual(binding.model_sha256, sha256(self.raw).hexdigest())` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_classifier_evaluation.py#L275) | 6 related ClassifierEvaluationTests scenarios, each with private fixture state. | `helper: tests.test_classifier_evaluation.ClassifierEvaluationTests.check_test_training_access_absent_wrong_purpose_and_empty_reject_before_artifact_read: self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_color_histogram

Color histogram calculation, comparison and desktop display

Source: [tests/test_color_histogram.py](../../tests/test_color_histogram.py). Category: Recipe, replay and reference infrastructure.

### ColorHistogramTests

Fixtures: `setUpClass` at line 45.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_native_sample_selection_and_empty_transition_preserve_all_observations`](../../tests/test_color_histogram.py#L57) | native sample selection and empty transition preserve all observations | `self.assertEqual(view.samples.count(), 2); self.assertEqual(view.plot.series.sample, 'first_1')` | unrecorded / — |
| [`test_invalid_or_incomplete_observations_never_form_a_complete_chart`](../../tests/test_color_histogram.py#L84) | invalid or incomplete observations never form a complete chart | `self.assertRaises(ValueError); self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_conditioned_shape_pooling

Conditioned pooling uses the same Gaussian model without dropping directions.

Source: [tests/test_conditioned_shape_pooling.py](../../tests/test_conditioned_shape_pooling.py). Category: Measurements and shape models.

### ConditionedShapePoolingTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_conditioned_shape_pooling.py#L171) | 8 related ConditionedShapePoolingTests scenarios, each with private fixture state. | `helper: tests.test_conditioned_shape_pooling.ConditionedShapePoolingTests.check_test_anisotropic_diagonal_pooling_matches_independent_scalar_gaussian_oracle: self.assertEqual(result.state, 'applied')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_conditioned_shape_pooling.py#L182) | 2 related ConditionedShapePoolingTests scenarios, each with private fixture state. | `helper: tests.test_conditioned_shape_pooling.ConditionedShapePoolingTests.check_test_conditioned_signed_mean_cancellation_is_valid: self.assertEqual(result.state, 'applied')` | unrecorded / — |

## tests.test_core_recovery

Seed core extraction, recovery, label ownership and CUDA bounds

Source: [tests/test_core_recovery.py](../../tests/test_core_recovery.py). Category: Image analysis, calibration and rendering.

### CoreRecoveryTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_core_recovery.py#L445) | 8 related CoreRecoveryTests scenarios, each with private fixture state. | `helper: tests.test_core_recovery.CoreRecoveryTests.check_test_isolated_recovery_preserves_pre_erosion_contour_and_unseeded_component: np.testing.assert_array_equal(partition.labels.cpu().numpy() == 901, isolated)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_core_recovery.py#L456) | 6 related CoreRecoveryTests scenarios, each with private fixture state. | `helper: tests.test_core_recovery.CoreRecoveryTests.check_test_canvas_invalid_support_and_empty_statuses_remain_explicit: self.assertTrue(obj.touches_canvas)` | unrecorded / — |

## tests.test_correction_attempt_qc

Actual correction producers and their recorded terminal QC boundaries.

Source: [tests/test_correction_attempt_qc.py](../../tests/test_correction_attempt_qc.py). Category: Persistence, export, identity and outcomes.

### CorrectionAttemptQCTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_correction_attempt_qc.py#L120) | 3 related CorrectionAttemptQCTests scenarios, each with private fixture state. | `helper: tests.test_correction_attempt_qc.CorrectionAttemptQCTests.check_test_process_control_and_secondary_record_failure_preserve_original_exception: self.assertRaises(KeyboardInterrupt)` | unrecorded / — |
### CorrectionAttemptQtTests

Fixtures: `setUpClass` at line 130.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_actual_return_with_invalid_ports_records_validation_failure_without_outputs`](../../tests/test_correction_attempt_qc.py#L159) | actual return with invalid ports records validation failure without outputs | `self.assertIsNone(dialog.result); self.assertIsNone(dialog._prepared)` | unrecorded / — |

## tests.test_correction_project_dialog

Exact-image correction Keep, worker handoff and reopened project pins.

Source: [tests/test_correction_project_dialog.py](../../tests/test_correction_project_dialog.py). Category: Persistence, export, identity and outcomes.

### CorrectionSessionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_context_uses_exact_locator_for_identical_bytes_and_preserves_loaded_record`](../../tests/test_correction_project_dialog.py#L36) | context uses exact locator for identical bytes and preserves loaded record | `self.assertEqual(other.image.source.sha256, self.context.image.source.sha256); self.assertNotEqual(other.image.identifier, self.context.image.identifier)` | unrecorded / — |
| [`test_cancellation_and_wrong_capture_result_do_not_publish_master`](../../tests/test_correction_project_dialog.py#L53) | cancellation and wrong capture result do not publish master | `self.assertRaises(NativeRecipeCancelled); self.assertRaisesRegex(ValueError, 'frozen selection')` | unrecorded / — |
### CorrectionProjectDialogTests

Fixtures: `setUpClass` at line 67.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_window_preserves_distinct_same_byte_image_binding_and_history`](../../tests/test_correction_project_dialog.py#L123) | window preserves distinct same byte image binding and history | `self.assertEqual(window._project_correction_bindings, (first, captured)); self.assertNotEqual(first.image_id, captured.image_id)` | unrecorded / — |

## tests.test_critical_review_fixes

Earlier regression fixes spanning annotation coordinates, learning and export

Source: [tests/test_critical_review_fixes.py](../../tests/test_critical_review_fixes.py). Category: Image analysis, calibration and rendering.

### CriticalIntegrityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_critical_review_fixes.py#L277) | 10 related CriticalIntegrityTests scenarios, each with private fixture state. | `helper: tests.test_critical_review_fixes.CriticalIntegrityTests.check_test_checkpoint_cache_uses_content_and_result_rejects_replaced_weights: self.assertEqual(loader.call_count, 2)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_critical_review_fixes.py#L290) | 10 related CriticalIntegrityTests scenarios, each with private fixture state. | `helper: tests.test_critical_review_fixes.CriticalIntegrityTests.check_test_streamed_evaluation_both_families_and_hybrid: self.assertFalse(report['scientifically_validated'])` | unrecorded / — |

## tests.test_desktop_terminal_history

Earlier desktop rejection boundaries and bounded evidence-only history review.

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
| [`test_validation_and_recovery`](../../tests/test_desktop_terminal_history.py#L244) | 2 related DesktopTerminalHistoryTests scenarios, each with private fixture state. | `helper: tests.test_desktop_terminal_history.DesktopTerminalHistoryTests.check_test_project_apply_failure_keeps_existing_image_and_drafts_before_session_clear: self.assertFalse(self.window._open_project(self.master))` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_desktop_terminal_history.py#L249) | 1 related DesktopTerminalHistoryTests scenarios, each with private fixture state. | `helper: tests.test_desktop_terminal_history.DesktopTerminalHistoryTests.check_test_reopened_disk_history_ack_export_keeps_current_analysis_drafts_and_reviews_unchanged: self.assertEqual(table.rowCount(), 1)` | unrecorded / — |

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

Fixtures: `setUpClass` at line 16.

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
| [`test_workflow_lifecycle`](../../tests/test_entrypoint.py#L107) | 3 related EntrypointTests scenarios, each with private fixture state. | `helper: tests.test_entrypoint.EntrypointTests.check_test_importing_entrypoint_does_not_import_gui_or_ml_packages: self.assertEqual(completed.stdout.strip(), '')` | unrecorded / — |

## tests.test_fill_roi_recipe

Public small-object ROI equality and complete native working paths.

Source: [tests/test_fill_roi_recipe.py](../../tests/test_fill_roi_recipe.py). Category: Recipe, replay and reference infrastructure.

### FillROIContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_fill_roi_recipe.py#L143) | 3 related FillROIContractTests scenarios, each with private fixture state. | `helper: tests.test_fill_roi_recipe.FillROIContractTests.check_test_all_twenty_eight_variants_and_reverse_order_preserve_base_contracts: self.assertEqual(len(factory.BASE_IDS), 28)` | unrecorded / — |
### FillROIIntegrationTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_fill_roi_recipe.py#L243) | 2 related FillROIIntegrationTests scenarios, each with private fixture state. | `helper: tests.test_fill_roi_recipe.FillROIIntegrationTests.check_test_crop_local_connectivity_area_equality_holes_exterior_and_shortcuts: torch.testing.assert_close(actual, expected)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_fill_roi_recipe.py#L248) | 1 related FillROIIntegrationTests scenarios, each with private fixture state. | `helper: tests.test_fill_roi_recipe.FillROIIntegrationTests.check_test_invalid_bounds_strict_cancel_and_input_preservation: self.assertRaises((TypeError, ValueError))` | unrecorded / — |

## tests.test_grouped_comparison

Compact synthetic engineering witnesses, not biological validation.

Source: [tests/test_grouped_comparison.py](../../tests/test_grouped_comparison.py). Category: Recipe, replay and reference infrastructure.

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

Fixtures: `setUpClass` at line 15, `setUp` at line 20.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_gui_layout.py#L134) | 9 related GuiLayoutTests scenarios, each with private fixture state. | `helper: tests.test_gui_layout.GuiLayoutTests.check_test_every_shortcut_is_unique: self.assertEqual({key: count for key, count in shortcuts.items() if count > 1}, {})` | unrecorded / — |

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

Fixtures: `setUpClass` at line 11.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_pilot_batch_rims_match_the_calibrated_96_mm_outer_edges`](../../tests/test_image_calibration.py#L218) | pilot batch rims match the calibrated 96 mm outer edges | `helper: tests.test_image_calibration.ImageCalibrationTests.check_test_pilot_batch_rims_match_the_calibrated_96_mm_outer_edges: self.assertIsNotNone(calibration.pixels_per_mm)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_image_calibration.py#L221) | 2 related ImageCalibrationTests scenarios, each with private fixture state. | `helper: tests.test_image_calibration.ImageCalibrationTests.check_test_scale_is_rejected_when_longest_ticks_are_not_major_increments: self.assertEqual(len(ruler.metric_tick_segments), 151)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_image_calibration.py#L226) | 2 related ImageCalibrationTests scenarios, each with private fixture state. | `helper: tests.test_image_calibration.ImageCalibrationTests.check_test_swatch_grid_snaps_to_shared_printed_edges: self.assertLess(float(np.max(np.abs(detected - expected))), 2.5)` | unrecorded / — |

## tests.test_image_list_controls

Native image-list actions, file preservation and edit ownership.

Source: [tests/test_image_list_controls.py](../../tests/test_image_list_controls.py). Category: Image analysis, calibration and rendering.

### ImageListControlsTests

Fixtures: `setUpClass` at line 28.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_header_open_and_row_removal_preserve_files_and_unresolved_project_records`](../../tests/test_image_list_controls.py#L31) | header open and row removal preserve files and unresolved project records | `self.assertTrue(window._open_project(project)); self.assertNotIn(window.open_action, window.workflow_toolbar.actions())` | unrecorded / — |
| [`test_removal_protects_edits_and_busy_work_then_clears_selected_native_display`](../../tests/test_image_list_controls.py#L98) | removal protects edits and busy work then clears selected native display | `self.assertFalse(window._remove_image(first)); busy.assert_called_once()` | unrecorded / — |

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

Fixtures: `setUpClass` at line 16, `setUp` at line 24.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_selected_seed_warning_is_exact_and_does_not_block_apply`](../../tests/test_instance_continuity_ui.py#L55) | selected seed warning is exact and does not block apply | `self.assertFalse(warning.isHidden()); self.assertEqual(warning.text(), '⚠ Seed 7 has 2 disconnected areas.')` | unrecorded / — |
| [`test_global_warning_and_identity_cache_update_for_a_new_draft`](../../tests/test_instance_continuity_ui.py#L101) | global warning and identity cache update for a new draft | `self.assertIn('1 seed ID has disconnected areas: 2', warning.text()); self.assertEqual(banner.text(), warning.text())` | unrecorded / — |

## tests.test_instance_mask_import

Instance mask loading, source identity and validation

Source: [tests/test_instance_mask_import.py](../../tests/test_instance_mask_import.py). Category: Annotation and interactive image review.

### InstanceMaskImportTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_instance_mask_import.py#L332) | 7 related InstanceMaskImportTests scenarios, each with private fixture state. | `helper: tests.test_instance_mask_import.InstanceMaskImportTests.check_test_corrected_png_tiff_and_npz_preserve_sparse_ids_exactly: self.assertTrue(cv2.imwrite(str(paths[0]), labels))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_instance_mask_import.py#L342) | 2 related InstanceMaskImportTests scenarios, each with private fixture state. | `helper: tests.test_instance_mask_import.InstanceMaskImportTests.check_test_corrected_import_rejects_shape_type_range_and_corruption: self.assertTrue(cv2.imwrite(str(colour), np.zeros((5, 6, 3), np.uint8)))` | unrecorded / — |

## tests.test_instance_mask_import_ui

Instance mask import dialogs and desktop application

Source: [tests/test_instance_mask_import_ui.py](../../tests/test_instance_mask_import_ui.py). Category: Annotation and interactive image review.

### InstanceMaskImportUiTests

Fixtures: `setUpClass` at line 20, `setUp` at line 28.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_instance_mask_import_ui.py#L331) | 6 related InstanceMaskImportUiTests scenarios, each with private fixture state. | `helper: tests.test_instance_mask_import_ui.InstanceMaskImportUiTests.check_test_panel_import_is_draft_with_provenance_warning_and_undo: chooser.assert_not_called()` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_instance_mask_import_ui.py#L340) | 2 related InstanceMaskImportUiTests scenarios, each with private fixture state. | `helper: tests.test_instance_mask_import_ui.InstanceMaskImportUiTests.check_test_validation_failure_warns_without_mutating_draft_or_history: warning.assert_called_once()` | unrecorded / — |

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

## tests.test_kmeans_model_artifact

Persistent model metadata, legacy compatibility and full-file identity.

Source: [tests/test_kmeans_model_artifact.py](../../tests/test_kmeans_model_artifact.py). Category: Recipe, replay and reference infrastructure.

### KMeansModelArtifactTests

Fixtures: `setUp` at line 59.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_kmeans_model_artifact.py#L185) | 5 related KMeansModelArtifactTests scenarios, each with private fixture state. | `helper: tests.test_kmeans_model_artifact.KMeansModelArtifactTests.check_test_legacy_codec_bytes_and_identity_remain_unchanged: self.assertEqual(artifact.model, models.parse_kmeans_model(raw))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_kmeans_model_artifact.py#L193) | 3 related KMeansModelArtifactTests scenarios, each with private fixture state. | `helper: tests.test_kmeans_model_artifact.KMeansModelArtifactTests.check_test_checksum_and_numeric_tampering_rejects: self.assertRaises(models.KMeansModelError)` | unrecorded / — |

## tests.test_kmeans_training_directory

Directory selection, bounded decoding and CUDA fits against public PlantCV.

Source: [tests/test_kmeans_training_directory.py](../../tests/test_kmeans_training_directory.py). Category: Recipe, replay and reference infrastructure.

### KMeansTrainingDirectoryTests

Fixtures: `setUp` at line 75.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_duplicate_occurrences_upload_once_and_preserve_bgr_channel_order`](../../tests/test_kmeans_training_directory.py#L277) | duplicate occurrences upload once and preserve bgr channel order | `helper: tests.test_kmeans_training_directory.KMeansTrainingDirectoryTests.check_test_duplicate_occurrences_upload_once_and_preserve_bgr_channel_order: self.assertEqual(upload.call_count, 1)` | unrecorded / — |
| [`test_custom_stream_fit_is_bounded_and_changes_during_fit_withhold_result`](../../tests/test_kmeans_training_directory.py#L281) | custom stream fit is bounded and changes during fit withhold result | `helper: tests.test_kmeans_training_directory.KMeansTrainingDirectoryTests.check_test_custom_stream_fit_is_bounded_and_changes_during_fit_withhold_result: self.assertEqual(fitted.training.training_rows.features.device.type, 'cuda')` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_kmeans_training_directory.py#L284) | 6 related KMeansTrainingDirectoryTests scenarios, each with private fixture state. | `helper: tests.test_kmeans_training_directory.KMeansTrainingDirectoryTests.check_test_regular_decode_skips_are_hashed_and_nonfiles_are_preserved: self.assertEqual([item.status for item in manifest.occurrences], ['accepted', 'decode_skipped', 'decode_skipped'…` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_kmeans_training_directory.py#L293) | 4 related KMeansTrainingDirectoryTests scenarios, each with private fixture state. | `helper: tests.test_kmeans_training_directory.KMeansTrainingDirectoryTests.check_test_depth_alpha_and_header_pixel_limits_reject_before_unsupported_analysis: self.assertRaisesRegex(native.KMeansDirectoryError, 'eight-bit')` | unrecorded / — |

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
| [`test_workflow_lifecycle`](../../tests/test_learning_workflow.py#L291) | 5 related LearningWorkflowTests scenarios, each with private fixture state. | `helper: tests.test_learning_workflow.LearningWorkflowTests.check_test_export_dialog_returns_review_and_split_metadata: self.assertEqual(options.manifest_path, Path(temporary) / 'manifest.json')` | unrecorded / — |

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
| [`test_workflow_lifecycle`](../../tests/test_manual_seed_centre_persistence.py#L165) | 3 related ManualSeedCentreStoreTests scenarios, each with private fixture state. | `helper: tests.test_manual_seed_centre_persistence.ManualSeedCentreStoreTests.check_test_round_trip_preserves_exact_source_floats_mode_and_empty_replace: self.assertIsNotNone(loaded)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_manual_seed_centre_persistence.py#L171) | 2 related ManualSeedCentreStoreTests scenarios, each with private fixture state. | `helper: tests.test_manual_seed_centre_persistence.ManualSeedCentreStoreTests.check_test_changed_source_bytes_are_rejected_without_mutating_sidecar: self.assertRaises(ManualSeedCentreFingerprintMismatch)` | unrecorded / — |

## tests.test_manual_seed_centre_ui

Manual centre drawing, selection and desktop persistence

Source: [tests/test_manual_seed_centre_ui.py](../../tests/test_manual_seed_centre_ui.py). Category: Annotation and interactive image review.

### ManualSeedCentreUiTests

Fixtures: `setUpClass` at line 18.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_manual_seed_centre_ui.py#L388) | 2 related ManualSeedCentreUiTests scenarios, each with private fixture state. | `helper: tests.test_manual_seed_centre_ui.ManualSeedCentreUiTests.check_test_qt_add_manual_drag_auto_drag_and_right_delete_use_marker_semantics: self.assertTrue(succeeded, error)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_manual_seed_centre_ui.py#L393) | 3 related ManualSeedCentreUiTests scenarios, each with private fixture state. | `helper: tests.test_manual_seed_centre_ui.ManualSeedCentreUiTests.check_test_completed_edit_inverse_projects_autosaves_and_invalidates_only_dependents: self.assertIsNotNone(key)` | unrecorded / — |

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

## tests.test_median_roi_recipe

Version compatibility, public median ROI values and native working paths.

Source: [tests/test_median_roi_recipe.py](../../tests/test_median_roi_recipe.py). Category: Recipe, replay and reference infrastructure.

### MedianROIContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_median_roi_recipe.py#L124) | 1 related MedianROIContractTests scenarios, each with private fixture state. | `helper: tests.test_median_roi_recipe.MedianROIContractTests.check_test_versions_coexist_and_forged_contracts_reject: self.assertEqual(len(set(factory.BASE_IDS)), 56)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_median_roi_recipe.py#L128) | 2 related MedianROIContractTests scenarios, each with private fixture state. | `helper: tests.test_median_roi_recipe.MedianROIContractTests.check_test_all_six_action_orders_preserve_nondefault_independent_controls: self.assertTrue(factory.is_median_roi_recipe(recipe))` | unrecorded / — |
### MedianROIIntegrationTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_invalid_bounds_reject_before_filter_allocation_and_cancel_preserves_input`](../../tests/test_median_roi_recipe.py#L137) | invalid bounds reject before filter allocation and cancel preserves input | `self.assertRaises((TypeError, ValueError)); worker.assert_not_called()` | unrecorded / — |

## tests.test_method_qc

Method quality-control facts, provenance and exported summaries

Source: [tests/test_method_qc.py](../../tests/test_method_qc.py). Category: Persistence, export, identity and outcomes.

### RecordedMethodQCTests

Fixtures: `setUp` at line 36.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_method_qc.py#L113) | 2 related RecordedMethodQCTests scenarios, each with private fixture state. | `helper: tests.test_method_qc.RecordedMethodQCTests.check_test_strict_schema_revision_nonfinite_duplicates_and_bounds_reject: self.assertRaises(ValueError)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_method_qc.py#L118) | 1 related RecordedMethodQCTests scenarios, each with private fixture state. | `helper: tests.test_method_qc.RecordedMethodQCTests.check_test_layout_only_reference_changes_preserve_semantic_qc_identity: self.assertEqual(original.revision, actual.revision)` | unrecorded / — |
### NativeMethodQCTests

Fixtures: `setUp` at line 125.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_legacy_archive_without_sidecar_derives_only_verified_stored_facts`](../../tests/test_method_qc.py#L145) | legacy archive without sidecar derives only verified stored facts | `self.assertRaises(FileNotFoundError); self.assertEqual(derived, quality)` | unrecorded / — |
### MethodQCQtTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_real_qt_export_ack_load_and_stale_record_preserve_unknown_eligibility`](../../tests/test_method_qc.py#L161) | real qt export ack load and stale record preserve unknown eligibility | `self.assertEqual(dialog.acknowledgment['qc_revision'], quality.revision); self.assertIn('Recorded acknowledgment: local reviewer', dialog.status.text())` | unrecorded / — |
| [`test_actual_reference_graph_button_captures_selected_stage_only`](../../tests/test_method_qc.py#L191) | actual reference graph button captures selected stage only | `self.assertEqual(dialog.quality.payload['identity']['recipe_identity'], attachment.recipe.identity.analytical_dict()); self.assertEqual(dialog.quality.payload['identity']['node_outcomes'][0]['stage_id'], self.stage.stage_id)` | unrecorded / — |
### RealReferenceMethodQCTests

Fixtures: `setUpClass` at line 217.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_actual_pinned_stage_inspection_roundtrip_and_missing_transition_invalidate_ack`](../../tests/test_method_qc.py#L235) | actual pinned stage inspection roundtrip and missing transition invalidate ack | `self.assertEqual(inspection.status, InspectionStatus.VERIFIED, inspection.detail); self.assertEqual(observed['checkpoint_sha256'], record.checkpoint_sha256)` | unrecorded / — |

## tests.test_method_qc_snapshot

Explicit project ownership and actual portable QC consumers.

Source: [tests/test_method_qc_snapshot.py](../../tests/test_method_qc_snapshot.py). Category: Persistence, export, identity and outcomes.

### MethodQCSnapshotTests

Fixtures: `setUp` at line 31.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_method_qc_snapshot.py#L147) | 3 related MethodQCSnapshotTests scenarios, each with private fixture state. | `helper: tests.test_method_qc_snapshot.MethodQCSnapshotTests.check_test_exported_sidecar_is_not_retained_without_explicit_action_and_legacy_snapshot_restores: self.assertFalse(history.load_project_method_qc(self.root, self.master).records)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_method_qc_snapshot.py#L153) | 2 related MethodQCSnapshotTests scenarios, each with private fixture state. | `helper: tests.test_method_qc_snapshot.MethodQCSnapshotTests.check_test_changed_objects_and_restore_byte_limits_reject_before_publication: self.assert_no_restore(package)` | unrecorded / — |
### MethodQCSnapshotQtTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_actual_retain_action_restored_graph_history_and_multiple_acknowledgments_are_inert`](../../tests/test_method_qc_snapshot.py#L166) | actual retain action restored graph history and multiple acknowledgments are inert | `self.assertIn('Retained 2', viewer.status.text()); self.assertEqual(child.quality, self.quality)` | unrecorded / — |

## tests.test_morphology_roi_recipe

Public rectangle-helper equality and integrated native ROI morphology.

Source: [tests/test_morphology_roi_recipe.py](../../tests/test_morphology_roi_recipe.py). Category: Recipe, replay and reference infrastructure.

### MorphologyROIContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_morphology_roi_recipe.py#L140) | 3 related MorphologyROIContractTests scenarios, each with private fixture state. | `helper: tests.test_morphology_roi_recipe.MorphologyROIContractTests.check_test_historical_mask_identity_tracks_shared_callback_implementation: self.assertRaises(bridge.NativeMaskStageError)` | unrecorded / — |
### MorphologyROIIntegrationTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_morphology_roi_recipe.py#L217) | 3 related MorphologyROIIntegrationTests scenarios, each with private fixture state. | `helper: tests.test_morphology_roi_recipe.MorphologyROIIntegrationTests.check_test_crop_boundary_is_independent_and_real_controls_change_pixels: self.assertFalse(torch.equal(crop_first[12:28, 12:28], full_first[12:28, 12:28]))` | unrecorded / — |

## tests.test_naive_bayes_plots

Compact default-style SVGs versus the actual pinned public plotting helper.

Source: [tests/test_naive_bayes_plots.py](../../tests/test_naive_bayes_plots.py). Category: Recipe, replay and reference infrastructure.

### BayesPDFPlotTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_naive_bayes_plots.py#L150) | 1 related BayesPDFPlotTests scenarios, each with private fixture state. | `helper: tests.test_naive_bayes_plots.BayesPDFPlotTests.check_test_exact_tables_hashes_renderer_metadata_and_no_global_rng_change: self.assertEqual(before[0], after[0])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_naive_bayes_plots.py#L154) | 2 related BayesPDFPlotTests scenarios, each with private fixture state. | `helper: tests.test_naive_bayes_plots.BayesPDFPlotTests.check_test_bad_models_labels_and_forged_bundles_fail_before_publication: self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_naive_bayes_training_directory

Real decoded directories against the pinned public binary trainer.

Source: [tests/test_naive_bayes_training_directory.py](../../tests/test_naive_bayes_training_directory.py). Category: Recipe, replay and reference infrastructure.

### BayesDirectoryTrainingTests

Fixtures: `setUp` at line 65.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_naive_bayes_training_directory.py#L185) | 2 related BayesDirectoryTrainingTests scenarios, each with private fixture state. | `helper: tests.test_naive_bayes_training_directory.BayesDirectoryTrainingTests.check_test_caps_precede_decode_and_upload_and_invalid_arguments_consume_no_rng: self.assertRaises(native.BayesDirectoryError)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_naive_bayes_training_directory.py#L190) | 1 related BayesDirectoryTrainingTests scenarios, each with private fixture state. | `helper: tests.test_naive_bayes_training_directory.BayesDirectoryTrainingTests.check_test_exact_walk_record_bound_is_independent_of_aggregate_entry_count: self.assertEqual(len(selected.walk), native.MAX_DIRECTORY_ENTRIES)` | unrecorded / — |

## tests.test_native_calibrated_range_recipe

Complete corrected range-mask workflow with physical observations and replay.

Source: [tests/test_native_calibrated_range_recipe.py](../../tests/test_native_calibrated_range_recipe.py). Category: Recipe, replay and reference infrastructure.

### RangeCompositionContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_explicit_range_before_mask_median_local_traits_and_canonical_persistence`](../../tests/test_native_calibrated_range_recipe.py#L67) | explicit range before mask median local traits and canonical persistence | `self.assertEqual(len(recipe.stages), 24); self.assertEqual(RecipeSpec.from_json(recipe.to_json()), recipe)` | unrecorded / — |
| [`test_old_plans_and_joint_range_admission_boundary`](../../tests/test_native_calibrated_range_recipe.py#L89) | old plans and joint range admission boundary | `self.assertEqual(old, 64 * pixels + 16 * min(pixels, factory.peaks.MAX_CANDIDATES) + 256 * 1024 * 1024); self.assertEqual(factory.retained_bound(pixels, range_arm=True), old + 16 * pixels)` | unrecorded / — |
### CalibratedRangeTests

Fixtures: `setUpClass` at line 112, `tearDownClass` at line 124, `setUp` at line 126.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_additional_numerical_regressions`](../../tests/test_native_calibrated_range_recipe.py#L184) | additional numerical regressions | `helper: tests.test_native_calibrated_range_recipe.CalibratedRangeTests.check_test_five_modes_use_exact_corrected_roi_pixels_and_local_range_replay: np.testing.assert_array_equal(outputs['mask'].cpu(), public[case['name'] + '_mask'])` | unrecorded / — |

## tests.test_native_calibrated_seed_recipe

Real composed card/seed execution, coordinate binding and desktop persistence.

Source: [tests/test_native_calibrated_seed_recipe.py](../../tests/test_native_calibrated_seed_recipe.py). Category: Recipe, replay and reference infrastructure.

### CalibratedSeedTests

Fixtures: `setUp` at line 93, `tearDown` at line 100.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_calibrated_seed_recipe.py#L228) | 3 related CalibratedSeedTests scenarios, each with private fixture state. | `helper: tests.test_native_calibrated_seed_recipe.CalibratedSeedTests.check_test_actual_public_auto_crop_seed_chains_and_physical_observations: self.assertLessEqual(np.abs(corrected.astype(int) - expected['corrected'][..., ::-1].astype(int)).max(), 1)` | unrecorded / — |

## tests.test_native_card_recipe

Automatic-card working graph, CUDA cache boundaries and saved outputs.

Source: [tests/test_native_card_recipe.py](../../tests/test_native_card_recipe.py). Category: Recipe, replay and reference infrastructure.

### CardContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_additive_recipe_contract_and_retained_bound`](../../tests/test_native_card_recipe.py#L31) | additive recipe contract and retained bound | `self.assertEqual(len(recipe.stages), 9); self.assertEqual(recipe.parameters_for_runner(factory.STANDARD)['pos'], 3)` | unrecorded / — |
### NativeCardRecipeTests

Fixtures: `setUp` at line 52, `tearDown` at line 61.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_card_recipe.py#L139) | 3 related NativeCardRecipeTests scenarios, each with private fixture state. | `helper: tests.test_native_card_recipe.NativeCardRecipeTests.check_test_actual_public_auto_and_roi_default_position: self.assertEqual(actual.status, 'executed', [(n.stage_id, n.reason) for n in actual.nodes])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_card_recipe.py#L145) | 1 related NativeCardRecipeTests scenarios, each with private fixture state. | `helper: tests.test_native_card_recipe.NativeCardRecipeTests.check_test_workspace_rejection_precedes_cuda_upload: self.assertRaisesRegex(factory.card.AutomaticCardResourceError, 'fixture cap')` | unrecorded / — |

## tests.test_native_card_replay

Colour-card compact archive contracts and allocation-free replay admission.

Source: [tests/test_native_card_replay.py](../../tests/test_native_card_replay.py). Category: Recipe, replay and reference infrastructure.

### CardCompactContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_card_replay.py#L205) | 5 related CardCompactContractTests scenarios, each with private fixture state. | `helper: tests.test_native_card_replay.CardCompactContractTests.check_test_actual_archive_tags_preserve_mapping_sequence_types_and_source_order: self.assertEqual(decoded, replay.canonical(value))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_card_replay.py#L213) | 5 related CardCompactContractTests scenarios, each with private fixture state. | `helper: tests.test_native_card_replay.CardCompactContractTests.check_test_complete_escaped_json_byte_boundary_including_empty_container_envelopes: self.assertEqual(replay.decode(binding), replay.canonical(value))` | unrecorded / — |
### CardReplayAdmissionTests

Fixtures: `setUpClass` at line 225.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_card_replay.py#L301) | 4 related CardReplayAdmissionTests scenarios, each with private fixture state. | `helper: tests.test_native_card_replay.CardReplayAdmissionTests.check_test_full_and_local_raster_specs_retain_float64_labels: self.assertEqual(replay.raster_spec('roi_bgr', SHAPE, ROI), ((540, 380, 3), '\|u1'))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_card_replay.py#L308) | 1 related CardReplayAdmissionTests scenarios, each with private fixture state. | `helper: tests.test_native_card_replay.CardReplayAdmissionTests.check_test_tight_limits_reject_before_runtime_or_allocation_and_small_allocations_round: self.assertEqual(replay.plan(operation, SHAPE, parameters, bounds=ROI, selected_limit=planned.selected_by…` | unrecorded / — |

## tests.test_native_card_stage_replay

Saved direct card inputs, mixed frames, every product, guards and actual Qt.

Source: [tests/test_native_card_stage_replay.py](../../tests/test_native_card_stage_replay.py). Category: Recipe, replay and reference infrastructure.

### CardStageReplayTests

Fixtures: `setUpClass` at line 39, `tearDownClass` at line 52.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_card_stage_replay.py#L139) | 3 related CardStageReplayTests scenarios, each with private fixture state. | `helper: tests.test_native_card_stage_replay.CardStageReplayTests.check_test_every_full_and_roi_card_stage_matches_all_outputs_and_direct_selection: self.assertEqual(result.status, 'matched', result.reason)` | unrecorded / — |

## tests.test_native_checkpoint_inspection

Native checkpoint inspection, evidence limits and retained outputs

Source: [tests/test_native_checkpoint_inspection.py](../../tests/test_native_checkpoint_inspection.py). Category: Recipe, replay and reference infrastructure.

### NativeCheckpointInspectionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_native_checkpoint_inspection.py#L131) | 2 related NativeCheckpointInspectionTests scenarios, each with private fixture state. | `helper: tests.test_native_checkpoint_inspection.NativeCheckpointInspectionTests.check_test_unbound_missing_cancelled_and_absent_runtime_return_no_pixels: self.assertEqual(inspect_native_checkpoint(None, recipe, record).status, InspectionStatus.STORE_UNBOUND)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_native_checkpoint_inspection.py#L136) | 1 related NativeCheckpointInspectionTests scenarios, each with private fixture state. | `helper: tests.test_native_checkpoint_inspection.NativeCheckpointInspectionTests.check_test_native_catalogue_match_preserves_source_topology_and_parameter_contracts: self.assertTrue(_matches_native_catalogue(recipe, catalogue))` | unrecorded / — |

## tests.test_native_color_replay

Typed observations, public colour parity, isolated replay and Qt charts.

Source: [tests/test_native_color_replay.py](../../tests/test_native_color_replay.py). Category: Recipe, replay and reference infrastructure.

### ColourReplayContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_color_replay.py#L125) | 3 related ColourReplayContractTests scenarios, each with private fixture state. | `helper: tests.test_native_color_replay.ColourReplayContractTests.check_test_known_records_nonfinite_tokens_and_immutable_expected_metadata: self.assertIs(compact.validate(decoded, 'observations', PARAMETERS, 1, ValueError), decoded)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_color_replay.py#L131) | 1 related ColourReplayContractTests scenarios, each with private fixture state. | `helper: tests.test_native_color_replay.ColourReplayContractTests.check_test_nonfinite_wrong_field_bad_bin_shapes_and_inconsistent_availability_rejected: self.assertRaises(ValueError)` | unrecorded / — |
### ColourReplayTests

Fixtures: `setUpClass` at line 139, `tearDownClass` at line 151.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_four_spaces_compare_every_public_observation_and_replay_only_selected_arrays`](../../tests/test_native_color_replay.py#L184) | four spaces compare every public observation and replay only selected arrays | `self.assertEqual(result.status, 'matched', result.reason); self.assertEqual(calls, ['analysis.color'])` | unrecorded / — |

## tests.test_native_core_recovery_recipe

Native core/recovery graph contracts, CUDA execution and cache behavior

Source: [tests/test_native_core_recovery_recipe.py](../../tests/test_native_core_recovery_recipe.py). Category: Recipe, replay and reference infrastructure.

### CoreRecoveryGraphTests

Fixtures: `setUp` at line 102.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_core_recovery_recipe.py#L361) | 4 related CoreRecoveryGraphTests scenarios, each with private fixture state. | `helper: tests.test_native_core_recovery_recipe.CoreRecoveryGraphTests.check_test_actual_pinned_full_chain_and_pre_erosion_union: self.assertEqual(result.status, 'executed')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_core_recovery_recipe.py#L368) | 3 related CoreRecoveryGraphTests scenarios, each with private fixture state. | `helper: tests.test_native_core_recovery_recipe.CoreRecoveryGraphTests.check_test_nested_products_immutable_cache_mutation_regenerates_and_callback_mutation_clears: self.assertRaises(TypeError)` | unrecorded / — |

## tests.test_native_correction_dialog

Color correction dialog controls, workers and imported inputs

Source: [tests/test_native_correction_dialog.py](../../tests/test_native_correction_dialog.py). Category: Recipe, replay and reference infrastructure.

### NativeCorrectionPreparationTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_four_png_inputs_decode_to_explicit_cuda_ports_and_exact_hashes`](../../tests/test_native_correction_dialog.py#L54) | four png inputs decode to explicit cuda ports and exact hashes | `self.assertIsInstance(prepared, NativeCorrectionInputs); self.assertEqual(set(prepared.tensors), set(_PORTS))` | unrecorded / — |
### NativeCorrectionDialogTests

Fixtures: `setUpClass` at line 82.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_close_cancels_worker_and_waits_for_it_to_drain`](../../tests/test_native_correction_dialog.py#L104) | close cancels worker and waits for it to drain | `self.assertTrue(dialog.isVisible()); self.assertTrue(dialog._cancel.is_set())` | unrecorded / — |

## tests.test_native_correction_files

Four-file correction identity, bounded preparation and actual output archives.

Source: [tests/test_native_correction_files.py](../../tests/test_native_correction_files.py). Category: Recipe, replay and reference infrastructure.

### CorrectionContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_native_builder_preserves_historical_recipe_and_exact_four_ports`](../../tests/test_native_correction_files.py#L36) | native builder preserves historical recipe and exact four ports | `self.assertEqual(historical.to_json(), before); self.assertEqual(recipe.provenance.mode, RecipeProvenanceMode.NATIVE_COMPATIBLE)` | unrecorded / — |
| [`test_stream_read_bounds_hold_even_if_file_grows`](../../tests/test_native_correction_files.py#L49) | stream read bounds hold even if file grows | `self.assertRaises(NativeCorrectionError); self.assertEqual(stream.requests, [9])` | unrecorded / — |
### CorrectionFileTests

Fixtures: `setUp` at line 71, `tearDown` at line 78.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_correction_files.py#L194) | 4 related CorrectionFileTests scenarios, each with private fixture state. | `helper: tests.test_native_correction_files.CorrectionFileTests.check_test_every_raw_file_change_recomputes_only_correction: self.assertIsNot(current, prepared)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_correction_files.py#L201) | 1 related CorrectionFileTests scenarios, each with private fixture state. | `helper: tests.test_native_correction_files.CorrectionFileTests.check_test_report_identity_tampering_is_rejected_before_any_array_load: self.assertRaises(exports.NativeResultExportError)` | unrecorded / — |

## tests.test_native_correction_recipe

Native color correction recipe composition and execution

Source: [tests/test_native_correction_recipe.py](../../tests/test_native_correction_recipe.py). Category: Recipe, replay and reference infrastructure.

### NativeCorrectionRecipeTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_identity_height_workspace_and_source_contract_fail_before_calculation`](../../tests/test_native_correction_recipe.py#L99) | identity height workspace and source contract fail before calculation | `self.assertRaises(NativeCorrectionError); self.assertRaisesRegex(NativeCorrectionError, 'equal image heights')` | unrecorded / — |

## tests.test_native_crop_replay

Actual dual-frame calibrated crop replay, admission, callback and Qt checks.

Source: [tests/test_native_crop_replay.py](../../tests/test_native_crop_replay.py). Category: Recipe, replay and reference infrastructure.

### CropReplayContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_crop_replay.py#L80) | 3 related CropReplayContractTests scenarios, each with private fixture state. | `helper: tests.test_native_crop_replay.CropReplayContractTests.check_test_strict_roi_distinct_frames_and_shifted_equal_shapes: self.assertEqual(context.local.shape, shifted.local.shape)` | unrecorded / — |
### NativeCropStageReplayTests

Fixtures: `setUpClass` at line 90, `tearDownClass` at line 101.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_all_four_calibrated_and_four_diagnostic_graphs_support_crop_replay`](../../tests/test_native_crop_replay.py#L129) | all four calibrated and four diagnostic graphs support crop replay | `assert_stage_inputs_equal(self, recipe, self.recipe, prefix); self.assertEqual(recipe.stage(base.stages[0].stage_id), base.stages[0])` | unrecorded / — |
| [`test_public_crop_cancellation_family_caller_identity_and_mutation_priority`](../../tests/test_native_crop_replay.py#L166) | public crop cancellation family caller identity and mutation priority | `self.assertGreaterEqual(len(calls), 3); self.assertRaises(card.card.AutomaticCardCancelled)` | unrecorded / — |

## tests.test_native_extended_threshold_controls

Pinned control witnesses and desktop routing for the four working thresholds.

Source: [tests/test_native_extended_threshold_controls.py](../../tests/test_native_extended_threshold_controls.py). Category: Recipe, replay and reference infrastructure.

### ExtendedThresholdControlTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_additional_numerical_regressions`](../../tests/test_native_extended_threshold_controls.py#L97) | additional numerical regressions | `helper: tests.test_native_extended_threshold_controls.ExtendedThresholdControlTests.witness: np.testing.assert_array_equal(actual, expected, err_msg=f'{operation} control {base!r} -> {variant!r}')` | unrecorded / — |

## tests.test_native_extended_threshold_recipe

Extended threshold recipe variants, contracts and native execution

Source: [tests/test_native_extended_threshold_recipe.py](../../tests/test_native_extended_threshold_recipe.py). Category: Recipe, replay and reference infrastructure.

### NativeExtendedThresholdRecipeTests

Fixtures: `setUp` at line 67.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_each_recipe_matches_pinned_public_operation_and_returns_cuda_mask`](../../tests/test_native_extended_threshold_recipe.py#L81) | each recipe matches pinned public operation and returns cuda mask | `self.assertEqual(node.status, 'executed', node.reason); self.assertEqual(result.status, 'executed')` | unrecorded / — |
| [`test_cache_controls_source_scope_and_tensor_version_are_recipe_local`](../../tests/test_native_extended_threshold_recipe.py#L111) | cache controls source scope and tensor version are recipe local | `self.assertIs(first.node(operation_id).outputs['mask'], second.node(operation_id).outputs['mask']); self.assertEqual(self.cache.last_computed_nodes, ())` | unrecorded / — |

## tests.test_native_extended_threshold_replay

Pinned public masks, actual archives, and bounded extended-threshold replay.

Source: [tests/test_native_extended_threshold_replay.py](../../tests/test_native_extended_threshold_replay.py). Category: Recipe, replay and reference infrastructure.

### ExtendedThresholdReplayPlanTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_extended_threshold_replay.py#L119) | 4 related ExtendedThresholdReplayPlanTests scenarios, each with private fixture state. | `helper: tests.test_native_extended_threshold_replay.ExtendedThresholdReplayPlanTests.check_test_existing_fourteen_admission_plans_remain_unchanged: self.assertEqual(len(expected), 14)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_extended_threshold_replay.py#L126) | 1 related ExtendedThresholdReplayPlanTests scenarios, each with private fixture state. | `helper: tests.test_native_extended_threshold_replay.ExtendedThresholdReplayPlanTests.check_test_dual_distinct_modes_selected_bytes_and_cuda_boundary: self.assertEqual(same['conversion_modes'], ('RGB',))` | unrecorded / — |
### ExtendedThresholdReplayTests

Fixtures: `setUpClass` at line 134, `tearDownClass` at line 164.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_seven_actual_archives_match_pinned_public_masks_and_select_exact_saved_arrays`](../../tests/test_native_extended_threshold_replay.py#L202) | seven actual archives match pinned public masks and select exact saved arrays | `np.testing.assert_array_equal(saved.node(recipe.stages[1].stage_id).outputs['mask'].cpu().numpy(), expected); self.assertEqual(result.status, 'matched', result.reason)` | unrecorded / — |
| [`test_cancel_and_input_expected_output_mutation_withhold_every_product`](../../tests/test_native_extended_threshold_replay.py#L232) | cancel and input expected output mutation withhold every product | `self.assertTrue(changed); self.assertEqual(result.status, 'failed', result.reason)` | unrecorded / — |

## tests.test_native_image_inputs

Pinned public reader semantics and shared native preparation boundaries.

Source: [tests/test_native_image_inputs.py](../../tests/test_native_image_inputs.py). Category: Recipe, replay and reference infrastructure.

### ImageInputContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_source_pin_and_explicit_mode`](../../tests/test_native_image_inputs.py#L67) | source pin and explicit mode | `self.assertEqual(pin['reader']['source_sha256'], READER_SHA); self.assertEqual(inputs.PLANTCV_READER_SHA256, READER_SHA)` | unrecorded / — |
| [`test_snapshot_identity_size_bounds_and_changed_source`](../../tests/test_native_image_inputs.py#L77) | snapshot identity size bounds and changed source | `self.assertEqual((identity.size, identity.sha256), (len(raw), sha256(raw).hexdigest())); self.assertEqual(inputs.snapshot(path, identity), raw)` | unrecorded / — |
### NativePreparationBoundaryTests

Fixtures: `setUp` at line 94, `tearDown` at line 100.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_additional_numerical_regressions`](../../tests/test_native_image_inputs.py#L141) | additional numerical regressions | `helper: tests.test_native_image_inputs.NativePreparationBoundaryTests.check_test_cache_reuses_pixels_repairs_detached_facts_and_equal_size_replacement: self.assertIs(second.gray, first.gray)` | unrecorded / — |

## tests.test_native_kmeans_recipe

Frozen-centre bindings, public predictions, native caches and Qt execution.

Source: [tests/test_native_kmeans_recipe.py](../../tests/test_native_kmeans_recipe.py). Category: Recipe, replay and reference infrastructure.

### NativeKMeansRecipeTests

Fixtures: `setUpClass` at line 70, `setUp` at line 81, `tearDown` at line 92.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_public_bgr_and_gray_labels_masks_and_prepared_cache_reuse`](../../tests/test_native_kmeans_recipe.py#L100) | public bgr and gray labels masks and prepared cache reuse | `self.assertEqual(result.status, 'executed', [(n.stage_id, n.reason) for n in result.nodes]); np.testing.assert_array_equal(actual.cpu().numpy(), expected)` | unrecorded / — |
| [`test_category_edit_and_nested_mutation_recompute_only_mask_node`](../../tests/test_native_kmeans_recipe.py#L120) | category edit and nested mutation recompute only mask node | `self.assertEqual(self.cache.last_computed_nodes, (factory.MASK_STAGE_ID,)); self.assertEqual(selected.node(factory.MASK_STAGE_ID).outputs['class_names'], ('selected',))` | unrecorded / — |

## tests.test_native_kmeans_training

Real directory training, CUDA row caching and asynchronous desktop lifecycle.

Source: [tests/test_native_kmeans_training.py](../../tests/test_native_kmeans_training.py). Category: Recipe, replay and reference infrastructure.

### NativeKMeansTrainingTests

Fixtures: `setUp` at line 36.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_kmeans_training.py#L93) | 2 related NativeKMeansTrainingTests scenarios, each with private fixture state. | `helper: tests.test_native_kmeans_training.NativeKMeansTrainingTests.check_test_bad_parameters_and_retained_rows_obey_effective_bounds: self.assertRaises(ValueError)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_kmeans_training.py#L98) | 1 related NativeKMeansTrainingTests scenarios, each with private fixture state. | `helper: tests.test_native_kmeans_training.NativeKMeansTrainingTests.check_test_calculation_identity_rejects_changed_loaded_producer_contract: self.assertEqual(len(identity), 64)` | unrecorded / — |

## tests.test_native_local_stage_replay

Saved calibrated-ROI replay: exact frames, ancestors and bounded transfers.

Source: [tests/test_native_local_stage_replay.py](../../tests/test_native_local_stage_replay.py). Category: Recipe, replay and reference infrastructure.


## tests.test_native_mask_stage

Native mask-stage checkpoints and output identity

Source: [tests/test_native_mask_stage.py](../../tests/test_native_mask_stage.py). Category: Recipe, replay and reference infrastructure.

### NativeMaskIdentityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_loaded_implementation_identity_is_stable_and_rejects_replaced_code`](../../tests/test_native_mask_stage.py#L136) | loaded implementation identity is stable and rejects replaced code | `self.assertRegex(first, '^[0-9a-f]{64}$'); self.assertEqual(first, native_stage_module._implementation_digest())` | unrecorded / — |
### NativeMaskCudaStageTests

Fixtures: `setUpClass` at line 148, `setUp` at line 154, `tearDown` at line 167.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_original_reference_fill_recipe_is_not_mislabelled_as_native`](../../tests/test_native_mask_stage.py#L259) | original reference fill recipe is not mislabelled as native | `self.assertEqual(outcome.status, NativeMaskStageStatus.UNSUPPORTED); self.assertEqual(outcome.run_record.state, StageRunState.UNSUPPORTED)` | unrecorded / — |

## tests.test_native_model_replay

Archived model binding and shared native/replay resource admission.

Source: [tests/test_native_model_replay.py](../../tests/test_native_model_replay.py). Category: Recipe, replay and reference infrastructure.

### ModelReplayContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_model_replay.py#L197) | 7 related ModelReplayContractTests scenarios, each with private fixture state. | `helper: tests.test_native_model_replay.ModelReplayContractTests.check_test_bayes_archived_bytes_parse_with_all_origin_io_forbidden: self.assertEqual(actual, model)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_model_replay.py#L207) | 3 related ModelReplayContractTests scenarios, each with private fixture state. | `helper: tests.test_native_model_replay.ModelReplayContractTests.check_test_identity_selection_header_and_origin_tampering_rejects_before_parse: self.assertRaises(native.NativeModelReplayError)` | unrecorded / — |
### ModelReplayPlanTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_model_replay.py#L346) | 4 related ModelReplayPlanTests scenarios, each with private fixture state. | `helper: tests.test_native_model_replay.ModelReplayPlanTests.check_test_bayes_plan_charges_bgr_expected_masks_tables_scratch_comparison_and_audit: self.assertEqual(plan.selected_bytes, (3 + c) * p + model.source_bytes)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_model_replay.py#L353) | 2 related ModelReplayPlanTests scenarios, each with private fixture state. | `helper: tests.test_native_model_replay.ModelReplayPlanTests.check_test_exact_selected_and_cuda_cap_boundaries_reject_without_allocations: self.assertEqual(function(*args, selected_limit=plan.selected_bytes, cuda_limit=plan.cuda_live_bytes), plan)` | unrecorded / — |
### SharedKMeansShapePlanTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_pure_shape_plan_preserves_unique_centres_and_pass_counts`](../../tests/test_native_model_replay.py#L360) | pure shape plan preserves unique centres and pass counts | `self.assertEqual(plan.work, gaussian_work + 2 + 77 * (1 + 2 * passes)); self.assertEqual(plan.tile_pixels, 5)` | unrecorded / — |
| [`test_shape_admission_checks_padded_domain_controls_and_classifier_work`](../../tests/test_native_model_replay.py#L371) | shape admission checks padded domain controls and classifier work | `self.assertRaises(ValueError); self.assertRaises(ValueError)` | unrecorded / — |
### SharedKMeansCudaPlanTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_shape_and_real_cuda_input_plans_are_identical_for_gray_rgb_odd_even_and_singleton`](../../tests/test_native_model_replay.py#L388) | shape and real cuda input plans are identical for gray rgb odd even and singleton | `self.assertEqual(predict.preflight_kmeans_prediction(image, model), predict.preflight_kmeans_prediction_shape(*shape, model))` | unrecorded / — |

## tests.test_native_model_stage_replay

Portable model/class-map replay, lineage, guards and actual classifier Qt.

Source: [tests/test_native_model_stage_replay.py](../../tests/test_native_model_stage_replay.py). Category: Recipe, replay and reference infrastructure.

### ModelStageReplayTests

Fixtures: `setUpClass` at line 42, `tearDownClass` at line 65.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_bayes_every_mask_names_and_model_metadata_match_with_exact_selection`](../../tests/test_native_model_stage_replay.py#L90) | bayes every mask names and model metadata match with exact selection | `self.assertEqual(result.status, 'matched', result.reason); self.assertEqual(calls, [BAYES])` | unrecorded / — |
| [`test_gray_rgb_odd_even_and_both_kmeans_formats_select_only_active_image`](../../tests/test_native_model_stage_replay.py#L105) | gray rgb odd even and both kmeans formats select only active image | `self.assertEqual(result.status, 'matched', result.reason); self.assertEqual(calls, [stage])` | unrecorded / — |

## tests.test_native_naive_bayes_directory_training_dialog

Directory training, strict portable provenance, and actual Qt/project saves.

Source: [tests/test_native_naive_bayes_directory_training_dialog.py](../../tests/test_native_naive_bayes_directory_training_dialog.py). Category: Recipe, replay and reference infrastructure.


## tests.test_native_naive_bayes_plot_dialog

Actual async compact plots, model-first saves and full modal lifecycle.

Source: [tests/test_native_naive_bayes_plot_dialog.py](../../tests/test_native_naive_bayes_plot_dialog.py). Category: Recipe, replay and reference infrastructure.

### NativeBayesPlotDialogTests

Fixtures: `setUpClass` at line 41, `setUp` at line 44.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_render_errors_and_invalid_bundle_write_nothing_and_allow_model_only_retry`](../../tests/test_native_naive_bayes_plot_dialog.py#L75) | render errors and invalid bundle write nothing and allow model only retry | `self.assertFalse(target.exists()); self.assertIn('no files saved', dialog.status_label.text())` | unrecorded / — |
| [`test_reserved_output_and_file_dialog_reentrancy_cannot_replace_model_with_svg`](../../tests/test_native_naive_bayes_plot_dialog.py#L96) | reserved output and file dialog reentrancy cannot replace model with svg | `self.assertIsNone(dialog._worker); self.assertFalse((self.root / name).exists())` | unrecorded / — |

## tests.test_native_naive_bayes_recipe

Pinned class masks, asset bindings, cache invalidation and desktop execution.

Source: [tests/test_native_naive_bayes_recipe.py](../../tests/test_native_naive_bayes_recipe.py). Category: Recipe, replay and reference infrastructure.

### NativeNaiveBayesRecipeTests

Fixtures: `setUp` at line 58, `tearDown` at line 69.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_native_naive_bayes_recipe.py#L138) | 2 related NativeNaiveBayesRecipeTests scenarios, each with private fixture state. | `helper: tests.test_native_naive_bayes_recipe.NativeNaiveBayesRecipeTests.check_test_actual_class_masks_match_public_call_and_nested_mutation_recomputes_only_classifier: self.assertEqual(first.status, 'executed', [(n.stage_id, n.reason) for n in first.nodes])` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_native_naive_bayes_recipe.py#L143) | 2 related NativeNaiveBayesRecipeTests scenarios, each with private fixture state. | `helper: tests.test_native_naive_bayes_recipe.NativeNaiveBayesRecipeTests.check_test_reselecting_changed_model_preserves_prepared_image_and_image_node: self.assertNotEqual(source['source_sha256'], self.source['source_sha256'])` | unrecorded / — |

## tests.test_native_naive_bayes_training

Public sampled-RGB trainer, TSV handoff and frozen-model prediction gates.

Source: [tests/test_native_naive_bayes_training.py](../../tests/test_native_naive_bayes_training.py). Category: Recipe, replay and reference infrastructure.

### BayesSampleParsingTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_native_naive_bayes_training.py#L116) | 2 related BayesSampleParsingTests scenarios, each with private fixture state. | `helper: tests.test_native_naive_bayes_training.BayesSampleParsingTests.check_test_duplicate_headers_quoted_cells_whitespace_missing_and_extra_empty_columns: self.assertEqual(table.headers, ('"seed"', 'soil', '"seed"'))` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_native_naive_bayes_training.py#L121) | 1 related BayesSampleParsingTests scenarios, each with private fixture state. | `helper: tests.test_native_naive_bayes_training.BayesSampleParsingTests.check_test_malformed_uint8_and_unsupported_empty_labels_are_explicit: self.assertRaises(BayesTrainingError)` | unrecorded / — |
### NativeBayesSampleTrainingTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_final_high_level_callbacks_guard_samples_pdf_storage_and_raised_callbacks`](../../tests/test_native_naive_bayes_training.py#L132) | final high level callbacks guard samples pdf storage and raised callbacks | `self.assertRaisesRegex(BayesTrainingError, 'changed before publication')` | unrecorded / — |

## tests.test_native_naive_bayes_training_dialog

Actual Bayes training UI, atomic TSV saves and project-held provenance.

Source: [tests/test_native_naive_bayes_training_dialog.py](../../tests/test_native_naive_bayes_training_dialog.py). Category: Recipe, replay and reference infrastructure.

### NativeBayesTrainingWorkflowTests

Fixtures: `setUpClass` at line 89, `setUp` at line 92, `tearDown` at line 108.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_naive_bayes_training_dialog.py#L328) | 5 related NativeBayesTrainingWorkflowTests scenarios, each with private fixture state. | `helper: tests.test_native_naive_bayes_training_dialog.NativeBayesTrainingWorkflowTests.check_test_legacy_and_trained_selections_roundtrip_exact_tsv_and_independent_metadata: self.assertEqual(set(plain), set(factory.MODEL_FIELDS))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_naive_bayes_training_dialog.py#L336) | 3 related NativeBayesTrainingWorkflowTests scenarios, each with private fixture state. | `helper: tests.test_native_naive_bayes_training_dialog.NativeBayesTrainingWorkflowTests.check_test_typed_provenance_rejects_mismatched_model_counts_profiles_and_calculation_facts: self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_native_paper_replay

Actual paper saved-input replay, primitive schema/admission and Qt workflow.

Source: [tests/test_native_paper_replay.py](../../tests/test_native_paper_replay.py). Category: Recipe, replay and reference infrastructure.

### PaperReplayContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_paper_replay.py#L115) | 3 related PaperReplayContractTests scenarios, each with private fixture state. | `helper: tests.test_native_paper_replay.PaperReplayContractTests.check_test_schema_preserves_tuple_types_ids_bounds_and_strict_integer_fields: self.assertEqual(contracts.inventory(value, (2, 3)), rows)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_paper_replay.py#L121) | 1 related PaperReplayContractTests scenarios, each with private fixture state. | `helper: tests.test_native_paper_replay.PaperReplayContractTests.check_test_allocation_free_operation_limits_alias_facts_and_exact_cap: self.assertGreaterEqual(result['cuda_bytes'], 60 + contracts.AUDIT_CUDA_BYTES)` | unrecorded / — |
### PaperStageReplayTests

Fixtures: `setUpClass` at line 129, `tearDownClass` at line 147.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_all_nine_stage_instances_replay_only_selected_operation_and_preserve_archive`](../../tests/test_native_paper_replay.py#L157) | all nine stage instances replay only selected operation and preserve archive | `self.assertEqual(result.status, 'matched', result.reason); self.assertEqual(calls, [saved.node(stage).operation_id])` | unrecorded / — |

## tests.test_native_preprocessing

Native preprocessing chains, caching and public PlantCV parity

Source: [tests/test_native_preprocessing.py](../../tests/test_native_preprocessing.py). Category: Recipe, replay and reference infrastructure.

### NativePreprocessingContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_preprocessing.py#L584) | 5 related NativePreprocessingContractTests scenarios, each with private fixture state. | `helper: tests.test_native_preprocessing.NativePreprocessingContractTests.check_test_layout_is_display_only_and_stage_projections_are_exact: self.assertEqual(outcome.status, NativePreprocessingStatus.EXECUTED)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_preprocessing.py#L592) | 7 related NativePreprocessingContractTests scenarios, each with private fixture state. | `helper: tests.test_native_preprocessing.NativePreprocessingContractTests.check_test_precancelled_chain_records_cancellation_without_calling_a_bridge: self.assertEqual(outcome.status, NativePreprocessingStatus.CANCELLED)` | unrecorded / — |
### NativePreprocessingPlantCVParityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_native_binary_mean_and_restricted_otsu_chains_match_original_stage_pixels`](../../tests/test_native_preprocessing.py#L605) | native binary mean and restricted otsu chains match original stage pixels | `self.assertIn(native.status, {NativePreprocessingStatus.EXECUTED, NativePreprocessingStatus.EMPTY}); self.assertIsNotNone(native.final_checkpoint)` | unrecorded / — |

## tests.test_native_range_recipe

Public range outputs through real native, archive, Qt and batch paths.

Source: [tests/test_native_range_recipe.py](../../tests/test_native_range_recipe.py). Category: Recipe, replay and reference infrastructure.

### RangeContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_separate_canonical_recipe_roundtrip_preserves_reference_and_extended_graphs`](../../tests/test_native_range_recipe.py#L47) | separate canonical recipe roundtrip preserves reference and extended graphs | `self.assertEqual(RecipeSpec.from_json(recipe.to_json()), recipe); self.assertEqual(persistence.native_recipes_from_payload(persistence.native_recipes_to_payload((recipe,))), (recipe,))` | unrecorded / — |
| [`test_shared_admission_and_exact_replay_resource_boundaries`](../../tests/test_native_range_recipe.py#L64) | shared admission and exact replay resource boundaries | `self.assertEqual(plan['cuda_bytes'], factory.retained_bound(887 * 887)); self.assertEqual(plan['work_details']['primitive_output_bytes'], 4 * 887 * 887)` | unrecorded / — |
### RangeExecutionTests

Fixtures: `setUpClass` at line 86, `tearDownClass` at line 109.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_additional_numerical_regressions`](../../tests/test_native_range_recipe.py#L156) | additional numerical regressions | `helper: tests.test_native_range_recipe.RangeExecutionTests.check_test_fifteen_mode_endpoint_and_reversed_bound_cases_match_both_public_outputs: self.assertEqual(run.status, 'executed', [(n.status, n.reason) for n in run.nodes])` | unrecorded / — |

## tests.test_native_reader_recipe

Pinned codec grayscale and versioned desktop/project reader selection.

Source: [tests/test_native_reader_recipe.py](../../tests/test_native_reader_recipe.py). Category: Recipe, replay and reference infrastructure.

### ReaderContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_native_reader_recipe.py#L117) | 2 related ReaderContractTests scenarios, each with private fixture state. | `helper: tests.test_native_reader_recipe.ReaderContractTests.check_test_all_24_ids_have_additive_reader_versions_and_calibrated_controls_reject: self.assertEqual(len(reader.BASE_IDS), 24)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_native_reader_recipe.py#L122) | 1 related ReaderContractTests scenarios, each with private fixture state. | `helper: tests.test_native_reader_recipe.ReaderContractTests.check_test_all_24_action_orders_preserve_selected_parameters_layout_and_project_versions: self.assertEqual(reader.selected_mode(recipe), 'codec-gray')` | unrecorded / — |
### ReaderIntegrationTests

Fixtures: `setUp` at line 130, `tearDown` at line 136.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_reader_recipe.py#L228) | 3 related ReaderIntegrationTests scenarios, each with private fixture state. | `helper: tests.test_native_reader_recipe.ReaderIntegrationTests.check_test_mode_switch_reuses_rgb_same_mode_reuses_nodes_and_roi_edits_reuse_source: self.assertEqual(calls, [inputs.GRAY_DECODE_FLAGS])` | unrecorded / — |

## tests.test_native_recipe

Native recipe execution, cache behavior, cancellation and results

Source: [tests/test_native_recipe.py](../../tests/test_native_recipe.py). Category: Recipe, replay and reference infrastructure.

### NativeRecipeTests

Fixtures: `setUp` at line 139.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_native_recipe.py#L336) | 4 related NativeRecipeTests scenarios, each with private fixture state. | `helper: tests.test_native_recipe.NativeRecipeTests.check_test_cached_lab_table_mutation_is_rejected_before_node_execution: self.assertRaisesRegex(NativeRecipeError, 'Cached LAB conversion table')` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_native_recipe.py#L371) | 2 related NativeRecipeTests scenarios, each with private fixture state. | `helper: tests.test_native_recipe.NativeRecipeTests.check_test_real_canonical_graph_matches_pinned_public_preprocessing_watershed_and_color: self.assertEqual(result.recipe_identity, selected_recipe().identity)` | unrecorded / — |

## tests.test_native_recipe_dialog

Native recipe dialog selection, execution and result state

Source: [tests/test_native_recipe_dialog.py](../../tests/test_native_recipe_dialog.py). Category: Recipe, replay and reference infrastructure.

### NativeRecipeDialogTests

Fixtures: `setUpClass` at line 19.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_selected_parameters_round_trip_without_changing_historical_graph`](../../tests/test_native_recipe_dialog.py#L32) | selected parameters round trip without changing historical graph | `self.assertEqual(dialog.recipe.parameters_for_runner(stage)['distance'], 12); self.assertNotEqual(dialog.recipe.identity.settings_sha256, recipe.identity.settings_sha256)` | unrecorded / — |
| [`test_size_diagnostic_cap_retains_visible_measurements_and_clears_preview`](../../tests/test_native_recipe_dialog.py#L57) | size diagnostic cap retains visible measurements and clears preview | `self.assertTrue(cv2.imwrite(str(path), np.repeat(gray[..., None], 3, axis=2))); self.assertIsNotNone(dialog.result, dialog.status_label.text())` | unrecorded / — |
| [`test_close_requests_cancellation_and_waits_for_worker_to_drain`](../../tests/test_native_recipe_dialog.py#L87) | close requests cancellation and waits for worker to drain | `self.assertTrue(release.wait(10)); self.assertTrue(dialog.isVisible())` | unrecorded / — |

## tests.test_native_recovery_replay

Pure recovery contracts, maximum inventory and allocation-free admission.

Source: [tests/test_native_recovery_replay.py](../../tests/test_native_recovery_replay.py). Category: Recipe, replay and reference infrastructure.

### RecoveryReplayContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_recovery_replay.py#L144) | 5 related RecoveryReplayContractTests scenarios, each with private fixture state. | `helper: tests.test_native_recovery_replay.RecoveryReplayContractTests.check_test_primitive_decoder_preserves_types_and_exact_byte_depth_item_caps: self.assertEqual(contracts.decode(encode(value)), contracts.canonical(value))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_recovery_replay.py#L152) | 2 related RecoveryReplayContractTests scenarios, each with private fixture state. | `helper: tests.test_native_recovery_replay.RecoveryReplayContractTests.check_test_inventory_maximum_sparse_ids_and_downstream_limit_are_distinct: self.assertEqual(len(decoded['objects']), 32768)` | unrecorded / — |

## tests.test_native_recovery_stage_replay

Actual saved recovery inputs, compact contracts, guards and desktop replay.

Source: [tests/test_native_recovery_stage_replay.py](../../tests/test_native_recovery_stage_replay.py). Category: Recipe, replay and reference infrastructure.

### RecoveryStageReplayTests

Fixtures: `setUpClass` at line 33, `tearDownClass` at line 49.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_operations_match_every_product_select_direct_arrays_and_preserve_archives`](../../tests/test_native_recovery_stage_replay.py#L60) | operations match every product select direct arrays and preserve archives | `self.assertEqual(result.status, 'matched', result.reason); self.assertEqual(calls, [saved.node(stage).operation_id])` | unrecorded / — |

## tests.test_native_result_export

Actual native product persistence and saved-project batch bindings.

Source: [tests/test_native_result_export.py](../../tests/test_native_result_export.py). Category: Recipe, replay and reference infrastructure.

### NativeResultExportTests

Fixtures: `setUp` at line 47, `tearDown` at line 57.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_project_graph_cancel_or_changed_source_never_publishes`](../../tests/test_native_result_export.py#L85) | project graph cancel or changed source never publishes | `self.assertRaises(ValueError); self.assertRaises(NativeRecipeCancelled)` | unrecorded / — |
### NativeMetadataTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_ordered_mappings_tuples_and_nonfinite_values_are_unambiguous`](../../tests/test_native_result_export.py#L101) | ordered mappings tuples and nonfinite values are unambiguous | `self.assertEqual([name for name, _ in encoded['items']], ['kind', 'second']); self.assertEqual([value['value'] for value in values], ['nan', 'positive_infinity', 'negative_infinity'])` | unrecorded / — |
| [`test_metadata_depth_cycles_cpu_tensors_and_visit_caps_reject`](../../tests/test_native_result_export.py#L113) | metadata depth cycles cpu tensors and visit caps reject | `self.assertRaises(exports.NativeResultExportError); self.assertRaises(exports.NativeResultExportError)` | unrecorded / — |
| [`test_ambiguous_layout_and_unversioned_tensors_are_explicit`](../../tests/test_native_result_export.py#L128) | ambiguous layout and unversioned tensors are explicit | `self.assertNotIn(binding['name'], collector.roles); self.assertRaises(exports.NativeResultExportError)` | unrecorded / — |

## tests.test_native_seed_working_recipe

Complete native seed working workflows

Source: [tests/test_native_seed_working_recipe.py](../../tests/test_native_seed_working_recipe.py). Category: Recipe, replay and reference infrastructure.

### NativeWorkingSeedContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_working_copy_adds_only_effective_size_control_and_preserves_history`](../../tests/test_native_seed_working_recipe.py#L29) | working copy adds only effective size control and preserves history | `self.assertEqual(recipe.recipe_id, selected.recipe_id + '_working'); self.assertEqual(RecipeSpec.from_json(recipe.to_json()), recipe)` | unrecorded / — |
### NativeWorkingSeedCalculationTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_additional_numerical_regressions`](../../tests/test_native_seed_working_recipe.py#L76) | additional numerical regressions | `helper: tests.test_native_seed_working_recipe.NativeWorkingSeedCalculationTests.check_test_changed_thickness_recomputes_only_size_and_preserves_cuda_upstream_products: self.assertEqual(first.status, 'executed')` | unrecorded / — |

## tests.test_native_size_replay

Public size products, bounded typed replay, physical units and Qt state.

Source: [tests/test_native_size_replay.py](../../tests/test_native_size_replay.py). Category: Recipe, replay and reference infrastructure.

### SizeReplayContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_size_replay.py#L134) | 3 related SizeReplayContractTests scenarios, each with private fixture state. | `helper: tests.test_native_size_replay.SizeReplayContractTests.check_test_exact_nominal_datatypes_zero_bundle_immutable_decoder_and_caps: self.assertEqual(decoded['graph_1']['area']['datatype'], "<class 'int'>")` | unrecorded / — |
### SizeReplayTests

Fixtures: `setUpClass` at line 144, `tearDownClass` at line 158.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_controlled_diagnostic_omission_and_numerical_unavailability_classification`](../../tests/test_native_size_replay.py#L173) | controlled diagnostic omission and numerical unavailability classification | `self.assertEqual(result.status, 'matched', result.reason); self.assertEqual(set(result.outputs), {'observations'})` | unrecorded / — |

## tests.test_native_smpts_recipe

Native soybean paper workflow composition and outputs

Source: [tests/test_native_smpts_recipe.py](../../tests/test_native_smpts_recipe.py). Category: Recipe, replay and reference infrastructure.

### PaperRecipeExecutionTests

Fixtures: `setUp` at line 45.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_executable_graph_exact_outputs_crop_lineage_and_node_local_reuse`](../../tests/test_native_smpts_recipe.py#L57) | executable graph exact outputs crop lineage and node local reuse | `self.assertEqual(actual.status, 'executed', [(n.stage_id, n.status, n.reason) for n in actual.nodes]); self.assertTrue(torch.equal(actual.labels, expected['final_labels']))` | unrecorded / — |
| [`test_export_and_saved_project_batch_preserve_paper_provenance`](../../tests/test_native_smpts_recipe.py#L84) | export and saved project batch preserve paper provenance | `self.assertEqual(result.status, 'executed'); self.assertEqual(report['recipe']['provenance']['mode'], 'paper_reconstruction')` | unrecorded / — |

## tests.test_native_stage_replay

Actual saved-input CUDA stage replay, lineage and unavailable outcomes.

Source: [tests/test_native_stage_replay.py](../../tests/test_native_stage_replay.py). Category: Recipe, replay and reference infrastructure.

### NativeStageReplayPlanTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_native_stage_replay.py#L94) | 3 related NativeStageReplayPlanTests scenarios, each with private fixture state. | `helper: tests.test_native_stage_replay.NativeStageReplayPlanTests.check_test_cuda_envelope_boundary_full_fixture_and_uhd_scope: self.assertEqual(maximum, 7340032)` | unrecorded / — |
### NativeStageReplayTests

Fixtures: `setUpClass` at line 105, `tearDownClass` at line 120, `setUp` at line 126, `tearDown` at line 132.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_stage_replay.py#L303) | 4 related NativeStageReplayTests scenarios, each with private fixture state. | `helper: tests.test_native_stage_replay.NativeStageReplayTests.check_test_actual_raster_stage_replays_dispatch_only_target_preserve_archives_and_cuda: self.assertEqual(result.status, 'matched', result.reason)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_stage_replay.py#L310) | 2 related NativeStageReplayTests scenarios, each with private fixture state. | `helper: tests.test_native_stage_replay.NativeStageReplayTests.check_test_admission_cancellation_and_callback_tensor_mutation_withhold_outputs: self.assertEqual(result.status, 'failed', result.reason)` | unrecorded / — |

## tests.test_native_threshold_stage

Native threshold stage inputs, outputs, resources and provenance

Source: [tests/test_native_threshold_stage.py](../../tests/test_native_threshold_stage.py). Category: Recipe, replay and reference infrastructure.

### NativeThresholdContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_output_metadata_limit_is_checked_before_checkpoint_host_transfer`](../../tests/test_native_threshold_stage.py#L389) | output metadata limit is checked before checkpoint host transfer | `helper: tests.test_native_threshold_stage.NativeThresholdContractTests.check_test_output_metadata_limit_is_checked_before_checkpoint_host_transfer: self.assertRaisesRegex(NativeThresholdStageError, 'metadata exceeds')` | unrecorded / — |
| [`test_cuda_runtime_failure_is_recorded_without_an_output_checkpoint`](../../tests/test_native_threshold_stage.py#L392) | cuda runtime failure is recorded without an output checkpoint | `helper: tests.test_native_threshold_stage.NativeThresholdContractTests.check_test_cuda_runtime_failure_is_recorded_without_an_output_checkpoint: self.assertEqual(outcome.run_record.state, StageRunState.FAILED)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_native_threshold_stage.py#L395) | 4 related NativeThresholdContractTests scenarios, each with private fixture state. | `helper: tests.test_native_threshold_stage.NativeThresholdContractTests.check_test_loaded_code_hash_refuses_same_size_source_edit_after_import: self.assertRegex(baseline, '^[0-9a-f]{64}$')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_threshold_stage.py#L402) | 5 related NativeThresholdContractTests scenarios, each with private fixture state. | `helper: tests.test_native_threshold_stage.NativeThresholdContractTests.check_test_stale_source_frame_upstream_recipe_and_role_bindings_reject_before_execution: self.assertRaisesRegex(NativeThresholdStageError, 'source SHA')` | unrecorded / — |

## tests.test_native_watershed_diagnostic_recipe

Watershed diagnostic recipe variants and CUDA products

Source: [tests/test_native_watershed_diagnostic_recipe.py](../../tests/test_native_watershed_diagnostic_recipe.py). Category: Recipe, replay and reference infrastructure.

### WatershedDiagnosticRecipeContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_joint_admission_is_explicit_and_full_committed_frame_fits`](../../tests/test_native_watershed_diagnostic_recipe.py#L41) | joint admission is explicit and full committed frame fits | `self.assertLessEqual(bound, runner.MAX_RETAINED_CUDA_BYTES); self.assertRaises(runner.NativeRecipeError)` | unrecorded / — |
### WatershedDiagnosticRecipeCalculationTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_all_six_actual_graphs_draw_exact_public_pixels_in_their_source_or_roi_frame`](../../tests/test_native_watershed_diagnostic_recipe.py#L65) | all six actual graphs draw exact public pixels in their source or roi frame | `self.assertEqual(diagnostic.status, 'executed', diagnostic.reason); self.assertEqual(tuple(diagnostic.outputs['diagnostic'].shape[:2]), result.labels_coordinate_frame.shape)` | unrecorded / — |

## tests.test_native_watershed_diagnostic_replay

Saved watershed diagnostic stage replay and guards

Source: [tests/test_native_watershed_diagnostic_replay.py](../../tests/test_native_watershed_diagnostic_replay.py). Category: Recipe, replay and reference infrastructure.

### DiagnosticReplayPlanTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_watershed_diagnostic_replay.py#L88) | 2 related DiagnosticReplayPlanTests scenarios, each with private fixture state. | `helper: tests.test_native_watershed_diagnostic_replay.DiagnosticReplayPlanTests.check_test_new_plan_admits_full_fixture_and_preserves_old_contract_envelopes: self.assertEqual(plan['cuda_bytes'], max(128 * pixels + 128 * 1024 * 1024, 13 * pixels + 256 * 1024…` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_watershed_diagnostic_replay.py#L93) | 1 related DiagnosticReplayPlanTests scenarios, each with private fixture state. | `helper: tests.test_native_watershed_diagnostic_replay.DiagnosticReplayPlanTests.check_test_decoder_counts_keys_depth_bytes_and_rejects_nonprimitive_tags: self.assertRaises(ValueError)` | unrecorded / — |
### DiagnosticReplayTests

Fixtures: `setUpClass` at line 101, `tearDownClass` at line 122.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_palette_maximum_and_nested_compact_immutability`](../../tests/test_native_watershed_diagnostic_replay.py#L148) | palette maximum and nested compact immutability | `self.assertEqual(len(frozen['palette_source_triples']), 4097); self.assertRaises(TypeError)` | unrecorded / — |

## tests.test_native_watershed_replay_integration

Typed saved-stage resource, cancellation and actual asynchronous Qt checks.

Source: [tests/test_native_watershed_replay_integration.py](../../tests/test_native_watershed_replay_integration.py). Category: Recipe, replay and reference infrastructure.

### TypedReplayPlanTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_all_five_selected_byte_and_cuda_boundaries_preserve_full_fixture_admission`](../../tests/test_native_watershed_replay_integration.py#L26) | all five selected byte and cuda boundaries preserve full fixture admission | `self.assertEqual(plan['selected_limit_bytes'], 128 * 1024 * 1024); self.assertIsNone(plan['work_limit'])` | unrecorded / — |

## tests.test_native_watershed_stage

Native watershed stage execution and checkpoints

Source: [tests/test_native_watershed_stage.py](../../tests/test_native_watershed_stage.py). Category: Recipe, replay and reference infrastructure.

### NativeWatershedIdentityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_native_watershed_stage.py#L291) | 2 related NativeWatershedIdentityTests scenarios, each with private fixture state. | `helper: tests.test_native_watershed_stage.NativeWatershedIdentityTests.check_test_loaded_distance_and_marker_implementation_identity_is_stable: self.assertRegex(first, '^[0-9a-f]{64}$')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_native_watershed_stage.py#L296) | 1 related NativeWatershedIdentityTests scenarios, each with private fixture state. | `helper: tests.test_native_watershed_stage.NativeWatershedIdentityTests.check_test_archive_preflight_rejects_before_cuda_probe_or_primitive: self.assertRaisesRegex(NativeWatershedStageError, 'max_npz_bytes')` | unrecorded / — |
### NativeWatershedCudaStageTests

Fixtures: `setUpClass` at line 307, `setUp` at line 312, `tearDown` at line 317.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_original_reference_recipe_is_never_treated_as_a_native_internal_replay`](../../tests/test_native_watershed_stage.py#L413) | original reference recipe is never treated as a native internal replay | `self.assertEqual(outcome.status, NativeWatershedStageStatus.UNSUPPORTED); self.assertEqual(outcome.run_record.state, StageRunState.UNSUPPORTED)` | unrecorded / — |

## tests.test_native_watershed_stage_replay

Exact saved watershed products and typed source/local archive contracts.

Source: [tests/test_native_watershed_stage_replay.py](../../tests/test_native_watershed_stage_replay.py). Category: Recipe, replay and reference infrastructure.


## tests.test_neutral_calibration

Selected grid references, CUDA sampling, cache boundaries and saved facts.

Source: [tests/test_neutral_calibration.py](../../tests/test_neutral_calibration.py). Category: Image analysis, calibration and rendering.

### NeutralCalibrationTests

Fixtures: `setUp` at line 39.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_neutral_calibration.py#L220) | 8 related NeutralCalibrationTests scenarios, each with private fixture state. | `helper: tests.test_neutral_calibration.NeutralCalibrationTests.check_test_default_source_mode_preserves_legacy_gain_and_pixel_arithmetic: np.testing.assert_array_equal(result.channel_gains_bgr, tuple((float(x) for x in gains)))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_neutral_calibration.py#L231) | 7 related NeutralCalibrationTests scenarios, each with private fixture state. | `helper: tests.test_neutral_calibration.NeutralCalibrationTests.check_test_missing_selection_has_no_partial_or_automatic_fallback: self.assertEqual(result.neutral_balance.state, 'unavailable')` | unrecorded / — |
### NeutralSettingsTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_neutral_calibration.py#L342) | 6 related NeutralSettingsTests scenarios, each with private fixture state. | `helper: tests.test_neutral_calibration.NeutralSettingsTests.check_test_selection_parser_is_shared_canonical_and_strict: self.assertEqual(normalize_neutral_patch_selection(raw), expected)` | unrecorded / — |

## tests.test_node_optimization

Node parameter search, optimizer state and cache reuse

Source: [tests/test_node_optimization.py](../../tests/test_node_optimization.py). Category: Pipeline workspace and controls.

### OptimizationContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_node_optimization.py#L183) | 7 related OptimizationContractTests scenarios, each with private fixture state. | `helper: tests.test_node_optimization.OptimizationContractTests.check_test_every_card_and_control_has_a_disposition: self.assertEqual(set(nodes), set(OBJECTIVES) \| set(EXEMPT))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_node_optimization.py#L193) | 3 related OptimizationContractTests scenarios, each with private fixture state. | `helper: tests.test_node_optimization.OptimizationContractTests.check_test_edge_controls_invalidate_true_prototype_owner: self.assertIn('reference_texture_prototypes', affected)` | unrecorded / — |
### OptimizationReferenceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_node_optimization.py#L331) | 9 related OptimizationReferenceTests scenarios, each with private fixture state. | `helper: tests.test_node_optimization.OptimizationReferenceTests.check_test_only_complete_reviewed_connected_shapes_are_eligible: self.assertEqual(set(np.unique(result)), {0, 1})` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_node_optimization.py#L343) | 1 related OptimizationReferenceTests scenarios, each with private fixture state. | `helper: tests.test_node_optimization.OptimizationReferenceTests.check_test_failed_candidate_cache_is_discarded: self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_node_optimization_production

Production integration on two generated images; no accuracy claims.

Source: [tests/test_node_optimization_production.py](../../tests/test_node_optimization_production.py). Category: Pipeline workspace and controls.

### ProductionOptimizationTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_ui_samples_fit_local_references_without_a_library_pin`](../../tests/test_node_optimization_production.py#L21) | One bounded real UI -> frozen samples -> production fit witness. | `self.assertTrue(cv2.imwrite(str(path), image)); self.assertEqual(len(samples), 2)` | unrecorded / — |

## tests.test_node_optimization_ui

Node optimizer controls and desktop result application

Source: [tests/test_node_optimization_ui.py](../../tests/test_node_optimization_ui.py). Category: Pipeline workspace and controls.

### NodeOptimizationUITests

Fixtures: `setUpClass` at line 21.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_node_optimization_ui.py#L257) | 6 related NodeOptimizationUITests scenarios, each with private fixture state. | `helper: tests.test_node_optimization_ui.NodeOptimizationUITests.check_test_all_computational_cards_offer_shared_action: self.assertEqual(not inspector.optimization_button.isHidden(), bool(capability(node).searchable))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_node_optimization_ui.py#L266) | 2 related NodeOptimizationUITests scenarios, each with private fixture state. | `helper: tests.test_node_optimization_ui.NodeOptimizationUITests.check_test_completed_progress_does_not_cancel_success_and_apply_is_atomic: self.assertFalse(task.stop.is_set())` | unrecorded / — |

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

## tests.test_otsu_calculation_facts

Exact binary domains and single-pass Otsu cutoff facts through native execution.

Source: [tests/test_otsu_calculation_facts.py](../../tests/test_otsu_calculation_facts.py). Category: Image analysis, calibration and rendering.

### OtsuCalculationContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_otsu_calculation_facts.py#L112) | 2 related OtsuCalculationContractTests scenarios, each with private fixture state. | `helper: tests.test_otsu_calculation_facts.OtsuCalculationContractTests.check_test_facts_reject_bool_out_of_range_profile_polarity_and_false_source_claim: self.assertEqual(expected['object_type'], 'light')` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_otsu_calculation_facts.py#L117) | 1 related OtsuCalculationContractTests scenarios, each with private fixture state. | `helper: tests.test_otsu_calculation_facts.OtsuCalculationContractTests.check_test_near_tie_and_rare_class_fallback_facts_keep_existing_selection_guards: self.assertRaises(thresholds.OtsuAmbiguityError)` | unrecorded / — |
### OtsuCalculationCudaTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_otsu_calculation_facts.py#L200) | 3 related OtsuCalculationCudaTests scenarios, each with private fixture state. | `helper: tests.test_otsu_calculation_facts.OtsuCalculationCudaTests.check_test_result_and_compatibility_wrapper_each_use_one_histogram: self.assertEqual(histogram.call_count, 1)` | unrecorded / — |

## tests.test_peer_feature_register

Structural checks for the PlantCV/SMPTS operation-level feature register.

Source: [tests/test_peer_feature_register.py](../../tests/test_peer_feature_register.py). Category: Recipe, replay and reference infrastructure.

### PeerFeatureRegisterTests

Fixtures: `setUpClass` at line 21.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_peer_feature_register.py#L148) | 6 related PeerFeatureRegisterTests scenarios, each with private fixture state. | `helper: tests.test_peer_feature_register.PeerFeatureRegisterTests.check_test_all_families_and_operation_rows_have_independent_status_and_contracts: self.assertEqual([f'F{index:02}' for index in range(1, 23)], [item['id'] for item in features])` | unrecorded / — |

## tests.test_peer_recipe_catalogue

Focused integrity checks for the audited, non-executed peer recipes.

Source: [tests/test_peer_recipe_catalogue.py](../../tests/test_peer_recipe_catalogue.py). Category: Recipe, replay and reference infrastructure.

### PeerRecipeCatalogueTests

Fixtures: `setUpClass` at line 47.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_peer_recipe_catalogue.py#L409) | 10 related PeerRecipeCatalogueTests scenarios, each with private fixture state. | `helper: tests.test_peer_recipe_catalogue.PeerRecipeCatalogueTests.check_test_catalogue_json_and_recipe_identities_are_deterministic: self.assertEqual(restored.to_json(), first)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_peer_recipe_catalogue.py#L422) | 2 related PeerRecipeCatalogueTests scenarios, each with private fixture state. | `helper: tests.test_peer_recipe_catalogue.PeerRecipeCatalogueTests.check_test_catalogue_json_rejects_duplicate_keys_and_execution_claims: self.assertRaises(RecipeCatalogueError)` | unrecorded / — |

## tests.test_performance_report

Actual synchronized CUDA/resource/cache facts and explicit correction timing.

Source: [tests/test_performance_report.py](../../tests/test_performance_report.py). Category: Startup, resources and diagnostics.

### PerformanceContractTests

Fixtures: `setUp` at line 23.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_performance_report.py#L83) | 2 related PerformanceContractTests scenarios, each with private fixture state. | `helper: tests.test_performance_report.PerformanceContractTests.check_test_resource_controls_reject_invalid_and_unbounded_requests_before_cuda: self.assertRaises((ValueError, TypeError))` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_performance_report.py#L88) | 2 related PerformanceContractTests scenarios, each with private fixture state. | `helper: tests.test_performance_report.PerformanceContractTests.check_test_explicit_correction_timer_preserves_completed_and_interrupted_work: self.assertEqual(complete['status'], 'completed')` | unrecorded / — |
### ActualPerformanceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_performance_report.py#L158) | 1 related ActualPerformanceTests scenarios, each with private fixture state. | `helper: tests.test_performance_report.ActualPerformanceTests.check_test_actual_cold_warm_timings_peaks_and_transfer_trace_preserve_native_products: self.assertEqual(report['status'], 'executed', report['runs'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_performance_report.py#L162) | 3 related ActualPerformanceTests scenarios, each with private fixture state. | `helper: tests.test_performance_report.ActualPerformanceTests.check_test_measured_threshold_stops_following_runs_and_cancel_is_distinct: self.assertEqual(report['status'], 'threshold_exceeded')` | unrecorded / — |

## tests.test_physical_shape_fit_availability

Finite calibrated evidence may exceed the physical fitter's numeric domain.

Source: [tests/test_physical_shape_fit_availability.py](../../tests/test_physical_shape_fit_availability.py). Category: Measurements and shape models.

### PhysicalShapeFitAvailabilityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_physical_shape_fit_availability.py#L270) | 14 related PhysicalShapeFitAvailabilityTests scenarios, each with private fixture state. | `helper: tests.test_physical_shape_fit_availability.PhysicalShapeFitAvailabilityTests.check_test_real_calibrated_area_covariance_overflow_retains_all_dimensionless_groups: self.assertTrue(all((np.isfinite(m.vector()).all() for m in values)))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_physical_shape_fit_availability.py#L287) | 3 related PhysicalShapeFitAvailabilityTests scenarios, each with private fixture state. | `helper: tests.test_physical_shape_fit_availability.PhysicalShapeFitAvailabilityTests.check_test_parent_predictive_sum_overflow_has_dedicated_numeric_failure: self.assertRaises(model._PhysicalFitNumericUnavailable)` | unrecorded / — |

## tests.test_pilot_analysis

Committed pilot-image segmentation, calibration and end-to-end analysis

Source: [tests/test_pilot_analysis.py](../../tests/test_pilot_analysis.py). Category: Image analysis, calibration and rendering.

### PilotAnalysisTests

Fixtures: `setUpClass` at line 15.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_sparse_pilot_runs_diagnostics_without_foreground_references`](../../tests/test_pilot_analysis.py#L891) | sparse pilot runs diagnostics without foreground references | `helper: tests.test_pilot_analysis.PilotAnalysisTests.check_test_sparse_pilot_runs_diagnostics_without_foreground_references: self.assertGreater(result.count, 0)` | unrecorded / — |
| [`test_dense_lupin_requires_user_authored_foreground_reference`](../../tests/test_pilot_analysis.py#L894) | dense lupin requires user authored foreground reference | `helper: tests.test_pilot_analysis.PilotAnalysisTests.check_test_dense_lupin_requires_user_authored_foreground_reference: self.assertIsNotNone(automatic.perimeter_background_lab)` | unrecorded / — |
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
| [`test_workflow_lifecycle`](../../tests/test_pipeline.py#L1366) | 30 related PipelineModelTests scenarios, each with private fixture state. | `helper: tests.test_pipeline.PipelineModelTests.check_test_foreground_requires_authored_evidence_and_has_no_refinement_controls: self.assertFalse(any(('automatic' in key for key in foreground_keys)))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_pipeline.py#L1399) | 7 related PipelineModelTests scenarios, each with private fixture state. | `helper: tests.test_pipeline.PipelineModelTests.check_test_reconnection_rejects_unimplemented_typed_substitutions: self.assertRaisesRegex(ValueError, 'different data types\|requires\|not an authored calculation')` | unrecorded / — |
### NodeTimingRecorderTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_cpu_and_explicit_spans_are_aggregated_by_node`](../../tests/test_pipeline.py#L1411) | cpu and explicit spans are aggregated by node | `self.assertEqual(result['decode'], 0.125); self.assertGreater(result['analysis'], 0.0)` | unrecorded / — |
| [`test_cancellation_stops_at_the_next_timed_node_boundary`](../../tests/test_pipeline.py#L1438) | cancellation stops at the next timed node boundary | `self.assertRaises(AnalysisCancelled); self.assertRaises(AnalysisCancelled)` | unrecorded / — |
### BaselineSettingsTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_out_of_range_setting_is_rejected`](../../tests/test_pipeline.py#L1761) | out of range setting is rejected | `helper: tests.test_pipeline.BaselineSettingsTests.check_test_out_of_range_setting_is_rejected: self.assertRaises(ValueError)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_pipeline.py#L1764) | 6 related BaselineSettingsTests scenarios, each with private fixture state. | `helper: tests.test_pipeline.BaselineSettingsTests.check_test_composite_dish_and_layer_settings_are_validated: self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_pipeline_backend_adapters

Pinned public-call adapters beyond the seed template's elementary stages.

Source: [tests/test_pipeline_backend_adapters.py](../../tests/test_pipeline_backend_adapters.py). Category: Pipeline workspace and controls.

### PipelinePublicAdapterTests

Fixtures: `setUp` at line 21.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_additional_numerical_regressions`](../../tests/test_pipeline_backend_adapters.py#L167) | additional numerical regressions | `helper: tests.test_pipeline_backend_adapters.PipelinePublicAdapterTests.check_test_extended_threshold_and_range_public_calls_preserve_masks_and_rgb: self.assertTrue(torch.equal(actual[name], expected[name]), stage.operation_id + '.' + name)` | unrecorded / — |

## tests.test_pipeline_backend_archive

Actual public execution stays distinct through archive, QC and replay.

Source: [tests/test_pipeline_backend_archive.py](../../tests/test_pipeline_backend_archive.py). Category: Pipeline workspace and controls.


## tests.test_pipeline_backend_execution

Actual backend choices, public CPU adapters and node-local cache contracts.

Source: [tests/test_pipeline_backend_execution.py](../../tests/test_pipeline_backend_execution.py). Category: Pipeline workspace and controls.

### PipelineBackendPolicyTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_backend_execution.py#L159) | 4 related PipelineBackendPolicyTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_backend_execution.PipelineBackendPolicyTests.check_test_native_default_and_public_only_where_equivalent: self.assertTrue(all((resolve_backend(selected, stage.stage_id) == 'native' for stage in selected.recipe.stages)))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_pipeline_backend_execution.py#L166) | 5 related PipelineBackendPolicyTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_backend_execution.PipelineBackendPolicyTests.check_test_native_only_explicit_override_is_rejected: self.assertRaisesRegex(BackendUnavailable, 'input boundary')` | unrecorded / — |
### PipelineBackendPublicIntegrationTests

Fixtures: `setUp` at line 188.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_backend_execution.py#L278) | 2 related PipelineBackendPublicIntegrationTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_backend_execution.PipelineBackendPublicIntegrationTests.check_test_all_three_seed_factories_execute_the_actual_public_complete_pipeline: self.assertEqual(result.status, 'executed', [(node.stage_id, node.status, node.reason) for n…` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_pipeline_backend_execution.py#L283) | 1 related PipelineBackendPublicIntegrationTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_backend_execution.PipelineBackendPublicIntegrationTests.check_test_mutated_public_capture_invalidates_whole_group_only: self.assertEqual(second.status, 'executed')` | unrecorded / — |

## tests.test_pipeline_cable_routing

Pipeline cable geometry and routing

Source: [tests/test_pipeline_cable_routing.py](../../tests/test_pipeline_cable_routing.py). Category: Pipeline workspace and controls.

### PipelineCableRoutingTests

Fixtures: `setUpClass` at line 148.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_cable_routing.py#L516) | 8 related PipelineCableRoutingTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_cable_routing.PipelineCableRoutingTests.check_test_default_mode_preserves_the_established_cubic_paths: self.assertFalse(canvas.cable_bundling_enabled)` | unrecorded / — |

## tests.test_pipeline_documents

Standalone graph/layout transport, factory protection and frozen file loads.

Source: [tests/test_pipeline_documents.py](../../tests/test_pipeline_documents.py). Category: Pipeline workspace and controls.

### PipelineDocumentTests

Fixtures: `setUpClass` at line 37.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_documents.py#L223) | 2 related PipelineDocumentTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_documents.PipelineDocumentTests.check_test_full_roundtrip_preserves_both_primaries_instances_and_backend_choices: self.assertNotIn('backend_default', workflow.to_dict())` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_pipeline_documents.py#L228) | 2 related PipelineDocumentTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_documents.PipelineDocumentTests.check_test_invalid_files_atomic_failure_and_factory_names_cannot_replace_existing_bytes: self.assertRaises(PipelineDocumentError)` | unrecorded / — |

## tests.test_pipeline_inspector_actions

Pipeline inspector parameter editing and actions

Source: [tests/test_pipeline_inspector_actions.py](../../tests/test_pipeline_inspector_actions.py). Category: Pipeline workspace and controls.

### PipelineInspectorActionTests

Fixtures: `setUpClass` at line 14.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_inspector_actions.py#L172) | 4 related PipelineInspectorActionTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_inspector_actions.PipelineInspectorActionTests.check_test_procedural_fit_action_is_contextual_and_emits_loss_weights: self.assertTrue(inspector.procedural_fit_container.isHidden())` | unrecorded / — |

## tests.test_pipeline_ui

Pipeline desktop controls, graph state, navigation, workers and status

Source: [tests/test_pipeline_ui.py](../../tests/test_pipeline_ui.py). Category: Pipeline workspace and controls.

### PipelineCanvasTests

Fixtures: `setUpClass` at line 23.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_approved_reference_edge_fit_updates_live_node_parameters`](../../tests/test_pipeline_ui.py#L4270) | approved reference edge fit updates live node parameters | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_approved_reference_edge_fit_updates_live_node_parameters: self.assertAlmostEqual(node.parameters['reference_edge_similarity_scale'], 1.3)` | unrecorded / — |
| [`test_switching_images_cancels_all_previous_image_work_before_load`](../../tests/test_pipeline_ui.py#L4273) | switching images cancels all previous image work before load | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_switching_images_cancels_all_previous_image_work_before_load: self.assertTrue(analysis_task.cancelled)` | unrecorded / — |
| [`test_cancelled_previous_image_completion_cannot_install_stale_result`](../../tests/test_pipeline_ui.py#L4276) | cancelled previous image completion cannot install stale result | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_cancelled_previous_image_completion_cannot_install_stale_result: mark_complete.assert_not_called()` | unrecorded / — |
| [`test_failed_analysis_purges_its_partial_cache`](../../tests/test_pipeline_ui.py#L4279) | failed analysis purges its partial cache | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_failed_analysis_purges_its_partial_cache: self.assertNotIn(key, window._analysis_caches)` | unrecorded / — |
| [`test_superseded_analysis_is_cancelled_and_remains_visible`](../../tests/test_pipeline_ui.py#L4282) | superseded analysis is cancelled and remains visible | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_superseded_analysis_is_cancelled_and_remains_visible: self.assertTrue(task.cancelled)` | unrecorded / — |
| [`test_worker_progress_turns_only_the_current_revision_node_green`](../../tests/test_pipeline_ui.py#L4285) | worker progress turns only the current revision node green | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_worker_progress_turns_only_the_current_revision_node_green: self.assertTrue(succeeded, error)` | unrecorded / — |
| [`test_cleared_reference_layer_returns_to_paint_and_accepts_dabs`](../../tests/test_pipeline_ui.py#L4288) | cleared reference layer returns to paint and accepts dabs | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_cleared_reference_layer_returns_to_paint_and_accepts_dabs: self.assertTrue(window.image_view._reference_erase_enabled)` | unrecorded / — |
| [`test_perimeter_source_checkbox_invalidates_every_background_consumer`](../../tests/test_pipeline_ui.py#L4291) | perimeter source checkbox invalidates every background consumer | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_perimeter_source_checkbox_invalidates_every_background_consumer: self.assertTrue(window.keep_perimeter_background_reference_checkbox.isChecked())` | unrecorded / — |
| [`test_annotated_seed_foreground_checkbox_defaults_on_and_invalidates_consumers`](../../tests/test_pipeline_ui.py#L4294) | annotated seed foreground checkbox defaults on and invalidates consumers | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_annotated_seed_foreground_checkbox_defaults_on_and_invalidates_consumers: self.assertTrue(checkbox.isChecked())` | unrecorded / — |
| [`test_new_seed_preserves_the_selected_instance_tool`](../../tests/test_pipeline_ui.py#L4297) | new seed preserves the selected instance tool | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_new_seed_preserves_the_selected_instance_tool: self.assertTrue(image.save(str(image_path)))` | unrecorded / — |
| [`test_reference_masks_run_analysis_only_after_explicit_apply`](../../tests/test_pipeline_ui.py#L4300) | reference masks run analysis only after explicit apply | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_reference_masks_run_analysis_only_after_explicit_apply: self.assertEqual(analyses, [])` | unrecorded / — |
| [`test_instance_annotations_apply_separately_from_reference_masks`](../../tests/test_pipeline_ui.py#L4303) | instance annotations apply separately from reference masks | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_instance_annotations_apply_separately_from_reference_masks: self.assertIn(key, window._instance_annotations_dirty)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_ui.py#L4306) | 53 related PipelineCanvasTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_canvas_contains_default_nodes_and_live_edges: self.assertGreater(canvas.horizontalScrollBar().maximum(), 0)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_pipeline_ui.py#L4362) | 1 related PipelineCanvasTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_setting_change_invalidates_only_node_and_graph_dependents: self.assertEqual(window.pipeline.node('edge_gradients').status, NodeStatus.IDLE)` | unrecorded / — |
| [`test_calibration_nodes_render_references_and_metric_scale`](../../tests/test_pipeline_ui.py#L4713) | calibration nodes render references and metric scale | `helper: tests.test_pipeline_ui.PipelineCanvasTests.check_test_calibration_nodes_render_references_and_metric_scale: self.assertTrue(succeeded, error)` | unrecorded / — |

## tests.test_pipeline_workflow_batch

Saved workspace workflows execute through the ordinary project export route.

Source: [tests/test_pipeline_workflow_batch.py](../../tests/test_pipeline_workflow_batch.py). Category: Pipeline workspace and controls.


## tests.test_pipeline_workflow_ui

Main-workspace routing, editing and queued-delivery ownership.

Source: [tests/test_pipeline_workflow_ui.py](../../tests/test_pipeline_workflow_ui.py). Category: Pipeline workspace and controls.

### PipelineWorkflowUiTests

Fixtures: `setUpClass` at line 24, `setUp` at line 27, `tearDown` at line 34.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_workflow_ui.py#L455) | workflow lifecycle | `helper: tests.test_pipeline_workflow_ui.PipelineWorkflowUiTests.check_test_shared_project_nodes_and_two_way_image_selection: self.assertTrue(CONTEXT_NODE_IDS <= self.window.pipeline_canvas.graph.nodes.keys())` | unrecorded / — |

## tests.test_pipeline_workflows

Editable workspace contracts and real CUDA node-local execution.

Source: [tests/test_pipeline_workflows.py](../../tests/test_pipeline_workflows.py). Category: Pipeline workspace and controls.

### PipelineWorkflowTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_shared_project_context_preserves_factory_calculations`](../../tests/test_pipeline_workflows.py#L19) | shared project context preserves factory calculations | `self.assertTrue(CONTEXT_NODE_IDS <= graph.nodes.keys()); self.assertTrue(set(CONTEXT_CONNECTIONS) <= set(graph.connections))` | unrecorded / — |
| [`test_roundtrip_and_old_project_default`](../../tests/test_pipeline_workflows.py#L52) | roundtrip and old project default | `self.assertEqual(PipelineWorkflow.from_json(edited.to_json()), edited); self.assertEqual(dict(PipelineWorkflow.from_dict(legacy).display_state), {})` | unrecorded / — |
| [`test_source_and_working_operation_contracts_reject_tampering`](../../tests/test_pipeline_workflows.py#L90) | source and working operation contracts reject tampering | `self.assertRaises(ValueError); self.assertRaises(ValueError)` | unrecorded / — |
| [`test_real_run_to_node_rewire_bypass_and_layout_cache`](../../tests/test_pipeline_workflows.py#L100) | real run to node rewire bypass and layout cache | `self.assertEqual({n.stage_id for n in first.nodes if n.status != 'not_run'}, {source.stage_id, median.stage_id, threshold.stage_id}); self.assertEqual(set(cache.last_reused_nodes), {source.stage_id, median.stage_id, threshold.stage_id})` | unrecorded / — |
### PipelineWorkflowCardAuditTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_complete_card_and_calibrated_otsu_workflows_share_final_audit_scope`](../../tests/test_pipeline_workflows.py#L203) | complete card and calibrated otsu workflows share final audit scope | `self.assertTrue(cv2.imwrite(str(path), image)); self.assertEqual(result.status, 'executed', [(n.stage_id, n.reason) for n in result.nodes])` | unrecorded / — |

## tests.test_pipeline_workspace_files

Desktop freeze, factory copies and complete graph/layout file integration.

Source: [tests/test_pipeline_workspace_files.py](../../tests/test_pipeline_workspace_files.py). Category: Pipeline workspace and controls.

### PipelineWorkspaceFileTests

Fixtures: `setUpClass` at line 25, `setUp` at line 28, `tearDown` at line 35.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_pipeline_workspace_files.py#L246) | 4 related PipelineWorkspaceFileTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_workspace_files.PipelineWorkspaceFileTests.check_test_freeze_guards_edits_but_allows_layout_run_and_backend_choice: self.assertTrue(self.files.frozen)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_pipeline_workspace_files.py#L253) | 1 related PipelineWorkspaceFileTests scenarios, each with private fixture state. | `helper: tests.test_pipeline_workspace_files.PipelineWorkspaceFileTests.check_test_invalid_backend_load_is_atomic_and_unavailable_assets_survive_resave: self.assertFalse(self.files.load_pipeline(bad))` | unrecorded / — |

## tests.test_plantcv_card_detection

CUDA color card detection geometry and public-source parity

Source: [tests/test_plantcv_card_detection.py](../../tests/test_plantcv_card_detection.py). Category: Numerical primitives and public-source parity.

### AutomaticCardTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_card_detection.py#L294) | 7 related AutomaticCardTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_card_detection.AutomaticCardTests.check_test_actual_complete_public_detector_and_bbox_both_methods_default_domain: np.testing.assert_array_equal(card.mask_color_card_cuda(source).cpu().numpy(), expected['bbox'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_card_detection.py#L304) | 4 related AutomaticCardTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_card_detection.AutomaticCardTests.check_test_missing_interior_chip_keeps_source_24_grid_labels_and_23_contours: self.assertEqual(len(result.geometry.selected_contour_indices), 23)` | unrecorded / — |

## tests.test_plantcv_card_preprocess

CUDA color card image preparation and source semantics

Source: [tests/test_plantcv_card_preprocess.py](../../tests/test_plantcv_card_preprocess.py). Category: Numerical primitives and public-source parity.

### ColorCardPreprocessTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_card_preprocess.py#L296) | 11 related ColorCardPreprocessTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_card_preprocess.ColorCardPreprocessTests.check_test_all_kernel_identities_and_integer_byte_weights_match_pinned_original: self.assertEqual(set(card.GAUSSIAN_KERNEL_SHA256), set(range(3, 256, 2)))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_card_preprocess.py#L310) | 4 related ColorCardPreprocessTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_card_preprocess.ColorCardPreprocessTests.check_test_binary_uncertainty_withholds_complete_mask_and_preserves_input: self.assertRaises(card.ColorCardPreprocessPrecisionError)` | unrecorded / — |

## tests.test_plantcv_color

CUDA color-space conversion and PlantCV-compatible image analysis

Source: [tests/test_plantcv_color.py](../../tests/test_plantcv_color.py). Category: Numerical primitives and public-source parity.

### ColorStatisticsTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_hue_median_excludes_zero_hue_samples`](../../tests/test_plantcv_color.py#L63) | hue median excludes zero hue samples | `self.assertAlmostEqual(mean, 2.0); self.assertAlmostEqual(deviation, 0.0)` | unrecorded / — |
| [`test_positive_byte_median_excludes_zero_samples`](../../tests/test_plantcv_color.py#L69) | positive byte median excludes zero samples | `self.assertAlmostEqual(mean, 2 / 255 * 100); self.assertAlmostEqual(median, 2 / 255 * 100)` | unrecorded / — |
### PlantCVColorParityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_color.py#L214) | 4 related PlantCVColorParityTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_color.PlantCVColorParityTests.check_test_all_histograms_statistics_bins_and_labels_match_public_call: self.assertEqual([item.pixel_count for item in result.labels], [70, 18])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_color.py#L221) | 2 related PlantCVColorParityTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_color.PlantCVColorParityTests.check_test_missing_label_failure_returns_no_partial_complete_bundle: self.assertEqual(reference['error_type'], 'UnboundLocalError')` | unrecorded / — |

## tests.test_plantcv_color_card

CUDA color card chip sampling and correction facts

Source: [tests/test_plantcv_color_card.py](../../tests/test_plantcv_color_card.py). Category: Numerical primitives and public-source parity.

### PlantCVColorCardParityTests

Fixtures: `setUpClass` at line 65.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_color_card.py#L156) | 4 related PlantCVColorCardParityTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_color_card.PlantCVColorCardParityTests.check_test_standard_matrix_matches_all_four_public_orientations_and_flooring: np.testing.assert_array_equal(actual, expected)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_color_card.py#L163) | 3 related PlantCVColorCardParityTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_color_card.PlantCVColorCardParityTests.check_test_empty_clipped_roi_preserves_public_failure: self.assertEqual(reference['result']['error_type'], 'IndexError')` | unrecorded / — |

## tests.test_plantcv_contours

CUDA contour extraction, topology and public geometry

Source: [tests/test_plantcv_contours.py](../../tests/test_plantcv_contours.py). Category: Numerical primitives and public-source parity.

### BinaryContourTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_contours.py#L159) | 5 related BinaryContourTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_contours.BinaryContourTests.check_test_noncontiguous_nonoverlapping_input_preserves_view_and_source: self.assertEqual([c.tolist() for c in result.contours], expected['contours'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_contours.py#L167) | 1 related BinaryContourTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_contours.BinaryContourTests.check_test_validation_scratch_cancel_and_first_or_final_callback_mutation: self.assertRaises(contours.BinaryContourResourceError)` | unrecorded / — |

## tests.test_plantcv_correction

CUDA polynomial color correction and numerical compatibility

Source: [tests/test_plantcv_correction.py](../../tests/test_plantcv_correction.py). Category: Numerical primitives and public-source parity.

### PlantCVCorrectionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_correction.py#L161) | 3 related PlantCVCorrectionTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_correction.PlantCVCorrectionTests.check_test_chip_rgb_means_and_headers_match_pinned_public_call: self.assertEqual(headers, tuple(expected['headers']))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_correction.py#L167) | 3 related PlantCVCorrectionTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_correction.PlantCVCorrectionTests.check_test_tiled_chip_reduction_and_affine_cancellation_preserve_input: np.testing.assert_array_equal(get_color_matrix_cuda(image, mask)[1], expected)` | unrecorded / — |

## tests.test_plantcv_distance

CUDA distance transform and public-source numerical parity

Source: [tests/test_plantcv_distance.py](../../tests/test_plantcv_distance.py). Category: Numerical primitives and public-source parity.

### PlantCVDistanceReferenceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_plantcv_source_pin_and_reference_wheel_are_explicit`](../../tests/test_plantcv_distance.py#L134) | plantcv source pin and reference wheel are explicit | `self.assertEqual((config['version'], config['commit']), (PLANTCV_VERSION, PLANTCV_COMMIT)); self.assertEqual(PLANTCV_VERSION, '4.11.3')` | unrecorded / — |
### PlantCVDistancePrimitiveTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_distance.py#L263) | 3 related PlantCVDistancePrimitiveTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_distance.PlantCVDistancePrimitiveTests.check_test_pinned_fixed_point_transform_matches_varied_masks_exactly: self.assertEqual(metadata['opencv'], OPENCV_REFERENCE_VERSION)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_distance.py#L269) | 1 related PlantCVDistancePrimitiveTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_distance.PlantCVDistancePrimitiveTests.check_test_strided_input_validation_cancellation_and_resource_cap: self.assertFalse(view.is_contiguous())` | unrecorded / — |

## tests.test_plantcv_extended_thresholds

CUDA extended threshold algorithms and input boundaries

Source: [tests/test_plantcv_extended_thresholds.py](../../tests/test_plantcv_extended_thresholds.py). Category: Numerical primitives and public-source parity.

### ExtendedThresholdTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_extended_thresholds.py#L473) | 13 related ExtendedThresholdTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_extended_thresholds.ExtendedThresholdTests.check_test_large_gaussian_kernel_bits_match_pinned_opencv: self.assertEqual(result.returncode, 0, result.stderr)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_extended_thresholds.py#L489) | 9 related ExtendedThresholdTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_extended_thresholds.ExtendedThresholdTests.check_test_gaussian_float32_ambiguous_byte_rounding_withholds_complete_mask: self.assertRaises(extended.ExtendedThresholdPrecisionError)` | unrecorded / — |

## tests.test_plantcv_fill

CUDA small-object and hole filling with public PlantCV parity

Source: [tests/test_plantcv_fill.py](../../tests/test_plantcv_fill.py). Category: Numerical primitives and public-source parity.

### PlantCVFillTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_area_threshold_equality_four_connectivity_and_border_objects`](../../tests/test_plantcv_fill.py#L213) | area threshold equality four connectivity and border objects | `helper: tests.test_plantcv_fill.PlantCVFillTests.check_test_area_threshold_equality_four_connectivity_and_border_objects: torch.testing.assert_close(actual.cpu(), torch.from_numpy(expected), rtol=0, atol=0)` | unrecorded / — |
| [`test_diagonal_contact_does_not_join_foreground_components`](../../tests/test_plantcv_fill.py#L217) | diagonal contact does not join foreground components | `helper: tests.test_plantcv_fill.PlantCVFillTests.check_test_diagonal_contact_does_not_join_foreground_components: self.assertFalse(bool(torch.any(actual).item()))` | unrecorded / — |
| [`test_empty_and_full_masks_keep_source_area_boundary_semantics`](../../tests/test_plantcv_fill.py#L221) | empty and full masks keep source area boundary semantics | `helper: tests.test_plantcv_fill.PlantCVFillTests.check_test_empty_and_full_masks_keep_source_area_boundary_semantics: self.assertTrue(bool(torch.all(retained == 255).item()))` | unrecorded / — |
| [`test_committed_real_image_mask_performance_and_memory_diagnostic`](../../tests/test_plantcv_fill.py#L229) | committed real image mask performance and memory diagnostic | `helper: tests.test_plantcv_fill.PlantCVFillTests.check_test_committed_real_image_mask_performance_and_memory_diagnostic: self.assertIsNotNone(gray, 'committed image fixture must decode')` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_fill.py#L232) | 2 related PlantCVFillTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_fill.PlantCVFillTests.check_test_source_pin_and_full_runner_pixel_cap: self.assertEqual((PLANTCV_VERSION, PLANTCV_COMMIT), ('4.11.3', '77e3322fd0cd5a9bef18bb99402f20c8f2db87a9'))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_fill.py#L237) | 2 related PlantCVFillTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_fill.PlantCVFillTests.check_test_binary_domain_parameters_cancellation_and_resource_limit_are_strict: self.assertRaisesRegex(ValueError, 'CUDA')` | unrecorded / — |

## tests.test_plantcv_kmeans

CUDA k-means analysis and public-model compatibility

Source: [tests/test_plantcv_kmeans.py](../../tests/test_plantcv_kmeans.py). Category: Numerical primitives and public-source parity.

### PlantCVKMeansMaskTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans.py#L159) | 3 related PlantCVKMeansMaskTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans.PlantCVKMeansMaskTests.check_test_separate_mask_order_and_bytes_match_pinned_public_call: self.assertEqual(list(actual), list(expected))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans.py#L165) | 4 related PlantCVKMeansMaskTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans.PlantCVKMeansMaskTests.check_test_k_zero_and_empty_category_failure_match_public: self.assertEqual(self.public(labels, 0, [None])[0], {'mapping': {}})` | unrecorded / — |

## tests.test_plantcv_kmeans_classifier

CUDA k-means class assignment and model boundaries

Source: [tests/test_plantcv_kmeans_classifier.py](../../tests/test_plantcv_kmeans_classifier.py). Category: Numerical primitives and public-source parity.

### KMeansClassifierTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_classifier.py#L142) | 2 related KMeansClassifierTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_classifier.KMeansClassifierTests.check_test_integer_ties_duplicates_fractional_separated_and_256th_class_match_public: self.assertEqual(actual.cpu().tolist(), expected)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_classifier.py#L147) | 4 related KMeansClassifierTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_classifier.KMeansClassifierTests.check_test_fractional_tie_near_tie_and_late_tile_withhold_complete_vector: self.assertEqual(references[0], references[1])` | unrecorded / — |

## tests.test_plantcv_kmeans_control

Compact control decisions checked against actual pinned sklearn methods.

Source: [tests/test_plantcv_kmeans_control.py](../../tests/test_plantcv_kmeans_control.py). Category: Numerical primitives and public-source parity.

### KMeansControlTests

Fixtures: `setUpClass` at line 144.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_control.py#L276) | 6 related KMeansControlTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_control.KMeansControlTests.check_test_preupdate_reassignment_timer_matches_actual_source_at_zero_and_threshold: self.assertEqual(trace, expected)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_control.py#L285) | 3 related KMeansControlTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_control.KMeansControlTests.check_test_invalid_control_state_and_calls_fail_before_state_changes: self.assertRaises(native.KMeansControlError)` | unrecorded / — |
### KMeansReassignmentCudaTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_cuda_update_then_reassignment_matches_actual_source_including_failure_order`](../../tests/test_plantcv_kmeans_control.py#L294) | cuda update then reassignment matches actual source including failure order | `self.assertEqual(sha256(result.centres.cpu().numpy().tobytes()).hexdigest(), expected['disabled_sha']); self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_plantcv_kmeans_conversion

Offline source-model conversion and shared-codec import compatibility.

Source: [tests/test_plantcv_kmeans_conversion.py](../../tests/test_plantcv_kmeans_conversion.py). Category: Numerical primitives and public-source parity.

### KMeansConversionStructureTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_conversion.py#L210) | 4 related KMeansConversionStructureTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_conversion.KMeansConversionStructureTests.check_test_shared_codec_and_help_imports_never_load_desktop_or_ml_dependencies: self.assertEqual(result.returncode, 0, result.stderr)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_conversion.py#L217) | 2 related KMeansConversionStructureTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_conversion.KMeansConversionStructureTests.check_test_atomic_replace_failure_and_output_cap_preserve_existing_file: self.assertRaises(conversion.KMeansConversionError)` | unrecorded / — |
### KMeansPublicConversionTests

Fixtures: `setUpClass` at line 227.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_converted_models_match_public_heldout_labels_and_ordered_masks`](../../tests/test_plantcv_kmeans_conversion.py#L290) | converted models match public heldout labels and ordered masks | `helper: tests.test_plantcv_kmeans_conversion.KMeansPublicConversionTests.check_test_converted_models_match_public_heldout_labels_and_ordered_masks: self.assertEqual(sha256(labels.cpu().numpy().tobytes()).hexdigest(), case['labels'])` | unrecorded / — |
| [`test_source_fractional_tie_remains_withheld_after_conversion`](../../tests/test_plantcv_kmeans_conversion.py#L294) | source fractional tie remains withheld after conversion | `helper: tests.test_plantcv_kmeans_conversion.KMeansPublicConversionTests.check_test_source_fractional_tie_remains_withheld_after_conversion: self.assertRaises(scoring.KMeansDecisionAmbiguous)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_conversion.py#L297) | 2 related KMeansPublicConversionTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_conversion.KMeansPublicConversionTests.check_test_actual_public_training_models_preserve_exact_centres_and_conversion_provenance: self.assertEqual((model.patch_size, model.channels, len(model.centres)), (case['p'], case['c'…` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_conversion.py#L302) | 1 related KMeansPublicConversionTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_conversion.KMeansPublicConversionTests.check_test_unfitted_wrong_type_malformed_and_oversized_models_leave_output_untouched: self.assertEqual(result.returncode, 0, result.stderr)` | unrecorded / — |

## tests.test_plantcv_kmeans_export

Compact fitted-centre export, identity and transfer/cancellation boundaries.

Source: [tests/test_plantcv_kmeans_export.py](../../tests/test_plantcv_kmeans_export.py). Category: Numerical primitives and public-source parity.

### KMeansFitExportTests

Fixtures: `setUpClass` at line 21.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_export.py#L143) | 5 related KMeansFitExportTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_export.KMeansFitExportTests.check_test_early_stopped_fit_and_ordered_fractional_signed_zero_centres_roundtrip: self.assertLess(self.result.n_steps, self.result.plan.n_steps)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_export.py#L151) | 2 related KMeansFitExportTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_export.KMeansFitExportTests.check_test_only_bounded_centres_transfer_and_source_tensors_remain_unchanged: self.assertEqual(transfer.call_count, 1)` | unrecorded / — |

## tests.test_plantcv_kmeans_features

CUDA k-means features and input conversion

Source: [tests/test_plantcv_kmeans_features.py](../../tests/test_plantcv_kmeans_features.py). Category: Numerical primitives and public-source parity.

### PlantCVKMeansFeatureTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_features.py#L175) | 3 related PlantCVKMeansFeatureTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_features.PlantCVKMeansFeatureTests.check_test_gray_rgb_odd_even_and_split_row_tiles_match_actual_public_patches: np.testing.assert_array_equal(actual, reference['features'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_features.py#L181) | 3 related PlantCVKMeansFeatureTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_features.PlantCVKMeansFeatureTests.check_test_custom_stream_and_cancellation_preserve_source: self.assertEqual(starts, expected_starts)` | unrecorded / — |

## tests.test_plantcv_kmeans_fit

Complete native fitting compared to traced actual public MiniBatchKMeans.fit.

Source: [tests/test_plantcv_kmeans_fit.py](../../tests/test_plantcv_kmeans_fit.py). Category: Numerical primitives and public-source parity.

### KMeansFitTests

Fixtures: `setUpClass` at line 137.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_fit.py#L393) | 4 related KMeansFitTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_fit.KMeansFitTests.check_test_actual_public_fit_matches_all_rng_draws_initializations_steps_and_final_outputs: self.assertEqual(error, expected['error'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_fit.py#L400) | 5 related KMeansFitTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_fit.KMeansFitTests.check_test_all_reduction_profile_withholds_fit_when_initialization_comparison_is_ambiguous: self.assertIsNone(result.source_n_threads)` | unrecorded / — |

## tests.test_plantcv_kmeans_intervals

Certified compact numeric decisions and actual pinned scalar-state parity.

Source: [tests/test_plantcv_kmeans_intervals.py](../../tests/test_plantcv_kmeans_intervals.py). Category: Numerical primitives and public-source parity.

### KMeansIntervalTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_intervals.py#L177) | 6 related KMeansIntervalTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_intervals.KMeansIntervalTests.check_test_point_sequences_preserve_actual_pinned_convergence_bits_and_branch_order: self.assertTrue(value is None or value.is_point)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_intervals.py#L186) | 3 related KMeansIntervalTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_intervals.KMeansIntervalTests.check_test_interval_tolerance_stop_bypass_and_ambiguity_preserve_state_order: self.assertEqual(stop, expected)` | unrecorded / — |

## tests.test_plantcv_kmeans_minibatch

Exact pinned sklearn dense updates and CUDA row-order/resource contracts.

Source: [tests/test_plantcv_kmeans_minibatch.py](../../tests/test_plantcv_kmeans_minibatch.py). Category: Numerical primitives and public-source parity.

### KMeansMiniBatchTests

Fixtures: `setUpClass` at line 68.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_minibatch.py#L288) | 5 related KMeansMiniBatchTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_minibatch.KMeansMiniBatchTests.check_test_actual_public_partial_fit_and_cython_updates_match_binary64_bytes: self.assert_reference(result, index)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_minibatch.py#L296) | 5 related KMeansMiniBatchTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_minibatch.KMeansMiniBatchTests.check_test_initial_mid_tile_and_final_cancel_withhold_outputs_and_preserve_inputs: self.assertRaises(native.KMeansMiniBatchCancelled)` | unrecorded / — |

## tests.test_plantcv_kmeans_model

K-means model structure, identity and validation

Source: [tests/test_plantcv_kmeans_model.py](../../tests/test_plantcv_kmeans_model.py). Category: Numerical primitives and public-source parity.

### KMeansModelTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_model.py#L139) | 2 related KMeansModelTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_model.KMeansModelTests.check_test_frozen_rows_order_duplicates_and_raw_vs_numeric_identity: self.assertEqual(model.centres, ((0.0,), (2.5,), (0.0,)))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_model.py#L144) | 6 related KMeansModelTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_model.KMeansModelTests.check_test_encoding_rejects_forged_models_before_creating_an_encoder: self.assertRaises(native.KMeansModelError)` | unrecorded / — |

## tests.test_plantcv_kmeans_plusplus

Actual public/private pinned k-means++ IDs, centres and RNG state on CUDA.

Source: [tests/test_plantcv_kmeans_plusplus.py](../../tests/test_plantcv_kmeans_plusplus.py). Category: Numerical primitives and public-source parity.

### KMeansPlusPlusTests

Fixtures: `setUpClass` at line 71.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_plusplus.py#L225) | 3 related KMeansPlusPlusTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_plusplus.KMeansPlusPlusTests.check_test_actual_public_private_and_estimator_init_match_ids_centres_and_state: self.assert_reference(result, schedule, expected)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_plusplus.py#L231) | 4 related KMeansPlusPlusTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_plusplus.KMeansPlusPlusTests.check_test_early_distance_candidate_and_final_cancellation_withhold_centres: self.assertRaises(native.KMeansPlusPlusCancelled)` | unrecorded / — |

## tests.test_plantcv_kmeans_predict

CUDA k-means predictions and source-model agreement

Source: [tests/test_plantcv_kmeans_predict.py](../../tests/test_plantcv_kmeans_predict.py). Category: Numerical primitives and public-source parity.

### KMeansPredictionTests

Fixtures: `setUpClass` at line 90.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_predict.py#L218) | 3 related KMeansPredictionTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_predict.KMeansPredictionTests.check_test_full_images_match_public_predict_joblib_path_gray_rgb_odd_even_and_fitted_model: np.testing.assert_array_equal(actual.cpu().numpy(), reference['labels'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_predict.py#L224) | 3 related KMeansPredictionTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_predict.KMeansPredictionTests.check_test_fractional_public_tie_withholds_complete_prediction: self.assertTrue(np.all(np.asarray(reference['labels']) == 0))` | unrecorded / — |

## tests.test_plantcv_kmeans_preprocess

CUDA k-means image preparation and feature order

Source: [tests/test_plantcv_kmeans_preprocess.py](../../tests/test_plantcv_kmeans_preprocess.py). Category: Numerical primitives and public-source parity.

### KMeansGaussianPreprocessTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_preprocess.py#L192) | 4 related KMeansGaussianPreprocessTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_preprocess.KMeansGaussianPreprocessTests.check_test_odd_even_gray_rgb_and_degenerate_tile_halos_match_public_patch_extract: np.testing.assert_array_equal(actual.cpu().numpy(), expected)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_preprocess.py#L199) | 3 related KMeansGaussianPreprocessTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_preprocess.KMeansGaussianPreprocessTests.check_test_custom_stream_tile_repartition_and_half_boundary_withholding: np.testing.assert_array_equal(tiled.cpu().numpy(), expected)` | unrecorded / — |

## tests.test_plantcv_kmeans_random

Exact random-call outputs/state from actual pinned MiniBatchKMeans.fit.

Source: [tests/test_plantcv_kmeans_random.py](../../tests/test_plantcv_kmeans_random.py). Category: Numerical primitives and public-source parity.

### KMeansRandomTests

Fixtures: `setUpClass` at line 79.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_random.py#L195) | 5 related KMeansRandomTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_random.KMeansRandomTests.check_test_replay_actual_public_fit_call_sequence_matches_outputs_and_every_state: self.assertFalse(call['weighted'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_random.py#L203) | 2 related KMeansRandomTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_random.KMeansRandomTests.check_test_invalid_seed_sizes_and_domains_fail_without_advancing_state: self.assertRaises(native.KMeansRandomError)` | unrecorded / — |

## tests.test_plantcv_kmeans_sampled_data

Actual pinned public sampled patches, CUDA storage and source RNG order.

Source: [tests/test_plantcv_kmeans_sampled_data.py](../../tests/test_plantcv_kmeans_sampled_data.py). Category: Numerical primitives and public-source parity.

### KMeansSampledDataTests

Fixtures: `setUpClass` at line 64.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_sampled_data.py#L255) | 5 related KMeansSampledDataTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_sampled_data.KMeansSampledDataTests.check_test_gray_bgr_odd_even_degenerate_and_replacement_modes_match_actual_public: self.assertEqual(result.device, source.device)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_sampled_data.py#L263) | 5 related KMeansSampledDataTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_sampled_data.KMeansSampledDataTests.check_test_zero_fraction_floor_matches_actual_public_failure_before_gaussian: self.assertEqual(expected['error'], 'ValueError')` | unrecorded / — |

## tests.test_plantcv_kmeans_tolerance

Pinned NumPy/sklearn tolerance values and bounded CUDA producers.

Source: [tests/test_plantcv_kmeans_tolerance.py](../../tests/test_plantcv_kmeans_tolerance.py). Category: Numerical primitives and public-source parity.

### KMeansToleranceTests

Fixtures: `setUpClass` at line 45.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_tolerance.py#L163) | 3 related KMeansToleranceTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_tolerance.KMeansToleranceTests.check_test_actual_pinned_two_pass_variance_and_shift_values_are_enclosed: self.assertLessEqual(result.lo, expected)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_tolerance.py#L169) | 3 related KMeansToleranceTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_tolerance.KMeansToleranceTests.check_test_caps_invalid_values_and_overflow_are_explicit: self.assertRaises(native.KMeansToleranceError)` | unrecorded / — |

## tests.test_plantcv_kmeans_training

Ordered CUDA image training compared with pinned public PlantCV and sklearn.

Source: [tests/test_plantcv_kmeans_training.py](../../tests/test_plantcv_kmeans_training.py). Category: Numerical primitives and public-source parity.

### KMeansImageTrainingTests

Fixtures: `setUpClass` at line 65.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_training.py#L318) | 7 related KMeansImageTrainingTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_training.KMeansImageTrainingTests.check_test_actual_public_patches_and_complete_fit_match_gray_bgr_and_source_defaults: self.assertEqual(_sha(result.training_rows.features), expected['features'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_training.py#L328) | 5 related KMeansImageTrainingTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_training.KMeansImageTrainingTests.check_test_cancellation_and_every_assembly_callback_mutation_withhold_all_rows: self.assertRaisesRegex(native.KMeansImageTrainingError, 'changed')` | unrecorded / — |

## tests.test_plantcv_kmeans_training_data

Actual pinned public training patches and bounded native CUDA allocation.

Source: [tests/test_plantcv_kmeans_training_data.py](../../tests/test_plantcv_kmeans_training_data.py). Category: Numerical primitives and public-source parity.

### KMeansTrainingDataTests

Fixtures: `setUpClass` at line 51.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_training_data.py#L216) | 4 related KMeansTrainingDataTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_training_data.KMeansTrainingDataTests.check_test_unpadded_gray_bgr_odd_even_and_degenerate_patches_match_actual_public: self.assertEqual(result.dtype, torch.uint16)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_training_data.py#L223) | 4 related KMeansTrainingDataTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_training_data.KMeansTrainingDataTests.check_test_early_gaussian_and_final_cancellation_return_no_matrix: self.assertGreater(total, 10)` | unrecorded / — |

## tests.test_plantcv_kmeans_training_score

Actual pinned Lloyd/inertia calls and CUDA training-score contracts.

Source: [tests/test_plantcv_kmeans_training_score.py](../../tests/test_plantcv_kmeans_training_score.py). Category: Numerical primitives and public-source parity.

### KMeansTrainingScoreTests

Fixtures: `setUpClass` at line 58.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_kmeans_training_score.py#L253) | 4 related KMeansTrainingScoreTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_training_score.KMeansTrainingScoreTests.check_test_actual_pinned_labels_rows_and_inertia_threads_match_certified_result: self.assert_reference(result, index, threads)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_kmeans_training_score.py#L260) | 4 related KMeansTrainingScoreTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_kmeans_training_score.KMeansTrainingScoreTests.check_test_fractional_bisector_in_late_tile_withholds_all_labels: self.assertRaises(native.KMeansTrainingScorePrecisionError)` | unrecorded / — |

## tests.test_plantcv_median

CUDA median filtering and public PlantCV pixel parity

Source: [tests/test_plantcv_median.py](../../tests/test_plantcv_median.py). Category: Numerical primitives and public-source parity.

### PlantCVMedianSourceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_public_source_and_transitive_helper_evidence_are_distinguished`](../../tests/test_plantcv_median.py#L134) | public source and transitive helper evidence are distinguished | `self.assertEqual((config['version'], config['commit']), (PLANTCV_VERSION, PLANTCV_COMMIT)); self.assertEqual(PLANTCV_VERSION, '4.11.3')` | unrecorded / — |
### PlantCVMedianPrimitiveTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_plantcv_median.py#L283) | 2 related PlantCVMedianPrimitiveTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_median.PlantCVMedianPrimitiveTests.check_test_cancellation_before_during_and_after_tiles_preserves_input: self.assertRaises(MedianCancelledError)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_median.py#L288) | 4 related PlantCVMedianPrimitiveTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_median.PlantCVMedianPrimitiveTests.check_test_scalar_and_tuple_kernels_preserve_shape_dtype_device_and_input: self.assertEqual(result.dtype, torch.uint8)` | unrecorded / — |
### PlantCVMedianReferenceTests

Fixtures: `setUpClass` at line 300.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_median.py#L415) | 3 related PlantCVMedianReferenceTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_median.PlantCVMedianReferenceTests.check_test_odd_scalar_no_roi_matches_the_actual_reference_runner: np.testing.assert_array_equal(actual.cpu().numpy(), reference.arrays['image'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_median.py#L421) | 1 related PlantCVMedianReferenceTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_median.PlantCVMedianReferenceTests.check_test_source_undefined_large_reflect_geometry_is_rejected_before_work: self.assertRaises(UnsupportedMedianGeometryError)` | unrecorded / — |

## tests.test_plantcv_minrect

CUDA minimum-area rectangle geometry and degeneracies

Source: [tests/test_plantcv_minrect.py](../../tests/test_plantcv_minrect.py). Category: Numerical primitives and public-source parity.

### PinnedMinRectTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_minrect.py#L100) | 3 related PinnedMinRectTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_minrect.PinnedMinRectTests.check_test_exact_pinned_rectangles_hulls_boxes_ties_degeneracy_and_point_clouds: np.testing.assert_array_equal(np.r_[actual[0], actual[1], actual[2]], np.r_[original['rect'][0], original['rect'][1], orig…` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_minrect.py#L106) | 1 related PinnedMinRectTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_minrect.PinnedMinRectTests.check_test_bounds_before_hull_and_cancel_before_after_hull_and_long_loops: self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_plantcv_morphology

CUDA erosion/dilation and public PlantCV pixel parity

Source: [tests/test_plantcv_morphology.py](../../tests/test_plantcv_morphology.py). Category: Numerical primitives and public-source parity.

### PlantCVMorphologySourceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_public_source_hashes_match_the_committed_release_registry`](../../tests/test_plantcv_morphology.py#L56) | public source hashes match the committed release registry | `self.assertEqual(config['reference'], 'plantcv'); self.assertEqual(config['version'], PLANTCV_VERSION)` | unrecorded / — |
| [`test_transitive_helper_hash_is_recorded_as_environment_evidence_only`](../../tests/test_plantcv_morphology.py#L80) | transitive helper hash is recorded as environment evidence only | `self.assertEqual(hashlib.sha256(raw).hexdigest(), PLANTCV_HELPER_REFERENCE_ENV_SHA256); self.assertNotIn('plantcv/plantcv/_helpers.py', source_paths)` | unrecorded / — |
### PlantCVMorphologyPrimitiveTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_plantcv_morphology.py#L188) | 1 related PlantCVMorphologyPrimitiveTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_morphology.PlantCVMorphologyPrimitiveTests.check_test_kernel_and_iteration_domain_matches_reference_runner_limits: self.assertEqual((MIN_KERNEL_SIZE, MAX_KERNEL_SIZE), (2, 255))` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_morphology.py#L192) | 5 related PlantCVMorphologyPrimitiveTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_morphology.PlantCVMorphologyPrimitiveTests.check_test_erode_and_dilate_preserve_gray_values_and_cuda_residency: self.assertEqual(result.device, gray.device)` | unrecorded / — |
### PlantCVMorphologyRunnerComparisonTests

Fixtures: `setUpClass` at line 205.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_morphology.py#L281) | 3 related PlantCVMorphologyRunnerComparisonTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_morphology.PlantCVMorphologyRunnerComparisonTests.check_test_largest_public_kernel_has_bounded_cuda_memory_and_matches_runner: np.testing.assert_array_equal(actual.cpu().numpy(), result.arrays['mask'])` | unrecorded / — |

## tests.test_plantcv_naive_bayes

CUDA Naive Bayes masks versus the pinned PlantCV public function.

Source: [tests/test_plantcv_naive_bayes.py](../../tests/test_plantcv_naive_bayes.py). Category: Numerical primitives and public-source parity.

### PlantCVNaiveBayesModelTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_naive_bayes.py#L131) | 2 related PlantCVNaiveBayesModelTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_naive_bayes.PlantCVNaiveBayesModelTests.check_test_parsing_duplicate_channels_newlines_unknown_channels_and_model_hash: self.assertEqual(model.class_names, ('leaf', 'soil', 'shadow'))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_naive_bayes.py#L136) | 1 related PlantCVNaiveBayesModelTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_naive_bayes.PlantCVNaiveBayesModelTests.check_test_malformed_nonfinite_negative_missing_and_oversized_models_are_explicit: self.assertRaises(native.NaiveBayesError)` | unrecorded / — |
### PlantCVNaiveBayesTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_naive_bayes.py#L229) | 4 related PlantCVNaiveBayesTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_naive_bayes.PlantCVNaiveBayesTests.check_test_public_masks_class_order_duplicate_channel_rows_and_input_preservation: self.assert_masks(result, _reference(image, raw), bgr.device)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_naive_bayes.py#L236) | 1 related PlantCVNaiveBayesTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_naive_bayes.PlantCVNaiveBayesTests.check_test_invalid_frozen_models_preflight_and_cancellation_publish_no_partial_masks: self.assertRaises(native.NaiveBayesError)` | unrecorded / — |

## tests.test_plantcv_naive_bayes_sampling

Actual public binary Bayes sampler observations, RNG and CUDA boundaries.

Source: [tests/test_plantcv_naive_bayes_sampling.py](../../tests/test_plantcv_naive_bayes_sampling.py). Category: Numerical primitives and public-source parity.

### BayesImageSamplingTests

Fixtures: `setUpClass` at line 108.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_naive_bayes_sampling.py#L473) | 7 related BayesImageSamplingTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_naive_bayes_sampling.BayesImageSamplingTests.check_test_actual_public_independent_columns_default_dtype_and_complete_rng_state: self.assertEqual(state_record(result.rng_before), expected['before'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_naive_bayes_sampling.py#L483) | 8 related BayesImageSamplingTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_naive_bayes_sampling.BayesImageSamplingTests.check_test_source_invalid_background_follows_consumed_foreground_draw: self.assertEqual(expected['error'], 'ValueError')` | unrecorded / — |

## tests.test_plantcv_naive_bayes_training

Ordered CUDA KDE versus the actual pinned SciPy training dependency.

Source: [tests/test_plantcv_naive_bayes_training.py](../../tests/test_plantcv_naive_bayes_training.py). Category: Numerical primitives and public-source parity.

### BayesKdeTrainingTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_naive_bayes_training.py#L394) | 8 related BayesKdeTrainingTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_naive_bayes_training.BayesKdeTrainingTests.check_test_buffered_weighted_moment_bits_at_outer_boundaries_and_constant_outcomes: self.assertEqual([v.hex() for v in actual_stats[c][:5]], source['initial'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_naive_bayes_training.py#L405) | 4 related BayesKdeTrainingTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_naive_bayes_training.BayesKdeTrainingTests.check_test_constant_source_success_and_singular_failures_all_pdf_bins_without_variance_floor: self.assertTrue(all(('error' in source for source in channels)))` | unrecorded / — |

## tests.test_plantcv_peaks

CUDA peak detection, spacing and source semantics

Source: [tests/test_plantcv_peaks.py](../../tests/test_plantcv_peaks.py). Category: Numerical primitives and public-source parity.

### NmsVisitEstimateTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_separable_maxima_equal_square_pool_at_edges_short_dimensions_and_ties`](../../tests/test_plantcv_peaks.py#L222) | separable maxima equal square pool at edges short dimensions and ties | `self.assertTrue(torch.equal(peaks._spacing_maximum(values, kernel, radius), expected)); self.assertTrue(torch.equal(peaks._spacing_maximum(values, kernel, radius), expected))` | unrecorded / — |
| [`test_round_estimate_counts_four_axis_windows_dense_and_candidate_work`](../../tests/test_plantcv_peaks.py#L239) | round estimate counts four axis windows dense and candidate work | `self.assertEqual(peaks.MAX_NMS_SAMPLE_VISITS, 64000000000); self.assertEqual(peaks._nms_round_visits(shape, distance, candidates), shape[0] * shape[1] * (24 + 4 * kernel) + 64 * candidates)` | unrecorded / — |
### NmsVisitBudgetTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_plantcv_peaks.py#L310) | 3 related NmsVisitBudgetTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_peaks.NmsVisitBudgetTests.check_test_round_visit_floor_boundary_and_incomplete_output: self.assertRaisesRegex(peaks.PeakResourceError, 'round/sample-visit limit')` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_peaks.py#L316) | 1 related NmsVisitBudgetTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_peaks.NmsVisitBudgetTests.check_test_rectangular_roi_d2_and_d3_match_priority_greedy_oracle: self.assertEqual(selected, expected)` | unrecorded / — |
### PlantCVPeakContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_pinned_source_versions_and_contract_are_explicit`](../../tests/test_plantcv_peaks.py#L322) | pinned source versions and contract are explicit | `self.assertEqual((PLANTCV_VERSION, PLANTCV_COMMIT), ('4.11.3', '77e3322fd0cd5a9bef18bb99402f20c8f2db87a9')); self.assertEqual(PLANTCV_WATERSHED_SOURCE_SHA256, 'd5fd90bf4e064117ff47ee333d2b83e62134901fda36f6966ae1b0be4a82a712')` | unrecorded / — |
| [`test_cross_opening_and_priority_rounds_preserve_pinned_edge_cases`](../../tests/test_plantcv_peaks.py#L363) | cross opening and priority rounds preserve pinned edge cases | `torch.testing.assert_close(opened, expected_opening); torch.testing.assert_close(selected, torch.tensor([[0, 0], [0, 2], [0, 4], [0, 6], [0, 8]], dtype=torch.int64))` | unrecorded / — |
### PlantCVPeakCudaTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_peaks.py#L507) | 2 related PlantCVPeakCudaTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_peaks.PlantCVPeakCudaTests.check_test_real_full_resolution_fixture_matches_pinned_peak_local_max: self.assertIsNotNone(gray, 'committed fixture must decode')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_peaks.py#L512) | 1 related PlantCVPeakCudaTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_peaks.PlantCVPeakCudaTests.check_test_validation_cancellation_and_explicit_resource_limits: self.assertRaises(PeakCancelledError)` | unrecorded / — |

## tests.test_plantcv_range

CUDA range thresholding and public pixel parity

Source: [tests/test_plantcv_range.py](../../tests/test_plantcv_range.py). Category: Numerical primitives and public-source parity.

### PlantCVRangeContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_source_and_transitive_helper_pins_match_available_bytes`](../../tests/test_plantcv_range.py#L117) | source and transitive helper pins match available bytes | `self.assertEqual(PLANTCV_VERSION, '4.11.3'); self.assertEqual(PLANTCV_COMMIT, '77e3322fd0cd5a9bef18bb99402f20c8f2db87a9')` | unrecorded / — |
### PlantCVRangeConversionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_channel_byte_converters_match_pinned_opencv_and_plantcv_helpers`](../../tests/test_plantcv_range.py#L136) | channel byte converters match pinned opencv and plantcv helpers | `np.testing.assert_array_equal(actual, reference[key])` | unrecorded / — |
### PlantCVRangeCudaTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_rgb_hsv_cmyk_and_gray_match_actual_pinned_public_chain`](../../tests/test_plantcv_range.py#L325) | rgb hsv cmyk and gray match actual pinned public chain | `helper: tests.test_plantcv_range.PlantCVRangeCudaTests.check_test_rgb_hsv_cmyk_and_gray_match_actual_pinned_public_chain: np.testing.assert_array_equal(mask.cpu().numpy(), expected[case['name'] + '_mask'])` | unrecorded / — |
| [`test_lab_custom_range_and_masked_output_match_pinned_public_chain`](../../tests/test_plantcv_range.py#L329) | lab custom range and masked output match pinned public chain | `helper: tests.test_plantcv_range.PlantCVRangeCudaTests.check_test_lab_custom_range_and_masked_output_match_pinned_public_chain: np.testing.assert_array_equal(mask.cpu().numpy(), expected['lab_mask'])` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_range.py#L332) | 3 related PlantCVRangeCudaTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_range.PlantCVRangeCudaTests.check_test_cmyk_cuda_true_division_matches_every_pinned_channel_byte: self.assertEqual(actual.device, source.device)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_range.py#L338) | 2 related PlantCVRangeCudaTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_range.PlantCVRangeCudaTests.check_test_float32_byte_preparation_noncontiguous_input_and_cancellation: torch.testing.assert_close(mask, torch.tensor([[255, 255, 255, 0]], dtype=torch.uint8, device='cuda'))` | unrecorded / — |

## tests.test_plantcv_raster_spans

CUDA contour rasterization, spans and output masks

Source: [tests/test_plantcv_raster_spans.py](../../tests/test_plantcv_raster_spans.py). Category: Numerical primitives and public-source parity.

### OpenCvRasterSpanTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_raster_spans.py#L225) | 9 related OpenCvRasterSpanTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_raster_spans.OpenCvRasterSpanTests.check_test_adjacent_stroke_batches_match_pinned_pixels_closure_and_single_large_edge: self.assertEqual(actual, {tuple(p) for p in pixels})` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_raster_spans.py#L237) | 3 related OpenCvRasterSpanTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_raster_spans.OpenCvRasterSpanTests.check_test_adjacent_stroke_work_and_raw_limits_span_batches_and_poll_before_resume: self.assertTrue(next(batches))` | unrecorded / — |

## tests.test_plantcv_size

CUDA PlantCV-compatible shape/size measurements

Source: [tests/test_plantcv_size.py](../../tests/test_plantcv_size.py). Category: Numerical primitives and public-source parity.

### PlantCVSizeSparseContourTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_size.py#L532) | 7 related PlantCVSizeSparseContourTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_size.PlantCVSizeSparseContourTests.check_test_supported_single_contours_match_pinned_order_hierarchy_and_revisits: self.assertEqual(status, 'single_external_contour')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_size.py#L542) | 9 related PlantCVSizeSparseContourTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_size.PlantCVSizeSparseContourTests.check_test_hierarchy_validation_rejects_malformed_and_detached_forests_and_cancels: self.assertTrue(_valid_retr_hierarchy(valid))` | unrecorded / — |
### PlantCVSizePremeasurementTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_size.py#L825) | 9 related PlantCVSizePremeasurementTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_size.PlantCVSizePremeasurementTests.check_test_explicit_line_thickness_matches_all_pinned_observations: self.assertTrue(actual.plantcv_bundle_available, actual.plantcv_bundle_unavailable_reason)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_size.py#L837) | 3 related PlantCVSizePremeasurementTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_size.PlantCVSizePremeasurementTests.check_test_actual_cuda_boundary_exports_trace_exactly_and_cancel_without_mutation: self.assertEqual(topology.contour_status, 'single_external_contour')` | unrecorded / — |

## tests.test_plantcv_size_diagnostic

CUDA size diagnostic products and public compatibility

Source: [tests/test_plantcv_size_diagnostic.py](../../tests/test_plantcv_size_diagnostic.py). Category: Numerical primitives and public-source parity.

### SizeDiagnosticRasterTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_size_diagnostic.py#L186) | 2 related SizeDiagnosticRasterTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_size_diagnostic.SizeDiagnosticRasterTests.check_test_caliper_fallback_thickness_is_separate_from_requested_drawing: self.assertEqual(line.call_args.kwargs['thickness'], expected_thickness)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_size_diagnostic.py#L191) | 2 related SizeDiagnosticRasterTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_size_diagnostic.SizeDiagnosticRasterTests.check_test_shared_work_cap_rejects_before_polygon_allocation: self.assertRaises(raster.RasterSpanLimitError)` | unrecorded / — |
### SizeDiagnosticCudaTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_size_diagnostic.py#L359) | 4 related SizeDiagnosticCudaTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_size_diagnostic.SizeDiagnosticCudaTests.check_test_actual_rgb_diagnostics_match_public_images_and_preserve_input: np.testing.assert_array_equal(actual.cpu().numpy(), expected[..., ::-1].copy())` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_size_diagnostic.py#L366) | 2 related SizeDiagnosticCudaTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_size_diagnostic.SizeDiagnosticCudaTests.check_test_thickness_validation_and_mismatched_geometry_reject_before_clone: self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_plantcv_thresholds

CUDA binary/mean/Otsu thresholds and isolated public-source comparison

Source: [tests/test_plantcv_thresholds.py](../../tests/test_plantcv_thresholds.py). Category: Numerical primitives and public-source parity.

### PlantCVThresholdPrimitiveTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_thresholds.py#L207) | 5 related PlantCVThresholdPrimitiveTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_thresholds.PlantCVThresholdPrimitiveTests.check_test_mean_certificate_covers_both_polarities_fractional_offsets_and_safe_disagreement: self.assertRaises(AdaptiveMeanPrecisionError)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_thresholds.py#L215) | 2 related PlantCVThresholdPrimitiveTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_thresholds.PlantCVThresholdPrimitiveTests.check_test_otsu_accepts_plateaus_and_rejects_tied_or_near_partition_scores: self.assertEqual(_otsu_cutoff_from_histogram(tied), 10)` | unrecorded / — |
### PlantCVOriginalRunnerComparisonTests

Fixtures: `setUpClass` at line 225.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_thresholds.py#L378) | 4 related PlantCVOriginalRunnerComparisonTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_thresholds.PlantCVOriginalRunnerComparisonTests.check_test_binary_supported_otsu_and_mean_match_the_pinned_isolated_runner: np.testing.assert_array_equal(near_tie_result.arrays['mask'], np.array([[0, 0, 0, 255, 255]], dtype=np.uin…` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_thresholds.py#L385) | 1 related PlantCVOriginalRunnerComparisonTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_thresholds.PlantCVOriginalRunnerComparisonTests.check_test_maximum_mean_kernel_certifies_edges_and_withholds_float32_half_case: self.assertEqual(int(original.arrays['mask'][255, 255]), 255)` | unrecorded / — |

## tests.test_plantcv_watershed

CUDA watershed labeling and public-source numerical parity

Source: [tests/test_plantcv_watershed.py](../../tests/test_plantcv_watershed.py). Category: Numerical primitives and public-source parity.

### PlantCVWatershedTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_watershed.py#L209) | 3 related PlantCVWatershedTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_watershed.PlantCVWatershedTests.check_test_exact_pinned_heap_ties_plateaus_borders_and_random_partitions: self.assertEqual(actual.device, inputs[0].device)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_watershed.py#L215) | 2 related PlantCVWatershedTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_watershed.PlantCVWatershedTests.check_test_cancellation_and_resource_preflight_do_not_return_or_modify_inputs: self.assertRaises(WatershedCancelled)` | unrecorded / — |

## tests.test_plantcv_watershed_diagnostic

CUDA watershed diagnostic labels, plots and bounds

Source: [tests/test_plantcv_watershed_diagnostic.py](../../tests/test_plantcv_watershed_diagnostic.py). Category: Numerical primitives and public-source parity.

### WatershedPaletteTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_exact_public_palette_sampling_shuffle_and_global_rng_preservation`](../../tests/test_plantcv_watershed_diagnostic.py#L102) | exact public palette sampling shuffle and global rng preservation | `self.assertEqual(palette, tuple((tuple(c) for c in colors))); self.assertEqual(metadata['palette_count'], case['count'])` | unrecorded / — |
| [`test_invalid_controls_and_runtime_lut_binding`](../../tests/test_plantcv_watershed_diagnostic.py#L120) | invalid controls and runtime lut binding | `self.assertRaises(drawing.WatershedDiagnosticError); self.assertGreater(len(identity['registry']), 100)` | unrecorded / — |
### WatershedDiagnosticTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_plantcv_watershed_diagnostic.py#L319) | 4 related WatershedDiagnosticTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_watershed_diagnostic.WatershedDiagnosticTests.check_test_long_contour_streams_exact_public_pixels_with_bounded_cpu_geometry: np.testing.assert_array_equal(actual.image.cpu().numpy(), expected['image'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_plantcv_watershed_diagnostic.py#L326) | 4 related WatershedDiagnosticTests scenarios, each with private fixture state. | `helper: tests.test_plantcv_watershed_diagnostic.WatershedDiagnosticTests.check_test_invalid_inputs_sparse_source_palette_and_preallocation_resource_caps: self.assertRaises(drawing.WatershedDiagnosticError)` | unrecorded / — |

## tests.test_polynomial_correction_integrity

Whole-call input and retained-matrix integrity for polynomial correction.

Source: [tests/test_polynomial_correction_integrity.py](../../tests/test_polynomial_correction_integrity.py). Category: Image analysis, calibration and rendering.

### PolynomialCorrectionIntegrityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_polynomial_correction_integrity.py#L161) | 4 related PolynomialCorrectionIntegrityTests scenarios, each with private fixture state. | `helper: tests.test_polynomial_correction_integrity.PolynomialCorrectionIntegrityTests.check_test_target_inputs_are_guarded_during_source_extraction: self.assertRaises(correction.CorrectionMutation)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_polynomial_correction_integrity.py#L168) | 3 related PolynomialCorrectionIntegrityTests scenarios, each with private fixture state. | `helper: tests.test_polynomial_correction_integrity.PolynomialCorrectionIntegrityTests.check_test_cross_phase_mutation_takes_precedence_over_callback_exception: self.assertRaises(correction.CorrectionMutation)` | unrecorded / — |

## tests.test_portable_result_identity

Actual relocated native results, artifact facts and private review history.

Source: [tests/test_portable_result_identity.py](../../tests/test_portable_result_identity.py). Category: Persistence, export, identity and outcomes.

### ArtifactIdentityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_portable_result_identity.py#L137) | 3 related ArtifactIdentityTests scenarios, each with private fixture state. | `helper: tests.test_portable_result_identity.ArtifactIdentityTests.check_test_equivalent_verified_locations_keep_paths_but_share_content_identity: self.assertEqual(reports[0]['identity'], reports[1]['identity'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_portable_result_identity.py#L143) | 2 related ArtifactIdentityTests scenarios, each with private fixture state. | `helper: tests.test_portable_result_identity.ArtifactIdentityTests.check_test_changed_disappeared_and_newly_available_selections_reject_stale_capture: self.assertRaisesRegex(ValueError, 'changed after analysis')` | unrecorded / — |
### PrivateReviewLookupTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_only_absent_primary_uses_project_private_review_without_writing_shared_history`](../../tests/test_portable_result_identity.py#L150) | only absent primary uses project private review without writing shared history | `self.assertEqual(lookup_result_review(root, revision, project_path=project)[1], 'matched'); self.assertFalse((root / 'result-reviews').exists())` | unrecorded / — |

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

Fixtures: `setUpClass` at line 18.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_fit_task_passes_complete_coverage_to_core_options`](../../tests/test_procedural_fit_ui.py#L131) | fit task passes complete coverage to core options | `self.assertTrue(options.annotations_are_complete); self.assertEqual(options.false_positive_weight, 2.5)` | unrecorded / — |
| [`test_completed_whole_dish_fit_labels_reported_metrics`](../../tests/test_procedural_fit_ui.py#L223) | completed whole dish fit labels reported metrics | `self.assertIn('Whole-dish metrics', window.pipeline_inspector.procedural_fit_status_label.text())` | unrecorded / — |

## tests.test_procedural_instances

CUDA procedural instance splitting, ownership and analysis integration

Source: [tests/test_procedural_instances.py](../../tests/test_procedural_instances.py). Category: Image analysis, calibration and rendering.

### ProceduralInstanceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_procedural_instances.py#L1253) | 20 related ProceduralInstanceTests scenarios, each with private fixture state. | `helper: tests.test_procedural_instances.ProceduralInstanceTests.check_test_reference_cost_settings_never_change_automatic_predictions: self.assertGreater(first.count, 0)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_procedural_instances.py#L1276) | 11 related ProceduralInstanceTests scenarios, each with private fixture state. | `helper: tests.test_procedural_instances.ProceduralInstanceTests.check_test_generated_frankenstein_region_is_hard_rejected_by_axis_ratio: self.assertEqual(result.count, 0)` | unrecorded / — |

## tests.test_project_analysis_persistence

Project analysis settings, image locators, migration and atomic persistence

Source: [tests/test_project_analysis_persistence.py](../../tests/test_project_analysis_persistence.py). Category: Persistence, export, identity and outcomes.

### ProjectAnalysisPersistenceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_posix_project_can_store_source_locators_that_differ_only_by_case`](../../tests/test_project_analysis_persistence.py#L786) | posix project can store source locators that differ only by case | `helper: tests.test_project_analysis_persistence.ProjectAnalysisPersistenceTests.check_test_posix_project_can_store_source_locators_that_differ_only_by_case: self.assertEqual(loaded.issues, ())` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_project_analysis_persistence.py#L789) | 7 related ProjectAnalysisPersistenceTests scenarios, each with private fixture state. | `helper: tests.test_project_analysis_persistence.ProjectAnalysisPersistenceTests.check_test_round_trip_is_compact_ordered_and_resolves_relative_and_external_images: self.assertTrue(destination.name.endswith(PROJECT_ANALYSIS_EXTENSION))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_project_analysis_persistence.py#L799) | 9 related ProjectAnalysisPersistenceTests scenarios, each with private fixture state. | `helper: tests.test_project_analysis_persistence.ProjectAnalysisPersistenceTests.check_test_project_v1_and_v2_migrate_to_v3_and_v2_rejects_injected_graph_fields: self.assertEqual(loaded.document.reference_graphs, ())` | unrecorded / — |

## tests.test_project_analysis_ui

Project analysis save/reopen, dirty state, selections and desktop guards

Source: [tests/test_project_analysis_ui.py](../../tests/test_project_analysis_ui.py). Category: Persistence, export, identity and outcomes.

### ProjectAnalysisUiTests

Fixtures: `setUpClass` at line 22.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_project_analysis_ui.py#L1825) | 15 related ProjectAnalysisUiTests scenarios, each with private fixture state. | `helper: tests.test_project_analysis_ui.ProjectAnalysisUiTests.check_test_actions_and_profile_load_replace_only_portable_analysis_state: self.assertEqual(window.open_action.shortcut().toString(), QKeySequence(QKeySequence.StandardKey.Open).toString())` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_project_analysis_ui.py#L1843) | 8 related ProjectAnalysisUiTests scenarios, each with private fixture state. | `helper: tests.test_project_analysis_ui.ProjectAnalysisUiTests.check_test_incompatible_profile_is_rejected_before_confirmation_and_is_atomic: self.assertFalse(loaded)` | unrecorded / — |

## tests.test_project_correction_inputs

Per-image correction selections, byte-bound copies and portable projects.

Source: [tests/test_project_correction_inputs.py](../../tests/test_project_correction_inputs.py). Category: Persistence, export, identity and outcomes.

### ProjectCorrectionInputTests

Fixtures: `setUp` at line 37, `tearDown` at line 54.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_bound_path_cannot_resolve_outside_workspace`](../../tests/test_project_correction_inputs.py#L226) | bound path cannot resolve outside workspace | `helper: tests.test_project_correction_inputs.ProjectCorrectionInputTests.check_test_bound_path_cannot_resolve_outside_workspace: self.assertRaisesRegex(ValueError, 'escapes')` | unrecorded / — |
| [`test_actual_result_selection_and_relocated_four_files_preserve_pixels`](../../tests/test_project_correction_inputs.py#L230) | actual result selection and relocated four files preserve pixels | `helper: tests.test_project_correction_inputs.ProjectCorrectionInputTests.check_test_actual_result_selection_and_relocated_four_files_preserve_pixels: self.assertFalse(loaded.issues)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_project_correction_inputs.py#L233) | 5 related ProjectCorrectionInputTests scenarios, each with private fixture state. | `helper: tests.test_project_correction_inputs.ProjectCorrectionInputTests.check_test_four_ordered_roles_strict_fields_and_canonical_frames: self.assertRaises((ValueError, TypeError))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_project_correction_inputs.py#L241) | 2 related ProjectCorrectionInputTests scenarios, each with private fixture state. | `helper: tests.test_project_correction_inputs.ProjectCorrectionInputTests.check_test_conflicting_transport_identity_rejects_before_reads: self.assertRaisesRegex(ValueError, 'conflicting byte')` | unrecorded / — |

## tests.test_project_correction_runner

Pinned correction execution, origin archives and actual batch companions.

Source: [tests/test_project_correction_runner.py](../../tests/test_project_correction_runner.py). Category: Persistence, export, identity and outcomes.

### ProjectCorrectionRunnerTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_actual_transport_run_preserves_origins_and_verified_producer_identity`](../../tests/test_project_correction_runner.py#L44) | actual transport run preserves origins and verified producer identity | `self.assertEqual(result.input_origins.binding, binding); self.assertEqual(result.input_origins.project_source_path, document.images[0].source.path)` | unrecorded / — |
| [`test_legacy_schema3_local_correction_identity_and_arrays_remain_readable`](../../tests/test_project_correction_runner.py#L71) | legacy schema3 local correction identity and arrays remain readable | `self.assertIsNone(report.pop('correction_input_origins')); self.assertNotIn('correction_input_origins', report['checkpoint_identity']['image_semantics'])` | unrecorded / — |

## tests.test_project_native_recipes

Editable native project selections coexist with exact historical graph records.

Source: [tests/test_project_native_recipes.py](../../tests/test_project_native_recipes.py). Category: Persistence, export, identity and outcomes.

### ProjectNativeRecipeTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_project_native_recipes.py#L242) | 5 related ProjectNativeRecipeTests scenarios, each with private fixture state. | `helper: tests.test_project_native_recipes.ProjectNativeRecipeTests.check_test_schema5_roundtrip_preserves_overlapping_history_and_working_selections: self.assertFalse(loaded.issues)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_project_native_recipes.py#L250) | 5 related ProjectNativeRecipeTests scenarios, each with private fixture state. | `helper: tests.test_project_native_recipes.ProjectNativeRecipeTests.check_test_invalid_working_metadata_leaves_master_and_model_reads_untouched: self.assertRaises(InvalidProjectAnalysis)` | unrecorded / — |

## tests.test_project_reference_captures

Real public batch execution and reference-only publication/QC witnesses.

Source: [tests/test_project_reference_captures.py](../../tests/test_project_reference_captures.py). Category: Persistence, export, identity and outcomes.

### ProjectReferenceCaptureTests

Fixtures: `setUpClass` at line 37, `setUp` at line 98.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_project_reference_captures.py#L175) | 1 related ProjectReferenceCaptureTests scenarios, each with private fixture state. | `helper: tests.test_project_reference_captures.ProjectReferenceCaptureTests.check_test_independent_reopening_and_recorded_qc_do_not_probe_original_source: self.assertEqual(qc_from_reference_session(opened), quality)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_project_reference_captures.py#L179) | 2 related ProjectReferenceCaptureTests scenarios, each with private fixture state. | `helper: tests.test_project_reference_captures.ProjectReferenceCaptureTests.check_test_dependency_failure_keeps_actual_partial_call_states: self.assertEqual(capture['state'], 'partial')` | unrecorded / — |

## tests.test_public_capture_layout

Preview geometry and clearing contracts, independent of source calculations.

Source: [tests/test_public_capture_layout.py](../../tests/test_public_capture_layout.py). Category: Recipe, replay and reference infrastructure.

### PublicCaptureLayoutTests

Fixtures: `setUpClass` at line 23, `setUp` at line 26, `tearDown` at line 37.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_public_capture_layout.py#L86) | 2 related PublicCaptureLayoutTests scenarios, each with private fixture state. | `helper: tests.test_public_capture_layout.PublicCaptureLayoutTests.check_test_preview_fits_both_window_sizes_and_keeps_aspect_ratio: self.assertEqual((self.dialog.width(), self.dialog.height()), (width, height))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_public_capture_layout.py#L91) | 1 related PublicCaptureLayoutTests scenarios, each with private fixture state. | `helper: tests.test_public_capture_layout.PublicCaptureLayoutTests.check_test_compact_selection_and_invalidation_cannot_restore_stale_image_on_resize: self.assertTrue(self.dialog.preview.pixmap().isNull())` | unrecorded / — |

## tests.test_public_reference_attempt_qc

Desktop boundary witnesses; no inferred outputs from thrown exceptions.

Source: [tests/test_public_reference_attempt_qc.py](../../tests/test_public_reference_attempt_qc.py). Category: Recipe, replay and reference infrastructure.

### PublicReferenceAttemptBoundaryTests

Fixtures: `setUpClass` at line 32, `setUp` at line 35.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_public_reference_attempt_qc.py#L202) | 8 related PublicReferenceAttemptBoundaryTests scenarios, each with private fixture state. | `helper: tests.test_public_reference_attempt_qc.PublicReferenceAttemptBoundaryTests.check_test_failed_wrapper_retains_original_error_and_exact_request_bytes: self.assertEqual(row['expected_source_sha256'], sha256(self.source.read_bytes()).hexdigest())` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_public_reference_attempt_qc.py#L213) | 3 related PublicReferenceAttemptBoundaryTests scenarios, each with private fixture state. | `helper: tests.test_public_reference_attempt_qc.PublicReferenceAttemptBoundaryTests.check_test_supplied_expected_source_is_not_overwritten_by_changed_observation: self.assertEqual(row['expected_source_sha256'], original)` | unrecorded / — |
### PublicReferenceCompletedAttemptTests

Fixtures: `setUpClass` at line 223, `setUp` at line 249.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_actual_empty_return_is_not_recast_as_failed_execution`](../../tests/test_public_reference_attempt_qc.py#L271) | actual empty return is not recast as failed execution | `self.assertTrue(success); self.assertEqual(session.outcome.status, 'empty')` | unrecorded / — |

## tests.test_public_seed_acceptance

Finite actual public-chain acceptance; expected artifacts are never refreshed.

Source: [tests/test_public_seed_acceptance.py](../../tests/test_public_seed_acceptance.py). Category: Frozen acceptance matrices.

### PublicSeedAcceptanceInventoryTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_frozen_case_inventory_source_recipe_and_outcome_claims`](../../tests/test_public_seed_acceptance.py#L254) | frozen case inventory source recipe and outcome claims | `self.assertEqual(record['case_count'], 8); self.assertEqual([c['case_id'] for c in record['cases']], [p.case_id for p in profiles()])` | unrecorded / — |
| [`test_changed_expected_bytes_are_rejected_without_refresh`](../../tests/test_public_seed_acceptance.py#L289) | changed expected bytes are rejected without refresh | `self.assertRaisesRegex(ValueError, 'missing or changed'); self.assertEqual(BASELINE.read_bytes(), raw)` | unrecorded / — |
| [`test_all_selected_public_chains_match_frozen_numeric_and_outcome_record`](../../tests/test_public_seed_acceptance.py#L330) | all selected public chains match frozen numeric and outcome record | `self.assertEqual(_canonical_json(actual, name='fresh public record'), _canonical_json(expected, name='fixed public record')); self.assertEqual(len(provenance), 8)` | unrecorded / — |

## tests.test_reader_acceptance

Frozen public/native reader pixels, outcomes and preparation-cache evidence.

Source: [tests/test_reader_acceptance.py](../../tests/test_reader_acceptance.py). Category: Frozen acceptance matrices.

### ReaderAcceptanceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_frozen_inventory_and_exact_encoded_fixture_bytes`](../../tests/test_reader_acceptance.py#L276) | frozen inventory and exact encoded fixture bytes | `self.assertEqual(expected['schema'], 'seedfiddle.reader-acceptance.v1'); self.assertEqual(sum((case['public']['rgb']['status'] == 'success' for case in cases.values())), 23)` | unrecorded / — |
| [`test_actual_public_native_pixels_outcomes_and_cache_match_frozen_matrix`](../../tests/test_reader_acceptance.py#L312) | actual public native pixels outcomes and cache match frozen matrix | `assert_json_equal(self, actual, expected); self.assertEqual(len(public_successes), 23)` | unrecorded / — |

## tests.test_recipe_acceptance

Finite recipe lifecycle acceptance, separate from public/source conformance.

Source: [tests/test_recipe_acceptance.py](../../tests/test_recipe_acceptance.py). Category: Frozen acceptance matrices.

### RecipeAcceptanceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_complete_admitted_version_profile_inventory`](../../tests/test_recipe_acceptance.py#L588) | complete admitted version profile inventory | `self.assertEqual(len(profiles), 223); self.assertEqual(sum((label == 'canonical' for label, _ in profiles)), 69)` | unrecorded / — |
| [`test_selected_complete_mean_inputs_match_pinned_public_chains`](../../tests/test_recipe_acceptance.py#L597) | selected complete mean inputs match pinned public chains | `self.assertTrue(seeds.REFERENCE.is_file(), 'Pinned public PlantCV interpreter is required.'); self.assertEqual(script.count('ksize=9, offset=2,'), 1)` | unrecorded / — |
| [`test_all_selected_recipes_cold_warm_verified_archive_and_source_absent_replay`](../../tests/test_recipe_acceptance.py#L648) | all selected recipes cold warm verified archive and source absent replay | `helper: tests.acceptance_workflows.compare_functional_profiles: testcase.assertEqual(sha256(raw).hexdigest(), capture.BASELINE_SHA256)` | unrecorded / — |
| [`test_frozen_baseline_integrity`](../../tests/test_recipe_acceptance.py#L652) | frozen baseline integrity | `helper: tests.acceptance_workflows.verify_frozen_baselines: testcase.assertEqual(sha256((root / 'artifacts' / name).read_bytes()).hexdigest(), digest)` | unrecorded / — |
### AcceptanceDifferenceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_complete_comparison_preserves_array_order_keys_lengths_and_types`](../../tests/test_recipe_acceptance.py#L658) | complete comparison preserves array order keys lengths and types | `self.assertEqual(summary['total_changes'], 7); self.assertEqual({sample['path'] for sample in summary['samples']}, {'/a~1b~0c', '/a~1b~0c/0', '/a~1b~0c/1', '/a~1b~0c/2', '/removed', '/added', '/flag'})` | unrecorded / — |
| [`test_sampling_cap_does_not_hide_late_changes_and_values_are_bounded`](../../tests/test_recipe_acceptance.py#L669) | sampling cap does not hide late changes and values are bounded | `self.assertEqual(summary['total_changes'], 80); self.assertEqual(len(summary['samples']), 3)` | unrecorded / — |
| [`test_json_string_subclasses_and_object_order_compare_without_false_changes`](../../tests/test_recipe_acceptance.py#L682) | json string subclasses and object order compare without false changes | `assert_json_equal(self, {'b': [VersionString('2.13.0'), None], 'a': 1.5}, {'a': 1.5, 'b': ['2.13.0', None]}); self.assertEqual(json_difference_summary(1, 1.0)['total_changes'], 1)` | unrecorded / — |
| [`test_invalid_bounds_and_non_json_values_are_rejected_or_mismatched`](../../tests/test_recipe_acceptance.py#L745) | invalid bounds and non json values are rejected or mismatched | `self.assertRaises(ValueError); self.assertRaises(AssertionError)` | unrecorded / — |

## tests.test_recipe_attempt_qc

Actual recipe terminal boundaries, historical transport and Qt ownership.

Source: [tests/test_recipe_attempt_qc.py](../../tests/test_recipe_attempt_qc.py). Category: Persistence, export, identity and outcomes.

### RecipeAttemptQCTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_recipe_attempt_qc.py#L137) | 3 related RecipeAttemptQCTests scenarios, each with private fixture state. | `helper: tests.test_recipe_attempt_qc.RecipeAttemptQCTests.check_test_actual_selected_model_mutation_fails_before_result_without_format_claim: self.assertIsInstance(delivered[0], Exception)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_recipe_attempt_qc.py#L143) | 2 related RecipeAttemptQCTests scenarios, each with private fixture state. | `helper: tests.test_recipe_attempt_qc.RecipeAttemptQCTests.check_test_before_after_changes_are_observations_and_historical_loading_never_reopens_files: self.assertNotEqual(before, row['inputs_after'][0]['observation']['sha256'])` | unrecorded / — |

## tests.test_recipe_batch_qc

Actual graph-ledger, interrupted publication and portable batch QC witnesses.

Source: [tests/test_recipe_batch_qc.py](../../tests/test_recipe_batch_qc.py). Category: Persistence, export, identity and outcomes.

### RecipeBatchQCTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_legacy_rows_and_forged_embedded_links_never_invent_recorded_qc`](../../tests/test_recipe_batch_qc.py#L75) | legacy rows and forged embedded links never invent recorded qc | `self.assertRaises(ValueError); self.assertIsNone(batches.load_project_graph_qc(path, 0))` | unrecorded / — |
### ActualRecipeBatchQCTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_actual_selected_model_failure_records_binding_phase_without_native_archive`](../../tests/test_recipe_batch_qc.py#L112) | actual selected model failure records binding phase without native archive | `self.assertEqual(row['state'], 'failed'); self.assertEqual(row['invoked_executor'], attempts.NATIVE_PATH_EXECUTOR)` | unrecorded / — |

## tests.test_reference_calibrated_capture_session

One actual capture; replayed results test session orchestration only.

Source: [tests/test_reference_calibrated_capture_session.py](../../tests/test_reference_calibrated_capture_session.py). Category: Recipe, replay and reference infrastructure.

### CalibratedSessionAdmissionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_reference_calibrated_capture_session.py#L64) | 3 related CalibratedSessionAdmissionTests scenarios, each with private fixture state. | `helper: tests.test_reference_calibrated_capture_session.CalibratedSessionAdmissionTests.check_test_crop_watershed_bound_does_not_reject_large_parent: self.assertEqual(plan['watershed_shape'], [96, 128])` | unrecorded / — |
### CalibratedCaptureSessionTests

Fixtures: `setUpClass` at line 74.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_partial_manifest_and_bad_preparation_metadata_reject`](../../tests/test_reference_calibrated_capture_session.py#L140) | partial manifest and bad preparation metadata reject | `self.assertRaises(ValueError); self.assertRaisesRegex(ValueError, 'preparation')` | unrecorded / — |
| [`test_final_source_rehash_detects_bytes_even_with_declared_constant_stamp`](../../tests/test_reference_calibrated_capture_session.py#L168) | final source rehash detects bytes even with declared constant stamp | `self.assertRaisesRegex(ValueError, 'original bytes')` | unrecorded / — |

## tests.test_reference_calibrated_seed

Actual public composition checks, followed by explicitly injected faults.

Source: [tests/test_reference_calibrated_seed.py](../../tests/test_reference_calibrated_seed.py). Category: Recipe, replay and reference infrastructure.

### CalibratedSpecTests

Fixtures: `setUp` at line 80.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_composition_roundtrip_preserves_original_catalogue_and_selected_recipes`](../../tests/test_reference_calibrated_seed.py#L85) | composition roundtrip preserves original catalogue and selected recipes | `self.assertEqual(restored.sha256, selected.sha256); self.assertEqual(restored.seed_recipe.identity, selected.seed_recipe.identity)` | unrecorded / — |
| [`test_cancelled_preflight_and_reader_invalid_context_do_not_open_files`](../../tests/test_reference_calibrated_seed.py#L99) | cancelled preflight and reader invalid context do not open files | `self.assertRaises(ReferenceCancelledError); self.assertRaises(CheckpointCancelled)` | unrecorded / — |

## tests.test_reference_calibration_adapters

Explicit public ROI and size-scale contexts for a composed reference workflow.

Source: [tests/test_reference_calibration_adapters.py](../../tests/test_reference_calibration_adapters.py). Category: Recipe, replay and reference infrastructure.

### ReferenceCalibrationAdapterTests

Fixtures: `setUp` at line 82.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_reference_calibration_adapters.py#L177) | 3 related ReferenceCalibrationAdapterTests scenarios, each with private fixture state. | `helper: tests.test_reference_calibration_adapters.ReferenceCalibrationAdapterTests.check_test_roi_bounds_reject_before_object_construction_or_public_call: self.assertRaises(WorkerError)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_reference_calibration_adapters.py#L183) | 3 related ReferenceCalibrationAdapterTests scenarios, each with private fixture state. | `helper: tests.test_reference_calibration_adapters.ReferenceCalibrationAdapterTests.check_test_roi_is_exact_half_open_public_object_and_full_frame_rgb_adapter: np.testing.assert_array_equal(rgb_img, self.rgb[..., ::-1])` | unrecorded / — |

## tests.test_reference_capture

Public reference capture sessions, manifests and publication failures

Source: [tests/test_reference_capture.py](../../tests/test_reference_capture.py). Category: Recipe, replay and reference infrastructure.

### ReferenceCaptureContractsTests

Fixtures: `setUpClass` at line 39, `tearDownClass` at line 69.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_manifest_revision_and_strict_json_caps_reject`](../../tests/test_reference_capture.py#L115) | manifest revision and strict json caps reject | `self.assertRaises(ValueError); self.assertRaises(ValueError)` | unrecorded / — |
### PublicCaptureDesktopTests

Fixtures: `setUpClass` at line 151, `setUp` at line 156.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_capture.py#L305) | 3 related PublicCaptureDesktopTests scenarios, each with private fixture state. | `helper: tests.test_reference_capture.PublicCaptureDesktopTests.check_test_selection_controls_create_new_recipe_and_preserve_original: self.assertIsInstance(editor, QLineEdit)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_capture.py#L311) | 5 related PublicCaptureDesktopTests scenarios, each with private fixture state. | `helper: tests.test_reference_capture.PublicCaptureDesktopTests.check_test_invalid_parameter_does_not_start_worker: pool.assert_not_called()` | unrecorded / — |

## tests.test_reference_checkpoint_inspection

Reference checkpoint inspection and graph-stage evidence

Source: [tests/test_reference_checkpoint_inspection.py](../../tests/test_reference_checkpoint_inspection.py). Category: Recipe, replay and reference infrastructure.


## tests.test_reference_edge_fit

Reference-driven edge fitting and rendered evidence

Source: [tests/test_reference_edge_fit.py](../../tests/test_reference_edge_fit.py). Category: Annotation and interactive image review.

### ReferenceEdgeFitTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_edge_fit.py#L249) | 5 related ReferenceEdgeFitTests scenarios, each with private fixture state. | `helper: tests.test_reference_edge_fit.ReferenceEdgeFitTests.check_test_internal_prototype_sources_are_prominent_normal_maxima: self.assertTrue(bool(torch.all(selected[:, 6]).item()))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_edge_fit.py#L257) | 2 related ReferenceEdgeFitTests scenarios, each with private fixture state. | `helper: tests.test_reference_edge_fit.ReferenceEdgeFitTests.check_test_balanced_score_rewards_correct_support_and_rejects_cross_matches: self.assertLess(good.loss, crossed.loss)` | unrecorded / — |

## tests.test_reference_edge_strip_descriptor

Reference edge strip extraction, descriptors and overlays

Source: [tests/test_reference_edge_strip_descriptor.py](../../tests/test_reference_edge_strip_descriptor.py). Category: Annotation and interactive image review.

### ReferenceEdgeStripDescriptorTests

Fixtures: `setUpClass` at line 17.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_edge_strip_descriptor.py#L761) | 12 related ReferenceEdgeStripDescriptorTests scenarios, each with private fixture state. | `helper: tests.test_reference_edge_strip_descriptor.ReferenceEdgeStripDescriptorTests.check_test_adaptive_edge_shape_resolves_small_seeds_without_exceeding_cap: self.assertAlmostEqual(scale, 0.64)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_edge_strip_descriptor.py#L776) | 2 related ReferenceEdgeStripDescriptorTests scenarios, each with private fixture state. | `helper: tests.test_reference_edge_strip_descriptor.ReferenceEdgeStripDescriptorTests.check_test_physical_training_strip_uses_annotation_axis_not_wrong_image_tangent: self.assertEqual(len(observed), 2)` | unrecorded / — |

## tests.test_reference_edge_toolbar_revision

Edge reference toolbar controls, graph revisions and image-view state

Source: [tests/test_reference_edge_toolbar_revision.py](../../tests/test_reference_edge_toolbar_revision.py). Category: Annotation and interactive image review.

### ReferenceEdgeToolbarRevisionTests

Fixtures: `setUpClass` at line 20, `tearDownClass` at line 33.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_edge_toolbar_revision.py#L242) | 6 related ReferenceEdgeToolbarRevisionTests scenarios, each with private fixture state. | `helper: tests.test_reference_edge_toolbar_revision.ReferenceEdgeToolbarRevisionTests.check_test_toolbar_toggles_and_opacity_are_display_only: self.assertEqual(window.analyze_action.text(), 'Run pipeline')` | unrecorded / — |

## tests.test_reference_graph_persistence

Reference graph project serialization and historical identity

Source: [tests/test_reference_graph_persistence.py](../../tests/test_reference_graph_persistence.py). Category: Recipe, replay and reference infrastructure.

### ReferenceGraphPersistenceTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_graph_persistence.py#L305) | 3 related ReferenceGraphPersistenceTests scenarios, each with private fixture state. | `helper: tests.test_reference_graph_persistence.ReferenceGraphPersistenceTests.check_test_attachment_round_trip_retains_recipe_run_and_display_state: self.assertEqual(restored, attachment)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_graph_persistence.py#L311) | 4 related ReferenceGraphPersistenceTests scenarios, each with private fixture state. | `helper: tests.test_reference_graph_persistence.ReferenceGraphPersistenceTests.check_test_attachment_rejects_missing_extra_duplicate_and_stale_stage_runs: self.assertRaisesRegex(ReferenceGraphPersistenceError, 'exactly one')` | unrecorded / — |

## tests.test_reference_graph_readonly

Read-only public reference graph presentation and edit prevention

Source: [tests/test_reference_graph_readonly.py](../../tests/test_reference_graph_readonly.py). Category: Recipe, replay and reference infrastructure.

### ReferenceGraphReadOnlyTests

Fixtures: `setUpClass` at line 14.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_graph_readonly.py#L375) | 5 related ReferenceGraphReadOnlyTests scenarios, each with private fixture state. | `helper: tests.test_reference_graph_readonly.ReferenceGraphReadOnlyTests.check_test_canvas_blocks_direct_and_real_qt_edit_paths: self.assertTrue(canvas.read_only)` | unrecorded / — |

## tests.test_reference_history

Reference history revision selection and UI presentation

Source: [tests/test_reference_history.py](../../tests/test_reference_history.py). Category: Annotation and interactive image review.

### RasterUndoHistoryTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_history.py#L114) | 4 related RasterUndoHistoryTests scenarios, each with private fixture state. | `helper: tests.test_reference_history.RasterUndoHistoryTests.check_test_grouped_reference_edit_is_restored_atomically: self.assertTrue(history.record('background stroke', 'material', before, after))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_history.py#L121) | 1 related RasterUndoHistoryTests scenarios, each with private fixture state. | `helper: tests.test_reference_history.RasterUndoHistoryTests.check_test_limit_must_retain_at_least_five_commands: self.assertRaisesRegex(ValueError, 'at least five')` | unrecorded / — |

## tests.test_reference_library

Reference library aggregation, extraction, contracts and persistence

Source: [tests/test_reference_library.py](../../tests/test_reference_library.py). Category: Measurements and shape models.

### SpeciesReferenceLibraryTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_library.py#L341) | 6 related SpeciesReferenceLibraryTests scenarios, each with private fixture state. | `helper: tests.test_reference_library.SpeciesReferenceLibraryTests.check_test_extraction_uses_production_descriptor_schemas: self.assertTrue(cv2.imwrite(str(source_path), image))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_library.py#L350) | 1 related SpeciesReferenceLibraryTests scenarios, each with private fixture state. | `helper: tests.test_reference_library.SpeciesReferenceLibraryTests.check_test_publish_load_export_import_and_tamper_detection: self.assertEqual(loaded.manifest.content_sha256, pin.sha256)` | unrecorded / — |

## tests.test_reference_mask_inspection

Saved public mask stage inspection

Source: [tests/test_reference_mask_inspection.py](../../tests/test_reference_mask_inspection.py). Category: Recipe, replay and reference infrastructure.

### SelectedRecipeInspectionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_selected_values_are_allowed_but_changed_contracts_are_not`](../../tests/test_reference_mask_inspection.py#L19) | selected values are allowed but changed contracts are not | `self.assertNotEqual(selected.identity, canonical.identity); self.assertTrue(_matches_catalogue_with_selected_values(selected, canonical))` | unrecorded / — |

## tests.test_reference_mask_stage

Public mask-stage capture and checkpoint identity

Source: [tests/test_reference_mask_stage.py](../../tests/test_reference_mask_stage.py). Category: Recipe, replay and reference infrastructure.

### ReferenceMaskStageTests

Fixtures: `setUpClass` at line 87, `tearDownClass` at line 104.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_invalid_mask_values_and_frame_shape_are_rejected_before_runner`](../../tests/test_reference_mask_stage.py#L246) | invalid mask values and frame shape are rejected before runner | `self.assertRaisesRegex(ReferenceMaskStageError, 'outside uint8 0/255'); self.assertRaisesRegex(ReferenceMaskStageError, 'coordinate frame')` | unrecorded / — |
| [`test_internal_watershed_substages_are_explicitly_unsupported`](../../tests/test_reference_mask_stage.py#L272) | internal watershed substages are explicitly unsupported | `self.assertEqual(outcome.status, ReferenceMaskStageStatus.UNSUPPORTED); self.assertIn('watershed internals', outcome.run_record.reason)` | unrecorded / — |

## tests.test_reference_model_assets

Real classifier bytes survive project moves and portable snapshot restore.

Source: [tests/test_reference_model_assets.py](../../tests/test_reference_model_assets.py). Category: Recipe, replay and reference infrastructure.

### ReferenceModelAssetTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_model_assets.py#L309) | 5 related ReferenceModelAssetTests scenarios, each with private fixture state. | `helper: tests.test_reference_model_assets.ReferenceModelAssetTests.check_test_capture_preserves_history_and_deduplicates_exact_objects: self.assertFalse(loaded.issues)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_model_assets.py#L317) | 6 related ReferenceModelAssetTests scenarios, each with private fixture state. | `helper: tests.test_reference_model_assets.ReferenceModelAssetTests.check_test_invalid_or_stale_binding_never_replaces_master: self.assertRaises(InvalidProjectAnalysis)` | unrecorded / — |

## tests.test_reference_persistence

Reference region sidecars, migration, source binding and desktop lifecycle

Source: [tests/test_reference_persistence.py](../../tests/test_reference_persistence.py). Category: Persistence, export, identity and outcomes.

### ReferenceRegionStoreTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_reference_persistence.py#L347) | 6 related ReferenceRegionStoreTests scenarios, each with private fixture state. | `helper: tests.test_reference_persistence.ReferenceRegionStoreTests.check_test_invalid_shape_review_names_seed_and_required_repair: self.assertRaises(ReferenceRegionError)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_reference_persistence.py#L356) | 5 related ReferenceRegionStoreTests scenarios, each with private fixture state. | `helper: tests.test_reference_persistence.ReferenceRegionStoreTests.check_test_round_trip_preserves_material_and_instance_outputs_only: self.assertIsNotNone(loaded)` | unrecorded / — |
### ReferenceRegionMainWindowTests

Fixtures: `setUpClass` at line 368.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_persistence.py#L853) | 8 related ReferenceRegionMainWindowTests scenarios, each with private fixture state. | `helper: tests.test_reference_persistence.ReferenceRegionMainWindowTests.check_test_saved_snapshot_automatically_loads_as_immutable_applied_state: self.assertIsNotNone(key)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_persistence.py#L864) | 2 related ReferenceRegionMainWindowTests scenarios, each with private fixture state. | `helper: tests.test_reference_persistence.ReferenceRegionMainWindowTests.check_test_loose_corrected_shape_is_deferred_instead_of_rejected_against_raw_image: self.assertIn(key, window._pending_unbound_reference_bundles)` | unrecorded / — |

## tests.test_reference_preprocessing

Public preprocessing chains, capture, controls and cache identity

Source: [tests/test_reference_preprocessing.py](../../tests/test_reference_preprocessing.py). Category: Recipe, replay and reference infrastructure.

### ReferencePreprocessingContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_reference_preprocessing.py#L353) | 4 related ReferencePreprocessingContractTests scenarios, each with private fixture state. | `helper: tests.test_reference_preprocessing.ReferencePreprocessingContractTests.check_test_invalid_preprocessing_edge_is_rejected_before_runner_execution: self.assertRaisesRegex(ReferencePreprocessingError, 'dependencies')` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_reference_preprocessing.py#L360) | 2 related ReferencePreprocessingContractTests scenarios, each with private fixture state. | `helper: tests.test_reference_preprocessing.ReferencePreprocessingContractTests.check_test_full_recipe_identity_can_include_selected_downstream_values: self.assertEqual(outcome.recipe_identity, recipe.identity)` | unrecorded / — |

## tests.test_reference_recipes

Contract tests for reference recipes; fixtures here are not real recipes.

Source: [tests/test_reference_recipes.py](../../tests/test_reference_recipes.py). Category: Recipe, replay and reference infrastructure.

### ReferenceRecipeTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_recipes.py#L592) | 10 related ReferenceRecipeTests scenarios, each with private fixture state. | `helper: tests.test_reference_recipes.ReferenceRecipeTests.check_test_strict_json_round_trip_and_deep_immutability: self.assertEqual(RecipeSpec.from_json(encoded), recipe)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_recipes.py#L605) | 4 related ReferenceRecipeTests scenarios, each with private fixture state. | `helper: tests.test_reference_recipes.ReferenceRecipeTests.check_test_recipe_rejects_duplicate_missing_mistyped_and_cyclic_dependencies: self.assertRaisesRegex(RecipeContractError, 'unique')` | unrecorded / — |

## tests.test_reference_runner

Isolated PlantCV worker requests, limits, transport and failure outcomes

Source: [tests/test_reference_runner.py](../../tests/test_reference_runner.py). Category: Recipe, replay and reference infrastructure.

### ReferenceArrayProtocolTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_validation_and_recovery`](../../tests/test_reference_runner.py#L246) | 7 related ReferenceArrayProtocolTests scenarios, each with private fixture state. | `helper: tests.test_reference_runner.ReferenceArrayProtocolTests.check_test_rejects_object_nonfinite_and_oversized_arrays: self.assertRaises(ReferenceProtocolError)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_reference_runner.py#L256) | 1 related ReferenceArrayProtocolTests scenarios, each with private fixture state. | `helper: tests.test_reference_runner.ReferenceArrayProtocolTests.check_test_audited_import_compiles_verified_source_instead_of_unchecked_pyc: self.assertEqual(fake_namespace['binary'](), 'unchecked-pyc')` | unrecorded / — |
### ReferenceIdentityTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_source_tampering_is_rejected_before_worker_start`](../../tests/test_reference_runner.py#L263) | source tampering is rejected before worker start | `self.assertRaisesRegex(ReferenceIdentityError, 'source changed')` | unrecorded / — |
| [`test_environment_inventory_tampering_is_rejected`](../../tests/test_reference_runner.py#L279) | environment inventory tampering is rejected | `self.assertRaisesRegex(ReferenceIdentityError, 'differs from the hash-pinned requirements')` | unrecorded / — |
| [`test_smpts_original_backend_is_explicitly_unavailable`](../../tests/test_reference_runner.py#L296) | smpts original backend is explicitly unavailable | `self.assertEqual(status['backend'], 'smpts_reference'); self.assertFalse(status['available'])` | unrecorded / — |
| [`test_availability_probes_interpreter_and_reports_unavailable_desktop_python`](../../tests/test_reference_runner.py#L304) | availability probes interpreter and reports unavailable desktop python | `self.assertTrue(reference_status['available'], reference_status); self.assertEqual(reference_status['backend'], 'plantcv_reference')` | unrecorded / — |
### IsolatedPlantCVExecutionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_timeout_and_cancellation_remove_temporary_protocol_files`](../../tests/test_reference_runner.py#L443) | timeout and cancellation remove temporary protocol files | `helper: tests.test_reference_runner.IsolatedPlantCVExecutionTests.check_test_timeout_and_cancellation_remove_temporary_protocol_files: self.assertEqual(list(root.iterdir()), [])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_runner.py#L446) | 2 related IsolatedPlantCVExecutionTests scenarios, each with private fixture state. | `helper: tests.test_reference_runner.IsolatedPlantCVExecutionTests.check_test_missing_external_python_has_clear_availability_state: self.assertFalse(status['available'])` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_reference_runner.py#L451) | 2 related IsolatedPlantCVExecutionTests scenarios, each with private fixture state. | `helper: tests.test_reference_runner.IsolatedPlantCVExecutionTests.check_test_callback_exception_kills_reaps_and_closes_worker_process: self.assertRaisesRegex(LookupError, 'callback failed')` | unrecorded / — |

## tests.test_reference_seed_chain

Actual public seed-chain composition, controls and cancellation

Source: [tests/test_reference_seed_chain.py](../../tests/test_reference_seed_chain.py). Category: Recipe, replay and reference infrastructure.

### ReferenceSeedChainTests

Fixtures: `setUpClass` at line 35, `tearDownClass` at line 53.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_seed_chain.py#L211) | 1 related ReferenceSeedChainTests scenarios, each with private fixture state. | `helper: tests.test_reference_seed_chain.ReferenceSeedChainTests.check_test_actual_eight_public_calls_publish_bound_reloadable_chain: self.assertEqual(outcome.status, 'executed', [call.reason for call in outcome.calls])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_seed_chain.py#L215) | 3 related ReferenceSeedChainTests scenarios, each with private fixture state. | `helper: tests.test_reference_seed_chain.ReferenceSeedChainTests.check_test_source_frame_channel_and_recipe_mismatches_reject_before_calls: self.assertRaises(ValueError)` | unrecorded / — |

## tests.test_reference_session

Shared reference imports retain fresh PlantCV state and actual calculations.

Source: [tests/test_reference_session.py](../../tests/test_reference_session.py). Category: Recipe, replay and reference infrastructure.

### ReferenceSessionTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_changed_requests_match_one_shot_and_do_not_share_plantcv_state`](../../tests/test_reference_session.py#L11) | changed requests match one shot and do not share plantcv state | `self.assertEqual(answers, [[[0, 0, 255, 255]], [[0, 0, 0, 255]]]); self.assertEqual(answers[-1], json.loads(direct.stdout))` | unrecorded / — |

## tests.test_reference_setup

Pinned reference environment and source installation checks

Source: [tests/test_reference_setup.py](../../tests/test_reference_setup.py). Category: Recipe, replay and reference infrastructure.

### ReferenceSetupTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_generated_metadata_passes_runner_source_validator_when_committed_bytes_exist`](../../tests/test_reference_setup.py#L255) | generated metadata passes runner source validator when committed bytes exist | `helper: tests.test_reference_setup.ReferenceSetupTests.check_test_generated_metadata_passes_runner_source_validator_when_committed_bytes_exist: self.assertIn(url, sources_by_url)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_reference_setup.py#L258) | 3 related ReferenceSetupTests scenarios, each with private fixture state. | `helper: tests.test_reference_setup.ReferenceSetupTests.check_test_reconstructed_manifests_match_runner_hash_pins_and_all_locked_packages: self.assertEqual(len(records), MAX_SOURCE_FILES)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_setup.py#L264) | 5 related ReferenceSetupTests scenarios, each with private fixture state. | `helper: tests.test_reference_setup.ReferenceSetupTests.check_test_materialization_is_idempotent_and_missing_sources_stay_explicit_without_fetch: self.assertTrue(first.metadata_available)` | unrecorded / — |

## tests.test_reference_source_identity

Source association is preserved separately from annotation generation.

Source: [tests/test_reference_source_identity.py](../../tests/test_reference_source_identity.py). Category: Persistence, export, identity and outcomes.

### ReferenceSourceArchiveTests

Fixtures: `setUp` at line 26.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_source_identity.py#L134) | 4 related ReferenceSourceArchiveTests scenarios, each with private fixture state. | `helper: tests.test_reference_source_identity.ReferenceSourceArchiveTests.check_test_v7_round_trip_preserves_known_generation_and_unknown_history: self.assertEqual(loaded.archive_source_sha256, self.digest)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_source_identity.py#L141) | 3 related ReferenceSourceArchiveTests scenarios, each with private fixture state. | `helper: tests.test_reference_source_identity.ReferenceSourceArchiveTests.check_test_save_rejects_bad_identity_before_rasters_or_destination_creation: self.assertRaises(ReferenceRegionError)` | unrecorded / — |
### ReferenceSourceAnalysisTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_source_identity.py#L290) | 4 related ReferenceSourceAnalysisTests scenarios, each with private fixture state. | `helper: tests.test_reference_source_identity.ReferenceSourceAnalysisTests.check_test_saved_project_batch_forwards_archive_and_generation_bindings: self.assertEqual(len(run_project(root, master, root / 'output')), 1)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_source_identity.py#L297) | 2 related ReferenceSourceAnalysisTests scenarios, each with private fixture state. | `helper: tests.test_reference_source_identity.ReferenceSourceAnalysisTests.check_test_analyze_path_rejects_mismatch_before_decoding_and_forwards_verified_facts: self.assertRaises(ValueError)` | unrecorded / — |
### ReferenceSourceDesktopTests

Fixtures: `setUpClass` at line 305.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_worker_detaches_and_forwards_reference_identity_metadata`](../../tests/test_reference_source_identity.py#L310) | worker detaches and forwards reference identity metadata | `self.assertEqual(analyzer.call_args.kwargs['reference_archive_source_sha256'], 'a' * 64); self.assertEqual(analyzer.call_args.kwargs['reference_annotation_snapshot_source_sha256'], 'b' * 64)` | unrecorded / — |

## tests.test_reference_stage

Public stage capture, graph adaptation and checkpoint identity

Source: [tests/test_reference_stage.py](../../tests/test_reference_stage.py). Category: Recipe, replay and reference infrastructure.

### ReferenceStageContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_stage.py#L450) | 3 related ReferenceStageContractTests scenarios, each with private fixture state. | `helper: tests.test_reference_stage.ReferenceStageContractTests.check_test_exact_replay_bindings_are_required_before_runner_call: self.assertRaisesRegex(ReferenceStageError, 'source SHA')` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_stage.py#L456) | 7 related ReferenceStageContractTests scenarios, each with private fixture state. | `helper: tests.test_reference_stage.ReferenceStageContractTests.check_test_stale_recipe_parameter_source_and_environment_pins_are_rejected: self.assertRaisesRegex(ReferenceStageError, 'expected recipe identity')` | unrecorded / — |

## tests.test_reference_texture_collage

Reference texture collage layout and image rendering

Source: [tests/test_reference_texture_collage.py](../../tests/test_reference_texture_collage.py). Category: Annotation and interactive image review.

### ReferenceTextureCollageTests

Fixtures: `setUpClass` at line 19.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_texture_collage.py#L158) | 3 related ReferenceTextureCollageTests scenarios, each with private fixture state. | `helper: tests.test_reference_texture_collage.ReferenceTextureCollageTests.check_test_edge_strip_markers_use_source_pixel_offsets_and_lengths: self.assertAlmostEqual(centre.x1(), 18.6)` | unrecorded / — |

## tests.test_reference_undo

Reference annotation undo/redo and independent history

Source: [tests/test_reference_undo.py](../../tests/test_reference_undo.py). Category: Annotation and interactive image review.

### ReferenceUndoUiTests

Fixtures: `setUpClass` at line 23, `setUp` at line 41.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_undo.py#L458) | 7 related ReferenceUndoUiTests scenarios, each with private fixture state. | `helper: tests.test_reference_undo.ReferenceUndoUiTests.check_test_material_exclusivity_multi_undo_returns_to_clean_baseline: self.assertFalse(self.window.reference_undo_button.isEnabled())` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_undo.py#L468) | 1 related ReferenceUndoUiTests scenarios, each with private fixture state. | `helper: tests.test_reference_undo.ReferenceUndoUiTests.check_test_retired_boundary_reference_edits_are_ignored: self.assertNotIn(key, self.window._reference_masks_dirty)` | unrecorded / — |

## tests.test_reference_watershed_capture

Hook semantics, strict transport and actual pinned public watershed products.

Source: [tests/test_reference_watershed_capture.py](../../tests/test_reference_watershed_capture.py). Category: Recipe, replay and reference infrastructure.

### CaptureHookTests

Fixtures: `setUp` at line 51.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_watershed_capture.py#L178) | 3 related CaptureHookTests scenarios, each with private fixture state. | `helper: tests.test_reference_watershed_capture.CaptureHookTests.check_test_one_actual_target_call_owned_products_prior_profile_and_trace_restore: self.assertIs(sys.getprofile(), prior)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_watershed_capture.py#L184) | 4 related CaptureHookTests scenarios, each with private fixture state. | `helper: tests.test_reference_watershed_capture.CaptureHookTests.check_test_missing_local_retains_labels_but_no_partial_capture: self.assertEqual(result.metadata['status'], 'unavailable')` | unrecorded / — |
### CaptureProtocolTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_reference_watershed_capture.py#L244) | 2 related CaptureProtocolTests scenarios, each with private fixture state. | `helper: tests.test_reference_watershed_capture.CaptureProtocolTests.check_test_empty_coordinate_record_reconstructs_exact_array_without_relaxing_npz: self.assertEqual(arrays['coordinates'].shape, (0, 2))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_reference_watershed_capture.py#L249) | 2 related CaptureProtocolTests scenarios, each with private fixture state. | `helper: tests.test_reference_watershed_capture.CaptureProtocolTests.check_test_malformed_empty_sentinels_and_partial_capture_reject: self.assertRaisesRegex(protocol.ReferenceProtocolError, 'empty-coordinate')` | unrecorded / — |
### ActualPublicCaptureTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_post_call_runtime_drift_withholds_all_outputs`](../../tests/test_reference_watershed_capture.py#L290) | post call runtime drift withholds all outputs | `self.assertRaisesRegex(protocol.ReferenceExecutionError, 'runtime identity changed')` | unrecorded / — |

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
| [`test_workflow_lifecycle`](../../tests/test_review_workflows.py#L106) | 4 related ReviewWorkflowTests scenarios, each with private fixture state. | `helper: tests.test_review_workflows.ReviewWorkflowTests.check_test_batch_project_does_not_require_an_optional_species_library: self.assertEqual(run_project(root, master, root / 'results'), [root / 'result'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_review_workflows.py#L113) | 1 related ReviewWorkflowTests scenarios, each with private fixture state. | `helper: tests.test_review_workflows.ReviewWorkflowTests.check_test_annotation_redo_and_byte_budget: np.testing.assert_array_equal(undo.rasters[0], before)` | unrecorded / — |

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
| [`test_batch_rejected_cancelled_and_device_failure_publish_terminal_without_companions`](../../tests/test_run_outcomes.py#L255) | batch rejected cancelled and device failure publish terminal without companions | `self.assertLogs('seedvision.export.run_outcomes', level='ERROR'); self.assertRaises(raised)` | unrecorded / — |

## tests.test_seed_crop_artifacts

Seed crop artifacts, export identity and portable review

Source: [tests/test_seed_crop_artifacts.py](../../tests/test_seed_crop_artifacts.py). Category: Persistence, export, identity and outcomes.

### SeedCropArtifactTests

Fixtures: `setUpClass` at line 46.

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

Fixtures: `setUpClass` at line 17, `setUp` at line 25.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_selected_instance_view.py#L172) | 4 related SelectedInstanceImageViewTests scenarios, each with private fixture state. | `helper: tests.test_selected_instance_view.SelectedInstanceImageViewTests.check_test_selected_only_overlay_is_cropped_and_excludes_other_ids: self.assertIsNotNone(item)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_selected_instance_view.py#L179) | 1 related SelectedInstanceImageViewTests scenarios, each with private fixture state. | `helper: tests.test_selected_instance_view.SelectedInstanceImageViewTests.check_test_label_replacement_and_committed_edit_invalidate_bounds_cache: self.assertEqual(self.view._instance_bounds(1), (120, 100, 130, 110))` | unrecorded / — |
### SelectedInstanceMainWindowTests

Fixtures: `setUpClass` at line 186.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_show_dropdown_plumbs_to_view_and_tracks_analysis_availability`](../../tests/test_selected_instance_view.py#L194) | show dropdown plumbs to view and tracks analysis availability | `self.assertTrue(image.save(str(image_path))); self.assertEqual(selector.currentData(), 'all')` | unrecorded / — |

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

## tests.test_smpts_paper

Soybean paper method calculations and explicitly supported boundaries

Source: [tests/test_smpts_paper.py](../../tests/test_smpts_paper.py). Category: Numerical primitives and public-source parity.

### PaperPlanningTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_full_fixture_admitted_and_work_cuda_limits_are_explicit`](../../tests/test_smpts_paper.py#L47) | full fixture admitted and work cuda limits are explicit | `self.assertLess(paper.component_work(pixels), paper.MAX_COMPONENT_WORK); self.assertEqual(paper.retained_bound(pixels), 32 * pixels + 208 * 1024 * 1024)` | unrecorded / — |
| [`test_inventory_and_contact_controls_are_exact`](../../tests/test_smpts_paper.py#L58) | inventory and contact controls are exact | `self.assertEqual(paper.select_contacts(rows, (3, 2), 5), (1,)); self.assertEqual(paper.select_contacts(rows, (3, 2), 6), ())` | unrecorded / — |
### PaperCudaTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_complete_declared_variant_matches_isolated_pinned_opencv_substeps`](../../tests/test_smpts_paper.py#L332) | complete declared variant matches isolated pinned opencv substeps | `helper: tests.test_smpts_paper.PaperCudaTests.check_test_complete_declared_variant_matches_isolated_pinned_opencv_substeps: self.assertEqual(run.returncode, 0, run.stdout + run.stderr)` | unrecorded / — |
| [`test_workflow_lifecycle`](../../tests/test_smpts_paper.py#L335) | 6 related PaperCudaTests scenarios, each with private fixture state. | `helper: tests.test_smpts_paper.PaperCudaTests.check_test_components_match_opencv8_on_islands_diagonals_chains_and_random_masks: self.assert_gpu(actual['labels'], expected)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_smpts_paper.py#L344) | 4 related PaperCudaTests scenarios, each with private fixture state. | `helper: tests.test_smpts_paper.PaperCudaTests.check_test_component_limit_and_failed_convergence_publish_nothing: self.assertEqual(len(actual['components']), 4096)` | unrecorded / — |

## tests.test_species_library_dialog

Species reference library desktop editing

Source: [tests/test_species_library_dialog.py](../../tests/test_species_library_dialog.py). Category: Measurements and shape models.

### SpeciesLibraryDialogTests

Fixtures: `setUpClass` at line 18.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_dialog_presents_three_explicit_workflow_steps`](../../tests/test_species_library_dialog.py#L23) | dialog presents three explicit workflow steps | `self.assertEqual(dialog.tabs.count(), 3); self.assertEqual(dialog.tabs.tabText(0), '1. Use a library')` | unrecorded / — |
| [`test_source_selection_summary_and_fork_guidance_are_immediate`](../../tests/test_species_library_dialog.py#L51) | source selection summary and fork guidance are immediate | `self.assertIn('0 of 1', dialog.source_selection_label.text()); self.assertIn('Select an installed version', dialog.retained_sources_label.text())` | unrecorded / — |

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
| [`test_catalogue_fixture_reuses_only_templates_and_restores_lazy_import_aliases`](../../tests/test_test_runner.py#L156) | catalogue fixture reuses only templates and restores lazy import aliases | `self.assertIs(runner.build_recipe_catalogue(), first); self.assertRaises(runner.NativeRecipeError)` | unrecorded / — |
| [`test_invalid_module_exits_before_execution_and_reports_discovery_errors`](../../tests/test_test_runner.py#L181) | invalid module exits before execution and reports discovery errors | `self.assertEqual(process.returncode, 2, process.stderr); self.assertEqual(record['tests'], 0)` | unrecorded / — |
| [`test_partition_is_complete_and_fixture_timings_preserve_lifecycle`](../../tests/test_test_runner.py#L211) | partition is complete and fixture timings preserve lifecycle | `self.assertEqual({k: len(v) for k, v in groups.items()}, {'cpu-qt': 1, 'cuda': 1, 'acceptance': 1}); self.assertEqual({t.id() for v in groups.values() for t in v}, {t.id() for t in items})` | unrecorded / — |
| [`test_fixture_bundle_checks_hashes_and_never_overwrites_reviewed_files`](../../tests/test_test_runner.py#L241) | fixture bundle checks hashes and never overwrites reviewed files | `self.assertFalse(missing['passed']); self.assertTrue(restored['passed'], restored)` | unrecorded / — |

## tests.test_threshold_conformance

Public/native threshold checkpoint comparison and conformance

Source: [tests/test_threshold_conformance.py](../../tests/test_threshold_conformance.py). Category: Recipe, replay and reference infrastructure.

### ThresholdConformanceContractTests

Fixtures: no class-local setup/teardown methods.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_input_parameter_provenance_and_implementation_mismatches_precede_both_runs`](../../tests/test_threshold_conformance.py#L264) | input parameter provenance and implementation mismatches precede both runs | `self.assertRaisesRegex(ThresholdPairError, 'parameters differ'); self.assertRaisesRegex(ThresholdPairError, 'source')` | unrecorded / — |
| [`test_cancelled_pair_retains_distinct_failed_states_and_cannot_pass`](../../tests/test_threshold_conformance.py#L331) | cancelled pair retains distinct failed states and cannot pass | `self.assertEqual(run.call_count, 0); self.assertEqual(report['pair_status'], 'not_evaluated')` | unrecorded / — |
| [`test_report_publication_is_bounded_content_addressed_and_no_overwrite`](../../tests/test_threshold_conformance.py#L352) | report publication is bounded content addressed and no overwrite | `self.assertEqual(first, second); self.assertEqual(first[0].read_bytes(), json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8'))` | unrecorded / — |
| [`test_missing_reference_prerequisites_and_cuda_are_reported_as_unavailable`](../../tests/test_threshold_conformance.py#L365) | missing reference prerequisites and cuda are reported as unavailable | `self.assertEqual(outcome.reference_outcome.run_record.state, StageRunState.UNAVAILABLE); self.assertEqual(outcome.native_outcome.run_record.state, StageRunState.UNAVAILABLE)` | unrecorded / — |

## tests.test_trait_classifier

Software oracles for trait classification; all truth is explicitly synthetic.

Source: [tests/test_trait_classifier.py](../../tests/test_trait_classifier.py). Category: Learning and trait analysis.

### TraitClassifierTests

Fixtures: `setUpClass` at line 26.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_trait_classifier.py#L286) | 8 related TraitClassifierTests scenarios, each with private fixture state. | `helper: tests.test_trait_classifier.TraitClassifierTests.check_test_synthetic_grouped_training_predicts_supplied_definition_ids: self.assertEqual([x['predicted_trait_id'] for x in result['predictions']], ['dark', 'dark', 'light', 'light'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_trait_classifier.py#L297) | 9 related TraitClassifierTests scenarios, each with private fixture state. | `helper: tests.test_trait_classifier.TraitClassifierTests.check_test_group_physical_and_example_overlap_reject_before_optimization: self.assertRaisesRegex(ValueError, f'overlap by {key}')` | unrecorded / — |

## tests.test_trait_pipeline

Native selection/cache/report witnesses for the frozen trait classifier.

Source: [tests/test_trait_pipeline.py](../../tests/test_trait_pipeline.py). Category: Learning and trait analysis.

### TraitPipelineTests

Fixtures: `setUpClass` at line 28, `setUp` at line 32.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_trait_pipeline.py#L412) | 10 related TraitPipelineTests scenarios, each with private fixture state. | `helper: tests.test_trait_pipeline.TraitPipelineTests.check_test_selected_method_preference_is_shared_and_filters_disabled_branches: self.assertEqual(select_instances(**values, enabled=enabled), select_result(analysis, enabled))` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_trait_pipeline.py#L425) | 8 related TraitPipelineTests scenarios, each with private fixture state. | `helper: tests.test_trait_pipeline.TraitPipelineTests.check_test_label_corrected_pixels_and_support_changes_invalidate_features: self.assertIsNot(old, self.cache['trait_classifier.features'])` | unrecorded / — |

## tests.test_trait_review_ui

Actual frozen CUDA trait predictions must reach the native review widgets.

Source: [tests/test_trait_review_ui.py](../../tests/test_trait_review_ui.py). Category: Learning and trait analysis.

### TraitReviewUiTests

Fixtures: `setUpClass` at line 24, `setUp` at line 31.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_trait_review_ui.py#L170) | 1 related TraitReviewUiTests scenarios, each with private fixture state. | `helper: tests.test_trait_review_ui.TraitReviewUiTests.check_test_actual_mixed_predictions_definitions_and_scores_reach_review_without_refitting: self.assertNotEqual(*confidences)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_trait_review_ui.py#L174) | 2 related TraitReviewUiTests scenarios, each with private fixture state. | `helper: tests.test_trait_review_ui.TraitReviewUiTests.check_test_actual_absent_invalid_and_outside_support_abstentions_stay_explanatory: self.assertEqual(set(observed['rows']), {1, 2})` | unrecorded / — |

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

Fixtures: `setUp` at line 48.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_trait_training.py#L322) | 7 related TraitTrainingTests scenarios, each with private fixture state. | `helper: tests.test_trait_training.TraitTrainingTests.check_test_compact_training_atomic_reload_and_actual_prediction: self.assertEqual(model.model_sha256, report['model_sha256'])` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_trait_training.py#L332) | 7 related TraitTrainingTests scenarios, each with private fixture state. | `helper: tests.test_trait_training.TraitTrainingTests.check_test_manifest_and_controls_reject_before_any_raster_or_optimizer: self.assertRaises(ValueError)` | unrecorded / — |
### TraitTrainingUITests

Fixtures: `setUpClass` at line 346, `setUp` at line 350.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_trait_training.py#L506) | 3 related TraitTrainingUITests scenarios, each with private fixture state. | `helper: tests.test_trait_training.TraitTrainingUITests.check_test_dialog_snapshots_manifest_controls_and_activation_choice: self.assertEqual(request.configuration, TraitTrainingConfiguration(17, 0.04, 0.2, (0.2, 5.0), 13, 0.7))` | unrecorded / — |
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
| [`test_scan_sees_the_interface`](../../tests/test_ui_text_limits.py#L56) | scan sees the interface | `self.assertGreater(sum((1 for _ in constant_strings(COMMAND_CALLS))), 150); self.assertGreater(sum((1 for _ in constant_strings({'setToolTip'}))), 100)` | unrecorded / — |

## tests.test_visualization_layers

Analysis visualization layers, overlays, crops and display contracts

Source: [tests/test_visualization_layers.py](../../tests/test_visualization_layers.py). Category: Image analysis, calibration and rendering.

### AnalysisLayerTests

Fixtures: `setUpClass` at line 1177.

| Test method / source | Functionality exercised | Assertion evidence | Last full run |
| --- | --- | --- | --- |
| [`test_workflow_lifecycle`](../../tests/test_visualization_layers.py#L2421) | 39 related AnalysisLayerTests scenarios, each with private fixture state. | `helper: tests.test_visualization_layers.AnalysisLayerTests.check_test_reference_seed_traits_are_isolated_normalized_material_models: self.assertTrue(products.profile.coat_model_available)` | unrecorded / — |
| [`test_validation_and_recovery`](../../tests/test_visualization_layers.py#L2463) | 2 related AnalysisLayerTests scenarios, each with private fixture state. | `helper: tests.test_visualization_layers.AnalysisLayerTests.check_test_missing_other_probabilities_render_black_not_an_unrelated_layer: self.assertFalse(np.any(rgba[..., :3]))` | unrecorded / — |
