"""Explicit test selections; full discovery is always the union, unfiltered.

CPU/Qt is an allowlist, not an inference from skip decorators. CUDA contains
every remaining ID, so new tests cannot disappear from full CI. Core
intentionally overlaps the two exhaustive partition groups.
"""

CORE_TARGETS = (
    'tests.test_pipeline',
    'tests.test_analysis_settings_persistence',
    'tests.test_project_analysis_persistence',
    'tests.test_pipeline_workflow_ui',
)

CPU_QT_TARGETS = (
    'tests.test_deferred_callbacks',
    'tests.test_pipeline_documents',
    'tests.test_pipeline_workspace_files',
    'tests.test_default_pipeline_copies',
    'tests.test_image_list_controls',
    'tests.test_catalogue_nodes',
    'tests.test_catalogue_nodes_ui',
    'tests.test_pipeline',
    'tests.test_pipeline_cable_routing',
    'tests.test_pipeline_inspector_actions',
    'tests.test_pipeline_ui',
    'tests.test_pipeline_workflow_ui',
    'tests.test_analysis_settings_persistence',
    'tests.test_project_analysis_persistence',
    'tests.test_project_analysis_ui',
    'tests.test_reference_history',
    'tests.test_reference_library',
    'tests.test_reference_persistence',
    'tests.test_reference_undo',
    'tests.test_annotation_tools',
    'tests.test_annotation_centres',
    'tests.test_annotation_edge_sources',
    'tests.test_annotation_shape_revision',
    'tests.test_edge_trace_commit',
    'tests.test_instance_annotation_tiling',
    'tests.test_instance_continuity',
    'tests.test_instance_continuity_ui',
    'tests.test_instance_mask_import',
    'tests.test_instance_mask_import_ui',
    'tests.test_manual_seed_centre_persistence',
    'tests.test_annotation_transfer',
    'tests.test_manual_seed_centre_ui',
    'tests.test_shape_guided_fill',
    'tests.test_shape_guided_ui',
    'tests.test_selected_instance_view',
    'tests.test_measurements',
    'tests.test_measurement_descriptions',
    'tests.test_measurement_description_ui',
    'tests.test_measurement_scale_safety',
    'tests.test_seed_shape_model',
    'tests.test_conditioned_shape_pooling',
    'tests.test_local_shape_units',
    'tests.test_physical_shape_fit_availability',
    'tests.test_shape_pooling_overflow',
    'tests.test_shape_scoring_availability',
    'tests.test_shape_update_availability',
    'tests.test_shape_view_normalization',
    'tests.test_node_optimization',
    'tests.test_node_optimization_ui',
    'tests.test_bootstrap',
    'tests.test_entrypoint',
    'tests.test_diagnostics',
    'tests.test_icon_asset',
    'tests.test_resources',
    'tests.test_benchmark_manifest',
    'tests.test_benchmark_instances',
    'tests.test_test_runner',
    'tests.test_node_vocabulary',
    'tests.test_ui_text_limits',
    'tests.test_gui_layout',
)

# Mixed desktop/contract modules contain a few real CUDA calculations. Keep
# those in the CUDA partition; do not add capability skips to make CI green.
CUDA_TARGETS = (
    'tests.test_annotation_shape_revision.ShapeRevisionTests.test_curvature_prior_changes_only_trace_and_downstream_cache',
    'tests.test_pipeline_ui.PipelineCanvasTests.test_calibration_nodes_render_references_and_metric_scale',
)

def flatten(suite):
    import unittest
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def matches(identifier, targets):
    return any(identifier == name or identifier.startswith(name + '.') for name in targets)


def partition(tests):
    groups = {'cpu-qt': [], 'cuda': []}
    identifiers = set()
    for test in tests:
        name = test.id()
        if name in identifiers:
            raise ValueError('Duplicate discovered test ID: ' + name)
        identifiers.add(name)
        cpu = matches(name, CPU_QT_TARGETS) and not matches(name, CUDA_TARGETS)
        groups['cpu-qt' if cpu else 'cuda'].append(test)
    # A misspelled/deleted allowlist entry must not silently shrink the CPU job.
    for target in CPU_QT_TARGETS + CUDA_TARGETS:
        if not any(matches(name, (target,)) for name in identifiers):
            raise ValueError('Selection target resolves to no discovered tests: ' + target)
    return groups
