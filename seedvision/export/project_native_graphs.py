"""Execute saved native working graphs alongside existing instance reports."""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import shutil
import tempfile

from seedvision.benchmarking.native_recipe import (
    NativeRecipeCancelled, run_native_recipe_path,
)
from seedvision.benchmarking.recipes import RecipeProvenanceMode
from seedvision.export.native_results import export_native_recipe_result, _json_bytes
from seedvision.persistence.reference_graphs import validate_reference_graphs
from seedvision.persistence.native_recipes import validate_native_recipes
from seedvision.persistence.reference_regions import file_sha256
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.persistence import correction_inputs as correction_assets
from seedvision.persistence.project_analysis import ProjectImageRecord
from seedvision.export import method_qc as qc, recipe_attempt_qc as attempts, recipe_batch_qc as batch_qc
from seedvision.persistence.method_qc_history import retain_method_qc
from seedvision.benchmarking.native_correction_recipe import NativeCorrectionCancelled


def export_project_node_instances(instances, image_path, directory, *, expected_source_sha256,
                                 project_root=None, reference_model_assets=(), correction_bindings=(),
                                 project_image_record=None, default_context=None, cancel=None):
    """Publish independent catalogue calculations with their own caches/scopes.

    This companion ledger preserves repeated instances even when their canonical
    recipe identities match. Canonical recipe acceptance and its ledger remain
    unchanged. A failed calculation gets an explicit row, never a passing archive.
    """
    from seedvision.pipeline.catalogue_nodes import validate_node_instances, execute_instance
    from seedvision.timing import AnalysisCancelled
    instances = validate_node_instances(instances)
    if not instances:
        return None
    destination = Path(directory).absolute()
    if destination.exists():
        raise FileExistsError(f'Node instance destination already exists: {destination}')
    if file_sha256(image_path) != expected_source_sha256:
        raise ValueError('Source changed before independent catalogue execution.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.node-instances-', suffix='.incomplete', dir=destination.parent))
    rows = []
    try:
        for index, instance in enumerate(instances):
            if cancel is not None and cancel():
                raise NativeRecipeCancelled('Catalogue instance batch cancelled.')
            if file_sha256(image_path) != expected_source_sha256:
                raise ValueError('Source changed during independent catalogue execution.')
            row = {'instance_id': instance.instance_id, 'catalogue_key': instance.catalogue_key,
                   'target_node': instance.target_node, 'report': None, 'status': 'not_run', 'reason': ''}
            cache = PipelineAnalysisCache()
            result = prepared = None
            try:
                if instance.workflow is not None:
                    recipe = instance.workflow.recipe
                    if recipe.recipe_id == correction_assets.RECIPE_ID:
                        key = None if project_image_record is None else (
                            project_image_record.identifier, recipe.recipe_id, recipe.recipe_version,
                            recipe.identity.recipe_sha256, instance.instance_id)
                        binding = next((item for item in correction_bindings if item.key == key), None)
                        if binding is None or project_root is None:
                            raise ValueError('No verified per-image correction inputs are saved for this instance.')
                        from seedvision.pipeline.catalogue_nodes import execute_bound_correction
                        result, prepared = execute_bound_correction(instance, project_root, binding,
                            project_image_record, image_path, cache=cache, cancel=cancel)
                    else:
                        from seedvision.persistence.reference_models import model_asset_paths
                        paths = {} if project_root is None else model_asset_paths(project_root, recipe, reference_model_assets)
                        result = execute_instance(instance, image_path, cache=cache, cancel=cancel, model_assets=paths)
                    if result.source_sha256 != expected_source_sha256:
                        raise ValueError('Node instance result source differs from the project image.')
                    report = export_native_recipe_result(staging / f'instance-{index:03d}', recipe, result, cancel=cancel)
                    target = result.node(instance.target_node)
                    row['status'], row['reason'] = target.status, target.reason
                else:
                    from seedvision.export.results import result_report, export_report
                    from seedvision.pipeline.catalogue_nodes import default_graph, instance_nodes
                    from seedvision.persistence.analysis_settings import analysis_settings_profile_to_payload
                    result = execute_instance(instance, image_path, cache=cache, cancel=cancel,
                                              default_context=default_context)
                    graph = default_graph(instance)
                    report_data = result_report(result, enabled={identifier for identifier in instance_nodes(instance)
                                                                if graph.node(identifier).enabled},
                        settings=analysis_settings_profile_to_payload(instance.default_settings),
                        execution_provenance={'route': 'catalogue_node_batch', 'instance_id': instance.instance_id,
                                              'target_node': instance.target_node})
                    folder = export_report(staging / f'instance-{index:03d}', result, report_data)
                    report = folder / 'results.json'
                    row['status'] = report_data['state']
                row['report'] = report.relative_to(staging).as_posix()
            except (NativeRecipeCancelled, NativeCorrectionCancelled, AnalysisCancelled):
                raise
            except Exception as error:
                row.update(status='failed', report=None, reason=f'{type(error).__name__}: {error}'[:4096])
            finally:
                result = prepared = None
                cache.values.clear()
            rows.append(row)
        if cancel is not None and cancel():
            raise NativeRecipeCancelled('Catalogue instance batch cancelled before publication.')
        if file_sha256(image_path) != expected_source_sha256:
            raise ValueError('Source changed before independent catalogue publication.')
        payload = {'format': 'seedfiddle.catalogue-node-instances', 'version': 1,
                   'source_sha256': expected_source_sha256, 'instances': rows, 'scientifically_validated': False}
        (staging / 'node-instances.json').write_bytes(_json_bytes(payload))
        staging.rename(destination)
        return destination / 'node-instances.json'
    except BaseException:
        _remove_owned_directory(staging, destination.parent)
        raise


def _record_row_attempt(row, request, *, state, phase, error=None, executor=None, result=None):
    try:
        quality = attempts.recipe_attempt_qc(request, state=state, phase=phase, detail=row['reason'],
            exception=error, invoked=executor is not None, invoked_executor=executor,
            result_returned=result is not None, returned_result=result)
        row['method_qc'] = batch_qc.embedded_row_qc(quality)
    except Exception as recording_error:
        row['method_qc'] = None
        row['qc_recording_error'] = f'{type(recording_error).__name__}: {recording_error}'[:4096]
        if error is not None:
            error.add_note('Batch row QC capture failed: '+row['qc_recording_error'])


def _remove_owned_directory(directory, parent):
    if not directory.exists():
        return False
    info = directory.lstat()
    if (directory.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400
            or directory.resolve().parent != parent.resolve()):
        raise ValueError('Native graph cleanup escaped its owned staging directory.')
    shutil.rmtree(directory)
    return True


def _retain_rows(rows, root, owner):
    for index, row in enumerate(rows):
        quality = batch_qc.validate_row_qc(row, index)
        if quality is not None:
            retain_method_qc(root, quality, project_path=owner)


def _selected_inputs(recipe, image_path, model_paths):
    selected = {'source_image':image_path}
    for stage in recipe.stages:
        if stage.operation_id not in {'external.plantcv.kmeans_centres','external.plantcv.naive_bayes_pdf'}:
            continue
        selection = recipe.parameters_for_runner(stage.stage_id).get('model_source')
        path = model_paths.get(stage.stage_id)
        if path is None and isinstance(selection, Mapping) and type(selection.get('path')) is str:
            path = selection['path']
        if path is not None:
            selected['model.'+stage.stage_id] = path
    return selected


def export_project_native_graphs(attachments, image_path, directory, *,
                                expected_source_sha256, cache=None, cancel=None,
                                project_root=None, reference_model_assets=(), native_recipes=(),
                                correction_bindings=(), project_image_record=None, project_path=None,
                                pipeline_workflow=None):
    """Run exact saved native recipes and atomically publish their execution log.

    Original/reference graphs remain explicitly unavailable here. They require
    their declared original executor rather than a substituted CUDA backend.
    Historical graph stage records stay metadata-only and are never rewritten.
    """
    attachments = validate_reference_graphs(attachments)
    native_recipes = validate_native_recipes(native_recipes)
    if pipeline_workflow is not None:
        from seedvision.pipeline.workflows import PipelineWorkflow, validate_workflow
        if not isinstance(pipeline_workflow, PipelineWorkflow):
            raise TypeError('pipeline_workflow must be a validated PipelineWorkflow.')
        validate_workflow(pipeline_workflow)
    correction_bindings = correction_assets.validate_correction_binding_records(correction_bindings)
    bindings_by_key = {binding.key: binding for binding in correction_bindings}
    if not attachments and not native_recipes and pipeline_workflow is None:
        return None
    if cache is None:
        cache = PipelineAnalysisCache()
    if not isinstance(cache, PipelineAnalysisCache):
        raise TypeError('cache must be a PipelineAnalysisCache.')
    if project_path is not None and project_root is None:
        raise ValueError('Project QC ownership requires the explicitly selected workspace root.')
    entries = tuple((attachment.recipe, attachment, None) for attachment in attachments)
    entries += tuple((recipe, None, None) for recipe in native_recipes)
    if pipeline_workflow is not None:
        entries += ((pipeline_workflow.recipe, None, pipeline_workflow),)
    destination = Path(directory).absolute()
    batch_request = batch_qc.capture_recipe_batch_request(tuple(recipe for recipe,_,_ in entries),
        image_path, destination, expected_source_sha256,
        None if project_image_record is None else project_image_record.identifier)
    bounded_source = None
    if isinstance(project_image_record, ProjectImageRecord):
        for binding in correction_bindings:
            if binding.image_id != project_image_record.identifier:
                continue
            try:
                correction_assets.validate_correction_bindings((binding,), (project_image_record,),
                                                                attachments, native_recipes,
                                                                pipeline_workflow=pipeline_workflow)
                source = next(asset for asset in binding.inputs if asset.role == 'source_rgb')
                if source.source_sha256 != expected_source_sha256:
                    continue
                # Establish the byte count before using this descriptor as a
                # shared guard. An unusable selection fails only its own row.
                correction_assets.copy_verified_input(image_path, source)
            except (ValueError, OSError):
                continue  # Invalid selections still receive their own failed row.
            bounded_source = source
            break
    def check_source():
        if bounded_source is not None:
            correction_assets.copy_verified_input(image_path, bounded_source)
        elif file_sha256(image_path) != expected_source_sha256:
            raise ValueError('Project image bytes changed before native graph publication or execution.')
    staging = None
    correction_prepared = None
    executed_corrections = []
    rows = []
    phase, current_row_index = 'admission', None
    try:
        if destination.exists():
            raise FileExistsError(f'Native graph destination already exists: {destination}')
        check_source()
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.native-graphs-', suffix='.incomplete', dir=destination.parent))
        for index, (recipe, attachment, workflow) in enumerate(entries):
            phase, current_row_index = 'between_rows', None
            if cancel is not None and cancel():
                raise NativeRecipeCancelled('Project native graph execution cancelled.')
            phase, current_row_index = 'recipe_row', index
            row = {'kind': 'pipeline_workflow' if workflow is not None else 'historical_graph' if attachment is not None else 'working_recipe',
                'attachment_index': index if attachment is not None else None,
                'working_recipe_index': index-len(attachments) if attachment is None and workflow is None else None,
                'recipe_identity': recipe.identity.to_dict(),
                'historical_stage_runs': [record.to_dict() for record in attachment.stage_runs] if attachment else [],
                'historical_checkpoint_data': attachment.checkpoint_data_status if attachment else 'not_recorded',
                'selected_stage_id': attachment.selected_stage_id if attachment else None,
                'display': {'bundle_cables': attachment.bundle_cables,
                            'route_around_nodes': attachment.route_around_nodes} if attachment else None,
                'report': None, 'nodes': [], 'reason': '', 'correction_binding': None, 'method_qc':None}
            if workflow is not None:
                row['pipeline_workflow'] = workflow.to_dict()
            request = attempts.capture_recipe_attempt_request(recipe, {'source_image':image_path})
            from seedvision.benchmarking.native_smpts_recipe import is_executable_paper_recipe
            if workflow is None and recipe.provenance.mode is not RecipeProvenanceMode.NATIVE_COMPATIBLE and not is_executable_paper_recipe(recipe):
                row.update(status='unavailable', reason='Saved graph requires its declared original/reference executor.')
                _record_row_attempt(row, request, state='unavailable', phase='batch_selection')
                rows.append(row)
                continue
            result = None
            row_phase, executor = 'batch_selection', None
            try:
                check_source()
                options = {}
                if reference_model_assets:
                    from seedvision.persistence.reference_models import model_asset_paths
                    if project_root is None:
                        raise ValueError('Project model transport requires its workspace root.')
                    options['model_asset_paths'] = model_asset_paths(project_root, recipe, reference_model_assets)
                if recipe.recipe_id == correction_assets.RECIPE_ID:
                    key = None if project_image_record is None else (
                        project_image_record.identifier, recipe.recipe_id, recipe.recipe_version,
                        recipe.identity.recipe_sha256)
                    binding = bindings_by_key.get(key)
                    if binding is None:
                        correction_prepared = None
                        row.update(status='unavailable', reason='No exact per-image correction input selection is saved.')
                        _record_row_attempt(row, request, state='unavailable', phase='batch_selection')
                        rows.append(row)
                        continue
                    row['correction_binding'] = binding.to_dict()
                    if project_root is None:
                        raise ValueError('Project correction transport requires its workspace root.')
                    paths = {asset.role:correction_assets.resolve_correction_asset_path(project_root, asset.path)
                             for asset in binding.inputs}
                    request = attempts.capture_recipe_attempt_request(recipe, paths)
                    from seedvision.benchmarking.project_correction import run_project_correction
                    row_phase, executor = 'correction_path', attempts.PROJECT_CORRECTION_EXECUTOR
                    result, correction_prepared = run_project_correction(recipe, project_root, binding,
                        project_image_record, image_path, cache=cache, previous=correction_prepared, cancel=cancel, working_workflow=workflow)
                else:
                    # Prepared correction inputs are outside a generic graph's
                    # cache budget. Release them before any other analysis.
                    correction_prepared = None
                    selected_inputs = _selected_inputs(recipe, image_path, options.get('model_asset_paths',{}))
                    request = attempts.capture_recipe_attempt_request(recipe, selected_inputs)
                    row_phase, executor = 'native_path', attempts.NATIVE_PATH_EXECUTOR
                    if workflow is None:
                        result = run_native_recipe_path(recipe, image_path, cache=cache, cancel=cancel, **options)
                    else:
                        from seedvision.pipeline.workflows import run_workflow_path
                        result = run_workflow_path(workflow, image_path, cache=cache, cancel=cancel, **options)
                row_phase = 'result_validation'
                if result.source_sha256 != expected_source_sha256:
                    raise ValueError('Native result source differs from the saved project image.')
                row_phase = 'archive_export'
                report = export_native_recipe_result(staging/f'graph-{index:03d}', recipe, result, cancel=cancel)
                # The actual exporter already verified every stored array and
                # published this sidecar atomically with its archive. Preserve
                # that completed row even if cancellation arrives after return.
                quality = qc.load_method_qc(report.parent/qc.FILENAME)
                row.update(status=result.status, report=report.relative_to(staging).as_posix(),
                    nodes=[{'stage_id': node.stage_id, 'status': node.status, 'reason': node.reason,
                            'reused': node.reused} for node in result.nodes])
                row['method_qc'] = batch_qc.embedded_row_qc(quality, archive_path=row['report'])
                batch_qc.validate_row_qc(row, index, expected_source_sha256=expected_source_sha256)
                if recipe.recipe_id == correction_assets.RECIPE_ID:
                    executed_corrections.append(binding)
            except (NativeRecipeCancelled, NativeCorrectionCancelled) as error:
                row.update(status='cancelled', report=None, nodes=[], reason=f'{type(error).__name__}: {error}')
                _record_row_attempt(row, request, state='cancelled', phase=row_phase, error=error,
                                    executor=executor, result=result)
                rows.append(row)
                raise
            except (ValueError, OSError, RuntimeError) as error:
                correction_prepared = None
                row.update(status='failed', report=None, nodes=[], reason=f'{type(error).__name__}: {error}')
                _record_row_attempt(row, request, state='failed', phase=row_phase, error=error,
                                    executor=executor, result=result)
                failed_directory = staging/f'graph-{index:03d}'
                try:
                    _remove_owned_directory(failed_directory, staging)
                except Exception as cleanup_error:
                    error.add_note(f'Owned failed graph cleanup failed: {cleanup_error}')
                    rows.append(row)
                    raise error
            finally:
                # Only the node cache may retain dense products across graphs.
                # A previous run must not keep evicted products alive during the next.
                result = None
            rows.append(row)
        phase, current_row_index = 'batch_publication', None
        payload = {'format': 'seedfiddle.project-native-graphs', 'version': 3,
            'source_sha256': expected_source_sha256, 'graphs': rows,
            'project_image_id': None if project_image_record is None else project_image_record.identifier,
            'scientifically_validated': False}
        with (staging/'native-graphs.json').open('wb') as stream:
            encoded = _json_bytes(payload)
            if len(encoded) > batch_qc.MAX_LEDGER_BYTES:
                raise ValueError('Native graph ledger exceeds its metadata byte bound.')
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if cancel is not None and cancel():
            raise NativeRecipeCancelled('Project native graph execution cancelled before publication.')
        check_source()
        # Export callbacks may run after computation. Recheck all correction
        # transports without callbacks before the companion transaction commits.
        for binding in executed_corrections:
            correction_assets.verify_correction_bindings(project_root, (binding,), (project_image_record,),
                                                          attachments, native_recipes,
                                                          pipeline_workflow=pipeline_workflow)
        if project_path is not None:
            _retain_rows(rows, project_root, project_path)
        staging.rename(destination)
        return destination/'native-graphs.json'
    except BaseException as error:
        correction_prepared = None
        archives_state = 'not_created' if staging is None else 'unobserved'
        try:
            if staging is not None and _remove_owned_directory(staging, destination.parent):
                archives_state = 'staging_removed'
        except Exception as cleanup_error:
            archives_state = 'cleanup_unknown'
            error.add_note(f'Owned native graph staging cleanup failed: {cleanup_error}')
        if isinstance(error, Exception):
            try:
                quality = batch_qc.recipe_batch_attempt_qc(batch_request, rows, error=error, phase=phase,
                    current_row_index=current_row_index, archives_state=archives_state)
                error.batch_quality = quality  # Compact in-memory evidence survives secondary history failure.
                _retain_rows(rows, destination.parent, destination)
                retain_method_qc(destination.parent, quality, project_path=destination)
                if project_path is not None:
                    _retain_rows(rows, project_root, project_path)
                    retain_method_qc(project_root, quality, project_path=project_path)
            except Exception as recording_error:
                error.add_note(f'Batch graph QC retention failed: {type(recording_error).__name__}: {recording_error}')
        raise
    finally:
        correction_prepared = None
