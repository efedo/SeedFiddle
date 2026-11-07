"""Batch reports for a project's independent default-graph node instances."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

from seedvision.persistence.reference_regions import file_sha256
from seedvision.segmentation.baseline import PipelineAnalysisCache


def _remove_owned_directory(directory, parent):
    if not directory.exists():
        return False
    info = directory.lstat()
    if (directory.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400
            or directory.resolve().parent != parent.resolve()):
        raise ValueError('Node instance cleanup escaped its owned staging directory.')
    shutil.rmtree(directory)
    return True


def export_project_node_instances(instances, image_path, directory, *, expected_source_sha256,
                                  default_context=None, cancel=None):
    """Publish each independent calculation with its own cache and scope.

    Repeated instances each keep a row. A failed calculation gets an explicit
    row, never a passing report.
    """
    from seedvision.export.results import result_report, export_report
    from seedvision.persistence.analysis_settings import analysis_settings_profile_to_payload
    from seedvision.pipeline.catalogue_nodes import (default_graph, execute_instance, instance_nodes,
                                                     validate_node_instances)
    from seedvision.timing import AnalysisCancelled
    instances = validate_node_instances(instances)
    if not instances:
        return None
    destination = Path(directory).absolute()
    if destination.exists():
        raise FileExistsError(f'Node instance destination already exists: {destination}')
    if file_sha256(image_path) != expected_source_sha256:
        raise ValueError('Source changed before independent node instance execution.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.node-instances-', suffix='.incomplete', dir=destination.parent))
    rows = []
    try:
        for index, instance in enumerate(instances):
            if cancel is not None and cancel():
                raise AnalysisCancelled('Node instance batch cancelled.')
            if file_sha256(image_path) != expected_source_sha256:
                raise ValueError('Source changed during independent node instance execution.')
            row = {'instance_id': instance.instance_id, 'catalogue_key': instance.catalogue_key,
                   'target_node': instance.target_node, 'report': None, 'status': 'not_run', 'reason': ''}
            cache = PipelineAnalysisCache()
            result = None
            try:
                result = execute_instance(instance, image_path, cache=cache, cancel=cancel,
                                          default_context=default_context)
                graph = default_graph(instance)
                report_data = result_report(result, enabled={identifier for identifier in instance_nodes(instance)
                                                            if graph.node(identifier).enabled},
                    settings=analysis_settings_profile_to_payload(instance.default_settings),
                    execution_provenance={'route': 'catalogue_node_batch', 'instance_id': instance.instance_id,
                                          'target_node': instance.target_node})
                folder = export_report(staging / f'instance-{index:03d}', result, report_data)
                row['report'] = (folder / 'results.json').relative_to(staging).as_posix()
                row['status'] = report_data['state']
            except AnalysisCancelled:
                raise
            except Exception as error:
                row.update(status='failed', report=None, reason=f'{type(error).__name__}: {error}'[:4096])
            finally:
                result = None
                cache.values.clear()
            rows.append(row)
        if cancel is not None and cancel():
            raise AnalysisCancelled('Node instance batch cancelled before publication.')
        if file_sha256(image_path) != expected_source_sha256:
            raise ValueError('Source changed before independent node instance publication.')
        payload = {'format': 'seedfiddle.catalogue-node-instances', 'version': 1,
                   'source_sha256': expected_source_sha256, 'instances': rows, 'scientifically_validated': False}
        (staging / 'node-instances.json').write_bytes(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8'))
        staging.rename(destination)
        return destination / 'node-instances.json'
    except BaseException:
        _remove_owned_directory(staging, destination.parent)
        raise
