"""Run an exact per-image correction selection from pinned project inputs."""
from dataclasses import replace
from pathlib import Path
import sys
from types import MappingProxyType

from seedvision.benchmarking.native_recipe import NativeRecipeCancelled, _digest, _module_record, _execution_signature
from seedvision.benchmarking.native_correction_inputs import (
    run_native_correction_files, correction_file_implementation_sha256, _signatures,
)
from seedvision.benchmarking.native_correction_recipe import (
    NativeCorrectionCancelled, RECIPE_ID, STAGE_ID, _correction_output_signatures,
)
from seedvision.benchmarking.recipes import RecipeProvenanceMode, RecipeSpec
from seedvision.persistence import correction_inputs as _bindings
from seedvision.persistence import project_analysis as _project_metadata
from seedvision.persistence.project_analysis import (
    ProjectImageRecord, ProjectFileReference,
    _image_identifier, _validated_source_locator, _validated_shape, _validated_digest,
    _resolve_project_relative, _external_path,
)
from seedvision.segmentation.baseline import PipelineAnalysisCache


_CACHE_KEY = f'reference.recipe:{RECIPE_ID}:{STAGE_ID}'


def _cancel(cancel):
    if cancel is not None and cancel():
        raise NativeRecipeCancelled('Project correction cancelled before a complete result.')


def _implementation():
    constants = (_project_metadata.PROJECT_RELATIVE, _project_metadata.EXTERNAL_ABSOLUTE,
                 _project_metadata._SHA256_LENGTH)
    if constants != ('project_relative', 'external_absolute', 64):
        raise ValueError('Project correction locator constants differ from their supported contract.')
    # Verify only the locator/shape/digest dependency closure. Unrelated project
    # UI dataclasses have default factories outside this execution footprint.
    metadata = _module_record(_project_metadata, only=(
        '_image_identifier', '_validated_source_locator', '_validated_relative',
        '_validated_shape', '_exact_int', '_validated_digest',
        '_resolve_project_relative', '_external_path'))
    metadata['locator_constants'] = constants
    return {'correction_project_adapter': _digest(_module_record(sys.modules[__name__])),
            'correction_binding_validation': _digest(_module_record(_bindings)),
            'correction_project_metadata': _digest(metadata)}


def _project_source(root, image_record, image_path):
    if (not isinstance(image_record, ProjectImageRecord)
            or not isinstance(image_record.source, ProjectFileReference)):
        raise ValueError('Project correction requires the exact saved project image record.')
    location, path = _validated_source_locator(image_record.source.location, image_record.source.path)
    _validated_digest(image_record.source.sha256, 'correction project image SHA')
    _validated_shape(image_record.source_shape, 'correction project image shape')
    if image_record.identifier != _image_identifier(location, path):
        raise ValueError('Correction project image ID differs from its saved locator.')
    expected = (_resolve_project_relative(Path(root).resolve(), path, 'source image path')
                if location == 'project_relative' else _external_path(path))
    actual = Path(image_path).resolve(strict=True)
    if not expected.is_absolute() or expected.resolve(strict=True) != actual:
        raise ValueError('Correction project source path differs from its exact saved image locator.')
    return actual


def run_project_correction(recipe, root, binding, image_record, image_path, *,
                           cache=None, previous=None, cancel=None, working_workflow=None,
                           reference_runner=None):
    """Return correction outputs and one reusable prepared CUDA input bundle.

    A caller may retain the prepared bundle across adjacent correction runs for
    this image. Release it before other analyses and before the next image.
    Original files are archival metadata and are never used as fallback.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be callable or None.')
    if cache is None:
        cache = PipelineAnalysisCache()
    if not isinstance(cache, PipelineAnalysisCache):
        raise TypeError('cache must be a PipelineAnalysisCache or None.')
    try:
        _cancel(cancel)
        if not isinstance(recipe, RecipeSpec) or recipe.provenance.mode is not RecipeProvenanceMode.NATIVE_COMPATIBLE:
            raise ValueError('Project correction requires its explicit native-compatible recipe.')
        if not isinstance(binding, _bindings.ProjectCorrectionBinding):
            raise ValueError('Project correction requires an exact immutable input binding.')
        # Snapshot immutable metadata before any subsequent caller callbacks.
        binding = _bindings.ProjectCorrectionBinding.from_dict(binding.to_dict())
        _bindings.validate_correction_bindings((binding,), (image_record,), native_recipes=(recipe,))
        source_path = _project_source(root, image_record, image_path)
        origins = _bindings.CorrectionRunOrigins(binding, image_record.source.location, image_record.source.path)
        before = _implementation()
        source_asset = next(asset for asset in binding.inputs if asset.role == 'source_rgb')
        paths = {asset.role: _bindings.resolve_correction_asset_path(root, asset.path) for asset in binding.inputs}
        def poll():
            _cancel(cancel)
            return False
        asset_cancel = poll if cancel is not None else None
        _bindings.copy_verified_input(source_path, source_asset, cancel=asset_cancel)
        _bindings.verify_correction_bindings(root, (binding,), (image_record,),
                                             native_recipes=(recipe,), cancel=asset_cancel)
        result, prepared = run_native_correction_files(recipe, paths, cache=cache,
            previous=previous, cancel=asset_cancel, working_workflow=working_workflow,
            reference_runner=reference_runner)
        signatures = (_correction_output_signatures(result.node(STAGE_ID).outputs),
            _execution_signature(result.node(STAGE_ID)))
        _bindings.validate_correction_run_origins(recipe, origins, result.input_sources)
        if (result.source_sha256 != image_record.source.sha256
                or result.coordinate_frame.shape != tuple(image_record.source_shape)):
            raise ValueError('Project correction output differs from its exact project source frame.')
        # No caller callback may run after this final poll and bounded audit.
        _cancel(cancel)
        if before != _implementation():
            raise ValueError('Project correction adapter or binding validation changed during execution.')
        _bindings.validate_correction_bindings((binding,), (image_record,), native_recipes=(recipe,))
        if _project_source(root, image_record, image_path) != source_path:
            raise ValueError('Project correction image locator changed during execution.')
        _bindings.copy_verified_input(source_path, source_asset)
        _bindings.verify_correction_bindings(root, (binding,), (image_record,), native_recipes=(recipe,))
        if _signatures(prepared.tensors) != dict(prepared.tensor_signatures):
            raise ValueError('A prepared project correction input changed during execution.')
        if (_correction_output_signatures(result.node(STAGE_ID).outputs),
                _execution_signature(result.node(STAGE_ID))) != signatures:
            raise ValueError('A project correction output changed before publication.')
        runtime = dict(result.runtime) | before
        return replace(result, input_origins=origins, runtime=MappingProxyType(runtime),
                       implementation_sha256=correction_file_implementation_sha256(runtime)), prepared
    except NativeCorrectionCancelled as error:
        cache.values.pop(_CACHE_KEY, None)
        raise NativeRecipeCancelled(str(error)) from error
    except Exception:
        cache.values.pop(_CACHE_KEY, None)
        raise
