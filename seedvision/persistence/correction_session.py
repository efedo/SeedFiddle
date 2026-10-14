"""Compact desktop correction context and transactional input capture."""
from dataclasses import dataclass
from pathlib import Path

from seedvision.benchmarking.native_correction_recipe import _validate_recipe
from seedvision.benchmarking.native_recipe import NativeRecipeCancelled
from seedvision.benchmarking.native_correction_inputs import _hash_file
from seedvision.benchmarking.project_correction import _project_source
from seedvision.persistence import correction_inputs as assets
from seedvision.persistence.project_analysis import (
    ProjectFileReference, ProjectImageRecord,
    _image_identifier, _validated_source_locator, _validated_digest, _validated_shape,
    _canonical_relative, PROJECT_RELATIVE, EXTERNAL_ABSOLUTE,
)
from seedvision.persistence.instance_masks import read_source_raster_shape


@dataclass(frozen=True, slots=True)
class CorrectionProjectContext:
    root: Path
    image: ProjectImageRecord
    source_path: Path

    def __post_init__(self):
        if not isinstance(self.image, ProjectImageRecord):
            raise ValueError('Correction project context requires an exact image record.')
        if not isinstance(self.root, Path) or not isinstance(self.source_path, Path):
            raise ValueError('Correction project context requires native absolute paths.')
        if not self.root.is_absolute() or not self.source_path.is_absolute():
            raise ValueError('Correction project context paths must be absolute.')
        location, path = _validated_source_locator(self.image.source.location, self.image.source.path)
        if self.image.identifier != _image_identifier(location, path):
            raise ValueError('Correction project context ID differs from its image locator.')
        _validated_digest(self.image.source.sha256, 'correction project source SHA')
        _validated_shape(self.image.source_shape, 'correction project source frame')


@dataclass(frozen=True, slots=True)
class CorrectionKeepRequest:
    context: CorrectionProjectContext
    recipe: object
    binding: assets.ProjectCorrectionBinding
    generation: int
    pinned: bool = False

    def __post_init__(self):
        if not isinstance(self.context, CorrectionProjectContext):
            raise ValueError('Correction Keep requires an exact project image context.')
        _validate_recipe(self.recipe)
        if type(self.generation) is not int or self.generation < 0 or type(self.pinned) is not bool:
            raise ValueError('Correction Keep requires a current generation and explicit input origin.')
        assets.validate_correction_bindings((self.binding,), (self.context.image,), native_recipes=(self.recipe,))


def capture_correction_context(root, source_path, *, original_records=()):
    """Capture one explicitly selected image; never identify an image by its SHA."""
    root, source_path = Path(root).resolve(), Path(source_path).resolve()
    before = _hash_file(source_path)
    shape = read_source_raster_shape(source_path)
    if before != _hash_file(source_path):
        raise ValueError('Correction project source changed during metadata capture.')
    if source_path.is_relative_to(root):
        location, path = PROJECT_RELATIVE, _canonical_relative(source_path.relative_to(root), 'source image path')
    else:
        location, path = EXTERNAL_ABSOLUTE, source_path.as_posix()
    record = ProjectImageRecord(_image_identifier(location, path), ProjectFileReference(location, path, before), shape)
    original = next((item for item in original_records if item.identifier == record.identifier), None)
    if original is not None:
        if original.source != record.source or original.source_shape != record.source_shape:
            raise ValueError('The selected project image changed since its saved source record.')
        record = original
    return CorrectionProjectContext(root, record, source_path)


def capture_correction_keep(request, *, cancel=None):
    """Publish verified immutable input copies; return compact binding metadata."""
    if not isinstance(request, CorrectionKeepRequest):
        raise ValueError('Correction Keep requires its immutable metadata request.')
    def poll():
        if cancel is not None and cancel():
            raise NativeRecipeCancelled('Keeping correction inputs cancelled.')
        return False
    poll()
    context = request.context
    source = next(asset for asset in request.binding.inputs if asset.role == 'source_rgb')
    path = _project_source(context.root, context.image, context.source_path)
    assets.copy_verified_input(path, source, cancel=poll)
    captured = assets.capture_correction_bindings(context.root, (request.binding,), (context.image,),
        native_recipes=(request.recipe,), existing=None if request.pinned else (), cancel=poll)[0]
    poll()
    # No caller callbacks after the final bounded source/transport verification.
    _project_source(context.root, context.image, context.source_path)
    assets.copy_verified_input(path, source)
    assets.verify_correction_bindings(context.root, (captured,), (context.image,), native_recipes=(request.recipe,))
    return validate_correction_keep_result(request, captured)


def validate_correction_keep_result(request, captured):
    """Accept only this request's selection with canonical project transport."""
    from dataclasses import replace
    if not isinstance(request, CorrectionKeepRequest):
        raise ValueError('Correction Keep requires its immutable metadata request.')
    assets.validate_correction_bindings((captured,), (request.context.image,), native_recipes=(request.recipe,))
    expected = replace(request.binding, inputs=tuple(replace(item,
        path='native-correction/'+item.source_sha256+'.image') for item in request.binding.inputs))
    if captured != expected:
        raise ValueError('Kept correction inputs differ from the frozen selection.')
    return captured
