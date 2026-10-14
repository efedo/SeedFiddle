"""Separate a saved model selection's origin from its verified local transport."""
from pathlib import Path, PurePosixPath, PureWindowsPath


def validate_model_origin(path):
    """Validate either host's canonical absolute text without probing that host."""
    if type(path) is not str or not path or path != path.strip() or len(path) > 4096 or '\0' in path:
        raise ValueError('Selected model origin must be bounded canonical absolute text.')
    flavor = PureWindowsPath(path) if PureWindowsPath(path).is_absolute() else PurePosixPath(path)
    if not flavor.is_absolute() or '..' in flavor.parts or str(flavor) != path:
        raise ValueError('Selected model origin must be canonical and absolute.')
    return path


def resolve_model_path(origin, resolved_path=None):
    validate_model_origin(origin)
    if resolved_path is None and not Path(origin).is_absolute():
        raise ValueError('Selected model origin is unavailable on this host; bind a verified project copy.')
    candidate = origin if resolved_path is None else str(resolved_path)
    try:
        canonical = str(Path(candidate).resolve(strict=True))
    except RuntimeError as error:  # pathlib on supported Python versions: symlink loop
        raise ValueError('The resolved model path cannot be followed.') from error
    if canonical != candidate:
        raise ValueError('The resolved model path must be canonical and absolute.')
    return canonical
