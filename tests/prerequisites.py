"""Read-only test prerequisite checks."""


def check_prerequisites(mode):
    """Full and CUDA runs require a CUDA device; core, CPU/Qt and focused runs do not."""
    errors = []
    if mode in {'full', 'cuda'}:
        import torch
        if not torch.cuda.is_available():
            errors.append('CUDA is required for ' + mode + '; use --tier cpu-qt for the explicit CPU subset.')
    return {'passed': not errors, 'errors': errors}
