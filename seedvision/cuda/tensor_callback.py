"""Callback-scoped guards for live CUDA buffers without retaining tile tensors."""
import torch


class TensorCallbackMutation(RuntimeError):
    """A caller callback changed a live calculation tensor."""


class TensorCallbackReturnError(TypeError):
    """A cancellation callback returned something other than a bool."""


def _signature(tensor):
    storage=tensor.untyped_storage()
    return (str(tensor.device),str(tensor.dtype),tuple(tensor.shape),tuple(tensor.stride()),
            tensor.storage_offset(),tensor.data_ptr(),storage.data_ptr(),storage.nbytes(),tensor._version)


def checked_cancel(cancel, tensors=(), *, identity=None):
    """Check live tensor storage before/after a strict caller callback.

    Snapshots are local to this call, including its exception path. Intentional
    kernel writes between callbacks are allowed; no tile is retained afterward.
    Invalid raster types/layouts remain the primitive validator's responsibility.
    """
    if cancel is None:return False
    watched=tuple((value,_signature(value)) for value in tensors
                  if torch.is_tensor(value) and value.layout==torch.strided)
    metadata=identity() if identity is not None else None
    try:
        stopped=cancel()
        if type(stopped) is not bool:raise TensorCallbackReturnError('Cancellation callback must return bool.')
        return stopped
    finally:
        if (any(_signature(value)!=signature for value,signature in watched)
                or (identity is not None and identity()!=metadata)):
            raise TensorCallbackMutation('A callback changed live calculation tensor storage or contents.')
