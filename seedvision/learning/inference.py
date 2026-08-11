"""Memory-bounded, overlap-blended CUDA inference for either model family."""

from __future__ import annotations

from contextlib import nullcontext


def _tile_starts(length: int, tile: int, stride: int) -> tuple[int, ...]:
    if length <= tile:
        return (0,)
    starts = list(range(0, max(1, length - tile + 1), stride))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def tiled_predict(
    model,
    features,
    *,
    tile_size: int = 512,
    overlap: int = 96,
    use_mixed_precision: bool = True,
):
    """Blend dense tile predictions while retaining outputs on the model device."""

    import torch
    import torch.nn.functional as functional

    if features.ndim != 4 or features.shape[0] != 1:
        raise ValueError("Tiled inference expects one NCHW feature tensor.")
    tile_size = int(tile_size)
    overlap = int(overlap)
    if tile_size < 64 or overlap < 0 or overlap * 2 >= tile_size:
        raise ValueError("Require tile_size >= 64 and 0 <= 2*overlap < tile_size.")
    original_height, original_width = features.shape[-2:]
    pad_height = max(0, tile_size - original_height)
    pad_width = max(0, tile_size - original_width)
    if pad_height or pad_width:
        mode = "reflect" if min(original_height, original_width) > 1 else "replicate"
        features = functional.pad(features, (0, pad_width, 0, pad_height), mode=mode)
    height, width = features.shape[-2:]
    stride = tile_size - 2 * overlap
    y_starts = _tile_starts(height, tile_size, stride)
    x_starts = _tile_starts(width, tile_size, stride)
    axis = torch.hann_window(tile_size + 2, periodic=False, device=features.device)[1:-1]
    weight = (axis[:, None] * axis[None, :]).clamp_min(1e-3)[None, None]
    accumulated: dict[str, object] = {}
    denominators: dict[str, object] = {}
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_mixed_precision and features.device.type == "cuda"
        else nullcontext()
    )
    model.eval()
    with torch.inference_mode(), autocast:
        for y0 in y_starts:
            for x0 in x_starts:
                tile = features[..., y0 : y0 + tile_size, x0 : x0 + tile_size]
                outputs = model(tile)
                for name, output in outputs.items():
                    output = output.float()
                    if name not in accumulated:
                        accumulated[name] = torch.zeros(
                            (1, output.shape[1], height, width),
                            device=output.device,
                            dtype=torch.float32,
                        )
                        denominators[name] = torch.zeros(
                            (1, 1, height, width),
                            device=output.device,
                            dtype=torch.float32,
                        )
                    accumulated[name][..., y0 : y0 + tile_size, x0 : x0 + tile_size] += output * weight
                    denominators[name][..., y0 : y0 + tile_size, x0 : x0 + tile_size] += weight
    return {
        name: values[..., :original_height, :original_width]
        / denominators[name][..., :original_height, :original_width].clamp_min(1e-6)
        for name, values in accumulated.items()
    }
