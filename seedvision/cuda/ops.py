"""CUDA-first image primitives implemented with PyTorch tensors.

Images use NCHW tensors. Colour inputs retain OpenCV's BGR channel ordering so
the public application data structures do not need to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from threading import RLock

import numpy as np


def apply_contrastive_negative_evidence(
    positive_probability,
    negative_probability,
    *,
    strength: float = 0.95,
):
    """Suppress a class only where negative evidence is more specific.

    Positive and negative colour models can both match glass, pale seeds, or
    neutral backgrounds. A direct multiplicative negative mask destroys valid
    class evidence in those ambiguous regions. This contrastive form leaves an
    equally good positive match intact and scales suppression with the negative
    model's relative advantage. Painted coordinates receive no special case.
    """

    import torch

    positive = positive_probability.clamp(0.0, 1.0)
    negative = negative_probability.clamp(0.0, 1.0)
    contradiction = torch.relu(negative - positive) / (
        negative + positive + 1e-6
    )
    return positive * (
        1.0 - max(0.0, min(1.0, float(strength))) * contradiction
    )


class GpuRaster:
    """GPU-resident raster downloaded only by an explicit CPU consumer."""

    __array_priority__ = 1000

    def __init__(self, tensor, *, numpy_dtype=np.uint8, name: str = "") -> None:
        import torch

        if not torch.is_tensor(tensor):
            raise TypeError("GpuRaster requires a PyTorch tensor.")
        self._tensor = tensor.detach()
        self._numpy_dtype = np.dtype(numpy_dtype)
        self.name = str(name)
        self._host_cache: np.ndarray | None = None
        self._lock = RLock()
        self.download_count = 0

    @property
    def device(self):
        return self._tensor.device

    @property
    def gpu_resident(self) -> bool:
        return self._tensor.device.type == "cuda"

    @property
    def is_materialized(self) -> bool:
        return self._host_cache is not None

    @property
    def host_cache(self) -> np.ndarray | None:
        """Return the existing CPU mirror without causing a download."""

        return self._host_cache

    @property
    def dtype(self) -> np.dtype:
        return self._numpy_dtype

    @property
    def shape(self) -> tuple[int, ...]:
        shape = tuple(int(value) for value in self._tensor.shape)
        if len(shape) == 4:
            shape = shape[1:]
        if len(shape) == 3 and shape[0] == 1:
            return shape[1:]
        if len(shape) == 3:
            return (shape[1], shape[2], shape[0])
        return shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def gpu_tensor(self, *, device=None, dtype=None):
        tensor = self._tensor
        if device is not None and tensor.device != device:
            tensor = tensor.to(device=device)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def numpy(self, *, copy: bool = False) -> np.ndarray:
        with self._lock:
            if self._host_cache is None:
                self._host_cache = tensor_to_image(
                    self._tensor, dtype=self._numpy_dtype
                )
                self.download_count += 1
            return self._host_cache.copy() if copy else self._host_cache

    def release_host_cache(self) -> None:
        with self._lock:
            self._host_cache = None

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        values = self.numpy(copy=bool(copy))
        return values if dtype is None else values.astype(dtype, copy=False)

    def __getitem__(self, key):
        return self.numpy()[key]

    def __len__(self) -> int:
        return self.shape[0]

    def __eq__(self, other):
        return self.numpy() == other

    def __ne__(self, other):
        return self.numpy() != other

    def __gt__(self, other):
        return self.numpy() > other

    def __ge__(self, other):
        return self.numpy() >= other

    def __lt__(self, other):
        return self.numpy() < other

    def __le__(self, other):
        return self.numpy() <= other

    def astype(self, dtype, copy=True):
        return self.numpy().astype(dtype, copy=copy)

    def max(self, *args, **kwargs):
        if not args and not kwargs:
            return self._tensor.max().item()
        return self.numpy().max(*args, **kwargs)

    def min(self, *args, **kwargs):
        if not args and not kwargs:
            return self._tensor.min().item()
        return self.numpy().min(*args, **kwargs)

    def sum(self, *args, **kwargs):
        if not args and not kwargs:
            return self._tensor.sum().item()
        return self.numpy().sum(*args, **kwargs)

    def any(self, *args, **kwargs):
        if not args and not kwargs:
            return bool(self._tensor.any().item())
        return self.numpy().any(*args, **kwargs)

    def count_above(self, threshold: float) -> int:
        return int((self._tensor >= float(threshold)).sum().item())

    def __repr__(self) -> str:
        return (
            f"GpuRaster(name={self.name!r}, shape={self.shape}, "
            f"device={self.device}, materialized={self.is_materialized})"
        )


@dataclass(frozen=True, slots=True)
class CudaContext:
    """Resolved tensor device used by the complete numerical image path."""

    device: object
    name: str
    cuda: bool
    fallback_reason: str = ""

    @classmethod
    def resolve(
        cls,
        *,
        requested: str = "cuda",
        allow_cpu_fallback: bool = True,
    ) -> "CudaContext":
        import torch

        if requested not in {"cuda", "auto", "cpu"}:
            raise ValueError("requested device must be cuda, auto, or cpu")
        if requested == "cpu":
            return cls(torch.device("cpu"), "CPU", False, "CPU explicitly selected")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            return cls(device, torch.cuda.get_device_name(device), True)
        if requested == "cuda" and not allow_cpu_fallback:
            raise RuntimeError(
                "CUDA image analysis was requested but PyTorch cannot access "
                "an NVIDIA CUDA device."
            )
        return cls(torch.device("cpu"), "CPU", False, "CUDA unavailable")


def image_to_tensor(image: np.ndarray | GpuRaster, context: CudaContext | None = None):
    """Upload a BGR/gray NumPy image as a float32 NCHW tensor."""

    import torch

    context = context or CudaContext.resolve()
    if isinstance(image, GpuRaster):
        tensor = image.gpu_tensor(device=context.device, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:
            tensor = tensor[None]
        return tensor
    values = np.asarray(image)
    if values.ndim == 2:
        values = values[:, :, None]
    tensor = torch.from_numpy(np.ascontiguousarray(values)).to(
        device=context.device, dtype=torch.float32
    )
    return tensor.permute(2, 0, 1).unsqueeze(0)


def tensor_to_image(tensor, *, dtype=np.uint8) -> np.ndarray:
    """Download one NCHW/CHW tensor to an HWC/gray NumPy array."""

    values = tensor.detach()
    if values.ndim == 4:
        values = values[0]
    if values.ndim == 3:
        values = values.permute(1, 2, 0)
    result = values.contiguous().cpu().numpy()
    if result.ndim == 3 and result.shape[2] == 1:
        result = result[:, :, 0]
    if np.issubdtype(np.dtype(dtype), np.integer):
        result = np.rint(result)
        limits = np.iinfo(dtype)
        result = np.clip(result, limits.min, limits.max)
    return result.astype(dtype, copy=False)


def resize(tensor, size: tuple[int, int], *, mode: str = "bilinear"):
    import torch.nn.functional as functional

    kwargs = {} if mode in {"nearest", "area"} else {"align_corners": False}
    return functional.interpolate(tensor, size, mode=mode, **kwargs)


def bgr_to_gray(bgr):
    return (
        bgr[:, 0:1] * 0.114
        + bgr[:, 1:2] * 0.587
        + bgr[:, 2:3] * 0.299
    )


def bgr_to_lab(bgr):
    """Convert 0..255 BGR to OpenCV-compatible 0..255 CIE Lab."""

    import torch

    rgb = bgr[:, (2, 1, 0)] / 255.0
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        torch.pow((rgb + 0.055) / 1.055, 2.4),
    )
    red, green, blue = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    x = (0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047
    y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    z = (0.0193339 * red + 0.1191920 * green + 0.9503041 * blue) / 1.08883
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def transform(value):
        return torch.where(
            value > epsilon,
            torch.pow(value.clamp_min(0.0), 1.0 / 3.0),
            (kappa * value + 16.0) / 116.0,
        )

    fx, fy, fz = transform(x), transform(y), transform(z)
    lightness = (116.0 * fy - 16.0) * 2.55
    a = 500.0 * (fx - fy) + 128.0
    b = 200.0 * (fy - fz) + 128.0
    return torch.cat((lightness, a, b), dim=1).clamp(0.0, 255.0)


def gaussian_blur(tensor, sigma: float, *, radius_limit: int = 96):
    import torch
    import torch.nn.functional as functional

    sigma = float(max(0.01, sigma))
    radius = min(radius_limit, max(1, round(sigma * 3.0)))
    positions = torch.arange(
        -radius, radius + 1, device=tensor.device, dtype=tensor.dtype
    )
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel /= kernel.sum()
    channels = tensor.shape[1]
    horizontal = kernel.reshape(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.reshape(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    result = functional.pad(tensor, (radius, radius, 0, 0), mode="replicate")
    result = functional.conv2d(result, horizontal, groups=channels)
    result = functional.pad(result, (0, 0, radius, radius), mode="replicate")
    return functional.conv2d(result, vertical, groups=channels)


@lru_cache(maxsize=64)
def _ellipse_kernel_cpu(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    radius = size // 2
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    denominator = max(radius + 0.25, 0.25)
    return np.uint8((xx / denominator) ** 2 + (yy / denominator) ** 2 <= 1.0)


def _binary_convolution(mask, size: int):
    import torch
    import torch.nn.functional as functional

    kernel_np = _ellipse_kernel_cpu(size)
    kernel = torch.from_numpy(kernel_np).to(
        device=mask.device, dtype=torch.float32
    )[None, None]
    radius = kernel.shape[-1] // 2
    values = functional.conv2d(mask.float(), kernel, padding=radius)
    return values, float(kernel_np.sum())


def binary_dilate(mask, size: int):
    values, _ = _binary_convolution(mask, size)
    return values > 0.0


def binary_erode(mask, size: int):
    values, total = _binary_convolution(mask, size)
    return values >= total - 0.5


def binary_open(mask, size: int):
    return binary_dilate(binary_erode(mask, size), size)


def binary_close(mask, size: int):
    return binary_erode(binary_dilate(mask, size), size)


def otsu_threshold(values) -> float:
    """Calculate an 8-bit Otsu threshold entirely on the tensor device."""

    import torch

    flat = values.reshape(-1).clamp(0.0, 255.0)
    histogram = torch.histc(flat, bins=256, min=0.0, max=255.0)
    probability = histogram / histogram.sum().clamp_min(1.0)
    indices = torch.arange(256, device=flat.device, dtype=torch.float32)
    omega = torch.cumsum(probability, dim=0)
    mean = torch.cumsum(probability * indices, dim=0)
    total_mean = mean[-1]
    between = (total_mean * omega - mean).square() / (
        omega * (1.0 - omega)
    ).clamp_min(1e-12)
    between[(omega <= 0.0) | (omega >= 1.0)] = -1.0
    return float(torch.argmax(between).item())


def lab_colour_distribution(
    lab,
    initial_samples,
    eligible_mask,
    *,
    maximum_components: int = 4,
    fit_iterations: int = 6,
    refinement_iterations: int = 2,
    refinement_min_probability: float = 0.82,
    frequency_weight_power: float = 0.35,
    scale_multiplier: float = 1.0,
    combine_modes: str = "sum",
    scale_floors=(8.0, 3.0, 3.0),
    distance_weights=(1.0, 1.25, 1.25),
    maximum_fit_samples: int = 32768,
    output_mask=None,
):
    """Fit a robust multimodal Lab distribution and return class membership.

    The compact mixture is initialized deterministically from the painted class
    samples. Refinement admits only high-probability, eligible pixels and keeps
    the original painted samples in every round, preventing unlabeled image
    regions from overwhelming the user's constraints.
    """

    import torch

    values = initial_samples.reshape(-1, 3).to(device=lab.device, dtype=lab.dtype)
    if not int(values.shape[0]):
        raise ValueError("At least one Lab colour sample is required.")
    eligible = eligible_mask.bool()
    floors = torch.as_tensor(scale_floors, device=lab.device, dtype=lab.dtype)
    weights = torch.as_tensor(distance_weights, device=lab.device, dtype=lab.dtype)
    if combine_modes not in {"sum", "maximum"}:
        raise ValueError("combine_modes must be 'sum' or 'maximum'.")

    def limited(samples, limit: int = maximum_fit_samples):
        count = int(samples.shape[0])
        if count <= limit:
            return samples
        indices = torch.linspace(
            0,
            count - 1,
            limit,
            device=samples.device,
        ).round().long()
        return samples[indices]

    anchor_samples = limited(values)

    def fit(samples):
        samples = limited(samples)
        sample_count = int(samples.shape[0])
        component_count = min(
            max(1, int(maximum_components)),
            max(1, (sample_count + 63) // 64),
        )
        global_centre = torch.median(samples, dim=0).values
        global_scale = torch.maximum(
            torch.median(torch.abs(samples - global_centre), dim=0).values
            * 1.4826,
            floors,
        )
        centres = [global_centre]
        minimum_distance = torch.sum(
            ((samples - global_centre) / global_scale).square() * weights,
            dim=1,
        )
        for _ in range(1, component_count):
            next_centre = samples[torch.argmax(minimum_distance)]
            centres.append(next_centre)
            next_distance = torch.sum(
                ((samples - next_centre) / global_scale).square() * weights,
                dim=1,
            )
            minimum_distance = torch.minimum(
                minimum_distance, next_distance
            )
        centre_tensor = torch.stack(centres)
        assignments = torch.zeros(sample_count, device=lab.device, dtype=torch.long)
        for _ in range(max(1, int(fit_iterations))):
            delta = (samples[:, None, :] - centre_tensor[None, :, :]) / global_scale
            distance = torch.sum(delta.square() * weights, dim=2)
            assignments = torch.argmin(distance, dim=1)
            updated = []
            for component in range(component_count):
                members = samples[assignments == component]
                updated.append(
                    torch.median(members, dim=0).values
                    if int(members.shape[0])
                    else centre_tensor[component]
                )
            next_centres = torch.stack(updated)
            if torch.max(torch.abs(next_centres - centre_tensor)) < 0.05:
                centre_tensor = next_centres
                break
            centre_tensor = next_centres
        component_scales = []
        component_weights = []
        for component in range(component_count):
            members = samples[assignments == component]
            if not int(members.shape[0]):
                members = samples
            deviation = torch.median(
                torch.abs(members - centre_tensor[component]), dim=0
            ).values * 1.4826
            component_scales.append(torch.maximum(deviation, floors))
            component_weights.append(float(members.shape[0]) / sample_count)
        return (
            centre_tensor,
            torch.stack(component_scales),
            torch.as_tensor(component_weights, device=lab.device, dtype=lab.dtype),
            sample_count,
        )

    def membership(centres, scales, component_weights):
        # Frequency zero makes every represented colour mode equally strong;
        # frequency one applies the measured painted-area occurrence directly.
        # Normalizing by the largest weight keeps the dominant mode's centre at
        # probability one while still retaining less common painted colours.
        adjusted_weights = component_weights.clamp_min(1e-6).pow(
            max(0.0, float(frequency_weight_power))
        )
        adjusted_weights /= adjusted_weights.max().clamp_min(1e-6)
        weighted_probability = torch.zeros(
            lab.shape[:2], device=lab.device, dtype=lab.dtype
        )
        maximum_membership = torch.zeros_like(weighted_probability)
        # A full H×W×mode tensor becomes prohibitive once a reference is allowed
        # to retain dozens of colours. Evaluate a few modes at a time so the
        # control scales computation rather than peak raster memory. A fuzzy
        # union (the strongest weighted mode) also makes capacity independent:
        # splitting one colour cloud into more overlapping fitted modes must not
        # mechanically increase its probability.
        for start in range(0, int(centres.shape[0]), 4):
            stop = min(start + 4, int(centres.shape[0]))
            effective_scales = scales[start:stop] * max(
                0.05, float(scale_multiplier)
            )
            delta = (
                lab[:, :, None, :] - centres[None, None, start:stop, :]
            ) / effective_scales[None, None, :, :]
            distance = torch.sum(
                delta.square() * weights[None, None, None, :], dim=3
            )
            component_membership = torch.exp(-0.5 * distance)
            weighted = (
                component_membership
                * adjusted_weights[None, None, start:stop]
            )
            if combine_modes == "maximum":
                weighted_probability = torch.maximum(
                    weighted_probability,
                    torch.max(weighted, dim=2).values,
                )
            else:
                weighted_probability += torch.sum(weighted, dim=2)
            maximum_membership = torch.maximum(
                maximum_membership,
                torch.max(component_membership, dim=2).values,
            )
        return weighted_probability.clamp(0.0, 1.0), maximum_membership

    def painted_component_weights(centres, scales):
        """Measure mode occurrence from the user's immutable anchor pixels."""

        delta = (anchor_samples[:, None, :] - centres[None, :, :]) / scales[
            None, :, :
        ]
        distance = torch.sum(delta.square() * weights, dim=2)
        assignment = torch.argmin(distance, dim=1)
        counts = torch.bincount(assignment, minlength=int(centres.shape[0])).to(
            dtype=lab.dtype
        )
        # A fitted mode with no surviving anchor assignment should not acquire
        # authority merely because self-refinement found many image pixels.
        return counts / counts.sum().clamp_min(1.0)

    fit_samples = anchor_samples
    centres, scales, component_weights, sample_count = fit(fit_samples)
    component_weights = painted_component_weights(centres, scales)
    probability, refinement_probability = membership(
        centres, scales, component_weights
    )
    rounds_completed = 0
    for _ in range(max(0, int(refinement_iterations))):
        accepted = lab[
            eligible
            & (refinement_probability >= float(refinement_min_probability))
        ]
        if not int(accepted.shape[0]):
            break
        growth_limit = min(
            maximum_fit_samples,
            max(int(anchor_samples.shape[0]), 64) * 4,
        )
        accepted = limited(accepted, growth_limit)
        fit_samples = torch.cat((anchor_samples, accepted), dim=0)
        centres, scales, component_weights, sample_count = fit(fit_samples)
        component_weights = painted_component_weights(centres, scales)
        probability, refinement_probability = membership(
            centres, scales, component_weights
        )
        rounds_completed += 1
    probability = probability * (
        eligible if output_mask is None else output_mask.bool()
    )
    return (
        probability.clamp(0.0, 1.0),
        centres,
        scales,
        component_weights,
        sample_count,
        rounds_completed,
    )


def lab_colour_frequency_distribution(
    lab,
    reference_samples,
    eligible_mask,
    *,
    maximum_bins: int = 64,
    refinement_iterations: int = 2,
    refinement_min_probability: float = 0.82,
    frequency_weight_power: float = 0.0,
    scale_multiplier: float = 1.0,
    scale_floors=(8.0, 4.0, 4.0),
    distance_weights=(1.0, 1.25, 1.25),
    quantization_steps=(4.0, 3.0, 3.0),
    maximum_fit_samples: int = 32768,
):
    """Estimate colour membership from a reference-pixel frequency table.

    Unlike the compact regional mixture used for automatic estimates, this
    model never averages light and dark painted patterns into one prototype.
    Reference pixels are quantized into small Lab cells. If a reference spans
    more cells than the configured budget, representative observed cells are
    selected to preserve Lab-space coverage rather than frequency alone, and
    all observed occurrences are reassigned to those representatives. Cautious
    refinement may widen only anchored cells; it cannot create an unsupported
    colour mode.
    """

    import torch

    samples = reference_samples.reshape(-1, 3).to(
        device=lab.device, dtype=lab.dtype
    )
    if not int(samples.shape[0]):
        raise ValueError("At least one Lab reference pixel is required.")
    eligible = eligible_mask.bool()
    floors = torch.as_tensor(scale_floors, device=lab.device, dtype=lab.dtype)
    metric_weights = torch.as_tensor(
        distance_weights, device=lab.device, dtype=lab.dtype
    )
    steps = torch.as_tensor(
        quantization_steps, device=lab.device, dtype=lab.dtype
    )
    quantized = torch.round(samples / steps).to(torch.int16)
    cells, counts = torch.unique(
        quantized, dim=0, return_counts=True
    )
    observed_cells = cells
    maximum_bins = max(1, int(maximum_bins))
    if int(cells.shape[0]) > maximum_bins:
        # Keeping only the most frequent cells makes affirmative evidence
        # non-monotonic: painting a larger, more varied foreground region can
        # evict an earlier colour completely.  Retain the dominant cell first,
        # then greedily cover Lab space with observed cells.  Occurrence still
        # influences selection and the fitted weights, but cannot consume all
        # slots with many neighbouring shades from one large painted patch.
        candidate_centres = cells.to(dtype=lab.dtype) * steps
        normalized = candidate_centres / floors
        normalized = normalized * torch.sqrt(metric_weights)[None, :]
        selected = torch.empty(
            maximum_bins, device=lab.device, dtype=torch.long
        )
        selected[0] = torch.argmax(counts)
        selected_mask = torch.zeros(
            int(cells.shape[0]), device=lab.device, dtype=torch.bool
        )
        selected_mask[selected[0]] = True
        minimum_distance = torch.sum(
            (normalized - normalized[selected[0]]).square(), dim=1
        )
        relative_frequency = torch.sqrt(
            counts.to(dtype=lab.dtype)
            / counts.max().to(dtype=lab.dtype).clamp_min(1.0)
        )
        coverage_weight = 0.25 + 0.75 * relative_frequency
        for index in range(1, maximum_bins):
            score = minimum_distance * coverage_weight
            score = score.masked_fill(selected_mask, -1.0)
            next_item = torch.argmax(score)
            selected[index] = next_item
            selected_mask[next_item] = True
            distance = torch.sum(
                (normalized - normalized[next_item]).square(), dim=1
            )
            minimum_distance = torch.minimum(minimum_distance, distance)
        cells = cells[selected]
    centres = cells.to(dtype=lab.dtype) * steps

    # Reassign every observed colour cell to the nearest retained representative
    # while preserving its full occurrence count. This avoids spatial-stride
    # sampling dropping a small but valid painted seed colour when the total
    # reference mask grows beyond the refinement sample budget.
    observed_centres = observed_cells.to(dtype=lab.dtype) * steps
    observed_counts = counts.to(dtype=lab.dtype)
    observed_delta = (
        observed_centres[:, None, :] - centres[None, :, :]
    ) / floors
    observed_distance = torch.sum(
        observed_delta.square() * metric_weights[None, None, :], dim=2
    )
    observed_assignment = torch.argmin(observed_distance, dim=1)
    anchor_counts = torch.zeros(
        int(centres.shape[0]), device=lab.device, dtype=lab.dtype
    )
    anchor_counts.scatter_add_(0, observed_assignment, observed_counts)
    component_weights = anchor_counts / anchor_counts.sum().clamp_min(1.0)
    scales = floors[None].repeat(int(centres.shape[0]), 1)

    def evaluate(current_scales):
        probability = torch.zeros(lab.shape[:2], device=lab.device, dtype=lab.dtype)
        strongest = torch.zeros_like(probability)
        assignments = torch.zeros(
            lab.shape[:2], device=lab.device, dtype=torch.int16
        )
        adjusted = component_weights.clamp_min(1e-6).pow(
            max(0.0, float(frequency_weight_power))
        )
        adjusted /= adjusted.max().clamp_min(1e-6)
        for start in range(0, int(centres.shape[0]), 4):
            stop = min(start + 4, int(centres.shape[0]))
            effective_scales = current_scales[start:stop] * max(
                0.05, float(scale_multiplier)
            )
            delta = (
                lab[:, :, None, :] - centres[None, None, start:stop, :]
            ) / effective_scales[None, None, :, :]
            distance = torch.sum(
                delta.square() * metric_weights[None, None, None, :], dim=3
            )
            membership = torch.exp(-0.5 * distance)
            probability += torch.sum(
                membership * adjusted[None, None, start:stop], dim=2
            )
            chunk_strongest, chunk_index = torch.max(membership, dim=2)
            replace = chunk_strongest > strongest
            strongest = torch.maximum(strongest, chunk_strongest)
            assignments = torch.where(
                replace,
                (chunk_index + start).to(torch.int16),
                assignments,
            )
        return probability.clamp(0.0, 1.0) * eligible, strongest, assignments

    probability, strongest, assignments = evaluate(scales)
    rounds_completed = 0
    refined_sample_count = int(samples.shape[0])
    for _ in range(max(0, int(refinement_iterations))):
        accepted_mask = eligible & (
            strongest >= float(refinement_min_probability)
        )
        accepted = lab[accepted_mask]
        accepted_assignment = assignments[accepted_mask].long()
        if not int(accepted.shape[0]):
            break
        if int(accepted.shape[0]) > maximum_fit_samples:
            indices = torch.linspace(
                0,
                int(accepted.shape[0]) - 1,
                maximum_fit_samples,
                device=lab.device,
            ).round().long()
            accepted = accepted[indices]
            accepted_assignment = accepted_assignment[indices]
        updated_scales = []
        for component in range(int(centres.shape[0])):
            members = accepted[accepted_assignment == component]
            if int(members.shape[0]) < 4:
                updated_scales.append(scales[component])
                continue
            deviation = torch.median(
                torch.abs(members - centres[component]), dim=0
            ).values * 1.4826
            # Interior refinement can widen an anchored mode modestly but can
            # never move its centre or grow without bound into seed-like tray
            # colours (or vice versa).
            updated_scales.append(
                torch.minimum(torch.maximum(deviation, floors), floors * 3.0)
            )
        scales = torch.stack(updated_scales)
        refined_sample_count = int(samples.shape[0] + accepted.shape[0])
        probability, strongest, assignments = evaluate(scales)
        rounds_completed += 1
    return (
        probability.clamp(0.0, 1.0),
        centres,
        scales,
        component_weights,
        refined_sample_count,
        rounds_completed,
    )


def connected_components(mask):
    """Label an 8-connected binary tensor with GPU union/find operations.

    Returns a label tensor and a compact CPU statistics dictionary. Pixel-level
    propagation and reductions remain on the tensor device; only one row per
    component is downloaded.
    """

    import torch

    binary = mask[0, 0].bool() if mask.ndim == 4 else mask.bool()
    height, width = binary.shape
    count = height * width
    index = torch.arange(1, count + 1, device=binary.device).reshape(height, width)
    parent = torch.arange(count + 1, device=binary.device, dtype=torch.int64)
    active = index[binary]
    parent[1:] = torch.where(binary.reshape(-1), parent[1:], torch.zeros_like(parent[1:]))
    edges = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0a, y1a = max(0, -dy), min(height, height - dy)
        x0a, x1a = max(0, -dx), min(width, width - dx)
        y0b, y1b = y0a + dy, y1a + dy
        x0b, x1b = x0a + dx, x1a + dx
        valid = binary[y0a:y1a, x0a:x1a] & binary[y0b:y1b, x0b:x1b]
        if valid.any():
            edges.append(
                (
                    index[y0a:y1a, x0a:x1a][valid],
                    index[y0b:y1b, x0b:x1b][valid],
                )
            )
    if active.numel() == 0:
        labels = torch.zeros((1, 1, height, width), device=binary.device, dtype=torch.int32)
        return labels, {
            "area": np.zeros(1, np.int64),
            "left": np.zeros(1, np.int64),
            "top": np.zeros(1, np.int64),
            "width": np.zeros(1, np.int64),
            "height": np.zeros(1, np.int64),
            "centroid_x": np.zeros(1, np.float32),
            "centroid_y": np.zeros(1, np.float32),
        }
    if edges:
        edge_a = torch.cat([edge[0] for edge in edges])
        edge_b = torch.cat([edge[1] for edge in edges])
        for _ in range(40):
            for _ in range(4):
                parent = parent[parent]
            root_a = parent[edge_a]
            root_b = parent[edge_b]
            lower = torch.minimum(root_a, root_b)
            higher = torch.maximum(root_a, root_b)
            previous = parent[higher]
            parent.scatter_reduce_(0, higher, lower, reduce="amin", include_self=True)
            if torch.equal(previous, parent[higher]):
                break
    for _ in range(8):
        parent = parent[parent]
    roots = parent[index]
    roots = torch.where(binary, roots, torch.zeros_like(roots))
    unique_roots, inverse = torch.unique(roots, sorted=True, return_inverse=True)
    labels_2d = inverse.reshape(height, width).to(torch.int64)
    if unique_roots[0] != 0:
        labels_2d += 1
    component_count = int(labels_2d.max().item()) + 1
    flat_labels = labels_2d.reshape(-1)
    areas = torch.bincount(flat_labels, minlength=component_count)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=binary.device, dtype=torch.int64),
        torch.arange(width, device=binary.device, dtype=torch.int64),
        indexing="ij",
    )
    maximum = torch.iinfo(torch.int64).max
    left = torch.full((component_count,), maximum, device=binary.device, dtype=torch.int64)
    top = left.clone()
    right = torch.full((component_count,), -1, device=binary.device, dtype=torch.int64)
    bottom = right.clone()
    left.scatter_reduce_(0, flat_labels, xx.reshape(-1), reduce="amin", include_self=True)
    top.scatter_reduce_(0, flat_labels, yy.reshape(-1), reduce="amin", include_self=True)
    right.scatter_reduce_(0, flat_labels, xx.reshape(-1), reduce="amax", include_self=True)
    bottom.scatter_reduce_(0, flat_labels, yy.reshape(-1), reduce="amax", include_self=True)
    sum_x = torch.zeros(component_count, device=binary.device, dtype=torch.float64)
    sum_y = torch.zeros_like(sum_x)
    sum_x.scatter_add_(0, flat_labels, xx.reshape(-1).double())
    sum_y.scatter_add_(0, flat_labels, yy.reshape(-1).double())
    safe_area = areas.clamp_min(1).double()
    statistics = {
        "area": areas.cpu().numpy(),
        "left": left.cpu().numpy(),
        "top": top.cpu().numpy(),
        "width": (right - left + 1).clamp_min(0).cpu().numpy(),
        "height": (bottom - top + 1).clamp_min(0).cpu().numpy(),
        "centroid_x": (sum_x / safe_area).float().cpu().numpy(),
        "centroid_y": (sum_y / safe_area).float().cpu().numpy(),
    }
    return labels_2d[None, None].to(torch.int32), statistics


def oriented_connected_components(
    mask,
    tangent_x,
    tangent_y,
    *,
    maximum_gap: int = 2,
    tangent_tolerance_degrees: float = 24.0,
    curvature_policy: str = "prefer",
    curvature_tolerance_degrees: float = 6.0,
):
    """Link ridge pixels on-device using tangent-compatible adjacency.

    Unlike ordinary connected components, this does not join crossing edges
    merely because their pixels touch. Short gaps are bridged only between two
    facing trace endpoints when the displacement follows both endpoint
    tangents and the signed tangent orientation preserves the same gradient
    side.  The endpoint rule is important: without it, increasing
    ``maximum_gap`` can connect two uninterrupted parallel ridges through a
    diagonal pair of otherwise compatible pixels. No component raster or
    statistics are transferred to the CPU.

    Optional curvature gating classifies every non-adjacent candidate bridge
    from its endpoint tangents.  After orienting both axial tangents towards
    the bridge chord, equal-signed tangent-to-chord bends describe a C-shaped
    (single-turn) continuation while opposite signs describe an S-shaped
    continuation through an inflection.  ``prefer`` rejects only confident
    S-bends, whereas ``require`` rejects every S-bend outside the configured
    angular ambiguity.  Immediate pixel neighbours are deliberately exempt:
    a one-pixel staircase chord is too quantized to support a curvature test.
    This is a local convexity constraint on initial assignments, not a claim
    that an arbitrarily long component has been globally proven convex.
    """

    import torch

    binary = mask[0, 0].bool() if mask.ndim == 4 else mask.bool()
    tx = tangent_x[0, 0] if tangent_x.ndim == 4 else tangent_x
    ty = tangent_y[0, 0] if tangent_y.ndim == 4 else tangent_y
    height, width = binary.shape
    curvature_policy = str(curvature_policy).strip().lower()
    if curvature_policy not in {"off", "prefer", "require"}:
        raise ValueError(
            "Curvature policy must be 'off', 'prefer', or 'require'."
        )
    curvature_tolerance_degrees = float(curvature_tolerance_degrees)
    if not 0.0 <= curvature_tolerance_degrees <= 45.0:
        raise ValueError(
            "Curvature tolerance must be between 0 and 45 degrees."
        )
    required_curvature_deadband = float(
        np.sin(np.deg2rad(curvature_tolerance_degrees))
    )
    preferred_curvature_deadband = float(
        np.sin(
            np.deg2rad(
                min(
                    60.0,
                    max(
                        2.0 * curvature_tolerance_degrees,
                        0.5 * float(tangent_tolerance_degrees),
                    ),
                )
            )
        )
    )
    count = height * width
    index = torch.arange(
        1, count + 1, device=binary.device, dtype=torch.int64
    ).reshape(height, width)
    parent = torch.arange(count + 1, device=binary.device, dtype=torch.int64)
    parent[1:] = torch.where(
        binary.reshape(-1), parent[1:], torch.zeros_like(parent[1:])
    )
    orientation_limit = float(
        np.cos(np.deg2rad(tangent_tolerance_degrees))
    )
    adjacent_direction_limit = float(
        np.cos(np.deg2rad(min(75.0, tangent_tolerance_degrees * 1.6)))
    )
    # A long bridge is less constrained by the pixel grid than immediate
    # 8-neighbour adjacency, so do not give it the former broad (up to 75
    # degree) directional allowance.  Twenty-five degrees still admits a
    # four-pixel chord across a realistically curved seed rim, while rejecting
    # the diagonal cross-links that merge close parallel rims.
    bridge_direction_limit = float(
        np.cos(
            np.deg2rad(
                max(2.0, min(25.0, float(tangent_tolerance_degrees)))
            )
        )
    )

    maximum_gap = max(1, int(maximum_gap))
    neighbour_bits = None
    neighbour_offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    if maximum_gap > 1:
        # Encode the eight-neighbour occupancy once.  Gap candidates can then
        # test whether each ridge pixel is a genuine endpoint without eight
        # additional full-raster comparisons per candidate displacement.
        neighbour_bits = torch.zeros_like(binary, dtype=torch.int16)
        for bit, (neighbour_dy, neighbour_dx) in enumerate(neighbour_offsets):
            source_y0 = max(0, -neighbour_dy)
            source_y1 = min(height, height - neighbour_dy)
            source_x0 = max(0, -neighbour_dx)
            source_x1 = min(width, width - neighbour_dx)
            target_y0, target_y1 = (
                source_y0 + neighbour_dy,
                source_y1 + neighbour_dy,
            )
            target_x0, target_x1 = (
                source_x0 + neighbour_dx,
                source_x1 + neighbour_dx,
            )
            occupied = binary[
                target_y0:target_y1, target_x0:target_x1
            ].to(torch.int16)
            neighbour_bits[
                source_y0:source_y1, source_x0:source_x1
            ] |= occupied << bit

    def neighbour_cone_bits(ux: float, uy: float) -> tuple[int, int]:
        """Return narrow forward/backward endpoint-occupancy cones."""

        forward = 0
        backward = 0
        for bit, (neighbour_dy, neighbour_dx) in enumerate(neighbour_offsets):
            neighbour_length = float(np.hypot(neighbour_dx, neighbour_dy))
            projection = (
                neighbour_dx * ux + neighbour_dy * uy
            ) / neighbour_length
            if projection >= 0.9:
                forward |= 1 << bit
            elif projection <= -0.9:
                backward |= 1 << bit
        return forward, backward

    edge_a = []
    edge_b = []
    for gap in range(1, maximum_gap + 1):
        offsets = []
        for dy in range(-gap, gap + 1):
            for dx in range(-gap, gap + 1):
                if max(abs(dx), abs(dy)) != gap:
                    continue
                if dy < 0 or (dy == 0 and dx <= 0):
                    continue
                offsets.append((dy, dx))
        for dy, dx in offsets:
            y0a, y1a = max(0, -dy), min(height, height - dy)
            x0a, x1a = max(0, -dx), min(width, width - dx)
            y0b, y1b = y0a + dy, y1a + dy
            x0b, x1b = x0a + dx, x1a + dx
            tx_a, ty_a = tx[y0a:y1a, x0a:x1a], ty[y0a:y1a, x0a:x1a]
            tx_b, ty_b = tx[y0b:y1b, x0b:x1b], ty[y0b:y1b, x0b:x1b]
            signed_agreement = tx_a * tx_b + ty_a * ty_b
            axial_agreement = torch.abs(signed_agreement)
            length = float(np.hypot(dx, dy))
            ux, uy = dx / length, dy / length
            follows_a = torch.abs(tx_a * ux + ty_a * uy)
            follows_b = torch.abs(tx_b * ux + ty_b * uy)
            direction_limit = (
                adjacent_direction_limit
                if gap == 1
                else bridge_direction_limit
            )
            connected = (
                binary[y0a:y1a, x0a:x1a]
                & binary[y0b:y1b, x0b:x1b]
                & (axial_agreement >= orientation_limit)
                & (follows_a >= direction_limit)
                & (follows_b >= direction_limit)
            )
            if gap > 1:
                if curvature_policy != "off":
                    # Tangents are axial for this geometric test.  Orient
                    # each towards the A->B chord so the result is invariant
                    # to endpoint order, tangent sign and curve winding.
                    dot_a = tx_a * ux + ty_a * uy
                    dot_b = tx_b * ux + ty_b * uy
                    sign_a = torch.where(
                        dot_a >= 0.0,
                        torch.ones_like(dot_a),
                        -torch.ones_like(dot_a),
                    )
                    sign_b = torch.where(
                        dot_b >= 0.0,
                        torch.ones_like(dot_b),
                        -torch.ones_like(dot_b),
                    )
                    bend_a = sign_a * (tx_a * uy - ty_a * ux)
                    bend_b = sign_b * (ux * ty_b - uy * tx_b)
                    opposite_bends = bend_a * bend_b < 0.0
                    deadband = (
                        preferred_curvature_deadband
                        if curvature_policy == "prefer"
                        else required_curvature_deadband
                    )
                    resolved_bends = torch.minimum(
                        torch.abs(bend_a), torch.abs(bend_b)
                    ) > deadband
                    connected &= ~(opposite_bends & resolved_bends)

                # The tangent sign is inherited from the directed image
                # gradient. Facing boundaries of adjacent seeds therefore
                # have opposite signs even though their *axial* tangents are
                # parallel. Do not bridge across that semantic side change.
                connected &= signed_agreement >= orientation_limit

                # Require at least one end of the proposed bridge to be open
                # in its narrow chord direction. Requiring both ends to be
                # topological one-pixel endpoints is too brittle after ridge
                # thinning and junction removal: a harmless staircase pixel
                # can sit beside one end. Two uninterrupted parallel traces,
                # however, have continuation at both sampled pixels and are
                # therefore never eligible for a gap bridge.
                forward_bits, backward_bits = neighbour_cone_bits(ux, uy)
                bits_a = neighbour_bits[y0a:y1a, x0a:x1a]
                bits_b = neighbour_bits[y0b:y1b, x0b:x1b]
                a_is_facing_endpoint = (bits_a & forward_bits) == 0
                b_is_facing_endpoint = (bits_b & backward_bits) == 0
                connected &= a_is_facing_endpoint | b_is_facing_endpoint
            edge_a.append(index[y0a:y1a, x0a:x1a][connected])
            edge_b.append(index[y0b:y1b, x0b:x1b][connected])
    if edge_a:
        a = torch.cat(edge_a)
        b = torch.cat(edge_b)
        for _ in range(16):
            for _ in range(4):
                parent = parent[parent]
            root_a, root_b = parent[a], parent[b]
            lower = torch.minimum(root_a, root_b)
            higher = torch.maximum(root_a, root_b)
            parent.scatter_reduce_(
                0, higher, lower, reduce="amin", include_self=True
            )
    for _ in range(8):
        parent = parent[parent]
    roots = torch.where(binary, parent[index], torch.zeros_like(index))
    _, inverse = torch.unique(roots, sorted=True, return_inverse=True)
    return inverse.reshape(1, 1, height, width).to(torch.int32)


def _shift_with_invalid(coordinates, dy: int, dx: int):
    import torch

    shifted = torch.full_like(coordinates, -1)
    height, width = coordinates.shape[-2:]
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    target_y0, target_y1 = source_y0 + dy, source_y1 + dy
    target_x0, target_x1 = source_x0 + dx, source_x1 + dx
    shifted[..., target_y0:target_y1, target_x0:target_x1] = coordinates[
        ..., source_y0:source_y1, source_x0:source_x1
    ]
    return shifted


def distance_transform(mask):
    """Approximate Euclidean distance transform using CUDA jump flooding."""

    import torch

    binary = mask.bool()
    height, width = binary.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=binary.device, dtype=torch.float32),
        torch.arange(width, device=binary.device, dtype=torch.float32),
        indexing="ij",
    )
    seed = ~binary[0, 0]
    coordinates = torch.stack((yy, xx), dim=0)[None]
    coordinates = torch.where(seed[None, None], coordinates, torch.full_like(coordinates, -1))
    pixel = torch.stack((yy, xx), dim=0)[None]
    best_distance = torch.where(seed[None, None], torch.zeros_like(pixel[:, :1]), torch.full_like(pixel[:, :1], float("inf")))
    jump = 1
    while jump < max(height, width):
        jump *= 2
    jump //= 2
    while jump >= 1:
        for dy, dx in (
            (-jump, -jump), (-jump, 0), (-jump, jump),
            (0, -jump), (0, jump),
            (jump, -jump), (jump, 0), (jump, jump),
        ):
            candidate = _shift_with_invalid(coordinates, dy, dx)
            valid = candidate[:, 0:1] >= 0
            candidate_distance = (
                (candidate[:, 0:1] - pixel[:, 0:1]).square()
                + (candidate[:, 1:2] - pixel[:, 1:2]).square()
            )
            improve = valid & (candidate_distance < best_distance)
            best_distance = torch.where(improve, candidate_distance, best_distance)
            coordinates = torch.where(improve.expand_as(coordinates), candidate, coordinates)
        jump //= 2
    return torch.sqrt(best_distance.clamp_min(0.0)) * binary.float()


def warp_perspective(image, matrix, output_size: tuple[int, int]):
    """Apply a source-to-destination homography with CUDA grid sampling."""

    import torch
    import torch.nn.functional as functional

    output_height, output_width = output_size
    transform = torch.as_tensor(matrix, device=image.device, dtype=torch.float32)
    inverse = torch.linalg.inv(transform)
    yy, xx = torch.meshgrid(
        torch.arange(output_height, device=image.device, dtype=torch.float32),
        torch.arange(output_width, device=image.device, dtype=torch.float32),
        indexing="ij",
    )
    homogeneous = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    source = homogeneous @ inverse.T
    source_x = source[..., 0] / source[..., 2].clamp_min(1e-8)
    source_y = source[..., 1] / source[..., 2].clamp_min(1e-8)
    input_height, input_width = image.shape[-2:]
    grid_x = source_x / max(input_width - 1, 1) * 2.0 - 1.0
    grid_y = source_y / max(input_height - 1, 1) * 2.0 - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)[None]
    return functional.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def bilinear_sample(image, x, y):
    """Sample a single-channel image at arbitrary CUDA coordinates."""

    import torch

    values = image[0, 0] if image.ndim == 4 else image
    height, width = values.shape
    x0 = torch.floor(x).long().clamp(0, width - 1)
    y0 = torch.floor(y).long().clamp(0, height - 1)
    x1 = (x0 + 1).clamp(0, width - 1)
    y1 = (y0 + 1).clamp(0, height - 1)
    wx = x - x0.float()
    wy = y - y0.float()
    top = values[y0, x0] * (1.0 - wx) + values[y0, x1] * wx
    bottom = values[y1, x0] * (1.0 - wx) + values[y1, x1] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    outside = (x < 0) | (x > width - 1) | (y < 0) | (y > height - 1)
    return torch.where(outside, torch.zeros_like(sampled), sampled)


def gradient_magnitude(gray, *, scharr: bool = False):
    import torch
    import torch.nn.functional as functional

    if scharr:
        kernel_x = torch.tensor(
            ((-3.0, 0.0, 3.0), (-10.0, 0.0, 10.0), (-3.0, 0.0, 3.0)),
            device=gray.device,
            dtype=gray.dtype,
        )[None, None] / 32.0
    else:
        kernel_x = torch.tensor(
            ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)),
            device=gray.device,
            dtype=gray.dtype,
        )[None, None] / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    padded = functional.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = functional.conv2d(padded, kernel_x)
    gy = functional.conv2d(padded, kernel_y)
    return torch.sqrt(gx.square() + gy.square() + 1e-8), gx, gy
