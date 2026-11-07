"""Automatic Foreground, Background and Other reference areas.

Without any painting, an image still gets an initial estimate of each material
class. Foreground comes from confident seeds: the isolated reference seeds when
they were found, otherwise confident isolated seeds in the vessel. Other comes
from the vessel wall and from the colour chart and ruler. Background comes from
the perimeter ring outside the vessel, or from rings around the confident seeds
when there is no ring or it disagrees with them, or from the image border.

These areas are priors, not peers of painted references. ``authority`` gives
the weight a class's automatic area keeps against reviewed references, and
``thinned`` turns that weight into a spatially uniform subset of the area, so
every pooled consumer weighs it consistently. Pixel work stays on the tensor
device; only per-seed windows of a few thousand pixels use OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from seedvision.cuda.ops import GpuRaster, binary_dilate, binary_open


@dataclass(frozen=True, slots=True)
class ConfidentSeed:
    """A confidently isolated seed, as a local mask in image coordinates."""

    left: int
    top: int
    mask: np.ndarray
    diameter_px: float
    # "reference" (isolated reference seeds), "vessel" or "scene" (the backup
    # search inside a visible vessel or across the whole scene).
    source: str = "reference"


@dataclass(slots=True)
class AutomaticMaterialReferences:
    """Full-image automatic reference areas on the tensor device."""

    foreground: object | None
    background: object | None
    other: object | None
    foreground_samples_lab: object | None
    background_samples_lab: object | None
    background_prior_lab: tuple[float, float, float] | None
    foreground_source: str
    background_source: str
    other_sources: tuple[str, ...]
    seed_count: int
    perimeter_disagreement: float | None = None

    @property
    def replaces_perimeter(self) -> bool:
        return self.background is not None and self.background_samples_lab is not None

    def label_raster(self, shape_hw: tuple[int, int], perimeter_mask=None) -> GpuRaster:
        """One uint8 raster: 1 Foreground, 2 Background, 3 Other, 0 none."""

        import torch

        reference = next(
            (mask for mask in (self.foreground, self.background, self.other) if mask is not None),
            None,
        )
        device = reference.device if reference is not None else "cpu"
        labels = torch.zeros(shape_hw, dtype=torch.uint8, device=device)
        background = self.background
        if background is None and self.background_source == "perimeter ring" and perimeter_mask is not None:
            background = perimeter_mask
        for mask, value in ((background, 2), (self.other, 3), (self.foreground, 1)):
            if mask is not None:
                labels[mask.to(device=device, dtype=torch.bool)] = value
        return GpuRaster(labels[None, None], numpy_dtype=np.uint8, name="automatic material references")


def authority(reviewed_pixels: int, nominal_seed_area: float, floor: float, half_life: float) -> float:
    """Weight an automatic area keeps once reviewed references exist."""

    if reviewed_pixels <= 0:
        return 1.0
    reviewed_seed_areas = reviewed_pixels / max(1.0, float(nominal_seed_area))
    return float(floor + (1.0 - floor) * 2.0 ** (-reviewed_seed_areas / max(0.05, float(half_life))))


def thinned(mask, reviewed_pixels: int, weight: float):
    """Keep a uniform subset of ``mask`` worth ``weight`` reviewed pixels each.

    With no reviewed pixels the whole area is kept. Otherwise the automatic
    area keeps about ``weight * reviewed_pixels`` pixels, so a pooled fit gives
    it ``weight`` of the reviewed references' influence.
    """

    import torch

    if mask is None:
        return None
    total = int(mask.sum().item())
    if reviewed_pixels <= 0 or total == 0:
        return mask
    fraction = min(1.0, max(0.0, weight) * reviewed_pixels / total)
    if fraction >= 1.0:
        return mask
    if fraction <= 0.0:
        return torch.zeros_like(mask)
    height, width = mask.shape[-2:]
    yy = torch.arange(height, device=mask.device, dtype=torch.int64)[:, None]
    xx = torch.arange(width, device=mask.device, dtype=torch.int64)[None, :]
    # A fixed integer hash gives a deterministic, spatially uniform pattern.
    hashed = ((xx * 73856093) ^ (yy * 19349663)) % 1000003
    return mask & (hashed < round(fraction * 1000003))


def sampled_rows(lab_tensor, mask, maximum: int = 32768):
    """Up to ``maximum`` evenly spaced Lab rows of the pixels in ``mask``."""

    import torch

    if mask is None:
        return None
    flat = mask.reshape(-1)
    count = int(flat.sum().item())
    if count == 0:
        return None
    indices = torch.nonzero(flat, as_tuple=False).flatten()
    if count > maximum:
        indices = indices[torch.linspace(0, count - 1, maximum, device=indices.device).round().long()]
    return lab_tensor[0].reshape(3, -1)[:, indices].T.contiguous()


def polygon_mask(points: np.ndarray, shape_hw, device, *, dilation_px: int = 0):
    """Rasterize the convex hull of ``points`` on the tensor device.

    Only the polygon's bounding window is touched; ``dilation_px`` grows the
    shape by that many pixels in every direction.
    """

    import torch
    import torch.nn.functional as F

    height, width = shape_hw
    mask = torch.zeros(shape_hw, dtype=torch.bool, device=device)
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if len(points) < 3 or not np.isfinite(points).all():
        return mask
    points = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    if len(points) < 3:
        return mask
    pad = max(0, int(dilation_px))
    x0 = max(0, int(np.floor(points[:, 0].min())) - pad)
    x1 = min(width, int(np.ceil(points[:, 0].max())) + 1 + pad)
    y0 = max(0, int(np.floor(points[:, 1].min())) - pad)
    y1 = min(height, int(np.ceil(points[:, 1].max())) + 1 + pad)
    if x1 <= x0 or y1 <= y0:
        return mask
    yy, xx = torch.meshgrid(
        torch.arange(y0, y1, device=device, dtype=torch.float32),
        torch.arange(x0, x1, device=device, dtype=torch.float32),
        indexing="ij",
    )
    signed_area = 0.5 * float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1])
    )
    orientation = 1.0 if signed_area >= 0 else -1.0
    inside = torch.ones_like(xx, dtype=torch.bool)
    for (ax, ay), (bx, by) in zip(points, np.roll(points, -1, axis=0)):
        cross = (bx - ax) * (yy - ay) - (by - ay) * (xx - ax)
        inside &= cross * orientation >= 0.0
    if pad:
        grown = inside[None, None].float()
        grown = F.max_pool2d(grown, (1, 2 * pad + 1), stride=1, padding=(0, pad))
        grown = F.max_pool2d(grown, (2 * pad + 1, 1), stride=1, padding=(pad, 0))
        inside = grown[0, 0] > 0
    mask[y0:y1, x0:x1] = inside
    return mask


def vessel_rim_support(lab_tensor, dish, *, angles: int = 360) -> float:
    """Fraction of the fitted outer vessel edge that shows a visible rim.

    Along each of ``angles`` radial lines through the fitted outer ellipse, the
    lightness step nearest the ellipse must stand well above that line's
    typical variation. A glass dish shows its rim almost all the way round; a
    circle fitted to a seed pile or a paper edge does not. Lines mostly outside
    the image are left out.
    """

    import torch
    import torch.nn.functional as F

    height, width = lab_tensor.shape[-2:]
    outer_x, outer_y = dish.outer_axes
    if outer_x <= 4.0 or outer_y <= 4.0:
        return 0.0
    device = lab_tensor.device
    theta = torch.arange(angles, device=device, dtype=torch.float32) * (2.0 * np.pi / angles)
    rotation = np.deg2rad(float(getattr(dish, "ellipse_angle_degrees", 0.0)))
    px, py = outer_x * torch.cos(theta), outer_y * torch.sin(theta)
    dx = px * float(np.cos(rotation)) - py * float(np.sin(rotation))
    dy = px * float(np.sin(rotation)) + py * float(np.cos(rotation))
    radius = torch.sqrt(dx.square() + dy.square()).clamp_min(1.0)
    nx, ny = dx / radius, dy / radius
    samples = max(64, min(1024, int(0.30 * max(outer_x, outer_y))))
    fraction = torch.linspace(-0.15, 0.15, samples, device=device)
    offsets = fraction[None, :] * radius[:, None]
    xs = float(dish.center_x) + dx[:, None] + offsets * nx[:, None]
    ys = float(dish.center_y) + dy[:, None] + offsets * ny[:, None]
    grid = torch.stack(
        (xs / max(1, width - 1) * 2.0 - 1.0, ys / max(1, height - 1) * 2.0 - 1.0), dim=-1
    )[None]
    lightness = F.grid_sample(
        lab_tensor[:, 0:1].float(), grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0, 0]
    kernel = torch.exp(-0.5 * (torch.arange(-3, 4, device=device, dtype=torch.float32) / 1.5).square())
    lightness = F.conv1d(lightness[:, None], (kernel / kernel.sum())[None, None], padding=3)[:, 0]
    spacing = (offsets[:, 1:] - offsets[:, :-1]).clamp_min(1e-3)
    step = (lightness[:, 1:] - lightness[:, :-1]).abs() / spacing
    inside = ((xs >= 0) & (xs <= width - 1) & (ys >= 0) & (ys <= height - 1)).float().mean(dim=1)
    near = (fraction[1:].abs() <= 0.02)[None, :].expand_as(step)
    peak = torch.where(near, step, torch.zeros_like(step)).max(dim=1).values
    typical = step.median(dim=1).values
    visible = peak >= torch.clamp(3.0 * typical, min=1.0)
    counted = inside >= 0.8
    if int(counted.sum().item()) < angles // 4:
        return 0.0
    return float((visible & counted).sum().item() / counted.sum().item())


def _vessel_wall_mask(dish, pixels_per_mm, wall_width_mm: float, shape_hw, device):
    """The vessel wall between its inner and outer edges, as fitted ellipses.

    Without a detected inner edge the wall is the configured width inside the
    outer edge.
    """

    import torch

    height, width = shape_hw
    mask = torch.zeros(shape_hw, dtype=torch.bool, device=device)
    outer_x, outer_y = dish.outer_axes
    if outer_x <= 1.0 or outer_y <= 1.0:
        return mask
    wall = (
        wall_width_mm * pixels_per_mm
        if pixels_per_mm is not None and pixels_per_mm > 0
        else 0.02 * max(outer_x, outer_y)
    )
    inner_x, inner_y = dish.inner_axes
    if not (getattr(dish, "rim_pair_detected", False) and 1.0 < inner_x < outer_x and 1.0 < inner_y < outer_y):
        inner_x, inner_y = outer_x - wall, outer_y - wall
    if inner_x <= 0.0 or inner_y <= 0.0:
        return mask
    left, top, right, bottom = dish.bounds[:4]
    x0, y0 = max(0, int(left) - 2), max(0, int(top) - 2)
    x1, y1 = min(width, int(right) + 3), min(height, int(bottom) + 3)
    if x1 <= x0 or y1 <= y0:
        return mask
    yy, xx = torch.meshgrid(
        torch.arange(y0, y1, device=device, dtype=torch.float32) - float(dish.center_y),
        torch.arange(x0, x1, device=device, dtype=torch.float32) - float(dish.center_x),
        indexing="ij",
    )
    angle = np.deg2rad(float(getattr(dish, "ellipse_angle_degrees", 0.0)))
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    along = xx * cosine + yy * sine
    across = -xx * sine + yy * cosine
    inside_outer = (along / outer_x).square() + (across / outer_y).square() <= 1.0
    outside_inner = (along / inner_x).square() + (across / inner_y).square() >= 1.0
    mask[y0:y1, x0:x1] = inside_outer & outside_inner
    return mask


def ruler_outline(ruler) -> np.ndarray | None:
    """The ruler's outline in its detection frame, or ``None``."""

    if ruler is None:
        return None
    if len(getattr(ruler, "outline_points", ())) >= 3:
        return np.asarray(ruler.outline_points, np.float64)
    a = np.asarray(ruler.endpoint_a, np.float64)
    b = np.asarray(ruler.endpoint_b, np.float64)
    direction = b - a
    length = float(np.hypot(*direction))
    if length <= 0.0 or ruler.width_px <= 0.0:
        return None
    normal = np.array((-direction[1], direction[0])) / length * (ruler.width_px * 0.5)
    return np.array((a + normal, b + normal, b - normal, a - normal))


def automatic_material_references(
    lab_tensor,
    dish,
    seeds: tuple[ConfidentSeed, ...],
    seed_diameter: float,
    pixels_per_mm: float | None,
    *,
    vessel_present: bool,
    perimeter_samples_lab=None,
    chart_polygon: np.ndarray | None = None,
    ruler_polygon: np.ndarray | None = None,
    settings,
) -> AutomaticMaterialReferences:
    """Build the three automatic reference areas for one corrected image."""

    import torch

    device = lab_tensor.device
    height, width = lab_tensor.shape[-2:]
    diameter = max(4.0, float(seed_diameter))

    # Foreground: confident seed interiors, inset away from edges and shadows.
    seed_area = torch.zeros((height, width), dtype=torch.bool, device=device)
    foreground = torch.zeros_like(seed_area)
    inset = max(1.0, settings.automatic_seed_inset_fraction * diameter)
    gap = max(1.0, settings.automatic_background_gap_fraction * diameter)
    band = max(1.0, settings.automatic_background_band_fraction * diameter)
    rings = torch.zeros_like(seed_area)
    for seed in seeds:
        local = np.asarray(seed.mask, np.uint8)
        if not local.any():
            continue
        pad = int(np.ceil(gap + band)) + 2
        padded = cv2.copyMakeBorder(local, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        left, top = seed.left - pad, seed.top - pad
        inside_distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
        outside_distance = cv2.distanceTransform(np.uint8(padded == 0), cv2.DIST_L2, 5)
        interior = inside_distance >= inset
        ring = (outside_distance >= gap) & (outside_distance <= gap + band)
        # Clip the window to the image.
        cx0, cy0 = max(0, -left), max(0, -top)
        cx1 = min(padded.shape[1], width - left)
        cy1 = min(padded.shape[0], height - top)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        ix0, iy0 = left + cx0, top + cy0
        ix1, iy1 = left + cx1, top + cy1
        window = (slice(iy0, iy1), slice(ix0, ix1))
        seed_area[window] |= torch.from_numpy(padded[cy0:cy1, cx0:cx1] > 0).to(device)
        foreground[window] |= torch.from_numpy(interior[cy0:cy1, cx0:cx1]).to(device)
        ring_tensor = torch.from_numpy(ring[cy0:cy1, cx0:cx1]).to(device)
        if bool(ring_tensor.any().item()):
            ring_lab = lab_tensor[0, :, iy0:iy1, ix0:ix1]
            values = ring_lab[:, ring_tensor]
            # Drop the darkest part of each ring (cast shadows) and anything far
            # from the ring's own typical colour (neighbouring seeds, debris).
            darkness_limit = torch.quantile(values[0], settings.automatic_background_shadow_fraction)
            median = values.median(dim=1).values
            distance = torch.sqrt(
                (ring_lab - median[:, None, None]).square().sum(dim=0)
            )
            keep = ring_tensor & (ring_lab[0] >= darkness_limit) & (
                distance <= settings.automatic_background_colour_tolerance
            )
            rings[window] |= keep

    # Other: the vessel wall, and the chart and ruler wherever they lie. Seeds
    # heaped against the wall, or a fitted rim slightly inside the glass, put
    # seeds in the wall band. Seed-coloured blobs (closer to the confident
    # seeds than to the surface around the vessel, and at least 0.4 of a seed
    # wide) stay out of Other; the rim's own thin lines remain.
    other = torch.zeros((height, width), dtype=torch.bool, device=device)
    other_sources: list[str] = []
    if settings.automatic_other_reference:
        if vessel_present:
            wall = _vessel_wall_mask(
                dish, pixels_per_mm, settings.automatic_wall_width_mm, (height, width), device
            )
            wall &= ~seed_area
            surface = None
            if perimeter_samples_lab is not None and int(perimeter_samples_lab.shape[0]) >= 16:
                surface = perimeter_samples_lab.to(device=device, dtype=torch.float32).median(dim=0).values
            elif bool(rings.any().item()):
                surface = lab_tensor[0][:, rings & ~seed_area].float().median(dim=1).values
            if surface is not None and bool(foreground.any().item()) and bool(wall.any().item()):
                seed_colour = lab_tensor[0][:, foreground].float().median(dim=1).values
                rows = torch.nonzero(wall.any(dim=1)).flatten()
                columns = torch.nonzero(wall.any(dim=0)).flatten()
                y0, y1 = int(rows[0].item()), int(rows[-1].item()) + 1
                x0, x1 = int(columns[0].item()), int(columns[-1].item()) + 1
                pixels = lab_tensor[0, :, y0:y1, x0:x1].float()
                to_seed = (pixels - seed_colour[:, None, None]).square().sum(dim=0)
                to_surface = (pixels - surface[:, None, None]).square().sum(dim=0)
                kernel = max(3, int(0.4 * diameter) | 1)
                blobs = binary_dilate(
                    binary_open((to_seed < to_surface)[None, None], kernel), 3
                )[0, 0]
                wall[y0:y1, x0:x1] &= ~blobs
            other |= wall
            if bool(wall.any().item()):
                other_sources.append("vessel wall")
        if settings.automatic_chart_as_other:
            if chart_polygon is not None:
                chart_mask = polygon_mask(chart_polygon, (height, width), device)
                if bool(chart_mask.any().item()):
                    other |= chart_mask
                    other_sources.append("colour chart")
            if ruler_polygon is not None:
                ruler_mask = polygon_mask(ruler_polygon, (height, width), device)
                if bool(ruler_mask.any().item()):
                    other |= ruler_mask
                    other_sources.append("ruler")
    if not bool(other.any().item()):
        other = None
        other_sources = []

    foreground_source = "none"
    if settings.automatic_foreground_reference and seeds and bool(foreground.any().item()):
        if other is not None:
            foreground &= ~other
        foreground_source = {
            "reference": "isolated reference seeds",
            "vessel": "confident seeds in the vessel",
            "scene": "confident seeds in the scene",
        }[getattr(seeds[0], "source", "reference")]
    else:
        foreground = None

    # Background: perimeter ring, rings around the seeds, or the image border.
    background = None
    background_samples = None
    background_prior = None
    disagreement = None
    background_source = "none"
    if settings.automatic_background_reference:
        rings &= ~seed_area
        if other is not None:
            rings &= ~other
        ring_samples = sampled_rows(lab_tensor, rings) if bool(rings.any().item()) else None
        if ring_samples is not None and int(ring_samples.shape[0]) < 32:
            ring_samples = None
        perimeter_available = perimeter_samples_lab is not None and int(perimeter_samples_lab.shape[0]) >= 16
        if perimeter_available and ring_samples is not None:
            perimeter_median = perimeter_samples_lab.to(device=device, dtype=torch.float32).median(dim=0).values
            ring_median = ring_samples.float().median(dim=0).values
            # OpenCV 8-bit Lab: convert the lightness step to CIE units.
            delta = (perimeter_median - ring_median) * torch.tensor((100.0 / 255.0, 1.0, 1.0), device=device)
            disagreement = float(torch.sqrt(delta.square().sum()).item())
        if perimeter_available and (
            disagreement is None or disagreement <= settings.automatic_perimeter_disagreement
        ):
            background_source = "perimeter ring"
        elif ring_samples is not None:
            background, background_samples = rings, ring_samples
            background_source = "around confident seeds"
        else:
            border = max(2, round(settings.automatic_border_band_fraction * min(height, width)))
            edge = torch.zeros((height, width), dtype=torch.bool, device=device)
            edge[:border, :] = True
            edge[-border:, :] = True
            edge[:, :border] = True
            edge[:, -border:] = True
            edge &= ~seed_area
            if other is not None:
                edge &= ~other
            border_samples = sampled_rows(lab_tensor, edge)
            if border_samples is not None and int(border_samples.shape[0]) >= 64:
                median = border_samples.float().median(dim=0).values
                distance = torch.sqrt((lab_tensor[0].float() - median[:, None, None]).square().sum(dim=0))
                edge &= distance <= settings.automatic_background_colour_tolerance
                border_samples = sampled_rows(lab_tensor, edge)
                if border_samples is not None and int(border_samples.shape[0]) >= 64:
                    background, background_samples = edge, border_samples
                    background_source = "image border"
        if background_samples is not None:
            background_prior = tuple(
                float(value) for value in background_samples.float().median(dim=0).values.cpu().tolist()
            )
        if background is not None and foreground is not None:
            background &= ~foreground

    return AutomaticMaterialReferences(
        foreground=foreground,
        background=background,
        other=other,
        foreground_samples_lab=None if foreground is None else sampled_rows(lab_tensor, foreground),
        background_samples_lab=background_samples,
        background_prior_lab=background_prior,
        foreground_source=foreground_source,
        background_source=background_source,
        other_sources=tuple(other_sources),
        seed_count=len(seeds),
        perimeter_disagreement=disagreement,
    )


def describe(references: AutomaticMaterialReferences | None) -> str:
    """One status line naming the source of each automatic area."""

    if references is None:
        return "No automatic references"
    other = ", ".join(references.other_sources) if references.other_sources else "none"
    text = (
        f"Foreground: {references.foreground_source}; "
        f"Background: {references.background_source}; Other: {other}"
    )
    if references.perimeter_disagreement is not None and references.background_source != "perimeter ring":
        text += f" (perimeter ring differs by ΔE {references.perimeter_disagreement:.0f})"
    return text


_OVERLAY_COLOURS = {
    1: (40, 200, 70, 175),
    2: (240, 150, 30, 150),
    3: (210, 60, 210, 165),
}


def overlay_rgba(
    references: AutomaticMaterialReferences,
    shape_hw: tuple[int, int],
    *,
    perimeter_mask=None,
    maximum_dimension: int = 2048,
) -> np.ndarray:
    """A display-sized RGBA image of the three areas, coloured on the device.

    Foreground is green, Background orange and Other magenta. Only the
    downscaled RGBA image leaves the device.
    """

    import torch
    import torch.nn.functional as F

    labels = references.label_raster(shape_hw, perimeter_mask).gpu_tensor()
    height, width = shape_hw
    scale = min(1.0, maximum_dimension / max(height, width))
    if scale < 1.0:
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        labels = F.interpolate(labels.float(), size=size, mode="nearest").to(torch.uint8)
    labels = labels[0, 0]
    palette = torch.zeros((4, 4), dtype=torch.uint8, device=labels.device)
    for value, colour in _OVERLAY_COLOURS.items():
        palette[value] = torch.tensor(colour, dtype=torch.uint8, device=labels.device)
    return np.ascontiguousarray(palette[labels.long()].cpu().numpy())
