"""PlantCV's watershed debug drawing on a private CUDA RGB canvas.

The public watershed function returns labels, not this image. Palette generation
is compact CPU metadata; masks, bounding-box reductions and painting stay CUDA.
The source's palette triples are passed literally to OpenCV as BGR, then the
native presentation boundary reverses them to RGB. A local legacy RNG makes the
otherwise ambient source shuffle reproducible without changing global RNG state.
"""
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import torch

from . import plantcv_contours as contours
from . import plantcv_raster_spans as raster
from . import plantcv_size as size

MAX_PIXELS = 16_000_000
MAX_LABEL_ID = 65_536
MAX_LABELS = 4096
MAX_TILE_PIXELS = 1_000_000
MAX_CONTOURS = 65_536
MAX_POINTS = 1_000_000
MAX_SCAN_WORK = 8_000_000_000
MAX_GEOMETRY_WORK = 64_000_000
MAX_CPU_GEOMETRY_BYTES = 256 * 1024 * 1024
MAX_REGISTRY_COLORS = 1_000_000
SOURCE_SHA256 = 'd5fd90bf4e064117ff47ee333d2b83e62134901fda36f6966ae1b0be4a82a712'
PALETTE_SOURCE_SHA256 = '915191e9c21896155b7cbec9edeb1103f2f4a319dca86b0d02e0337a5bfefac0'
RNG_PROFILE = 'numpy.random.RandomState(MT19937).shuffle; local uint32 seed'


class WatershedDiagnosticError(ValueError):
    pass


class WatershedDiagnosticResourceError(WatershedDiagnosticError):
    pass


class WatershedDiagnosticUnsupported(NotImplementedError):
    pass


class WatershedDiagnosticCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WatershedDiagnostic:
    image: torch.Tensor
    metadata: object


def _colormap(name):
    if not isinstance(name, str) or not 1 <= len(name) <= 128:
        raise WatershedDiagnosticError('Select a registered Matplotlib colormap name.')
    try:
        cmap = matplotlib.colormaps[name]
    except KeyError as error:
        raise WatershedDiagnosticError('Unknown Matplotlib colormap: ' + name) from error
    if type(cmap) not in (LinearSegmentedColormap, ListedColormap):
        raise WatershedDiagnosticUnsupported('Custom colormap callables are not supported.')
    if type(cmap.N) is not int or not 1 <= cmap.N <= 65_536:
        raise WatershedDiagnosticResourceError('Colormap lookup table exceeds its compact bound.')
    # Initialize the private registry copy, including under/over/bad entries.
    cmap(np.asarray([0.0]), bytes=True)
    lut = np.asarray(cmap._lut)
    if lut.shape != (cmap.N + 3, 4) or not np.isfinite(lut).all():
        raise WatershedDiagnosticError('Colormap lookup table is invalid.')
    return cmap, sha256(lut.astype('<f8').tobytes()).hexdigest()


def palette_runtime_identity():
    """Bind registered LUTs as well as package versions for cache/replay identity."""
    names = tuple(sorted(matplotlib.colormaps))
    if len(names) > 1024:
        raise WatershedDiagnosticResourceError('Colormap registry exceeds its compact bound.')
    rows, colors = [], 0
    for name in names:
        cmap, digest = _colormap(name)
        colors += cmap.N + 3
        if colors > MAX_REGISTRY_COLORS:
            raise WatershedDiagnosticResourceError('Aggregate colormap LUT bound exceeded.')
        rows.append((name, cmap.N, type(cmap).__name__, digest))
    return {'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
            'registry': rows, 'rng_profile': RNG_PROFILE}


def watershed_palette(count, *, color_scale='gist_rainbow', random_seed=0):
    """Exact source byte sampling and shuffle for an explicitly seeded call."""
    if type(count) is not int or not 1 <= count <= MAX_LABELS + 1:
        raise WatershedDiagnosticResourceError('Palette length exceeds its label bound.')
    if type(random_seed) is not int or not 0 <= random_seed <= 0xffffffff:
        raise WatershedDiagnosticError('Palette seed must be a uint32 integer.')
    cmap, lut_digest = _colormap(color_scale)
    colors = cmap(np.linspace(0, 1, count), bytes=True)[:, :3].tolist()
    np.random.RandomState(random_seed).shuffle(colors)
    colors = tuple(tuple(int(channel) for channel in color) for color in colors)
    return colors, {'color_scale': color_scale, 'random_seed': random_seed,
        'rng_profile': RNG_PROFILE, 'numpy_version': np.__version__,
        'matplotlib_version': matplotlib.__version__, 'colormap_lut_sha256': lut_digest,
        'reference_numpy_version': '2.2.6', 'reference_matplotlib_version': '3.11.2',
        'reference_opencv_python_version': '4.11.0.86', 'reference_plantcv_version': '4.11.3',
        'palette_sha256': sha256(bytes(v for c in colors for v in c)).hexdigest(),
        'palette_source_triples': colors, 'palette_count': count,
        'palette_source_sha256': PALETTE_SOURCE_SHA256}


def _signature(value):
    storage = value.untyped_storage()
    return (value.device, value.dtype, value.layout, tuple(value.shape),
            tuple(value.stride()), value.data_ptr(), value.storage_offset(), value._version,
            storage.data_ptr(), storage.nbytes())


def _label_bounds(labels, guard):
    """One tiled CUDA reduction; only occupied label boxes leave the device."""
    height, width = map(int, labels.shape)
    minimum, maximum = int(labels.min().item()), int(labels.max().item())
    if minimum < 0 or maximum > MAX_LABEL_ID:
        raise WatershedDiagnosticError('Labels must lie in [0, 65536].')
    guard()
    shape = (maximum + 1,)
    lows_x = torch.full(shape, width, dtype=torch.int64, device=labels.device)
    lows_y = torch.full(shape, height, dtype=torch.int64, device=labels.device)
    highs_x = torch.full(shape, -1, dtype=torch.int64, device=labels.device)
    highs_y = torch.full(shape, -1, dtype=torch.int64, device=labels.device)
    flat = labels.view(-1)
    for start in range(0, flat.numel(), MAX_TILE_PIXELS):
        guard()
        stop = min(flat.numel(), start + MAX_TILE_PIXELS)
        ids = flat[start:stop].to(torch.int64)
        positions = torch.arange(start, stop, device=labels.device, dtype=torch.int64)
        xs, ys = positions % width, positions // width
        lows_x.scatter_reduce_(0, ids, xs, reduce='amin', include_self=True)
        lows_y.scatter_reduce_(0, ids, ys, reduce='amin', include_self=True)
        highs_x.scatter_reduce_(0, ids, xs, reduce='amax', include_self=True)
        highs_y.scatter_reduce_(0, ids, ys, reduce='amax', include_self=True)
        del ids, positions, xs, ys
    occupied = torch.nonzero(highs_x >= 0, as_tuple=False).flatten()
    if occupied.numel() > MAX_LABELS + 1 or int((occupied > 0).sum().item()) > MAX_LABELS:
        raise WatershedDiagnosticResourceError('Occupied label count exceeds 4096.')
    boxes = torch.stack((occupied, lows_x[occupied], lows_y[occupied],
                         highs_x[occupied] + 1, highs_y[occupied] + 1), dim=1).cpu().tolist()
    guard()
    return tuple(tuple(row) for row in boxes)


def render_watershed_diagnostic_rgb(image_rgb, labels, *, color_scale='gist_rainbow',
                                     random_seed=0, line_thickness=5, cancel=None):
    """Return a complete CUDA debug overlay and immutable compact provenance.

    Each label is traced in its CUDA bounding box, then translated back to the
    full frame. This preserves RETR_TREE/CHAIN_APPROX_NONE contour order while
    avoiding a full-frame scan for every seed. Dense reduction scratch is tiled
    at 1M pixels (<=64MiB plus <=2.1MiB label boxes); contour extraction keeps its
    existing 256MiB cap and painting its 96MiB cap. These phases do not overlap.
    Scan admission is 16P+32*sum(ROI pixels), at most8B estimated visits, separate
    from a shared 32M sparse-topology budget and 64M stroke-geometry budget
    (each individual forest also retains the source helper's 16M stroke bound).
    CPU drawing admission estimates192 bytes/expanded point (both point tuples)
    plus2048 bytes per retained batch/primitive row, plus compact forest/boxes.
    Adjacent CHAIN_APPROX_NONE edges are painted in batches of at most256 edges
    and8192 estimated rows, with a separate single-edge allowance.256MiB is the
    per-contour byte-model limit, not a whole-process RSS guarantee. Retained
    contours are processed one forest at a time; aggregate points/contours/spans
    and stroke pixels retain the shared limits. No partial image is published.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be callable or None')
    if type(line_thickness) is not int or not 1 <= line_thickness <= 32767:
        raise WatershedDiagnosticError('Line thickness must be an integer in [1,32767].')
    if (not all(torch.is_tensor(v) and v.layout == torch.strided and v.device.type == 'cuda'
                and v.is_contiguous() and not v.requires_grad for v in (image_rgb, labels))
            or image_rgb.dtype != torch.uint8 or image_rgb.ndim != 3 or image_rgb.shape[-1] != 3
            or labels.dtype != torch.int32 or labels.ndim != 2
            or tuple(image_rgb.shape[:2]) != tuple(labels.shape) or image_rgb.device != labels.device):
        raise WatershedDiagnosticError('Require contiguous same-device CUDA RGB uint8 and int32 labels.')
    height, width = map(int, labels.shape)
    if min(height, width) < 1 or height * width > MAX_PIXELS:
        raise WatershedDiagnosticResourceError('Diagnostic frame exceeds its 16M-pixel bound.')
    signatures = tuple(_signature(v) for v in (image_rgb, labels))
    canvas = None
    canvas_signature = None

    def guard():
        def check():
            if tuple(_signature(v) for v in (image_rgb, labels)) != signatures:
                raise WatershedDiagnosticError('Diagnostic input changed during calculation.')
            if canvas is not None and _signature(canvas) != canvas_signature:
                raise WatershedDiagnosticError('Private diagnostic canvas changed during callback.')
        check()
        try:
            stopped = False if cancel is None else cancel()
        finally:
            check()
        if type(stopped) is not bool:
            raise WatershedDiagnosticError('Cancellation callback must return exactly bool.')
        if stopped:
            raise WatershedDiagnosticCancelled('Watershed diagnostic cancelled.')
        return False

    def after_write():
        nonlocal canvas_signature
        canvas_signature = _signature(canvas)

    guard()
    with torch.no_grad():
        boxes = _label_bounds(labels, guard)
        palette, facts = watershed_palette(len(boxes), color_scale=color_scale, random_seed=random_seed)
        positive = tuple(box for box in boxes if box[0] > 0)
        if any(box[0] >= len(palette) for box in positive):
            raise WatershedDiagnosticUnsupported('Source palette[label_id] is undefined for these sparse IDs.')
        roi_pixels = sum((x1-x0)*(y1-y0) for _, x0, y0, x1, y1 in positive)
        scan_work = 16*height*width + 32*roi_pixels
        if scan_work > MAX_SCAN_WORK:
            raise WatershedDiagnosticResourceError('Aggregate diagnostic scan work exceeds its bound.')
        guard()
        canvas = image_rgb.clone()
        canvas_signature = _signature(canvas)
        point_count = contour_count = boundary_count = 0
        geometry_work, topology_work, paint_budget = 0, [0], [0, 0]
        max_cpu_geometry_bytes = 0
        try:
            for label, x0, y0, x1, y1 in positive:
                guard()
                submask = (labels[y0:y1, x0:x1] == label).to(torch.uint8).mul_(255)
                forest = contours.trace_binary_contours_cuda(submask, approximation='none',
                    cancel=guard, work_budget=topology_work)
                del submask
                forest_points = sum(len(contour) for contour in forest.contours)
                point_count += forest_points
                boundary_count += forest.boundary_point_count
                contour_count += len(forest.contours)
                if max(point_count, boundary_count) > MAX_POINTS or contour_count > MAX_CONTOURS:
                    raise WatershedDiagnosticResourceError('Aggregate diagnostic contour bound exceeded.')
                remaining = MAX_GEOMETRY_WORK - geometry_work
                offset = max(0, raster.MAX_RASTER_SPAN_WORK - remaining)
                local_work = [offset]
                for contour in forest.contours:
                    edge_rows = min(height, line_thickness+4)
                    batch_edges = min(256, max(1, 8192//edge_rows))
                    cpu_bytes = (16*forest_points + 160*len(boxes)
                        + 192*len(contour) + 2048*edge_rows*(batch_edges+1))
                    if cpu_bytes > MAX_CPU_GEOMETRY_BYTES:
                        raise WatershedDiagnosticResourceError('CPU drawing geometry byte-model bound exceeded.')
                    max_cpu_geometry_bytes = max(max_cpu_geometry_bytes, cpu_bytes)
                    points = tuple((int(x)+x0, int(y)+y0) for x, y in contour[:, 0, :])
                    for spans in raster.iter_adjacent_closed_polyline_spans(points, width, height,
                            thickness=line_thickness, cancel=guard, work_budget=local_work):
                        size._paint_rgb_spans_(canvas, spans, palette[label], paint_budget,
                            cancel=guard, after_write=after_write)
                        del spans
                    del points
                geometry_work += local_work[0] - offset
                del forest
            guard()
        except (contours.BinaryContourCancelled, raster.RasterSpanCancelled,
                size.PlantCVSizeCancelledError) as error:
            raise WatershedDiagnosticCancelled(str(error)) from error
        except (contours.BinaryContourResourceError, raster.RasterSpanLimitError) as error:
            raise WatershedDiagnosticResourceError(str(error)) from error
        except contours.BinaryContourUnsupportedError as error:
            raise WatershedDiagnosticUnsupported(str(error)) from error
        facts.update({'source_sha256': SOURCE_SHA256, 'line_thickness': line_thickness,
            'channel_boundary': 'native RGB; source BGR copy; literal palette triples are BGR draw colors',
            'kind': 'native_watershed_debug_diagnostic', 'public_return': 'labels only',
            'contour_mode': 'RETR_TREE/CHAIN_APPROX_NONE; closed LINE_8',
            'label_ids': tuple(box[0] for box in positive), 'scan_work': scan_work,
            'roi_pixels': roi_pixels, 'contour_points': point_count, 'contours': contour_count,
            'boundary_points': boundary_count, 'geometry_work': geometry_work,
            'topology_work': topology_work[0], 'max_cpu_geometry_estimate_bytes': max_cpu_geometry_bytes,
            'stroke_pixels': paint_budget[0], 'span_items': paint_budget[1]})
        guard()
        return WatershedDiagnostic(canvas, MappingProxyType(facts))
