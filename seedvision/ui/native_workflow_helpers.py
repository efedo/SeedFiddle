"""Shared honest parameter controls and sampled CUDA raster display."""
import json
import math
import numpy as np
import torch
from PySide6.QtGui import QImage
from seedvision.pipeline.model import ParameterSpec, ParameterSection

def configure_foreground_regions(graph, workflow):
    """Expose the working pipeline's region adapter on its owning mask node."""
    from seedvision.pipeline.workflows import FOREGROUND_OPERATIONS
    key = 'foreground_roi_bounds_xyxy'
    for stage in workflow.recipe.stages:
        if stage.operation_id not in FOREGROUND_OPERATIONS:
            continue
        node = graph.node(stage.stage_id)
        node.parameters[key] = json.dumps(workflow.foreground_regions.get(stage.stage_id))
        node.default_parameters[key] = 'null'
        if not any(spec.key == key for spec in node.parameter_specs):
            node.parameter_specs += (ParameterSpec(key, 'Foreground region (input pixels)', 'structured',
                description='null keeps the whole image. [left, top, right, bottom] excludes foreground outside '
                    'this rectangle in this node input frame while preserving image dimensions. '
                    'Select region draws it when the node uses source coordinates. '
                    'This explicit native mask adapter also follows PlantCV threshold calls.'),)
            node.parameter_sections += (ParameterSection('Foreground selection', (key,)),)

def configure_native_parameters(graph, recipe, *, typed=False, editable_only=False):
    graph.reference_recipe_identity = recipe.identity
    descriptions = {
        'external.image_inputs': 'Decode the selected image and prepare full-resolution RGB and grayscale pixels.',
        'filter.median_blur': 'Filter local grayscale noise before foreground selection.',
        'threshold.binary': 'Select foreground using the cutoff and light/dark polarity.',
        'threshold.mean': 'Select foreground using local mean intensity, neighborhood size and offset. Even sizes round up to odd.',
        'threshold.otsu': 'Select foreground using the supported Otsu partition and light/dark polarity.',
        'threshold.gaussian': 'Select foreground with the supported Gaussian kernel, offset and polarity. Even sizes round up to odd.',
        'threshold.triangle': 'Select foreground with the source histogram preamble and supported Otsu decision.',
        'threshold.texture': 'Select foreground from the GLCM property, kernel, offset, cutoff and border mode.',
        'threshold.dual_channels': 'Select foreground using the chosen channels and straight-line points.',
        'threshold.custom_range': 'Apply inclusive channel bounds and retain selected original RGB pixels. RGB bounds use B,G,R storage order; CMYK preserves the pinned C/Y swap.',
        'morphology.fill': 'Remove foreground components smaller than the selected pixel area.',
        'morphology.erode': 'Shrink the foreground with the selected square kernel and number of passes.',
        'morphology.dilate': 'Expand the foreground with the selected square kernel and number of passes.',
        'reference.internal.watershed.distance_map': 'Measure distance inside the prepared foreground.',
        'reference.internal.watershed.local_maxima': 'Select local distance peaks with the chosen minimum spacing.',
        'reference.internal.watershed.connected_markers': 'Group adjacent peak pixels into eight-connected markers.',
        'reference.internal.watershed.negate_distance': 'Negate the distance map to form the watershed elevation.',
        'reference.internal.watershed.masked_flood': 'Partition the original foreground using the markers and distance elevation.',
        'reference.internal.watershed.marker_count_observation': 'Report the reference object-count observation derived from connected markers.',
        'native.watershed_debug_diagnostic': 'Draw a separate watershed debug image with the source contours, colormap and an explicit reproducible shuffle seed. Palette and stroke edits reuse segmentation.',
        'analysis.color': 'Measure per-object channel distributions, circular hue and positive saturation/value summaries.',
        'analysis.size': 'Measure supported objects in pixels and draw their size diagnostic, including disconnected regions, holes and nested islands. Unsupported numerical geometry is unavailable; diagnostic resource limits preserve available measurements.',
    }
    for stage in recipe.stages:
        node = graph.nodes[stage.stage_id]
        if typed:
            node.category = 'SMPTS' if recipe.recipe_id.startswith('smpts_') else 'NATIVE' if recipe.recipe_id.startswith('seedfiddle_') else 'PlantCV'
        node.description = descriptions.get(stage.operation_id, stage.description)
        specs = []
        values = ({p.parameter_id: p.value for p in stage.parameters} if typed else {})
        for parameter in stage.parameters:
            if not parameter.affects_calculation or (typed and parameter.parameter_id in ('n_labels', 'model_source', 'label')):
                continue
            selected_value = parameter.value
            numeric = parameter.value_type in ('integer', 'number')
            if typed and parameter.value_type == 'json' and parameter.parameter_id in (
                    'ksize', 'threshold', 'size', 'iterations', 'distance', 'offset', 'min_distance'):
                defaults = {'ksize': 5 if stage.operation_id == 'filter.median_blur' else 3,
                    'threshold': 128, 'size': 50, 'iterations': 1, 'distance': 10,
                    'offset': 5, 'min_distance': 10}
                selected_value = defaults[parameter.parameter_id] if selected_value is None else selected_value
                numeric = type(selected_value) in (int, float)
            if parameter.choices:
                kind, value = 'choice', selected_value
            elif typed and (numeric or parameter.value_type == 'boolean'):
                kind = 'bool' if parameter.value_type == 'boolean' else 'int' if type(selected_value) is int else 'float'
                value = selected_value
            else:
                kind = 'text'
                value = (str(parameter.value) if parameter.value_type == 'string'
                    else json.dumps(recipe.parameters_for_runner(stage.stage_id)[parameter.parameter_id]))
                if typed and parameter.parameter_id in ('lower', 'upper', 'points', 'roi_bounds_xyxy', 'seed_roi_xyxy', 'validity_roi', 'color_chip_size', 'cat_list'):
                    kind = 'structured'
            values[parameter.parameter_id] = value
            description = parameter.description
            if parameter.parameter_id == 'ksize' and stage.operation_id in ('threshold.mean', 'threshold.gaussian'):
                maximum = 255 if stage.operation_id == 'threshold.mean' else 15
                description = f'Integer requested kernel from 3 to {maximum}; even sizes round up to odd. Uncertain rounding is unavailable.'
            specs.append(ParameterSpec(parameter.parameter_id, parameter.parameter_id.replace('_', ' '),
                kind, minimum=(parameter.minimum if parameter.minimum is not None else -2147483647) if kind in ('int', 'float') else None,
                maximum=(parameter.maximum if parameter.maximum is not None else 2147483647) if kind in ('int', 'float') else None,
                step=1 if kind == 'int' else 0.01, description=description, choices=tuple(parameter.choices)))
        node.parameters = values
        node.parameter_specs = tuple(specs)
        node.default_parameters = dict(values)
        node.parameter_sections = (ParameterSection('Selected calculation parameters', tuple(spec.key for spec in specs)),) if specs else ()
        node.bypassable = node.bypassable if typed else False
        node.implemented = True



def native_preview_image(raster, semantics, *, maximum=1024):
    stride = max(1, math.ceil(raster.shape[0]/maximum), math.ceil(raster.shape[1]/maximum))
    sampled = raster[::stride, ::stride]
    if semantics.get('color_order') == 'BGR':
        sampled = sampled.flip(-1)
    if semantics.get('positive_values') == 'chip IDs10..240':
        ids = sampled.to(torch.int64)
        sampled = torch.stack(((ids*53)%255, (ids*97)%255, (ids*193)%255), -1).to(torch.uint8)
    elif sampled.dtype in (torch.float32, torch.float64):
        if semantics.get('unit') == 'pixel' and 'distanceTransform' in semantics.get('source', ''):
            # Display-only compression makes small foreground distances visible
            # alongside large objects. Zero/background remains exactly black.
            values = sampled.clamp_min(0).log1p()
            level = values / values.max().clamp_min(1e-12)
            rgb = torch.stack(((3*level-1).clamp(0, 1),
                (1.5-3*(level-.5).abs()).clamp(0, 1), (1-2*level).clamp(0, 1)), -1)
            sampled = (rgb * (values > 0).unsqueeze(-1) * 255).to(torch.uint8)
        else:
            lower, upper = sampled.min(), sampled.max()
            sampled = ((sampled-lower)/(upper-lower).clamp_min(1e-12)*255).to(torch.uint8)
    elif sampled.dtype == torch.int32 or semantics.get('range') == '0..cluster_count-1':
        ids = sampled.to(torch.int64)
        # Distinct colours for sparse IDs throughout the recovery graph's
        # supported range; zero remains black (joint period16,193,265).
        sampled = torch.stack(((ids*53)%251, (ids*97)%253, (ids*193)%255), -1).to(torch.uint8)
    elif sampled.dtype == torch.bool:
        sampled = sampled.to(torch.uint8)*255
    elif sampled.dtype == torch.uint8 and semantics.get('positive_value') == 1:
        sampled = sampled*255
    array = np.ascontiguousarray(sampled.cpu().numpy())
    format = QImage.Format.Format_RGB888 if array.ndim == 3 else QImage.Format.Format_Grayscale8
    image = QImage(array.data, array.shape[1], array.shape[0], array.strides[0], format).copy()
    return image
