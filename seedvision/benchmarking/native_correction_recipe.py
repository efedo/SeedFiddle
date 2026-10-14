"""Execute the canonical correction stage on four explicit CUDA inputs.

Images and labelled masks retain their external source/frame identities. Only
compact fit matrices leave CUDA; this executor does not decode or detect cards.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import re
import sys
import time
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.native_recipe import (
    MAX_RETAINED_CUDA_BYTES, NativeRecipeNode, _CacheEntry, _digest,
    _module_record, _output_signatures, _runtime_records, _tensor_signature,
)
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeIdentity, RecipeSpec, RecipeProvenanceMode
from seedvision.cuda import plantcv_correction as _correction
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.persistence.correction_inputs import CorrectionRunOrigins


RECIPE_ID = "plantcv_color_card_correct_color"
STAGE_ID = "pcv_color_correct_color"
MAX_COMBINED_PIXELS = 64_000_000
_INPUTS = ("target_rgb", "target_mask", "source_rgb", "source_mask")


class NativeCorrectionError(ValueError):
    pass


class NativeCorrectionCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class NativeCorrectionResult:
    recipe_identity: RecipeIdentity
    target_source_sha256: str
    source_sha256: str
    target_coordinate_frame: CoordinateFrame
    coordinate_frame: CoordinateFrame
    preparation: Mapping[str, str]
    runtime: Mapping[str, str]
    implementation_sha256: str
    nodes: tuple[NativeRecipeNode, ...]
    status: str
    input_sources: Mapping[str, Mapping] | None = None
    input_origins: CorrectionRunOrigins | None = None

    @property
    def model_asset_paths(self):
        return None

    def node(self, stage_id):
        return next(node for node in self.nodes if node.stage_id == stage_id)


def _cancel(cancel):
    if cancel is not None and cancel():
        raise NativeCorrectionCancelled("Correction cancelled before a complete result.")


def build_native_correction_recipe(recipe=None):
    """Retain the four explicit ports with truthful native execution provenance."""
    selected = build_recipe_catalogue().recipe(RECIPE_ID) if recipe is None else recipe
    if not isinstance(selected, RecipeSpec) or selected.recipe_id != RECIPE_ID:
        raise NativeCorrectionError('Select the canonical colour-card correction recipe.')
    if selected.provenance.mode is RecipeProvenanceMode.NATIVE_COMPATIBLE:
        _validate_recipe(selected)
        return selected
    _validate_recipe(selected)
    provenance = replace(selected.provenance, mode=RecipeProvenanceMode.NATIVE_COMPATIBLE,
        backend_id='seedvision.benchmarking.native_correction_recipe', backend_version='1.0.0',
        evidence_note='Native four-input colour correction; supported computational domains '
            'and independent colour accuracy are evaluated separately.')
    return replace(selected, provenance=provenance)


def _validate_recipe(recipe):
    if not isinstance(recipe, RecipeSpec) or recipe.recipe_id != RECIPE_ID:
        raise NativeCorrectionError("Select the canonical colour-card correction recipe.")
    actual = recipe.to_dict()
    canonical = build_recipe_catalogue().recipe(RECIPE_ID)
    if recipe.provenance.mode is RecipeProvenanceMode.NATIVE_COMPATIBLE:
        canonical = build_native_correction_recipe(canonical)
    expected = canonical.to_dict()
    for document in (actual, expected):
        for name in ("layout", "title", "description"):
            document.pop(name)
        for stage in document["stages"]:
            stage.pop("title")
            stage.pop("description")
            for port in stage["ports"]:
                port.pop("label")
            for parameter in stage["parameters"]:
                parameter.pop("value")
    if actual != expected:
        raise NativeCorrectionError("Correction recipe source, topology or typed ports differ from the catalogue.")


def _validate_tensor(name, value, *, image, frame, device):
    expected = (*frame.shape, 3) if image else frame.shape
    if (not torch.is_tensor(value) or value.layout != torch.strided
            or value.device.type != "cuda" or value.device != device
            or value.dtype != torch.uint8 or tuple(value.shape) != expected
            or not value.is_contiguous()):
        raise NativeCorrectionError(f"{name} requires contiguous CUDA uint8 data in its exact frame.")


def _immutable_array(value):
    source = np.ascontiguousarray(value)
    return np.frombuffer(source.tobytes(), dtype=source.dtype).reshape(source.shape)


def _correction_output_signatures(outputs):
    # Readonly bytes do not prevent changes to NumPy shape/dtype metadata.
    # Hash only the compact contiguous tables; corrupted/oversized metadata
    # produces a different signature without reading an unbounded view.
    matrices = tuple((name, tuple(value.shape), value.dtype.str, value.strides,
        value.flags.writeable, value.__array_interface__['data'][0],
        sha256(value).hexdigest() if value.flags.c_contiguous and value.nbytes <= 255*4*8 else None)
        for name, value in outputs.items() if isinstance(value, np.ndarray))
    return _output_signatures(outputs), matrices


def _fit_cache_budget(cache, prefix, target_pixels, source_pixels):
    # BGR input copies, simultaneous BGR and RGB corrected outputs, plus
    # conservative scratch for the bounded float64 pixel tiles.
    requested = 3*target_pixels+9*source_pixels+64*1024*1024
    if requested > MAX_RETAINED_CUDA_BYTES:
        raise NativeCorrectionError("Correction exceeds the retained CUDA workspace budget.")
    storage, other_names = {}, []
    for name, entry in cache.values.items():
        if name.startswith("reference.recipe:") and not name.startswith(prefix):
            other_names.append(name)
            if isinstance(entry, _CacheEntry):
                for value in entry.node.outputs.values():
                    if torch.is_tensor(value) and value.device.type == "cuda":
                        allocation = value.untyped_storage()
                        storage[(str(value.device), allocation.data_ptr())] = allocation.nbytes()
    if sum(storage.values())+requested > MAX_RETAINED_CUDA_BYTES:
        for name in other_names:
            cache.values.pop(name, None)


def _result(recipe, target_sha, source_sha, target_frame, source_frame,
        runtime, implementation, node):
    preparation = {
        "target_rgb": "caller-supplied CUDA uint8 RGB; converted to BGR on CUDA",
        "source_rgb": "caller-supplied CUDA uint8 RGB; converted to BGR on CUDA",
        "target_mask": "caller-supplied CUDA uint8 chip labels in target frame",
        "source_mask": "caller-supplied CUDA uint8 chip labels in source frame",
    }
    return NativeCorrectionResult(recipe.identity, target_sha, source_sha,
        target_frame, source_frame, MappingProxyType(preparation),
        MappingProxyType(dict(runtime)), implementation, (node,), "executed")


def run_native_correction_recipe(recipe, target_rgb, target_mask, source_rgb,
        source_mask, *, target_source_sha256, source_sha256,
        target_coordinate_frame, coordinate_frame, cache=None, cancel=None):
    """Return every declared correction output with explicit RGB boundaries.

    Source and target may have different widths, but pinned correct_color's
    diagnostic montage requires equal heights even when debugging is disabled.
    Failure/cancellation returns no result and publishes no stale cache entry.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError("cancel must be callable or None")
    _cancel(cancel)
    _validate_recipe(recipe)
    if any(not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value)
            for value in (target_source_sha256, source_sha256)):
        raise NativeCorrectionError("Both external images require lowercase source SHA-256 identities.")
    if not all(isinstance(frame, CoordinateFrame)
            for frame in (target_coordinate_frame, coordinate_frame)):
        raise NativeCorrectionError("Both external images require explicit coordinate frames.")
    device = target_rgb.device if torch.is_tensor(target_rgb) else None
    if device is None or device.type != "cuda":
        raise NativeCorrectionError("Correction inputs must reside on one CUDA device.")
    tensors = dict(zip(_INPUTS, (target_rgb, target_mask, source_rgb, source_mask)))
    for name, value in tensors.items():
        frame = target_coordinate_frame if name.startswith("target_") else coordinate_frame
        _validate_tensor(name, value, image=name.endswith("_rgb"), frame=frame, device=device)
    target_pixels, source_pixels = target_mask.numel(), source_mask.numel()
    if target_pixels+source_pixels > MAX_COMBINED_PIXELS:
        raise NativeCorrectionError("Correction images exceed the combined pixel budget.")
    if target_rgb.shape[0] != source_rgb.shape[0]:
        raise NativeCorrectionError("Pinned correct_color requires equal image heights for its diagnostic montage.")
    if cache is None:
        cache = PipelineAnalysisCache()
    if not isinstance(cache, PipelineAnalysisCache):
        raise TypeError("cache must be an existing PipelineAnalysisCache")
    stage = recipe.stages[0]
    if recipe.parameters_for_runner(stage.stage_id):
        raise NativeCorrectionError("The canonical correction stage has no editable calculation parameters.")
    input_keys = {}
    for name, value in tensors.items():
        target = name.startswith("target_")
        input_keys[name] = _digest({
            "port": name,
            "source": target_source_sha256 if target else source_sha256,
            "frame": (target_coordinate_frame if target else coordinate_frame).to_dict(),
            "tensor": _tensor_signature(value),
        })
    records, runtime = _runtime_records(device)
    records[_correction.__name__] = _digest(_module_record(_correction))
    records[__name__] = _digest(_module_record(sys.modules[__name__]))
    implementation = _digest({"runtime": runtime, "modules": records})
    key = _digest({"recipe": recipe.identity.analytical_dict(),
        "inputs": input_keys, "implementation": implementation})
    prefix = f"reference.recipe:{recipe.recipe_id}:"
    name = prefix+stage.stage_id
    entry = cache.values.get(name)
    if (isinstance(entry, _CacheEntry) and entry.key == key
            and entry.signatures == _correction_output_signatures(entry.node.outputs)):
        _cancel(cancel)
        previous = entry.node
        node = NativeRecipeNode(previous.stage_id, previous.operation_id, previous.status,
            previous.outputs, previous.parameters, previous.input_keys, previous.product_key,
            reused=True, reason=previous.reason)
        cache.last_computed_nodes, cache.last_reused_nodes = (), (stage.stage_id,)
        cache.node_timings_seconds = {stage.stage_id: 0.0}
        return _result(recipe, target_source_sha256, source_sha256,
            target_coordinate_frame, coordinate_frame, runtime, implementation, node)
    cache.values.pop(name, None)
    cache.last_computed_nodes, cache.last_reused_nodes = (), ()
    cache.node_timings_seconds = {}
    _fit_cache_budget(cache, prefix, target_pixels, source_pixels)
    started = time.perf_counter()
    try:
        _cancel(cancel)
        target_bgr = target_rgb.flip(-1).contiguous()
        _cancel(cancel)
        source_bgr = source_rgb.flip(-1).contiguous()
        correction = _correction.correct_color_cuda(target_bgr, target_mask,
            source_bgr, source_mask, cancel=cancel)
        outputs = {
            "corrected_image": correction.corrected_bgr.flip(-1).contiguous(),
            "target_matrix": _immutable_array(correction.target_matrix),
            "source_matrix": _immutable_array(correction.source_matrix),
            "transformation_matrix": _immutable_array(correction.transformation_matrix),
        }
        if (set(outputs) != {port.name for port in stage.ports if port.direction == "output"}
                or outputs["transformation_matrix"].shape != (9, 9)):
            raise NativeCorrectionError("Correction did not return all canonical outputs.")
        _cancel(cancel)
        generation_name = prefix+"generation"
        generation = int(cache.values.get(generation_name, 0))+1
        node = NativeRecipeNode(stage.stage_id, stage.operation_id, "executed",
            MappingProxyType(outputs), MappingProxyType({}), MappingProxyType(input_keys),
            _digest([key, generation]))
        cache.values[generation_name] = generation
        cache.values[name] = _CacheEntry(key, node, _correction_output_signatures(outputs))
        cache.last_computed_nodes, cache.last_reused_nodes = (stage.stage_id,), ()
        cache.node_timings_seconds = {stage.stage_id: time.perf_counter()-started}
        return _result(recipe, target_source_sha256, source_sha256,
            target_coordinate_frame, coordinate_frame, runtime, implementation, node)
    except _correction.CorrectionCancelled as error:
        raise NativeCorrectionCancelled(str(error)) from error
