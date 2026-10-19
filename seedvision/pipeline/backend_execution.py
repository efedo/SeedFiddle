"""Actual per-node native/public execution for the desktop workflow evaluator.

Only explicitly equivalent public operations are selectable. Public CPU inputs
cross the audited subprocess boundary deliberately; returned raster products are
uploaded to the caller's device once and remain there for downstream nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
import json
import sys

import numpy as np
import torch

from seedvision.benchmarking.checkpoints import _freeze_json
from seedvision.benchmarking import reference_runner as reference


BACKENDS = ("native", "plantcv")
PUBLIC_OPERATIONS = frozenset({
    "threshold.binary", "threshold.mean", "threshold.otsu", "threshold.gaussian",
    "threshold.triangle", "threshold.texture", "threshold.dual_channels",
    "threshold.custom_range", "filter.median_blur", "morphology.fill",
    "morphology.erode", "morphology.dilate", "analysis.size", "analysis.color",
    "transform.correct_color",
    "analysis.naive_bayes_classifier",
    "color_card.means", "color_card.standard", "color_card.affine",
})
WATERSHED_OPERATIONS = tuple("reference.internal.watershed." + name for name in (
    "distance_map", "local_maxima", "connected_markers", "negate_distance",
    "masked_flood", "marker_count_observation"))


class BackendUnavailable(NotImplementedError):
    """The selected public operation has no equivalent admitted execution."""


@dataclass(frozen=True)
class BackendCapability:
    backends: tuple[str, ...]
    reason: str = ""
    atomic_group: tuple[str, ...] = ()


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _watershed_group(recipe):
    selected = {stage.operation_id: stage for stage in recipe.stages
                if stage.operation_id in WATERSHED_OPERATIONS}
    if len(selected) != len(WATERSHED_OPERATIONS):
        return (), "Public watershed requires the complete six-stage atomic group."
    group = tuple(selected[op].stage_id for op in WATERSHED_OPERATIONS)
    if sum(stage.operation_id in WATERSHED_OPERATIONS for stage in recipe.stages) != 6:
        return (), "Public watershed requires one unambiguous atomic group."
    distance, peaks, markers, negative, flood, count = group
    incoming = {(edge.target_stage, edge.target_port): (edge.source_stage, edge.source_port)
                for edge in recipe.dependencies}
    required = {(peaks, "distance"): (distance, "distance"),
        (markers, "peaks"): (peaks, "peaks"), (negative, "distance"): (distance, "distance"),
        (flood, "negative_distance"): (negative, "negative_distance"),
        (flood, "markers"): (markers, "markers"), (count, "markers"): (markers, "markers"),
        (count, "flood_complete"): (flood, "flood_complete")}
    masks = [incoming.get((node, "mask")) for node in (distance, peaks, flood)]
    if (any(incoming.get(key) != value for key, value in required.items())
            or masks[0] is None or len(set(masks)) != 1
            or incoming.get((flood, "rgb"), (None,))[0] in group
            or (flood, "rgb") not in incoming):
        return group, "Edited watershed connections cannot be executed as one equivalent public call."
    return group, ""


def backend_capability(recipe, stage_id):
    """Return choices for the actual operation, controls and connected ports."""
    stage = recipe.stage(stage_id)
    parameters = recipe.parameters_for_runner(stage_id)
    group = ()
    if stage.operation_id in WATERSHED_OPERATIONS:
        group, reason = _watershed_group(recipe)
        return BackendCapability(("native",) if reason else BACKENDS, reason, group)
    if stage.operation_id not in PUBLIC_OPERATIONS:
        reasons = {
            "external.image_inputs": "Image preparation is the explicit native CUDA input boundary.",
            "analysis.kmeans_predict": "This stage predicts a saved frozen model; public kmeans_classifier fits a new model.",
            "analysis.kmeans_masks": "Masks consume the saved predictor's categories, not a public model-fitting call.",
        }
        reason = reasons.get(stage.operation_id, '')
        if not reason:
            if stage.operation_id in ('color_card.preprocess', 'color_card.geometry', 'color_card.mask', 'color_card.scale'):
                reason = ('These detector intermediates include private geometry/ROI/scale ports that the public '
                    'detect_color_card return does not provide; their separate native controls remain active.')
            elif stage.operation_id == 'color_card.bgr':
                reason = 'This is the explicit native RGB-to-BGR tensor adapter, not a public algorithm call.'
            elif stage.operation_id.startswith('calibrated_seed.'):
                reason = 'This contextual image/frame/calibration adapter is native; downstream equivalent public nodes remain selectable.'
            elif stage.operation_id.startswith(('recovery.', 'smpts_paper.')):
                reason = 'This separate reconstruction/recovery algorithm has no equivalent PlantCV public callable.'
            elif stage.operation_id.startswith('external.'):
                reason = 'This native input/model transport validates saved source identities before calculations.'
            else:
                reason = 'This native stage has no equivalent public callable with the same declared inputs and outputs.'
        return BackendCapability(("native",), reason)
    reason = ""
    if parameters.get("roi_bounds_xyxy") is not None:
        reason = "The public operation processes a complete image; this node's local ROI is native-only."
    if (stage.operation_id == "filter.median_blur" and parameters.get("ksize") is not None
            and type(parameters.get("ksize")) is not int):
        reason = "This public median adapter accepts an odd scalar kernel, not a tuple kernel."
    if stage.operation_id in ("analysis.size", "analysis.color") and not isinstance(parameters.get("label"), str):
        reason = "The audited public adapter requires one explicit observation sample label."
    return BackendCapability(("native",) if reason else BACKENDS, reason)


def resolve_backend(workflow, stage_id):
    capability = backend_capability(workflow.recipe, stage_id)
    override = dict(getattr(workflow, "node_backends", None) or {}).get(stage_id)
    selected = override or getattr(workflow, "backend_default", "native")
    if selected not in BACKENDS:
        raise ValueError("Backend must be native or plantcv.")
    if selected not in capability.backends:
        if override is not None:
            raise BackendUnavailable(capability.reason)
        return "native"
    return selected


def validate_backend_selection(workflow):
    """Reject mixed public watershed internals and unsupported explicit choices."""
    if getattr(workflow, "backend_default", "native") not in BACKENDS:
        raise ValueError("Backend default must be native or plantcv.")
    overrides = getattr(workflow, "node_backends", None) or {}
    if not isinstance(overrides, Mapping) or set(overrides) - {s.stage_id for s in workflow.recipe.stages}:
        raise ValueError("Backend overrides must name registered workflow stages.")
    for stage in workflow.recipe.stages:
        resolve_backend(workflow, stage.stage_id)
    group, reason = _watershed_group(workflow.recipe)
    if group and not reason:
        choices = {resolve_backend(workflow, stage_id) for stage_id in group}
        if len(choices) != 1:
            raise BackendUnavailable("Public watershed is one atomic call; choose the same backend for all six stages.")
        if choices == {"plantcv"} and set(group) & set(workflow.disabled_nodes):
            raise BackendUnavailable("A public watershed atomic group cannot contain a bypassed internal stage.")


def validate_backend_provenance(value):
    if not isinstance(value, Mapping) or value.get("backend") != "plantcv":
        raise ValueError("Public execution requires PlantCV backend provenance.")
    if (value.get("schema") != "seedfiddle.pipeline-public-execution.v1"
            or not isinstance(value.get("operation"), str)
            or not isinstance(value.get("identity"), Mapping)
            or not isinstance(value.get("effective_parameters"), Mapping)
            or value.get("transport") != "CUDA/CPU input -> audited CPU subprocess -> CUDA/CPU output"):
        raise ValueError("Invalid public execution provenance.")
    expected = {"reference": "plantcv", "version": reference.PINNED_PLANTCV_VERSION,
        "commit": reference.PINNED_PLANTCV_COMMIT, "python": reference.PINNED_REFERENCE_PYTHON,
        "platform": reference.PINNED_REFERENCE_PLATFORM,
        "source_audit_sha256": reference.PINNED_SOURCE_AUDIT_SHA256,
        "requirements_lock_sha256": reference.PINNED_REQUIREMENTS_SHA256}
    identity = value['identity']
    if any(identity.get(name) != selected for name, selected in expected.items()):
        raise ValueError('Public backend provenance differs from its pinned runtime and source.')
    digest = identity.get('identity_digest')
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in '0123456789abcdef' for char in digest)):
        raise ValueError('Public backend provenance requires its verified identity digest.')
    if not isinstance(value['effective_parameters'].get('callable'), str):
        raise ValueError('Public backend provenance requires its actually executed callable.')
    return _freeze_json(dict(value), name="public backend provenance")


def validate_public_otsu_calculation(value, object_type):
    from seedvision.benchmarking.reference_stage import validate_otsu_cutoff_adapter
    validate_otsu_cutoff_adapter(value, object_type)
    return _freeze_json(dict(value), name="public Otsu cutoff adapter")


def _poll(cancel):
    if cancel is not None:
        stopped = cancel()
        if type(stopped) is not bool:
            raise ValueError("Backend cancellation callback must return bool.")
        if stopped:
            from seedvision.benchmarking.native_recipe import NativeRecipeCancelled
            raise NativeRecipeCancelled("Public workflow execution cancelled.")


def _cpu_array(value):
    if isinstance(value, np.ndarray):
        if value.nbytes > 1024 * 1024:
            raise ValueError("Compact public input exceeds its metadata byte limit.")
        return np.array(value, copy=True, order="C")
    if not torch.is_tensor(value) or value.layout != torch.strided:
        raise ValueError("Public raster ports require strided PyTorch tensors.")
    # A copy isolates the worker's input boundary even for CPU-only adapter tests.
    return np.array(value.detach().cpu().contiguous().numpy(), copy=True, order="C")


def _upload(value, device):
    return torch.from_numpy(np.array(value, copy=True, order="C")).to(device=device).contiguous()


def _observations(result):
    rows = result.observations.get("observations", {})
    if not isinstance(rows, Mapping):
        raise ValueError("The public call omitted its observation mapping.")
    return _freeze_json(dict(rows), name="public observations")


def _color_chart(observations):
    from seedvision.benchmarking.color_histogram import HueHistogramChart, HueHistogramSeries
    series = []
    for sample, rows in observations.items():
        histogram = rows.get("hue_frequencies")
        if histogram is not None:
            bins, values = tuple(histogram["label"]), tuple(histogram["value"])
            if bins != tuple(range(1, 360, 2)) or len(values) != 180:
                raise ValueError("The actual public hue chart must contain its pinned 180 bins.")
            series.append(HueHistogramSeries(int(sample.rsplit("_", 1)[-1]), sample, bins, values))
    return HueHistogramChart("available" if series else "empty", tuple(series))


class BackendExecution:
    """One execution's backend plan and audited, reusable public-call products."""
    def __init__(self, recipe, *, workflow=None, runner=None, cancel=None, cache=None,
                 prefix="", watch=None):
        self.recipe, self.workflow, self.cancel = recipe, workflow, cancel
        self.runner, self.cache, self.prefix, self.watch = runner, cache, prefix, watch
        self._identity = None
        self._group, self._group_reason = _watershed_group(recipe)
        if workflow is not None:
            validate_backend_selection(workflow)

    def backend_for(self, stage_id):
        return resolve_backend(self.workflow, stage_id) if self.workflow is not None else "native"

    def validate_cached_groups(self):
        """A changed captured local invalidates its whole public call's products."""
        if not self._group or self.cache is None:
            return
        name = self.prefix + "public-watershed:" + self._group[0]
        if self.backend_for(self._group[0]) != "plantcv":
            self.cache.values.pop(name, None)
            return
        from seedvision.benchmarking.native_recipe import _CacheEntry, _output_signatures, _evict_recipe_descendants, _execution_signature
        entry = self.cache.values.get(name)
        if (isinstance(entry, _CacheEntry)
                and entry.signatures == _output_signatures(entry.node.outputs)
                and entry.execution_signature == _execution_signature(entry.node)):
            return
        self.cache.values.pop(name, None)
        for stage_id in self._group:
            self.cache.values.pop(self.prefix + stage_id, None)
            _evict_recipe_descendants(self.recipe, stage_id, self.cache, self.prefix)

    def identity(self):
        if self._identity is None:
            _poll(self.cancel)
            if self.runner is None:
                root = Path(__file__).resolve().parents[2]
                self.runner = reference.PlantCVReferenceRunner(
                    root / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe")
            if isinstance(self.runner, reference.PlantCVReferenceRunner):
                status = reference.plantcv_reference_status(
                    python_executable=self.runner.python_executable,
                    source_manifest_path=self.runner.source_manifest_path, source_root=self.runner.source_root,
                    environment_manifest_path=self.runner.environment_manifest_path,
                    requirements_lock_path=self.runner.requirements_lock_path)
                if not status.get("available"):
                    raise BackendUnavailable(status.get("reason", "The pinned public runtime is unavailable."))
                self._identity = {"identity_digest": status["identity_digest"],
                    "python_executable": str(self.runner.python_executable),
                    "executable_sha256": sha256(self.runner.python_executable.read_bytes()).hexdigest()}
            else:
                # Explicit runner injection is for callers/tests supplying the
                # same checked ReferenceResult boundary, never an automatic fallback.
                identity = getattr(self.runner, "backend_identity", None)
                if not isinstance(identity, Mapping):
                    raise ValueError("Injected public runners require an explicit backend_identity.")
                self._identity = dict(identity)
            self._identity['adapter'] = self._adapter_identity()
            _poll(self.cancel)
        return self._identity

    def _adapter_identity(self):
        from seedvision.benchmarking.native_recipe import _module_record
        worker = Path(reference.__file__).with_name('plantcv_worker.py')
        return {'loaded_adapter': _digest(_module_record(sys.modules[__name__])),
            'loaded_runner': _digest(_module_record(reference)),
            'worker_source': sha256(worker.read_bytes()).hexdigest(),
            'public_operations': sorted(PUBLIC_OPERATIONS), 'watershed_operations': WATERSHED_OPERATIONS}

    def audit(self):
        if self.cache is not None:
            from seedvision.benchmarking.native_recipe import _CacheEntry, _execution_signature
            for name, entry in self.cache.values.items():
                if (name.startswith(self.prefix) and isinstance(entry, _CacheEntry)
                        and entry.execution_signature != _execution_signature(entry.node)):
                    raise ValueError('Cached backend or public-call provenance changed during execution.')
        if self._identity is not None and self._identity.get('adapter') != self._adapter_identity():
            raise ValueError('Public backend implementation changed during workflow execution.')

    def _group_inputs(self, nodes):
        flood = self._group[4]
        result, keys = {}, {}
        for edge in self.recipe.dependencies:
            if edge.target_stage == flood and edge.target_port in ("rgb", "mask"):
                node = nodes.get(edge.source_stage)
                if node is None or node.product_key is None or edge.source_port not in node.outputs:
                    raise BackendUnavailable("Public watershed requires its actual upstream RGB and mask products.")
                result[edge.target_port] = node.outputs[edge.source_port]
                keys[edge.target_port] = [node.product_key, edge.source_port]
        return result, keys

    def key_for(self, stage_id, nodes):
        backend = self.backend_for(stage_id)
        key = {"backend": backend}
        if backend == "plantcv":
            key["executor"] = self.identity()
            if stage_id in self._group:
                _, keys = self._group_inputs(nodes)
                key["atomic_group"] = {"stages": self._group, "inputs": keys,
                    "distance": self.recipe.parameters_for_runner(self._group[1])["distance"],
                    "label": self.recipe.parameters_for_runner(self._group[5])["label"]}
        return key

    def _call(self, operation, inputs, parameters):
        self.identity()
        _poll(self.cancel)
        total_bytes = 0
        for value in inputs.values():
            if torch.is_tensor(value):
                size = value.numel() * value.element_size()
            elif isinstance(value, np.ndarray):
                size = value.nbytes
                if size > 1024 * 1024:
                    raise BackendUnavailable('Compact public input exceeds its metadata byte limit.')
            else:
                raise ValueError('Public input ports require tensors or compact numeric arrays.')
            if size > reference.MAX_ARRAY_BYTES:
                raise BackendUnavailable('Public input exceeds the audited array byte limit before CPU transfer.')
            total_bytes += size
        if total_bytes > reference.MAX_INPUT_ARCHIVE_BYTES - 1024 * 1024:
            raise BackendUnavailable('Public inputs exceed the audited aggregate byte limit before CPU transfer.')
        from seedvision.benchmarking.native_recipe import _nested_signature
        def signatures():
            return (_nested_signature((), inputs), tuple((name, value.shape, value.dtype.str,
                sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
                for name, value in inputs.items() if isinstance(value, np.ndarray)))
        snapshots = signatures()
        arrays = {name: _cpu_array(value) for name, value in inputs.items()}
        _poll(self.cancel)
        result = self.runner.run(operation, arrays, parameters, cancel_event=self.cancel)
        _poll(self.cancel)
        if type(result) is not reference.ReferenceResult or result.operation != operation:
            raise ValueError("Public execution requires the checked ReferenceResult for the requested operation.")
        if snapshots != signatures():
            raise ValueError("Public operation input changed during the call.")
        from seedvision.benchmarking.reference_stage import _validate_reference_identity
        _validate_reference_identity(result, self.recipe)
        provenance = validate_backend_provenance({
            "schema": "seedfiddle.pipeline-public-execution.v1", "backend": "plantcv",
            "operation": operation, "identity": dict(result.identity),
            "effective_parameters": dict(result.effective_parameters),
            "transport": "CUDA/CPU input -> audited CPU subprocess -> CUDA/CPU output",
            "channel_semantics": dict(result.channel_semantics),
        })
        return result, provenance

    def _watershed(self, stage_id, nodes):
        from seedvision.benchmarking.native_recipe import NativeRecipeNode, _CacheEntry, _output_signatures, _execution_signature
        inputs, _ = self._group_inputs(nodes)
        identity = self.key_for(stage_id, nodes)
        key = _digest(identity)
        cache_name = self.prefix + "public-watershed:" + self._group[0]
        entry = self.cache.values.get(cache_name) if self.cache is not None else None
        if (not isinstance(entry, _CacheEntry) or entry.key != key
                or entry.signatures != _output_signatures(entry.node.outputs)
                or entry.execution_signature != _execution_signature(entry.node)):
            if self.cache is not None:
                self.cache.values.pop(cache_name, None)
            entry = None  # Release an obsolete captured tensor set before its replacement.
            result, provenance = self._call("segmentation.watershed.capture", inputs, {
                "distance": self.recipe.parameters_for_runner(self._group[1])["distance"],
                "label": self.recipe.parameters_for_runner(self._group[5])["label"]})
            capture = result.effective_parameters.get("capture", {})
            if capture.get("status") != "available":
                raise BackendUnavailable("The public watershed call did not yield complete captured locals.")
            device = inputs["mask"].device
            outputs = {name: _upload(result.arrays[name], device) for name in
                       ("distance", "peaks", "markers", "negative_distance", "labels")}
            outputs["coordinates"] = _upload(result.arrays.get("coordinates", np.empty((0, 2), dtype=np.int64)), device)
            label = self.recipe.parameters_for_runner(self._group[5])["label"]
            observation = _observations(result)[label]["estimated_object_count"]
            outputs["marker_count"] = _freeze_json({"sample": label, "variable": "estimated_object_count",
                **dict(observation)}, name="actual public marker observation")
            _poll(self.cancel)
            node = NativeRecipeNode(self._group[0], "segmentation.watershed.capture", "executed",
                MappingProxyType(outputs), MappingProxyType({}), MappingProxyType({}), key,
                backend="plantcv", backend_provenance=provenance)
            entry = _CacheEntry(key, node, _output_signatures(outputs), execution_signature=_execution_signature(node))
            if self.cache is not None:
                self.cache.values[cache_name] = entry
        if self.watch is not None:
            self.watch(entry.node.outputs)
        suffix = self.recipe.stage(stage_id).operation_id.rsplit(".", 1)[-1]
        names = {"distance_map": ("distance",), "local_maxima": ("coordinates", "peaks"),
            "connected_markers": ("markers",), "negate_distance": ("negative_distance",),
            "masked_flood": ("labels",), "marker_count_observation": ("marker_count",)}[suffix]
        outputs = {name: entry.node.outputs[name] for name in names}
        if suffix == "masked_flood":
            outputs["flood_complete"] = True
        return outputs, entry.node.backend_provenance, None

    def execute(self, stage, inputs, parameters, nodes):
        if self.backend_for(stage.stage_id) != "plantcv":
            raise ValueError("Only selected public nodes enter the public executor.")
        if stage.stage_id in self._group:
            return self._watershed(stage.stage_id, nodes)
        operation = stage.operation_id
        selected = dict(parameters)
        public_inputs = inputs
        if selected.get("roi_bounds_xyxy") is None:
            selected.pop("roi_bounds_xyxy", None)
        if operation == "threshold.custom_range":
            selected = {"channel": selected["channel"], "lower_thresh": selected["lower"],
                        "upper_thresh": selected["upper"]}
        if operation == "analysis.naive_bayes_classifier":
            from seedvision.cuda.plantcv_naive_bayes import _validate_model
            model = inputs['model']
            _validate_model(model)
            public_inputs = {'rgb': inputs['rgb_image'],
                'pdf_tables': np.asarray(model.probabilities, dtype=np.float64)}
            selected = {'class_names': list(model.class_names), 'model_source_sha256': model.source_sha256,
                'tables_sha256': model.tables_sha256}
        elif operation == 'color_card.means':
            public_inputs = {'rgb': inputs['bgr'].flip(-1).contiguous(),
                'mask': inputs['labeled_mask'].to(torch.uint8)}
            operation = 'transform.get_color_matrix'
        elif operation == 'color_card.standard':
            public_inputs = {'context': np.array([0], dtype=np.uint8)}
            operation = 'transform.std_color_matrix'
        elif operation == 'color_card.affine':
            public_inputs = {'rgb': inputs['bgr'].flip(-1).contiguous(),
                'source_matrix': np.asarray(inputs['source_matrix'], dtype=np.float64),
                'target_matrix': np.asarray(inputs['target_matrix'], dtype=np.float64)}
            operation = 'transform.affine_color_correction'
        result, provenance = self._call(operation, public_inputs, selected)
        device = next(value.device for value in inputs.values() if torch.is_tensor(value))
        calculation = None
        if operation == "analysis.color":
            observations = _observations(result)
            outputs = {"observations": observations, "histogram": _color_chart(observations)}
        elif operation == "analysis.size":
            outputs = {"observations": _observations(result), "diagnostic": _upload(result.arrays["analysis_image"], device)}
        elif operation == "transform.correct_color":
            outputs = {"corrected_image": _upload(result.arrays["corrected_image"], device)}
            for out, source in (("target_matrix", "target_color_matrix"),
                                ("source_matrix", "source_color_matrix"),
                                ("transformation_matrix", "transformation_matrix")):
                array = np.ascontiguousarray(result.arrays[source])
                outputs[out] = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
        elif operation == "analysis.naive_bayes_classifier":
            names = tuple(inputs['model'].class_names)
            if tuple(result.effective_parameters.get('class_names', ())) != names:
                raise ValueError('Public classifier class ordering differs from the selected frozen model.')
            outputs = {'class_masks': MappingProxyType({name: _upload(result.arrays[f'class_{index}'], device)
                for index, name in enumerate(names)}), 'class_names': names, 'model_identity': inputs['model_identity']}
        elif operation in ('transform.get_color_matrix', 'transform.std_color_matrix'):
            name = 'source_matrix' if operation == 'transform.get_color_matrix' else 'target_matrix'
            outputs = {name: tuple(tuple(float(value) for value in row) for row in result.arrays[name])}
        elif operation == 'transform.affine_color_correction':
            outputs = {'corrected_rgb': _upload(result.arrays['corrected_rgb'], device),
                'transformation_matrix': tuple(tuple(float(value) for value in row)
                    for row in result.arrays['transformation_matrix']),
                'metadata': _freeze_json({'fit': 'affine', 'numpy_version': result.identity['actual_distributions']['numpy']},
                    name='public affine fit')}
        else:
            outputs = {name: _upload(array, device) for name, array in result.arrays.items()}
        if operation == "threshold.otsu":
            from seedvision.benchmarking.reference_stage import validate_otsu_cutoff_adapter
            adapter = result.observations.get("otsu_cutoff_adapter")
            validate_otsu_cutoff_adapter(adapter, selected["object_type"])
            calculation = _freeze_json(dict(adapter), name="public Otsu cutoff adapter")
        _poll(self.cancel)
        return outputs, provenance, calculation
