from __future__ import annotations

from tests.functional_units import functional_units

import io
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock

import numpy as np

from seedvision.benchmarking.reference_runner import (
    MAX_ARRAY_ELEMENTS,
    MAX_ARRAYS,
    PlantCVReferenceRunner,
    ReferenceCancelledError,
    ReferenceIdentityError,
    ReferenceProtocolError,
    ReferenceTimeoutError,
    _run_child,
    _strict_json_loads,
    _read_npz,
    plantcv_reference_status,
    smpts_reference_status,
    validate_array_mapping,
)
from seedvision.benchmarking.plantcv_worker import (
    _AuditedPlantCVSourceFinder,
    WorkerError,
    _read_request,
    _run_operation,
    _verify_callable_origin,
    _verify_package_import_location,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTHON = ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"
SOURCE_ROOT = ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3"
SOURCE_MANIFEST = ROOT / "artifacts" / "reference-sources" / "plantcv-source-audit.json"
ENVIRONMENT_MANIFEST = ROOT / "artifacts" / "reference-sources" / "plantcv-environment.json"
REQUIREMENTS_LOCK = ROOT / "config" / "plantcv-reference-requirements.txt"
HAS_REFERENCE_ENV = all(
    path.exists() for path in (REFERENCE_PYTHON, SOURCE_ROOT, SOURCE_MANIFEST, ENVIRONMENT_MANIFEST, REQUIREMENTS_LOCK)
)


@functional_units
class ReferenceArrayProtocolTests(unittest.TestCase):

    def check_test_rejects_object_nonfinite_and_oversized_arrays(self) -> None:
        with self.assertRaises(ReferenceProtocolError):
            validate_array_mapping({"values": np.array([object()], dtype=object)})
        with self.assertRaises(ReferenceProtocolError):
            validate_array_mapping({"values": np.array([np.nan], dtype=np.float32)})
        with self.assertRaises(ReferenceProtocolError):
            validate_array_mapping({"bad/name": np.ones((2,), dtype=np.uint8)})
        for reserved in ("file", "allow_pickle"):
            with self.subTest(reserved=reserved), self.assertRaises(ReferenceProtocolError):
                validate_array_mapping({reserved: np.ones((2,), dtype=np.uint8)})

    def check_test_array_inventory_limits_precede_any_contiguous_copy(self) -> None:
        base = np.zeros((1,), dtype=np.uint8)
        per_array_elements = (MAX_ARRAY_ELEMENTS // 2) + 1
        views = {
            "first": np.lib.stride_tricks.as_strided(base, shape=(per_array_elements,), strides=(0,)),
            "second": np.lib.stride_tricks.as_strided(base, shape=(per_array_elements,), strides=(0,)),
        }
        with mock.patch("seedvision.benchmarking.reference_runner.np.ascontiguousarray") as make_contiguous:
            with self.assertRaisesRegex(ReferenceProtocolError, "aggregate"):
                validate_array_mapping(views)
            make_contiguous.assert_not_called()

        too_many = {
            f"array_{index}": np.zeros((1,), dtype=np.uint8)
            for index in range(MAX_ARRAYS + 1)
        }
        with mock.patch("seedvision.benchmarking.reference_runner.np.ascontiguousarray") as make_contiguous:
            with self.assertRaisesRegex(ReferenceProtocolError, "array limit"):
                validate_array_mapping(too_many)
            make_contiguous.assert_not_called()

    def check_test_strict_json_rejects_nonfinite_constants_and_duplicate_keys(self) -> None:
        for document in (
            '{"value":NaN}', '{"value":Infinity}', '{"value":1,"value":2}',
            '{"value":1e999}', '{"nested":{"value":-1e999}}',
        ):
            with self.subTest(document=document), self.assertRaises(ValueError):
                _strict_json_loads(document)

        with tempfile.TemporaryDirectory() as temp_name:
            for parameters in ('{"x":NaN}', '{"x":1,"x":2}', '{"x":1e999}'):
                request = Path(temp_name) / "request.json"
                request.write_text(
                    '{"protocol_version":1,"request_id":"0123456789abcdef0123456789abcdef",'
                    '"operation":"threshold.otsu","parameters":' + parameters + ',"input_arrays":{}}',
                    encoding="utf-8",
                )
                with self.subTest(parameters=parameters), self.assertRaisesRegex(WorkerError, "invalid JSON"):
                    _read_request(request)

    def check_test_rejects_pickle_and_oversized_npy_header_before_array_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            object_path = root / "object.npz"
            np.savez_compressed(object_path, values=np.array([{"unsafe": True}], dtype=object))
            with self.assertRaises(ReferenceProtocolError):
                _read_npz(object_path, max_archive_bytes=2 * 1024 * 1024)

            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                header,
                {
                    "descr": np.lib.format.dtype_to_descr(np.dtype("uint8")),
                    "fortran_order": False,
                    "shape": (MAX_ARRAY_ELEMENTS + 1,),
                },
            )
            bomb_path = root / "large-header.npz"
            with zipfile.ZipFile(bomb_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("large.npy", header.getvalue())
            with self.assertRaises(ReferenceProtocolError):
                _read_npz(bomb_path, max_archive_bytes=2 * 1024 * 1024)


    def check_test_npz_keyword_names_are_rejected_during_archive_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            for name in ("file", "allow_pickle"):
                path = Path(temp_name) / f"{name}.npz"
                np.savez_compressed(path, values=np.ones((2,), dtype=np.uint8))
                # Rewrite the safe member name to prove the decoder independently
                # rejects keys that collide with NumPy's savez keyword API.
                with zipfile.ZipFile(path, "r") as source:
                    payload = source.read("values.npy")
                with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as target:
                    target.writestr(f"{name}.npy", payload)
                with self.subTest(name=name), self.assertRaises(ReferenceProtocolError):
                    _read_npz(path, max_archive_bytes=2 * 1024 * 1024)

    def check_test_shadow_package_origin_is_rejected_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            distribution = root / "site-packages"
            package = distribution / "plantcv"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            shadow = root / "shadow" / "plantcv.py"
            shadow.parent.mkdir()
            shadow.write_text("raise RuntimeError('executed shadow')", encoding="utf-8")
            fake_spec = SimpleNamespace(origin=str(shadow))
            with mock.patch("seedvision.benchmarking.plantcv_worker.importlib.util.find_spec", return_value=fake_spec):
                with self.assertRaisesRegex(WorkerError, "outside the verified distribution"):
                    _verify_package_import_location(distribution)

    def check_test_audited_import_compiles_verified_source_instead_of_unchecked_pyc(self) -> None:
        fullname = "plantcv.plantcv.threshold.threshold_methods"
        relative = "plantcv/plantcv/threshold/threshold_methods.py"
        package_names = ("plantcv", "plantcv.plantcv", "plantcv.plantcv.threshold")
        module_names = (*package_names, fullname)
        missing = object()
        previous_modules = {name: sys.modules.get(name, missing) for name in module_names}
        previous_attrs: list[tuple[object, str, object]] = []
        finder = None
        with tempfile.TemporaryDirectory() as temp_name:
            distribution = Path(temp_name) / "site-packages"
            source_path = distribution / relative
            source_path.parent.mkdir(parents=True)
            verified_source = b"def binary():\n    return 'verified-source'\n"
            source_path.write_bytes(verified_source)
            fake_source = Path(temp_name) / "fake.py"
            fake_source.write_text("def binary():\n    return 'unchecked-pyc'\n", encoding="utf-8")
            cache_path = Path(importlib.util.cache_from_source(str(source_path)))
            py_compile.compile(
                str(fake_source),
                cfile=str(cache_path),
                dfile=str(source_path),
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
            )

            standard_loader = importlib.machinery.SourceFileLoader(fullname, str(source_path))
            standard_code = standard_loader.get_code(fullname)
            fake_namespace = {"__name__": fullname}
            exec(standard_code, fake_namespace)
            self.assertEqual(fake_namespace["binary"](), "unchecked-pyc")

            for name in package_names:
                package = importlib.util.module_from_spec(
                    importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
                )
                package.__path__ = [str(distribution.joinpath(*name.split(".")))]
                sys.modules[name] = package
                if "." in name:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = sys.modules[parent_name]
                    previous_attrs.append((parent, child_name, getattr(parent, child_name, missing)))
                    setattr(parent, child_name, package)

            record = {
                "path": relative,
                "bytes": len(verified_source),
                "sha256": hashlib.sha256(verified_source).hexdigest(),
            }
            finder = _AuditedPlantCVSourceFinder([record], distribution)
            sys.meta_path.insert(0, finder)
            try:
                module = importlib.import_module(fullname)
                fake_pcv = SimpleNamespace(threshold=SimpleNamespace(binary=module.binary))
                self.assertEqual(module.binary(), "verified-source")
                _verify_callable_origin("threshold.binary", fake_pcv, [record], distribution, finder)
            finally:
                if finder in sys.meta_path:
                    sys.meta_path.remove(finder)
                for parent, child_name, previous in reversed(previous_attrs):
                    if previous is missing:
                        try:
                            delattr(parent, child_name)
                        except AttributeError:
                            pass
                    else:
                        setattr(parent, child_name, previous)
                for name, previous in previous_modules.items():
                    if previous is missing:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = previous

    def check_test_rejects_operation_outside_audited_callable_allowlist(self) -> None:
        runner = PlantCVReferenceRunner(sys.executable)
        with self.assertRaisesRegex(ReferenceProtocolError, "callable allowlist"):
            runner.run("plantcv.arbitrary_python", {"gray": np.zeros((2, 2), dtype=np.uint8)})

    def test_validation_and_recovery(self):
        """7 related ReferenceArrayProtocolTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_rejects_object_nonfinite_and_oversized_arrays)
        self.scenario(self.check_test_array_inventory_limits_precede_any_contiguous_copy)
        self.scenario(self.check_test_strict_json_rejects_nonfinite_constants_and_duplicate_keys)
        self.scenario(self.check_test_rejects_pickle_and_oversized_npy_header_before_array_load)
        self.scenario(self.check_test_npz_keyword_names_are_rejected_during_archive_preflight)
        self.scenario(self.check_test_shadow_package_origin_is_rejected_before_import)
        self.scenario(self.check_test_rejects_operation_outside_audited_callable_allowlist)

    def test_workflow_lifecycle(self):
        """1 related ReferenceArrayProtocolTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_audited_import_compiles_verified_source_instead_of_unchecked_pyc)


class ReferenceIdentityTests(unittest.TestCase):
    @unittest.skipUnless(HAS_REFERENCE_ENV, "isolated PlantCV reference environment is not provisioned")
    def test_source_tampering_is_rejected_before_worker_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            altered_root = Path(temp_name) / "source"
            altered_root.mkdir()
            (altered_root / "LICENSE").write_text("tampered source", encoding="utf-8")
            runner = PlantCVReferenceRunner(
                REFERENCE_PYTHON,
                source_root=altered_root,
                source_manifest_path=SOURCE_MANIFEST,
                environment_manifest_path=ENVIRONMENT_MANIFEST,
                requirements_lock_path=REQUIREMENTS_LOCK,
            )
            with self.assertRaisesRegex(ReferenceIdentityError, "source changed"):
                runner.run("threshold.otsu", {"gray": np.zeros((4, 4), dtype=np.uint8)})

    @unittest.skipUnless(HAS_REFERENCE_ENV, "isolated PlantCV reference environment is not provisioned")
    def test_environment_inventory_tampering_is_rejected(self) -> None:
        inventory = json.loads(ENVIRONMENT_MANIFEST.read_text(encoding="utf-8"))
        package = next(iter(inventory))
        inventory[package] = "0.0.0-tampered"
        with tempfile.TemporaryDirectory() as temp_name:
            altered = Path(temp_name) / "environment.json"
            altered.write_text(json.dumps(inventory), encoding="utf-8")
            runner = PlantCVReferenceRunner(
                REFERENCE_PYTHON,
                source_manifest_path=SOURCE_MANIFEST,
                source_root=SOURCE_ROOT,
                environment_manifest_path=altered,
                requirements_lock_path=REQUIREMENTS_LOCK,
            )
            with self.assertRaisesRegex(ReferenceIdentityError, "differs from the hash-pinned requirements"):
                runner.run("threshold.otsu", {"gray": np.zeros((4, 4), dtype=np.uint8)})

    def test_smpts_original_backend_is_explicitly_unavailable(self) -> None:
        status = smpts_reference_status()
        self.assertEqual(status["backend"], "smpts_reference")
        self.assertFalse(status["available"])
        self.assertEqual(status["state"], "unavailable")
        self.assertIn("not been recovered or executed", status["reason"])

    @unittest.skipUnless(HAS_REFERENCE_ENV, "isolated PlantCV reference environment is not provisioned")
    def test_availability_probes_interpreter_and_reports_unavailable_desktop_python(self) -> None:
        # The identity probe has a 10 s budget sized for an idle machine. This
        # test checks identity, not latency, so give the probe headroom when
        # the parallel full suite is saturating the CPU and disk.
        from unittest.mock import patch
        import seedvision.benchmarking.reference_runner as runner_module

        original_run_child = runner_module._run_child

        def patient_run_child(command, *, timeout_seconds, **kwargs):
            return original_run_child(command, timeout_seconds=max(timeout_seconds, 60.0), **kwargs)

        self.enterContext(patch.object(runner_module, "_run_child", patient_run_child))
        reference_status = plantcv_reference_status(python_executable=REFERENCE_PYTHON)
        self.assertTrue(reference_status["available"], reference_status)
        self.assertEqual(reference_status["backend"], "plantcv_reference")
        self.assertEqual(reference_status["state"], "environment_and_source_verified")

        desktop_status = plantcv_reference_status(python_executable=sys.executable)
        self.assertEqual(desktop_status["backend"], "plantcv_reference")
        self.assertFalse(desktop_status["available"], desktop_status)
        self.assertEqual(desktop_status["state"], "unavailable")


@functional_units
class IsolatedPlantCVExecutionTests(unittest.TestCase):

    def check_test_timeout_and_cancellation_remove_temporary_protocol_files(self) -> None:
        image = np.zeros((3, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as isolated_temp_root:
            runner = PlantCVReferenceRunner(REFERENCE_PYTHON, temporary_root=isolated_temp_root)
            root = Path(isolated_temp_root)
            self.assertEqual(list(root.iterdir()), [])
            with self.assertRaises(ReferenceTimeoutError):
                runner.run("threshold.otsu", {"gray": image}, timeout_seconds=0.01)
            self.assertEqual(list(root.iterdir()), [])

            cancellation = threading.Event()
            cancellation.set()
            with self.assertRaises(ReferenceCancelledError):
                runner.run("threshold.otsu", {"gray": image}, cancel_event=cancellation)
            self.assertEqual(list(root.iterdir()), [])

            def raising_callback() -> bool:
                raise RuntimeError("callback failure")

            with self.assertRaisesRegex(RuntimeError, "callback failure"):
                runner.run("threshold.otsu", {"gray": image}, cancel_event=raising_callback)
            self.assertEqual(list(root.iterdir()), [])

    def check_test_missing_external_python_has_clear_availability_state(self) -> None:
        status = plantcv_reference_status(python_executable=Path(tempfile.gettempdir()) / "missing-reference-python.exe")
        self.assertFalse(status["available"])
        self.assertIn("does not exist", status["reason"])

    def check_test_callback_exception_kills_reaps_and_closes_worker_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            child_holder: dict[str, subprocess.Popen[bytes]] = {}
            real_popen = subprocess.Popen

            def tracked_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
                child = real_popen(*args, **kwargs)
                child_holder["child"] = child
                return child

            def raising_callback() -> bool:
                raise LookupError("cancellation callback failed")

            with mock.patch("seedvision.benchmarking.reference_runner.subprocess.Popen", side_effect=tracked_popen):
                with self.assertRaisesRegex(LookupError, "callback failed"):
                    _run_child(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        cwd=Path(temp_name), env=os.environ.copy(), timeout_seconds=20,
                        cancel_event=raising_callback,
                    )
            child = child_holder["child"]
            self.assertIsNotNone(child.poll())
            self.assertTrue(child.stdout.closed)
            self.assertTrue(child.stderr.closed)

    def check_test_auto_color_adapter_preserves_chip_observations_and_calibration(self) -> None:
        expected_metadata = {
            "median_color_chip_size": 1440.0,
            "median_color_chip_width": 40.0,
            "median_color_chip_height": 36.0,
        }
        outputs = SimpleNamespace(observations={}, measurements={}, metadata=expected_metadata)
        params = SimpleNamespace(unit="mm", px_width=1.5, px_height=1.25)

        def fake_auto_correct(*, rgb_img: np.ndarray, **kwargs: object) -> np.ndarray:
            np.testing.assert_array_equal(rgb_img[0, 0], (10, 30, 250))
            return np.broadcast_to(np.array((3, 2, 1), dtype=np.uint8), rgb_img.shape).copy()

        pcv = SimpleNamespace(
            transform=SimpleNamespace(auto_correct_color=fake_auto_correct), outputs=outputs, params=params
        )
        request = {
            "operation": "transform.auto_correct_color",
            "parameters": {"color_chip_size": "classic"},
        }
        rgb = np.broadcast_to(np.array((250, 30, 10), dtype=np.uint8), (2, 3, 3)).copy()
        arrays, observations, effective, _, semantics = _run_operation(request, {"rgb": rgb}, Path("."), pcv)
        np.testing.assert_array_equal(arrays["corrected_image"][0, 0], (1, 2, 3))
        self.assertEqual(observations["metadata"], expected_metadata)
        self.assertEqual(effective["calibration_effects"]["px_width"], 1.5)
        self.assertEqual(effective["calibration_effects"]["px_height"], 1.25)
        self.assertIn("RGB uint8 HWC", semantics["corrected_image"])

    def check_test_large_kernel_and_watershed_parameters_are_rejected_before_call(self) -> None:
        image = np.zeros((8, 8), dtype=np.uint8)
        pcv = mock.Mock()
        cases = (
            ("morphology.erode", {"gray": image}, {"ksize": 100_000, "iterations": 1}),
            ("morphology.dilate", {"gray": image}, {"ksize": 3, "iterations": 100_000}),
            ("filter.median_blur", {"gray": image}, {"ksize": 100_000}),
            ("threshold.mean", {"gray": image}, {"ksize": 100_000, "offset": 1}),
            (
                "segmentation.watershed",
                {"rgb": np.zeros((8, 8, 3), dtype=np.uint8), "mask": image},
                {"distance": 100_000},
            ),
            (
                "transform.auto_correct_color",
                {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
                {"block_size": 100_001},
            ),
            (
                "transform.auto_correct_color",
                {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
                {"radius": 100_000},
            ),
        )
        for operation, arrays, parameters in cases:
            request = {"operation": operation, "parameters": parameters}
            with self.subTest(operation=operation), self.assertRaisesRegex(WorkerError, "supported range"):
                _run_operation(request, arrays, Path("."), pcv)
        pcv.assert_not_called()

    @unittest.skipUnless(HAS_REFERENCE_ENV, "isolated PlantCV reference environment is not provisioned")
    def test_timeout_and_cancellation_remove_temporary_protocol_files(self):
        self.scenario(self.check_test_timeout_and_cancellation_remove_temporary_protocol_files, labelled=False)

    def test_validation_and_recovery(self):
        """2 related IsolatedPlantCVExecutionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_missing_external_python_has_clear_availability_state)
        self.scenario(self.check_test_large_kernel_and_watershed_parameters_are_rejected_before_call)

    def test_workflow_lifecycle(self):
        """2 related IsolatedPlantCVExecutionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_callback_exception_kills_reaps_and_closes_worker_process)
        self.scenario(self.check_test_auto_color_adapter_preserves_chip_observations_and_calibration)


if __name__ == "__main__":
    unittest.main()
