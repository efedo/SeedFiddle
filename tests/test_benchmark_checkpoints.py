from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile
from unittest.mock import patch

import numpy as np

import seedvision.benchmarking.checkpoints as checkpoint_module
from seedvision.benchmarking.checkpoints import (
    CheckpointCancelled,
    CheckpointError,
    CheckpointIdentity,
    CheckpointLimits,
    CheckpointMismatch,
    CoordinateFrame,
    ExecutionRecord,
    ExecutionStatus,
    InvalidCheckpoint,
    MeasurementContract,
    RasterRole,
    load_checkpoint,
    load_stage_input,
    publish_checkpoint,
)


def _identity(*, upstream: str | None = None) -> CheckpointIdentity:
    return CheckpointIdentity(
        source_sha256="a" * 64,
        coordinate_frame=CoordinateFrame(
            frame_id="corrected_image_v1",
            shape=(7, 9),
            origin="top_left_pixel_centres",
            axes="x_right_y_down",
            units="pixel",
            transform_sha256="b" * 64,
        ),
        stage_id="instance_partition",
        recipe_id="plantcv_seed_recipe",
        recipe_version="4.2.1",
        recipe_sha256="c" * 64,
        graph_sha256="d" * 64,
        settings_sha256="e" * 64,
        backend_id="plantcv_reference",
        backend_version="4.2.1",
        environment_versions={"python": "3.12.4", "numpy": "2.1.0", "plantcv": "4.2.1"},
        image_semantics={
            "color_order": "BGR",
            "dtype": "uint8",
            "value_range": [0, 255],
            "channel_conversion": "none",
            "geometric_transform": "source_to_corrected_homography",
            "interpolation": "linear",
        },
        mask_semantics={
            "foreground_polarity": "nonzero_is_foreground",
            "representation": "boolean",
            "connectivity": 8,
            "validity_region": "all_pixels",
            "border_policy": "constant_background",
        },
        instance_semantics={
            "label_dtype": "int32",
            "background_label": 0,
            "label_id_mapping": "identity_with_background_zero",
            "split_boundary_ownership": "watershed_line_background",
            "visibility": "all_detected_instances",
            "eligibility": "reviewed_complete_instances_only",
        },
        upstream_checkpoint_sha256=upstream,
    )


def _measurement(*, eligible: bool = True) -> MeasurementContract:
    return MeasurementContract(
        definition_id="seedfiddle.instance_area",
        definition_version="2",
        mask_type="instance_partition",
        units="pixel^2",
        calibration_status="uncalibrated",
        uncertainty="not_estimated",
        eligible=eligible,
        reason="" if eligible else "No complete reviewed instance labels.",
    )


class _CountedTensor:
    """Tiny CUDA-shaped stand-in proving preflight happens before .to()."""

    def __init__(self, shape=(2, 3)):
        self.shape = shape
        self.dtype = "torch.uint8"
        self.device = "cuda:0"
        self.transfers = 0

    def numel(self):
        return int(np.prod(self.shape))

    def element_size(self):
        return 1

    def detach(self):
        return self

    def contiguous(self):
        return self

    def to(self, *, device, copy):
        self.transfers += 1
        return self

    def numpy(self):
        return np.zeros(self.shape, dtype=np.uint8)


class BenchmarkCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def publish(self, *, identity=None, arrays=None, **kwargs):
        values = (
            {"mask": np.eye(7, 9, dtype=np.bool_), "labels": np.arange(63, dtype=np.int32).reshape(7, 9)}
            if arrays is None else arrays
        )
        roles = kwargs.pop(
            "raster_roles",
            {
                "mask": RasterRole("mask", np.dtype(np.bool_).str),
                "labels": RasterRole("instances", np.dtype(np.int32).str),
            },
        )
        return publish_checkpoint(
            self.root,
            _identity() if identity is None else identity,
            values,
            execution=ExecutionRecord(ExecutionStatus.AVAILABLE, elapsed_ms=1.25, transfer_ms=0.2),
            measurement=_measurement(),
            raster_roles=roles,
            allow_cpu_transfer=True,
            **kwargs,
        )

    def test_real_npz_round_trip_preserves_identity_semantics_and_dtype(self) -> None:
        reference = self.publish(metadata={"adapter": "audited", "warnings": ["line policy recorded"]})

        loaded = load_checkpoint(self.root, reference.checkpoint_sha256, expected_identity=_identity())

        self.assertEqual(loaded.checkpoint_sha256, reference.checkpoint_sha256)
        self.assertEqual(loaded.identity, _identity())
        self.assertEqual(loaded.identity.coordinate_frame.shape, (7, 9))
        self.assertEqual(loaded.identity.image_semantics["color_order"], "BGR")
        self.assertEqual(loaded.execution.status, ExecutionStatus.AVAILABLE)
        self.assertTrue(loaded.measurement.eligible)
        self.assertEqual(loaded.arrays["labels"].dtype, np.dtype("int32"))
        self.assertEqual(loaded.arrays["mask"].dtype, np.dtype("bool"))
        self.assertEqual(loaded.raster_roles["labels"].kind, "instances")
        self.assertEqual(loaded.raster_roles["mask"].kind, "mask")
        self.assertFalse(loaded.arrays["labels"].flags.writeable)
        with self.assertRaises(ValueError):
            loaded.arrays["labels"].setflags(write=True)
        self.assertEqual(loaded.metadata["warnings"], ("line policy recorded",))

    def test_cpu_serialization_requires_explicit_benchmark_opt_in(self) -> None:
        with self.assertRaisesRegex(CheckpointError, "allow_cpu_transfer=True"):
            publish_checkpoint(
                self.root,
                _identity(),
                {"mask": np.ones((2, 2), dtype=np.uint8)},
                execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
                measurement=_measurement(),
            )

    def test_stage_replay_requires_exact_source_frame_recipe_and_upstream_identity(self) -> None:
        upstream = "f" * 64
        identity = _identity(upstream=upstream)
        reference = self.publish(identity=identity)
        expected_variants = (
            replace(identity, source_sha256="1" * 64),
            replace(identity, stage_id="instance_measurement"),
            replace(identity, coordinate_frame=replace(identity.coordinate_frame, shape=(8, 9))),
            replace(identity, coordinate_frame=replace(identity.coordinate_frame, transform_sha256="2" * 64)),
            replace(identity, recipe_sha256="3" * 64),
            replace(identity, graph_sha256="4" * 64),
            replace(identity, settings_sha256="5" * 64),
            replace(identity, backend_version="4.2.2"),
            replace(identity, environment_versions={"python": "3.12.4", "numpy": "2.2.0", "plantcv": "4.2.1"}),
            replace(identity, upstream_checkpoint_sha256="6" * 64),
        )

        for expected in expected_variants:
            with self.subTest(expected=expected):
                with self.assertRaises(CheckpointMismatch):
                    load_checkpoint(self.root, reference.checkpoint_sha256, expected_identity=expected)

        loaded = load_stage_input(
            self.root,
            reference.checkpoint_sha256,
            expected_identity=identity,
        )
        self.assertEqual(loaded.identity.upstream_checkpoint_sha256, upstream)
        with self.assertRaisesRegex(CheckpointMismatch, "exact upstream"):
            load_stage_input(self.root, reference.checkpoint_sha256, expected_identity=_identity())
        with self.assertRaises(CheckpointMismatch):
            load_checkpoint(
                self.root,
                reference.checkpoint_sha256,
                expected_measurement=_measurement(eligible=False),
            )

    def test_content_address_is_immutable_and_different_content_gets_new_identity(self) -> None:
        first = self.publish()
        repeated = self.publish()
        changed = self.publish(arrays={"mask": np.zeros((7, 9), dtype=np.bool_), "labels": np.arange(63, dtype=np.int32).reshape(7, 9)})

        self.assertEqual(first.checkpoint_sha256, repeated.checkpoint_sha256)
        self.assertNotEqual(first.checkpoint_sha256, changed.checkpoint_sha256)
        self.assertTrue(first.path.is_dir())
        self.assertTrue(changed.path.is_dir())

    def test_tampered_recipe_metadata_and_coordinate_metadata_are_rejected(self) -> None:
        for field, mutate in (
            ("recipe_sha256", lambda identity: identity.__setitem__("recipe_sha256", "9" * 64)),
            ("coordinate_frame", lambda identity: identity["coordinate_frame"].__setitem__("shape", [70, 90])),
        ):
            with self.subTest(field=field):
                reference = self.publish()
                manifest_path = reference.path / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest["identity"])
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(InvalidCheckpoint, "tampered"):
                    load_checkpoint(self.root, reference.checkpoint_sha256)
                # Restore a clean address before the next mutation.
                shutil_path = reference.path
                for child in shutil_path.iterdir():
                    child.unlink()
                shutil_path.rmdir()

    def test_changed_npz_values_and_dtype_are_rejected(self) -> None:
        for change_dtype in (False, True):
            with self.subTest(change_dtype=change_dtype):
                reference = self.publish()
                with np.load(reference.path / "arrays.npz", allow_pickle=False) as archive:
                    values = {name: archive[name] for name in archive.files}
                values["labels"] = values["labels"].astype(np.int64) if change_dtype else values["labels"] + 1
                with (reference.path / "arrays.npz").open("wb") as stream:
                    np.savez(stream, **values)
                with self.assertRaises(InvalidCheckpoint):
                    load_checkpoint(self.root, reference.checkpoint_sha256)
                for child in reference.path.iterdir():
                    child.unlink()
                reference.path.rmdir()

    def test_valid_empty_and_unavailable_execution_states_stay_distinct(self) -> None:
        empty = publish_checkpoint(
            self.root,
            _identity(),
            {"labels": np.zeros((7, 9), dtype=np.int32)},
            execution=ExecutionRecord(ExecutionStatus.EMPTY, warnings=("valid input, no seeds",)),
            measurement=_measurement(eligible=False),
            allow_cpu_transfer=True,
        )
        failed = publish_checkpoint(
            self.root,
            _identity(),
            {},
            execution=ExecutionRecord(ExecutionStatus.FAILED, reason="Reference adapter crashed."),
            measurement=_measurement(eligible=False),
            allow_cpu_transfer=True,
        )

        self.assertIs(load_checkpoint(self.root, empty.checkpoint_sha256).execution.status, ExecutionStatus.EMPTY)
        loaded_failed = load_checkpoint(self.root, failed.checkpoint_sha256)
        self.assertIs(loaded_failed.execution.status, ExecutionStatus.FAILED)
        self.assertEqual(loaded_failed.arrays, {})
        with self.assertRaises(CheckpointMismatch):
            load_stage_input(self.root, empty.checkpoint_sha256, expected_identity=_identity())
        for status in (ExecutionStatus.FAILED, ExecutionStatus.UNSUPPORTED):
            unavailable = publish_checkpoint(
                self.root,
                _identity(upstream="f" * 64),
                {},
                execution=ExecutionRecord(status, reason=f"{status.value} for this adapter."),
                measurement=_measurement(eligible=False),
                allow_cpu_transfer=True,
            )
            with self.subTest(status=status):
                with self.assertRaisesRegex(CheckpointMismatch, status.value):
                    load_stage_input(
                        self.root,
                        unavailable.checkpoint_sha256,
                        expected_identity=_identity(upstream="f" * 64),
                    )

    def test_compact_empty_coordinate_table_round_trips_immutably(self) -> None:
        reference = publish_checkpoint(
            self.root,
            _identity(),
            {"centres_xy": np.empty((0, 2), dtype=np.float32)},
            execution=ExecutionRecord(ExecutionStatus.EMPTY),
            measurement=_measurement(eligible=False),
            allow_cpu_transfer=True,
        )
        loaded = load_checkpoint(self.root, reference.checkpoint_sha256)

        self.assertEqual(loaded.arrays["centres_xy"].shape, (0, 2))
        self.assertEqual(loaded.arrays["centres_xy"].dtype, np.dtype(np.float32))
        self.assertFalse(loaded.arrays["centres_xy"].flags.writeable)
        with self.assertRaises(ValueError):
            loaded.arrays["centres_xy"].setflags(write=True)
        self.assertEqual(dict(loaded.raster_roles), {})

    def test_array_and_metadata_limits_and_cancellation_are_enforced(self) -> None:
        limited = CheckpointLimits(max_array_bytes=8, max_total_array_bytes=8)
        with self.assertRaisesRegex(CheckpointError, "Array mask exceeds"):
            self.publish(arrays={"mask": np.ones((3, 3), dtype=np.uint8)}, raster_roles={}, limits=limited)
        with self.assertRaisesRegex(CheckpointError, "metadata"):
            self.publish(metadata={"large": "x" * 100}, limits=CheckpointLimits(max_metadata_bytes=32))
        with self.assertRaises(CheckpointCancelled):
            self.publish(cancel=lambda: True)
        self.assertEqual(list(self.root.glob("*.checkpoint")), [])
        self.assertEqual(list(self.root.glob(".*")), [])

    def test_aggregate_budget_is_checked_before_any_tensor_cpu_transfer(self) -> None:
        first = _CountedTensor()
        second = _CountedTensor()
        with self.assertRaisesRegex(CheckpointError, "total limit"):
            publish_checkpoint(
                self.root,
                _identity(),
                {"first": first, "second": second},
                execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
                measurement=_measurement(),
                allow_cpu_transfer=True,
                raster_roles={},
                limits=CheckpointLimits(max_array_bytes=8, max_total_array_bytes=8),
            )
        self.assertEqual((first.transfers, second.transfers), (0, 0))

    def test_npz_container_limit_is_preflighted_before_tensor_transfer(self) -> None:
        expected_archive = BytesIO()
        np.savez(expected_archive, a=np.zeros((2, 3), dtype=np.uint8))
        exact_size = len(expected_archive.getvalue())

        too_tight = _CountedTensor()
        with self.assertRaisesRegex(CheckpointError, "array archive requires"):
            publish_checkpoint(
                self.root,
                _identity(),
                {"a": too_tight},
                execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
                measurement=_measurement(),
                allow_cpu_transfer=True,
                raster_roles={},
                limits=CheckpointLimits(max_npz_bytes=exact_size - 1),
            )
        self.assertEqual(too_tight.transfers, 0)

        # The exact format size remains admissible and verifies the estimate
        # includes both the NPY header and ZIP container/member overhead.
        exact_limit = _CountedTensor()
        published = publish_checkpoint(
            self.root,
            _identity(),
            {"a": exact_limit},
            execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
            measurement=_measurement(),
            allow_cpu_transfer=True,
            raster_roles={},
            limits=CheckpointLimits(max_npz_bytes=exact_size),
        )
        self.assertEqual(exact_limit.transfers, 1)
        self.assertEqual((published.path / "arrays.npz").stat().st_size, exact_size)
        self.assertEqual(load_checkpoint(self.root, published.checkpoint_sha256).arrays["a"].shape, (2, 3))

    def test_npz_keyword_collisions_are_rejected_before_tensor_transfer(self) -> None:
        for reserved_name in ("file", "allow_pickle"):
            with self.subTest(name=reserved_name):
                tensor = _CountedTensor()
                with self.assertRaisesRegex(CheckpointError, "Invalid checkpoint array name"):
                    publish_checkpoint(
                        self.root,
                        _identity(),
                        {reserved_name: tensor},
                        execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
                        measurement=_measurement(),
                        allow_cpu_transfer=True,
                        raster_roles={},
                    )
                self.assertEqual(tensor.transfers, 0)

    def test_loader_rejects_npz_keyword_collisions_in_rehashed_manifests(self) -> None:
        for reserved_name in ("file", "allow_pickle"):
            with self.subTest(name=reserved_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                reference = publish_checkpoint(
                    root,
                    _identity(),
                    {"compact": np.arange(6, dtype=np.uint8)},
                    execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
                    measurement=_measurement(),
                    allow_cpu_transfer=True,
                    raster_roles={},
                )
                manifest_path = reference.path / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["arrays"][0]["name"] = reserved_name
                manifest["checkpoint_sha256"] = checkpoint_module._content_sha256(manifest)
                digest = manifest["checkpoint_sha256"]
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                os.replace(reference.path, root / f"{digest}.checkpoint")

                with self.assertRaisesRegex(InvalidCheckpoint, "invalid array name"):
                    load_checkpoint(root, digest)

    def test_raster_roles_bind_frame_dtype_and_semantic_ranges(self) -> None:
        with self.assertRaisesRegex(CheckpointError, "differs"):
            self.publish(raster_roles={"mask": RasterRole("mask", np.dtype(np.bool_).str), "labels": RasterRole("instances", np.dtype(np.int16).str)})
        with self.assertRaisesRegex(ValueError, "rasters must use HW"):
            self.publish(raster_roles={"mask": RasterRole("mask", np.dtype(np.bool_).str), "labels": RasterRole("instances", np.dtype(np.int32).str, layout="HWC")})
        with self.assertRaisesRegex(CheckpointError, "representation"):
            self.publish(arrays={"mask": np.full((7, 9), 255, dtype=np.uint8), "labels": np.arange(63, dtype=np.int32).reshape(7, 9)}, raster_roles={"mask": RasterRole("mask", np.dtype(np.uint8).str), "labels": RasterRole("instances", np.dtype(np.int32).str)})

    def test_image_raster_role_binds_channels_dtype_shape_and_value_range(self) -> None:
        identity = _identity()
        raster_roles = {"image": RasterRole("image", np.dtype(np.uint8).str, layout="HWC")}
        common = {
            "execution": ExecutionRecord(ExecutionStatus.AVAILABLE),
            "measurement": _measurement(),
            "raster_roles": raster_roles,
            "allow_cpu_transfer": True,
        }
        good = publish_checkpoint(self.root, identity, {"image": np.zeros((7, 9, 3), dtype=np.uint8)}, **common)
        self.assertEqual(load_checkpoint(self.root, good.checkpoint_sha256).arrays["image"].shape, (7, 9, 3))
        for value in (
            np.zeros((7, 8, 3), dtype=np.uint8),
            np.zeros((7, 9, 3), dtype=np.int16),
        ):
            with self.subTest(dtype=value.dtype, shape=value.shape):
                with self.assertRaises(CheckpointError):
                    publish_checkpoint(self.root, identity, {"image": value}, **common)
        bounded_identity = replace(identity, image_semantics={**identity.image_semantics, "value_range": [0, 200]})
        with self.assertRaisesRegex(CheckpointError, "declared range"):
            publish_checkpoint(
                self.root,
                bounded_identity,
                {"image": np.full((7, 9, 3), 201, dtype=np.uint8)},
                **common,
            )

    def test_loader_rejects_semantically_invalid_raster_role_even_with_rehashed_manifest(self) -> None:
        reference = self.publish()
        manifest_path = reference.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        label_spec = next(item for item in manifest["arrays"] if item["name"] == "labels")
        label_spec["role"]["dtype"] = np.dtype(np.int16).str
        manifest["checkpoint_sha256"] = checkpoint_module._content_sha256(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        changed_digest = manifest["checkpoint_sha256"]
        changed_path = self.root / f"{changed_digest}.checkpoint"
        os.replace(reference.path, changed_path)

        with self.assertRaisesRegex(InvalidCheckpoint, "semantic contract"):
            load_checkpoint(self.root, changed_digest)

    def test_post_staging_cancellation_cleans_temporary_directory(self) -> None:
        checks = {"count": 0}

        def cancel_after_npz_stage():
            checks["count"] += 1
            return checks["count"] == 5

        with self.assertRaises(CheckpointCancelled):
            self.publish(cancel=cancel_after_npz_stage)
        self.assertEqual(list(self.root.glob("*.checkpoint")), [])
        self.assertEqual(list(self.root.glob(".*")), [])

    def test_concurrent_same_content_winner_leaves_no_losing_stage(self) -> None:
        original_replace = os.replace
        state = {"nested_publish": False}
        identity = _identity()
        arrays = {"mask": np.eye(7, 9, dtype=np.bool_), "labels": np.arange(63, dtype=np.int32).reshape(7, 9)}

        def race_replace(source, destination):
            if Path(destination).name.endswith(".checkpoint") and not state["nested_publish"]:
                state["nested_publish"] = True
                publish_checkpoint(
                    self.root,
                    identity,
                    arrays,
                    execution=ExecutionRecord(ExecutionStatus.AVAILABLE, elapsed_ms=1.25, transfer_ms=0.2),
                    measurement=_measurement(),
                    raster_roles={"mask": RasterRole("mask", np.dtype(np.bool_).str), "labels": RasterRole("instances", np.dtype(np.int32).str)},
                    allow_cpu_transfer=True,
                )
            return original_replace(source, destination)

        with patch("seedvision.benchmarking.checkpoints.os.replace", side_effect=race_replace):
            published = self.publish()
        self.assertTrue(published.path.is_dir())
        self.assertEqual(list(self.root.glob(".*")), [])

    def test_exact_array_size_limit_round_trips_with_bounded_header_slack(self) -> None:
        limits = CheckpointLimits(
            max_array_bytes=64,
            max_total_array_bytes=64,
            max_npz_bytes=1024,
        )
        reference = publish_checkpoint(
            self.root,
            _identity(),
            {"mask": np.ones((8, 8), dtype=np.uint8)},
            execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
            measurement=_measurement(),
            allow_cpu_transfer=True,
            limits=limits,
        )

        loaded = load_checkpoint(self.root, reference.checkpoint_sha256, limits=limits)
        self.assertEqual(loaded.arrays["mask"].nbytes, 64)

    def test_forged_huge_npy_shape_is_rejected_before_np_load(self) -> None:
        reference = self.publish()
        archive_path = reference.path / "arrays.npz"
        with zipfile.ZipFile(archive_path, "r") as archive:
            original = {name: archive.read(name) for name in archive.namelist()}
        forged_header = BytesIO()
        np.lib.format.write_array_header_1_0(
            forged_header,
            {
                "descr": np.dtype(np.bool_).str,
                "fortran_order": False,
                "shape": (10**12, 9),
            },
        )
        original["mask.npy"] = forged_header.getvalue()
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, payload in original.items():
                archive.writestr(name, payload)

        with self.assertRaisesRegex(InvalidCheckpoint, "header differs"):
            load_checkpoint(self.root, reference.checkpoint_sha256)

    def test_archive_path_replacement_after_preflight_cannot_change_deserialized_snapshot(self) -> None:
        reference = self.publish()
        original_load = np.load
        replaced = {"done": False}

        def replace_archive_after_preflight(file, *args, **kwargs):
            if isinstance(file, BytesIO) and not replaced["done"]:
                replaced["done"] = True
                (reference.path / "arrays.npz").write_bytes(b"replacement after preflight")
            return original_load(file, *args, **kwargs)

        with patch("seedvision.benchmarking.checkpoints.np.load", side_effect=replace_archive_after_preflight):
            loaded = load_checkpoint(self.root, reference.checkpoint_sha256)
        self.assertTrue(replaced["done"])
        self.assertEqual(loaded.arrays["labels"].shape, (7, 9))

    def test_adversarial_duplicate_unsafe_object_and_high_ratio_archives_are_rejected(self) -> None:
        # Duplicate and traversal-like names are rejected before NumPy opens a member.
        for archive_case in ("duplicate", "unsafe", "object"):
            with self.subTest(archive_case=archive_case):
                reference = self.publish()
                archive_path = reference.path / "arrays.npz"
                with zipfile.ZipFile(archive_path, "r") as original:
                    contents = {name: original.read(name) for name in original.namelist()}
                if archive_case == "duplicate":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                            archive.writestr("mask.npy", contents["mask.npy"])
                            archive.writestr("mask.npy", contents["mask.npy"])
                            archive.writestr("labels.npy", contents["labels.npy"])
                elif archive_case == "unsafe":
                    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                        archive.writestr("../mask.npy", contents["mask.npy"])
                        archive.writestr("labels.npy", contents["labels.npy"])
                else:
                    forged_object = BytesIO()
                    np.lib.format.write_array(
                        forged_object,
                        np.ones((7, 9), dtype=object),
                        allow_pickle=True,
                    )
                    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                        archive.writestr("mask.npy", forged_object.getvalue())
                        archive.writestr("labels.npy", contents["labels.npy"])
                with self.assertRaises(InvalidCheckpoint):
                    load_checkpoint(self.root, reference.checkpoint_sha256)
                for child in reference.path.iterdir():
                    child.unlink()
                reference.path.rmdir()

        reference = publish_checkpoint(
            self.root,
            _identity(),
            {"bits": np.zeros((1024 * 1024,), dtype=np.uint8)},
            execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
            measurement=_measurement(),
            allow_cpu_transfer=True,
        )
        with (reference.path / "arrays.npz").open("wb") as stream:
            np.savez_compressed(stream, bits=np.zeros((1024 * 1024,), dtype=np.uint8))
        with self.assertRaisesRegex(InvalidCheckpoint, "compression ratio"):
            load_checkpoint(self.root, reference.checkpoint_sha256)

    def test_checkpoint_metadata_accepts_only_strict_json_primitives(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON primitive"):
            self.publish(metadata={"bad": object()})
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.publish(metadata={"bad": float("nan")})


if __name__ == "__main__":
    unittest.main()
