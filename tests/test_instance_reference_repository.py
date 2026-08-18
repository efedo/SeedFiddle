from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "seed-instance-references"
EXPECTED_IMAGE_NAMES = (
    "IMG_0002c.JPG",
    "IMG_9632c.JPG",
    "IMG_9636c.JPG",
    "IMG_9641c.JPG",
    "IMG_9666c.JPG",
    "IMG_9667c.JPG",
    "IMG_9668c.JPG",
    "IMG_9670c.JPG",
    "IMG_9685c.JPG",
    "IMG_9689c.JPG",
    "IMG_9974c.JPG",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_file(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise AssertionError(f"{field} must be relative: {value}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise AssertionError(f"{field} escapes {root}: {value}") from error
    if not resolved.is_file():
        raise AssertionError(f"{field} does not exist: {resolved}")
    return resolved


def _disconnected_identifiers(labels: np.ndarray) -> set[int]:
    """Return IDs with more than one exact categorical 8-component.

    The scan creates one union-find node per horizontal nonzero run.  Adjacent
    rows are joined when same-ID runs overlap or touch diagonally.  This avoids
    allocating a full-size boolean raster for every instance in a dense mask.
    """

    parents: list[int] = []
    ranks: list[int] = []
    node_labels: list[int] = []

    def create_node(identifier: int) -> int:
        node = len(parents)
        parents.append(node)
        ranks.append(0)
        node_labels.append(identifier)
        return node

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if ranks[first_root] < ranks[second_root]:
            first_root, second_root = second_root, first_root
        parents[second_root] = first_root
        if ranks[first_root] == ranks[second_root]:
            ranks[first_root] += 1

    # Runs use half-open x extents.  For 8-connectivity, [a, b) and [c, d)
    # on adjacent rows touch whenever c <= b and d >= a; equality represents
    # a one-pixel diagonal contact.
    previous: list[tuple[int, int, int, int]] = []
    for row in labels:
        boundaries = np.flatnonzero(
            np.concatenate(
                (
                    np.asarray((True,)),
                    row[1:] != row[:-1],
                    np.asarray((True,)),
                )
            )
        )
        current: list[tuple[int, int, int, int]] = []
        previous_start = 0
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            identifier = int(row[int(start)])
            if identifier == 0:
                continue
            node = create_node(identifier)
            start_int = int(start)
            end_int = int(end)
            while (
                previous_start < len(previous)
                and previous[previous_start][1] < start_int
            ):
                previous_start += 1
            candidate = previous_start
            while candidate < len(previous) and previous[candidate][0] <= end_int:
                old_start, old_end, old_identifier, old_node = previous[candidate]
                if old_end >= start_int and old_identifier == identifier:
                    union(node, old_node)
                candidate += 1
            current.append((start_int, end_int, identifier, node))
        previous = current

    components: dict[int, set[int]] = {}
    for node, identifier in enumerate(node_labels):
        components.setdefault(identifier, set()).add(find(node))
    return {
        identifier
        for identifier, roots in components.items()
        if len(roots) != 1
    }


class InstanceReferenceRepositoryTests(unittest.TestCase):
    def test_manifest_and_every_reference_mask_are_self_consistent(self) -> None:
        manifest_path = REFERENCE_ROOT / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("version"), 1)
        entries = payload.get("entries")
        self.assertIsInstance(entries, list)
        assert isinstance(entries, list)

        expected_paths = {f"images/{name}" for name in EXPECTED_IMAGE_NAMES}
        actual_paths = {
            entry.get("image_path")
            for entry in entries
            if isinstance(entry, dict)
        }
        self.assertEqual(actual_paths, expected_paths)
        self.assertEqual(len(entries), len(expected_paths))
        expected_mask_names = {
            f"{Path(name).stem}.instances.png" for name in EXPECTED_IMAGE_NAMES
        }
        actual_mask_names = {
            entry.get("mask_path")
            for entry in entries
            if isinstance(entry, dict)
        }
        self.assertEqual(actual_mask_names, expected_mask_names)
        self.assertEqual(
            {path.name for path in REFERENCE_ROOT.glob("IMG_*.instances.png")},
            expected_mask_names,
        )

        for entry_value in entries:
            self.assertIsInstance(entry_value, dict)
            assert isinstance(entry_value, dict)
            image_path_text = entry_value.get("image_path")
            with self.subTest(image=image_path_text):
                image_path = _relative_file(
                    PROJECT_ROOT, image_path_text, "image_path"
                )
                mask_path = _relative_file(
                    REFERENCE_ROOT, entry_value.get("mask_path"), "mask_path"
                )
                self.assertEqual(mask_path.parent, REFERENCE_ROOT.resolve())
                self.assertEqual(entry_value.get("coordinate_space"), "source")
                self.assertIs(entry_value.get("reviewed"), False)
                self.assertIsInstance(entry_value.get("notes"), str)
                self.assertTrue(entry_value.get("notes"))
                self.assertEqual(entry_value.get("external_reference_count"), 2)
                self.assertEqual(
                    entry_value.get("image_sha256"), _sha256(image_path)
                )
                self.assertEqual(
                    entry_value.get("mask_sha256"), _sha256(mask_path)
                )

                source = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                self.assertIsNotNone(source)
                assert source is not None
                expected_shape = (
                    int(entry_value.get("height", -1)),
                    int(entry_value.get("width", -1)),
                )
                self.assertEqual(source.shape, expected_shape)
                del source

                labels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                self.assertIsNotNone(labels)
                assert labels is not None
                self.assertEqual(labels.ndim, 2)
                self.assertEqual(labels.dtype, np.uint16)
                self.assertEqual(labels.shape, expected_shape)
                identifiers = np.unique(labels)
                self.assertGreater(len(identifiers), 1)
                self.assertEqual(int(identifiers[0]), 0)
                instance_count = int(entry_value.get("instance_count", -1))
                self.assertTrue(
                    np.array_equal(
                        identifiers[1:],
                        np.arange(1, instance_count + 1, dtype=identifiers.dtype),
                    ),
                    "nonzero instance IDs must be contiguous from 1",
                )
                dish_count = int(entry_value.get("dish_instance_count", -1))
                external_count = int(
                    entry_value.get("external_reference_count", -1)
                )
                self.assertGreater(dish_count, 0)
                self.assertEqual(dish_count + external_count, instance_count)
                disconnected = _disconnected_identifiers(labels)
                self.assertEqual(
                    disconnected,
                    set(),
                    f"IDs with more than one 8-connected component: "
                    f"{sorted(disconnected)}",
                )
                del labels


if __name__ == "__main__":
    unittest.main()
