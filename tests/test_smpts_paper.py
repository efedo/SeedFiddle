
from tests.functional_units import functional_units
from collections.abc import Mapping
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.cuda import smpts_paper as paper
from tests.test_plantcv_watershed import _REFERENCE


def canonical_components(mask):
    count, raw, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    first = sorted((int(np.flatnonzero(raw == i)[0]), i) for i in range(1, count))
    lookup = np.zeros(count, np.int32)
    for i, (_, old) in enumerate(first, 1):
        lookup[old] = i
    labels = lookup[raw]
    rows = []
    for i in range(1, count):
        yy, xx = np.nonzero(labels == i)
        rows.append({'label_id': i, 'area_pixels': len(xx),
            'bounds_xyxy': (int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1)})
    return labels, tuple(rows)


def contact_fixture():
    mask = np.zeros((110, 120), np.uint8)
    mask[2:5, 2:7] = 255  # Box15: unchanged.
    mask[2:6, 15:20] = 255  # Box20: equality is unchanged.
    mask[15:18, 2:42] = 255  # Box120: selected and erased.
    mask[40:67, 5:32] = 255
    mask[40:67, 50:77] = 255
    mask[52:55, 32:50] = 255  # Thin bridge; selected parent splits into two.
    yy, xx = np.indices(mask.shape)
    rgb = np.stack((xx % 256, yy % 256, (xx+3*yy) % 256), axis=-1).astype(np.uint8)
    return mask, rgb


class PaperPlanningTests(unittest.TestCase):
    def test_full_fixture_admitted_and_work_cuda_limits_are_explicit(self):
        pixels = 2080*3120
        self.assertLess(paper.component_work(pixels), paper.MAX_COMPONENT_WORK)
        self.assertEqual(paper.retained_bound(pixels), 64*pixels+192*1024*1024)
        self.assertLess(paper.retained_bound(pixels), paper.MAX_CUDA_BYTES)
        for value in (0, True, 64_000_001):
            with self.assertRaises(ValueError):
                paper.component_work(value)
        with self.assertRaises(paper.PaperResourceError):
            paper.retained_bound(64_000_000)

    def test_inventory_and_contact_controls_are_exact(self):
        rows = ({'label_id': 1, 'area_pixels': 6, 'bounds_xyxy': (0, 0, 2, 3)},)
        self.assertEqual(paper.select_contacts(rows, (3, 2), 5), (1,))
        self.assertEqual(paper.select_contacts(rows, (3, 2), 6), ())
        for threshold in (True, -1, 1.0):
            with self.assertRaises(ValueError):
                paper.select_contacts(rows, (3, 2), threshold)
        for bad in ((dict(rows[0], label_id=2),), (dict(rows[0], bounds_xyxy=(0, 0, 3, 3)),),
                (dict(rows[0], area_pixels=7),), [rows[0]]):
            with self.assertRaises(ValueError):
                paper.select_contacts(bad, (3, 2), 5)
        self.assertLessEqual(paper.metadata_bound(4096), paper.MAX_CPU_METADATA_BYTES)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class PaperCudaTests(unittest.TestCase):
    def gpu(self, value):
        return torch.as_tensor(value, device='cuda')

    def assert_gpu(self, actual, expected):
        self.assertEqual(actual.device.type, 'cuda')
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)

    def check_test_components_match_opencv8_on_islands_diagonals_chains_and_random_masks(self):
        rng = np.random.default_rng(3847)
        cases = [np.zeros((1, 1), np.uint8), np.full((1, 4097), 255, np.uint8),
            np.eye(97, dtype=np.uint8)*255, np.full((83, 41), 255, np.uint8)]
        for shape in ((1, 63), (71, 1), (3, 7), (7, 3), (37, 61), (111, 117)):
            for probability in (.1, .5, .9):
                cases.append((rng.random(shape) < probability).astype(np.uint8)*255)
        hole = np.full((33, 35), 255, np.uint8)
        hole[3:29, 4:30] = 0
        hole[10:15, 12:17] = 255
        cases.append(hole)
        for mask in cases:
            with self.subTest(shape=mask.shape, foreground=int(np.count_nonzero(mask))):
                expected, rows = canonical_components(mask)
                actual = paper.components(self.gpu(mask))
                self.assert_gpu(actual['labels'], expected)
                self.assertEqual(actual['components'], rows)
                self.assertTrue(all(isinstance(r, Mapping) for r in actual['components']))
                if rows:
                    with self.assertRaises(TypeError):
                        actual['components'][0]['area_pixels'] = 0

    def check_test_binary_median_and_intersection_exact_with_replicated_borders(self):
        rng = np.random.default_rng(98)
        for shape in ((1, 23), (23, 1), (13, 17)):
            mask = (rng.random(shape) < .5).astype(np.uint8)*255
            gpu = self.gpu(mask)
            for size in (1, 3, 5, 13, 31):
                self.assert_gpu(paper.binary_median(gpu, size), cv2.medianBlur(mask, size))
            reverse = np.ascontiguousarray(mask[::-1, ::-1])
            self.assert_gpu(paper.intersection(gpu, self.gpu(reverse)), cv2.bitwise_and(mask, reverse))
        bad = self.gpu(np.array([[0, 127], [255, 0]], np.uint8))
        for call in (lambda: paper.binary_median(bad), lambda: paper.intersection(bad, bad),
                lambda: paper.components(bad)):
            with self.assertRaises(ValueError):
                call()

    def check_test_targeted_erosion_split_disappearance_and_packed_original_rgb(self):
        mask, rgb = contact_fixture()
        source, rows = canonical_components(mask)
        selected = tuple(r['label_id'] for r in rows
            if (r['bounds_xyxy'][2]-r['bounds_xyxy'][0])*(r['bounds_xyxy'][3]-r['bounds_xyxy'][1]) > 20)
        target = np.isin(source, selected).astype(np.uint8)*255
        eroded = cv2.erode(target, np.ones((13, 13), np.uint8), borderType=cv2.BORDER_REPLICATE)
        processed = np.where(np.isin(source, selected), eroded, mask).astype(np.uint8)
        final, final_rows = canonical_components(processed)
        actual = paper.extract_from_intersection(self.gpu(mask), self.gpu(rgb), 20, 13)
        for key, expected in (('source_labels', source), ('selected_mask', target),
                ('eroded_selected_mask', eroded), ('mask', processed), ('final_labels', final)):
            self.assert_gpu(actual[key], expected)
        self.assertEqual(actual['source_components'], rows)
        self.assertEqual(actual['final_components'], final_rows)
        self.assertEqual(actual['selected_parent_ids'], (3, 4))
        self.assertEqual(actual['disappeared_parent_ids'], (3,))
        self.assertEqual([r['parent_id'] for r in actual['crops']], [1, 2, 4, 4])
        offset = 0
        for row in actual['crops']:
            x0, y0, x1, y1 = row['bounds_xyxy']
            local_mask = (final[y0:y1, x0:x1] == row['label_id']).astype(np.uint8)*255
            local_rgb = np.where(local_mask[..., None] == 255, rgb[y0:y1, x0:x1], 0)
            size = (x1-x0)*(y1-y0)
            self.assertEqual(row['offset_pixels'], offset)
            self.assertEqual(row['crop_shape'], (y1-y0, x1-x0))
            self.assert_gpu(actual['packed_mask'][offset:offset+size].reshape(local_mask.shape), local_mask)
            self.assert_gpu(actual['packed_rgb'][3*offset:3*(offset+size)].reshape(local_rgb.shape), local_rgb)
            offset += size
        self.assertEqual(actual['packed_rgb'].numel(), 3*offset)
        self.assertEqual(actual['packed_mask'].numel(), offset)
        self.assert_gpu(actual['mask'][2:6, :20], mask[2:6, :20])

    def check_test_replicated_erosion_frame_edges_and_selected_window_controls(self):
        for shape in ((1, 19), (21, 1), (19, 23)):
            mask = np.full(shape, 255, np.uint8)
            mask[-1, -1] = 0
            source = paper.components(self.gpu(mask))
            for size in (1, 3, 13, 31):
                actual = paper.targeted_erosion(self.gpu(mask), source['labels'],
                    source['components'], (1,), size)
                expected = cv2.erode(mask, np.ones((size, size), np.uint8), borderType=cv2.BORDER_REPLICATE)
                self.assert_gpu(actual['mask'], expected)

    def check_test_component_limit_and_failed_convergence_publish_nothing(self):
        mask = np.zeros((127, 127), np.uint8)
        mask[::2, ::2] = 255
        actual = paper.components(self.gpu(mask))
        self.assertEqual(len(actual['components']), 4096)
        mask = np.zeros((129, 129), np.uint8)
        mask[::2, ::2] = 255
        with patch.object(paper, '_statistics') as stats:
            with self.assertRaises(paper.PaperResourceError):
                paper.components(self.gpu(mask))
            stats.assert_not_called()
        with patch.object(paper, 'UNION_ROUNDS', 0):
            with self.assertRaises(paper.PaperResourceError):
                paper.components(self.gpu(np.full((3, 3), 255, np.uint8)))

    def check_test_stalled_hooks_fail_verification_and_alternate_stream_tiles_are_stable(self):
        mask = self.gpu(np.full((19, 23), 255, np.uint8))
        real = paper._program(mask)
        class Stalled:
            def launch(self, *args):
                return real.launch(*args)
            def launch_1d(self, name, *args):
                if name != 'paper_hook':
                    return real.launch_1d(name, *args)
        with patch.object(paper, '_program', return_value=Stalled()):
            with self.assertRaises(paper.PaperResourceError):
                paper.components(mask)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), patch.object(paper, 'TILE', 7):
            actual = paper.components(mask)
        stream.synchronize()
        self.assert_gpu(actual['labels'], np.ones((19, 23), np.int32))

    def check_test_every_component_workspace_is_guarded_before_callbacks(self):
        original = paper._Guard.watch
        for target in range(9):
            watched = []
            mask = self.gpu(np.full((7, 9), 255, np.uint8))
            def watch(guard, *tensors):
                original(guard, *tensors)
                watched.extend(tensors)
            def mutate():
                if len(watched) > target:
                    watched[target].add_(1)
                return False
            with patch.object(paper._Guard, 'watch', watch):
                with self.assertRaises(paper.PaperMutationError):
                    paper.components(mask, cancel=mutate)

    def check_test_overlapping_component_boxes_pack_only_their_own_pixels(self):
        mask = np.zeros((35, 37), np.uint8)
        mask[1:34, 1:36] = 255
        mask[3:32, 3:34] = 0
        mask[12:23, 13:24] = 255
        rgb = np.full((*mask.shape, 3), (33, 81, 119), np.uint8)
        actual = paper.extract_from_intersection(self.gpu(mask), self.gpu(rgb), contact_box_area=100_000)
        self.assertEqual(len(actual['crops']), 2)
        self.assertEqual([r['parent_id'] for r in actual['crops']], [1, 2])
        for row in actual['crops']:
            x0, y0, x1, y1 = row['bounds_xyxy']
            count = (x1-x0)*(y1-y0)
            offset = row['offset_pixels']
            expected = (actual['final_labels'][y0:y1, x0:x1] == row['label_id']).to(torch.uint8)*255
            self.assertTrue(torch.equal(actual['packed_mask'][offset:offset+count].reshape(expected.shape), expected))

    def check_test_empty_extraction_and_compact_crop_limit(self):
        mask = np.zeros((9, 11), np.uint8)
        rgb = np.full((9, 11, 3), 121, np.uint8)
        actual = paper.extract_from_intersection(self.gpu(mask), self.gpu(rgb))
        self.assertEqual(actual['crops'], ())
        self.assertEqual(actual['disappeared_parent_ids'], ())
        self.assertEqual(tuple(actual['packed_rgb'].shape), (0,))
        mask[:] = 255
        with patch.object(paper, 'MAX_CROP_BYTES', 100):
            with self.assertRaises(paper.PaperResourceError):
                paper.extract_from_intersection(self.gpu(mask), self.gpu(rgb), 1000)

    def check_test_cancellation_mutation_and_malformed_partition_withhold_results(self):
        mask, rgb = contact_fixture()
        gpu, image = self.gpu(mask), self.gpu(rgb)
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls == 12
        with self.assertRaises(paper.PaperCancelled):
            paper.extract_from_intersection(gpu, image, cancel=cancel)
        calls = 0
        def mutate():
            nonlocal calls
            calls += 1
            if calls == 8:
                image[0, 0, 0] = 11
            return False
        with self.assertRaises(paper.PaperMutationError):
            paper.extract_from_intersection(gpu, image, cancel=mutate)
        source = paper.components(gpu)
        bad = source['labels'].clone()
        bad[0, 0] = -1
        with self.assertRaises(ValueError):
            paper.targeted_erosion(gpu, bad, source['components'], (3,), 13)
        with self.assertRaises(TypeError):
            paper.components(gpu, cancel=lambda: 1)

    def check_test_complete_declared_variant_matches_isolated_pinned_opencv_substeps(self):
        mask, rgb = contact_fixture()
        gray = np.where(mask, 220, 10).astype(np.uint8)
        script = r'''
from importlib.metadata import version
import sys
import cv2
import numpy as np
assert version('opencv-python') == '4.11.0.86'
assert version('numpy') == '2.2.6'
with np.load(sys.argv[1], allow_pickle=False) as data:
    gray = data['gray']
a = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)
m = cv2.medianBlur(a, 5)
_, g = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
b = cv2.bitwise_and(m, g)
count, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
selected = [i for i in range(1, count) if int(stats[i, 2])*int(stats[i, 3]) > 20]
target = np.isin(labels, selected).astype(np.uint8)*255
eroded = cv2.erode(target, np.ones((13, 13), np.uint8), borderType=cv2.BORDER_REPLICATE)
final = np.where(np.isin(labels, selected), eroded, b).astype(np.uint8)
np.savez(sys.argv[2], adaptive_mask=a, median_mask=m, global_mask=g,
    intersection_mask=b, selected_mask=target, eroded_selected_mask=eroded, mask=final)
'''
        with tempfile.TemporaryDirectory() as temporary:
            source, expected = Path(temporary)/'source.npz', Path(temporary)/'expected.npz'
            np.savez(source, gray=gray)
            run = subprocess.run([str(_REFERENCE), '-c', script, str(source), str(expected)],
                capture_output=True, text=True, timeout=90)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            actual = paper.reconstruct(self.gpu(gray), self.gpu(rgb), contact_box_area=20)
            with np.load(expected, allow_pickle=False) as values:
                for key in values.files:
                    self.assert_gpu(actual[key], values[key])
                labels, rows = canonical_components(values['mask'])
                self.assert_gpu(actual['final_labels'], labels)
                self.assertEqual(actual['final_components'], rows)

    @unittest.skipUnless(_REFERENCE.is_file(), 'Pinned OpenCV reference Python required')
    def test_complete_declared_variant_matches_isolated_pinned_opencv_substeps(self):
        self.scenario(self.check_test_complete_declared_variant_matches_isolated_pinned_opencv_substeps, labelled=False)

    def test_workflow_lifecycle(self):
        """6 related PaperCudaTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_components_match_opencv8_on_islands_diagonals_chains_and_random_masks)
        self.scenario(self.check_test_binary_median_and_intersection_exact_with_replicated_borders)
        self.scenario(self.check_test_targeted_erosion_split_disappearance_and_packed_original_rgb)
        self.scenario(self.check_test_replicated_erosion_frame_edges_and_selected_window_controls)
        self.scenario(self.check_test_every_component_workspace_is_guarded_before_callbacks)
        self.scenario(self.check_test_overlapping_component_boxes_pack_only_their_own_pixels)

    def test_validation_and_recovery(self):
        """4 related PaperCudaTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_component_limit_and_failed_convergence_publish_nothing)
        self.scenario(self.check_test_stalled_hooks_fail_verification_and_alternate_stream_tiles_are_stable)
        self.scenario(self.check_test_empty_extraction_and_compact_crop_limit)
        self.scenario(self.check_test_cancellation_mutation_and_malformed_partition_withhold_results)


if __name__ == '__main__':
    unittest.main()
