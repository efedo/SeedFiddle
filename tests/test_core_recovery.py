
from tests.functional_units import functional_units
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.cuda import core_recovery as recovery
from tests.test_plantcv_watershed import _REFERENCE, _PINS_SCRIPT


_ORACLE = _PINS_SCRIPT + r'''
import sys
import numpy as np
from skimage.segmentation import watershed
with np.load(sys.argv[1], allow_pickle=False) as values:
    results = {}
    for i in range(int(sys.argv[3])):
        domain = (values[f'f{i}'] == 255) & (values[f'v{i}'] == 255)
        results[f'l{i}'] = watershed(values[f'e{i}'], values[f'm{i}'], mask=domain)
    np.savez(sys.argv[2], **results)
'''


def _pair(shape=(19, 23)):
    fg = np.zeros(shape, dtype=np.uint8)
    fg[2:-2, 2:-2] = 255
    valid = np.full(shape, 255, dtype=np.uint8)
    markers = np.zeros(shape, dtype=np.int32)
    markers[shape[0]//2, 4] = 7
    markers[shape[0]//2, -5] = 2**31-1
    elevation = np.zeros(shape, dtype=np.float32)
    return fg, markers, valid, elevation


def _cuda(values):
    return tuple(torch.as_tensor(value, device='cuda') for value in values)


def _interface(labels):
    band = np.zeros(labels.shape, dtype=bool)
    for a, b, target_a, target_b in (
        (labels[:-1], labels[1:], band[:-1], band[1:]),
        (labels[:, :-1], labels[:, 1:], band[:, :-1], band[:, 1:])):
        differing = (a > 0) & (b > 0) & (a != b)
        target_a |= differing
        target_b |= differing
    return band


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for core recovery')
@functional_units
class CoreRecoveryTests(unittest.TestCase):
    def reference(self, cases):
        self.assertTrue(_REFERENCE.is_file(), 'Pinned source interpreter is required.')
        arrays = {f'{name}{i}': array for i, case in enumerate(cases)
                  for name, array in zip(('f', 'm', 'v', 'e'), case)}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            np.savez(folder/'input.npz', **arrays)
            completed = subprocess.run([str(_REFERENCE), '-c', _ORACLE,
                str(folder/'input.npz'), str(folder/'output.npz'), str(len(cases))],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with np.load(folder/'output.npz', allow_pickle=False) as values:
                return [values[f'l{i}'].copy() for i in range(len(cases))]

    def assert_products(self, partition, case, expected):
        fg, markers, valid, _ = case
        domain = (fg == 255) & (valid == 255)
        np.testing.assert_array_equal(partition.labels.cpu().numpy(), expected)
        np.testing.assert_array_equal(partition.domain.cpu().numpy(), domain)
        np.testing.assert_array_equal(partition.unresolved.cpu().numpy(), domain & (expected == 0))
        interface = _interface(expected)
        np.testing.assert_array_equal(partition.interface.cpu().numpy(), interface)
        ids = sorted(int(x) for x in np.unique(markers) if x > 0)
        self.assertEqual(partition.label_ids.cpu().tolist(), ids)
        self.assertEqual([o.label_id for o in partition.objects], ids)
        self.assertEqual(partition.unresolved_pixels, int(np.count_nonzero(domain & (expected == 0))))
        self.assertEqual(partition.interface_pixels, int(interface.sum()))
        self.assertEqual(sum(o.assigned_pixels for o in partition.objects)+partition.unresolved_pixels,
                         int(domain.sum()))
        self.assertEqual(partition.plan.validity_clipped_foreground_pixels,
                         int(np.count_nonzero(fg == 255))-int(domain.sum()))
        for obj in partition.objects:
            selected = expected == obj.label_id
            yy, xx = np.nonzero(selected)
            self.assertEqual(obj.seed_pixels, int(np.count_nonzero(markers == obj.label_id)))
            self.assertEqual(obj.assigned_pixels, int(selected.sum()))
            self.assertEqual(obj.interface_pixels, int(np.count_nonzero(selected & interface)))
            self.assertEqual(obj.bbox, (int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1))
        np.testing.assert_array_equal(partition.labels.cpu().numpy()[markers > 0], markers[markers > 0])


    def check_test_isolated_recovery_preserves_pre_erosion_contour_and_unseeded_component(self):
        fg = np.zeros((31, 43), dtype=np.uint8)
        yy, xx = np.indices(fg.shape)
        isolated = (yy-15)**2+(xx-13)**2 <= 81
        unseeded = (yy-15)**2+(xx-35)**2 <= 16
        fg[isolated | unseeded] = 255
        markers = np.zeros_like(fg, dtype=np.int32)
        core = (yy-15)**2+(xx-13)**2 <= 16
        markers[core] = 901
        case = fg, markers, np.full_like(fg, 255), np.zeros(fg.shape, np.float32)
        partition = recovery.recover_core_labels(*_cuda(case))
        np.testing.assert_array_equal(partition.labels.cpu().numpy() == 901, isolated)
        np.testing.assert_array_equal(partition.unresolved.cpu().numpy(), unseeded)
        self.assertGreater(partition.objects[0].assigned_pixels, partition.objects[0].seed_pixels)
        self.assertEqual(partition.status, 'unresolved_foreground')
        annotation = recovery.annotate_core_recovery(partition)
        self.assertTrue(annotation.objects[0].segmentation_measurement_allowed)
        self.assertTrue(annotation.objects[0].uncontested_visible_partition)
        self.assertFalse(annotation.objects[0].scientific_validation)

    def check_test_radius_and_policy_change_annotation_reuse_labels_and_match_clipped_dilation(self):
        case = _pair()
        partition = recovery.recover_core_labels(*_cuda(case))
        labels = partition.labels.clone()
        expected = _interface(labels.cpu().numpy())
        domain = partition.domain.cpu().numpy().astype(bool)
        for radius in range(4):
            with patch.object(recovery, '_flood', side_effect=AssertionError('partition must be cached')):
                annotation = recovery.annotate_core_recovery(partition, uncertainty_radius=radius)
                flagged = recovery.annotate_core_recovery(partition, uncertainty_radius=radius,
                                                          measurement_policy='flag_uncertain')
            self.assertIs(annotation.partition, partition)
            np.testing.assert_array_equal(annotation.uncertainty_band.cpu().numpy(), expected)
            np.testing.assert_array_equal(flagged.uncertainty_band.cpu().numpy(), expected)
            for obj in annotation.objects:
                assigned = labels.cpu().numpy() == obj.label_id
                self.assertEqual(obj.uncertainty_pixels, int(np.count_nonzero(expected & assigned)))
                self.assertFalse(obj.segmentation_measurement_allowed)
                self.assertFalse(obj.uncontested_visible_partition)
                self.assertFalse(obj.scientific_validation)
            self.assertTrue(all(o.segmentation_measurement_allowed for o in flagged.objects))
            self.assertTrue(all(not o.uncontested_visible_partition for o in flagged.objects))
            expanded = expected.copy()
            expanded[1:] |= expected[:-1]
            expanded[:-1] |= expected[1:]
            expanded[:, 1:] |= expected[:, :-1]
            expanded[:, :-1] |= expected[:, 1:]
            expected = expanded & domain
        torch.testing.assert_close(partition.labels, labels)

    def check_test_canvas_invalid_support_and_empty_statuses_remain_explicit(self):
        fg = np.full((7, 9), 255, np.uint8)
        valid = fg.copy()
        valid[:, 4] = 0
        markers = np.zeros(fg.shape, np.int32)
        markers[3, 2] = 5
        partition = recovery.recover_core_labels(*_cuda((fg, markers, valid, np.zeros(fg.shape, np.float32))))
        obj = partition.objects[0]
        self.assertTrue(obj.touches_canvas)
        self.assertTrue(obj.touches_invalid_support)
        self.assertFalse(obj.touches_unresolved)
        self.assertEqual(partition.unresolved_pixels, 7*4)
        annotation = recovery.annotate_core_recovery(partition)
        self.assertEqual(annotation.objects[0].reasons, ('canvas_boundary', 'invalid_support'))
        self.assertFalse(annotation.objects[0].segmentation_measurement_allowed)
        for foreground, validity, status in ((np.zeros_like(fg), fg, 'empty_foreground'),
                (fg, np.zeros_like(fg), 'all_foreground_excluded_by_validity'),
                (fg, fg, 'unresolved_foreground')):
            result = recovery.recover_core_labels(*_cuda((foreground, np.zeros_like(markers),
                validity, np.zeros(fg.shape, np.float32))))
            self.assertEqual(result.status, status)
            self.assertEqual(result.objects, ())

    def check_test_invalid_rasters_fail_before_partition_and_static_limits_before_compilation(self):
        for index, value in ((0, 1), (2, 1), (1, -1), (3, float('nan')), (3, float('inf'))):
            case = list(_pair())
            case[index] = case[index].copy()
            case[index][0, 0] = value
            with self.subTest(index=index, value=value), patch.object(recovery, '_flood') as flood:
                with self.assertRaises(ValueError):
                    recovery.recover_core_labels(*_cuda(case))
                flood.assert_not_called()
        case = list(_pair())
        case[1][0, 0] = 1
        with self.assertRaises(ValueError):
            recovery.recover_core_labels(*_cuda(case))
        case = list(_pair())
        case[2][9, 4] = 0
        with self.assertRaises(ValueError):
            recovery.recover_core_labels(*_cuda(case))
        inputs = _cuda(_pair())
        for name in ('MAX_RASTER_PIXELS', 'MAX_LIVE_BYTES', 'MAX_WORK', 'MAX_CPU_METADATA_BYTES'):
            with self.subTest(limit=name), patch.object(recovery, name, 1), \
                    patch.object(recovery, 'fixed_kernel_program') as compile:
                with self.assertRaises(recovery.CoreRecoveryResourceError):
                    recovery.recover_core_labels(*inputs)
                compile.assert_not_called()
        for values in ((inputs[0].cpu(),)+inputs[1:],
                (inputs[0], inputs[1].to(torch.int64), inputs[2], inputs[3]),
                tuple(x.t() for x in inputs), (inputs[0][:0],)+tuple(x[:0] for x in inputs[1:])):
            with self.assertRaises(ValueError):
                recovery.recover_core_labels(*values)



    def check_test_all_inputs_and_owned_workspace_are_guarded_across_caller_callbacks(self):
        original_watch = recovery._Guard.watch
        # Four inputs, table/state, compact/sorted IDs, three domain products,
        # watershed output/three queue arrays/state, then summary stats/state.
        for target in range(18):
            inputs, watched, fired = _cuda(_pair()), [], False
            def watch(guard, *tensors):
                original_watch(guard, *tensors)
                watched.extend(tensors)
            def cancel():
                nonlocal fired
                if len(watched) > target:
                    watched[target].add_(1)
                    fired = True
                return False
            with self.subTest(target=target), patch.object(recovery._Guard, 'watch', watch):
                with self.assertRaises(recovery.CoreRecoveryMutationError):
                    recovery.recover_core_labels(*inputs, cancel=cancel)
            self.assertTrue(fired)
        inputs = _cuda(_pair())
        def raises_after_mutating():
            inputs[0].zero_()
            raise RuntimeError('caller error')
        with self.assertRaises(recovery.CoreRecoveryMutationError):
            recovery.recover_core_labels(*inputs, cancel=raises_after_mutating)

    def check_test_cancellation_return_type_and_final_checkpoint_withhold_publication(self):
        inputs = _cuda(_pair())
        calls = 0
        def count():
            nonlocal calls
            calls += 1
            return False
        recovery.recover_core_labels(*inputs, cancel=count)
        final, calls = calls, 0
        def last():
            nonlocal calls
            calls += 1
            return calls == final
        with self.assertRaises(recovery.CoreRecoveryCancelled):
            recovery.recover_core_labels(*inputs, cancel=last)
        for value in (None, 0, np.bool_(False), 'no'):
            with self.subTest(value=value), self.assertRaises(TypeError):
                recovery.recover_core_labels(*inputs, cancel=lambda: value)
        with self.assertRaises(recovery.CoreRecoveryCancelled):
            recovery.recover_core_labels(*inputs, cancel=lambda: True)

    def check_test_retained_partition_metadata_tensor_and_annotation_callback_mutations_fail(self):
        partition = recovery.recover_core_labels(*_cuda(_pair()))
        for changed in (replace(partition, objects=()), replace(partition, status='other'),
                replace(partition, unresolved_pixels=1), replace(partition, profile='other')):
            with self.assertRaises(recovery.CoreRecoveryMutationError):
                recovery.annotate_core_recovery(changed)
        partition.labels.zero_()
        with self.assertRaises(recovery.CoreRecoveryMutationError):
            recovery.annotate_core_recovery(partition)
        partition = recovery.recover_core_labels(*_cuda(_pair()))
        def cancel():
            partition.validity.fill_(0)
            return True
        with self.assertRaises(recovery.CoreRecoveryMutationError):
            recovery.annotate_core_recovery(partition, cancel=cancel)
        partition = recovery.recover_core_labels(*_cuda(_pair()))
        for value in (-1, 9, True, 1.5):
            with self.subTest(radius=value), self.assertRaises(ValueError):
                recovery.annotate_core_recovery(partition, uncertainty_radius=value)
        with self.assertRaises(ValueError):
            recovery.annotate_core_recovery(partition, measurement_policy='other')
        class MutablePolicy:
            def __eq__(self, other):
                return True
        class StringPolicy(str):
            pass
        for policy in (MutablePolicy(), StringPolicy('flag_uncertain'), np.str_('flag_uncertain'), None, 0):
            with self.subTest(policy=type(policy).__name__), \
                    patch.object(recovery, 'fixed_kernel_program') as compile:
                with self.assertRaises(ValueError):
                    recovery.annotate_core_recovery(partition, measurement_policy=policy,
                        cancel=lambda: self.fail('Invalid policy must fail before callbacks.'))
                compile.assert_not_called()

    def check_test_in_place_frozen_record_changes_are_detected_before_annotation(self):
        for target in ('object', 'plan'):
            partition = recovery.recover_core_labels(*_cuda(_pair()))
            if target == 'object':
                object.__setattr__(partition.objects[0], 'assigned_pixels',
                                   partition.objects[0].assigned_pixels+1)
            else:
                object.__setattr__(partition.plan, 'work', partition.plan.work+1)
            with self.subTest(target=target), self.assertRaises(recovery.CoreRecoveryMutationError):
                recovery.annotate_core_recovery(partition)

    def check_test_callback_record_changes_and_changed_snapshot_withhold_publication(self):
        for target in ('recovery_plan', 'partition_object', 'partition_snapshot',
                       'annotation_plan', 'annotation_status', 'annotation_parent_snapshot'):
            for raises in (False, True):
                inputs = _cuda(_pair())
                annotation = target.startswith('annotation_')
                partition = recovery.recover_core_labels(*inputs) if annotation else None
                code = (recovery.annotate_core_recovery if annotation else recovery.recover_core_labels).__code__
                fired = []
                def cancel():
                    frame = sys._getframe()
                    while frame is not None and frame.f_code is not code:
                        frame = frame.f_back
                    if frame is None or fired:
                        return False
                    values = frame.f_locals
                    if target in ('recovery_plan', 'annotation_plan'):
                        if 'plan_snapshot' not in values:
                            return False
                        plan = values['plan']
                        object.__setattr__(plan, 'work', plan.work+1)
                    elif target == 'annotation_status':
                        if 'object_snapshot' not in values:
                            return False
                        status = values['objects'][0]
                        object.__setattr__(status, 'scientific_validation', True)
                    else:
                        retained = values.get('partition')
                        if retained is None or not retained._snapshot:
                            return False
                        facts = retained.objects[0]
                        object.__setattr__(facts, 'assigned_pixels', facts.assigned_pixels+1)
                        if target.endswith('snapshot'):
                            object.__setattr__(retained, '_snapshot', recovery._partition_snapshot(retained))
                    fired.append(True)
                    if raises:
                        raise RuntimeError('caller exception after record change')
                    return False
                with self.subTest(target=target, raises=raises), self.assertRaises(recovery.CoreRecoveryMutationError):
                    if annotation:
                        recovery.annotate_core_recovery(partition, cancel=cancel)
                    else:
                        recovery.recover_core_labels(*inputs, cancel=cancel)
                self.assertTrue(fired)

    def check_test_unknown_negative_outside_and_lost_marker_outputs_are_never_published(self):
        for failure in ('unknown', 'negative', 'outside', 'lost_marker'):
            def invalid_flood(elevation, markers, domain, capacity, guard):
                output = markers.clone()
                if failure == 'unknown':
                    output[3, 3] = 99
                elif failure == 'negative':
                    output[3, 3] = -7
                elif failure == 'outside':
                    output[0, 0] = 7
                else:
                    output[9, 4] = 0
                guard.watch(output)
                return output
            with self.subTest(failure=failure), patch.object(recovery, '_flood', invalid_flood):
                with self.assertRaises(recovery.CoreRecoveryMutationError):
                    recovery.recover_core_labels(*_cuda(_pair()))

    def check_test_queue_overflow_and_stalled_flood_are_bounded_failures(self):
        original_program = recovery.fixed_kernel_program
        for failure in ('overflow', 'stall'):
            steps = 0
            class Program:
                def __init__(self, program):
                    self.program = program
                def launch_1d(self, *args):
                    return self.program.launch_1d(*args)
                def launch(self, name, tensors, integers):
                    nonlocal steps
                    if name == 'seedvision_ws_step':
                        steps += 1
                        if failure == 'stall':
                            return
                    self.program.launch(name, tensors, integers)
                    if failure == 'overflow' and name == 'seedvision_ws_init':
                        tensors[-1][3] = 1
            def compile(source, device):
                return Program(original_program(source, device))
            with self.subTest(failure=failure), patch.object(recovery, 'fixed_kernel_program', compile), \
                    patch.object(recovery, 'KERNEL_BATCH_PIXELS', 17):
                with self.assertRaises(recovery.CoreRecoveryResourceError):
                    recovery.recover_core_labels(*_cuda(_pair()))
            self.assertLessEqual(steps, (15*19+16)//17)

    def check_test_symmetric_checkerboard_and_diagonal_only_contacts(self):
        for fg, markers, expected in (
                (np.full((2, 2), 255, np.uint8), np.array([[1, 2], [2, 1]], np.int32),
                 np.ones((2, 2), dtype=bool)),
                (np.array([[255, 0], [0, 255]], np.uint8), np.array([[1, 0], [0, 2]], np.int32),
                 np.zeros((2, 2), dtype=bool))):
            result = recovery.recover_core_labels(*_cuda((fg, markers, np.full_like(fg, 255),
                                                         np.zeros(fg.shape, np.float32))))
            np.testing.assert_array_equal(result.interface.cpu().numpy(), expected)
            torch.testing.assert_close(result.labels, torch.as_tensor(markers, device='cuda'))

    def check_test_retained_input_backing_and_annotation_resource_preflight(self):
        inputs = list(_cuda(_pair()))
        storage = torch.zeros(2_000_000, dtype=torch.uint8, device='cuda')
        storage[:inputs[0].numel()].copy_(inputs[0].reshape(-1))
        inputs[0] = storage[:inputs[0].numel()].reshape(inputs[0].shape)
        partition = recovery.recover_core_labels(*inputs)
        expected = storage.numel()+inputs[1].numel()*4+inputs[2].numel()+inputs[3].numel()*4
        self.assertEqual(partition.plan.input_backing_bytes, expected)
        with patch.object(recovery, 'MAX_LIVE_BYTES', expected-1), \
                patch.object(recovery, 'fixed_kernel_program') as compile:
            with self.assertRaises(recovery.CoreRecoveryResourceError):
                recovery.recover_core_labels(*inputs)
            compile.assert_not_called()
        for name in ('MAX_LIVE_BYTES', 'MAX_WORK', 'MAX_CPU_METADATA_BYTES'):
            with self.subTest(limit=name), patch.object(recovery, name, 1), \
                    patch.object(recovery, 'fixed_kernel_program') as compile:
                with self.assertRaises(recovery.CoreRecoveryResourceError):
                    recovery.annotate_core_recovery(partition, uncertainty_radius=8)
                compile.assert_not_called()
        with patch.object(recovery._watershed, 'MAX_WORKSPACE_BYTES', 1), \
                patch.object(recovery, '_flood') as flood:
            with self.assertRaises(recovery.CoreRecoveryResourceError):
                recovery.recover_core_labels(*inputs)
            flood.assert_not_called()

    def check_test_annotation_guards_parent_and_each_owned_buffer_even_when_callback_raises(self):
        original_watch = recovery._Guard.watch
        for target in range(13):
            partition = recovery.recover_core_labels(*_cuda(_pair()))
            watched, fired = [], False
            def watch(guard, *tensors):
                original_watch(guard, *tensors)
                watched.extend(tensors)
            def cancel():
                nonlocal fired
                if len(watched) > target:
                    watched[target].add_(1)
                    fired = True
                    raise RuntimeError('caller raised after a write')
                return False
            with self.subTest(target=target), patch.object(recovery._Guard, 'watch', watch):
                with self.assertRaises(recovery.CoreRecoveryMutationError):
                    recovery.annotate_core_recovery(partition, uncertainty_radius=3, cancel=cancel)
            self.assertTrue(fired)


    def test_workflow_lifecycle(self):
        """8 related CoreRecoveryTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_isolated_recovery_preserves_pre_erosion_contour_and_unseeded_component)
        self.scenario(self.check_test_radius_and_policy_change_annotation_reuse_labels_and_match_clipped_dilation)
        self.scenario(self.check_test_all_inputs_and_owned_workspace_are_guarded_across_caller_callbacks)
        self.scenario(self.check_test_in_place_frozen_record_changes_are_detected_before_annotation)
        self.scenario(self.check_test_unknown_negative_outside_and_lost_marker_outputs_are_never_published)
        self.scenario(self.check_test_symmetric_checkerboard_and_diagonal_only_contacts)
        self.scenario(self.check_test_retained_input_backing_and_annotation_resource_preflight)
        self.scenario(self.check_test_annotation_guards_parent_and_each_owned_buffer_even_when_callback_raises)

    def test_validation_and_recovery(self):
        """6 related CoreRecoveryTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_canvas_invalid_support_and_empty_statuses_remain_explicit)
        self.scenario(self.check_test_invalid_rasters_fail_before_partition_and_static_limits_before_compilation)
        self.scenario(self.check_test_cancellation_return_type_and_final_checkpoint_withhold_publication)
        self.scenario(self.check_test_retained_partition_metadata_tensor_and_annotation_callback_mutations_fail)
        self.scenario(self.check_test_callback_record_changes_and_changed_snapshot_withhold_publication)
        self.scenario(self.check_test_queue_overflow_and_stalled_flood_are_bounded_failures)


if __name__ == '__main__':
    unittest.main()
