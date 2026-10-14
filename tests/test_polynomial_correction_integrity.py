"""Whole-call input and retained-matrix integrity for polynomial correction."""
import unittest
from unittest.mock import patch

import torch

from seedvision.cuda import plantcv_correction as correction
from tests.test_native_correction_recipe import _cuda_inputs


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for polynomial correction')
class PolynomialCorrectionIntegrityTests(unittest.TestCase):
    def invoke(self, inputs, callback):
        state = {'phase': 'entry', 'extracted': [], 'copied': [], 'fit': {}}
        extract, apply, matrices = (correction.get_color_matrix_cuda,
            correction._apply, correction._matrices)
        solve = correction.np.linalg.solve

        def tracked_extract(image, mask, **kwargs):
            state['phase'] = 'target' if image is inputs[0] else 'source'
            result = extract(image, mask, **kwargs)
            state['extracted'].append(result[1])
            return result

        def tracked_matrices(source, target):
            result = matrices(source, target)
            state['copied'] = list(result)
            return result

        def tracked_apply(*args, **kwargs):
            state['phase'] = 'apply'
            state['fit']['coefficients'] = args[1]
            return apply(*args, **kwargs)

        def tracked_solve(*args, **kwargs):
            result = solve(*args, **kwargs)
            state['fit']['inverse'] = result
            return result

        with patch.object(correction, 'get_color_matrix_cuda', tracked_extract), \
                patch.object(correction, '_matrices', tracked_matrices), \
                patch.object(correction, '_apply', tracked_apply), \
                patch.object(correction.np.linalg, 'solve', tracked_solve):
            return correction.correct_color_cuda(*inputs, cancel=lambda: callback(state))

    def test_target_inputs_are_guarded_during_source_extraction(self):
        for index in (0, 1):
            with self.subTest(input_index=index):
                inputs, changed = _cuda_inputs(), []

                def callback(state):
                    if state['phase'] == 'source' and not changed:
                        inputs[index].reshape(-1)[0].add_(1)
                        changed.append(index)
                    return False

                with self.assertRaises(correction.CorrectionMutation):
                    self.invoke(inputs, callback)
                self.assertEqual(changed, [index])

    def test_already_consumed_inputs_are_guarded_during_application(self):
        for index in (0, 1, 3):
            with self.subTest(input_index=index):
                inputs, changed = _cuda_inputs(), []

                def callback(state):
                    if state['phase'] == 'apply' and not changed:
                        inputs[index].reshape(-1)[0].add_(1)
                        changed.append(index)
                    return False

                with self.assertRaises(correction.CorrectionMutation):
                    self.invoke(inputs, callback)
                self.assertEqual(changed, [index])

    def test_retained_chip_matrices_are_guarded_across_phases(self):
        cases = (('source', 'extracted', 0), ('apply', 'copied', 0), ('apply', 'copied', 1))
        for phase, field, index in cases:
            with self.subTest(phase=phase, field=field, matrix_index=index):
                changed = []

                def callback(state):
                    if state['phase'] == phase and not changed:
                        state[field][index][0, 1] += .001
                        changed.append(index)
                    return False

                with self.assertRaises(correction.CorrectionMutation):
                    self.invoke(_cuda_inputs(), callback)
                self.assertEqual(changed, [index])

    def test_cross_phase_mutation_takes_precedence_over_callback_exception(self):
        inputs, changed = _cuda_inputs(), []
        marker = RuntimeError('caller failure after target mutation')

        def callback(state):
            if state['phase'] == 'source' and not changed:
                inputs[0].reshape(-1)[0].add_(1)
                changed.append(True)
                raise marker
            return False

        with self.assertRaises(correction.CorrectionMutation):
            self.invoke(inputs, callback)
        self.assertTrue(changed)

    def test_unmodified_stop_and_caller_exception_keep_their_meaning(self):
        for kind in ('stop', 'exception'):
            with self.subTest(kind=kind):
                marker = RuntimeError('unchanged caller failure')

                def callback(state):
                    if state['phase'] == 'apply':
                        if kind == 'exception':
                            raise marker
                        return True
                    return False

                expected = RuntimeError if kind == 'exception' else correction.CorrectionCancelled
                with self.assertRaises(expected) as caught:
                    self.invoke(_cuda_inputs(), callback)
                if kind == 'exception':
                    self.assertIs(caught.exception, marker)

    def test_private_fit_mutation_wins_over_stop_and_caller_exception(self):
        for field in ('inverse', 'coefficients'):
            for kind in ('continue', 'stop', 'exception'):
                with self.subTest(field=field, kind=kind):
                    changed = []

                    def callback(state):
                        if state['phase'] == 'apply' and not changed:
                            state['fit'][field][0, 0] += .01
                            changed.append(True)
                            if kind == 'exception':
                                raise RuntimeError('caller failure after fit mutation')
                            return kind == 'stop'
                        return False

                    with self.assertRaises(correction.CorrectionMutation):
                        self.invoke(_cuda_inputs(), callback)
                    self.assertEqual(changed, [True])

    def test_malformed_inputs_keep_the_intended_validation_errors(self):
        invalid_images = (None, [], torch.zeros((2, 3, 3), dtype=torch.uint8),
            torch.zeros((2, 3), device='cuda', dtype=torch.uint8),
            torch.sparse_coo_tensor(torch.empty((3, 0), dtype=torch.int64),
                torch.empty(0), (2, 3, 3)))
        for image in invalid_images:
            with self.subTest(image_type=type(image).__name__):
                inputs = list(_cuda_inputs())
                inputs[0] = image
                with self.assertRaisesRegex(ValueError, 'contiguous CUDA HxWx3 uint8 BGR'):
                    correction.correct_color_cuda(*inputs)
        with self.assertRaisesRegex(TypeError, 'cancel must be a zero-argument callable or None'):
            correction.correct_color_cuda(*_cuda_inputs(), cancel=0)


if __name__ == '__main__':
    unittest.main()
