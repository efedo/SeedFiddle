"""Actual saved recovery inputs, compact contracts, guards and desktop replay."""
from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch

import cv2
import numpy as np
import torch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from seedvision.benchmarking import native_core_recovery_recipe as recovery, native_recovery_replay as contracts
from seedvision.benchmarking import native_recipe as runner, native_stage_replay as replay
from seedvision.export import native_results as archives
from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
from tests.test_native_core_recovery_recipe import selected_recipe, parameter, fixture
from tests.replay_fixtures import archive_hashes as hashes, changed_archive, replay_attempt

STAGES=(recovery.CORES,recovery.VALIDITY,recovery.CLIP,recovery.PARTITION,recovery.ANNOTATION,recovery.MEASUREMENTS)
ARRAY_COUNTS={recovery.CORES:2,recovery.VALIDITY:2,recovery.CLIP:3,
    recovery.PARTITION:9,recovery.ANNOTATION:10,recovery.MEASUREMENTS:0}


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class RecoveryStageReplayTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir='artifacts');cls.root=Path(cls.temporary.name)
        cls.image=cls.root/'source.png';gray,rgb=fixture(touching=False)
        assert cv2.imwrite(str(cls.image),rgb)
        cls.cases={}
        for name,roi in (('full',None),('roi',{'shape':[64,96],'bounds':[0,1,60,64]})):
            recipe=parameter(selected_recipe(),recovery.VALIDITY,'validity_roi',roi)
            saved=runner.run_native_recipe_path(recipe,cls.image)
            assert saved.node(recovery.CORES).status=='executed'
            assert saved.node(recovery.VALIDITY).status=='executed'
            assert saved.node(recovery.CLIP).status=='executed'
            for stage in STAGES:assert saved.node(stage).status=='executed',saved.node(stage).reason
            directory=cls.root/name;archives.export_native_recipe_result(directory,recipe,saved)
            cls.cases[name]=(recipe,saved,directory)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def attempt(self,stage,case='roi',directory=None,cancel=None,change=None,observer=None):
        return replay_attempt(directory or self.cases[case][2],stage,
            cancel=cancel,inventory_change=change,observer=observer,
            observer_codes=(recovery.core._Guard.watch.__code__,) if observer else ())

    def changed(self,case='roi',**nodes):
        recipe,saved,_=self.cases[case]
        return changed_archive(self.root,recipe,saved,nodes)

    def test_operations_match_every_product_select_direct_arrays_and_preserve_archives(self):
        from tests.replay_fixtures import hidden_origins
        with hidden_origins(self.image):
            for case,(_,saved,directory) in self.cases.items():
                before=hashes(directory)
                for stage in STAGES:
                    with self.subTest(case=case,stage=stage):
                        result,calls,selected=self.attempt(stage,case)
                        self.assertEqual(result.status,'matched',result.reason)
                        self.assertEqual(calls,[saved.node(stage).operation_id]);self.assertEqual(len(selected),1)
                        self.assertEqual(len(selected[0]),ARRAY_COUNTS[stage])
                        self.assertEqual(set(result.outputs),set(saved.node(stage).outputs))
                        for name,value in result.outputs.items():
                            if torch.is_tensor(value):self.assertTrue(torch.equal(value,saved.node(stage).outputs[name]))
                        self.assertEqual(set(result.metadata['comparison']['by_port']),set(result.outputs))
                        self.assertEqual(result.metadata['frame_scope'],'source');self.assertFalse(result.scientifically_validated)
                        if stage==recovery.PARTITION:
                            self.assertIs(result.outputs['partition']['tensors']['labels'],result.outputs['labels'])
                            self.assertEqual(set(result.metadata['comparison']['by_port']['partition']['by_field']),
                                {*contracts.PARTITION_TENSORS,'metadata'})
                        elif stage==recovery.ANNOTATION:
                            self.assertIs(result.outputs['annotation']['uncertainty_band'],result.outputs['uncertainty_band'])
                        elif stage==recovery.MEASUREMENTS:
                            self.assertEqual(result.metadata['plan']['selected_bytes'],0)
                            self.assertIn('not recomputed',result.metadata['upstream_statistics'])
                self.assertEqual(hashes(directory),before)


























if __name__=='__main__':unittest.main()
