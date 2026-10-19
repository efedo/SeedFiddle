"""Saved direct card inputs, mixed frames, every product, guards and actual Qt."""

from tests.functional_units import functional_units
from dataclasses import replace
import os
from pathlib import Path
import shutil
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
from seedvision.benchmarking import native_card_recipe as card, native_card_replay as contracts
from seedvision.benchmarking import native_recipe as runner, native_stage_replay as replay
from seedvision.benchmarking import native_calibrated_seed_recipe as calibrated
from seedvision.benchmarking import native_watershed_diagnostic_recipe as diagnostic
from seedvision.export import native_results as archives
from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
from tests.test_native_card_recipe import fixture, _parameter
from tests.replay_fixtures import (archive_hashes as hashes, changed_archive,
    replay_attempt, assert_stage_inputs_equal)

BGR=card.PREFIX+'bgr'
ROI=(40,40,420,580)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
@functional_units
class CardStageReplayTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir='artifacts');cls.root=Path(cls.temporary.name)
        cls.image=cls.root/'card.png';assert cv2.imwrite(str(cls.image),fixture(varied=True))
        cls.cases={}
        for name,bounds in (('full',None),('roi',ROI)):
            recipe=_parameter(card.build_native_card_recipe(),card.PREPROCESS,'roi_bounds_xyxy',bounds)
            recipe=_parameter(recipe,card.SCALE,'color_chip_size',[7,11])
            saved=runner.run_native_recipe_path(recipe,cls.image)
            assert saved.status=='executed',[(node.stage_id,node.reason) for node in saved.nodes]
            directory=cls.root/name;archives.export_native_recipe_result(directory,recipe,saved)
            cls.cases[name]=(recipe,saved,directory)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def attempt(self,stage,case='roi',directory=None,cancel=None,inventory_change=None):
        return replay_attempt(directory or self.cases[case][2],stage,
            cancel=cancel,inventory_change=inventory_change)

    def changed(self,case='roi',**nodes):
        recipe,saved,_=self.cases[case]
        return changed_archive(self.root,recipe,saved,nodes)

    def check_test_every_full_and_roi_card_stage_matches_all_outputs_and_direct_selection(self):
        from tests.replay_fixtures import hidden_origins
        with hidden_origins(self.image):
            counts={BGR:2,card.PREPROCESS:5,card.GEOMETRY:2,card.MASK:3,card.SCALE:0,
                    card.MEANS:2,card.STANDARD:1,card.AFFINE:2}
            for case,(recipe,saved,directory) in self.cases.items():
                before=hashes(directory)
                for stage in recipe.stages[1:]:
                    with self.subTest(case=case,stage=stage.stage_id):
                        result,calls,selected=self.attempt(stage.stage_id,case)
                        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,[stage.operation_id])
                        self.assertEqual(len(selected),1);self.assertEqual(len(selected[0]),counts[stage.stage_id])
                        self.assertEqual(set(result.outputs),set(saved.node(stage.stage_id).outputs))
                        self.assertEqual(set(result.metadata['comparison']['by_port']),set(result.outputs))
                        for name,value in result.outputs.items():
                            if torch.is_tensor(value):self.assertTrue(torch.equal(value,saved.node(stage.stage_id).outputs[name]))
                        self.assertFalse(result.scientifically_validated)
                self.assertEqual(hashes(directory),before)



    def check_test_all_nine_actual_card_graph_variants_replay_canonical_shared_prefix(self):
        canonical=card.build_native_card_recipe()
        variants=[canonical]
        for kind in ('binary','mean','otsu','range'):
            recipe=calibrated.build_native_calibrated_seed_recipe(kind)
            variants.extend((recipe,diagnostic.build_native_watershed_diagnostic_recipe(recipe)))
        self.assertEqual(len(variants),9)
        prefix=tuple(stage.stage_id for stage in canonical.stages[1:])
        for recipe in variants:
            with self.subTest(recipe=recipe.recipe_id):
                runner._validate_recipe(recipe)
                assert_stage_inputs_equal(self,recipe,canonical,prefix)
                # Composition adds source_frame, but consumes exactly the same
                # RGB boundary. All eight prefixes therefore have one numerical
                # contract; downstream numerical profiles remain in acceptance.
                source=recipe.stage(canonical.stages[0].stage_id)
                for port in canonical.stages[0].ports:self.assertIn(port,source.ports)
        saved=runner.run_native_recipe_path(canonical,self.image)
        self.assertTrue(all(saved.node(stage).status=='executed' for stage in prefix))
        directory=self.root/'canonical-prefix';archives.export_native_recipe_result(directory,canonical,saved)
        for stage in (BGR,card.PREPROCESS,card.SCALE,card.AFFINE):
            result,calls,_=self.attempt(stage,directory=directory)
            self.assertEqual(result.status,'matched',result.reason)
            self.assertEqual(calls,[canonical.stage(stage).operation_id])
        self.assertIsNone(saved.node(card.SCALE).outputs['metadata']['unit'])
        for recipe_id in diagnostic.BASE_IDS[:3]:
            recipe=diagnostic.canonical_base(recipe_id+diagnostic.SUFFIX)
            self.assertFalse(any(stage.operation_id in contracts.PORTS for stage in recipe.stages))








    def check_test_singular_affine_inputs_and_ignored_chip_ids_match_saved_direct_fit(self):
        recipe,saved,_=self.cases['roi'];means=saved.node(card.MEANS);affine=saved.node(card.AFFINE)
        rows=tuple((-1000.0,.2,.3,.4) for _ in range(24))
        direct=card.execute_card_stage('color_card.affine',{'bgr':saved.node(BGR).outputs['bgr'],
            'source_matrix':rows,'target_matrix':saved.node(card.STANDARD).outputs['target_matrix']},{},None)
        directory=self.changed(**{card.MEANS:replace(means,outputs={'source_matrix':rows}),
            card.AFFINE:replace(affine,outputs=direct)})
        result,calls,_=self.attempt(card.AFFINE,directory=directory)
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,['color_card.affine'])











    def test_workflow_lifecycle(self):
        """3 related CardStageReplayTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_every_full_and_roi_card_stage_matches_all_outputs_and_direct_selection)
        self.scenario(self.check_test_all_nine_actual_card_graph_variants_replay_canonical_shared_prefix)
        self.scenario(self.check_test_singular_affine_inputs_and_ignored_chip_ids_match_saved_direct_fit)


if __name__=='__main__':unittest.main()
