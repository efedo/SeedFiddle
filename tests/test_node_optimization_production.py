"""Production integration on two generated images; no accuracy claims."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import torch

from seedvision.pipeline import build_default_pipeline
from seedvision.optimization.runtime import Sample, ProductionEvaluator, production_settings
from seedvision.optimization.search import optimize, apply_report
from seedvision.persistence.reference_regions import SeedInstanceAnnotation
from seedvision.segmentation.baseline import analyze_path


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA production pipeline required')
class ProductionOptimizationTests(unittest.TestCase):
    def test_shared_proposal_reproduces_final_two_image_scores(self):
        with TemporaryDirectory() as directory:
            image = np.full((384,512,3),225,np.uint8)
            cv2.circle(image,(256,190),160,(180,180,180),3)
            labels = np.zeros(image.shape[:2],np.uint16)
            for identifier,centre in enumerate(((220,150),(290,220)),1):
                cv2.circle(image,centre,25,(30,70,115),-1)
                cv2.circle(labels,centre,25,identifier,-1)
            graph = build_default_pipeline()
            first = Path(directory)/'first.png'
            cv2.imwrite(str(first),image)
            initial = analyze_path(first, **production_settings(graph),
                enabled_nodes=frozenset(key for key,node in graph.nodes.items() if node.enabled))
            shape = initial.calibration.corrected_bgr.shape[:2]
            labels = cv2.warpPerspective(labels,initial.calibration.affine_matrix,
                (shape[1],shape[0]),flags=cv2.INTER_NEAREST)
            traits = tuple(SeedInstanceAnnotation(i,shape_reviewed=True,outline_visibility='complete',pose='flat') for i in (1,2))
            samples = []
            for index in range(2):
                path = Path(directory)/f'example-{index}.png'
                cv2.imwrite(str(path),np.clip(image.astype(np.int16)-index*8,0,255).astype(np.uint8))
                samples.append(Sample(str(index),path,{'foreground_reference_mask':labels>0,'seed_instance_traits':traits},
                    shape,initial.calibration.affine_matrix,labels=labels,seed_diameter=50))
            selected = {'background_likelihood','reference_edge_probability','procedural_instances'}
            report = optimize(graph,ProductionEvaluator(graph,samples),selected=selected,maximum_evaluations=3)
            self.assertEqual(report.original,{key:dict(node.parameters) for key,node in graph.nodes.items()})
            for row in report.nodes:
                if row.node_id in selected:
                    self.assertEqual(set(row.before),{'0','1'})
                    self.assertEqual(set(row.after),{'0','1'})
                    self.assertEqual(len(row.trials),3)
            apply_report(graph,report)
            reproduced = ProductionEvaluator(graph,samples).final_scores(graph)
            for key,score in report.final_after.items():
                self.assertAlmostEqual(reproduced[key],score,places=6)


if __name__ == '__main__':
    unittest.main()
