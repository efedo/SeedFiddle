"""Shared reference imports retain fresh PlantCV state and actual calculations."""
import json
from pathlib import Path
import subprocess
import unittest

from tests.reference_session import reference_session


class ReferenceSessionTests(unittest.TestCase):
    def test_changed_requests_match_one_shot_and_do_not_share_plantcv_state(self):
        root=Path(__file__).resolve().parents[1]
        executable=str(root/'artifacts/plantcv-reference-env/Scripts/python.exe')
        source='''import json, numpy as np
import plantcv.plantcv as pcv
assert not hasattr(pcv,"fixture_private_state")
pcv.fixture_private_state=True
print(json.dumps(pcv.threshold.binary(np.array([[0,80,160,240]],dtype=np.uint8),threshold=THRESHOLD,object_type="light").tolist()))
'''
        with reference_session() as session:
            answers=[]
            for threshold in (100,200):
                script=source.replace('THRESHOLD',str(threshold))
                answer=subprocess.run([executable,'-c',script],cwd=root,capture_output=True,text=True,check=True,timeout=60)
                answers.append(json.loads(answer.stdout))
            direct=session.original_run([executable,'-c',source.replace('THRESHOLD','200')],cwd=root,
                capture_output=True,text=True,check=True,timeout=60)
            self.assertEqual(answers,[[[0,0,255,255]],[[0,0,0,255]]])
            self.assertEqual(answers[-1],json.loads(direct.stdout))
            self.assertEqual(session.stats['worker_starts'],1)
            self.assertEqual(session.stats['actual_operations'],2)
        self.assertEqual(session.children,{})
