"""Compact default-style SVGs versus the actual pinned public plotting helper."""

from tests.functional_units import functional_units
from concurrent.futures import ThreadPoolExecutor
import builtins
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from seedvision.cuda.plantcv_naive_bayes import CHANNELS, parse_plantcv_pdf
from seedvision.export import naive_bayes_plots as native

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/plantcv-reference-env/Scripts/python.exe'
ORACLE = r'''
import hashlib,importlib,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
module=importlib.import_module('plantcv.learn.naive_bayes')
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == sys.argv[2]
assert matplotlib.__version__ == '3.11.2'
root=Path(sys.argv[1]); observations=[]
for case in json.loads(sys.stdin.read()):
    out=root/str(len(observations)); out.mkdir()
    records=[]
    original=plt.savefig
    def save(path):
        axes=plt.gca()
        records.append(dict(lines=[dict(label=line.get_label(),x=line.get_xdata().tolist(),
            y=line.get_ydata().tolist()) for line in axes.lines],
            legend=[text.get_text() for text in axes.get_legend().get_texts()]))
        original(path)
    plt.savefig=save
    try:
        with matplotlib.rc_context({k:v for k,v in matplotlib.rcParamsDefault.items() if k!='backend'}):
            for c,channel in enumerate(('hue','saturation','value')):
                module._plot_pdf(channel,str(out),**{name:np.asarray(tables[c],dtype=np.float64)
                    for name,tables in zip(case['names'],case['tables'])})
    finally:
        plt.savefig=original; plt.close('all')
    observations.append(records)
(root/'observations.json').write_text(json.dumps(observations,ensure_ascii=False),encoding='utf-8')
'''


def pdf_model(names=('zeta', 'alpha')):
    rows = ['class\tchannel\t'+'\t'.join(map(str, range(256)))]
    for c, channel in enumerate(CHANNELS):
        for index, name in enumerate(names):
            values = [(1+index)*(1+c)*(.001+((x-53*index)**2 % 173)/10000) for x in range(256)]
            rows.append(name+'\t'+channel+'\t'+'\t'.join(map(repr, values)))
    return parse_plantcv_pdf(('\n'.join(rows)+'\n').encode('utf-8'))


def normalized_svg(raw):
    tree = ET.fromstring(raw)
    ids = {node.attrib['id']:f'id{i}' for i,node in enumerate(tree.iter()) if 'id' in node.attrib}
    for node in tree.iter():
        if node.tag.endswith('}date'):
            node.text = ''
        for key, value in list(node.attrib.items()):
            if key == 'id':
                node.attrib[key] = ids[value]
            else:
                node.attrib[key] = re.sub(r'#([A-Za-z0-9_]+)',
                    lambda match:'#'+ids.get(match[1], match[1]), value)
    return ET.tostring(tree)


@functional_units
class BayesPDFPlotTests(unittest.TestCase):

    def check_test_exact_tables_hashes_renderer_metadata_and_no_global_rng_change(self):
        model = pdf_model()
        before = np.random.get_state()
        result = native.render_naive_bayes_plots(model)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0]); np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(result.model_sha256, model.source_sha256)
        self.assertEqual(result.tables_sha256, model.tables_sha256)
        metadata = json.loads(result.metadata_json)
        self.assertEqual(metadata['renderer']['matplotlib'], matplotlib.__version__)
        self.assertEqual(metadata['renderer']['backend'], 'matplotlib.backends.backend_svg.FigureCanvasSVG')
        for record,plot in zip(metadata['plots'],result.plots):
            self.assertEqual(record['sha256'], sha256(plot.svg_bytes).hexdigest())
            self.assertEqual(record['bytes'], len(plot.svg_bytes))


    def check_test_bad_models_labels_and_forged_bundles_fail_before_publication(self):
        model = pdf_model()
        for invalid in [None,replace(model,class_names=('x'*32769,'y')),
                replace(model,class_names=('\ud800','y')),replace(model,tables_sha256='a'*64)]:
            with self.subTest(invalid=type(invalid)),patch.object(native,'_renderer_modules') as loader:
                with self.assertRaises(ValueError):
                    native.render_naive_bayes_plots(invalid)
                loader.assert_not_called()
        result = native.render_naive_bayes_plots(model)
        first = result.plots[0]
        for invalid in [replace(result,model_sha256='b'*64),replace(result,class_names=tuple(reversed(model.class_names))),
                replace(result,plots=result.plots[:2]),replace(result,plots=(replace(first,filename='../hue.svg'),)+result.plots[1:]),
                replace(result,plots=(replace(first,svg_bytes=first.svg_bytes+b'changed'),)+result.plots[1:]),
                replace(result,plots=(replace(first,svg_bytes=b'<not-svg/>',sha256=sha256(b'<not-svg/>').hexdigest()),)+result.plots[1:]),
                replace(result,renderer=result.renderer[:-1]+(result.renderer[0],))]:
            with self.subTest(invalid=invalid.model_sha256),self.assertRaises(ValueError):
                native.validate_naive_bayes_plots(invalid,model)



    def check_test_queued_cancel_and_concurrent_jobs_keep_models_and_rc_state_separate(self):
        model = pdf_model()
        stop,started = threading.Event(),threading.Event()
        outcome = []
        def queued():
            started.set()
            try:
                native.render_naive_bayes_plots(model,cancel=stop.is_set)
            except Exception as error:
                outcome.append(error)
        native._RENDER_LOCK.acquire()
        try:
            thread = threading.Thread(target=queued)
            thread.start(); self.assertTrue(started.wait(2)); stop.set(); thread.join(2)
            self.assertFalse(thread.is_alive())
        finally:
            native._RENDER_LOCK.release()
        self.assertEqual(len(outcome),1)
        self.assertIsInstance(outcome[0],native.BayesPlotCancelled)
        models = [model,pdf_model(('different',))]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(native.render_naive_bayes_plots,models))
        self.assertEqual([r.tables_sha256 for r in results],[m.tables_sha256 for m in models])

    def test_workflow_lifecycle(self):
        """1 related BayesPDFPlotTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_exact_tables_hashes_renderer_metadata_and_no_global_rng_change)

    def test_validation_and_recovery(self):
        """2 related BayesPDFPlotTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_bad_models_labels_and_forged_bundles_fail_before_publication)
        self.scenario(self.check_test_queued_cancel_and_concurrent_jobs_keep_models_and_rc_state_separate)


if __name__ == '__main__':
    unittest.main()
