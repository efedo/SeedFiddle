"""Compact default-style SVGs versus the actual pinned public plotting helper."""
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


class BayesPDFPlotTests(unittest.TestCase):
    def test_actual_pinned_helper_order_curves_legend_and_normalized_svg(self):
        models = [pdf_model(), pdf_model(('one',)), pdf_model(('épi', '$x^2$', '_hidden'))]
        with tempfile.TemporaryDirectory() as directory:
            run = subprocess.run([str(REFERENCE), '-X', 'utf8', '-c', ORACLE,
                directory, native.PLANTCV_TRAINER_SHA256], input=json.dumps([
                    dict(names=m.class_names,tables=m.probabilities) for m in models]),
                text=True, capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            observations = json.loads((Path(directory)/'observations.json').read_text(encoding='utf-8'))
            for index, model in enumerate(models):
                result = native.render_naive_bayes_plots(model)
                self.assertEqual(tuple(p.filename for p in result.plots), native.FILENAMES)
                for c, plot in enumerate(result.plots):
                    with self.subTest(case=index,channel=c):
                        observed = observations[index][c]
                        self.assertEqual([line['label'] for line in observed['lines']], list(model.class_names))
                        self.assertEqual(observed['legend'], [name for name in model.class_names if not name.startswith('_')])
                        for line,tables in zip(observed['lines'],model.probabilities):
                            self.assertEqual(line['x'], list(range(256)))
                            self.assertEqual(line['y'], list(tables[c]))
                        expected = (Path(directory)/str(index)/plot.filename).read_bytes()
                        self.assertEqual(normalized_svg(plot.svg_bytes), normalized_svg(expected))

    def test_exact_tables_hashes_renderer_metadata_and_no_global_rng_change(self):
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

    def test_maximum_training_class_label_domain_and_independent_figures(self):
        names = tuple(f'c{i}-'+chr(65+i%26)*250 for i in range(32))
        model = pdf_model(names)
        figure = plt.figure()
        plt.plot([3,2,1], label='existing figure')
        before = plt.get_fignums()
        try:
            with matplotlib.rc_context({'lines.linewidth':11,'figure.figsize':[2,2]}):
                result = native.render_naive_bayes_plots(model)
                self.assertEqual(matplotlib.rcParams['lines.linewidth'],11)
                self.assertEqual(matplotlib.rcParams['figure.figsize'],[2,2])
            self.assertEqual(plt.get_fignums(),before)
            self.assertEqual(len(figure.axes[0].lines),1)
            self.assertLess(sum(len(p.svg_bytes) for p in result.plots),native.MAX_TOTAL_SVG_BYTES)
        finally:
            plt.close(figure)

    def test_bad_models_labels_and_forged_bundles_fail_before_publication(self):
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

    def test_streaming_and_aggregate_caps_raise_without_returning_partial_bundle(self):
        model = pdf_model()
        with patch.object(native,'MAX_SVG_BYTES',64),self.assertRaisesRegex(ValueError,'byte bound'):
            native.render_naive_bayes_plots(model)
        with patch.object(native,'MAX_TOTAL_SVG_BYTES',64),self.assertRaisesRegex(ValueError,'byte bound'):
            native.render_naive_bayes_plots(model)
        self.assertEqual(len(native.render_naive_bayes_plots(model).plots),3)

    def test_cancel_callback_mutation_and_renderer_errors_release_lock(self):
        model = pdf_model()
        for callback in [lambda:True,lambda:1]:
            with self.assertRaises((native.BayesPlotCancelled,native.BayesPlotError)):
                native.render_naive_bayes_plots(model,cancel=callback)
        changed = replace(model)
        def mutate():
            object.__setattr__(changed,'source_bytes',changed.source_bytes+1)
            raise RuntimeError('callback failed')
        with self.assertRaisesRegex(ValueError,'changed'):
            native.render_naive_bayes_plots(changed,cancel=mutate)
        calls = []
        def cancel_later():
            calls.append(None)
            return len(calls)>6
        with self.assertRaises(native.BayesPlotCancelled):
            native.render_naive_bayes_plots(model,cancel=cancel_later)
        with (patch.object(native,'_renderer_modules',side_effect=ImportError('missing runtime')),
                self.assertRaises(ImportError)):
            native.render_naive_bayes_plots(model)
        original_import = builtins.__import__
        def missing(name,*args,**kwargs):
            if name == 'matplotlib':
                raise ImportError('missing runtime')
            return original_import(name,*args,**kwargs)
        with patch.object(builtins,'__import__',side_effect=missing),self.assertRaisesRegex(ValueError,'runtime dependency'):
            native.render_naive_bayes_plots(model)
        from matplotlib.figure import Figure
        with patch.object(Figure,'savefig',side_effect=ValueError('renderer failed')),self.assertRaisesRegex(ValueError,'renderer failed'):
            native.render_naive_bayes_plots(model)
        self.assertEqual(len(native.render_naive_bayes_plots(model).plots),3)

    def test_queued_cancel_and_concurrent_jobs_keep_models_and_rc_state_separate(self):
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


if __name__ == '__main__':
    unittest.main()
