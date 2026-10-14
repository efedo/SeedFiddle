"""Compact PlantCV-style PDF plots; no image access or output publication.

The pinned trainer plots ordered classes at implicit x=0..255, uses legend
loc=best and writes one default-style SVG per HSV channel. We use independent
noninteractive Figures and serialize our Matplotlib jobs. Factory defaults are
explicit; custom pyplot state/styles and byte-identical SVG are not promised.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import re
from threading import Lock
import xml.etree.ElementTree as ET

from seedvision.cuda.plantcv_naive_bayes import (
    CHANNELS, MAX_CLASSES, NaiveBayesModel, _validate_model)

FILENAMES = tuple(channel + '_pdf.svg' for channel in CHANNELS)
MAX_LABEL_BYTES = 32_768
MAX_TOTAL_LABEL_BYTES = 65_536
MAX_SVG_BYTES = 8 * 1024 * 1024
MAX_TOTAL_SVG_BYTES = 24 * 1024 * 1024
PLANTCV_TRAINER_SHA256 = 'f59c95c493d189b6110d47a60a97c86e306ab87583a2f128eff8c389aa6b56eb'
_RENDER_LOCK = Lock()


class BayesPlotError(ValueError):
    """Unsupported compact data, renderer or output size."""


class BayesPlotCancelled(RuntimeError):
    """No complete plot bundle is published after cancellation."""


@dataclass(frozen=True, slots=True)
class BayesPDFPlot:
    filename: str
    svg_bytes: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class BayesPDFPlots:
    model_sha256: str
    tables_sha256: str
    class_names: tuple[str, ...]
    renderer: tuple[tuple[str, str], ...]
    plots: tuple[BayesPDFPlot, ...]

    @property
    def metadata_json(self):
        """Descriptive plot identity, separate from the numerical model record."""
        return json.dumps(dict(schema=1, kind='native_bayes_pdf_plots',
            model_sha256=self.model_sha256, tables_sha256=self.tables_sha256,
            class_names=self.class_names, renderer=dict(self.renderer),
            plots=[dict(filename=p.filename, bytes=len(p.svg_bytes), sha256=p.sha256)
                   for p in self.plots]), ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), allow_nan=False)


def _validate_compact_model(model):
    if (type(model) is not NaiveBayesModel or type(model.class_names) is not tuple
            or not 1 <= len(model.class_names) <= MAX_CLASSES):
        raise BayesPlotError('Plots require a bounded parsed PlantCV PDF model.')
    total = 0
    for name in model.class_names:
        if type(name) is not str or not name or len(name) > MAX_LABEL_BYTES:
            raise BayesPlotError('PDF plot class label exceeds the compact label bound.')
        try:
            size = len(name.encode('utf-8'))
        except UnicodeError as error:
            raise BayesPlotError('PDF plot labels must be valid UTF-8.') from error
        total += size
        if size > MAX_LABEL_BYTES or total > MAX_TOTAL_LABEL_BYTES:
            raise BayesPlotError('PDF plot class labels exceed the compact label bound.')
    _validate_model(model)


def _facts(model):
    return (model.class_names, model.probabilities, model.source_sha256,
            model.source_bytes, model.tables_sha256)


def _poll(cancel, model, frozen):
    if _facts(model) != frozen:
        raise BayesPlotError('PDF model changed during plotting.')
    if cancel is not None:
        try:
            stopped = cancel()
        finally:
            if _facts(model) != frozen:
                raise BayesPlotError('PDF model changed during plotting.')
        if type(stopped) is not bool:
            raise BayesPlotError('Plot cancellation must return bool.')
        if stopped:
            raise BayesPlotCancelled('PDF plotting stopped before a complete bundle.')


class _BoundedSVG(io.BytesIO):
    def write(self, data):
        if self.tell() + len(data) > MAX_SVG_BYTES:
            raise BayesPlotError('PDF SVG exceeds the compact output byte bound.')
        return super().write(data)


def _renderer_modules():
    try:
        import matplotlib
        from matplotlib.backends.backend_svg import FigureCanvasSVG
        from matplotlib.figure import Figure
        import numpy
    except ImportError as error:
        raise BayesPlotError('PDF plots require the Matplotlib runtime dependency.') from error
    return matplotlib, numpy, Figure, FigureCanvasSVG


def render_naive_bayes_plots(model: NaiveBayesModel, *, cancel=None) -> BayesPDFPlots:
    """Render all three SVG payloads from compact fitted tables, or return none.

    No pyplot figure manager or interactive backend is used. Cancellation while
    waiting for our renderer lock or between classes/channels returns no bundle.
    Rendering cannot undo any already consumed training RNG draws.
    """
    if cancel is not None and not callable(cancel):
        raise BayesPlotError('Plot cancellation must be callable or None.')
    _validate_compact_model(model)
    frozen = _facts(model)
    _poll(cancel, model, frozen)
    while not _RENDER_LOCK.acquire(timeout=.05):
        _poll(cancel, model, frozen)
    try:
        _poll(cancel, model, frozen)
        mpl, numpy, Figure, Canvas = _renderer_modules()
        # A private default-style context prevents application pyplot style and
        # GUI backend state from changing these independent SVG Figures.
        defaults = deepcopy({k: v for k, v in mpl.rcParamsDefault.items() if k != 'backend'})
        rc_sha = sha256(json.dumps({k: str(v) for k, v in defaults.items()},
            sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        renderer = tuple(dict(matplotlib=mpl.__version__, numpy=numpy.__version__,
            backend='matplotlib.backends.backend_svg.FigureCanvasSVG',
            profile='matplotlib-factory-default-svg-v1', rcparams_sha256=rc_sha,
            plantcv_trainer_sha256=PLANTCV_TRAINER_SHA256).items())
        plots = []
        with mpl.rc_context(defaults):
            for channel, filename in enumerate(FILENAMES):
                _poll(cancel, model, frozen)
                figure = Figure()
                Canvas(figure)
                try:
                    axes = figure.subplots()
                    for name, tables in zip(model.class_names, model.probabilities):
                        _poll(cancel, model, frozen)
                        axes.plot(tables[channel], label=name)
                    axes.legend(loc='best')
                    with _BoundedSVG() as stream:
                        figure.savefig(stream, format='svg')
                        svg = stream.getvalue()
                    if sum(len(p.svg_bytes) for p in plots) + len(svg) > MAX_TOTAL_SVG_BYTES:
                        raise BayesPlotError('PDF SVG bundle exceeds the compact output byte bound.')
                    plots.append(BayesPDFPlot(filename, svg, sha256(svg).hexdigest()))
                finally:
                    figure.clear()
                _poll(cancel, model, frozen)
        bundle = BayesPDFPlots(model.source_sha256, model.tables_sha256,
            model.class_names, renderer, tuple(plots))
        validate_naive_bayes_plots(bundle, model)
        _poll(cancel, model, frozen)
        return bundle
    finally:
        _RENDER_LOCK.release()


def validate_naive_bayes_plots(bundle, model):
    """Check complete compact payload identity before any UI file publication."""
    _validate_compact_model(model)
    if (type(bundle) is not BayesPDFPlots
            or type(bundle.model_sha256) is not str or type(bundle.tables_sha256) is not str
            or bundle.model_sha256 != model.source_sha256
            or bundle.tables_sha256 != model.tables_sha256
            or type(bundle.class_names) is not tuple or bundle.class_names != model.class_names
            or type(bundle.plots) is not tuple or len(bundle.plots) != 3
            or type(bundle.renderer) is not tuple or len(bundle.renderer) != 6):
        raise BayesPlotError('PDF plot bundle differs from the selected model.')
    for pair in bundle.renderer:
        if (type(pair) is not tuple or len(pair) != 2
                or any(type(value) is not str or not value or len(value) > 512 for value in pair)):
            raise BayesPlotError('PDF renderer identity is invalid.')
    renderer = dict(bundle.renderer)
    if (set(renderer) != {'matplotlib', 'numpy', 'backend', 'profile',
                         'rcparams_sha256', 'plantcv_trainer_sha256'}
            or renderer['backend'] != 'matplotlib.backends.backend_svg.FigureCanvasSVG'
            or renderer['profile'] != 'matplotlib-factory-default-svg-v1'
            or not re.fullmatch('[0-9a-f]{64}', renderer['rcparams_sha256'])
            or renderer['plantcv_trainer_sha256'] != PLANTCV_TRAINER_SHA256):
        raise BayesPlotError('PDF renderer identity is invalid.')
    total = 0
    for filename, plot in zip(FILENAMES, bundle.plots):
        if (type(plot) is not BayesPDFPlot or plot.filename != filename
                or type(plot.svg_bytes) is not bytes or not 0 < len(plot.svg_bytes) <= MAX_SVG_BYTES
                or type(plot.sha256) is not str
                or sha256(plot.svg_bytes).hexdigest() != plot.sha256):
            raise BayesPlotError('PDF plot payload identity is invalid.')
        total += len(plot.svg_bytes)
        try:
            if ET.fromstring(plot.svg_bytes).tag != '{http://www.w3.org/2000/svg}svg':
                raise BayesPlotError('PDF plot payload is not an SVG document.')
        except ET.ParseError as error:
            raise BayesPlotError('PDF plot payload is not an SVG document.') from error
    if total > MAX_TOTAL_SVG_BYTES:
        raise BayesPlotError('PDF SVG bundle exceeds the compact output byte bound.')
