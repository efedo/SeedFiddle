"""Bounded compact output contract for watershed debug-image replay."""
from collections.abc import Mapping
from hashlib import sha256
import json
import re

from seedvision.benchmarking.checkpoints import _freeze_json
from seedvision.cuda import plantcv_watershed_diagnostic as drawing

MAX_METADATA_BYTES = 256*1024
MAX_METADATA_ITEMS = 30_000
MAX_METADATA_DEPTH = 4
MAX_STRING_BYTES = 1024
KEYS = frozenset(('color_scale','random_seed','rng_profile','numpy_version',
    'matplotlib_version','colormap_lut_sha256','reference_numpy_version',
    'reference_matplotlib_version','reference_opencv_python_version',
    'reference_plantcv_version','palette_sha256','palette_source_triples',
    'palette_count','palette_source_sha256','source_sha256','line_thickness',
    'channel_boundary','kind','public_return','contour_mode','label_ids','scan_work',
    'roi_pixels','contour_points','contours','boundary_points','geometry_work',
    'topology_work','max_cpu_geometry_estimate_bytes','stroke_pixels','span_items'))


def decode(binding, error_type):
    """Decode primitive tagged metadata, counting keys, nodes and encoded bytes."""
    visits = encoded = 0
    def charge(amount):
        nonlocal encoded
        encoded += amount
        if encoded > MAX_METADATA_BYTES:
            raise error_type('Diagnostic metadata exceeds its encoded-byte bound.')
    def visit(amount=1):
        nonlocal visits
        visits += amount
        if visits > MAX_METADATA_ITEMS:
            raise error_type('Diagnostic metadata exceeds its traversal bound.')
    def walk(value, depth=0):
        visit()
        if depth > MAX_METADATA_DEPTH:
            raise error_type('Diagnostic metadata exceeds its traversal bound.')
        if type(value) is str:
            if len(value) > MAX_STRING_BYTES:
                raise error_type('Diagnostic metadata string exceeds its bound.')
            try:
                encoded_string = value.encode('utf-8')
            except UnicodeEncodeError as error:
                raise error_type('Diagnostic metadata string is not valid UTF-8.') from error
            if len(encoded_string) > MAX_STRING_BYTES:
                raise error_type('Diagnostic metadata string exceeds its bound.')
            charge(len(json.dumps(value,ensure_ascii=False).encode('utf-8')))
            return value
        if type(value) in (int,bool) or value is None:
            if type(value) is int and value.bit_length() > 64:
                raise error_type('Diagnostic metadata integer exceeds its compact bound.')
            charge(len(json.dumps(value)))
            return value
        if (type(value) is not dict or set(value)!= {'kind','items'}
                or type(value['items']) is not list):
            raise error_type('Diagnostic metadata requires primitive tagged containers.')
        if type(value['kind']) is not str or value['kind'] not in ('mapping','tuple','list'):
            raise error_type('Diagnostic metadata contains an unsupported tag.')
        charge(32+len(value['items']))  # conservative tag/delimiter allowance
        visit()  # bounded kind value; its bytes are included in the tag allowance
        visit()  # encoded items list
        if value['kind'] in ('tuple','list'):
            return tuple(walk(item,depth+1) for item in value['items'])
        if value['kind']=='mapping':
            result = {}
            for item in value['items']:
                visit()  # encoded key/value pair list
                if type(item) is not list or len(item)!=2 or type(item[0]) is not str or item[0] in result:
                    raise error_type('Diagnostic metadata contains an invalid mapping entry.')
                charge(4)
                key = walk(item[0],depth+1)
                result[key] = walk(item[1],depth+1)
            return result
        raise error_type('Diagnostic metadata contains an unsupported tag.')
    return walk(binding)


def validate(value, parameters, pixels, error_type):
    """Validate structure/provenance, preserving valid numeric output differences."""
    if not isinstance(value,Mapping) or set(value)!=KEYS:
        raise error_type('Diagnostic metadata must have its exact 31-field schema.')
    def integer(name, low, high):
        item = value[name]
        if type(item) is not int or not low<=item<=high:
            raise error_type('Diagnostic metadata integer is outside its domain: '+name)
        return item
    count = integer('palette_count',1,drawing.MAX_LABELS+1)
    integer('random_seed',0,0xffffffff)
    integer('line_thickness',1,32767)
    for name in ('color_scale','random_seed','line_thickness'):
        if type(value[name]) is not type(parameters[name]) or value[name]!=parameters[name]:
            raise error_type('Diagnostic metadata differs from its selected parameter: '+name)
    fixed = {'rng_profile':drawing.RNG_PROFILE,'numpy_version':drawing.np.__version__,
        'matplotlib_version':drawing.matplotlib.__version__,
        'reference_numpy_version':'2.2.6','reference_matplotlib_version':'3.11.2',
        'reference_opencv_python_version':'4.11.0.86','reference_plantcv_version':'4.11.3',
        'palette_source_sha256':drawing.PALETTE_SOURCE_SHA256,'source_sha256':drawing.SOURCE_SHA256,
        'channel_boundary':'native RGB; source BGR copy; literal palette triples are BGR draw colors',
        'kind':'native_watershed_debug_diagnostic','public_return':'labels only',
        'contour_mode':'RETR_TREE/CHAIN_APPROX_NONE; closed LINE_8'}
    for name, expected in fixed.items():
        if type(value[name]) is not str or value[name]!=expected:
            raise error_type('Diagnostic metadata has inconsistent fixed provenance: '+name)
    if type(value['color_scale']) is not str or not 1<=len(value['color_scale'])<=128:
        raise error_type('Diagnostic colormap name exceeds its domain.')
    for name in ('colormap_lut_sha256','palette_sha256'):
        if type(value[name]) is not str or re.fullmatch('[0-9a-f]{64}',value[name]) is None:
            raise error_type('Diagnostic metadata digest is malformed: '+name)
    palette = value['palette_source_triples']
    if (type(palette) is not tuple or len(palette)!=count
            or any(type(row) is not tuple or len(row)!=3
                   or any(type(c) is not int or not 0<=c<=255 for c in row) for row in palette)):
        raise error_type('Diagnostic palette requires its exact byte triples.')
    if sha256(bytes(c for row in palette for c in row)).hexdigest()!=value['palette_sha256']:
        raise error_type('Diagnostic palette bytes differ from their recorded digest.')
    ids = value['label_ids']
    if (type(ids) is not tuple or len(ids)>drawing.MAX_LABELS
            or any(type(i) is not int or not 1<=i<min(count,drawing.MAX_LABEL_ID+1) for i in ids)
            or any(a>=b for a,b in zip(ids,ids[1:]))):
        raise error_type('Diagnostic label IDs violate their ordered palette-index domain.')
    limits = {'scan_work':drawing.MAX_SCAN_WORK,'roi_pixels':pixels*drawing.MAX_LABELS,
        'contour_points':drawing.MAX_POINTS,'contours':drawing.MAX_CONTOURS,
        'boundary_points':drawing.MAX_POINTS,'geometry_work':drawing.MAX_GEOMETRY_WORK,
        'topology_work':drawing.size.MAX_CONTOUR_WORK_UNITS,
        'max_cpu_geometry_estimate_bytes':drawing.MAX_CPU_GEOMETRY_BYTES,
        'stroke_pixels':drawing.size.MAX_SIZE_DIAGNOSTIC_PIXELS,
        'span_items':drawing.size.MAX_SIZE_DIAGNOSTIC_SPAN_ITEMS}
    for name, maximum in limits.items():
        integer(name,0,maximum)
    return _freeze_json(value,name='watershed diagnostic replay metadata')


def compare(actual, expected, poll):
    """Compare validated compact outputs; retain at most 32 mismatch paths."""
    different = compared = 0
    paths = []
    absent = object()
    def leaf(a,b,path):
        nonlocal different, compared
        poll()
        if isinstance(a,Mapping) and isinstance(b,Mapping):
            for key in sorted(set(a)|set(b)):
                leaf(a.get(key),b.get(key),path+'.'+key)
            return
        if type(a) is tuple or type(b) is tuple:
            left = a if type(a) is tuple else ()
            right = b if type(b) is tuple else ()
            for index in range(max(len(left),len(right))):
                leaf(left[index] if index<len(left) else absent,
                     right[index] if index<len(right) else absent,path+'['+str(index)+']')
            if left or right or type(a) is tuple and type(b) is tuple:
                return
        compared += 1
        if type(a) is not type(b) or a!=b:
            different += 1
            if len(paths)<32:
                paths.append(path)
    leaf(actual,expected,'metadata')
    return {'mismatched_elements':different,'compared_elements':compared,
            'shape_equal':True,'kind':'compact_metadata','mismatch_paths':tuple(paths)}
