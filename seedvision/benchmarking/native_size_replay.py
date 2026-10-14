"""Bounded primitive size observations and physical-unit replay contracts."""
from collections.abc import Mapping
import json
import math
from types import MappingProxyType

from seedvision.cuda import plantcv_size as size
from . import native_calibrated_seed_recipe as units

MAX_METADATA_BYTES = 8*1024*1024
MAX_METADATA_ITEMS = 100_000
MAX_METADATA_DEPTH = 8
MAX_STRING_BYTES = 2048
MAX_LABEL_BYTES = 1024
FIELDS = frozenset(('trait','method','scale','datatype','value','label'))
# datatype is a source declaration, not necessarily the concrete value type.
SCHEMA = (
    ('in_bounds','whether the plant goes out of bounds ','none','bool','none'),
    ('area','area','pixels','int','pixels'),
    ('convex_hull_area','convex hull area','pixels','int','pixels'),
    ('solidity','solidity','none','float','none'),
    ('perimeter','perimeter','pixels','int','pixels'),
    ('total_edge_length','total length of object edges','pixels','int','pixels'),
    ('width','width','pixels','int','pixels'),
    ('height','height','pixels','int','pixels'),
    ('longest_path','longest path','pixels','int','pixels'),
    ('center_of_mass','center of mass','none','tuple',('x','y')),
    ('convex_hull_vertices','convex hull vertices','none','int','none'),
    ('object_in_frame','object in frame','none','bool','none'),
    ('ellipse_center','ellipse center','none','tuple',('x','y')),
    ('ellipse_major_axis','ellipse major axis length','pixels','int','pixels'),
    ('ellipse_minor_axis','ellipse minor axis length','pixels','int','pixels'),
    ('ellipse_angle','ellipse major axis angle','degrees','float','degrees'),
    ('ellipse_eccentricity','ellipse eccentricity','none','float','none'),
)


def state():
    return (MAX_METADATA_BYTES,MAX_METADATA_ITEMS,MAX_METADATA_DEPTH,MAX_STRING_BYTES,MAX_LABEL_BYTES,
        tuple(sorted(FIELDS)),SCHEMA,size.PLANTCV_SIZE_METHOD,
        size.MAX_RASTER_PIXELS,size.MAX_TOPOLOGY_BOUNDARY_POINTS,size.MAX_TILE_PIXELS,
        size.MAX_CONTOUR_TRACE_STEPS,size.MAX_CONTOUR_SCAN_ROWS,size.MAX_CONTOUR_COMPONENTS,
        size.MAX_CONTOUR_AGGREGATE_POINTS,size.MAX_CONTOUR_WORK_UNITS,
        size.MAX_SIZE_DIAGNOSTIC_POINTS,size.MAX_SIZE_DIAGNOSTIC_PIXELS,
        size.MAX_SIZE_DIAGNOSTIC_BATCH_PIXELS,size.MAX_SIZE_DIAGNOSTIC_SPAN_ITEMS,
        size.MAX_SIZE_DIAGNOSTIC_SCRATCH_BYTES,size.MAX_SIZE_DIAGNOSTIC_LABELS,
        size.MAX_RASTER_SPAN_WORK,size.MAX_RASTER_SPAN_SEGMENTS,
        units.AREA_VARIABLES,units.LINEAR_VARIABLES,units.UNSCALED_VARIABLES)


PINNED_STATE = state()


def controls(parameters, error_type):
    if (not isinstance(parameters,Mapping) or set(parameters) not in (
            {'n_labels','label'},{'n_labels','label','line_thickness'})):
        raise error_type('Size replay requires its exact historical or working controls.')
    count=parameters['n_labels'];label=parameters['label']
    if type(count) is not int or not 1<=count<=128:
        raise error_type('Size label count must be an integer in [1,128].')
    thickness=parameters.get('line_thickness',5)
    if type(thickness) is not int or not 1<=thickness<=32767:
        raise error_type('Size line thickness must be an integer in [1,32767].')
    names=(label,)*count if type(label) is str else label
    if (not isinstance(names,(tuple,list)) or len(names)!=count
            or any(type(value) is not str for value in names)):
        raise error_type('Size labels require a string or one string per ID.')
    for value in names:
        if len(value)>MAX_LABEL_BYTES:
            raise error_type('Size sample label exceeds its UTF-8 bound.')
        try:raw=value.encode('utf-8')
        except UnicodeEncodeError as error:raise error_type('Size label is not valid UTF-8.') from error
        if len(raw)>MAX_LABEL_BYTES:raise error_type('Size sample label exceeds its UTF-8 bound.')
    return tuple(f'{value}_{i}' for i,value in enumerate(names,1))


def _walk(root, error_type, poll, tagged):
    """Freeze only primitive mappings/sequences, with captured traversal limits."""
    visits=encoded=0
    max_bytes,max_items,max_depth,max_string=MAX_METADATA_BYTES,MAX_METADATA_ITEMS,MAX_METADATA_DEPTH,MAX_STRING_BYTES
    def charge(amount):
        nonlocal encoded
        encoded+=amount
        if encoded>max_bytes:raise error_type('Size metadata exceeds its encoded-byte bound.')
    def walk(value,depth=0):
        nonlocal visits
        visits+=1
        if visits>max_items or depth>max_depth:raise error_type('Size metadata exceeds its traversal bound.')
        if poll is not None and visits%256==0:poll()
        if type(value) is str:
            if len(value)>max_string:raise error_type('Size metadata string exceeds its bound.')
            try:raw=value.encode('utf-8')
            except UnicodeEncodeError as error:raise error_type('Size metadata is not UTF-8.') from error
            if len(raw)>max_string:raise error_type('Size metadata string exceeds its bound.')
            charge(len(json.dumps(value,ensure_ascii=False).encode('utf-8')));return value
        if value is None or type(value) in (bool,int,float):
            if type(value) is int and value.bit_length()>64:raise error_type('Size metadata integer exceeds its bound.')
            if type(value) is float and not math.isfinite(value):raise error_type('Size metadata must be finite.')
            charge(len(json.dumps(value)));return value
        if tagged:
            if (type(value) is not dict or set(value)!={'kind','items'} or type(value['items']) is not list):
                raise error_type('Size metadata requires tagged primitive mappings/sequences.')
            kind,items=value['kind'],value['items'];charge(32+len(items))
            if kind in ('tuple','list'):return tuple(walk(item,depth+1) for item in items)
            if kind!='mapping':raise error_type('Size metadata contains an unsupported tag.')
            result={}
            for item in items:
                if type(item) is not list or len(item)!=2 or type(item[0]) is not str or item[0] in result:
                    raise error_type('Size metadata has an invalid mapping entry.')
                key=walk(item[0],depth+1);charge(4);result[key]=walk(item[1],depth+1)
            return MappingProxyType(result)
        if type(value) in (tuple,list):
            charge(32+len(value));return tuple(walk(item,depth+1) for item in value)
        if isinstance(value,Mapping):
            charge(32+len(value));result={}
            for key,item in value.items():
                if type(key) is not str:raise error_type('Size metadata keys must be strings.')
                result[walk(key,depth+1)]=walk(item,depth+1);charge(4)
            return MappingProxyType(result)
        raise error_type('Fresh size metadata has an unknown type.')
    return walk(root)


def decode(binding, error_type, poll=None):
    return _walk(binding,error_type,poll,True)


def canonical(value, error_type=None, poll=None):
    if error_type is None:error_type=ValueError
    return _walk(value,error_type,poll,False)


def _number(value, error_type, positive=False):
    if (type(value) not in (int,float) or not math.isfinite(value)
            or positive and value<=0):
        raise error_type('Size metadata requires finite numeric values in its source domain.')


def observations(value, parameters, error_type, unit='pixels'):
    names=controls(parameters,error_type)
    if not isinstance(value,Mapping) or set(value)!=set(names):
        raise error_type('Size sample records differ from the effective label controls.')
    for sample in names:
        values=value[sample]
        if not isinstance(values,Mapping) or set(values)!={row[0] for row in SCHEMA}:
            raise error_type('Size requires exactly seventeen observation variables.')
        for variable,trait,scale,datatype,label in SCHEMA:
            row=values[variable]
            if not isinstance(row,Mapping) or set(row)!=FIELDS:
                raise error_type('Size observation requires its exact six fields.')
            if variable in units.AREA_VARIABLES+units.LINEAR_VARIABLES:scale=label=unit
            fixed={'trait':trait,'method':'plantcv.plantcv.within_frame' if variable=='in_bounds' else size.PLANTCV_SIZE_METHOD,
                'scale':scale,'datatype':"<class '"+datatype+"'>",'label':label}
            if any(type(row[key]) is not type(expected) or row[key]!=expected for key,expected in fixed.items()):
                raise error_type('Size observation differs from its fixed source descriptor contract.')
            measurement=row['value']
            if datatype=='bool':
                if type(measurement) is not bool:raise error_type('Size frame observations require bool values.')
            elif datatype=='tuple':
                if type(measurement) is not tuple or len(measurement)!=2:
                    raise error_type('Size coordinates require a two-value tuple.')
                for coordinate in measurement:_number(coordinate,error_type)
            else:_number(measurement,error_type)
    return value


def scale(value, error_type):
    if not isinstance(value,Mapping) or set(value)!={'px_width','px_height','unit'}:
        raise error_type('Size units require the exact card scale record.')
    if all(item is None for item in value.values()):return 1.0,1.0,'pixels'
    width,height,unit=value['px_width'],value['px_height'],value['unit']
    _number(width,error_type,True);_number(height,error_type,True)
    if type(unit) is not str or unit!='mm' or not math.isfinite(width*height):
        raise error_type('Size scale requires mm and a finite area factor.')
    return width,height,unit


def metadata(value, error_type):
    fixed={'area_variables':units.AREA_VARIABLES,'linear_variables':units.LINEAR_VARIABLES,
        'coordinate_units':'ROI-local pixels'}
    if not isinstance(value,Mapping) or set(value)!=set(fixed)|{'px_width','px_height','unit','area_factor','source_area_scale_field'}:
        raise error_type('Size units metadata requires its exact eight fields.')
    _number(value['px_width'],error_type,True);_number(value['px_height'],error_type,True)
    _number(value['area_factor'],error_type)
    if (type(value['unit']) is not str or value['unit'] not in ('pixels','mm')
            or type(value['source_area_scale_field']) is not str or value['source_area_scale_field']!=value['unit']
            or any(type(value[key]) is not type(expected) or value[key]!=expected for key,expected in fixed.items())):
        raise error_type('Size unit metadata differs from its fixed source descriptors.')
    return value


def compare(actual, expected, poll):
    """Compare every primitive leaf and retain at most32 differing paths."""
    mismatched=compared=0;paths=[];absent=object()
    def walk(a,b,path):
        nonlocal mismatched,compared
        if compared%256==0:poll()
        if isinstance(a,Mapping) and isinstance(b,Mapping):
            for key in sorted(set(a)|set(b)):walk(a.get(key,absent),b.get(key,absent),path+'.'+key)
            return
        if type(a) is tuple or type(b) is tuple:
            left=a if type(a) is tuple else ();right=b if type(b) is tuple else ()
            for i in range(max(len(left),len(right))):
                walk(left[i] if i<len(left) else absent,right[i] if i<len(right) else absent,path+'['+str(i)+']')
            if left or right or type(a) is tuple and type(b) is tuple:return
        compared+=1
        if type(a) is not type(b) or a!=b:
            mismatched+=1
            if len(paths)<32:paths.append(path)
    walk(actual,expected,'size')
    return {'mismatched_elements':mismatched,'compared_elements':compared,'shape_equal':True,
        'kind':'typed_size_metadata','mismatch_paths':tuple(paths)}
