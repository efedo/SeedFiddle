"""Bounded typed colour-observation records for saved-input replay.

Archived type names are data. Only the six explicit record schemas below are
accepted; nothing is imported or constructed from an archived type name.
"""
from collections.abc import Mapping
from dataclasses import fields
import json
import math
from types import MappingProxyType

from seedvision.cuda import plantcv_color as color
from . import color_histogram as chart

MAX_METADATA_BYTES = 8*1024*1024
MAX_METADATA_ITEMS = 1_000_000
MAX_METADATA_DEPTH = 10
MAX_STRING_BYTES = 2048
MAX_LABEL_BYTES = 1024
RECORDS = MappingProxyType({cls.__module__+'.'+cls.__qualname__:cls for cls in (
    color.ColorAnalysisResult,color.ColorLabelResult,color.ColorHistogram,
    color.ColorStatistic,chart.HueHistogramChart,chart.HueHistogramSeries)})
SCHEMAS = MappingProxyType({name:frozenset(field.name for field in fields(cls)) for name,cls in RECORDS.items()})
NONFINITE = frozenset(('nan','positive_infinity','negative_infinity'))


def state():
    """Snapshot actual compact-contract constants without retaining mutable maps."""
    return (MAX_METADATA_BYTES,MAX_METADATA_ITEMS,MAX_METADATA_DEPTH,MAX_STRING_BYTES,MAX_LABEL_BYTES,
        tuple((name,id(cls),tuple(sorted(SCHEMAS[name]))) for name,cls in RECORDS.items()),
        tuple(sorted(NONFINITE)),color.MAX_LABELS,color.MAX_IMAGE_PIXELS,color.TILE_PIXELS,color.MAX_OUTPUT_BYTES,
        color._METHOD,tuple(color._MODES),tuple((key,tuple(value)) for key,value in color._VARIABLES.items()))


PINNED_STATE=state()


def controls(parameters, error_type):
    if set(parameters)!= {'n_labels','colorspaces','label'}:
        raise error_type('Colour replay requires its exact three controls.')
    count=parameters['n_labels'];space=parameters['colorspaces'];label=parameters['label']
    if type(count) is not int or not 1<=count<=color.MAX_LABELS:
        raise error_type(f'Colour label count must be an integer in [1,{color.MAX_LABELS}].')
    if type(space) is not str or space.lower() not in ('rgb','lab','hsv','all'):
        raise error_type('Colour space must be rgb, lab, hsv or all.')
    names=(label,)*count if type(label) is str else label
    if (not isinstance(names,(tuple,list)) or len(names)!=count
            or any(type(value) is not str for value in names)):
        raise error_type('Colour labels require a string or one string per ID.')
    if any(len(value)>MAX_LABEL_BYTES for value in names):
        raise error_type('Colour replay sample labels exceed their UTF-8 byte bound.')
    try:
        if any(len(value.encode('utf-8'))>MAX_LABEL_BYTES for value in names):
            raise error_type('Colour replay sample labels exceed their UTF-8 byte bound.')
    except UnicodeEncodeError as error:
        raise error_type('Colour sample label is not valid UTF-8.') from error
    return space.lower(),tuple(f'{value}_{i}' for i,value in enumerate(names,1))


def _record(name, values):
    return MappingProxyType({'record':name,'fields':MappingProxyType(values)})


def decode(binding, error_type, poll=None):
    """Decode known records, charging keys, containers and JSON byte overhead."""
    visits=encoded=0
    max_bytes,max_items,max_depth,max_string=MAX_METADATA_BYTES,MAX_METADATA_ITEMS,MAX_METADATA_DEPTH,MAX_STRING_BYTES
    def visit(amount=1):
        nonlocal visits
        visits+=amount
        if visits>max_items:raise error_type('Colour metadata exceeds its traversal bound.')
    def charge(amount):
        nonlocal encoded
        encoded+=amount
        if encoded>max_bytes:
            raise error_type('Colour metadata exceeds its encoded-byte bound.')
    def walk(value,depth=0):
        visit()
        if poll is not None and visits%256==0:poll()
        if depth>max_depth:
            raise error_type('Colour metadata exceeds its traversal bound.')
        if type(value) is str:
            if len(value)>max_string:raise error_type('Colour metadata string exceeds its bound.')
            try:raw=value.encode('utf-8')
            except UnicodeEncodeError as error:raise error_type('Colour metadata is not UTF-8.') from error
            if len(raw)>max_string:raise error_type('Colour metadata string exceeds its bound.')
            charge(len(json.dumps(value,ensure_ascii=False).encode('utf-8')));return value
        if value is None or type(value) in (bool,int,float):
            if type(value) is int and value.bit_length()>64:
                raise error_type('Colour metadata integer exceeds its bound.')
            if type(value) is float and not math.isfinite(value):
                raise error_type('Colour nonfinite values require explicit tags.')
            charge(len(json.dumps(value)));return value
        if type(value) is not dict:raise error_type('Colour metadata requires explicit record/tuple tags.')
        if set(value)=={'kind','value'} and value['kind']=='nonfinite':
            if type(value['value']) is not str or value['value'] not in NONFINITE:
                raise error_type('Colour nonfinite tag is malformed.')
            charge(64);return ('nonfinite',value['value'])
        if set(value)=={'kind','items'} and value['kind']=='tuple' and type(value['items']) is list:
            charge(32+len(value['items']));visit(2)
            return tuple(walk(item,depth+1) for item in value['items'])
        if set(value)!= {'kind','type','fields'} or value['kind']!='record':
            raise error_type('Colour metadata contains an unsupported tag.')
        name=value['type'];values=value['fields']
        if type(name) is not str or name not in SCHEMAS or type(values) is not dict or set(values)!=SCHEMAS[name]:
            raise error_type('Colour record name or exact fields are malformed.')
        charge(80);walk(name,depth+1);visit(3)
        decoded={}
        for key,item in values.items():
            walk(key,depth+1);charge(4);decoded[key]=walk(item,depth+1)
        return _record(name,decoded)
    return walk(binding)


def canonical(value):
    """Convert only actual known dataclasses; normalize nonfinite value tokens."""
    name=type(value).__module__+'.'+type(value).__qualname__
    if name in RECORDS and type(value) is RECORDS[name]:
        return _record(name,{field.name:canonical(getattr(value,field.name)) for field in fields(value)})
    if type(value) is tuple:return tuple(canonical(item) for item in value)
    if type(value) is float and not math.isfinite(value):
        return ('nonfinite','nan' if math.isnan(value) else 'positive_infinity' if value>0 else 'negative_infinity')
    if value is None or type(value) in (str,int,float,bool):return value
    raise ValueError('Fresh colour output has an unknown metadata type.')


def validate(value, port, parameters, pixels, error_type):
    """Check shape/types and static provenance; valid numeric changes compare."""
    space,names=controls(parameters,error_type)
    def record(item,cls):
        name=cls.__module__+'.'+cls.__qualname__
        if (not isinstance(item,Mapping) or set(item)!= {'record','fields'} or item['record']!=name
                or not isinstance(item['fields'],Mapping) or set(item['fields'])!=SCHEMAS[name]):
            raise error_type('Colour product has an unexpected record schema.')
        return item['fields']
    def fixed(item,values):
        if any(type(item[key]) is not type(expected) or item[key]!=expected for key,expected in values.items()):
            raise error_type('Colour metadata differs from its fixed source contract.')
    def sequence(value,length):
        if type(value) is not tuple or len(value)!=length:
            raise error_type('Colour tuple has an incorrect shape.')
        return value
    def number(value,low,high):
        if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
            raise error_type('Colour numeric product is outside its domain.')
    def bins(values,variable):
        expected=(tuple(range(1,360,2)) if variable=='hue_frequencies' else
            tuple(range(-128,128)) if variable in ('green-magenta_frequencies','blue-yellow_frequencies') else
            tuple(round(i/255*100,2) for i in range(256)) if variable in ('lightness_frequencies','saturation_frequencies','value_frequencies') else
            tuple(range(256)))
        sequence(values,len(expected))
        # Well-typed wrong bin values are historical products, so compare them.
        for value in values:number(value,min(expected),max(expected))
    if port=='observations':
        root=record(value,color.ColorAnalysisResult)
        fixed(root,{'status':'available','colorspace':space,'error_type':None,'error':None,'failed_sample':None})
        selected=tuple(mode for mode in color._MODES if space=='all' or mode.lower()==space)
        variables=tuple(variable for mode in selected for variable in color._VARIABLES[mode])
        statistic_names=(('saturation_mean','saturation_median','value_mean','value_median') if space in ('hsv','all') else ())+('hue_circular_mean','hue_circular_std','hue_median')
        rows=sequence(root['labels'],len(names));population=0
        for i,row in enumerate(rows,1):
            sample=record(row,color.ColorLabelResult)
            fixed(sample,{'label_id':i,'sample':names[i-1]})
            count=sample['pixel_count']
            if type(count) is not int or not 0<=count<=pixels:raise error_type('Colour pixel count is outside its domain.')
            population+=count
            for histogram,variable in zip(sequence(sample['histograms'],len(variables)),variables):
                item=record(histogram,color.ColorHistogram)
                fixed(item,{'variable':variable,'trait':variable.replace('_',' '),'method':color._METHOD,
                    'scale':'frequency','units':'percent of selected pixels','datatype':"<class 'list'>"})
                bins(item['bin_labels'],variable)
                for frequency in sequence(item['values'],len(item['bin_labels'])):number(frequency,0,100)
            for statistic,variable in zip(sequence(sample['statistics'],len(statistic_names)),statistic_names):
                item=record(statistic,color.ColorStatistic)
                unit='percent' if variable.startswith(('saturation','value')) else 'degrees'
                fixed(item,{'variable':variable,'trait':'hue circular standard deviation' if variable=='hue_circular_std' else variable.replace('_',' '),
                    'method':color._METHOD,'scale':unit,'units':unit,'datatype':"<class 'float'>"})
                data=item['value'];nonfinite=type(data) is tuple and len(data)==2 and data[0]=='nonfinite' and data[1] in NONFINITE
                if nonfinite:
                    if item['available'] is not False or type(item['unavailable_reason']) is not str or not item['unavailable_reason']:
                        raise error_type('Nonfinite colour statistic requires an unavailable state/reason.')
                else:
                    number(data,0,math.inf)
                    if item['available'] is not True or item['unavailable_reason'] is not None:
                        raise error_type('Finite colour statistic requires its available state.')
        if population>pixels:raise error_type('Colour label populations exceed the raster.')
    elif port=='histogram':
        root=record(value,chart.HueHistogramChart)
        fixed(root,{'status':'available' if space in ('hsv','all') else 'empty','variable':'hue_frequencies',
            'x_title':'Hue bin centre (degrees)','y_title':'Pixels (%)',
            'rendering':'native Qt; pinned PlantCV returned data, independent rendering'})
        for i,row in enumerate(sequence(root['series'],len(names) if space in ('hsv','all') else 0),1):
            item=record(row,chart.HueHistogramSeries)
            fixed(item,{'label_id':i,'sample':names[i-1]});bins(item['bins'],'hue_frequencies')
            for frequency in sequence(item['frequencies'],180):number(frequency,0,100)
    else:raise error_type('Unknown colour replay product port.')
    return value


def compare(actual,expected,poll):
    """Count every leaf; tagged NaNs compare as their explicit canonical token."""
    mismatched=compared=0;paths=[];absent=object()
    def walk(a,b,path):
        nonlocal mismatched,compared
        if compared%256==0:poll()
        if isinstance(a,Mapping) and isinstance(b,Mapping):
            for key in sorted(set(a)|set(b)):walk(a.get(key,absent),b.get(key,absent),path+'.'+key)
            return
        if type(a) is tuple or type(b) is tuple:
            left=a if type(a) is tuple else ();right=b if type(b) is tuple else ()
            for i in range(max(len(left),len(right))):walk(left[i] if i<len(left) else absent,right[i] if i<len(right) else absent,path+'['+str(i)+']')
            if left or right or type(a) is tuple and type(b) is tuple:return
        compared+=1
        if type(a) is not type(b) or a!=b:
            mismatched+=1
            if len(paths)<32:paths.append(path)
    walk(actual,expected,'colour')
    return {'mismatched_elements':mismatched,'compared_elements':compared,'shape_equal':True,
        'kind':'typed_colour_metadata','mismatch_paths':tuple(paths)}
