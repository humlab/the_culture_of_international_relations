# -*- coding: utf-8 -*-
import os
import sys
import logging
import inspect
import types

def getLogger(name='cultural_treaties', level=logging.INFO):
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

logger = getLogger(__name__)

__cwd__ = os.path.abspath(__file__) if '__file__' in globals() else os.getcwd()

sys.path.append(__cwd__)

def extend(target, *args, **kwargs):
    target = dict(target)
    for source in args:
        target.update(source)
    target.update(kwargs)
    return target

def ifextend(target, source, p):
    return extend(target, source) if p else target

def extend_single(target, source, name):
    if name in source:
        target[name] = source[name]
    return target

class SimpleStruct(types.SimpleNamespace):
    
    def __init__(self, **kwargs):
        super(SimpleStruct, self).__init__(**kwargs)
        
    def update(self, **kwargs):
        self.__dict__.update(kwargs)

def flatten(l):
    return [item for sublist in l for item in sublist]

def project_series_to_range(series, low, high):
    norm_series = series / series.max()
    return norm_series.apply(lambda x: low + (high - low) * x)

def project_to_range(value, low, high):
    return low + (high - low) * value

def clamp_values(values, low_high):
    mw = max(values)
    return [ project_to_range(w / mw, low_high[0], low_high[1]) for w in values ]

def filter_kwargs(f, args):
    try:
        return { k: args[k] for k in args.keys() if k in inspect.getargspec(f).args }
    except:
        return args

def cpif_deprecated(source, target, name):
    logger.debug('use of cpif is deprecated')
    if name in source:
        target[name] = source[name]
    return target

def dict_subset(d, keys):
    if keys is None:
        return d
    return { k: v for (k, v) in d.items() if k in keys }

def dict_split(d, fn):
    """Splits a dictionary into two parts based on predicate """
    true_keys = { k for k in d.keys() if fn(d, k) }
    return { k: d[k] for k in true_keys }, { k: d[k] for k in set(d.keys()) - true_keys }

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = dict(zip(list_of_dicts[0], zip(*[d.values() for d in list_of_dicts])))
    return dict_of_lists