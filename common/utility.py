# -*- coding: utf-8 -*-
import os
import sys
import time
import pandas as pd
import shutil
import zipfile
import logging
import inspect

from itertools import product
        
def getLogger(name='cultural_treaties', level=logging.INFO):
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

logger = getLogger(__name__)

__cwd__ = os.path.abspath(__file__) if '__file__' in globals() else os.getcwd()

sys.path.append(__cwd__)

def extend(target, source):
    target.update(source)
    return target
    
def ifextend(target, source, p):
    return extend(target, source) if p else target

def extend_single(target, source, name):
    if name in source:
        target[name] = source[name]
    return target

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
        return { k:args[k] for k in args.keys() if k in inspect.getargspec(f).args }
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
    return { k:v for (k,v) in d.items() if k in keys }


