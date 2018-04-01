# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:15:29 2018

@author: HITS
"""
import imp
from pathlib import Path


def get_model(mod_fn, classes):
    "Get model on dynamic."
    model_path = Path(mod_fn)
    return imp.load_source(model_path.stem, str(model_path)).model(classes)
