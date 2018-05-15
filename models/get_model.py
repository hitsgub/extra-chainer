import imp
from pathlib import Path


def get_model(mod_fn, classes):
    "Get model on dynamic."
    model_path = Path(mod_fn)
    return imp.load_source(model_path.stem, str(model_path)).model(classes)
