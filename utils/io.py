import datetime
import logging
from pathlib import Path


def create_log(outdir, logname='log.txt'):
    "Create log file and set logging target to the file."
    fname = Path(outdir, logname)
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=fname, level=logging.DEBUG)
    return


def create_result_dir(outdir, modelpath, header=''):
    "Create result directory using the current time to set the unique name."
    modelname = Path(modelpath).stem
    now = datetime.datetime.now()
    strnow = now.strftime('%y%m%d_%H%M%S_%f')
    result_dir = Path(outdir, '{}_{}_{}'.format(header, modelname, strnow))
    if not result_dir.exists():
        result_dir.mkdir(parents=True)
    return result_dir
