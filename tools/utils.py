""" General purpose utility functions to limit duplication of code """

import logging
import numpy as np
import os
import torch

from copy import copy
from pathlib import Path
from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def DotDict(in_dict):
    out_dict = copy(in_dict)
    for k, v in out_dict.items():
        if isinstance(v, dict):
            out_dict[k] = DotDict(v)
    return dotdict(out_dict)


def append2dict(source, data):
    for k in data.keys():
        if isinstance(data[k], list):
            source[k] += data[k].astype(np.float32)
        else:
            source[k].append(data[k].astype(np.float32))

def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)


def params2torch(params, dtype=torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}


def prepare_params(params, frame_mask, dtype=np.float32):
    return {k: v[frame_mask].astype(dtype) for k, v in params.items()}


def np2torch(item):
    out = {}
    for k, v in item.items():
        if v == []:
            continue
        if isinstance(v, list):
            try:
                out[k] = torch.from_numpy(np.concatenate(v))
            except:
                out[k] = torch.from_numpy(np.array(v))
        elif isinstance(v, dict):
            out[k] = np2torch(v)
        elif isinstance(v, Data):
            out[k] = v
            logging.debug(f"Object of class {Data} is not converted to the tensor!")
        else:
            out[k] = torch.from_numpy(v)
    return out


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array


def get_project_root():
    """ Return string with a path to the root folder in the project """
    project_root = Path(__file__).parent.parent
    return str(project_root)


def makepath(desired_path, isfile=False):
    """
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    """
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):
            os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path):
            os.makedirs(desired_path)
    return desired_path


def makelogger(logfile_path=None, mode='w'):
    """ 
    Initializes and configures the logger. 
    Args:
        logfile_path (str): Desired path to a file where the logs are exported.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # if the logging directory is given, logs will be stored in a file
    if logfile_path:
        os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
        fh = logging.FileHandler('%s' % logfile_path, mode=mode)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


INTENTS = ['lift', 'pass', 'offhand', 'use', 'all']
OBJECTS = ['airplane', 'alarmclock', 'apple', 'banana', 'binoculars', 'bowl', 'camera', 'phone', 'cube', 'cup',
           'cylinder', 'doorknob', 'elephant', 'eyeglasses', 'flashlight', 'flute', 'hammer', 'hand', 'headphones',
           'knife', 'lightbulb', 'mouse', 'mug', 'fryingpan', 'piggybank', 'pyramid', 'duck', 'scissors', 'sphere',
           'stanfordbunny', 'stapler', 'toothbrush', 'toothpaste', 'torus', 'train', 'teapot', 'waterbottle',
           'wineglass', 'watch', 'gamecontroller', 'stamp']
SUBJECTS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
OBJECTS_SIZES = ['airplane', 'alarmclock', 'apple', 'banana', 'binoculars', 'bowl', 'camera', 'phone', 'cubelarge',
                 'cubemedium', 'cubesmall', 'cup', 'cylinderlarge', 'cylindermedium', 'cylindersmall', 'doorknob',
                 'elephant', 'eyeglasses', 'flashlight', 'flute', 'hammer', 'hand', 'headphones', 'knife', 'lightbulb',
                 'mouse', 'mug', 'fryingpan', 'piggybank', 'pyramidlarge', 'pyramidmedium', 'pyramidsmall',
                 'duck', 'scissors', 'spherelarge', 'spheremedium', 'spheresmall', 'stanfordbunny', 'stapler',
                 'toothbrush', 'toothpaste', 'toruslarge', 'torusmedium', 'torussmall', 'train', 'teapot',
                 'waterbottle', 'wineglass', 'watch', 'gamecontroller', 'stamp']
