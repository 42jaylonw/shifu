import os
from datetime import datetime
import random
import numpy as np
import torch


def datetime_logdir(log_root, run_name):
    return os.path.join(log_root, datetime.now().strftime('%Y%m%d-%H:%M:%S') + '_' + run_name)


def latest_logdir(log_root, run_name=''):
    target_runs = []
    for run in os.listdir(log_root):
        if run_name in run:
            target_runs.append(run)
    target_runs.sort()
    latest_run = target_runs[-1]
    print(f'found the latest logdir: {latest_run}')
    return os.path.join(log_root, latest_run)


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        last_run = latest_logdir(root)
    except:
        raise ValueError("No runs in this directory: " + root)

    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
