from __future__ import print_function

import os
import json
import logging
import numpy as np
from datetime import datetime

import torchvision.utils as vutils

def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False):
    from PIL import Image
    tensor = tensor.cpu()
    grid = vutils.make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, range=range, scale_each=scale_each)
    #ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    ndarr = grid.byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)
