import logging
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset
import pickle


def read_pickle(filepath):
    with open(filepath, 'rb')as f:
        return pickle.load(f)


def write_pickle(filepath, data):
    with open(filepath, 'wb')as f:
        pickle.dump(data, f)


def set_seed(num):
    os.environ['PYTHONHASHSEED'] = str(num)
    np.random.seed(num)
    random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)