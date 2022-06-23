import datetime
import os
from math import sqrt

import numpy as np
import torchvision.transforms as standard_transformers
import torchvision.utils as utils
from tensorboardX import SummaryWriter
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

ckp_path = '../../ckpt'
exp_name = 'voc_psp_net'
writer = SummaryWriter(os.path.join(ckp_path, 'exp', exp_name))

args = {
    'train_batch_size': 1,
    'lr': 1e-2 / sqrt(16 / 4),
    'lr_decay': 0.9,
    'max_ite': 3e4,
    'longer_size': 512,
    'crop_size': 473,
    'stride_rate': 2/3,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'print_freq': 10,
    'val_save_to_img_file': True,
    'val_img_sample_rate': 0.01,
    'val_img_display_size': 384
}


#def main():



