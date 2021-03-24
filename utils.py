import os
import shutil
from torch.utils.tensorboard import SummaryWriter
import torch


def scale_pattern(mask, p_scale=10000):
    if p_scale == 1:
        mask_s = torch.tanh(mask) / 2 + 0.5
    if p_scale == 10000:
        mask_s = mask
    return mask_s


def scale_theta(mask, theta_div, theta_bound):
    mask_s = torch.tanh(mask / theta_div) * theta_bound
    return mask_s


def para2dir(para):
    path = ''
    for key in para:
        path += key
        path += '/'
        path += str(para[key])
        path += '/'
    path = path[:-1]
    return path


def get_log_writer(para):
    base_dir = para2dir(para)

    directory = 'logs/{}'.format(base_dir)
    if os.path.exists(directory):
        shutil.rmtree(directory)
        os.makedirs(directory)
    else:
        os.makedirs(directory)

    save_dir = 'save/' + base_dir
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        os.makedirs(save_dir)
    return SummaryWriter(directory), save_dir
