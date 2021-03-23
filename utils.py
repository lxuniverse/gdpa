import os
import shutil
from torch.utils.tensorboard import SummaryWriter


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
