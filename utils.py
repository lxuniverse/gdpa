import os
import shutil
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from data import normalize_vggface


def attack_batch_pgd(inputs, targets, model, pgd_iter=20, alpha=1, epsilon=16):
    model.eval()
    device = 'cuda'
    inputs, targets = inputs.to(device), targets.to(device)

    delta = torch.zeros_like(inputs, requires_grad=True)
    for t in range(pgd_iter):
        loss = nn.CrossEntropyLoss()(model((inputs + delta)[:, [2, 1, 0], :, :]), targets)
        loss.backward()
        delta.data = (delta + inputs.shape[0] * alpha * delta.grad.data).clamp(-epsilon / 255, epsilon / 255)
        delta.grad.zero_()
    perturbed_input = (inputs + delta.detach()).clamp(0, 1)
    outputs = model(normalize_vggface(perturbed_input))
    _, predicted = outputs.max(1)
    return (~predicted.eq(targets)).sum().item(), targets.size(0), perturbed_input


def scale_pattern(mask, p_scale=10000):
    if p_scale == 1:
        mask_s = torch.tanh(mask) / 2 + 0.5
    if p_scale == 10000:
        mask_s = mask
    return mask_s


def scale_theta(mask, theta_div):
    mask_s = torch.tanh(mask / theta_div) * 0.8
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
