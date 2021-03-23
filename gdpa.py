import torch
import torch.nn.functional as F
import torchvision
import time
from tqdm import tqdm
import numpy as np
import argparse
from utils import get_log_writer
from models import load_generator, load_model_vggface
from data import load_vggface_unnormalized, normalize_vggface, load_imagenet_unnormalize, normalize_imagenet
import torchvision.models as models

def scale_pattern(mask, p_scale=10000):
    if p_scale == 1:
        mask_s = torch.tanh(mask) / 2 + 0.5
    if p_scale == 10000:
        mask_s = mask
    return mask_s


def scale_theta(mask, theta_div, theta_bound):
    mask_s = torch.tanh(mask / theta_div) * theta_bound
    return mask_s


def move_m_p(aff_theta, pattern_s, alpha=1):
    bs = pattern_s.size()[0]
    device = 'cuda'
    image_with_patch = torch.zeros(bs, 3, 224, 224, device=device)
    mask_with_patch = torch.zeros(bs, 1, 224, 224, device=device)
    start = 111 - pattern_s.size()[2] // 2
    end = start + pattern_s.size()[2]
    image_with_patch[:, :, start:end, start:end] = pattern_s
    mask_with_patch[:, :, start:end, start:end] = alpha

    rot_theta = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).unsqueeze(0).to(device).repeat(bs, 1, 1)
    theta_batch = torch.cat((rot_theta, aff_theta.unsqueeze(2)), 2)
    grid = F.affine_grid(theta_batch, image_with_patch.size(), align_corners=True)
    pattern_s = F.grid_sample(image_with_patch, grid, align_corners=True)
    mask_s = F.grid_sample(mask_with_patch, grid, align_corners=True)
    return mask_s, pattern_s


def perturb_image(inputs, mp_generator, devide_theta, theta_bound, alpha=1, p_scale=10000):
    mask_generated, pattern_generated, aff_theta = mp_generator(inputs)

    aff_theta = scale_theta(aff_theta, devide_theta, theta_bound)
    pattern_s = scale_pattern(pattern_generated, p_scale=p_scale)

    mask_s, pattern_s = move_m_p(aff_theta, pattern_s, alpha=alpha)

    inputs = inputs * (1 - mask_s) + pattern_s * mask_s
    inputs = inputs.clamp(0, 1)
    return inputs


def train_gen_batch(inputs, targets, model, mp_generator,
                    optimizer_gen, criterion,
                    loss_l_gen, devide_theta, normalize_func, theta_bound, alpha=1, p_scale=10000):
    mp_generator.train()
    model.eval()

    device = 'cuda'
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer_gen.zero_grad()
    inputs = perturb_image(inputs, mp_generator, devide_theta, theta_bound, alpha=alpha, p_scale=p_scale)
    outputs = model(normalize_func(inputs))
    loss = -criterion(outputs, targets)
    loss.backward()
    optimizer_gen.step()
    loss_l_gen.append(loss.cpu().detach().numpy())
    _, predicted = outputs.max(1)
    return (~predicted.eq(targets)).sum().item(), targets.size(0), inputs


def test_gen_batch(inputs, targets, model, mp_generator,
                   optimizer_gen, devide_theta, normalize_func, theta_bound, alpha=1, p_scale=10000):
    mp_generator.eval()
    model.eval()

    device = 'cuda'
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer_gen.zero_grad()
    inputs = perturb_image(inputs, mp_generator, devide_theta, theta_bound, alpha=alpha, p_scale=p_scale)
    outputs = model(normalize_func(inputs))
    _, predicted = outputs.max(1)
    return (~predicted.eq(targets)).sum().item(), targets.size(0), inputs


def train(dataloader, dataloader_val, model, mp_generator, optimizer_gen, scheduler, criterion,
          epochs, devide_theta, alpha, normalize_func, writer, theta_bound):
    for epoch in range(epochs):
        start_time = time.time()
        print('epoch: {}'.format(epoch))
        # training
        loss_l_gen = []
        correct_gen = 0
        total_gen = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
            correct_batch, total_batch, final_ims_gen = train_gen_batch(inputs, targets, model,
                                                                        mp_generator,
                                                                        optimizer_gen, criterion,
                                                                        loss_l_gen, devide_theta, normalize_func, theta_bound, alpha=alpha,
                                                                        p_scale=10000)
            correct_gen += correct_batch
            total_gen += total_batch
        # training log
        loss = np.array(loss_l_gen).mean()
        asr = correct_gen / total_gen
        writer.add_scalar('train_gen/loss', loss, epoch)
        writer.add_scalar('train_gen/asr', asr, epoch)
        final_ims_gen = torchvision.utils.make_grid(final_ims_gen)
        writer.add_image('final_im_gen/{}'.format(epoch), final_ims_gen, epoch)
        # testing
        correct_gen2 = 0
        total_gen2 = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader_val)):
            correct_batch, total_batch, final_ims_gen = test_gen_batch(inputs, targets, model,
                                                                       mp_generator,
                                                                       optimizer_gen, devide_theta, normalize_func, theta_bound, alpha=alpha,
                                                                       p_scale=10000)
            correct_gen2 += correct_batch
            total_gen2 += total_batch
        asr = correct_gen2 / total_gen2
        writer.add_scalar('train_gen2/asr', asr, epoch)
        final_ims_gen = torchvision.utils.make_grid(final_ims_gen)
        writer.add_image('final_im_gen2/{}'.format(epoch), final_ims_gen, epoch)
        # scheduler
        scheduler.step()
        end_time = time.time()
        print(end_time - start_time)


def get_para():
    # input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='gdpa_beta2')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=int, default=3000)
    args = parser.parse_args()
    # fixed parameters
    epochs = 50
    lr_gen = 0.0005
    alpha = args.alpha
    patch_size = args.size
    dataset = 'vggface'
    imagenet_model = 'resnet50'
    # para
    para = {'exp': args.exp, 'device': 'cuda', 'beta': args.beta, 'lr_gen': lr_gen,
            'epochs': epochs, 'alpha': alpha, 'patch_size': patch_size, 'dataset': dataset, 'imagenet_model': imagenet_model}
    print(para)
    return para


def main():
    para = get_para()
    writer, base_dir = get_log_writer(para)
    # data
    if para['dataset'] == 'vggface':
        dataloader, dataloader_val = load_vggface_unnormalized(32)
        normalize_func = normalize_vggface
    elif para['dataset'] == 'imagenet':
        dataloader, dataloader_val = load_imagenet_unnormalize(32)
        normalize_func = normalize_imagenet
    # clf model
    if para['dataset'] == 'vggface':
        model_path = '/home/xli62/uap/phattacks/glass/donemodel/new_ori_model.pt'
        model_train = load_model_vggface(model_path)
    elif para['dataset'] == 'imagenet':
        if para['imagenet_model'] == 'resnet50':
            model_train = models.resnet50(pretrained=True)
        if para['imagenet_model'] == 'vgg16':
            model_train = models.vgg16(pretrained=True)
        if para['imagenet_model'] == 'vgg19':
            model_train = models.vgg19(pretrained=True)
    model_train = model_train.to(para['device'])
    model_train.eval()
    # gen model
    mp_generator = load_generator(para['patch_size'], 3, 1, 64, 'resnet_6blocks').to(para['device'])
    # training setting
    optimizer_gen = torch.optim.Adam([
        {'params': mp_generator.parameters(), 'lr': para['lr_gen']}
    ], lr=0.1, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=50, gamma=0.2)
    criterion = torch.nn.CrossEntropyLoss()
    # bound for theta
    theta_bound = 1 - (para['patch_size'] / 224.0)
    # train and test
    train(dataloader, dataloader_val, model_train, mp_generator, optimizer_gen, scheduler,
          criterion, para['epochs'], para['beta'], para['alpha'], normalize_func, writer, theta_bound)


if __name__ == '__main__':
    main()
