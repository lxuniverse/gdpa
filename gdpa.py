import torch
import torch.nn.functional as F
import torchvision
import time
from tqdm import tqdm
import numpy as np
import argparse
from utils import get_log_writer, scale_theta, scale_pattern
from models import load_generator, load_model_vggface
from data import load_vggface_unnormalized, normalize_vggface, load_imagenet_unnormalize, normalize_imagenet
import torchvision.models as models


def move_m_p(aff_theta, pattern_s, device, alpha=1):
    bs = pattern_s.size()[0]
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


def perturb_image(inputs, mp_generator, devide_theta, device, alpha=1, p_scale=10000):
    mask_generated, pattern_generated, aff_theta = mp_generator(inputs)
    aff_theta = scale_theta(aff_theta, devide_theta)
    pattern_s = scale_pattern(pattern_generated, p_scale=p_scale)
    mask_s, pattern_s = move_m_p(aff_theta, pattern_s, device, alpha=alpha)
    inputs = inputs * (1 - mask_s) + pattern_s * mask_s
    inputs = inputs.clamp(0, 1)
    return inputs


def train_gen_batch(inputs, targets, model, mp_generator, optimizer_gen, criterion,
                    loss_l_gen, devide_theta, normalize_func, device, alpha=1, p_scale=10000):
    mp_generator.train()
    model.eval()

    inputs, targets = inputs.to(device), targets.to(device)
    optimizer_gen.zero_grad()
    inputs = perturb_image(inputs, mp_generator, devide_theta, device, alpha=alpha, p_scale=p_scale)
    outputs = model(normalize_func(inputs))
    loss = -criterion(outputs, targets)
    loss.backward()
    optimizer_gen.step()
    loss_l_gen.append(loss.cpu().detach().numpy())
    _, predicted = outputs.max(1)
    return (~predicted.eq(targets)).sum().item(), targets.size(0), inputs


def test_gen_batch(inputs, targets, model, mp_generator,
                   optimizer_gen, devide_theta, normalize_func, device, alpha=1, p_scale=10000):
    mp_generator.eval()
    model.eval()

    inputs, targets = inputs.to(device), targets.to(device)
    optimizer_gen.zero_grad()
    inputs = perturb_image(inputs, mp_generator, devide_theta, device, alpha=alpha, p_scale=p_scale)
    outputs = model(normalize_func(inputs))
    _, predicted = outputs.max(1)
    return (~predicted.eq(targets)).sum().item(), targets.size(0), inputs


def gdpa(dataloader, dataloader_val, model, mp_generator, optimizer_gen, scheduler, criterion,
          epochs, devide_theta, alpha, normalize_func, writer, device):
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
                                                                        loss_l_gen, devide_theta, normalize_func,
                                                                        device, alpha=alpha, p_scale=10000)
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
                                                                       optimizer_gen, devide_theta, normalize_func,
                                                                       device, alpha=alpha, p_scale=10000)
            correct_gen2 += correct_batch
            total_gen2 += total_batch
        # testing log
        asr = correct_gen2 / total_gen2
        writer.add_scalar('test_gen/asr', asr, epoch)
        final_ims_gen = torchvision.utils.make_grid(final_ims_gen)
        writer.add_image('final_im_test/{}'.format(epoch), final_ims_gen, epoch)
        # scheduler
        scheduler.step()
        # time
        end_time = time.time()
        print(end_time - start_time)


def get_args():
    # input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='gdpa')
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=int, default=3000)
    parser.add_argument('--dataset', type=str, default='vggface')
    parser.add_argument('--data_path', type=str, default='/home/xli62/uap/phattacks/glass/Data')
    parser.add_argument('--vgg_model_path', type=str,
                        default='/home/xli62/uap/phattacks/glass/donemodel/new_ori_model.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_gen', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    para = {'exp': args.exp, 'beta': args.beta, 'lr_gen': args.lr_gen,
            'epochs': args.epochs, 'alpha': args.alpha, 'patch_size': args.patch_size, 'dataset': args.dataset}
    writer, base_dir = get_log_writer(para)
    # data
    if para['dataset'] == 'vggface':
        dataloader, dataloader_val = load_vggface_unnormalized(args.batch_size, args.data_path)
        normalize_func = normalize_vggface
    elif para['dataset'] == 'imagenet':
        dataloader, dataloader_val = load_imagenet_unnormalize(args.batch_size, args.data_path)
        normalize_func = normalize_imagenet
    # clf model
    if para['dataset'] == 'vggface':
        model_train = load_model_vggface(args.vgg_model_path)
    elif para['dataset'] == 'imagenet':
        model_train = models.vgg19(pretrained=True)
    model_train = model_train.to(args.device)
    model_train.eval()
    # gen model
    mp_generator = load_generator(para['patch_size'], 3, 64).to(args.device)
    # training setting
    optimizer_gen = torch.optim.Adam([
        {'params': mp_generator.parameters(), 'lr': para['lr_gen']}
    ], lr=0.1, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=50, gamma=0.2)
    criterion = torch.nn.CrossEntropyLoss()
    # train and test
    gdpa(dataloader, dataloader_val, model_train, mp_generator, optimizer_gen, scheduler,
          criterion, para['epochs'], para['beta'], para['alpha'], normalize_func, writer, args.device)


if __name__ == '__main__':
    main()
