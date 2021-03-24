import torch
import torchvision
import time
from tqdm import tqdm
import numpy as np
from data import load_vggface_unnormalized, normalize_vggface
from models import load_model_vggface, load_generator
from utils import get_log_writer
from gdpa import perturb_image
import argparse


def train_gen_batch(inputs, targets, model, mp_generator,
                    optimizer_gen, criterion,
                    loss_l_gen, devide_theta, theta_bound, device, alpha=1, p_scale=10000):
    mp_generator.train()
    model.eval()

    inputs, targets = inputs.to(device), targets.to(device)
    optimizer_gen.zero_grad()
    inputs = perturb_image(inputs, mp_generator, devide_theta, theta_bound, device, alpha=alpha, p_scale=p_scale)
    outputs = model(normalize_vggface(inputs[:, [2, 1, 0], :, :]))
    loss = -criterion(outputs, targets)
    loss.backward()
    optimizer_gen.step()
    loss_l_gen.append(loss.cpu().detach().numpy())
    _, predicted = outputs.max(1)
    return (~predicted.eq(targets)).sum().item(), targets.size(0), inputs


def train_clf_batch(inputs, targets, model, mp_generator,
                    optimizer_clf, criterion,
                    loss_l_clf, devide_theta, theta_bound, device):
    mp_generator.eval()
    model.train()
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer_clf.zero_grad()
    inputs = perturb_image(inputs, mp_generator, devide_theta, theta_bound, device)
    outputs = model(normalize_vggface(inputs))
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer_clf.step()
    loss_l_clf.append(loss.cpu().detach().numpy())
    _, predicted = outputs.max(1)
    return (~predicted.eq(targets)).sum().item(), targets.size(0), inputs


def gdpa_at(dataloader, model, mp_generator, optimizer_gen, optimizer_clf, scheduler, criterion, epochs,
            devide_theta, writer, save_freq, patch_size, theta_bound, device):
    for epoch in range(epochs):
        start_time = time.time()
        print('epoch: {}'.format(epoch))
        loss_l_gen = []
        correct_gen = 0
        total_gen = 0
        loss_l_clf = []
        correct_clf = 0
        total_clf = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
            # train generator
            correct_batch, total_batch, final_ims_gen = train_gen_batch(inputs, targets, model,
                                                                        mp_generator,
                                                                        optimizer_gen, criterion,
                                                                        loss_l_gen, devide_theta, theta_bound, device)
            correct_gen += correct_batch
            total_gen += total_batch
            # train clf
            correct_batch, total_batch, final_ims_clf = train_clf_batch(inputs, targets, model,
                                                                        mp_generator,
                                                                        optimizer_clf, criterion,
                                                                        loss_l_clf, devide_theta, theta_bound, device)
            correct_clf += correct_batch
            total_clf += total_batch

        scheduler.step()
        # log generator
        loss = np.array(loss_l_gen).mean()
        asr = correct_gen / total_gen
        writer.add_scalar('train_gen/loss', loss, epoch)
        writer.add_scalar('train_gen/asr', asr, epoch)
        final_ims_gen = torchvision.utils.make_grid(final_ims_gen)
        writer.add_image('final_im_gen/{}'.format(epoch), final_ims_gen, epoch)
        # log clf
        loss = np.array(loss_l_clf).mean()
        asr = correct_clf / total_clf
        writer.add_scalar('train_clf/loss', loss, epoch)
        writer.add_scalar('train_clf/asr', asr, epoch)
        final_ims_clf = torchvision.utils.make_grid(final_ims_clf)
        writer.add_image('final_im_clf/{}'.format(epoch), final_ims_clf, epoch)
        # save model
        if (epoch + 1) % save_freq == 0:
            torch.save(model.state_dict(), 'at_model/at_classifier_size_{}_epoch_{}.pt'.format(patch_size, epoch))
        # time
        end_time = time.time()
        print(end_time - start_time)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/xli62/uap/phattacks/glass/Data')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr_gen', type=float, default=0.0001)
    parser.add_argument('--lr_clf', type=float, default=0.0001)
    parser.add_argument('--beta', type=int, default=3000)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--patch_size', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--vgg_model_path', type=str,
                        default='/home/xli62/uap/phattacks/glass/donemodel/new_ori_model.pt')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    para = {'exp': 'exp_at', 'lr_gen': args.lr_gen, 'lr_clf': args.lr_clf,
            'epochs': args.epochs, 'size': args.patch_size}
    writer, base_dir = get_log_writer(para)
    # data
    dataloader, dataloader_val = load_vggface_unnormalized(args.batch_size, args.data_path)
    # model
    model_train = load_model_vggface(args.vgg_model_path)
    model_train = model_train.to(args.device)
    model_train.eval()
    # gdpa generator
    mp_generator = load_generator(args.patch_size, 3, 64).to(args.device)
    # training setting
    optimizer_gen = torch.optim.Adam([
        {'params': mp_generator.parameters(), 'lr': args.lr_gen}
    ], lr=0.1, betas=(0.5, 0.9))
    optimizer_clf = torch.optim.Adam([
        {'params': model_train.parameters(), 'lr': args.lr_clf}
    ], lr=0.1, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=50, gamma=0.2)
    criterion = torch.nn.CrossEntropyLoss()
    theta_bound = 1 - (args.patch_size / 224.0)
    # main logic
    gdpa_at(dataloader, model_train, mp_generator, optimizer_gen, optimizer_clf, scheduler,
            criterion, args.epochs, args.beta, writer, args.save_freq, args.patch_size, theta_bound, args.device)


if __name__ == '__main__':
    main()
