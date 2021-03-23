import torch
import torchvision
import time
from tqdm import tqdm
import numpy as np
from data import load_vggface_unnormalized
from models import load_model_vggface, load_generator
from utils import get_log_writer
from gdpa import perturb_image, normalize_vggface, train_gen_batch


def train_clf_batch(inputs, targets, model, mp_generator,
                    optimizer_clf, criterion,
                    loss_l_clf, devide_theta):
    mp_generator.eval()
    model.train()

    device = 'cuda'
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer_clf.zero_grad()
    inputs = perturb_image(inputs, mp_generator, devide_theta)
    outputs = model(normalize_vggface(inputs))
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer_clf.step()
    loss_l_clf.append(loss.cpu().detach().numpy())
    _, predicted = outputs.max(1)
    return (~predicted.eq(targets)).sum().item(), targets.size(0), inputs


def at(dataloader, model, mp_generator, optimizer_gen, optimizer_clf, scheduler, criterion, epochs,
       devide_theta, writer, save_freq, size):
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
            correct_batch, total_batch, final_ims_gen = train_gen_batch(inputs, targets, model,
                                                                        mp_generator,
                                                                        optimizer_gen, criterion,
                                                                        loss_l_gen, devide_theta)
            correct_gen += correct_batch
            total_gen += total_batch
            correct_batch, total_batch, final_ims_clf = train_clf_batch(inputs, targets, model,
                                                                        mp_generator,
                                                                        optimizer_clf, criterion,
                                                                        loss_l_clf, devide_theta)
            correct_clf += correct_batch
            total_clf += total_batch

        scheduler.step()

        loss = np.array(loss_l_gen).mean()
        asr = correct_gen / total_gen
        writer.add_scalar('train_gen/loss', loss, epoch)
        writer.add_scalar('train_gen/asr', asr, epoch)
        final_ims_gen = torchvision.utils.make_grid(final_ims_gen)
        writer.add_image('final_im_gen/{}'.format(epoch), final_ims_gen, epoch)

        loss = np.array(loss_l_clf).mean()
        asr = correct_clf / total_clf
        writer.add_scalar('train_clf/loss', loss, epoch)
        writer.add_scalar('train_clf/asr', asr, epoch)
        final_ims_clf = torchvision.utils.make_grid(final_ims_clf)
        writer.add_image('final_im_clf/{}'.format(epoch), final_ims_clf, epoch)

        if (epoch + 1) % save_freq == 0:
            torch.save(model.state_dict(), 'at_model/at_classifier_size_{}_epoch_{}.pt'.format(size, epoch))

        end_time = time.time()
        print(end_time - start_time)


def main():
    epochs = 1000
    lr_gen = 0.0001
    lr_clf = 0.0001
    beta = 3000
    glass_iters = 100
    save_freq = 50
    size = 70
    para = {'exp': 'exp_at', 'device': 'cuda', 'lr_gen': lr_gen,
            'lr_clf': lr_clf, 'epochs': epochs, 'glass_iters': glass_iters, 'size': size}
    writer, base_dir = get_log_writer(para)

    dataloader, dataloader_val = load_vggface_unnormalized(32)

    model_path = '/home/xli62/uap/phattacks/glass/donemodel/new_ori_model.pt'
    model_train = load_model_vggface(model_path)
    model_train = model_train.to(para['device'])
    model_train.eval()

    mp_generator = load_generator(size, 3, 1, 64, 'resnet_6blocks').to(para['device'])

    optimizer_gen = torch.optim.Adam([
        {'params': mp_generator.parameters(), 'lr': lr_gen}
    ], lr=0.1, betas=(0.5, 0.9))

    optimizer_clf = torch.optim.Adam([
        {'params': model_train.parameters(), 'lr': lr_clf}
    ], lr=0.1, betas=(0.5, 0.9))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=50, gamma=0.2)
    criterion = torch.nn.CrossEntropyLoss()

    at(dataloader, model_train, mp_generator, optimizer_gen, optimizer_clf, scheduler,
       criterion, epochs, beta, writer, save_freq, size)


if __name__ == '__main__':
    main()
