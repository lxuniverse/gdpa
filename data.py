import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets
import os
import numpy as np
from torch.utils.data import Subset

data_mean_imagenet = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1).to('cuda')
data_std_imagenet = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1).to('cuda')


def normalize_imagenet(x):
    return (x - data_mean_imagenet) / data_std_imagenet


data_mean_vggface = torch.tensor([0.367035294117647, 0.41083294117647057, 0.5066129411764705]).unsqueeze(1).unsqueeze(
    1).to('cuda')
data_std_vggface = torch.tensor([1 / 255, 1 / 255, 1 / 255]).unsqueeze(1).unsqueeze(1).to('cuda')


def normalize_vggface(x):
    x = x[:, [2, 1, 0], :, :]
    return (x - data_mean_vggface) / data_std_vggface


def load_imagenet_unnormalize(bs, path):
    num_train_im = 10000
    dataset_train = datasets.ImageFolder(path + '/train', transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]))

    targets = np.array(dataset_train.targets)
    id_l = []
    for c in range(1000):
        c_idx = np.where(targets == c)[0][:int(num_train_im / 1000)]
        id_l.append(c_idx)
    subset = np.hstack(id_l)
    dataset_train = Subset(dataset_train, subset)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=bs, shuffle=True,
        num_workers=4, pin_memory=True)

    dataloader_val = torch.utils.data.DataLoader(
        datasets.ImageFolder(path + '/val', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=True)

    return dataloader_train, dataloader_val


def load_vggface_unnormalized(bs, data_path):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
        ]),
    }

    data_dir = data_path  # change this if the data is in different loaction
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}

    dataloader_train = torch.utils.data.DataLoader(image_datasets['train'], batch_size=bs,
                                                   shuffle=True, drop_last=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(image_datasets['test'], batch_size=bs,
                                                  shuffle=False, drop_last=True, num_workers=4)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    print(class_names)
    print(dataset_sizes)
    return dataloader_train, dataloader_test
