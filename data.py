import os

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def get_dataloaders(data_dir, train_batch_size, test_batch_size, data_transforms=None, shuffle_train=True, shuffle_test=False):

    if data_transforms is None:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(128),
                transforms.RandomResizedCrop((128,128)),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(),
                transforms.RandomAffine(degrees=0.),
                transforms.ToTensor(),
                transforms.Normalize((0.449,), (0.226,))
            ]),
            'test': transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop((128,128)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.449,), (0.226,))
            ]),
        }

    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                ['train', 'test']}
    dataloaders = {'train': DataLoader(datasets['train'], batch_size=train_batch_size, shuffle=shuffle_train),
                   'test': DataLoader(datasets['test'], batch_size=test_batch_size, shuffle=shuffle_test)}
    return dataloaders
