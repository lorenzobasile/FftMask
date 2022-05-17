import os
from deeprobust.image.attack.pgd import PGD
from deeprobust.image.attack.fgsm import FGSM
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset

def get_dataloaders(data_dir, train_batch_size, test_batch_size, data_transforms=None, shuffle_train=False, shuffle_test=False):

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

class AdversarialDataset(Dataset):

    def __init__(self, model, adversarytype, dataloader, eps):
        self.clean_imgs=torch.empty(0,1,128,128)
        self.adv_imgs=torch.empty(0,1,128,128)
        self.labels=torch.empty(0, dtype=torch.int64)
        device=torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
        adv_xy_list = []
        for x, y in dataloader:
            x=x.to(device)
            y=y.to(device)
            if adversarytype=='FGSM':
                adversary = FGSM(model, 'cuda')
                x_adv = adversary.generate(x, y, epsilon=eps)
            if adversarytype=='PGD':
                adversary = PGD(model, 'cuda')
                x_adv = adversary.generate(x, y, epsilon=eps, step_size=eps/3, num_steps=10)
            self.clean_imgs=torch.cat((self.clean_imgs, x.detach().cpu()))
            self.adv_imgs=torch.cat((self.adv_imgs, x_adv.detach().cpu()))
            self.labels=torch.cat((self.labels, y.detach().cpu()))
            self.labels.type(torch.LongTensor)

    def __len__(self):
        return len(self.clean_imgs)

    def __getitem__(self, idx):
        return self.clean_imgs[idx], self.adv_imgs[idx], self.labels[idx]
