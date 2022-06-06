import os
import foolbox as fb
import torchvision
import torch
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
        c="advdata/clean"+adversarytype+".pt"
        a="advdata/adv"+adversarytype+".pt"
        l="advdata/lbl"+adversarytype+".pt"
        if os.path.isfile(c) and os.path.isfile(a) and os.path.isfile(l):
            self.clean_imgs=torch.load(c)
            self.adv_imgs=torch.load(a)
            self.labels=torch.load(l)
            return
        self.clean_imgs=torch.empty(0,1,128,128)
        self.adv_imgs=torch.empty(0,1,128,128)
        self.labels=torch.empty(0, dtype=torch.int64)
        device=model.device
        for k, (x, y) in enumerate(dataloader):
            print("batch ", k)
            x=x.to(device)
            y=y.to(device)
            if adversarytype=='FGSM':
                adversary = fb.attacks.FGSM()
            elif adversarytype=='PGD':
                adversary = fb.attacks.PGD(steps=10, abs_stepsize=eps/3)
            else:
                adversary = fb.attacks.L2CarliniWagnerAttack()
            x_adv, clipped, is_adv = adversary(model, x, y, epsilons=eps)
            self.clean_imgs=torch.cat((self.clean_imgs, x.detach().cpu()))
            self.adv_imgs=torch.cat((self.adv_imgs, x_adv.detach().cpu()))
            self.labels=torch.cat((self.labels, y.detach().cpu()))
            self.labels.type(torch.LongTensor)
        torch.save(self.clean_imgs, c)
        torch.save(self.adv_imgs, a)
        torch.save(self.labels, l)
    def __len__(self):
        return len(self.clean_imgs)

    def __getitem__(self, idx):
        return self.clean_imgs[idx], self.adv_imgs[idx], self.labels[idx]
