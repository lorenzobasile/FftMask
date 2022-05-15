import torch
from deeprobust.image.attack.pgd import PGD
from deeprobust.image.attack.fgsm import FGSM

from torch.utils.data import TensorDataset, DataLoader, Dataset

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

def train(model, dataloaders, n_epochs, optimizer, scheduler=None):

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    for epoch in range(n_epochs):
        print("Epoch: ", epoch+1, '/', n_epochs)
        model.train()
        for x, y in dataloaders['train']:
            x=x.to(device)
            y=y.to(device)
            out=model(x)
            l=loss(out, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.eval()
        for i in ['train', 'test']:
            correct=0
            with torch.no_grad():
                for x, y in dataloaders[i]:
                    out=model(x.to(device))
                    correct+=(torch.argmax(out, axis=1)==y.to(device)).sum().item()
            print("Accuracy on "+i+" set: ", correct/len(dataloaders[i].dataset))


def ADVtrain(model, base_model, adversarytype, dataloaders, n_epochs, optimizer, eps, hybrid=False, scheduler=None):

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    for epoch in range(n_epochs):
        print("Epoch: ", epoch, '/', n_epochs)
        model.train()
        correct=0
        correct_adv=0
        for x, x_adv, y in dataloaders['train']:
            x=x.to(device)
            x_adv=x_adv.to(device)
            y=y.to(device)
            out=model(x)
            if hybrid:
                l=loss(out, y)
                l+=model.mask.weight.abs().sum()*0.0001
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            out_adv=model(x_adv)
            correct += (torch.argmax(out, axis=1) == y).sum().item()
            correct_adv += (torch.argmax(out_adv, axis=1) == y).sum().item()
            l=loss(out_adv, y)
            l+=model.mask.weight.abs().sum()*0.0001
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        print(f"\n\nClean Accuracy on training set: {correct / len(dataloaders['train'].dataset) * 100:.5f} %")
        print(f"Adversarial Accuracy on training set: {correct_adv / len(dataloaders['train'].dataset) * 100:.5f} %")
        if scheduler is not None:
            scheduler.step()
        model.eval()
        correct_adv=0
        correct=0
        for x, x_adv, y in dataloaders['test']:
            x=x.to(device)
            x_adv=x_adv.to(device)
            y=y.to(device)
            out = model(x)
            out_adv=model(x_adv)
            correct_adv += (torch.argmax(out_adv, axis=1) == y).sum().item()
            correct += (torch.argmax(out, axis=1) == y).sum().item()
        print(f"Clean Accuracy on test set: {correct / len(dataloaders['test'].dataset) * 100:.5f} %")
        print(f"Adversarial Accuracy on test set: {correct_adv / len(dataloaders['test'].dataset) * 100:.5f} %")
