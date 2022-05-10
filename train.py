import torch
from deeprobust.image.attack.pgd import PGD


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


def ADVtrain(model, adversarytype, dataloaders, n_epochs, optimizer, eps, scheduler=None):

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    for epoch in range(n_epochs):

        print("Epoch: ", epoch, '/', n_epochs)

        model.train()
        correct=0
        correct_adv=0
        for x, y in dataloaders['train']:
            x=x.to(device)
            y=y.to(device)
            if adversarytype=='FGSM':
                adversary = FGSM(model, 'cuda')
                x_adv = adversary.generate(x, y, epsilon=eps)
            if adversarytype=='PGD':
                adversary = PGD(model, 'cuda')
                x_adv = adversary.generate(x, y, epsilon=eps, step_size=eps/3, num_steps=10)
            out=model(x)
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
        for x, y in dataloaders['test']:
            x=x.to(device)
            y=y.to(device)
            out = model(x)
            if adversarytype=='FGSM':
                out_adv = model(adversary.generate(x, y, epsilon=eps))
            if adversarytype=='PGD':
                out_adv = model(adversary.generate(x, y, epsilon=eps, step_size=eps/3, num_steps=10))
            correct_adv += (torch.argmax(out_adv, axis=1) == y).sum().item()
            correct += (torch.argmax(out, axis=1) == y).sum().item()
        print(f"Clean Accuracy on test set: {correct / len(dataloaders['test'].dataset) * 100:.5f} %")
        print(f"Adversarial Accuracy on test set: {correct_adv / len(dataloaders['test'].dataset) * 100:.5f} %")
