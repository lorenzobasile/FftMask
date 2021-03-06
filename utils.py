import torch
import numpy as np
import matplotlib.pyplot as plt



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


def single(models, x, y, n_epochs, optimizers, lam, idx, path):

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda:0" if next(models[0].parameters()).is_cuda else "cpu")

    n=len(x)
    x=x.to(device)
    y=y.to(device)
    for epoch in range(n_epochs):
        print("Epoch: ", epoch+1, '/', n_epochs)
        l=loss(out, y[i].reshape(1))
        penalty=models[i].mask.weight.abs().sum()
        l+=penalty*lam
        optimizers[i].zero_grad()
        l.backward()
        optimizers[i].step()
        for i in range(n):
            models[i].train()
            out=models[i](x[i])
            if epoch==n_epochs-1:
                correct=torch.argmax(out, axis=1)==y[i]
                mask=np.fft.fftshift(models[i].mask.weight.detach().cpu().reshape(128,128))
                plt.figure()
                plt.imshow(mask, cmap='Blues')
                plt.colorbar()
                plt.savefig(path+"figures/"+str(idx)+("correct" if correct else "wrong")+".png")

                np.save(path+"masks/"+str(idx)+("correct" if correct else "wrong")+".npy", mask)

                '''
                recon_adv=models[i].mask(x_adv[i]).detach().cpu().reshape(128,128)
                adv=x_adv[i].detach().cpu().reshape(128,128)


                plt.figure()
                plt.imshow(recon_adv, cmap='gray')
                plt.savefig("figures/single2/"+"recon"+str(i)+".png")

                plt.figure()
                plt.imshow(adv, cmap='gray')
                plt.savefig("figures/single2/"+"adv"+str(i)+".png")
                '''
                idx+=1





def ADVtrain(model, base_model, adversarytype, dataloaders, n_epochs, optimizer, eps, lam, hybrid=False, scheduler=None):

    clean=[]
    adv=[]
    penalties=[]
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
                penalty=model.mask.weight.abs().sum()
                penalties.append(penalty.detach().cpu().numpy())
                l+=penalty*lam
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                model.mask.weight.data.clamp_(0.)
            out_adv=model(x_adv)
            correct += (torch.argmax(out, axis=1) == y).sum().item()
            correct_adv += (torch.argmax(out_adv, axis=1) == y).sum().item()
            l=loss(out_adv, y)
            penalty=model.mask.weight.abs().sum()
            penalties.append(penalty.detach().cpu().numpy())
            l+=penalty*lam
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            model.mask.weight.data.clamp_(0.)
        clean_acc=correct / len(dataloaders['train'].dataset) * 100
        adv_acc=correct_adv / len(dataloaders['train'].dataset) * 100
        print(f"\n\nClean Accuracy on training set: {clean_acc:.5f} %")
        print(f"Adversarial Accuracy on training set: {adv_acc:.5f} %")
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
        clean_acc=correct / len(dataloaders['test'].dataset) * 100
        adv_acc=correct_adv / len(dataloaders['test'].dataset) * 100
        print(f"Clean Accuracy on test set: {clean_acc:.5f} %")
        print(f"Adversarial Accuracy on test set: {adv_acc:.5f} %")
        clean.append(clean_acc)
        adv.append(adv_acc)
    return clean, adv, penalties
