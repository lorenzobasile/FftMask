import matplotlib.pyplot as plt
import timm
import torch
import argparse
import numpy as np
from model import Classifier, MaskedClf, Mask
from data import get_dataloaders, AdversarialDataset
import os
from torch.utils.data import DataLoader



parser = argparse.ArgumentParser(description='PyTorch ImageNette adversarial evaluation and training')
parser.add_argument('--model', type=str, default='vgg11', help="network architecture")
parser.add_argument('--attack', type=str, default='PGD', help="adversarial attack")
parser.add_argument('--epsilon', type=float, default=0.01, help="epsilon")
parser.add_argument('--data', type=str, default='./data/imagenette2-320/', help='path to dataset')
parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
parser.add_argument('--N', type=int, default=5, help='images to save')


args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists('figures/'+args.attack+"/"+str(args.epsilon)):
        os.makedirs('figures/'+args.attack+"/"+str(args.epsilon))
dataloaders = get_dataloaders(data_dir=args.data, train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)
adv_dataloaders = {'train': DataLoader(AdversarialDataset(None, args.attack, dataloaders['train'], args.epsilon, 'train'), batch_size=args.train_batch_size, shuffle=True),
                   'test': DataLoader(AdversarialDataset(None, args.attack, dataloaders['test'], args.epsilon, 'test'), batch_size=args.test_batch_size, shuffle=False)}
filenames=["lambda_"+str(lam) for lam in [0, 1e-05, 0.0001, 0.001, 0.01]]

ps=[0, 1, 2, float('inf')]

for filename in filenames:
    base_model = Classifier(args.model).to(device)
    m=Mask().to(device)
    model=MaskedClf(m, base_model)
    model.load_state_dict(torch.load("trained_models/"+args.attack+"/"+str(args.epsilon)+"/"+filename+".pt"))

    correct=0
    correct_adv=0
    correct_masked=0
    correct_adv_masked=0
    batch_norm=torch.zeros(len(adv_dataloaders['test']), len(ps))
    for b, (x, x_adv, y) in enumerate(adv_dataloaders['test']):
        x=x.to(device)
        x_adv=x_adv.to(device)
        y=y.to(device)
        out = base_model(x)
        out_adv = base_model(x_adv)
        out_masked=model(x)
        out_adv_masked=model(x_adv)
        for i, p in enumerate(ps):
            batch_norm[b, i]=torch.norm(x_adv-x, p)
        correct_adv += (torch.argmax(out_adv, axis=1) == y).sum().item()
        correct += (torch.argmax(out, axis=1) == y).sum().item()
        correct_adv_masked += (torch.argmax(out_adv_masked, axis=1) == y).sum().item()
        correct_masked += (torch.argmax(out_masked, axis=1) == y).sum().item()

    for i, p in enumerate(ps):
        print(f"Avg {p} norm of the attack: {torch.mean(batch_norm, axis=0)}")
    print(f"Clean Accuracy on test set: {correct / len(adv_dataloaders['test'].dataset) * 100:.5f} %")
    print(f"Adversarial Accuracy on test set: {correct_adv / len(adv_dataloaders['test'].dataset) * 100:.5f} %")

    print(f"Clean Accuracy on test set (after mask training): {correct_masked / len(adv_dataloaders['test'].dataset) * 100:.5f} %")
    print(f"Adversarial Accuracy on test set (after mask training): {correct_adv_masked / len(adv_dataloaders['test'].dataset) * 100:.5f} %")

    clean, adv, label = next(iter(adv_dataloaders['test']))
    clean=clean.to(device)
    adv=adv.to(device)

    for n in range(args.N):

        recon_clean=m(clean[n]).detach().cpu().reshape(128,128)
        recon_adv=m(adv[n]).detach().cpu().reshape(128,128)
        clean=clean[n].detach().cpu().reshape(128,128)
        adv=adv[n].detach().cpu().reshape(128,128)


        plt.figure()
        plt.imshow(recon_clean, cmap='gray')
        plt.savefig("figures/"+args.attack+"/"+str(args.epsilon)+"/"+filename+str(n)+"recon_clean.png")
        plt.figure()
        plt.imshow(recon_adv, cmap='gray')
        plt.savefig("figures/"+args.attack+"/"+str(args.epsilon)+"/"+filename+str(n)+"recon_adv.png")
        plt.figure()
        plt.imshow(clean, cmap='gray')
        plt.savefig("figures/"+args.attack+"/"+str(args.epsilon)+"/"+str(n)+"clean.png")
        plt.figure()
        plt.imshow(adv, cmap='gray')
        plt.savefig("figures/"+args.attack+"/"+str(args.epsilon)+"/"+str(n)+"adv.png")
