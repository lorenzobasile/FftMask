import matplotlib.pyplot as plt
import timm
import torch
import argparse
import numpy as np
from model import MaskedClf, Mask
from data import get_dataloaders, AdversarialDataset
import os
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='PyTorch ImageNette ADV Finetune')
parser.add_argument('--model', type=str, default='vgg11', help="network architecture")
parser.add_argument('--attack', type=str, default='PGD', help="adversarial attack")
parser.add_argument('--epsilon', type=float, default=0.01, help="epsilon")
parser.add_argument('--data', type=str, default='./data/imagenette2-320/', help='path to dataset')
parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
args = parser.parse_args()
filenames=["lambda_"+str(lam) for lam in [0, 1e-05, 0.0001, 0.001, 0.01]]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists('figures/'+args.attack+str(args.epsilon)):
        os.makedirs('figures/'+args.attack+str(args.epsilon))
#dataloaders = get_dataloaders(data_dir=args.data, train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)
#adv_dataloaders = {'train': DataLoader(AdversarialDataset(None, args.attack, dataloaders['train'], args.epsilon, 'train'), batch_size=args.train_batch_size, shuffle=True),
#                   'test': DataLoader(AdversarialDataset(None, args.attack, dataloaders['test'], args.epsilon, 'test'), batch_size=args.test_batch_size, shuffle=False)}
'''
for filename in filenames:
    base_model = timm.create_model(args.model, pretrained=True, num_classes=10)
    base_model.features[0]=torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    base_model = base_model.to(device)
    m=Mask().to(device)
    model=MaskedClf(m, base_model)
    model.load_state_dict(torch.load("trained_models/"+args.attack+str(args.epsilon)+"/"+filename+".pt"))

    print("Accuracy evaluation")
    correct=0
    correct_adv=0
    for x, x_adv, y in adv_dataloaders['test']:
        x=x.to(device)
        x_adv=x_adv.to(device)
        y=y.to(device)
        out = base_model(x)
        out_adv = base_model(x_adv)
        correct_adv += (torch.argmax(out_adv, axis=1) == y).sum().item()
        correct += (torch.argmax(out, axis=1) == y).sum().item()
    print(f"Clean Accuracy on test set: {correct / len(adv_dataloaders['test'].dataset) * 100:.5f} %")
    print(f"Adversarial Accuracy on test set: {correct_adv / len(adv_dataloaders['test'].dataset) * 100:.5f} %")
    

    plt.figure()
    plt.imshow(np.fft.fftshift(model.mask.weight.detach().cpu().reshape(128,128)), cmap='Blues') #bwr for diverging
    plt.colorbar()
    plt.savefig("figures/"+args.attack+str(args.epsilon)+"/"+filename+".png")

    clean, adv, label = next(iter(adv_dataloaders['test']))
    clean=clean.to(device)
    adv=adv.to(device)
    recon_clean=m(clean[1]).detach().cpu().reshape(128,128)
    recon_adv=m(adv[1]).detach().cpu().reshape(128,128)
    clean=clean[1].detach().cpu().reshape(128,128)
    adv=adv[1].detach().cpu().reshape(128,128)

    print("difference (linf): ", torch.norm(adv-clean, float('inf')))
    print(adv, clean)
    print("difference (l2): ", torch.norm(adv-clean, 2))
    
    plt.figure()
    plt.imshow(recon_clean, cmap='gray')
    plt.savefig("figures/"+args.attack+str(args.epsilon)+"/"+filename+"recon_clean.png")
    plt.figure()
    plt.imshow(recon_adv, cmap='gray')
    plt.savefig("figures/"+args.attack+str(args.epsilon)+"/"+filename+"recon_adv.png")
    plt.figure()
    plt.imshow(clean, cmap='gray')
    plt.savefig("figures/"+args.attack+str(args.epsilon)+"/"+filename+"clean.png")
    plt.figure()
    plt.imshow(adv, cmap='gray')
    plt.savefig("figures/"+args.attack+str(args.epsilon)+"/"+filename+"adv.png")



'''

base_model = timm.create_model(args.model, pretrained=True, num_classes=10)
base_model.features[0]=torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
base_model = base_model.to(device)
correct=0
correct_adv=0
m=Mask().to(device)
model=MaskedClf(m, base_model)
model.load_state_dict(torch.load("trained_models/DF0.1/lambda_1e-05.pt"))

df=np.fft.fftshift(model.mask.weight.detach().cpu().reshape(128,128))


model.load_state_dict(torch.load("trained_models/PGD0.1/lambda_1e-05.pt"))

pgd=np.fft.fftshift(model.mask.weight.detach().cpu().reshape(128,128))

plt.figure()
plt.imshow(pgd-df)
plt.colorbar()
plt.savefig("figures/difference.png")
