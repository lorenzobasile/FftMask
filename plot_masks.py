import matplotlib.pyplot as plt
import timm
import torch
import argparse
import numpy as np
from model import Classifier, MaskedClf, Mask
from data import get_dataloaders, AdversarialDataset
import os
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='PyTorch ImageNette ADV Finetune')
parser.add_argument('--model', type=str, default='vgg11', help="network architecture")
parser.add_argument('--attack1', type=str, default='PGD', help="adversarial attack")
parser.add_argument('--epsilon1', type=float, default=0.01, help="epsilon")
parser.add_argument('--lambda1', type=float, default=0, help='lasso coef')
parser.add_argument('--attack2', type=str, default='PGD', help="adversarial attack")
parser.add_argument('--epsilon2', type=float, default=0.01, help="epsilon")
parser.add_argument('--lambda2', type=float, default=0, help='lasso coef')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path='figures/vs/'+args.attack1+"eps"+str(args.epsilon1)+"lam"+str(args.lambda1)+args.attack2+"eps"+str(args.epsilon2)+"lam"+str(args.lambda2)

if not os.path.exists(path):
        os.makedirs(path)
base_model = Classifier(args.model)
base_model = base_model.to(device)
m=Mask().to(device)
model=MaskedClf(m, base_model)
model.load_state_dict(torch.load("trained_models/"+args.attack1+"/"+str(args.epsilon1)+"/lambda_"+str(args.lambda1)+".pt"))

mask1=np.fft.fftshift(model.mask.weight.detach().cpu().reshape(128,128))


model.load_state_dict(torch.load("trained_models/"+args.attack2+"/"+str(args.epsilon2)+"/lambda_"+str(args.lambda2)+".pt"))

mask2=np.fft.fftshift(model.mask.weight.detach().cpu().reshape(128,128))

plt.figure()
plt.imshow(mask1, cmap='Blues')
plt.colorbar()
plt.savefig(path+"/mask1.png")

plt.figure()
plt.imshow(mask2, cmap='Blues')
plt.colorbar()
plt.savefig(path+"/mask2.png")

plt.figure()
plt.imshow(mask2-mask1, cmap='bwr')
plt.colorbar()
plt.savefig(path+"/mask2-mask1.png")
