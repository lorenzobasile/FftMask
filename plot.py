import matplotlib.pyplot as plt
import timm
import torch
import argparse
import numpy as np
from model import MaskedClf, Mask
import os

parser = argparse.ArgumentParser(description='PyTorch ImageNette ADV Finetune')
parser.add_argument('--model', type=str, default='vgg11', help="network architecture")

args = parser.parse_args()

#filenames=['PGD_epsilon_0.01_lambda_0.01', 'PGD_epsilon_0.01_lambda_0.001', 'PGD_epsilon_0.01_lambda_0.0001', 'PGD_epsilon_0.01_lambda_1e-05', 'PGD_epsilon_0.01_lambda_0']
filename='PGD_INFTY_epsilon_0.01_lambda_1e-05'
filename2='PGD_INFTY_epsilon_0.01_lambda_0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists('figures/'+args.model):
        os.makedirs('figures/'+args.model)
'''
for filename in filenames:
    print(f'\nTraining {args.model} model...')
    base_model = timm.create_model(args.model, pretrained=True, num_classes=10)
    base_model.features[0]=torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    base_model = base_model.to(device)
    correct=0
    correct_adv=0
    m=Mask().to(device)
    model=MaskedClf(m, base_model)
    model.load_state_dict(torch.load("trained_models/"+args.model+"/"+filename+".pt"))

    plt.figure()
    plt.imshow(np.fft.fftshift(model.mask.weight.detach().cpu().reshape(128,128)))
    plt.colorbar()
    plt.savefig("figures/"+args.model+"/"+filename+".png")
'''


base_model = timm.create_model(args.model, pretrained=True, num_classes=10)
base_model.features[0]=torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
base_model = base_model.to(device)
correct=0
correct_adv=0
m=Mask().to(device)
model=MaskedClf(m, base_model)
model.load_state_dict(torch.load("trained_models/"+args.model+"/"+filename+".pt"))

pgd=np.fft.fftshift(model.mask.weight.detach().cpu().reshape(128,128))


model.load_state_dict(torch.load("trained_models/"+args.model+"/"+filename2+".pt"))

fgsm=np.fft.fftshift(model.mask.weight.detach().cpu().reshape(128,128))

plt.figure()
plt.imshow(pgd-fgsm)
plt.colorbar()
plt.savefig("figures/"+args.model+"/difference.png")

