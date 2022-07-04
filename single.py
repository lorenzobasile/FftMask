import foolbox
from utils import ADVtrain, single
import timm
import torch
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

from data import get_dataloaders, AdversarialDataset
from model import MaskedClf, Mask

parser = argparse.ArgumentParser(description='PyTorch ImageNette adversarial evaluation and training')
parser.add_argument('--model', type=str, default='vgg11', help="network architecture")
parser.add_argument('--attack', type=str, default='PGD', help="adversarial attack")
parser.add_argument('--epsilon', type=float, default=0.01, help="epsilon")
parser.add_argument('--lam', type=float, default=0.01, help="lambda")
parser.add_argument('--data', type=str, default='./data/imagenette2-320/', help='path to dataset')
parser.add_argument('--train_batch_size', type=int, default=4, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='test batch size')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

args = parser.parse_args()

eps=args.epsilon

dataloaders = get_dataloaders(data_dir=args.data, train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_model = timm.create_model(args.model, pretrained=True, num_classes=10)
base_model.features[0]=torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
base_model = base_model.to(device)
base_model.load_state_dict(torch.load("trained_models/clean.pt"))
base_model.eval()
fmodel = foolbox.models.PyTorchModel(base_model, bounds=(-np.inf,np.inf))

adv_dataloaders = {'train': DataLoader(AdversarialDataset(fmodel, args.attack, dataloaders['train'], args.epsilon, 'train'), batch_size=args.train_batch_size, shuffle=False),
                   'test': DataLoader(AdversarialDataset(fmodel, args.attack, dataloaders['test'], args.epsilon, 'test'), batch_size=args.test_batch_size, shuffle=False)}

print(len(dataloaders['train'].dataset), len(dataloaders['test'].dataset), len(adv_dataloaders['train'].dataset), len(adv_dataloaders['test'].dataset))


models=[]
optimizers=[]

for i in range(args.test_batch_size):
    model=MaskedClf(Mask().to(device), base_model)
    for p in model.clf.parameters():
        p.requires_grad=False
    models.append(model)
    optimizers.append(torch.optim.Adam(model.parameters(), lr=0.01))
    
single(models,  adv_dataloaders, 100, optimizers, args.lam)
