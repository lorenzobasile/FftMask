import matplotlib.pyplot as plt
import timm
import torch
import argparse
import numpy as np
from data import get_dataloaders
from model import MaskedClf, Mask

parser = argparse.ArgumentParser(description='PyTorch ImageNette ADV Finetune')
parser.add_argument('--model', type=str, default='vgg11', help="network architecture")
parser.add_argument('--data', type=str, default='./data/imagenette2-320/', help='path to dataset')
parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

args = parser.parse_args()

eps=0.01

filenames=['FGSMvgg11lambda0.01_2', 'FGSMvgg11lambda0.001_2', 'FGSMvgg11lambda0.0001_2', 'FGSMvgg11lambda1e-05_2']

# get dataloaders
dataloaders = get_dataloaders(data_dir=args.data,
                              train_batch_size=args.train_batch_size,
                              test_batch_size=args.test_batch_size)

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for filename in filenames:
    print(f'\nTraining {args.model} model...')
    base_model = timm.create_model(args.model, pretrained=True, num_classes=10)
    base_model.features[0]=torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    base_model = base_model.to(device)
    base_model.load_state_dict(torch.load("trained_models/"+ args.model + ".pt"))
    correct=0
    correct_adv=0
    m=Mask().to(device)
    model=MaskedClf(m, base_model)
    model.load_state_dict(torch.load("trained_models/"+filename+".pt"))

    plt.figure()
    plt.imshow(np.fft.fftshift(model.mask.weight.detach().cpu().reshape(128,128)))
    plt.colorbar()
    plt.savefig(filename+".png")
