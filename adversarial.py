import foolbox
from utils import ADVtrain
import timm
import torch
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data import get_dataloaders, AdversarialDataset
from model import MaskedClf, Mask

parser = argparse.ArgumentParser(description='PyTorch ImageNette adversarial evaluation and training')
parser.add_argument('--model', type=str, default='vgg11', help="network architecture")
parser.add_argument('--attack', type=str, default='PGD', help="adversarial attack")
parser.add_argument('--epsilon', type=float, default=0.01, help="epsilon")
parser.add_argument('--data', type=str, default='./data/imagenette2-320/', help='path to dataset')
parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
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

adv_dataloaders = {'train': DataLoader(AdversarialDataset(base_model, args.attack, dataloaders['train'], args.epsilon, 'train'), batch_size=args.train_batch_size, shuffle=True),
                   'test': DataLoader(AdversarialDataset(base_model, args.attack, dataloaders['test'], args.epsilon, 'test'), batch_size=args.test_batch_size, shuffle=False)}

print(len(dataloaders['train'].dataset), len(dataloaders['test'].dataset), len(adv_dataloaders['train'].dataset), len(adv_dataloaders['test'].dataset))


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
'''
lambdas=[0, 1e-5, 1e-4]

for lam in lambdas:
    print("L1 penalty: ", lam)

    m=Mask().to(device)
    model=MaskedClf(m, base_model)

    for p in model.clf.parameters():
        p.requires_grad=False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                args.lr,
                epochs=args.epochs,
                steps_per_epoch=len(dataloaders['train']),
                pct_start=0.1
            )

    clean, adv, penalties=ADVtrain(model, model.clf, args.attack, adv_dataloaders, args.epochs, optimizer, args.epsilon, lam, hybrid=True, scheduler=scheduler)
    torch.save(model.state_dict(), "trained_models/"+args.attack+"/"+str(args.epsilon)+"_lambda_"+str(lam)+".pt")

    plt.figure()
    plt.semilogy(penalties)
    plt.savefig("figures/"+args.attack+"/"+str(args.epsilon)+"_lambda_"+str(lam)+"_penalty.png")
'''
