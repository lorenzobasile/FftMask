import timm
import torch
import argparse
from data import get_dataloaders
from train import train

parser = argparse.ArgumentParser(description='PyTorch ImageNette Fimetune')
parser.add_argument('--model', type=str, default='vgg11', help="network architecture")
parser.add_argument('--data', type=str, default='./data/imagenette2-320/', help='path to dataset')
parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

args = parser.parse_args()

# get dataloaders
dataloaders = get_dataloaders(data_dir=args.data,
                              train_batch_size=args.train_batch_size,
                              test_batch_size=args.test_batch_size)

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print(f'\nTraining {args.model} model...')
model = timm.create_model(args.model, pretrained=True, num_classes=10)
model.features[0]=torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(dataloaders['train']),
            pct_start=0.1
        )

train(model, dataloaders, args.epochs, optimizer, scheduler)
torch.save(model.state_dict(), "trained_models/"+ args.model + ".pt")
