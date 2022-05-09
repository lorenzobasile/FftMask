from deeprobust.image.attack.pgd import PGD
from train import ADVtrain

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
base_model = timm.create_model(args.model, pretrained=True, num_classes=10)
base_model.features[0]=torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
base_model = base_model.to(device)
base_model.load_state_dict(torch.load("trained_models/"+ args.model + ".pt"))
adversary = PGD(base_model, 'cuda')

for x, y in dataloaders['test']:
    x=x.to(device)
    y=y.to(device)
    out = base_model(x)
    out_adv = base_model(adversary.generate(x, y, epsilon=eps, step_size=eps/3, num_steps=10))
    correct_adv += (torch.argmax(out_adv, axis=1) == y).sum().item()
    correct += (torch.argmax(out, axis=1) == y).sum().item()
print(f"Clean Accuracy on test set: {correct / len(dataloaders['test'].dataset) * 100:.5f} %")
print(f"Adversarial Accuracy on test set: {correct_adv / len(dataloaders['test'].dataset) * 100:.5f} %")


'''
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(dataloaders['train']),
            pct_start=0.1
        )

ADVtrain(base_model, dataloaders, args.epochs, optimizer, scheduler)
torch.save(model.state_dict(), "trained_models/"+ args.model + ".pt")
'''
