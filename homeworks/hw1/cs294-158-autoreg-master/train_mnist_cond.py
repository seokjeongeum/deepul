import os
from os.path import exists, join
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torchvision import transforms

from deepul_helper.models import PixelCNN
from deepul_helper.utils import to_one_hot


DEVICE = torch.device('cuda')

def create_cond_input(x, y):
    if args.cond_mode == 'class':
        return to_one_hot(y, 10, DEVICE).float()
    elif args.cond_mode == 'image':
        return F.interpolate(x, scale_factor=0.25, mode='bilinear')


def train(model, train_loader, optimizer, epoch):
    model.train()

    train_losses = []
    pbar = tqdm(total=len(train_loader.dataset))
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        cond = create_cond_input(x, y)
        loss = model.nll(x, cond=cond)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(x.shape[0])
        train_losses.append(loss.item())
        avg_loss = np.mean(train_losses[-50:])
        pbar.set_description(f'Epoch {epoch}, Train Loss {avg_loss:.4f}')
    pbar.close()


def eval(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            cond = create_cond_input(x, y)
            loss = model.nll(x, cond=cond)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)
    avg_loss = avg_loss.item()
    print(f'Test Loss {avg_loss:.4f}')


def sample(model, test_loader, epoch):
    if args.cond_mode == 'class':
        cond = torch.arange(10).unsqueeze(1).repeat(1, 10).view(-1).to(DEVICE)
        cond = to_one_hot(cond, 10, DEVICE)
        samples = model.sample(100, cond=cond)
        save_image(samples, join('out', name, 'samples', f'epoch{epoch}.png'), nrow=10)
    elif args.cond_mode == 'image':
        x = next(iter(test_loader))[0][:32]
        cond = F.interpolate(x, scale_factor=0.25, mode='bilinear').to(DEVICE)
        samples = model.sample(32, cond=cond)
        cond = F.interpolate(cond, scale_factor=4, mode='bilinear').cpu()
        images = torch.stack((cond, samples), dim=1)
        images = images.view(-1, *images.shape[2:])
        save_image(images, join('out', name, 'samples', f'epoch{epoch}.png'), nrow=8)


def train_epochs(model, optimizer, train_loader, test_loader):
    sample(model, test_loader, -1)
    torch.save(model.state_dict(), join('out', name, 'checkpoints', f'epoch-1_state_dict'))
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, epoch)
        eval(model, test_loader)
        sample(model, test_loader, epoch)
        torch.save(model.state_dict(), join('out', name, 'checkpoints', f'epoch{epoch}_state_dict'))


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x > 0.5).float()
    ])
    train_dset = MNIST('data', transform=transform, train=True, download=True)
    test_dset = MNIST('data', transform=transform, train=False, download=True)

    train_loader = data.DataLoader(train_dset, batch_size=args.bs, shuffle=True,
                                   pin_memory=True, num_workers=2)
    test_loader = data.DataLoader(test_dset, batch_size=args.bs, shuffle=True,
                                  pin_memory=True, num_workers=2)

    return train_loader, test_loader

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not exists(join('out', name)):
        os.makedirs(join('out', name, 'samples'))
        os.makedirs(join('out', name, 'checkpoints'))

    if args.cond_mode == 'class':
        conditional_size = (10,)
    elif args.cond_mode == 'image':
        conditional_size = (1, 7, 7)
    else:
        raise Exception('Invalid cond_mode', args.cond_mode)

    if args.model == 'pixelcnn':
        model = PixelCNN(DEVICE, conditional_size=conditional_size)
    else:
        raise Exception('Invalid model', args.model)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader, test_loader = load_data()

    dset_examples = next(iter(train_loader))[0]
    save_image(dset_examples, join('out', name, 'dset_examples.png'))

    train_epochs(model, optimizer, train_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--model', type=str, default='pixelcnn', help='pixelcnn')
    parser.add_argument('--cond_mode', type=str, default='class', help='class | image')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    args = parser.parse_args()

    name = f'{args.model}_{args.cond_mode}'
    main()

