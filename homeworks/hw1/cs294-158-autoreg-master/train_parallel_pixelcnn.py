import os
import time
from os.path import exists, join
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from torchvision import datasets

from deepul_helper.models import ParallelPixelCNN


DEVICE = torch.device('cuda')

def process_x(x):
    x = F.interpolate(x, scale_factor=8)
    return (x > 0.5).float()


def expand_subsampled(samples):
    images = []
    for i in range(5):
        stride = 2 ** (5 - i)
        img = samples[:, :, ::stride, ::stride]
        img = F.interpolate(img, size=(224, 224))
        images.append(img)
    images.append(samples)
    images = torch.stack(images, dim=1).view(-1, 1, 224, 224)
    return images



def train(model, train_loader, optimizer, epoch):
    model.train()

    train_losses = []
    pbar = tqdm(total=len(train_loader.dataset))
    for x, _ in train_loader:
        x = x.to(DEVICE)
        x = process_x(x)
        loss = model.nll(x)
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
        for x, _ in data_loader:
            x = x.to(DEVICE)
            x = process_x(x)
            loss = model.nll(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)
    avg_loss = avg_loss.item()
    print(f'Test Loss {avg_loss:.4f}')
    return avg_loss


def sample(model, epoch):
    samples = model.sample(10)
    images = expand_subsampled(samples)
    save_image(images, join('out', name, 'samples', f'epoch{epoch}.png'), nrow=6)


def train_epochs(model, optimizer, train_loader, test_loader):
    start = time.time()
    sample(model, -1)
    print('Sampling took', time.time() - start, 'seconds')
    torch.save(model.state_dict(), join('out', name, 'checkpoints', f'epoch-1_state_dict'))
    test_losses = []
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, epoch)
        test_loss = eval(model, test_loader)
        sample(model, epoch)
        torch.save(model.state_dict(), join('out', name, 'checkpoints', f'epoch{epoch}_state_dict'))
        test_losses.append(test_loss)
    np.save(join('out', name, 'test_losses.npy'), test_losses)


def load_data():
    train_dset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_dset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

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

    model = ParallelPixelCNN(DEVICE)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader, test_loader = load_data()

    dset_examples = next(iter(train_loader))[0][:10]
    dset_examples = process_x(dset_examples)
    dset_examples = expand_subsampled(dset_examples)
    save_image(dset_examples, join('out', name, 'dset_examples.png'), nrow=6)

    train_epochs(model, optimizer, train_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=7e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    args = parser.parse_args()

    name = 'parallel_pixelcnn'
    main()

