import os
import time
from os.path import exists, join
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.optim as optim
import torch.utils.data as data
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torchvision import transforms

from deepul_helper.models import RNN, MADE, PixelCNN, WaveNet, Transformer


DEVICE = torch.device('cuda')

def train(model, train_loader, optimizer, epoch):
    model.train()

    train_losses = []
    pbar = tqdm(total=len(train_loader.dataset))
    for x, _ in train_loader:
        x = x.to(DEVICE)
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
            loss = model.nll(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)
    avg_loss = avg_loss.item()
    print(f'Test Loss {avg_loss:.4f}')
    return avg_loss


def sample(model, epoch):
    samples = model.sample(64)
    save_image(samples, join('out', name, 'samples', f'epoch{epoch}.png'), nrow=8)


def train_epochs(model, optimizer, train_loader, test_loader):
    start = time.time()
    # sample(model, -1)
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

    if args.model == 'rnn':
        model = RNN(DEVICE, append_loc=False)
    elif args.model == 'rnn_loc':
        model = RNN(DEVICE, append_loc=True)
    elif args.model == 'made':
        if args.order_id == 0:
            # Natural ordering
            ordering = np.arange(784)
        elif args.order_id == 1:
            # Random ordering
            ordering = np.random.permutation(784)
        elif args.order_id == 2:
            # Even then odd indices
            ordering = np.concatenate((np.arange(0, 784, 2), np.arange(1, 784, 2)))
        elif args.order_id == 3:
            # By columns from left to right
            ordering = np.arange(784).reshape(28, 28).T.reshape(-1)
        elif args.order_id == 4:
            # Top to midwayk, then bottom to midway
            ordering = np.concatenate((np.arange(784 // 2), np.arange(784 // 2, 784)[::-1]))
        elif args.order_id == 5:
            # Even rows, then odd rows
            ordering = np.arange(784).reshape(14, 2, 28).transpose((1, 0, 2)).reshape(784)
        else:
            raise Exception('Invalid order_id', args.order_id)
        model = MADE(DEVICE, ordering=ordering)
    elif args.model == 'pixelcnn':
        model = PixelCNN(DEVICE)
    elif args.model == 'wavenet':
        model = WaveNet(DEVICE, append_loc=False)
    elif args.model == 'wavenet_loc':
        model = WaveNet(DEVICE, append_loc=True)
    elif args.model == 'transformer':
        model = Transformer(DEVICE, mode='none')
    elif args.model == 'transformer_loc':
        model = Transformer(DEVICE, mode='pixel_location')
    elif args.model == 'transformer_posenc':
        model = Transformer(DEVICE, mode='pos_encoding')
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
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--model', type=str, default='rnn', help='rnn | rnn_loc | made | pixelcnn')
    parser.add_argument('--order_id', type=int, default=0, help='MADE order id')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    args = parser.parse_args()

    if args.model == 'made':
        name = f'{args.model}_{args.order_id}'
    else:
        name = args.model

    main()

