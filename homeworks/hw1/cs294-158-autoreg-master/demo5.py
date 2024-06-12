import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from deepul_helper.models import MaskConv2d

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class PixelCNN(nn.Module):
    name = 'PixelCNN'
    def __init__(self, n_layers):
        super().__init__()
        model = [MaskConv2d('A', 1, 1, 3, padding=1)]
        for _ in range(n_layers):
            model.extend([MaskConv2d('B', 1, 1, 3, padding=1)])
        model.append(MaskConv2d('B', 1, 1, 3,padding=1))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x)

class GatedConv2d(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, k=3, padding=1):
        super().__init__()
        self.vertical = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=k,
                                  padding=padding, bias=False)
        self.horizontal = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, k),
                                    padding=(0, padding), bias=False)
        self.vtohori = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1, bias=False)
        self.hori_res = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        self.vmask[:, :, k // 2 + 1:, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2 + 1:] = 0
        if mask_type == 'A':
            self.hmask[:, :, :, k // 2] = 0

    def gate(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.tanh(x1) * torch.sigmoid(x2)

    def down_shift(self, x):
        x = x[:, :, :-1, :]
        pad = nn.ZeroPad2d((0, 0, 1, 0))
        return pad(x)

    def forward(self, x):
        vx, hx = x.chunk(2, dim=1)

        self.vertical.weight.data *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx = self.vertical(vx)
        new_hx = self.horizontal(hx)
        new_hx = new_hx + self.vtohori(self.down_shift(vx))

        vx = self.gate(vx)
        new_hx = self.gate(new_hx)
        hx = hx + self.hori_res(new_hx)

        return torch.cat((vx, hx), dim=1)

class GatedPixelCNN(nn.Module):
    name = 'StackedPixelCNN'
    def __init__(self, n_layers):
        super().__init__()
        model = [GatedConv2d('A', 1, 1, 3, padding=1)]
        for _ in range(n_layers):
            model.extend([GatedConv2d('B', 1, 1, 3, padding=1)])
        model.append(GatedConv2d('B', 1, 1, 3, padding=1))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(torch.cat((x, x), dim=1)).chunk(2, dim=1)[1]


def plot_receptive_field(model, data):
    out = model(data)
    out[0, 0, 5, 5].backward()
    print(data.grad)
    grad = data.grad.detach().cpu().numpy()[0, 0]
    grad = np.abs(grad)
    grad = np.minimum(grad, np.median(grad[grad > 0]))
    if np.max(grad) != 0:
        grad /= np.max(grad)

    plt.figure()
    plt.imshow(grad)
    plt.title(f'Gradient wrt pixel (5, 5), {model.name} {n_layers} layers')
    plt.show()


x = torch.randn(1, 1, 10, 10).cuda()
x.requires_grad = True

for i, n_layers in enumerate([1, 5, 10, 50]):
    pixelcnn = PixelCNN(n_layers=n_layers).cuda()
    plot_receptive_field(pixelcnn, x)
    x.grad.zero_()

for i, n_layers in enumerate([1, 5, 10, 50]):
    gated_pixelcnn = GatedPixelCNN(n_layers=n_layers).cuda()
    plot_receptive_field(gated_pixelcnn, x)
    x.grad.zero_()
