import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask()

    def forward(self, input):
        return F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def create_mask(self):
        k = self.kernel_size[0]
        self.mask[:, :, :k // 2] = 1
        self.mask[:, :, k // 2, :k // 2] = 1


class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        kernel_size_1 = (kernel_size // 2, kernel_size)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size_1)
        self.pad1 = nn.ZeroPad2d((
            kernel_size_1[1] // 2, # pad left
            kernel_size_1[1] // 2, # pad right
            kernel_size_1[0], # pad top
            0 # pad bottom
        ))

        kernel_size_2 = (1, kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size_2, bias=False)
        self.pad2 = nn.ZeroPad2d((
            kernel_size_2[1],
            0,
            0,
            0
        ))

    def forward(self, x):
        out1 = self.conv1(self.pad1(x))[:, :, :-1, :]
        out2 = self.conv2(self.pad2(x))[:, :, :, :-1]
        return out1 + out2

x = torch.randn(128, 32, 64, 64).cuda() # Initialize random input data

def speed_test(conv, n_trials=100):
    conv(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_trials):
        conv(x)
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return elapsed

speedups = []
ks = list(range(3, 21, 2))
for k in ks:
    conv_with_mask = MaskConv2d(32, 32, kernel_size=k, padding=k // 2).cuda()
    conv_no_mask = CausalConv2d(32, 32, k).cuda()

    time_mask = speed_test(conv_with_mask)
    time_no_mask = speed_test(conv_no_mask)

    speedup = time_mask / time_no_mask
    speedups.append(speedup)

    print(f'Kernel Size {k}, Speedup {speedup:.2f}x')

plt.figure()
plt.plot(ks, speedups)
plt.title('Padded Convolutions vs Masked Convolutions')
plt.xlabel('kernel size')
plt.ylabel('Speedup of Padded over Masked')








