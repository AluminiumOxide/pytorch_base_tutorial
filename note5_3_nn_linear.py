import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.linear = Linear(196608, 10) # 32*32*3*64

    def forward(self, x):
        x = torch.flatten(x)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    module = Module()
    for step,data in enumerate(dataloader):
        imgs, targets = data
        output = module(imgs)
        print('input shape: {} output shape: {} which is {}'.format(imgs.shape,output.shape,output))

