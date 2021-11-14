import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, download=True,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        x = self.maxpool(x)
        return x


if __name__ == '__main__':
    module = Module()
    writer = SummaryWriter("logs")
    for step,data in enumerate(dataloader):
        imgs, targets = data
        output = module(imgs)
        print('input shape: {} output shape: {}'.format(imgs.shape,output.shape))

        writer.add_images("input", imgs, step)
        writer.add_images("output", output, step)

    writer.close()
