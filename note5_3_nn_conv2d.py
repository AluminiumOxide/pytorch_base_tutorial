import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),download=False)

dataloader = DataLoader(dataset, batch_size=64)

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.conv1(x)
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

