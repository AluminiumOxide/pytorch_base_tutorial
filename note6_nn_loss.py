import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset, batch_size=1)


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.model = Sequential(
            Conv2d(3, 16, 5),
            MaxPool2d(2, 2),
            Conv2d(16, 32, 5),
            MaxPool2d(2, 2),
            Flatten(),  # 注意一下,线性层需要进行展平处理
            Linear(32*5*5, 120),
            Linear(120, 84),
            Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    loss = nn.CrossEntropyLoss()
    module = Module()
    for data in dataloader:
        imgs, targets = data
        outputs = module(imgs)
        result_loss = loss(outputs, targets)

        print(' pedict:{}\n targets:{}\n loss:{}\n'.format(outputs.detach().numpy(),targets,result_loss))
