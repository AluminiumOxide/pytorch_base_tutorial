import torch
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
    module = Module()
    loss = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        module = module.cuda()
        loss = loss.cuda()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(12):
        running_loss = 0.0

        for imgs, targets in dataloader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            outputs = module(imgs)
            result_loss = loss(outputs, targets)
            result_loss.backward()
            optimizer.step()


            running_loss = running_loss + result_loss
        print(running_loss)




# def closure():
#     optimizer.zero_grad()
#     outputs = module(imgs)
#     result_loss = loss(outputs, targets)
#     result_loss.backward()
#     return loss
#
# optimizer.step(closure)