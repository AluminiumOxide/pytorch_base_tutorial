import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_set = torchvision.datasets.CIFAR10(root='./CIFAR10',train=True, download=False)
# 测试集
test_set = torchvision.datasets.CIFAR10(root='./CIFAR10',train=False, transform=torchvision.transforms.ToTensor(), download=False)

img,target = test_set[0]
print(img.shape)
print(target)

test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter('logs')
step = 0
for data in test_loader:
    imgs,target = data
    print(img.shape)
    print(target)
    writer.add_images("test_data",imgs, step)
    step = step+1

writer.close()



