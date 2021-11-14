import torchvision

train_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='./CIFAR10',train=False,download=True)
