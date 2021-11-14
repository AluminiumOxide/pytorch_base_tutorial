import torchvision
import torch
from torch import nn


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),  # 注意一下,线性层需要进行展平处理
            nn.Linear(32*5*5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


module = Module()
# 保存方式1,模型结构+张量
torch.save(module, "module.pth")
# 保存方式2，张量（推荐）
torch.save(module.state_dict(), "module_state_dict.pth")

# 加载方式1 对应保存方式1,同时加载模型结构+张量
load_module = torch.load("module.pth")
# 加载方式2 对应保存方式2,加载模型后加载张量(必须先实例化模型)
module.load_state_dict(torch.load("module_state_dict.pth"))
print(module)





