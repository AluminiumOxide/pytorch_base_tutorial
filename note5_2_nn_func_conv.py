import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = Image.open('dataset/cat_vs_dog/train/cat/cat.0.jpg')  # 打开图像并展示
    plt.figure("input")
    plt.imshow(img)
    trans_totensor = transforms.ToTensor()  # 准备PIL和Tensor之间的转换
    trans_toPIL = transforms.ToPILImage()
    img_tensor = trans_totensor(img)  # PIL转Tensor并设置到对应形状
    img_tensor = torch.reshape(img_tensor, (1, 3, 374, 500))
    filters = torch.randn(1,3,3,3)  # 随机生成卷积核，为了简单，输出通道为1，输入通道和图像一致设置为3
    output = F.conv2d(img_tensor, filters, padding=1)
    output = torch.reshape(output, (1, 374, 500)) # 输出通道如果没reshape，是4维的，转成3维图像并变回PIL
    output = trans_toPIL(output)
    plt.figure("output")  # 输出图像
    plt.imshow(output)
    plt.show()
