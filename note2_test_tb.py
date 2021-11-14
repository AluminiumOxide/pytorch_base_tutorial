from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

image_path = 'dataset/cat_vs_dog/train/cat/cat.0.jpg' # 图像目录
img_PIL = Image.open(image_path)  # 打开图片文件（PILimage）
img_array = np.array(img_PIL)    # 转成numpy格式
print(type(img_array))
print(img_array.shape)  # (374, 500, 3)
writer.add_image('cat', img_array, 1, dataformats='HWC')

x = range(100)
for i in x:
    writer.add_scalar('y=2x', i * 2, i)

writer.close()


# from PIL import Image
# import numpy as np
#
# image_path = 'dataset/cat_vs_dog/train/cat/cat.0.jpg'
# img = Image.open(image_path)
# print(type(img))
# img_array = np.array(img)
# print(type(img_array))
