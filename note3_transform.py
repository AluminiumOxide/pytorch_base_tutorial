from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import cv2

# 打开图片
img_path = 'dataset/cat_vs_dog/train/cat/cat.0.jpg'
img = Image.open(img_path)

# 创建event文件目录
writer = SummaryWriter('logs')

# ToTensor使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('img_tensor', img_tensor)

# ToPILImage 略

# Normalize
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image('Normalize', img_norm)

# resize
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("resize", img_resize, 0)

# compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([
    trans_resize_2,
    trans_totensor
])
img_resize_2 = trans_compose(img)
writer.add_image("resize",img_resize_2, 1)

# randomcrop
trans_random = transforms.RandomCrop((128,256))
trans_compose_2 = transforms.Compose([
    trans_random,
    trans_totensor
])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("randomcrop", img_crop, i)


writer.close()


