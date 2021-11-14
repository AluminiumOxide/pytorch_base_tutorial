from torch.utils.data import Dataset
from PIL import Image
import os

class LypData(Dataset):
    def __init__(self,root_dir , label_dir):
        self.root_dir = root_dir  # 根目录
        self.label_dir = label_dir  # 每类的名称
        self.path = os.path.join(self.root_dir, self.label_dir)  # 每类的目录
        self.img_path = os.listdir(self.path)  # 每张图片名称

    def __getitem__(self, item):
        img_name = self.img_path[item]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    root_dir = 'dataset/cat_vs_dog/train'
    cat_label_dir = 'cat'
    dataset_obj = LypData(root_dir, cat_label_dir)
