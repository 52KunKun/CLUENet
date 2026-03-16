from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt,label_txt,data_root, transform, loader=default_loader):
        label_fh = open(label_txt,'r')
        self.label_dict = {}
        index = 0
        for line in label_fh:
            line = line.strip('\n')
            line = line.rstrip()
            self.label_dict[line] = index
            index+=1  # 建立标签

        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split('/')[0]
            imgs.append((line+".jpg", self.label_dict[words]))
        self.data_root = data_root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.data_root, fn))

        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)