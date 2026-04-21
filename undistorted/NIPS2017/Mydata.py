from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import os
class ImageNetDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.file.ImageId.iloc[idx] + '.png')
        image = Image.open(img_name)
        image = self.transform(image)
        labels = self.file.TrueLabel.iloc[idx]
        labels = np.array(labels)
        return image, labels - 1


def channel_F_to_L(img):
    img = img.swapaxes(0, 2)
    img = img.swapaxes(0, 1)
    return img

def channel_L_to_F(img):
    img = img.swapaxes(0, 2)
    img = img.swapaxes(1, 2)
    return img


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

