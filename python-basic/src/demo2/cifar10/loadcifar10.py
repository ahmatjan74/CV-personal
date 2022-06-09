from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

label_name = ['airplane', 'automobile', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse',
              'ship', 'truck']
label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx;
print(label_dict)


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataSet(Dataset):
    def __int__(self, im_list, transform=None, loader=default_loader):
        super(MyDataSet, self).__int__()
        imgs = []
        for im_item in im_list:
            im_label_name = im_item.split('/')[-2]
            imgs.append([im_item, label_dict[im_label_name]])
    #     todo:

    def __getitem__(self, item):
        raise

    def __len__(self):
        raise