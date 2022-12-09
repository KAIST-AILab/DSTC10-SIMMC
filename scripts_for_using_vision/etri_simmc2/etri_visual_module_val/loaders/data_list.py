import numpy as np
import os
import os.path
from PIL import Image
import torch
from torch.utils.data import Dataset

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1:]
            label[-1] = label[-1].strip()
            label_list.append(list(map(int, label)))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


def Imageloader(image_path, transform=None):
    img = pil_loader(image_path)
    img = transform(img)
    return img


class Imagelists(object):
    def __init__(self, image_list, root="./data",
                 transform=None, target_transform=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)
        

class ImagelistsAllDomain(Dataset):
    def __init__(self, image_list, root="./data",
                 transform=None, target_transform=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list)
        # print('label prior:', labels)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, i):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[i])
        target = self.labels[i]
        if len(target) < 5:  # 5: number of attributes of clothes
            target += [-1] * (5 - len(target))
            # target = np.concatenate([target, np.array([-1] * (5 - len(target)))])
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, torch.tensor(target, dtype=torch.long)
        else:
            return img, torch.tensor(target, dtype=torch.long), self.imgs[i]

    def __len__(self):
        return len(self.imgs)