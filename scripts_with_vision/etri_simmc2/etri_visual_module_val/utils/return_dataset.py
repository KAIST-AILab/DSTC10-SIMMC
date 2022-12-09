import os
import sys
import torch
from torchvision import transforms
from .data_list import Imagelists, Imageloader


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def return_dataset(args):
    base_path = './txt'
    img_root = './data/crop_trainval'
    image_set_file_s = os.path.join(base_path, 'crop_img_label_train.txt')
    image_set_file_t_val_sp1 = os.path.join(base_path, 'crop_img_label_val_cloth_store.txt')
    image_set_file_t_val_sp2 = os.path.join(base_path, 'crop_img_label_val_cloth_store_paul.txt')
    image_set_file_t_val_sp3 = os.path.join(base_path, 'crop_img_label_val_cloth_store_woman.txt')

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists(image_set_file_s, root=img_root, transform=data_transforms['train'])
    target_dataset_val_sp1 = Imagelists(image_set_file_t_val_sp1, root=img_root, transform=data_transforms['test'])
    target_dataset_val_sp2 = Imagelists(image_set_file_t_val_sp2, root=img_root, transform=data_transforms['test'])
    target_dataset_val_sp3 = Imagelists(image_set_file_t_val_sp3, root=img_root, transform=data_transforms['test'])
    # print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 256
    elif args.net == 'vgg':
        bs = 48
    else:
        bs = 128
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=8, shuffle=True,
                                                drop_last=True)
    target_loader_val_sp1 = \
        torch.utils.data.DataLoader(target_dataset_val_sp1,
                                    batch_size=bs,
                                    num_workers=8,
                                    shuffle=False, drop_last=False)
    target_loader_val_sp2 = \
        torch.utils.data.DataLoader(target_dataset_val_sp2,
                                    batch_size=bs,
                                    num_workers=8,
                                    shuffle=False, drop_last=False)
    target_loader_val_sp3 = \
        torch.utils.data.DataLoader(target_dataset_val_sp3,
                                    batch_size=bs,
                                    num_workers=8,
                                    shuffle=False, drop_last=False)
    return source_loader, target_loader_val_sp1, target_loader_val_sp2, target_loader_val_sp3


def return_dataset_test(args):
    base_path = './txt'
    img_root = './data/crop_trainval'
    image_set_file_t_val_sp1 = os.path.join(base_path, 'crop_img_label_val_cloth_store.txt')
    image_set_file_t_val_sp2 = os.path.join(base_path, 'crop_img_label_val_cloth_store_paul.txt')
    image_set_file_t_val_sp3 = os.path.join(base_path, 'crop_img_label_val_cloth_store_woman.txt')

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_val_sp1 = Imagelists(image_set_file_t_val_sp1, root=img_root, transform=data_transforms['test'])
    target_dataset_val_sp2 = Imagelists(image_set_file_t_val_sp2, root=img_root, transform=data_transforms['test'])
    target_dataset_val_sp3 = Imagelists(image_set_file_t_val_sp3, root=img_root, transform=data_transforms['test'])
    # print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 256
    elif args.net == 'vgg':
        bs = 48
    else:
        bs = 128
    target_loader_val_sp1 = \
        torch.utils.data.DataLoader(target_dataset_val_sp1,
                                    batch_size=bs,
                                    num_workers=8,
                                    shuffle=False, drop_last=False)
    target_loader_val_sp2 = \
        torch.utils.data.DataLoader(target_dataset_val_sp2,
                                    batch_size=bs,
                                    num_workers=8,
                                    shuffle=False, drop_last=False)
    target_loader_val_sp3 = \
        torch.utils.data.DataLoader(target_dataset_val_sp3,
                                    batch_size=bs,
                                    num_workers=8,
                                    shuffle=False, drop_last=False)
    return target_loader_val_sp1, target_loader_val_sp2, target_loader_val_sp3


def return_single_image(args, file_name):
    base_path = './txt'
    img_root = './data/crop_trainval'
    file_path = os.path.join(img_root, file_name)
    # image_set_file_t_val_sp1 = os.path.join(base_path, 'crop_img_label_val_cloth_store.txt')

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image = Imageloader(file_path, transform=data_transforms['test'])
    return image


def save_dstc_dataset(net_name, cropped_images_path):
    if net_name == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transform = transforms.Compose([
        ResizeImage(256),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dstc_data = dict()
    for a_dir in os.listdir(cropped_images_path):
        print(os.path.join(cropped_images_path, a_dir))
        scene_name = a_dir.rsplit('/', 1)[-1]
        dstc_data[scene_name] = dict()
        files = os.listdir(os.path.join(cropped_images_path, a_dir))
        for a_file in files:
            print(a_file)
            image_feature = Imageloader(os.path.join(cropped_images_path, a_dir, a_file), transform=data_transform)
            index, _ = a_file.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('_')
            dstc_data[scene_name][index] = image_feature

    torch.save(dstc_data, './dstc_data.pth')
                

if __name__ == '__main__':
    save_dstc_dataset(net_name='resnet', cropped_images_path='../images_cropped')

      