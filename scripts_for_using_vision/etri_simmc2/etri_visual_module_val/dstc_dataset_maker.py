import os
import sys
import torch
import argparse
from torchvision import transforms
from torch.autograd import Variable
from loaders.data_list import Imagelists, Imageloader
from model.resnet import resnet34


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def save_dstc_dataset(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_size = 224
    data_transform = transforms.Compose([
        ResizeImage(256),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    vision_backbone = resnet34()
    if args.resnet_checkpoint is None :
        vision_backbone.to(device)
        print('Defualt!')
    else:
        vision_backbone.to(device)
        vision_backbone.load_state_dict(torch.load(args.resnet_checkpoint))
        vision_backbone.eval()

    im_data = torch.FloatTensor(1)
    im_data = im_data.cuda()
    im_data = Variable(im_data)
    
    dstc_data = dict()
    cropped_images_path = args.cropped_images_path
    with torch.no_grad():
        for a_dir in os.listdir(cropped_images_path):
            print(os.path.join(cropped_images_path, a_dir))
            scene_name = a_dir.rsplit('/', 1)[-1]
            dstc_data[scene_name] = dict()
            files = os.listdir(os.path.join(cropped_images_path, a_dir))
            for a_file in files:
                print(a_file)
                image_feature = Imageloader(os.path.join(cropped_images_path, a_dir, a_file), transform=data_transform)
                im_data.resize_(image_feature.size()).copy_(image_feature)
                im_data.unsqueeze_(0)
                feat_out = vision_backbone(im_data)
                index, _ = a_file.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('_')
                dstc_data[scene_name][index] = feat_out

    torch.save(dstc_data, args.savefile)
    
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resnet_checkpoint",
        default=None
    )
    parser.add_argument(
        "--savefile",
        type=str,
        required=True
    )
    parser.add_argument(
        "--cropped_images_path",
        type=str,
        required=True
    )
    args = parser.parse_args()
    save_dstc_dataset(args=args)
