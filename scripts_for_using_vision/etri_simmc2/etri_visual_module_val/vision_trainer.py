import os
import argparse
from tqdm import trange, tqdm
import torch
from torch import nn
from torchvision import transforms
from loaders.data_list import Imagelists, Imageloader, ImagelistsAllDomain
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from torch.optim import AdamW as torch_AdamW


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


CELoss = nn.CrossEntropyLoss()


def return_dataset():
    base_path = './txt'
    img_root = './images_cropped' 
    image_set_file_train = os.path.join(base_path, 'train_labels.txt')
    image_set_file_devtest = os.path.join(base_path, 'devtest_labels.txt')
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


    train_dataset = ImagelistsAllDomain(image_set_file_train, root=img_root, transform=data_transforms['train'])
    devtest_dataset = ImagelistsAllDomain(image_set_file_devtest, root=img_root, transform=data_transforms['test'])

    bs = 128
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,
                                                num_workers=8, shuffle=True,
                                                drop_last=True)
    devtest_loader = \
        torch.utils.data.DataLoader(devtest_dataset,
                                    batch_size=bs,
                                    num_workers=8,
                                    shuffle=False, drop_last=False)
    return train_loader, devtest_loader


def train(args, model, heads, train_loader, devtest_loader):
    optimizer_grouped_parameters = [
        {
            'params': model.parameters()
        },
        {
            'params': heads['fashion_a'].parameters()
        },
        {
            'params': heads['fashion_c'].parameters()
        },
        {
            'params': heads['fashion_p'].parameters()
        },
        {
            'params': heads['fashion_s'].parameters()
        },
        {
            'params': heads['fashion_t'].parameters()
        },
        {
            'params': heads['furniture_c'].parameters()
        },
        {
            'params': heads['furniture_t'].parameters()
        },
    ]
    optimizer = torch_AdamW(optimizer_grouped_parameters, lr=0.0005)
    
    nets = [model, heads['fashion_a'],  heads['fashion_c'], heads['fashion_p'], heads['fashion_s'], 
            heads['fashion_t'], heads['furniture_c'], heads['furniture_t']]
    for net in nets:
        net.train()
        net.zero_grad()

    global_step = 0
    epochs_trained = 0
    train_iterator = trange(
        epochs_trained,
        int(args.train_epochs),
        desc="Epoch",
    )

    for _ in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Train Iteration")
        for step, batch in enumerate(epoch_iterator):
            img = batch[0]
            label = batch[1]
            batch_size = len(label)
            # print('bs:', len(label))
            batch_loss = 0
            for inbatch_idx in range(len(label)):
                feat_out = model(img[inbatch_idx][None, ...].cuda())
                # print('label:', label[inbatch_idx])
                if label[inbatch_idx][-1] != -1:
                    out_a = heads['fashion_a'](feat_out)
                    out_c = heads['fashion_c'](feat_out)
                    out_p = heads['fashion_p'](feat_out)
                    out_s = heads['fashion_s'](feat_out)
                    out_t = heads['fashion_t'](feat_out)
                    # print('out_s:', out_s)
                    # print('torch.tensor([label[inbatch_idx][3]).cuda()]', torch.tensor([label[inbatch_idx][3]).cuda()])
                    loss = CELoss(out_a, torch.tensor([label[inbatch_idx][0]]).cuda()) +\
                    CELoss(out_c, torch.tensor([label[inbatch_idx][1]]).cuda()) +\
                    CELoss(out_p, torch.tensor([label[inbatch_idx][2]]).cuda()) +\
                    CELoss(out_s, torch.tensor([label[inbatch_idx][3]]).cuda()) +\
                    CELoss(out_t, torch.tensor([label[inbatch_idx][4]]).cuda()) 
                else:
                    out_c = heads['furniture_c'](feat_out)
                    out_t = heads['furniture_t'](feat_out)
                    loss = CELoss(out_c, torch.tensor([label[inbatch_idx][0]]).cuda()) +\
                    CELoss(out_t, torch.tensor([label[inbatch_idx][1]]).cuda())

                batch_loss += loss
            batch_loss /= batch_size
            batch_loss.backward()
            optimizer.step()
            for net in nets:
                net.zero_grad()
            global_step += 1

            if global_step % args.eval_step == 0:
                result = evaluate(args, model, heads, devtest_loader, global_step)
                for net in nets:
                    net.train()

            if global_step % args.save_step == 0:
                print('checkpoint saving!!')
                output_dir = os.path.join(
                    args.output_dir, "{}-{}".format('checkpoint', global_step)
                )
                os.makedirs(output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
                torch.save({
                    'fashion_a': heads['fashion_a'].state_dict(),
                    'fashion_c': heads['fashion_c'].state_dict(),
                    'fashion_p': heads['fashion_p'].state_dict(),
                    'fashion_s': heads['fashion_s'].state_dict(),
                    'fashion_t': heads['fashion_t'].state_dict(),
                    'furniture_c': heads['furniture_c'].state_dict(),
                    'furniture_t': heads['furniture_t'].state_dict(),
                }, os.path.join(output_dir, 'heads_net.pt'))                


def evaluate(args, model, heads, devtest_loader, global_step):
    nets = [model, heads['fashion_a'],  heads['fashion_c'], heads['fashion_p'], heads['fashion_s'], 
            heads['fashion_t'], heads['furniture_c'], heads['furniture_t']]
    for net in nets:
        net.eval()

    def add_dicts(d1, d2):
        return {k: d1[k] + d2[k] for k in d1}

    report = {
        'fashion_a': [0,0], 'fashion_c': [0,0], 'fashion_p': [0,0] ,'fashion_s': [0,0], 'fashion_t': [0,0],
        'furniture_c': [0,0], 'furniture_t': [0,0]
    }

    with torch.no_grad():
        epoch_iterator = tqdm(devtest_loader, desc="Evaluation")
        for batch_idx, batch in enumerate(epoch_iterator):
            img = batch[0]
            label = batch[1]
            batch_size = len(label)
            for inbatch_idx in range(len(label)):
                feat_out = model(img[inbatch_idx][None, ...].cuda())
                if label[inbatch_idx][-1] != -1:
                    out_a = heads['fashion_a'](feat_out)
                    out_c = heads['fashion_c'](feat_out)
                    out_p = heads['fashion_p'](feat_out)
                    out_s = heads['fashion_s'](feat_out)
                    out_t = heads['fashion_t'](feat_out)
                    report['fashion_a'][0] += (out_a.argmax() == label[inbatch_idx][0]).float()
                    report['fashion_c'][0] += (out_c.argmax() == label[inbatch_idx][1]).float()
                    report['fashion_p'][0] += (out_p.argmax() == label[inbatch_idx][2]).float()
                    report['fashion_s'][0] += (out_s.argmax() == label[inbatch_idx][3]).float()
                    report['fashion_t'][0] += (out_t.argmax() == label[inbatch_idx][4]).float() 
                    report['fashion_a'][1] += 1
                    report['fashion_c'][1] += 1
                    report['fashion_p'][1] += 1
                    report['fashion_s'][1] += 1
                    report['fashion_t'][1] += 1
                else:
                    out_c = heads['furniture_c'](feat_out)
                    out_t = heads['furniture_t'](feat_out)
                    report['furniture_c'][0] += (out_c.argmax() == label[inbatch_idx][0]).float()
                    report['furniture_t'][0] += (out_t.argmax() == label[inbatch_idx][1]).float()
                    report['furniture_c'][1] += 1
                    report['furniture_t'][1] += 1
    
    report_result = {
        'fashion_a': report['fashion_a'][0]/report['fashion_a'][1],
        'fashion_c': report['fashion_c'][0]/report['fashion_c'][1],
        'fashion_p': report['fashion_p'][0]/report['fashion_p'][1],
        'fashion_s': report['fashion_s'][0]/report['fashion_s'][1],
        'fashion_t': report['fashion_t'][0]/report['fashion_t'][1],
        'furniture_c': report['furniture_c'][0]/report['furniture_c'][1],
        'furniture_t': report['furniture_t'][0]/report['furniture_t'][1],
    }
    report_result['average'] = sum([v for v in report_result.values()]) / len(report_result)

    print(report_result)
    with open(args.eval_result_file, 'a') as writer:
        writer.write(str(global_step) + '\n')
        for key in report_result.keys():
            writer.write("%s = %s\n" % (key, str(report_result[key])))
        writer.write('\n')

    return report_result
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ETRI vision module training')
    parser.add_argument('--output_dir', type=str, default='./vision_module_ckpt',
                        help='dir to output checkpoint of trained model')
    parser.add_argument('--eval_result_file', type=str, default='./vision_module_ckpt/eval_result')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=100)
    parser.add_argument('--save_step', type=int, default=100)

    args = parser.parse_args()

    train_loader, devtest_loader = return_dataset()

    G = resnet34()
    inc = 512

    fashion_a = Predictor_deep(num_class=11, inc=inc)
    fashion_c = Predictor_deep(num_class=84, inc=inc)
    fashion_p = Predictor_deep(num_class=36, inc=inc)
    fashion_s = Predictor_deep(num_class=6, inc=inc)
    fashion_t = Predictor_deep(num_class=18, inc=inc)

    furniture_c = Predictor_deep(num_class=9, inc=inc)
    furniture_t = Predictor_deep(num_class=10, inc=inc)

    G.cuda()
    fashion_a.cuda()
    fashion_c.cuda()
    fashion_p.cuda()
    fashion_s.cuda()
    fashion_t.cuda()
    furniture_c.cuda()
    furniture_t.cuda()

    heads = {'fashion_a': fashion_a, 'fashion_c': fashion_c, 'fashion_p': fashion_p, 
             'fashion_s': fashion_s, 'fashion_t': fashion_t,
             'furniture_c': furniture_c, 'furniture_t': furniture_t}
    
    train(args, G, heads, train_loader, devtest_loader)

    
