import argparse
import os
import torch
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.return_dataset import return_dataset_test

# Validation settings
parser = argparse.ArgumentParser(description='SIMMC 2.0 visual module')
parser.add_argument('--checkpath', type=str, default='./checkpoints',
                    help='dir to load checkpoint')
parser.add_argument('--net', type=str, default='resnet34', metavar='B',
                    help='backbone architecture')
args = parser.parse_args()

print('network %s' %(args.net))
loader_val_sp1, loader_val_sp2, loader_val_sp3 = return_dataset_test(args)

if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

if "resnet" in args.net:
    F1_a = Predictor_deep(num_class=11, inc=inc)
    F1_c = Predictor_deep(num_class=63, inc=inc)
    F1_p = Predictor_deep(num_class=36, inc=inc)
    F1_s = Predictor_deep(num_class=6,  inc=inc)
    F1_t = Predictor_deep(num_class=18, inc=inc)
else:
    F1_a = Predictor(num_class=11, inc=inc)
    F1_c = Predictor(num_class=63, inc=inc)
    F1_p = Predictor(num_class=36, inc=inc)
    F1_s = Predictor(num_class=6,  inc=inc)
    F1_t = Predictor(num_class=18, inc=inc)

G.cuda()
F1_a.cuda()
F1_c.cuda()
F1_p.cuda()
F1_s.cuda()
F1_t.cuda()

G.load_state_dict(torch.load(os.path.join(args.checkpath, "G_model_{}_best.pth.tar". format(args.net))))
F1_a.load_state_dict(torch.load(os.path.join(args.checkpath, "F1_{}_model_{}_best.pth.tar".format('assetType', args.net))))
F1_c.load_state_dict(torch.load(os.path.join(args.checkpath, "F1_{}_model_{}_best.pth.tar".format('color', args.net))))
F1_p.load_state_dict(torch.load(os.path.join(args.checkpath, "F1_{}_model_{}_best.pth.tar".format('pattern', args.net))))
F1_s.load_state_dict(torch.load(os.path.join(args.checkpath, "F1_{}_model_{}_best.pth.tar".format('sleeveLength', args.net))))
F1_t.load_state_dict(torch.load(os.path.join(args.checkpath, "F1_{}_model_{}_best.pth.tar".format('type', args.net))))

im_data = torch.FloatTensor(1)
gt_labels = torch.LongTensor(1)

im_data = im_data.cuda()
gt_labels = gt_labels.cuda()

im_data = Variable(im_data)
gt_labels = Variable(gt_labels)
if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


def eval(loader):
    G.eval()
    F1_a.eval()
    F1_c.eval()
    F1_p.eval()
    F1_s.eval()
    F1_t.eval()
    num_img = 0
    cor_a, cor_c, cor_p, cor_s, cor_t = 0, 0, 0, 0, 0
    
    with torch.no_grad():
        for batch_idx, data_val in enumerate(loader):
            im_data.resize_(data_val[0].size()).copy_(data_val[0])
            gt_labels.resize_(data_val[1].size()).copy_(data_val[1])
            feat_out = G(im_data)
            out_a = F1_a(feat_out)
            out_c = F1_c(feat_out)
            out_p = F1_p(feat_out)
            out_s = F1_s(feat_out)
            out_t = F1_t(feat_out)
            num_img += im_data.size(0)

            pred_a = out_a.data.max(1)[1]
            pred_c = out_c.data.max(1)[1]
            pred_p = out_p.data.max(1)[1]
            pred_s = out_s.data.max(1)[1]
            pred_t = out_t.data.max(1)[1]

            cor_a += pred_a.eq(gt_labels[:, 0]).cpu().sum()
            cor_c += pred_c.eq(gt_labels[:, 1]).cpu().sum()
            cor_p += pred_p.eq(gt_labels[:, 2]).cpu().sum()
            cor_s += pred_s.eq(gt_labels[:, 3]).cpu().sum()
            cor_t += pred_t.eq(gt_labels[:, 4]).cpu().sum()

        acc_a = 100. * cor_a / num_img
        acc_c = 100. * cor_c / num_img
        acc_p = 100. * cor_p / num_img
        acc_s = 100. * cor_s / num_img
        acc_t = 100. * cor_t / num_img

        test_log = 'acc_assetType: {:.2f}%  acc_color: {:.2f}%  acc_pattern: {:.2f}%  acc_sleeveLength: {:.2f}%  acc_type: {:.2f}%\n'.\
            format(acc_a, acc_c, acc_p, acc_s, acc_t)
    print(test_log)



print("Val set sp1")
eval(loader_val_sp1)

print("Val set sp2")
eval(loader_val_sp2)

print("Val set sp3")
eval(loader_val_sp3)
