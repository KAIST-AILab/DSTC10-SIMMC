import argparse
import os
import torch
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.return_dataset import return_single_image
from txt.category_list.label_to_category import labels_to_category
import ipdb
# Validation settings
parser = argparse.ArgumentParser(description='SIMMC 2.0 visual module')
parser.add_argument('--checkpath', type=str, default='./checkpoints',
                    help='dir to load checkpoint')
parser.add_argument('--net', type=str, default='resnet34', metavar='B',
                    help='backbone architecture')
args = parser.parse_args()

print('network %s' %(args.net))

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
im_data = im_data.cuda()
im_data = Variable(im_data)

def eval_single_image(img):
    G.eval()
    F1_a.eval()
    F1_c.eval()
    F1_p.eval()
    F1_s.eval()
    F1_t.eval()
    
    with torch.no_grad():
        im_data.resize_(img.size()).copy_(img)
        im_data.unsqueeze_(0)
        print(im_data.size())
        ipdb.set_trace()
        feat_out = G(im_data) # 512 dim
        print(feat_out.size())
        out_a = F1_a(feat_out)
        out_c = F1_c(feat_out)
        out_p = F1_p(feat_out)
        out_s = F1_s(feat_out)
        out_t = F1_t(feat_out)

        pred_a = out_a.data.max(1)[1].cpu()[0].item()
        pred_c = out_c.data.max(1)[1].cpu()[0].item()
        pred_p = out_p.data.max(1)[1].cpu()[0].item()
        pred_s = out_s.data.max(1)[1].cpu()[0].item()
        pred_t = out_t.data.max(1)[1].cpu()[0].item()

        pred_att = labels_to_category([pred_a, pred_c, pred_p, pred_s, pred_t])
        test_log = 'assetType: {} ({})\ncolor: {} ({})\npattern: {} ({})\nsleeveLength: {} ({})\ntype: {} ({})\n'\
            .format(pred_att[0], pred_a, pred_att[1], pred_c, pred_att[2], pred_p, pred_att[3], pred_s, pred_att[4], pred_t)
    print(test_log)

# image file path
file_name = 'cloth_store_1_1_1/cloth_store_1_1_1_2.png'
input_img = return_single_image(args, file_name)
eval_single_image(img=input_img)

