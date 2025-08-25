'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:43
Description: 
'''

import cv2
import torch
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from util.norm import SpecificNorm
from PerceptFace import PerceptFace

imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
def mynorm(image):
    image_temp = image * imagenet_std
    return image_temp + imagenet_mean

def mynorm_(image):
    image_temp = image - imagenet_mean
    return image_temp / imagenet_std

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


import argparse
def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--FFM_pretrain', type=str,
                        default='pretrained/90000_net_G.pth',
                        help='load the pretrained model from the specified location')

    parser.add_argument("--Arc_path", type=str, default='pretrained_models/arcface_checkpoint.tar', help="run ONNX model via TRT")
    parser.add_argument('--pic_a_path', default='original/4.jpg', type=str)
    parser.add_argument('--save_path', default='protect/',type=str)



    parser.add_argument('--lambda_wa', type=float, default=100.0)
    parser.add_argument('--lambda_id', type=float, default=1)
    parser.add_argument('--lambda_rec', type=float, default=1)

    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=1, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)  # todo
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.999)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)

    parser.add_argument('--n_samples', dest='n_samples', type=int, default=1, help='# of sample images')

    parser.add_argument('--gpu', dest='gpu', action='store_true', default=True)

    return parser.parse_args(args)

# seed_torch()
args = parse()
print(args)
args.lr_base = args.lr
args.betas = (args.beta1, args.beta2)
start_epoch, epoch_iter = 1, 0
torch.nn.Module.dump_patches = True


model = PerceptFace(args)
model.load('premodels/MSE_new_all_loss_id_5_rec_5_wa_5_step_40000.pt')
model.eval()
spNorm =SpecificNorm()

start_epoch, epoch_iter = 1, 0
crop_size = 224

torch.nn.Module.dump_patches = True

mode = 'None'


spNorm = SpecificNorm()
app = Face_detect_crop(name='antelope', root='./insightface_func/models')
app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)

with torch.no_grad():
    ############## Forward Pass ######################

    pic_b = args.pic_a_path
    img_b_whole = cv2.imread(pic_b)

    img_b_align_crop_list, b_mat_list = app.get(img_b_whole, crop_size)
    # detect_results = None
    swap_result_list = []

    b_align_crop_tenor_list = []

    for b_align_crop in img_b_align_crop_list:
        b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None, ...].cuda()

        b_align_crop_tenor = mynorm_(b_align_crop_tenor)
        swap_result,_=model.generate_protected(b_align_crop_tenor)
        swap_result=swap_result[0]
        swap_result=mynorm(swap_result)

        swap_result_list.append(swap_result)
        b_align_crop_tenor_list.append(b_align_crop_tenor)


    net = None

    reverse2wholeimage(b_align_crop_tenor_list,swap_result_list, b_mat_list, crop_size, img_b_whole, \
        os.path.join(args.out_path, 'result_whole_swapsigle26.jpg'),norm = spNorm)

    print(' ')

    print('************ Done ! ************')
