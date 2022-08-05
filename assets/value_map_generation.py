import sys
import math
import os
import random
import os.path as osp
import numpy as np
import datetime
import random
import torch
import glob
import time
import cv2
import argparse
import torch.nn as nn
import torch.nn.functional as F
#from tqdm import tqdm
import iresnet
from scrfd import SCRFD
from utils import norm_crop
from torch.nn import functional as F
import new_cam_def
import capture_gradient
import model
from collections import OrderedDict
import vit

def tv_loss(input_noise):
    temp1 = torch.cat((input_noise[:, :, 1:, :], input_noise[:, :, -1, :].unsqueeze(2)), 2)
    temp2 = torch.cat((input_noise[:, :, :, 1:], input_noise[:, :, :, -1].unsqueeze(3)), 3)
    print(temp1.shape, temp2.shape)
    temp = (input_noise - temp1) ** 2 + (input_noise - temp2) ** 2
    return temp.sum()

def value_map_capture(model, gaus, im_a, im_v, is_cuda, device, iteration):
    assets_path = 'assets'
    detector = SCRFD(model_file=osp.join(assets_path, 'det_10g.onnx'))
    ctx_id = -1 if not is_cuda else 0
    detector.prepare(ctx_id, det_thresh=0.5, input_size=(160, 160))
    bboxes, kpss = detector.detect(im_a, max_num=1)
    if bboxes.shape[0]==0:
        return im_a
    att_img, M = norm_crop(im_a, kpss[0], image_size=112)
    bboxes, kpss = detector.detect(im_v, max_num=1)
    if bboxes.shape[0]==0:
        return im_a
    vic_img, _ = norm_crop(im_v, kpss[0], image_size=112)

    att_img = att_img[:,:,::-1]
    vic_img = vic_img[:,:,::-1]

    # get victim feature
    vic_img = torch.Tensor(vic_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    vic_img.div_(255).sub_(0.5).div_(0.5)
    vic_feats1 = model.forward(vic_img)
    vic_feats1 = F.normalize(vic_feats1)

    # process input
    att_img = torch.Tensor(att_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    att_img.div_(255).sub_(0.5).div_(0.5)
    att_img_ = att_img.clone()
    att_img.requires_grad = True


    max_similarity = 0
    momentum_begin = 1
    #vic_feats1 = model.forward(vic_img)
    #vic_feats1 = F.normalize(vic_feats1)
    for i in range(iteration):
         model.zero_grad()
         adv_images = att_img.clone()

         # get adv feature
         adv_feats1 = model.forward(adv_images)
         adv_feats1 = F.normalize(adv_feats1)

         # caculate loss and backward
         loss1 = torch.exp(torch.tensor(-20.0)*(torch.dot(adv_feats1[0],vic_feats1[0])-torch.tensor(0.50)))
         #loss1 = torch.mean(torch.square(adv_feats1 - vic_feats1))
         #loss2 = torch.sum((adv_images - att_img_).abs()) / (112*112)
         loss2 = tv_loss(adv_images - att_img_)
         loss = loss1 + loss2 * 0.05
         loss.backward(retain_graph=True)
         loss_mini_res50 = torch.dot(adv_feats1[0],vic_feats1[0])

         grad = att_img.grad.data.clone()
         '''if momentum_begin == 1:
               grad = grad
               momentum_begin = 0
            else:
               grad = next_grad * 0.9 + grad
            next_grad = grad.clone()'''
         grad = gaus(grad)
         grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)

         if momentum_begin == 1:
             grad = grad
             momentum_begin = 0
         else:
             grad = next_grad * 0.9 + grad
         next_grad = grad.clone()
         sum_grad = grad

         #update training adv img
         att_img.data = att_img.data - torch.sign(sum_grad) / 255
         att_img.data = torch.clamp(att_img.data, -1.0, 1.0)
         att_img = att_img.data.requires_grad_(True)
         #conditioned update mask w-h
         #current_loss1 = loss1.data.cpu().detach().numpy()
         current_loss_mini_res50 = loss_mini_res50.data.cpu().detach().numpy()
         print('current similarity in value map is:' + str(current_loss_mini_res50))
         if current_loss_mini_res50 > max_similarity:
             max_similarity = current_loss_mini_res50
             best_img = att_img.clone()
         else:
             continue
         model.zero_grad()
    return best_img

