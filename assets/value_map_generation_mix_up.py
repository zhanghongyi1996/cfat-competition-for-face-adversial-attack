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

def compute_ig(att_img,vic_img,model,vic_feats1,device,basic_att_img):
    step_width = 40
    epsilon = 3.0/255
    baseline = vic_img
    compare_inputs = np.asarray([baseline + (float(wk) / step_width) * (basic_att_img - baseline) for wk in
                      range(step_width * 7 // 8, step_width + 1)])
    compare_inputs = torch.from_numpy(compare_inputs)
    compare_inputs = compare_inputs.to(device, dtype=torch.float)
    scaled_inputs = [baseline + (float(wk) / step_width) * (att_img - baseline) for wk in
                     range(step_width * 7 // 8, step_width + 1)]
    scaled_inputs = np.asarray(scaled_inputs)
    scaled_inputs = scaled_inputs + np.random.uniform(-epsilon,epsilon,scaled_inputs.shape)
    last_t = torch.from_numpy(np.asarray([(wk) / step_width for wk in range(step_width * 7 // 8, step_width + 1)])).unsqueeze(0).unsqueeze(0).unsqueeze(0).permute(3, 0, 1, 2).to(device,dtype=torch.float)
    scaled_inputs = torch.from_numpy(scaled_inputs)
    scaled_inputs = scaled_inputs.to(device, dtype=torch.float)
    scaled_inputs.requires_grad_(True)
    att_out = F.normalize(model(scaled_inputs))
    loss = torch.sum(torch.square(att_out - vic_feats1)) + torch.sum(torch.abs(scaled_inputs - compare_inputs)) / 300
    model.zero_grad()
    loss.backward(retain_graph=True)
    grads = scaled_inputs.grad.data * (last_t / torch.sum(last_t))
    avg_grads = torch.sum(grads, dim=0)
    integrated_grad = avg_grads
    IG = integrated_grad.unsqueeze(0).cpu().detach().numpy()
    del integrated_grad,avg_grads,grads,loss,att_out,last_t
    return IG

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
    vic_img_fake = vic_img.clone().cpu().detach().squeeze().numpy()

    # process input
    att_img = torch.Tensor(att_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    att_img.div_(255).sub_(0.5).div_(0.5)
    att_img_ = att_img.clone()
    fake_att_img_ = att_img_.clone().cpu().detach().squeeze().numpy()
    att_img.requires_grad = True


    max_similarity = 0
    momentum_begin = 1
    #vic_feats1 = model.forward(vic_img)
    #vic_feats1 = F.normalize(vic_feats1)
    for i in range(iteration):
         model.zero_grad()
         adv_images = att_img.clone()
         pure_images = adv_images.clone().cpu().detach().squeeze().numpy()
         # get adv feature
         adv_feats1 = model.forward(adv_images)
         adv_feats1 = F.normalize(adv_feats1)

         # caculate loss and backward
         #loss1 = torch.exp(torch.tensor(-20.0)*(torch.dot(adv_feats1[0],vic_feats1[0])-torch.tensor(0.50)))
         #loss1 = torch.mean(torch.square(adv_feats1 - vic_feats1))
         #loss2 = torch.sum((adv_images - att_img_).abs()) / (112*112)
         #loss = loss1
         #loss.backward(retain_graph=True)
         loss_mini_res50 = torch.dot(adv_feats1[0],vic_feats1[0])
         IG = compute_ig(pure_images, vic_img_fake, model, vic_feats1, device, fake_att_img_)

         grad = torch.from_numpy(IG)
         grad = grad.to(device)
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

