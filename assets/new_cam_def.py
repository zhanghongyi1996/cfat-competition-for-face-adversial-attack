import sys
import os
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

def new_cam_func(model_name, image_input, coefficient_selection, loop_min, value_minimum):
    # extract feature layer
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().detach().numpy())

    # extract pos layer
    position_blobs = []

    def hook_position(module, input, output):
        position_blobs.append(output.data.cpu().detach().numpy())

    device = torch.device('cpu')
    if model_name == 'w600k_r50.pth':
        model = iresnet.iresnet50()
    elif model_name == 'glint360k_r100.pth':
        model = iresnet.iresnet100()
    else:
        print('error model name')
        return
    weight = osp.join('assets', model_name)
    model.load_state_dict(torch.load(weight, map_location=device))
    finalconv_name = 'bn2'
    result_name = 'fc'
    model.eval().to(device)
    model._modules.get(finalconv_name).register_forward_hook(hook_feature)
    model._modules.get(result_name).register_forward_hook(hook_position)

    params = list(model.parameters())
    fc_weight = np.squeeze(params[-4].data.numpy())
    # fc_bias = np.squeeze(params[-3].data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 112x112
        size_upsample = (112, 112)
        bz, nc, h, w = feature_conv.shape  # (1, 512, 7, 7)
        output_cam = []
        for idx in class_idx:
            cam = np.sum((weight_softmax[idx].reshape(nc, h * w)) * (feature_conv.reshape((nc, h * w))), axis=0)
            cam = cam.reshape((h, w))
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    # proprocess image
    detector = SCRFD(model_file=osp.join('assets', 'det_10g.onnx'))
    detector.prepare(-1, det_thresh=0.5, input_size=(160, 160))
    #str_idname = "%03d" % label_number
    #origin_att_img = cv2.imread('images/' + str_idname + '/1.png')
    origin_att_img = image_input
    # h, w, c = origin_att_img.shape
    bboxes, kpss = detector.detect(origin_att_img, max_num=1)
    att_img, M = norm_crop(origin_att_img, kpss[0], image_size=112)
    image_return = att_img.copy()
    att_img = att_img[:, :, ::-1]
    att_img = torch.Tensor(att_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    att_img.div_(255).sub_(0.5).div_(0.5)

    result_model = model(att_img)

    coefficient = abs(result_model.detach().numpy()[0])
    sorted_coefficient = np.sort(coefficient)
    map_in_list = []
    good_cam = []
    for i in range(coefficient_selection):
        fit = np.where(coefficient == sorted_coefficient[511 - i])
        class_max = fit[0]
        CAM = returnCAM(features_blobs[0], fc_weight, class_max)

        final = CAM[0]
        final = final - np.min(final)
        final_img = final / np.max(final)

        # for i in range(112):
        #     for j in range(112):
        #         if final_img[i][j] < loop_min:
        #             final_img[i][j] = 0
        #
        # final_img = np.uint8(255 * final_img)
        # cass = []
        # for i in range(112):
        #     for j in range(112):
        #         if final_img[i][j] > 0:
        #             cass.append(final_img[i][j])
        # ret, thresh = cv2.threshold(final_img, min(cass) - 1, 255, cv2.THRESH_BINARY)
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # center = []
        # area = 0
        # for c in contours:
        #     x, y, w, h = cv2.boundingRect(c)
        #     area = area + w * h
        area = np.count_nonzero(final_img >= loop_min)
        # print(area)
        if area > 1500:
            # print(str(fit[0][0]) + ' is Bad heat map!')
            continue
        else:
            map_in_list.append(fit[0][0])
            good_cam.append(CAM[0])

    final_result = np.zeros((112, 112))
    for i in range(len(good_cam)):
        final_result = final_result + good_cam[i]
    final_result = final_result - np.min(final_result)
    all_final_img = final_result / np.max(final_result)
    img_copy = np.copy(all_final_img)

    for i in range(112):
        for j in range(112):
            if all_final_img[i][j] < value_minimum:
                all_final_img[i][j] = 0

    all_final_img = np.uint8(255 * all_final_img)
    result_cass = []
    for i in range(112):
        for j in range(112):
            if all_final_img[i][j] > 0:
                result_cass.append(all_final_img[i][j])

    ret, thresh = cv2.threshold(all_final_img, min(result_cass) - 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_center = []
    all_width = []
    all_height = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        print(x, y, w, h)
        final_center.append([int(x + w / 2), int(y + h / 2)])
        all_width.append(w)
        all_height.append(h)
    print(final_center)
    print(all_width)
    print(all_height)

    return final_center, all_width, all_height, img_copy

