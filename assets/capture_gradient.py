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


def capture_new_center(img_copy, value_minimum):
    img_img = np.copy(img_copy)
    for i in range(112):
        for j in range(112):
            if img_img[i][j] < value_minimum:
                img_img[i][j] = 0

    all_final_img = np.uint8(255 * img_img)
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
        final_center.append([int(x + w / 2), int(y + h / 2)])
        all_width.append(w)
        all_height.append(h)
    return final_center, all_width, all_height
