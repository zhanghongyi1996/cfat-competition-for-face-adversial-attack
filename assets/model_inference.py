import argparse

import cv2
import os
import os.path as osp
import numpy as np
import torch
import iresnet

from skimage import transform as trans
from scrfd import SCRFD
from utils import norm_crop

@torch.no_grad()
def inference(detector, net, img):
    bboxes, kpss = detector.detect(img, max_num=1)
    if bboxes.shape[0]==0:
        return None
    bbox = bboxes[0]
    kp = kpss[0]
    aimg0,_ = norm_crop(img, kp, image_size=112)

    aimg = cv2.cvtColor(aimg0, cv2.COLOR_BGR2RGB)
    aimg = np.transpose(aimg, (2, 0, 1))
    aimg = torch.from_numpy(aimg).unsqueeze(0).float()
    aimg.div_(255).sub_(0.5).div_(0.5)
    aimg = aimg.cuda()
    feat = net(aimg).cpu().numpy().flatten()
    feat /= np.sqrt(np.sum(np.square(feat)))
    return feat, bbox

def cal_score(diff, facebox):
    #diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mask_area = cv2.countNonZero(diff)
    #print(facebox)
    face_area = (facebox[2]- facebox[0]) * (facebox[3] - facebox[1])
    ratio = mask_area / face_area * 100
    print("face_area", face_area)
    print("mask_area", mask_area)
    s = 100 - ratio
    return s

if __name__ == "__main__":

    #init face detection
    detector = SCRFD(model_file = 'assets/det_10g.onnx')
    detector.prepare(0, det_thresh=0.5, input_size=(160, 160))

    #model-1
    net1 = iresnet.iresnet50()
    weight = 'assets/w600k_r50.pth'
    net1.load_state_dict(torch.load(weight, map_location=torch.device('cuda')))
    net1.eval().cuda()

    #model-2
    net2 = iresnet.iresnet100()
    weight = 'assets/glint360k_r100.pth'
    net2.load_state_dict(torch.load(weight, map_location=torch.device('cuda')))
    net2.eval().cuda()

    res = []
    checklist = []
    suc = 0
    scores = []
    version = "e2e"
    #version = "shrink"
    mask_limits = []
    #for i in range(1,2):
    #for i in range(1,86):
    #adv_imgs = os.listdir("output_wh/")
    #adv_imgs.sort()
    for i in range(1,101):
        #attname = adv_imgs[i-1]
        #print(attname)
        #s = float(attname.split('.')[0].split('_')[-1])
        #vic = cv2.imread('images/006/1.png')
        #att = cv2.imread('006_2.png')
        vic = cv2.imread('images/%03d/1.png'%i)
        att = cv2.imread('output_%s/%03d_2.png'%(version,i))
        att0 = cv2.imread('images/%03d/0.png'%i)
        _, bbox_att0 = inference(detector, net1, att0)
        diff = att - att0
        diff = diff[:,:,0]
        s = cal_score(diff, bbox_att0)
        print("score:",s)
        #mask_limits.append(area)

        #mask = cv2.imread('%s_mask/%03d_shrinkmask.png'%(version,i))
        #mask = cv2.imread('%s_mask/mask_for_%03d.png'%(version,i))

        #att = cv2.imread('output/%03d_2.png'%i)

        feat_vic, bbox_vic = inference(detector, net1, vic)
        feat_att, bbox_att = inference(detector, net1, att)
        sim1 = np.dot(feat_vic,feat_att)
        #s = score(mask, bbox_att)
        if sim1 > 0.3:
           suc += 1
           scores.append(s)
        else:
           s = 0
           scores.append(s)
        res.append(sim1)
        print(sim1)

        feat_vic, bbox_vic = inference(detector, net2, vic)
        feat_att, bbox_att = inference(detector, net2, att)
        sim2 = np.dot(feat_vic,feat_att)
        if sim2 > 0.3:
           suc += 1
           scores.append(s)
        else:
           s = 0
           scores.append(s)
        res.append(sim2)
        print(sim2)
        if (sim1 < 0.3 or sim2 < 0.3):
           checklist.append(i)
    scores.sort()
    print(scores)
    print("final score:", scores[100])
    #np.savetxt("res.txt", np.array(res))
    #np.savetxt("limits.txt", np.array(mask_limits))
    #np.savetxt("check_mask_id.txt", np.array(checklist))
    print("success:", suc)

