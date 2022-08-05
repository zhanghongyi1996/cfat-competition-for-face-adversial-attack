import sys
import pip
sys.path.append("./assets")
#sys.path.append("./assets/tools_for_match")
import math
import os
#os.system("python ./assets/tool_install.py")
#sys.path.append("./assets/tools_for_match/timm-0.6.5/")
#sys.path.append("./assets/tools_for_match/scs-3.2.0/")
#sys.path.append("./assets/tools_for_match/osqp-0.6.2.post5/")
#sys.path.append("./assets/tools_for_match/ecos-2.0.10/")
#sys.path.append("./assets/tools_for_match/cvxpy-1.2.1/")
#sys.path.append("./assets/tools_for_match/cvxopt-1.3.0/")
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
import value_map_generation
import value_map_generation_mix_up
import cvxpy as cp
import vit
import shufflenet_v2

class Guassian_con:

    def __init__(self, sigma, device):
        super(Guassian_con, self).__init__()
        self.kernal_size = np.int(np.round(3 * sigma) * 2 + 1)
        Guassian_one_dimension = cv2.getGaussianKernel(self.kernal_size,sigma,ktype=cv2.CV_32F)
        Guassian_two_dimension = Guassian_one_dimension * Guassian_one_dimension.T
        Gaussian_filter_torch = torch.FloatTensor(Guassian_two_dimension).unsqueeze(0).unsqueeze(0)
        Gaussian_filter_torch = Gaussian_filter_torch.expand((3, 1, self.kernal_size, self.kernal_size))
        #self.Gaussian_weight = nn.Parameter(data=Gaussian_filter_torch, requires_grad=False)
        #self.Gaussian_weight = Gaussian_filter_torch
        self.Gaussian_weight = Gaussian_filter_torch.to(device)
    def __call__(self, x):
        x = F.conv2d(input=x, weight=self.Gaussian_weight, bias=None, stride=1, padding=(self.kernal_size - 1) // 2, dilation=1, groups=3)
        return x
class PyFAT:

    def __init__(self, N = 10):
        os.environ['PYTHONHASHSEED'] = str(1)
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        self.device = torch.device('cpu')
        self.is_cuda = False
        self.num_iter = 200
        self.alpha = 1.0/255
        self.k = 1.0/512
        self.gradient_ratio = 0.0
        self.photo_number = N
        self.model_list = []
        self.beta = 0.03
        self.sigma = 0.84089642
        self.step_width = 40
        self.epsilon = 3.0/255

    def set_cuda(self):
        self.is_cuda = True
        self.device = torch.device('cuda')
        torch.cuda.manual_seed_all(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def load(self, assets_path):
        detector = SCRFD(model_file=osp.join(assets_path, 'det_10g.onnx'))
        ctx_id = -1 if not self.is_cuda else 0
        detector.prepare(ctx_id, det_thresh=0.5, input_size=(160, 160))
        img_shape = (112,112)

        model1 = iresnet.iresnet50()
        weight1 = osp.join(assets_path, 'w600k_r50.pth')
        model1.load_state_dict(torch.load(weight1, map_location=self.device))
        model1.eval().to(self.device)
        self.model_list.append(model1)


        model2 = vit.VisionTransformer(embed_dim=256,num_classes=512,num_heads=8,patch_size=9)
        weight2 = osp.join(assets_path, 'vitmodel.pt')
        model2.load_state_dict(torch.load(weight2, map_location=self.device))
        model2.eval().to(self.device)
        self.model_list.append(model2)

        #model3 = shufflenet_v2.ShuffleNet_v2_face(scale=2, num_features=512)
        #weight3 = osp.join(assets_path, 'shufflenet_final.pt')
        #model3.load_state_dict(torch.load(weight3, map_location=self.device))
        #model3.eval().to(self.device)
        #self.model_list.append(model3)

        model3 = shufflenet_v2.ShuffleNet_v2_face(scale=2, num_features=512)
        weight3 = osp.join(assets_path, 'pure_shufflenet.pth')
        model3.load_state_dict(torch.load(weight3, map_location=self.device))
        model3.eval().to(self.device)
        self.model_list.append(model3)

        #model4 = iresnet.iresnet18(num_features=256)
        #weight4 = osp.join(assets_path, 'all_r18_model.pt')
        #weight4 = osp.join(assets_path, 'latest_r18_adv.pt')
        #model4.load_state_dict(torch.load(weight4, map_location=self.device))
        #model4.eval().to(self.device)
        #self.model_list.append(model4)

        model4 = vit.VisionTransformer(embed_dim=256,num_classes=512,num_heads=8,patch_size=9)
        weight4 = osp.join(assets_path, 'pure_vit.pth')
        model4.load_state_dict(torch.load(weight4, map_location=self.device))
        model4.eval().to(self.device)
        self.model_list.append(model4)

        #model5 = iresnet.iresnet18(num_features=256)
        #weight5 = osp.join(assets_path, 'ms1_r18_company.pt')
        #model5.load_state_dict(torch.load(weight5, map_location=self.device))
        #model5.eval().to(self.device)
        #self.model_list.append(model5)
        model5 = iresnet.iresnet50()
        weight5 = osp.join(assets_path, 'pure_r50_adv.pth')
        model5.load_state_dict(torch.load(weight5, map_location=self.device))
        model5.eval().to(self.device)
        self.model_list.append(model5)

        self.detector = detector
        self.gs_c = Guassian_con(self.sigma, self.device)

    def size(self):
        return self.photo_number

    def tv_loss(self, input_noise):
        temp1 = torch.cat((input_noise[:, :, 1:, :], input_noise[:, :, -1, :].unsqueeze(2)), 2)
        temp2 = torch.cat((input_noise[:, :, :, 1:], input_noise[:, :, :, -1].unsqueeze(3)), 3)
        print(temp1.shape,temp2.shape)
        temp = (input_noise - temp1) ** 2 + (input_noise - temp2) ** 2
        return temp.sum()

    def gen_mask(self, cs, ws, hs):
        mask_np = np.ones((112,112,3)) 
        for i in range(len(cs)):
            cx,cy = cs[i][0],cs[i][1]
            xmin = max(0, int(cx - ws[i]/2))
            xmax = min(112, int(cx + ws[i]/2))
            ymin = max(0, int(cy - hs[i]/2))
            ymax = min(112, int( cy + hs[i]/2))
            mask_np[ymin:ymax, xmin:xmax, :] = 0
            mask = torch.Tensor(mask_np.transpose(2, 0, 1)).unsqueeze(0)
        return mask

    def compute_ig(self,att_img,vic_img,model,vic_feats1):
        baseline = vic_img
        scaled_inputs = [baseline + (float(wk) / self.step_width) * (att_img - baseline) for wk in
                         range(self.step_width * 7 // 8, self.step_width + 1)]
        scaled_inputs = np.asarray(scaled_inputs)
        scaled_inputs = scaled_inputs + np.random.uniform(-self.epsilon,self.epsilon,scaled_inputs.shape)
        last_t = torch.from_numpy(np.asarray([(wk) / self.step_width for wk in range(self.step_width * 7 // 8, self.step_width + 1)])).unsqueeze(0).unsqueeze(0).unsqueeze(0).permute(3, 0, 1, 2).to(self.device,dtype=torch.float)
        scaled_inputs = torch.from_numpy(scaled_inputs)
        scaled_inputs = scaled_inputs.to(self.device, dtype=torch.float)
        scaled_inputs.requires_grad_(True)
        att_out = F.normalize(model(scaled_inputs))
        #loss3 = 1 - gra_ratio
        loss = torch.sum(torch.square(att_out - vic_feats1))
        model.zero_grad()
        loss.backward(retain_graph=True)
        #print(torch.sum(last_t))
        grads = scaled_inputs.grad.data * (last_t / torch.sum(last_t))
        #grads = scaled_inputs.grad.data
        #print(grads.shape)
        avg_grads = torch.sum(grads, dim=0)
        #avg_grads = torch.mean(grads, dim=0)
        #delta_X = scaled_inputs[-1] - scaled_inputs[0]
        integrated_grad = avg_grads
        IG = integrated_grad.unsqueeze(0).cpu().detach().numpy()
        #print(IG.shape)
        del integrated_grad,avg_grads,grads,loss,att_out,last_t
        return IG


    def create_anchors(self, pre_grad):
        #pre_grad : numpy , 1x112x112
        #anchor_ratios = [0.33, 0.5, 1, 2, 3]
        anchor_ratios = [1]
        anchor_scales = [4]
        #window = 8
        window = 4
        #anchor_strides = [[7,7],[9,9],[13,13],[3, 13], [13, 3], [7,13], [13, 7],[5,9], [9,5], [5, 13], [13, 5]]
        anchor_strides = [[12, 12],[16, 16],[20, 20],[24, 24],[26, 26],[26, 20],[26, 16],[26, 12],[26, 8],[8, 26],[12, 26],[16, 26],[20, 26],[22, 8],[22, 12],[22, 16],[8, 22],[12, 22],[16, 22]]
        kinds = len(anchor_strides) * len(anchor_ratios)
        anchor_num = window * window * kinds
        anchors = np.zeros((anchor_num, 6), dtype=np.float32)
        #for k in anchor_scales:
        for i in range(window):
            for j in range(window):
                for m, r in enumerate(anchor_ratios):
                    for n, s in enumerate(anchor_strides):
                        #w = int(math.sqrt(s*s/r))
                        #h = int(s * math.sqrt(r))
                        w = s[0]
                        h = s[1]
                        x0 = (112/window)*i + (112/window/2)
                        y0 = (112/window)*j + (112/window/2)
                        x1 = min(max(0, int(x0 - w/2)), 112)
                        y1 = min(max(0, int(y0 - h/2)), 112)
                        x2 = min(max(0, int(x0 + w/2)), 112)
                        y2 = min(max(0, int(y0 + h/2)), 112)
                        area = (x2-x1)*(y2-y1)
                        #if self.method == 'cvx':
                        score = np.sum(pre_grad[y1:y2, x1:x2])
                            #score = np.sum(pre_grad[y1:y2, x1:x2])  * 1.0/ area
                        #else:
                            #score = np.sum(pre_grad[y1:y2, x1:x2])  * 1.0/ area
                        anchor_box = [x1, y1, x2, y2, area, score]
                        anchors[(4*i + j)*len(anchor_ratios)*len(anchor_strides) + len(anchor_strides)*m + n, :] = anchor_box
        return anchors, len(anchor_ratios) * len(anchor_strides)

    def opt_anchors(self, anchors, aw,  max_area, anchors_per_pt):

        size = anchors.shape[0]
        areas = anchors[:, 4]
        #print(areas[:100])
        scores = anchors[:, 5]
        anchor_used = np.zeros((aw*aw, aw*aw*anchors_per_pt))
        for m in range(aw*aw):
            start = anchors_per_pt * m
            end = anchors_per_pt * (m+1)
            anchor_used[m, start:end] = 1
        #print("total anchor size:", size)
        x = cp.Variable(size, integer=True)
        #obj = cp.Maximize(cp.sum(cp.multiply(x, scores)))
        obj = cp.Maximize(scores@x)
        #obj = cp.Maximize(scores@x + 0.5 * areas@x / max_area )
        cond = [0 <= x,x<=1, cp.sum(x) <= 5, anchor_used@x <= 1, areas@x <= max_area]
        #cond = [0 <= x,x<=1, cp.sum(x) <= 5, cp.sum(cp.multiply(x, areas)) <= max_area]
        prob = cp.Problem(obj, cond)
        prob.solve(solver='GLPK_MI')
        print("max sum scores:", prob.value)
        #np.where return a tuple
        opt_index = np.where(x.value==1)[0].tolist()
        print("opt index:", opt_index)
        mask = np.ones((3, 112, 112))
        for idx in opt_index:
            xmin = int(anchors[idx, 0])
            ymin = int(anchors[idx, 1])
            xmax = int(anchors[idx, 2])
            ymax = int(anchors[idx, 3])
            mask[:, ymin:ymax, xmin:xmax] = 0
        #mask = mask[0]
        #mask_area = cv2.countNonZero(mask)
        #print(mask_area)
        mask = torch.Tensor(mask.copy()).unsqueeze(0).to(self.device)

        return mask


    def generate(self, im_a, im_v, tim):
        tc = datetime.datetime.now()
        judge = 0
        #model_selection
        if tim < 5:
            self.model1 = self.model_list[tim]
            # use_mix_up
            #anchor_design = 1
            mix_up_method = 1
            anchor_area = 0.12
        else:
            self.model1 = self.model_list[tim-5]
            # not_use_mix_up
            #anchor_design = 1
            mix_up_method = 0
            anchor_area = 0.12
        #Generate start
        h, w, c = im_a.shape
        assert len(im_a.shape) == 3
        assert len(im_v.shape) == 3
        bboxes, kpss = self.detector.detect(im_a, max_num=1)
        facebox_area = (bboxes[0][2] - bboxes[0][0]) * (bboxes[0][3] - bboxes[0][1])
        if bboxes.shape[0]==0:
            return im_a
        att_img, M = norm_crop(im_a, kpss[0], image_size=112)
        bboxes, kpss = self.detector.detect(im_v, max_num=1)
        if bboxes.shape[0]==0:
            return im_a
        vic_img, _ = norm_crop(im_v, kpss[0], image_size=112)

        att_img = att_img[:,:,::-1]
        vic_img = vic_img[:,:,::-1]

        # get victim feature
        vic_img = torch.Tensor(vic_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        vic_img.div_(255).sub_(0.5).div_(0.5)
        vic_feats1 = self.model1.forward(vic_img)
        vic_feats1 = F.normalize(vic_feats1)
        vic_img_fake = vic_img.clone().cpu().detach().squeeze().numpy()

        # process input
        att_img = torch.Tensor(att_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        att_img.div_(255).sub_(0.5).div_(0.5)
        att_img_ = att_img.clone()
        att_img.requires_grad = True

        #gra_ratio = torch.Tensor(np.array(self.gradient_ratio).copy()).to(self.device)
        #gra_ratio = gra_ratio.clone()
        #gra_ratio.requires_grad = True
        
        prev_loss1 = 100
        prev_loss2 = 100

        momentum_begin = 1
        
        #generate value map
        if mix_up_method == 1:
            print("Generate mix up value map")
            value_map = value_map_generation_mix_up.value_map_capture(self.model1, self.gs_c, im_a, im_v, self.is_cuda, self.device, iteration = 20)
        else:
            print("Generate normal value map")
            value_map = value_map_generation.value_map_capture(self.model1, self.gs_c, im_a, im_v, self.is_cuda, self.device, iteration = 20)
        diff_val = value_map - att_img_
        value_map = torch.sum(torch.abs(diff_val[0]), dim=0).cpu().detach().numpy()
        print('value map generation done!')
        anchors, nums = self.create_anchors(value_map)
        max_area = int(112 * 112 * anchor_area)
        aw = 4
        mask = self.opt_anchors(anchors, aw,  max_area, nums)
        self.mask = mask
        print('mask generation done!')

        accumulate_step = 0
        max_similarity = 0
        #for i in tqdm(range(self.num_iter)):
        for i in range(self.num_iter):
            self.model1.zero_grad()
            mask = self.mask.clone()
            mask = mask.to(self.device)
            adv_images = att_img.clone()

            if mix_up_method == 0:
               # get adv feature
               adv_feats1 = self.model1.forward(adv_images)
               adv_feats1 = F.normalize(adv_feats1)

               # caculate loss and backward
               loss1 = torch.exp(torch.tensor(-20.0)*(torch.dot(adv_feats1[0],vic_feats1[0])-torch.tensor(0.50)))
               loss2 = self.tv_loss(adv_images-att_img_) * 0.001
               #loss3 = 1 - gra_ratio
               loss = loss1 + loss2
               loss.backward(retain_graph=True)
               grad = att_img.grad.data.clone()
               #loss_mini_res50 = torch.dot(adv_feats1[0],vic_feats1[0])
            else:
                pure_images = adv_images.cpu().detach().squeeze().numpy()
                IG = self.compute_ig(pure_images, vic_img_fake, self.model1, vic_feats1)
                grad = torch.from_numpy(IG)
                grad = grad.to(self.device)

            '''if momentum_begin == 1:
                grad = grad
                momentum_begin = 0
            else:
                grad = next_grad * 0.9 + grad
            next_grad = grad.clone()'''
            grad = self.gs_c(grad)
            #print(grad.shape)
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)

            if momentum_begin == 1:
                grad = grad
                momentum_begin = 0
            else:
                grad = next_grad * 0.9 + grad
            next_grad = grad.clone()
            sum_grad = grad

            #mask_ratio_grad = gra_ratio.grad.data.clone()
            #r_grad = mask_ratio_grad

            #update training adv img
            att_img.data = att_img.data - torch.sign(sum_grad) * self.alpha * (1 - mask)
            att_img.data = torch.clamp(att_img.data, -1.0, 1.0)
            att_img = att_img.data.requires_grad_(True)
            #calculate new similarity of att_img
            #current_loss1 = loss1.data.cpu().detach().numpy()
            #current_loss3 = loss3.data.cpu().detach().numpy()
            atv_feats1 = self.model1.forward(att_img.clone())
            atv_feats1 = F.normalize(atv_feats1)
            loss_mini_res50 = torch.dot(atv_feats1[0],vic_feats1[0])
            current_loss_mini_res50 = loss_mini_res50.data.cpu().detach().numpy()

            diff = att_img - att_img_
            diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5
            diff = cv2.warpAffine(src=diff, M=M, dsize=(w, h), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)
            final_mask = diff[:,:,0]
            mask_area = cv2.countNonZero(final_mask)
            print(facebox_area)
            print(mask_area)
            raw_score = 100 -  mask_area / facebox_area * 100
            #check new photo and similarity
            diff_new_version = diff[:,:,::-1]
            my_new_img = im_a + diff_new_version
            my_bboxes, my_kpss = self.detector.detect(my_new_img, max_num=1)
            if my_bboxes.shape[0] == 0:
                current_for_check_res50 = current_loss_mini_res50
            else:
                my_new_att_img, my_M = norm_crop(my_new_img, my_kpss[0], image_size=112)
                my_new_att_img = my_new_att_img[:,:,::-1]
                att_img_for_check = torch.Tensor(my_new_att_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
                att_img_for_check.div_(255).sub_(0.5).div_(0.5)
                #ta = datetime.datetime.now()
                new_att_feats1 = self.model1.forward(att_img_for_check)
                #tb = datetime.datetime.now()
                #print('---------', (tb-ta).total_seconds() )
                new_att_feats1 = F.normalize(new_att_feats1)
                loss_for_check_res50 = torch.dot(new_att_feats1[0],vic_feats1[0])
                current_for_check_res50 = loss_for_check_res50.data.cpu().detach().numpy()

            #calculate the max similarity in this turn
            if current_for_check_res50 > max_similarity:
                max_similarity = current_for_check_res50
                best_img = att_img.clone()
                raw_score_for_best = np.copy(raw_score)


            if current_loss_mini_res50 > 0.70 and current_for_check_res50 > 0.70 and raw_score > 95:
                print(torch.dot(adv_feats1[0],vic_feats1[0]))
                print(current_for_check_res50)
                att_img = adv_images.clone()
                break
            if current_for_check_res50 > 0.70 and current_loss_mini_res50 > 0.70:
                print("mask update!")
                print(current_for_check_res50)
                #gra_ratio.data = gra_ratio.data - self.beta * torch.sign(r_grad)
                #gra_ratio = gra_ratio.data.requires_grad_(True)
                #if tim < 5:
                    #cs,ws,hs=capture_gradient.capture_new_center(img_copy = self.heatmap, value_minimum = gra_ratio.data.item())
                    #previous_att_img = best_img.clone()
                    #max_similarity = 0
                    #self.mask = self.gen_mask(cs, ws, hs)
                #else:
                    #self.mask = self.mask
                previous_att_img = best_img.clone()

                #self.mask = self.gen_mask(cs, ws, hs)
                #previous_att_img = best_img.clone()
                #max_similarity = 0
                #update adv img
                att_img.data = att_img.data*(1 - mask) + att_img_* mask 
                #prev_loss1 = current_loss1
                accumulate_step = 0
                judge = 1
                last_raw_score = np.copy(raw_score_for_best)
                print('last raw score is ' + str(last_raw_score))
            else:
                accumulate_step += 1
            #if accumulate_step >= 5 and anchor_design == 0:
                #for i in range(len(cs)):
                    #cs[i][0] += int(2*(random.random() - 0.5))
                    #cs[i][1] += int(2*(random.random() - 0.5))
                #self.mask = self.gen_mask(cs, ws, hs)
                #att_img.data = att_img.data*(1 - mask) + att_img_* mask
            # calculate time consumption
            td = datetime.datetime.now()
            print('---------', (td - tc).total_seconds())
            if (td-tc).total_seconds() > 93:
                print('Time nearly comes to 100s, must break')
                break

        diff = att_img - att_img_
        diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5
        diff = cv2.warpAffine(src=diff, M=M, dsize=(w, h), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)
        final_mask = diff[:,:,0]
        mask_area = cv2.countNonZero(final_mask)
        raw_score = 100 -  mask_area / facebox_area * 100 
        if current_for_check_res50 > 0.90 and current_loss_mini_res50 > 0.90:
           print("success!")
        else:
           print("failed!")
           if judge == 1:
              diff = previous_att_img - att_img_
              diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5
              diff = cv2.warpAffine(src=diff, M=M, dsize=(w, h), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)
              print("load last result successfully")
              raw_score = last_raw_score
           else:
              print("completely failed!")
              diff = best_img - att_img_
              diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5
              diff = cv2.warpAffine(src=diff, M=M, dsize=(w, h), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)
              print("current similarity is " + str(max_similarity))
              print("current score is " + str(raw_score_for_best))

        diff_bgr = diff[:,:,::-1]
        adv_img = im_a + diff_bgr
        return adv_img

def main(args):

    # make directory
    save_dir = args.output
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idname in range(1, 101):
        tool = PyFAT(N=10)
        if args.device=='cuda':
            tool.set_cuda()
        tool.load('assets')
        str_idname = "%03d"%idname
        iddir = osp.join('images', str_idname)
        att = osp.join(iddir, '0.png')
        vic = osp.join(iddir, '1.png')
        origin_att_img = cv2.imread(att)
        origin_vic_img = cv2.imread(vic)
        print('new photo start')
        for turn in range(tool.size()):
            print('new turn start')
            #tc = datetime.datetime.now()
            adv_img = tool.generate(origin_att_img, origin_vic_img, turn)
            #print("raw score:", raw_score)
            #td = datetime.datetime.now()
            #print('---------', (td-tc).total_seconds() )
            #if (td-tc).total_seconds() >= 100:
                #print("run out of time")
            #save_name = '{}_2_{}.png'.format(str_idname, str(raw_score))
            print('new turn end')
            save_name = '{}_fake_'.format(str_idname) + str(turn) + '_2.png'
            cv2.imwrite(save_dir + '/' + save_name, adv_img)
        print('new photo end')
        #save_name = '{}_2.png'.format(str_idname)
        #cv2.imwrite(save_dir + '/' + save_name, adv_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='output directory', type=str, default='output/')
    parser.add_argument('--device', help='device to use', type=str, default='cpu')
    args = parser.parse_args()
    main(args)

