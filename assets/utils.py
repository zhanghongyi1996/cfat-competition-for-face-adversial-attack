import numpy as np
import cv2
from skimage import transform as trans
import os

arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041] ], dtype=np.float32 )

def estimate_norm(lmk, image_size):
    assert lmk.shape==(5,2)
    tform = trans.SimilarityTransform()
    _src = float(image_size)/112 * arcface_src
    tform.estimate(lmk, _src)
    M = tform.params[0:2,:]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img,M, (image_size, image_size), borderValue = 0.0)
    return warped, M

def eye_mask():
    mask = np.ones((112,112,3), dtype=np.uint8)
    mask *= 255
    mask[35:65,30:80,:] = 0
    cv2.imwrite("eye_mask.png", mask)

def union_mask():
    eye_mask = cv2.imread("eye_mask.png") 
    savedir = "./crop_mask"
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    areas = []
    size = 112*112
    for i in range(1,101):
        cam_mask = cv2.imread("all_mask/mask_for_%03d.png"%i)
        savepath = os.path.join(savedir, "mask_for_%03d.png"%i)
        union = cam_mask + eye_mask 
        area = size - cv2.countNonZero(union[:,:,0])
        if area < 10:
           union = cam_mask
           area = size - cv2.countNonZero(union[:,:,0])
        print(area)
        areas.append(area)
        cv2.imwrite(savepath, union)
    areas.sort()
    print(areas[0],areas[50], areas[-1])

if __name__ =="__main__":
     eye_mask()
     union_mask()

