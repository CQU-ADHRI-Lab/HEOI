import cv2 as cv
import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from utils.utils_cv import show_cv2_img

Root_dir = '/home/nan/dataset/VR/raw_image_224'
Target_dir = '/home/nan/dataset/VR/flow'

# Root_dir = '/home/yang/Desktop/CVPR2019/data/attention/VR/raw_image_224'
# Target_dir = '/home/yang/Desktop/CVPR2019/data/attention/VR/flow'


src_list = os.listdir(Root_dir)
tgt_list = os.listdir(Target_dir)

hsv = np.zeros((224,224,3))
hsv[...,1] = 255
hsv = hsv.astype('float32')

for sub1 in src_list:
    print(sub1)
    sub1_list = os.listdir(os.path.join(Root_dir, sub1))
    sub1_dir = os.path.join(Target_dir, sub1)
    if not os.path.exists(sub1_dir):
        os.mkdir(sub1_dir)
    for sub2 in sub1_list:
        print(sub2)
        sub2_list = sorted(os.listdir(os.path.join(Root_dir, sub1, sub2)))
        # check directory
        sub2_dir = os.path.join(Target_dir, sub1, sub2)
        if not os.path.exists(sub2_dir):
            os.mkdir(sub2_dir)

        img_path = os.path.join(Root_dir, sub1, sub2, sub2_list[0])
        prev = cv2.imread(img_path)
        # show_cv2_img(prev)
        prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
        prev_path = os.path.join(Target_dir, sub1, sub2, sub2_list[0][:-4]+'.npy')
        for i in range(1,len(sub2_list)):
            img_path = os.path.join(Root_dir, sub1, sub2, sub2_list[i])
            next_path = os.path.join(Target_dir, sub1, sub2, sub2_list[i][:-4]+'.npy')
            next = cv2.imread(img_path)
            # show_cv2_img(next)
            next = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prev,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
            # hsv[...,0] = ang*180/np.pi/2
            # hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
            # bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
            # show_cv2_img(bgr)
            np.save(prev_path,flow)
            prev = next
            prev_path = next_path
        np.save(prev_path, flow)

