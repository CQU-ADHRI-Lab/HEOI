import cv2 as cv
import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
# from utils.cv_utils import show_cv2_img

Root_dir = '/home/zhangfuchun/nzx/TIA/raw_image_224'
Target_dir = '/home/zhangfuchun/nzx/TIA/flow'

if not os.path.exists(Target_dir):
    os.mkdir(Target_dir)

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
    # S08_V02_T20161225_234439_00_FYL_PD9432_G03  
    for sub2 in sub1_list:
        
        sub2_list = sorted(os.listdir(os.path.join(Root_dir, sub1, sub2)))
        # check directory
        sub2_dir = os.path.join(Target_dir, sub1, sub2)
        if not os.path.exists(sub2_dir):
            os.mkdir(sub2_dir)

        img_path = os.path.join(Root_dir, sub1, sub2, sub2_list[0])
        # print(img_path)
        prev = cv2.imread(img_path)
        # show_cv2_img(prev)
        prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
        # prev_path = os.path.join(Target_dir, sub1, sub2, sub2_list[0][:-4]+'.npy')
        # print(prev_path)
        flow = None
        next = None
        # print(sub2_list)
        for i in range(len(sub2_list)):
            img_path = os.path.join(Root_dir, sub1, sub2, sub2_list[i])
            next_path = os.path.join(Target_dir, sub1, sub2, sub2_list[i][:-4]+'.npy')

            if not os.path.exists(next_path):
                next = cv2.imread(img_path)
                next = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
                flow = cv.calcOpticalFlowFarneback(prev,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                np.save(next_path,flow)
                prev = next
                print(next_path)
            else:
                continue

           
            

