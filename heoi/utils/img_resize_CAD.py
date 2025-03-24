
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os


Root_dir = '/home/cqu/nzx/CAD/raw_image'
Target_dir = '/home/cqu/nzx/CAD/raw_image_448'
ReSize = 448

if not os.path.exists(Target_dir):
        os.mkdir(Target_dir)
        

src_list = os.listdir(Root_dir)
tgt_list = os.listdir(Target_dir)

for sub1 in src_list:
    print(sub1)
    sub1_list = os.listdir(os.path.join(Root_dir, sub1))
    sub1_dir = os.path.join(Target_dir, sub1)
    if not os.path.exists(sub1_dir):
        os.mkdir(sub1_dir)
    for img_name in sub1_list:
        print(f'{sub1}-{img_name}')
        targte_path = os.path.join(Target_dir, sub1, img_name)
        if os.path.exists(targte_path):
            continue
        img_path = os.path.join(Root_dir, sub1, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (ReSize,ReSize))
        
        
        cv2.imwrite(targte_path, img)