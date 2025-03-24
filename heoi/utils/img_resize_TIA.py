
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os


# Root_dir = '/home/yang/Desktop/CVPR2019/data/attention/TIA/raw_image'
# Target_dir = '/home/yang/Desktop/CVPR2019/data/attention/TIA/raw_image_224'


Root_dir = '/home/nan/dataset/CAD120/CAD120/raw_image'
Target_dir = '/home/nan/dataset/CAD120/CAD120/raw_image_224'


src_list = os.listdir(Root_dir)
tgt_list = os.listdir(Target_dir)

for sub1 in src_list:
    print(sub1)
    sub1_list = os.listdir(os.path.join(Root_dir, sub1))
    sub1_dir = os.path.join(Target_dir, sub1)
    if not os.path.exists(sub1_dir):
        os.mkdir(sub1_dir)
    for sub2 in sub1_list:
        print(sub2)
        sub2_list = os.listdir(os.path.join(Root_dir, sub1, sub2))
        # check directory
        sub2_dir = os.path.join(Target_dir, sub1, sub2)
        if not os.path.exists(sub2_dir):
            os.mkdir(sub2_dir)
        for img_name in sub2_list:
            img_path = os.path.join(Root_dir, sub1, sub2, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224,224))
            targte_path = os.path.join(Target_dir, sub1, sub2, img_name)
            cv2.imwrite(targte_path, img)