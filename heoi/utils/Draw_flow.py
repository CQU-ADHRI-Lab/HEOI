
import cv2 as cv
import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from utils.utils_cv import show_cv2_img

img_path = '/home/yang/Desktop/CVPR2019/data/attention/CAD120/raw_image/Subject1_rgbd_images_making_cereal_1204142055/0345.png'
img = cv2.imread(img_path)
show_cv2_img(img)

flow_file = '/home/yang/Desktop/CVPR2019/data/attention/CAD120/flow/Subject1_rgbd_images_making_cereal_1204142055/0345.npy'
flow = np.load(flow_file)
hsv = np.zeros((224,224,3))
hsv[...,1] = 255
hsv = hsv.astype('uint8')

mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
cv.imwrite('./flow_plot.jpg',bgr)
show_cv2_img(bgr)