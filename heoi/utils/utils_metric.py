import numpy as np
import torch
import torch.nn as nn
from collections import deque
# from sklearn.metrics import average_precision_score
# from PIL import Image
# from sklearn.metrics import roc_auc_score
cosine_similarity = nn.CosineSimilarity()

def argmax_pts(heatmap):
    
    idx=np.unravel_index(heatmap.argmax(),heatmap.shape)
    pred_y,pred_x=map(float,idx)

    return pred_x,pred_y

def euclid_dist(pred,target,type='avg'):
    
    batch_dist=0.
    sample_dist=0.

    batch_size=pred.shape[0]
    pred_H,pred_W=pred.shape[1:]


    sample_dist_list=[]
    mean_gt_gaze_list=[]
    for b_idx in range(batch_size):

        pred_x,pred_y=argmax_pts(pred[b_idx])
        norm_p=np.array([pred_x,pred_y])/np.array([pred_W,pred_H])
        # print(norm_p,target[b_idx])
        # plt.imshow(pred[b_idx])
        # plt.show()


        b_target=target[b_idx]
        valid_target=b_target[b_target!=-1].view(-1,2)
        valid_target=valid_target.numpy()
        sample_dist=valid_target-norm_p

        sample_dist = np.sqrt(np.power(sample_dist[:, 0], 2) + np.power(sample_dist[:, 1], 2))



        if type=='avg':
            mean_gt_gaze = np.mean(valid_target, 0)
            sample_avg_dist = mean_gt_gaze - norm_p
            sample_avg_dist = np.sqrt(np.power(sample_avg_dist[0], 2) + np.power(sample_avg_dist[1], 2))

            sample_dist=float(sample_avg_dist)
        elif type=='min':
            sample_dist=float(np.min(sample_dist))

        elif type=="retained":
            mean_gt_gaze = np.mean(valid_target, 0)
            sample_avg_dist = mean_gt_gaze - norm_p
            sample_avg_dist = np.sqrt(np.power(sample_avg_dist[0], 2) + np.power(sample_avg_dist[1], 2))
            sample_dist=float(sample_avg_dist)

            mean_gt_gaze_list.append(mean_gt_gaze)
            sample_dist_list.append(sample_dist)

        else:
            raise NotImplemented

        batch_dist+=sample_dist

    euclid_dist=batch_dist/float(batch_size)

    if type=="retained":

        return mean_gt_gaze_list,sample_dist_list

    return euclid_dist

#Metric functions 
def euclid_dist(output, target, l):
    total = 0
    fulltotal = 0

    output = output.float()
    target = target.float()
    for i in range(l):

        predy = ((output[i] / 227.0) / 227.0) 
        predx = ((output[i] % 227.0) / 227.0) 

        ct = 0
        for j in range(100):
            ground_x = target[i][2*j]
            ground_y = target[i][2*j + 1]

            if ground_x == -1 or ground_y == -1:
                break

            temp = np.sqrt(np.power((ground_x - predx), 2) + np.power((ground_y - predy), 2))
            total += temp
            ct += 1

        total = total / float(ct * 1.0)

        fulltotal += total

    fulltotal = fulltotal / float(l * 1.0)

    return 


def L2_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def angle_eva(norm_p, head_position):
    #Ang
    # predicted_direction = torch.Tensor([norm_p]) - head_position
    average_gaze_direction = None
    predicted_direction = norm_p- head_position
    average_cos = cosine_similarity(predicted_direction.cpu(), average_gaze_direction).numpy()
    average_cos = np.maximum(np.minimum(average_cos, 1.0), -1.0)
    average_Ang = np.arccos(average_cos) * 180 / np.pi
    
    
def Gaussian_Heatmap(img, pt, sigma):
    # Draw a 2D gaussian

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    img = img/np.max(img)
    # img = torch.FloatTensor(img)
    return img

def angle_compute(v1,v2):
    x=np.array(v1)
    y=np.array(v2)

    # 分别计算两个向量的模：
    module_x=np.sqrt(x.dot(x))
    module_y=np.sqrt(y.dot(y))

    # 计算两个向量的点积
    dot_value=x.dot(y)

    # 计算夹角的cos值：
    cos_theta=dot_value/(module_x*module_y)

    # 求得夹角（弧度制）：
    angle_radian=np.arccos(cos_theta)

    # 转换为角度值：
    angle_value=angle_radian*180/np.pi
    return angle_value

def main():
    predicted_direction = torch.Tensor((1,1))
    gt_gaze_direction = torch.Tensor((0,1))
    # print(predicted_direction)
    # # print("here")
    # average_cos = cosine_similarity(predicted_direction, gt_gaze_direction).numpy()
    # print(average_cos)
    
    # average_cos = np.maximum(np.minimum(average_cos, 1.0), -1.0)
    average_Ang = angle_compute(predicted_direction,gt_gaze_direction)
    print(average_Ang)

if __name__ == '__main__':
    main()