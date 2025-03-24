from __future__ import absolute_import
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def auc(heatmap, onehot_im, is_im=True):
    if is_im:
        auc_score = roc_auc_score(np.reshape(onehot_im,onehot_im.size), np.reshape(heatmap,heatmap.size))

    else:
        auc_score = roc_auc_score(onehot_im, heatmap)
    return auc_score


def ap(label, pred):
    return average_precision_score(label, pred)


def argmax_pts(heatmap):
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = map(float,idx)
    return pred_x, pred_y


def L2_dist(p1, p2):
    p1 = p1.cpu().numpy()
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def angle_compute(v1,v2):
    x = np.array(v1)
    y = np.array(v2)

    # 分别计算两个向量的模：
    module_x = np.sqrt(x.dot(x))
    module_y = np.sqrt(y.dot(y))

    # 计算两个向量的点积
    dot_value = x.dot(y)

    # 计算夹角的cos值：
    cos_theta = dot_value / (module_x * module_y)
    if cos_theta > 1 or cos_theta < -1:
        return -1

    # 求得夹角（弧度制）：
    angle_radian = np.arccos(cos_theta)

    # 转换为角度值：
    angle_value = angle_radian * 180 / np.pi
    return angle_value


# is_cross
def gaze_direction_box_cross_metric(gaze_box, head_point, gaze_point):
    is_correct = False
    point_on_bottom_edge = two_point_to_downedge_point(gaze_point, head_point)
    gaze_line = [point_on_bottom_edge, head_point]
    four_lines = four_lines_of_box(gaze_box)
    for each_line in four_lines:
        is_cross = IsIntersec(gaze_line[0], gaze_line[1],
                            each_line[0], each_line[1])
        if is_cross:
            is_correct = True
            break
    return is_correct


def two_point_to_downedge_point(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    deta_y = abs(y1-y2)
    if deta_y < float(5/480):
        return_p = (1, min(y1, y2))
    else:
        if y1 > y2:
            p_upper = [x2, y2]
            p_down = [x1, y1]
        else:
            p_upper = [x1, y1]
            p_down = [x2, y2]

        X = p_down[0] - (p_upper[0] - p_down[0]) * (p_down[1] - 1)/(p_upper[1] - p_down[1])
        return_p = (X, 1)
    return return_p


def four_lines_of_box(gaze_box):
    x1, y1, x2, y2 = gaze_box[0], gaze_box[1], gaze_box[2], gaze_box[3]
    lines = []
    p11 = (x1, y1)
    p22 = (x1, y2)
    lines.append((p11, p22))  # left

    p11 = (x1, y1)
    p22 = (x2, y1)
    lines.append((p11, p22))  # top

    p11 = (x2, y2)
    p22 = (x2, y1)
    lines.append((p11, p22))  # right

    p11 = (x1, y2)
    p22 = (x2, y2)
    lines.append((p11, p22))  # down

    return lines


def cross(p1, p2, p3):
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    return x1 * y2 - x2 * y1


def IsIntersec(p1, p2, p3, p4):
    if (max(p1[0], p2[0]) >= min(p3[0], p4[0])
            and max(p3[0], p4[0]) >= min(p1[0], p2[0])
            and max(p1[1], p2[1]) >= min(p3[1], p4[1])
            and max(p3[1], p4[1]) >= min(p1[1], p2[1])):

        if (cross(p1, p2, p3) * cross(p1, p2, p4) <= 0
                and cross(p3, p4, p1) * cross(p3, p4, p2) <= 0):
            D = 1
        else:
            D = 0
    else:
        D = 0
    return D


