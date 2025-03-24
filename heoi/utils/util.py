from __future__ import print_function
from PIL import Image
from cv2 import INTER_LINEAR
import numpy as np
import os,torchvision,math,torch,random,colorsys,cv2
from matplotlib import pyplot as plt


def IOU_compute(boxA,boxB):
    # [x1,y1,x2,y2] = [bbox1[0],bbox1[1],bbox1[2],bbox1[3]]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def euclideanDistance(p1,p2):
    return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )

def mkdirs(paths):
    overall_path = paths[0]
    mkdir(overall_path)
    for path in paths[1:]:
        overall_path = os.path.join(overall_path,path)
        mkdir(overall_path)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

########################################################################################
###########################   post and after processing    ##########################
########################################################################################
# 归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# 标准化
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def generate_guassian(image,mu,sigma,w,h):
    guassian_map = np.zeros((w,h))
    guassian_map = np.exp(-0.5 * mu / sigma ** 2)
    guassian_map[guassian_map<0] = 0
    guassian_map = np.uint8(guassian_map)
    
    jetmap1 = cv2.applyColorMap(255 - guassian_map, cv2.COLORMAP_JET)
    out = cv2.addWeighted(image, 0.5, jetmap1, 0.5, 0, out)
    
########################################################################################
###########################   image, tensor, numpy, opencv    ##########################
########################################################################################
def tensor2im(img, imtype=np.uint8):
    # select a sample or create grid if img is a batch
    # if len(img.shape) == 4:
    #     nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
    #     img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    image_numpy = img.numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t*255.0

    if image_numpy_t.shape[-1] == 1:
        image_numpy_t = np.tile(image_numpy_t, (1,1,3))

    return image_numpy_t.astype(imtype)



def show_cv2_img(img, title='img'):
    '''
    Display cv2 image
    :param img: cv::mat
    :param title: title
    :return: None
    '''
    img = img[...,::-1]
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    jetmap1 = jetmap2 = None
    brightness = 1.0 if bright else 0.7   # 3元表达式
    hsv = [(i / N, 1, brightness) for i in range(N)]   #  H-  , S-饱和度(0-1), V-明度(0-1)
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))   # lambda argument_list : function_expression
    random.shuffle(colors)
    # colors = [color*255 for color in colors]
    cv2.applyColorMap()
    out1 = cv2.addWeighted(jetmap1, 0.5, jetmap2, 0.5, 0, jetmap2)

    return colors

def cv2_commmon_functions(image = None):

    bbox = [1,2,3,4]
    text = 'text'
    color255 = (255,0,0)
    b_lt = (int(float(bbox[1])), int(float(bbox[0])))
    b_rb = (int(float(bbox[3])), int(float(bbox[2])))
    imgae_save_name = "1.png"
    raw_image = cv2.imread('1.png')
    cv2.putText(image, str(text), ((int(float(bbox[1])), int(float(bbox[0]) - 5))), cv2.FONT_HERSHEY_SIMPLEX, 1, color255,thickness=2)
    cv2.rectangle(image, b_lt, b_rb, color255, 2)
    cv2.circle(image, center=(b_lt,b_rb), radius=2,color=(255,0,255),thickness=2,lineType=8)
    cv2.line(image, b_lt, b_lt, (255,255,0), 5) 
    cv2.imwrite(imgae_save_name,raw_image)
    cv2.namedWindow("Frame", 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.resize(image, (400, 400), interpolation=INTER_LINEAR)

def draw_color_skeleton(self,image,body_18_3):
    self.kpt_color = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    # self.kpt_line = [ (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
    self.kpt_draw = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    self.kpt_line = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
                            (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
    H = image.shape[0]
    W = image.shape[1]
    centers = {}
    for i in range(18):
        xx = body_18_3[i,0]
        yy = body_18_3[i,1]
        xx = int(xx)
        yy = int(yy)
        centers[i] = (xx, yy)
        if i in self.kpt_draw:
            if xx!=0 and yy!=0:
                image = cv2.circle(image, (xx, yy), 3, self.kpt_color[i], thickness=4, lineType=8, shift=0)

    for pair_order, pair in enumerate(self.kpt_line):
        # if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
        #     continue
        p1 = centers[pair[0]]
        p2 = centers[pair[1]]
        if p1!=(0,0) and p2!=(0,0):
            image = cv2.line(image, p1, p2, self.kpt_color[pair_order], 4)

    return image

 ### mouse case, to define the clicked objects
def on_mouse(event, x, y, flags, pos):
    # att_box_num.append('one')
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print(x, y)
       
        print('you have pressed point (', x, y, ")")
    cv2.setMouseCallback('Frame', on_mouse)
    flag = cv2.waitKey(0)  # this line is novel
    if flag == 49:
        # att_box_num.append('one')
        print('you have pressed 1')
    elif flag == 50:
        # att_box_num.append('two')
        print('you have pressed 2')
    elif flag == 51:
        jump = 1
        # continue
    else:
        pass

def json():
    att_obj_ann_name = os.path.join
    with open(att_obj_ann_name, 'r') as file:
        atts_objs = json.load(file)
    
    variable = None
    save_name = os.path.join()
    with open(save_name, 'w') as f:
        json.dump(variable, f) 
    
    frame_index = 1
    att_ann_name = os.path.join(f'AttentionBoxes_{frame_index:04d}.json')
    
       
def main():
    import torch.nn as nn
    from torchvision.ops import nms, roi_align, roi_pool
    import torch
    
    # fp = torch.tensor(list(range(5 * 5))).float()
    # fp = torch.randn([2, 512, 112, 112]).float
    fp = torch.FloatTensor(10,512,224,224)
    # fp = fp.view(1, 1, 5, 5)
    # print(fp.)
    # [batch_id, x1, y1, x2, y2]
    # boxes = torch.tensor([[0,0, 0, 56, 56],[1,56,56,112,112]]).float()
    # boxes = [[0,0, 0, 56, 56],[1,56,56,112,112]] 
    # pooled_features = roi_pool(fp, boxes, [4, 4])
    # print(pooled_features)
    # print(pooled_features.shape)
    x = torch.rand(2,2)
    print(x)
    y = torch.rand(2,2)
    # print(y)
    a = x*y
    print(x)
    # print(y)
    print(a)
    # label_dens = np.random.randint(0,2,(5,1))
    # print(label_dens)
    # label_one_hot = np.eye(N=label_dens.shape[0], M=2)
    # label_one_hot = label_one_hot[label_dens]
    # # label_one_hot = label_one_hot.squeeze()
    # print(label_one_hot)
    # print(label_one_hot.shape)
    
    # m = nn.Softmax(dim=1)
    # input = torch.randn(3, 2)
    # print(input)
    # output = m(input)
    # print(output)
if __name__ == '__main__':
    main()
    
    
    

