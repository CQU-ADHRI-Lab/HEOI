import cv2
from matplotlib import pyplot as plt
import numpy as np
import random, colorsys, cv2
import torchvision
import math

def read_cv2_img(path):
    '''
    Read color images
    :param path: Path to image
    :return: Only returns color images
    '''
    img = cv2.imread(path, -1)

    if img is not None:
        if len(img.shape) != 3:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

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

def show_images_row(imgs, titles, rows=1):
    '''
       Display grid of cv2 images image
       :param img: list [cv::mat]
       :param title: titles
       :return: None
    '''
    assert ((titles is None) or (len(imgs) == len(titles)))
    num_images = len(imgs)

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, num_images + 1)]

    fig = plt.figure()
    for n, (image, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(rows, np.ceil(num_images / float(rows)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        ax.set_title(title)
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
    
def tensor2im(img, imtype=np.uint8, unnormalize=False, idx=0, nrows=None):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()

    if unnormalize:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)

    image_numpy = img.numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t*255.0

    if image_numpy_t.shape[-1] == 1:
        image_numpy_t = np.tile(image_numpy_t, (1,1,3))

    return image_numpy_t.astype(imtype)

def tensor2maskim(mask, imtype=np.uint8, idx=0, nrows=1):
    im = tensor2im(mask, imtype=imtype, idx=idx, unnormalize=False, nrows=nrows)
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=-1)
    return im