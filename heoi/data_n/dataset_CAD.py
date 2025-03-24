import os
import sys
import cv2
import json
import numpy as np
from scipy import ndimage
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_n.dataset_base_n import DatasetBase
from torch.utils.data import DataLoader
from options.test_opt import TOptions
from utils.util import normalization

class CustomDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(CustomDataset, self).__init__(opt, is_for_train)
        self.opt = opt
        self._is_for_train = is_for_train
        self.mode = 'train' if self._is_for_train else 'test'

        self._root_path = self._opt.data_root + 'CAD/'
        self.raw_image_path = self._root_path + 'raw_image_224'
        self.human_box_mask_path = self._root_path + 'human_box_binary_mask'
        self.object_box_mask_path = self._root_path + 'object_binary_mask'
        self.human_skeleton_mask_path = self._root_path + 'human_skeleton_mask'
        self.human_body_part_path = self._root_path + 'human_body_part_box'

        self.H = 480
        self.W = 640
        self.input = self.opt.image_size

        self.keys = []
        self._prepare_keys()

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        key = self.keys[index]
        data_dict = self._prepare_data_by_key(key)
        sample = data_dict

        return sample

    def __len__(self):
        return self._dataset_size

    def _prepare_keys(self):
        if self.mode == "train":
            file_list_path = os.path.join(self._root_path, f'{self.mode}_list_dense.txt')
        elif self.mode == "test":
            file_list_path = os.path.join(self._root_path, f'{self.mode}_list_dense_ObjDet.txt')
        with open(file_list_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip()
            self.keys.append(line)

        self._dataset_size = len(self.keys)

    def _prepare_data_by_key(self, key):
        data = {'human_box_mask': None,
                'object_box_mask': None,
                'image':None,
                'frame_name': None,
                'index':None,
                'label':None,
                'human_skeleton_mask':None,
                'human_body_part_mask':None,
                'box':None
                }
       
        "Subject3_rgbd_images_arranging_objects_0510143426-0000-x1-y1-x2-y2-class-flag"
        key = key.strip()
        data['index'] = key

        "frame_name"
        [video_base_name, frame_index] = key.split('-')[0:2]

        frame_name = f'{video_base_name}-{frame_index}'
        data['frame_name'] = frame_name
        
        
        "image"
        image_name = os.path.join(self.raw_image_path, video_base_name,'{:04d}.png'.format(int(frame_index)))
        image = np.transpose(cv2.imread(image_name), (2, 0, 1)) # from (480,640,3) to (3, 480, 640)
        data['image'] = image
        
        
        "target object box"
        [x1,y1,x2,y2] = key.split('-')[2:6]
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        
        box = f'{x1}-{y1}-{x2}-{y2}'
        data['box'] = box
        
        if self.mode == "train":
            "object box mask"
            object_box_mask_name = os.path.join(self.object_box_mask_path, self.mode, f'{key}.png')
            object_box_mask = cv2.imread(object_box_mask_name)
            data['object_box_mask'] = np.transpose(object_box_mask, (2, 0, 1))
        elif self.mode == "test":
        
            y1 = int(int(y1) * self.input / self.H)
            y2 = int(int(y2) * self.input / self.H)
            x1 = int(int(x1) * self.input / self.W)   
            x2 = int(int(x2) * self.input / self.W)
        
            "object box mask"
            object_box_mask = np.zeros((1, self.input, self.input))
            object_box_mask[0,y1:y2,x1:x2] = 255
            data["object_box_mask"] = object_box_mask
        
        
        "human box mask"
        human_box_mask_name = os.path.join(self.human_box_mask_path, self.mode, f'{frame_name}.png')
        data['human_box_mask'] = np.transpose(cv2.imread(human_box_mask_name), (2, 0, 1))

        "human skeleton mask"
        human_skeleton_mask_name = os.path.join(self.human_skeleton_mask_path, self.mode, f'{frame_name}.png')
        data['human_skeleton_mask'] = np.transpose(cv2.imread(human_skeleton_mask_name), (2, 0, 1))
        
        "human body part mask"
        self.tem_one_data = np.zeros((1,self.input, self.input))
        self.tem_one_data = self.tem_one_data * [0] 
        body_part_name = os.path.join(self.human_body_part_path, self.mode, f'{frame_name}.json')
        with open(body_part_name, 'r') as file:
            body_parts = json.load(file)
        r_wrist_box = body_parts['r_wrist_box']
        l_wrist_box = body_parts['l_wrist_box']
        head_box = body_parts['head_box']
        if len(r_wrist_box)>0:
            [x_min, y_min, x_max, y_max] = r_wrist_box
            y1 = int(y_min * self.input / self.H)
            y2 = int(y_max * self.input / self.H)
            x1 = int(x_min * self.input / self.W)
            x2 = int(x_max * self.input / self.W)
            self.tem_one_data[0,y1:y2,x1:x2] = 255
        if len(l_wrist_box)>0:
            [x_min, y_min, x_max, y_max] = l_wrist_box
            y1 = int(y_min * self.input / self.H)
            y2 = int(y_max * self.input / self.H)
            x1 = int(x_min * self.input / self.W)
            x2 = int(x_max * self.input / self.W)
            self.tem_one_data[0,y1:y2,x1:x2] = 255
        if len(head_box)>0:
            [x_min, y_min, x_max, y_max] = head_box
            y1 = int(y_min * self.input / self.H)
            y2 = int(y_max * self.input / self.H)
            x1 = int(x_min * self.input / self.W)
            x2 = int(x_max * self.input / self.W)
            self.tem_one_data[0,y1:y2,x1:x2] = 255
        data['human_body_part_mask'] = self.tem_one_data
        
        "label"
        
        label = key.split('-')[-1]
       
        "if label for one"
        data['label'] = int(label)
        
        return data


def main():
    aa = '/home/nan/dataset/TIA/raw_image'
    x = os.listdir(aa)
    y = {}
    for idx, name in enumerate(sorted(x)):
        y[name] = idx
    print(y)


if __name__ == '__main__':
    # main()
   

    opt = TOptions().parse()
    dataset_test = CustomDataset(opt, is_for_train=False)
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=4,
                                 shuffle=False,
                                 num_workers=2,
                                 drop_last=True)

    for bs, test_bs in enumerate(dataloader_test):
        print(bs)
