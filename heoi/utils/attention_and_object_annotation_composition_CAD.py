from genericpath import exists
import os
import sys
import cv2
import json
import pickle
import numpy as np

root_path = '/home/cqu/nzx/CAD/'
annotation_object_path = root_path + 'allobjects_annotation_name'
annotation_attention_path = root_path + 'attention_annotation_name'
detection_object_path = root_path + 'CAD120_focusloss_detection'
train_file = os.path.join(root_path, 'train_list.txt')
test_file = os.path.join(root_path, 'test_list.txt')

def attention_annotation_check(type):
    # if not os.path.exists(os.path.join(root_path, 'attention_object_annotation')):
    #             os.mkdir(os.path.join(root_path, 'attention_object_annotation'))
    # if not os.path.exists(os.path.join(root_path, 'attention_object_annotation', 'train')):
    #             os.mkdir(os.path.join(root_path, 'attention_object_annotation', 'train'))
    # if not os.path.exists(os.path.join(root_path, 'attention_object_annotation', 'test')):
    #             os.mkdir(os.path.join(root_path, 'attention_object_annotation', 'test'))
    if type == 'train':
        file_name = train_file
    if type == 'test':
        file_name = test_file
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for key in lines:
            print(key)
            key = key.strip() # key is like "Subject1_rgbd_images_arranging_objects_0510175411-0304"  
            [video_base_name, frame_index] = key.split("-")
            obj_list = load_annotated_objects('CAD', key, annotation_object_path) # annotation_obj is a list like [["box", 328, 315, 486, 393]] stored by json format 
            att_list = load_attentions('CAD', key, annotation_attention_path) 
            org_box_num = int(len(obj_list))
            att_box_num = int(len(att_list))
            assert org_box_num>=att_box_num
            assert att_box_num>=1
            assert att_box_num<3       
            if att_box_num == 2:
                if att_list[0] == att_list[1]:
                    print(key)
            for att in att_list:
                x1, y1, x2, y2 = att[1], att[2], att[3], att[4]
                assert x1<x2
                assert y1<y2
                
            # if att_box_num == 2:

            #     if att_list[0] == att_list[1]:
            #         # print(key)
            #         tem_list = []
            #         tem_list.append(att_list[0]) 
            #         name = os.path.join(annotation_attention_path, video_base_name,'attention-{}.json'.format(frame_index)) 
            #         with open(name, 'w') as file:
            #             json.dump(tem_list,file)   


def attention_and_object_annotation_composition(type = None):
    if not os.path.exists(os.path.join(root_path, 'attention_object_attotation')):
                os.mkdir(os.path.join(root_path, 'attention_object_attotation'))
    if not os.path.exists(os.path.join(root_path, 'attention_object_attotation', 'train')):
                os.mkdir(os.path.join(root_path, 'attention_object_attotation', 'train'))
    if not os.path.exists(os.path.join(root_path, 'attention_object_attotation', 'test')):
                os.mkdir(os.path.join(root_path, 'attention_object_attotation', 'test'))

    if type == 'train':
        file_name = train_file
    if type == 'test':
        file_name = test_file

    with open(file_name, 'r') as file:
        lines = file.readlines()
        for key in lines:
            print(key)
            tem_dict = {}
            tem_dict['attentions'] = []
            tem_dict['objects'] = []
            
            key = key.strip() # key is like "Subject1_rgbd_images_arranging_objects_0510175411-0304"  
            obj_list = load_annotated_objects('CAD', key, annotation_object_path) # annotation_obj is a list like [["box", 328, 315, 486, 393]] stored by json format 
            att_list = load_attentions('CAD', key, annotation_attention_path) 
            org_box_num = int(len(obj_list))
            att_box_num = int(len(att_list))
            assert org_box_num>=att_box_num
            assert att_box_num>=1
            assert att_box_num<3
            have_att_num = 0
            for j in range(org_box_num):
                [name, x1, y1, x2, y2] = obj_list[j]
                tem_obj = [int(x1), int(y1), int(x2), int(y2)]
                assert isinstance(tem_obj,list)
                is_attention = 0
                
                for i in range(att_box_num):
                    [name, x1_a, y1_a, x2_a, y2_a] = att_list[i]
                    tem_att = [int(x1_a), int(y1_a), int(x2_a), int(y2_a)]
                    assert isinstance(tem_att,list)
                    Iou = IOU_compute(tem_obj, tem_att)
                    if Iou > 0.99:
                        is_attention=1
                        break

                if is_attention == 0:
                    tem_dict['objects'].append(obj_list[j])
                if is_attention == 1:
                    tem_dict['attentions'].append(obj_list[j])
                    have_att_num+=1

            
            if have_att_num == att_box_num:
                pass
            else:
                print('ERROR:{}'.format(key)) 
            assert have_att_num == att_box_num 
            assert (len(tem_dict['objects']) + len(tem_dict['attentions']) ) == org_box_num
            dict_name = os.path.join(root_path, 'attention_object_attotation', type, '{}.json'.format(key) ) 
            with open(dict_name, 'w') as f:
                    json.dump(tem_dict, f) 


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



def load_annotated_objects(pattern, frame_name, path): 
        if pattern == 'CAD': # key is like "Subject1_rgbd_images_arranging_objects_0510175411-0304" 
            [video_base_name, frame_index] = frame_name.split("-")
            all_objs_name = os.path.join(path, video_base_name,
                                         "all_objects_{:04d}.json".format(
                                             int(frame_index) + 1))
            with open(all_objs_name, 'r') as f:
                all_objs = json.load(f)
            return all_objs
        # key is like "c14_open_door-S53_V10_T20161228_034010_00_FHXL2_PD9thFloor_G05-138"
        if pattern == 'TIA':
            [cate, video_base_name, frame_index] = frame_name.split('-')  
            frame_index = int(frame_index)
            all_objs_name = os.path.join(path, cate, video_base_name, 'allobjects_{:04d}.json'.format(frame_index))
            with open(all_objs_name, 'r') as f:
                all_objs = json.load(f)
            return all_objs


def load_attentions(pattern, frame_name, path):
        if pattern == 'CAD': # key is like "Subject1_rgbd_images_arranging_objects_0510175411-0304"  
            # path = self.data_path + 'attention_annotation_name'
            [video_base_name, frame_index] = frame_name.split("-")
            name = os.path.join(path, video_base_name,
                                'attention-{}.json'.format(frame_index))
            with open(name, 'r') as file:
                attentions = json.load(file)
            return attentions
        elif pattern == 'TIA':
            # path = self.data_path + 'attention_annotation_name'
            [task, video_base_name, frame_index] = frame_name.split("-")
            name = os.path.join(path, task, video_base_name,
                                f'AttentionBoxes_{frame_index.zfill(4)}.json')
            with open(name, 'r') as file:
                attentions = json.load(file)
            return attentions

def main():

    attention_annotation_check('train')
    attention_annotation_check('test')
    # attention_and_object_annotation_composition('train') 
    # attention_and_object_annotation_composition('test')

if __name__ == '__main__':
    main()
