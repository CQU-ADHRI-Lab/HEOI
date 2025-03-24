import os
import sys
import cv2
import json
import pickle
import numpy as np

root_path = '/home/zhangfuchun/nzx/CAD/'
annotation_object_path = root_path + 'allobjects_annotation_name'
annotation_attention_path = root_path + 'attention_annotation_name'
detection_object_path = root_path + 'CAD120_focusloss_detection'
train_file = os.path.join(root_path, 'train_list.txt')
test_file = os.path.join(root_path, 'test_list.txt')
focus_loss_IOU_threshold = 0.6
# how many annotation objects 
def detection_object_stat():
    all_stat_res = {}
    all_stat_res['obj_num'] = 0
    all_stat_res['image_num'] = 0
    with open(test_file, 'r') as file:
        lines = file.readlines()
        all_stat_res['image_num'] += len(lines)
        for key in lines:
            key = key.strip() # key is like "Subject1_rgbd_images_arranging_objects_0510175411-0304"  
            have_obj, detected_objects = load_detected_objects('CAD',key,detection_object_path)
            # obj_num = int(len(detected_objects))
            # print(obj_num)
            if have_obj == 0:
                if 0 not in all_stat_res:
                    all_stat_res['0'] = []
                    all_stat_res['0'].append(key)
                else:
                    all_stat_res['0'].append(key)
            else:
                obj_num = int(len(detected_objects))
                print(obj_num)
                all_stat_res['obj_num'] += obj_num
                if obj_num not in all_stat_res:
                    all_stat_res[obj_num] = []
                    all_stat_res[obj_num].append(key)
                else:
                    all_stat_res[obj_num].append(key)

    all_stat_res['ave_obj_num'] = all_stat_res['obj_num']/all_stat_res['image_num']
    print("test_set: image_num : {}".format(all_stat_res['image_num']))
    print("test_set: obj_num : {} ".format(all_stat_res['obj_num']) )
    print("test_set: ave_obj_num : {:.2f}".format(all_stat_res['ave_obj_num'])) 

    for key, value in all_stat_res.items():
        if key == 'obj_num'or key == 'image_num' or key == 'ave_obj_num':
            pass
        else:
            num_obj = len(value)
            print("test_set: obj_num=={} {:.2f}%".format(key,num_obj/all_stat_res['image_num']*100 ))

              


def annotation_objects_stat():

    all_stat_res = {}
    all_stat_res['train_obj_num'] = 0
    all_stat_res['train_image_num'] = 0
    all_stat_res['one_obj_name'] = []
    all_stat_res['two_obj_name'] = []
    all_stat_res['three_obj_name'] = []
    all_stat_res['zero_obj_name'] = []
    all_stat_res['threePlus_obj_name'] = [] 

    stat_res = {}
    stat_res['train_obj_num'] = 0
    stat_res['train_image_num'] = 0
    stat_res['one_obj_name'] = []
    stat_res['two_obj_name'] = []
    stat_res['three_obj_name'] = []
    stat_res['zero_obj_name'] = []
    stat_res['threePlus_obj_name'] = []

    with open(train_file, 'r') as file:
        lines = file.readlines()
        stat_res['train_image_num'] = len(lines)
        all_stat_res['train_image_num'] += len(lines)
        for key in lines:
            key = key.strip() # key is like "Subject1_rgbd_images_arranging_objects_0510175411-0304"                 
            annotation_obj = load_annotated_objects('CAD', key, annotation_object_path) # annotation_obj is a list like [["box", 328, 315, 486, 393]] stored by json format
            obj_num = len(annotation_obj)
            stat_res['train_obj_num'] += obj_num
            all_stat_res['train_obj_num'] += obj_num
            if obj_num == 0:
                stat_res['zero_obj_name'].append(key)
                all_stat_res['zero_obj_name'].append(key)
            elif obj_num == 1:
                stat_res['one_obj_name'].append(key)
                all_stat_res['one_obj_name'].append(key)
            elif obj_num == 2:
                stat_res['two_obj_name'].append(key)
                all_stat_res['two_obj_name'].append(key)
            elif obj_num == 3:
                stat_res['three_obj_name'].append(key)
                all_stat_res['three_obj_name'].append(key)
            else:
                stat_res['threePlus_obj_name'].append(key)
                all_stat_res['threePlus_obj_name'].append(key)

    stat_res['ave_obj_num'] = stat_res['train_obj_num']/stat_res['train_image_num']
    print("train_set: train_image_num : {}".format(stat_res['train_image_num']))
    print("train_set: train_obj_num : {} ".format(stat_res['train_obj_num']) )
    print("train_set: zero_obj_num : {}".format(len(stat_res['zero_obj_name'])))
    print("train_set: one_obj_num : {} {:.2f}%".format(len(stat_res['one_obj_name']),len(stat_res['one_obj_name'])/stat_res['train_image_num']*100 ))
    print("train_set: two_obj_num : {} {:.2f}%".format(len(stat_res['two_obj_name']),len(stat_res['two_obj_name'])/stat_res['train_image_num']*100 ))
    print("train_set: three_obj_num : {} {:.2f}%".format(len(stat_res['three_obj_name']),len(stat_res['three_obj_name'])/stat_res['train_image_num']*100 ))
    print("train_set: threePlus_obj_num : {} {:.2f}%".format(len(stat_res['threePlus_obj_name']),len(stat_res['threePlus_obj_name'])/stat_res['train_image_num']*100))
    print("train_set: ave_obj_num : {:.2f}".format(stat_res['ave_obj_num']))

    print(len(all_stat_res['three_obj_name']))

    stat_res['train_obj_num'] = 0
    stat_res['train_image_num'] = 0
    stat_res['one_obj_name'] = []
    stat_res['two_obj_name'] = []
    stat_res['three_obj_name'] = []
    stat_res['zero_obj_name'] = []
    stat_res['threePlus_obj_name'] = []

    with open(test_file, 'r') as file:
        lines = file.readlines()
        stat_res['train_image_num'] = len(lines)
        all_stat_res['train_image_num'] += len(lines)
        for key in lines:
            key = key.strip() # key is like "Subject1_rgbd_images_arranging_objects_0510175411-0304"                 
            annotation_obj = load_annotated_objects('CAD', key, annotation_object_path) # annotation_obj is a list like [["box", 328, 315, 486, 393]] stored by json format
            obj_num = len(annotation_obj)

            stat_res['train_obj_num'] += obj_num
            all_stat_res['train_obj_num'] += obj_num
            if obj_num == 0:
                stat_res['zero_obj_name'].append(key)
                all_stat_res['zero_obj_name'].append(key)
            elif obj_num == 1:
                stat_res['one_obj_name'].append(key)
                all_stat_res['one_obj_name'].append(key)
            elif obj_num == 2:
                stat_res['two_obj_name'].append(key)
                all_stat_res['two_obj_name'].append(key)
            elif obj_num == 3:
                stat_res['three_obj_name'].append(key)
                all_stat_res['three_obj_name'].append(key)
            else:
                stat_res['threePlus_obj_name'].append(key)
                all_stat_res['threePlus_obj_name'].append(key)

    print(len(all_stat_res['three_obj_name']))

    stat_res['ave_obj_num'] = stat_res['train_obj_num']/stat_res['train_image_num']
    print("test_set: test_image_num : {}".format(stat_res['train_image_num']))
    print("test_set: test_obj_num : {} ".format(stat_res['train_obj_num']) )
    print("test_set: zero_obj_num : {}".format(len(stat_res['zero_obj_name'])))
    print("test_set: one_obj_num : {} {:.2f}%".format(len(stat_res['one_obj_name']),len(stat_res['one_obj_name'])/stat_res['train_image_num']*100 ))
    print("test_set: two_obj_num : {} {:.2f}%".format(len(stat_res['two_obj_name']),len(stat_res['two_obj_name'])/stat_res['train_image_num']*100 ))
    print("test_set: three_obj_num : {} {:.2f}%".format(len(stat_res['three_obj_name']),len(stat_res['three_obj_name'])/stat_res['train_image_num']*100 ))
    print("test_set: threePlus_obj_num : {} {:.2f}%".format(len(stat_res['threePlus_obj_name']),len(stat_res['threePlus_obj_name'])/stat_res['train_image_num']*100))
    print("test_set: ave_obj_num : {:.2f}".format(stat_res['ave_obj_num']))

    all_stat_res['ave_obj_num'] = all_stat_res['train_obj_num']/all_stat_res['train_image_num']
    print("train_and_test_set: image_num : {}".format(all_stat_res['train_image_num']))
    print("train_and_test_set: obj_num : {} ".format(all_stat_res['train_obj_num']) )
    print("train_and_test_set: zero_obj_num : {}".format(len(all_stat_res['zero_obj_name'])))
    print("train_and_test_set: one_obj_num : {} {:.2f}%".format(len(all_stat_res['one_obj_name']),len(all_stat_res['one_obj_name'])/all_stat_res['train_image_num']*100 ))
    print("train_and_test_set: two_obj_num : {} {:.2f}%".format(len(all_stat_res['two_obj_name']),len(all_stat_res['two_obj_name'])/all_stat_res['train_image_num']*100 ))
    print("train_and_test_set: three_obj_num : {} {:.2f}%".format(len(all_stat_res['three_obj_name']),len(all_stat_res['three_obj_name'])/all_stat_res['train_image_num']*100 ))
    print("train_and_test_set: threePlus_obj_num : {} {:.2f}%".format(len(all_stat_res['threePlus_obj_name']),len(all_stat_res['threePlus_obj_name'])/all_stat_res['train_image_num']*100))
    print("train_and_test_set: ave_obj_num : {:.2f}".format(all_stat_res['ave_obj_num']))



def load_attentions(self, pattern, frame_name):
        if pattern == 'CAD': # key is like "Subject1_rgbd_images_arranging_objects_0510175411-0304"  
            path = self.data_path + 'attention_annotation_name'
            [video_base_name, frame_index] = frame_name.split("-")
            name = os.path.join(path, video_base_name,
                                'attention-{}.json'.format(frame_index))
            with open(name, 'r') as file:
                attentions = json.load(file)
            return attentions
        elif pattern == 'TIA':
            path = self.data_path + 'attention_annotation_name'
            [task, video_base_name, frame_index] = frame_name.split("-")
            name = os.path.join(path, task, video_base_name,
                                f'AttentionBoxes_{frame_index.zfill(4)}.json')
            with open(name, 'r') as file:
                attentions = json.load(file)
            return attentions

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

def load_detected_objects(pattern, frame_name, path):
    flag = 1
    path_name = None
    if pattern == 'CAD':
        [cate, frame_index] = frame_name.split('-')
        path_name = os.path.join(path, cate,'{}.json'.format(frame_index)) 

    elif pattern == 'TIA': # AttentionBoxes_0000
        [cate, video_base_name, frame_index] = frame_name.split('-')
        path_name = os.path.join(path, cate, video_base_name, 'AttentionBoxes_{:04d}.json'.format(int(frame_index)))

    # print(path_name)

    if not os.path.exists(path_name):
        flag = 0
        filtered_objects = []
        # print('not')

    else:
        with open(path_name, 'r') as f:
            objects = json.load(f)
        # print(len(objects))
        filtered_objects = filter_objtect(objects)
        # print(len(filtered_objects))
        obj_num = len(filtered_objects)
        if obj_num < 1:
            flag = 0
    return flag, filtered_objects

def filter_objtect(detected_objects):
        # filter_objects
        filtered_objects = []
        for obj in detected_objects:
            [name, x1, y1, x2, y2, prob] = obj
            if prob > focus_loss_IOU_threshold:
                filtered_objects.append(obj)
        return filtered_objects



def main():

    # annotation_objects_stat()
    detection_object_stat()

if __name__ == '__main__':
    main()
