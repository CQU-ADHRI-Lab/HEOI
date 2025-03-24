from genericpath import exists
import os
import sys
import cv2
import json
import pickle
import numpy as np

root_path = '/home/cqu/nzx/CAD/'
att_obj_annotation_path = root_path + 'attention_object_annotation'
skeleton_ann_path = root_path + 'skeleton_openpose_json'
train_file = os.path.join(root_path, 'train_list.txt')
test_file = os.path.join(root_path, 'test_list.txt')

H = 480
W = 640
I = 224

is_save = 1       
is_show = 0


def get_human_box(type):
    if type == 'train':
        file_name = train_file
    if type == 'test':
        file_name = test_file
    No_human_detection_list = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for key in lines:
            key = key.strip()
            [video_base_name, frame_index] = key.split("-") 
            
            # load skeleton
            openpose_file = os.path.join(skeleton_ann_path,video_base_name,'{}_{:012d}_keypoints.json'.format(video_base_name, int(frame_index))) 
            with open(openpose_file, 'r') as output:
                skeletons = json.load(output)
            persons = skeletons['people']  # person is a list
            num = len(persons)
            
            # find human box
            human_box = []
            have_good_human_flag = 1
            if num == 0:
                have_good_human_flag = 0
            if num > 0:
                person = chooseONEperson(persons) 
                body = np.reshape(person['pose_keypoints_2d'], (18, 3)) # body is a array
                confidence_points = confidence_filter(body, 0.1)
                point_num = len(confidence_points)
                if point_num < 6:
                    have_good_human_flag = 0
                else:
                    x_list = []
                    y_list = []
                    for p in confidence_points:
                        [x,y] = p
                        x_list.append(x)
                        y_list.append(y)
                        
                    x_min = min(x_list)
                    x_max = max(x_list)
                    y_min = min(y_list)
                    y_max = max(y_list)
                    
                    assert x_min>0
                    # assert x_max<W
                    
                    assert y_min>0
                    assert y_max<H
                    
                    x00 = int(x_min*I/W)
                    y00 = int(y_min*I/H)
                    w00 = int(x_max*I/W-x_min*I/W)
                    h00 = int(y_max*I/H-y_min*I/H)
                    x11 = int(x_max*I/W)
                    y11 = int(y_max*I/H) 
                    x11 = min(x11, 224) 
                    y00 = max(0, y00-int(0.1*h00)) 
                    
                    
                    # assert x00>0
                    # assert y00>0
                    # assert y_min>0
                    # assert y_max<H
                    
                    human_box = ['human', x00, y00, x11, y11]
                
            if have_good_human_flag == 0:
                No_human_detection_list.append(key)
                print(key)
            
            if is_show == 1 and have_good_human_flag == 1:
                image_name = os.path.join(root_path, 'raw_image_224', video_base_name,'{:04d}.png'.format(int(frame_index)))
                raw_image = cv2.imread(image_name)
                cv2.rectangle(raw_image, (human_box[1], human_box[2]), (human_box[3], human_box[4]),(255, 0, 0), thickness=2)
                if not os.path.exists(os.path.join(root_path, 'human_box_visual')):
                    os.mkdir(os.path.join(root_path, 'human_box_visual'))
                if not os.path.exists(os.path.join(root_path, 'human_box_visual', 'train')):
                    os.mkdir(os.path.join(root_path, 'human_box_visual', 'train'))
                if not os.path.exists(os.path.join(root_path, 'human_box_visual', 'test')):
                    os.mkdir(os.path.join(root_path, 'human_box_visual', 'test'))
                    
                save_name = os.path.join(root_path, 'human_box_visual', type, '{}.png'.format(key) ) 
                cv2.imwrite(save_name, raw_image)
                print(key)
                # cv2.imshow('1', raw_image)
                # cv2.waitKey(0)
            
            if is_save ==1 :
                if not os.path.exists(os.path.join(root_path, 'human_box_annotation')):
                    os.mkdir(os.path.join(root_path, 'human_box_annotation'))
                if not os.path.exists(os.path.join(root_path, 'human_box_annotation', 'train')):
                    os.mkdir(os.path.join(root_path, 'human_box_annotation', 'train'))
                if not os.path.exists(os.path.join(root_path, 'human_box_annotation', 'test')):
                    os.mkdir(os.path.join(root_path, 'human_box_annotation', 'test'))
                save_name =  os.path.join(root_path, 'human_box_annotation', type, '{}.json'.format(key) ) 
                with open(save_name, 'w') as f:
                    json.dump(human_box, f) 
                                                                                                 
                    
def confidence_filter(data, con_threshold):
        # data is n*3 array
        # to filter the face, body, hand with low confidence
        n = data.shape[0]
        confidence_points = []
        for i in range(n):
            con = data[i, 2]
            x = data[i,0]
            y = data[i,1] 
            if con < con_threshold:
                data[i] = [0.0, 0.0, 0]
            else:
                confidence_points.append([x,y])
        return confidence_points
                       
def chooseONEperson(peoples):
        # peoples are the list of people from openpose, each people is a dict
        bestpeople = None
        max_confidece = 0
        for people in peoples:
            pose_keypoints_2d = people['pose_keypoints_2d']
            # array,  the number is 54 = 18*3 , (x, y , confidence)
            pose_keypoints_2d = np.reshape(pose_keypoints_2d, (18, 3))
            confidence = np.sum(pose_keypoints_2d, axis=0)[2]
            if confidence > max_confidece:
                max_confidece = confidence
                bestpeople = people
        return bestpeople
    
def get_box_name(type):
    if type == 'train':
        file_name = train_file
    if type == 'test':
        file_name = test_file
    object_name = []
    all_stat_res = {}
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for key in lines:
            key = key.strip()
            print(key)
            att_obj_ann_name = os.path.join(att_obj_annotation_path, type, '{}.json'.format(key))
            with open (att_obj_ann_name, 'r') as f:
                att_obj_ann = json.load(f)
            att_list = att_obj_ann['attentions']
            obj_list = att_obj_ann['objects']
            
            # num
            num = len(att_list) + len(obj_list)
            if num not in all_stat_res:
                all_stat_res[num] = []
                all_stat_res[num].append(key)
            else:
                all_stat_res[num].append(key)
                
            for att in att_list:
                obj_name_id = att[0]
                if obj_name_id not in object_name:
                    object_name.append(obj_name_id)
            
            for att in obj_list:
                obj_name_id = att[0]
                if obj_name_id not in object_name:
                    object_name.append(obj_name_id) 
    print(object_name)
    for key, value in all_stat_res.items():
        print("num=={} {}".format(key,len(value)))

def main():
    # get_box_name('train') 
    get_box_name('test')
    # get_human_box('train') 
    # get_human_box('test') 

if __name__ == '__main__':
    main()