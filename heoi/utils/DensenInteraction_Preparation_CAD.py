import os,json,cv2,statistics
from matplotlib import image
import numpy as np
from util import mkdir, mkdirs, euclideanDistance,IOU_compute

class DenseInteraction_Preparation():
    def __init__(self):
        self.pattern = 'CAD'
        self.input = 224
        self.skeleton_joint = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
                            (1, 8), (8, 9), (9, 10), (1, 11), (11, 12),
                            (12, 13),(1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
            
        if self.pattern == 'CAD':
            self.root_path = '/home/cqu/nzx/CAD'
            self.image_path = '/home/cqu/nzx/CAD/raw_image_224'
            self.raw_image_path = '/home/cqu/nzx/CAD/raw_image'
            # self.attention_object_ann_path = '/home/cqu/nzx/CAD/attention_object_annotation'
            self.human_skeleton_path = '/home/cqu/nzx/CAD/skeleton_openpose_json'
            self.tain_list = '/home/cqu/nzx/CAD/train_list.txt'
            self.test_list = '/home/cqu/nzx/CAD/test_list.txt'
            self.tain_dense_list = '/home/cqu/nzx/CAD/train_list_dense.txt'
            self.test_dense_list = '/home/cqu/nzx/CAD/test_list_dense.txt'
            self.H = 480
            self.W = 640
            self.focus_loss_IOU_threshold = 0.6
        
        if self.pattern == 'TIA':
            self.root_path = '/home/cqu/nzx/TIA'
            self.image_path = '/home/cqu/nzx/TIA/raw_image_224'
            self.raw_image_path = '/home/cqu/nzx/TIA/raw_image'
            # self.attention_object_ann_path = '/home/cqu/nzx/CAD/attention_object_annotation'
            self.human_skeleton_path = '/home/cqu/nzx/TIA/skeleton'
            self.tain_list = '/home/cqu/nzx/TIA/train_list.txt'
            self.test_list = '/home/cqu/nzx/TIA/test_list.txt'
            self.tain_dense_list = '/home/cqu/nzx/TIA/train_list_dense.txt'
            self.test_dense_list = '/home/cqu/nzx/TIA/test_list_dense.txt'
            self.H = 1080
            self.W = 1920
            self.focus_loss_IOU_threshold = 0.6
    
    def extract_testing_index_obj_detection(self,):
        train_txt_name = os.path.join(self.root_path,'test_list_dense_ObjDet.txt')
        train_txt = open(train_txt_name, 'a')
        with open(os.path.join(self.test_list), 'r') as file:
            lines = file.readlines()
        for line in lines:
            print(line)
            line = line.strip()  
            
            att_obj_ann_name = os.path.join(self.root_path, 'attention_object_annotation', 'test', f'{line}.json')
            with open(att_obj_ann_name, 'r') as file:
                atts_objs = json.load(file)
            atts = atts_objs['attentions']
            # objs = atts_objs['objects']
            assert len(atts)==1 or len(atts)==2

            # [name, x_min, y_min, x_max, y_max, prob] = obj
            name = line.split('-')[0]
            flag, objs =  self.load_object_detection(pattern=self.pattern, frame_name=line)  
            # print(flag)   
            if flag:
                # print("okay")
                for obj in objs:
                    obj_pos = obj[1:5]
                    obj_is_att = 0
                    for att in atts:
                        [c,x_min, y_min, x_max, y_max] = att
                        y_min = int(y_min * self.input / self.H)
                        y_max = int(y_max * self.input / self.H)
                        x_min = int(x_min * self.input / self.W)
                        x_max = int(x_max * self.input / self.W)
                        att_pos = [x_min, y_min, x_max, y_max]
                        iou_rate = IOU_compute(obj_pos, att_pos)
                        if iou_rate > 0.5:
                            [x_min, y_min, x_max, y_max] = obj_pos
                            c = obj[0]
                            train_txt.write(f"{line}-{x_min}-{y_min}-{x_max}-{y_max}-{c}-1\n")
                            obj_is_att = 1
                            break
                    if obj_is_att == 0:
                        [x_min, y_min, x_max, y_max] = obj_pos
                        c = obj[0]
                        train_txt.write(f"{line}-{x_min}-{y_min}-{x_max}-{y_max}-{c}-0\n")
        
        train_txt.close()
        
    def extract_training_index(self,):
        "Subject3_rgbd_images_arranging_objects_0510143426-0000-x1-y1-x2-y2-class-flag"
        train_txt_name = os.path.join(self.root_path,'train_list_dense.txt')
        train_txt = open(train_txt_name, 'a')
        
        with open(os.path.join(self.tain_list), 'r') as file:
            lines = file.readlines()
        for line in lines:
            print(line)
            line = line.strip()          
            att_obj_ann_name = os.path.join(self.root_path, 'attention_object_annotation', 'train', f'{line}.json')
            with open(att_obj_ann_name, 'r') as file:
                atts_objs = json.load(file)
            atts = atts_objs['attentions']
            objs = atts_objs['objects']
            assert len(atts)==1 or len(atts)==2
            for att in atts:
                [c,x_min, y_min, x_max, y_max] = att
                y_min = int(y_min * self.input / self.H)
                y_max = int(y_max * self.input / self.H)
                x_min = int(x_min * self.input / self.W)
                x_max = int(x_max * self.input / self.W)
                train_txt.write(f"{line}-{x_min}-{y_min}-{x_max}-{y_max}-{c}-1\n")
            obj_num = len(objs)
            if obj_num>0:
                for obj in objs:
                    [c,x_min, y_min, x_max, y_max] = obj
                    y_min = int(y_min * self.input / self.H)
                    y_max = int(y_max * self.input / self.H)
                    x_min = int(x_min * self.input / self.W)
                    x_max = int(x_max * self.input / self.W)
                    train_txt.write(f"{line}-{x_min}-{y_min}-{x_max}-{y_max}-{c}-0\n")
        
        train_txt.close()
    
    def extract_test_index(self,):
        "Subject3_rgbd_images_arranging_objects_0510143426-0000-x1_y1_x2_y2-class-flag"
        test_txt_name = os.path.join(self.root_path,'test_list_dense.txt')
        test_txt = open(test_txt_name, 'a')
        
        with open(os.path.join(self.test_list), 'r') as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip()
            print(line)
            [cate, frame_index] = line.split('-')           
            att_obj_ann_name = os.path.join(self.root_path, 'attention_object_annotation', 'test', f'{line}.json')
            with open(att_obj_ann_name, 'r') as file:
                atts_objs = json.load(file)
            atts = atts_objs['attentions']
            objs = atts_objs['objects']
            assert len(atts)==1 or len(atts)==2
            for att in atts:
                [c,x_min, y_min, x_max, y_max] = att
                y_min = int(y_min * self.input / self.H)
                y_max = int(y_max * self.input / self.H)
                x_min = int(x_min * self.input / self.W)
                x_max = int(x_max * self.input / self.W)
                test_txt.write(f"{line}-{x_min}-{y_min}-{x_max}-{y_max}-{c}-1\n")
            obj_num = len(objs)
            if obj_num>0:
                for obj in objs:
                    [c,x_min, y_min, x_max, y_max] = obj
                    y_min = int(y_min * self.input / self.H)
                    y_max = int(y_max * self.input / self.H)
                    x_min = int(x_min * self.input / self.W)
                    x_max = int(x_max * self.input / self.W)
                    test_txt.write(f"{line}-{x_min}-{y_min}-{x_max}-{y_max}-{c}-0\n")
        
        test_txt.close()                    

    def extract_human_skeleton_mask(self,):
        is_true = True
        if is_true:
            with open(os.path.join(self.tain_list), 'r') as file:
                lines = file.readlines()
            mkdirs([self.root_path, 'human_skeleton_mask', 'train'])
            save_type = 'train'
        else:
            with open(os.path.join(self.test_list), 'r') as file:
                lines = file.readlines()
            mkdirs([self.root_path, 'human_skeleton_mask', 'test'])
            save_type = 'test'
        
        for line in lines:
            print(line)
            self.skeleton_mask = np.zeros((224,224))
            line = line.strip()
            [cate, frame_index] = line.split('-') 
            human_sketon_ann_name = os.path.join(self.human_skeleton_path,cate,'{}_{:012d}_keypoints.json'.format(cate, int(frame_index)))
            with open (human_sketon_ann_name, 'r') as file:
                skeletons = json.load(file)  # it is a list
            persons = skeletons['people']  # person is a list
            num = len(persons)
            if num>0:
                person = self.chooseONEperson(persons)
                body = np.reshape(person['pose_keypoints_2d'],(18, 3))  # body is a array
                for pair in self.skeleton_joint:
                    if body[pair[0], 2] > 0.1 and body[pair[1], 2] > 0.1:
                        # line segment is available
                        p1 = (int((body[pair[0], 0]* self.input / self.W)),
                                int((body[pair[0], 1])* self.input / self.H))
                        p2 = (int((body[pair[1], 0]* self.input / self.W) ),
                                int((body[pair[1], 1])* self.input / self.H))
                        if p1 == p2:
                            continue
                        else:
                            # Draw line on images
                            cv2.line(self.skeleton_mask, p1, p2, (255), 10) 
            
            save_name = os.path.join(self.root_path, 'human_skeleton_mask',save_type,f"{line}.png")
            cv2.imwrite(save_name,self.skeleton_mask)
            
    def extract_human_box_mask(self,):
        is_true = False
        if is_true:
            human_box_ann_path = '/home/cqu/nzx/CAD/human_box_annotation/train'
            with open(os.path.join(self.tain_list), 'r') as file:
                lines = file.readlines()
            mkdirs([self.root_path, 'human_box_binary_mask', 'train'])
            save_type = 'train'
        else:
            human_box_ann_path = '/home/cqu/nzx/CAD/human_box_annotation/test'
            with open(os.path.join(self.test_list), 'r') as file:
                lines = file.readlines()
            mkdirs([self.root_path, 'human_box_binary_mask', 'test'])
            save_type = 'test'
        
        for line in lines:
            line = line.strip()
            # print(line)
            [cate, frame_index] = line.split('-') 
            human_box_ann_name = os.path.join(human_box_ann_path,f"{line}.json")
            with open (human_box_ann_name, 'r') as file:
                human_box = json.load(file)   
            
            tem_num = len(human_box)
            if tem_num == 0:
                print(line)
            else:
                [x00, y00, x11, y11] = human_box[1:]
            self.tem_one_data = np.zeros((self.input, self.input))
            self.tem_one_data = self.tem_one_data * [0]              
            " here should be noticed that y is first and x is second"
            self.tem_one_data[y00:y11,x00:x11] = 255
            human_box_binary_mask_name = os.path.join(self.root_path, 'human_box_binary_mask',save_type,f"{line}.png")
            # save_image(self.tem_one_data,human_box_binary_mask_name)
            cv2.imwrite(human_box_binary_mask_name,self.tem_one_data)

    def extract_object_box_mask(self,):
        # mkdirs([self.root_path, 'object_binary_mask', 'train'])
        # mkdirs([self.root_path, 'object_binary_mask', 'test'])
        # mkdirs([self.root_path, 'visualization_check', 'train'])
        # mkdirs([self.root_path, 'visualization_check', 'test'])
        
        is_true = True
        if is_true:
            obj_binary_mask_save_path = '/home/cqu/nzx/CAD/object_binary_mask/train'
            with open(os.path.join(self.tain_dense_list), 'r') as file:
                lines = file.readlines()
            save_type = 'train'
        else:
            obj_binary_mask_save_path = '/home/cqu/nzx/CAD/object_binary_mask/test'
            with open(os.path.join(self.test_dense_list), 'r') as file:
                lines = file.readlines()
            save_type = 'test'
        
        "Subject3_rgbd_images_arranging_objects_0510143426-0000-x1-y1-x2-y2-class-flag"
        for line in lines:
            line = line.strip()
            print(line)
            [video_base_name, frame_index] = line.split('-')[0:2]
            [x1,y1,x2,y2] = line.split('-')[2:6]
            [x1,y1,x2,y2] = [int(x1),int(y1),int(x2),int(y2)] 
            
            "object binary mask"
            self.tem_one_data = np.zeros((self.input, self.input))
            self.tem_one_data = self.tem_one_data * [0] 
            self.tem_one_data[y1:y2,x1:x2] = 255
            obj_binary_mask_save_name = os.path.join(obj_binary_mask_save_path, f'{line}.png')
            cv2.imwrite(obj_binary_mask_save_name,self.tem_one_data)
            
            "draw on raw image to check"
            # image_name = os.path.join(self.image_path, video_base_name,'{:04d}.png'.format(int(frame_index)))
            # raw_image = cv2.imread(image_name)
            # cv2.rectangle(raw_image, (x1,y1), (x2,y2),(255, 0, 0), thickness=2)
            # imgae_save_name = os.path.join(self.root_path, 'visualization_check',save_type, f"{video_base_name}-{frame_index}.png")
            # cv2.imwrite(imgae_save_name,raw_image)
        
      
    def human_skeleton_statis(self,):
        is_true = True
        if is_true:
            # human_skeleton_path = '/home/cqu/nzx/CAD/'
            with open(os.path.join(self.tain_list), 'r') as file:
                lines = file.readlines()
            # mkdirs([self.root_path, 'human_head_box', 'train'])
            save_type = 'train'
        else:
            # human_skeleton_path = '/home/cqu/nzx/CAD/'
            with open(os.path.join(self.test_list), 'r') as file:
                lines = file.readlines()
            # mkdirs([self.root_path, 'human_head_box', 'test'])
            save_type = 'test'
        
        head_center_x_statis = []
        head_center_y_statis = []
        d_statis = []
        neck_rlip_d_statis = []
        neck_lrip_d_statis = []
        nose_neck_not_exist = 0
        have_head_center_num = 0
        joint_num = {}
        zero_egiht_eleven_co_not_exist_num = 0
        have_person_frame_num = 0
        wrist_distance_statis = []
        for line in lines:
            line = line.strip()
            [cate, frame_index] = line.split('-') 
            human_sketon_ann_name = os.path.join(self.human_skeleton_path,cate,'{}_{:012d}_keypoints.json'.format(cate, int(frame_index)))
            with open (human_sketon_ann_name, 'r') as file:
                skeletons = json.load(file)  # it is a list
            persons = skeletons['people']  # person is a list
            num = len(persons)
            if num >0:
                head_list = [0, 14, 15, 16, 17]
                person = self.chooseONEperson(persons)
                body = np.reshape(person['pose_keypoints_2d'],(18, 3))  # body is a array
                # body = self.confidence_filter(body, 0.1)
                have_person_frame_num+=1
                "head center"
                head_center_x = []
                head_center_y = []
                for i in head_list:
                    p = body[i,2]
                    if p>0.1:
                        head_center_x.append(body[i,0])
                        head_center_y.append(body[i,1])
                assert len(head_center_x) == len(head_center_y)
                if len(head_center_x) <1:
                    print(f'head_not_detected frame: {line}')
                else:
                    have_head_center_num+=1
                    head_x = statistics.mean(head_center_x)
                    head_y = statistics.mean(head_center_y)
                    head_center_x_statis.append(head_x)
                    head_center_y_statis.append(head_y)
                    
                "nose neck distance"
                if body[0,2]>0.1 and body[1,2]>0.1:
                    d = euclideanDistance([body[0,0],body[0,1]],[body[1,0],body[1,1]])
                    d_statis.append(d)
                else:
                    nose_neck_not_exist+=1
                
                "joint statis"
                for i in range(18):
                    if body[i,2]>0.1:
                        if i not in joint_num:
                            joint_num[i] = 1
                        else:
                            joint_num[i]+=1
                
                ""
                if body[0,2]<0.1 and body[8,2]<0.1 and  body[11,2]<0.1:
                    zero_egiht_eleven_co_not_exist_num+=1
                if body[1,2]>0.1 and body[8,2]>0.1:
                    d = euclideanDistance([body[1,0],body[1,1]],[body[8,0],body[8,1]])
                    neck_rlip_d_statis.append(d)
                if body[1,2]>0.1 and body[11,2]>0.1:
                    d = euclideanDistance([body[1,0],body[1,1]],[body[11,0],body[11,1]])
                    neck_lrip_d_statis.append(d)
                
            else:
                print(f"No exist skeleton frame： {line}") 
                
        head_center_x_mean = statistics.mean(head_center_x_statis)    
        head_center_y_mean = statistics.mean(head_center_y_statis)
        neck_nose_d_mean = statistics.mean(d_statis)
        neck_rlip_d_statis_mean = statistics.mean(neck_rlip_d_statis)
        neck_lrip_d_statis_mean = statistics.mean(neck_lrip_d_statis)
        print(f"have_person_frame_num == {have_person_frame_num}")
        print(f"nose_neck_not_exist_num == {nose_neck_not_exist}\n")
        print(f"have head center num =={have_head_center_num}\n")   
        print(f"zero_egiht_eleven_co_not_exist_num=={zero_egiht_eleven_co_not_exist_num}")
        print(f"head_center_x_statis == {head_center_x_mean}\n")
        print(f"head_center_y_statis == {head_center_y_mean}\n")
        print(f"neck_nose_d_mean == {neck_nose_d_mean}\n") 
        print(f"neck_rlip_d_statis_mean == {neck_rlip_d_statis_mean}\n")   
        print(f"neck_lrip_d_statis_mean == {neck_lrip_d_statis_mean}\n")    
        print(f"joint num == {joint_num}")  
                    
                            
    def human_body_part_extraction(self,):
        dis_threhold = 0.9
        is_true = True
        if is_true:
            with open(os.path.join(self.tain_list), 'r') as file:
                lines = file.readlines()
            mkdirs([self.root_path, 'human_body_part_box', 'train'])
            save_type = 'train'
        else:
            with open(os.path.join(self.test_list), 'r') as file:
                lines = file.readlines()
            mkdirs([self.root_path, 'human_body_part_box', 'test'])
            save_type = 'test'
        
        self.l_wrist_distance = 25
        self.r_wrist_distance = 25
        for line in lines:
            line = line.strip()
            [cate, frame_index] = line.split('-') 
            human_sketon_ann_name = os.path.join(self.human_skeleton_path,cate,'{}_{:012d}_keypoints.json'.format(cate, int(frame_index)))
            with open (human_sketon_ann_name, 'r') as file:
                skeletons = json.load(file)  # it is a list
            persons = skeletons['people']  # person is a list
            num = len(persons)
            
            body_part = {}
            if num >0:        
                image_name = os.path.join(self.raw_image_path, cate,'{:04d}.png'.format(int(frame_index)))
                raw_image = cv2.imread(image_name)       
                person = self.chooseONEperson(persons)
                body = np.reshape(person['pose_keypoints_2d'],(18, 3))  # body is a array
                # body = self.confidence_filter(body, 0.1)"

                "wrist"
                if body[4,2]>0.1:
                    r_wrist_point = [int(body[4,0]),int(body[4,1]+5)]
                    body_part['r_wrist_center'] = r_wrist_point
                    if body[2,2]>0.1 and body[3,2]>0.1:
                        self.r_wrist_distance = euclideanDistance([body[3,0],body[3,1]],[body[2,0],body[2,1]])
                        self.r_wrist_distance = 0.5*self.r_wrist_distance
                    x1 = r_wrist_point[0]-self.r_wrist_distance
                    x2 = r_wrist_point[0]+self.r_wrist_distance
                    y1 = r_wrist_point[1]-self.r_wrist_distance
                    y2 = r_wrist_point[1]+self.r_wrist_distance
                    body_part['r_wrist_box'] = [int(x1), int(y1), int(x2), int(y2)]
                    "visual to check"
                    # cv2.rectangle(raw_image, (int(x1), int(y1)), (int(x2), int(y2)),(0, 255, 0), thickness=2)
                else:
                    body_part['r_wrist_box'] = []
                    body_part['r_wrist_center'] = []
                    
                if body[7,2]>0.1:
                    l_wrist_point = [int(body[7,0]),int(body[7,1]+5)]
                    body_part['l_wrist_center'] = l_wrist_point
                    if body[5,2]>0.1 and body[6,2]>0.1:
                        self.r_wrist_distance = euclideanDistance([body[5,0],body[5,1]],[body[6,0],body[6,1]])
                        self.r_wrist_distance = 0.5*self.r_wrist_distance
                    x1 = l_wrist_point[0]-self.l_wrist_distance
                    x2 = l_wrist_point[0]+self.l_wrist_distance
                    y1 = l_wrist_point[1]-self.l_wrist_distance
                    y2 = l_wrist_point[1]+self.l_wrist_distance
                    body_part['l_wrist_box'] = [int(x1), int(y1), int(x2), int(y2)]
                    "visual to check"
                    # cv2.rectangle(raw_image, (int(x1), int(y1)), (int(x2), int(y2)),(0, 255, 0), thickness=2)
                else:
                    body_part['l_wrist_box'] = []
                    body_part['l_wrist_center'] = []
                
                "head center"
                head_list = [0, 14, 15, 16, 17]
                head_center_x = []
                head_center_y = []
                for i in head_list:
                    p = body[i,2]
                    if p>0.1:
                        head_center_x.append(body[i,0])
                        head_center_y.append(body[i,1])
                assert len(head_center_x) == len(head_center_y)
                if len(head_center_x) == 0:
                    print(f'head_center_not_detected frame: {line}')
                else:
                    self.head_center_x = statistics.mean(head_center_x)
                    self.head_center_y = statistics.mean(head_center_y)
                
                "head"
                if len(head_center_x)>0 and body[1,2]>0.1:
                    
                    self.hean_neck_d = euclideanDistance([self.head_center_x,self.head_center_y],[body[1,0],body[1,1]])
                    self.hean_neck_d *= dis_threhold
                    x1 = self.head_center_x-self.hean_neck_d
                    x2 = self.head_center_x+self.hean_neck_d
                    y1 = self.head_center_y-self.hean_neck_d
                    y2 = self.head_center_y+self.hean_neck_d
                    
                    body_part['head_center'] = [int(self.head_center_x), int(self.head_center_y)]
                    body_part['head_box'] = [int(x1), int(y1), int(x2), int(y2)]

                    "visual to check"                   
                    # cv2.rectangle(raw_image, (int(x1), int(y1)), (int(x2), int(y2)),(255, 0, 0), thickness=2)
                    # self.draw_skeleton(raw_image,body)
                    # cv2.circle(raw_image, center=(int(self.head_center_x), int(self.head_center_y)), radius=3,color=(255,0,0),thickness=3,lineType=8)
                  
                else:
                    body_part['head_box']=[]
                    body_part['head_center']  = []
                    print(f"head box failed frame: {line}")
                
                # imgae_save_name = os.path.join(self.root_path, 'human_body_part_box_visualization',save_type, f"{line}.png")
                # cv2.imwrite(imgae_save_name,raw_image)
                
            else:
                body_part['head_box']=[]
                body_part['head_center'] = []
                body_part['l_wrist_box'] = []
                body_part['r_wrist_box'] = []
                body_part['l_wrist_center'] = []
                body_part['r_wrist_center'] = []
                print(f"No exist skeleton frame： {line}") 
            
            "save"
            save_name = os.path.join(self.root_path, 'human_body_part_box',save_type, f"{line}.json")
            with open(save_name, 'w') as f:
                json.dump(body_part, f) 
            
                
    def draw_skeleton(self,image,body):
        
        for pair in self.skeleton_joint:
            if body[pair[0], 2] > 0.1 and body[pair[1], 2] > 0.1:
                # line segment is available
                p1 = (int((body[pair[0], 0])),
                        int((body[pair[0], 1])))
                p2 = (int((body[pair[1], 0]) ),
                        int((body[pair[1], 1])))
                if p1 == p2:
                    continue
                else:
                    # Draw line on images
                    cv2.line(image, p1, p2, (255,255,0), 3) 
                    cv2.circle(image, center=p1, radius=2,color=(0,0,255),thickness=2,lineType=8)
                    cv2.circle(image, center=p2, radius=2,color=(0,0,255),thickness=2,lineType=8)
        # return image
                                    
                    
    def extract_human_joint_guassian_mask(self,):

        pass
    def chooseONEperson(self, peoples):
        # peoples are the list of people from openpose, each people is a dict
        bestpeople = None
        max_confidece = -100
        for people in peoples:
            pose_keypoints_2d = people['pose_keypoints_2d']
            # array,  the number is 54 = 18*3 , (x, y , confidence)
            pose_keypoints_2d = np.reshape(pose_keypoints_2d, (18, 3))
            confidence = np.sum(pose_keypoints_2d, axis=0)[2]
            if confidence > max_confidece:
                max_confidece = confidence
                bestpeople = people
        return bestpeople
    
    def confidence_filter(self, data, con_threshold):
        # data is n*3 array
        # to filter the face, body, hand with low confidence
        n = data.shape[0]
        for i in range(n):
            con = data[i, 2]
            if con < con_threshold:
                data[i] = [0.0, 0.0, 0]
        return data

    def load_object_detection(self, pattern, frame_name):
        flag = 1
        path_name = None
        if pattern == 'CAD':
            # path = self.root_path + ''
            path = os.path.join(self.root_path,'CAD120_focusloss_detection')
            [cate, frame_index] = frame_name.split('-')
            path_name = os.path.join(path, cate,
                                     '{}.json'.format(frame_index))

        elif pattern == 'TIA':
            # path = self.root_path + 'TIA_backup/'
            path = os.path.join(self.root_path,'TIA_backup','focusloss_detection')
            [cate, video_base_name, frame_index] = frame_name.split('-')
            path_name = os.path.join(path, cate, video_base_name,
                                     f'{self.img_name[:-3]}json')

        if not os.path.exists(path_name):
            print("path not exists",path_name)
            flag = 0
            filtered_objects = []

        else:
            with open(path_name, 'r') as f:
                objects = json.load(f)
            filtered_objects = self.filter_objtect(objects)
            obj_num = len(filtered_objects)
            if obj_num < 1:
                flag = 0
        return flag, filtered_objects
    
    def filter_objtect(self, detected_objects):
        # filter_objects
        filtered_objects = []
        for obj in detected_objects:
            [name, x1, y1, x2, y2, prob] = obj
            if prob > self.focus_loss_IOU_threshold:
                filtered_objects.append(obj)
        return filtered_objects
    
def main():
    xx = DenseInteraction_Preparation()
    # xx.extract_test_index()
    # xx.extract_training_index()
    # xx.extract_human_box_mask()
    # xx.extract_object_box_mask()
    # xx.human_skeleton_statis()
    # xx.human_body_part_extraction()
    # xx.extract_human_skeleton_mask()
    xx.extract_testing_index_obj_detection()

if __name__ == '__main__':
    main()