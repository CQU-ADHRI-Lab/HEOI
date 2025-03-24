import os,json,cv2,statistics
from matplotlib import image
import numpy as np
from util import mkdir, mkdirs, euclideanDistance

class DenseInteraction_Preparation():
    def __init__(self):
        self.pattern = 'TIA'
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
    
    def generate_dense_train_test_list(self,):
        "Subject3_rgbd_images_arranging_objects_0510143426-0000-x1-y1-x2-y2-class-flag"
        "c14_open_door-S53_V10_T20161228_034010_00_FHXL2_PD9thFloor_G05-138"
        is_train = False
        if is_train:
            txt_name = os.path.join(self.root_path,'train_list_dense.txt')
            txt = open(txt_name, 'a')
            self.list = self.tain_list
            self.mode = 'train'           
            mkdirs([self.root_path, 'attention_object_annotation_name','train'])
           
        else:
            txt_name = os.path.join(self.root_path,'test_list_dense.txt')
            txt = open(txt_name, 'a')
            self.list = self.test_list
            self.mode = 'test'
            mkdirs([self.root_path, 'attention_object_annotation_name','test'])
           
            
        with open(os.path.join(self.list), 'r') as file:
            lines = file.readlines()
        for line in lines:
            att_obj_dict = {'attentions':[],
                            'objects':[]}
            
            # print(line)
            line = line.strip()      
            [cate, video_base_name, frame_index] = line.strip().split('-')    
            frame_index = int(frame_index)
            att_ann_name = os.path.join(self.root_path, 'attention_annotation_name', cate, video_base_name, f'AttentionBoxes_{frame_index:04d}.json')
            with open(att_ann_name, 'r') as file:
                atts = json.load(file)
            # atts = atts_objs['attentions']
            # objs = atts_objs['objects']
            att_ann_name = os.path.join(self.root_path, 'otherobjects_annotation_name', cate, video_base_name,f'otherobjects_{frame_index:04d}.json')
            with open(att_ann_name, 'r') as file:
                objs = json.load(file)
                
            assert len(atts)==1 or len(atts)==2
            for att in atts:
                [c,x_min, y_min, x_max, y_max] = att
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                y_max = min(self.H, y_max)
                
                if 0<=x_min<x_max<=self.W and 0<=y_min<y_max<=self.H:
                    att_obj_dict['attentions'].append([c,x_min, y_min, x_max, y_max])
                    y_min = int(y_min * self.input / self.H)
                    y_max = int(y_max * self.input / self.H)
                    x_min = int(x_min * self.input / self.W)
                    x_max = int(x_max * self.input / self.W)
                    txt.write(f"{line}-{x_min}-{y_min}-{x_max}-{y_max}-{c}-1\n")
                else:
                    print('att',line, att)
                # assert 0<x_min<x_max<self.W
                # assert 0<y_min<y_max<self.H
                
            obj_num = len(objs)
            if obj_num>0:
                for obj in objs:
                    [c,x_min, y_min, x_max, y_max] = obj
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    # assert 0<x_min<x_max<self.W
                    # assert 0<y_min<y_max<self.H
                    y_max = min(self.H, y_max)
                    if 0<=x_min<x_max<=self.W and 0<=y_min<y_max<=self.H:
                        att_obj_dict['objects'].append([c,x_min, y_min, x_max, y_max])
                        y_min = int(y_min * self.input / self.H)
                        y_max = int(y_max * self.input / self.H)
                        x_min = int(x_min * self.input / self.W)
                        x_max = int(x_max * self.input / self.W)
                        txt.write(f"{line}-{x_min}-{y_min}-{x_max}-{y_max}-{c}-0\n")
                    else:
                        print('obj',line, obj)
                    
            save_name = os.path.join(self.root_path, 'attention_object_annotation_name',self.mode, f"{line}.json")
            with open(save_name, 'w') as f:
                json.dump(att_obj_dict, f) 
        
        txt.close()
    
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
        is_true = False
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
        
        test_visual_num = 0
        test_visual_num_max = 30
        test_flag = 0
        
        for line in lines:
            # print(line)
            self.skeleton_mask = np.zeros((224,224))
            line = line.strip()
            # [cate, frame_index] = line.split('-') 
            # human_sketon_ann_name = os.path.join(self.human_skeleton_path,cate,'{}_{:012d}_keypoints.json'.format(cate, int(frame_index)))
            [cate, video_base_name, frame_index] = line.strip().split('-')
            frame_index = int(frame_index)
            human_sketon_ann_name = os.path.join(self.human_skeleton_path,cate,video_base_name,'{}_{:012d}_keypoints.json'.format(video_base_name, frame_index))
            
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
                            cv2.line(self.skeleton_mask, p1, p2, (255), 5) 
            
            if np.random.random_sample() <= 0.005 and test_visual_num<test_visual_num_max :
                test_visual_num+=1
                test_flag = 1
            else:
                test_flag = 0
            if test_flag == 1 and test_visual_num< test_visual_num_max:
                print(line)
                mkdirs([self.root_path, 'human_skeleton_mask_test_visual'])
                save_name = os.path.join(self.root_path, 'human_skeleton_mask_test_visual',f"{line}.png")
                cv2.imwrite(save_name,self.skeleton_mask)
            # if test_visual_num == test_visual_num_max:
            #     break
            save_name = os.path.join(self.root_path, 'human_skeleton_mask',save_type,f"{line}.png")
            cv2.imwrite(save_name,self.skeleton_mask)
        print(save_type,'finished!!!')
    
    def get_human_box(self,type):
        if type == 'train':
            file_name = self.tain_list
            mkdirs([self.root_path, 'human_box','train'])
            mkdirs([self.root_path, 'human_box_test_visual','train'])
        if type == 'test':
            file_name = self.test_list
            mkdirs([self.root_path, 'human_box','test'])
            mkdirs([self.root_path, 'human_box_test_visual','test'])
            
        No_human_detection_list = []
        with open(file_name, 'r') as file:
            lines = file.readlines()
        
        test_visual_num = 0
        test_visual_num_max = 30
        for line in lines:
            
            [cate, video_base_name, frame_index] = line.strip().split('-')
            frame_index = int(frame_index)
            openpose_file = os.path.join(self.human_skeleton_path,cate,video_base_name,'{}_{:012d}_keypoints.json'.format(video_base_name, frame_index))
            with open(openpose_file, 'r') as output:
                skeletons = json.load(output)
            persons = skeletons['people']  # person is a list
            num = len(persons)
            
            test_flag = 0
            if np.random.random_sample() <= 0.005 and test_visual_num<test_visual_num_max :
                test_visual_num+=1
                test_flag = 1
                # raw image
                raw_images = sorted(os.listdir(os.path.join(self.raw_image_path, cate, video_base_name)))
                image_name = os.path.join(self.raw_image_path, cate, video_base_name, raw_images[frame_index])
                print(image_name)
                self.image = cv2.imread(image_name)
            else:
                test_flag = 0
                 
            # find human box
            human_box = []
            have_good_human_flag = 1
            if num == 0:
                have_good_human_flag = 0
            if num > 0:
                person = self.chooseONEperson(persons) 
                body = np.reshape(person['pose_keypoints_2d'], (18, 3)) # body is a array
                confidence_points = self.human_skeleton_points_filter(body, 0.1)
                point_num = len(confidence_points)
                if point_num < 4:
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
                    x_min,y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    w00 = int(x_max-x_min)
                    h00 = int(y_max-y_min)
                    x_min = x_min - 0.1*w00
                    y_min = y_min - 0.1*h00
                    x_max = x_max + 0.1*w00
                    y_max = y_max + 0.1*h00
                    
                    x_min = max(0,x_min)
                    y_min = max(0,y_min)
                    x_max = min(self.W, x_max)
                    y_max = min(self.H, y_max)
                    
                    x_min,y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    
                    assert 0<=x_min<x_max<=self.W
                    assert 0<=y_min<y_max<=self.H
                    I = 224
                    "visual to check"
                    if test_visual_num<test_visual_num_max and test_flag == 1:
                        imgae_save_name = os.path.join(self.root_path, 'human_box_test_visual',type, f"{line}.png")
                        cv2.rectangle(self.image, (x_min,y_min), (x_max, y_max),(255, 0, 0), thickness=2)
                        cv2.imwrite(imgae_save_name,self.image)
                    
                    x00 = int(x_min*I/self.W)
                    y00 = int(y_min*I/self.H)
                    # w00 = int(x_max*I/self.W-x_min*I/self.W)
                    # h00 = int(y_max*I/self.H-y_min*I/self.H)
                    x11 = int(x_max*I/self.W)
                    y11 = int(y_max*I/self.H) 
                    # x11 = min(x11, 224) 
                    # y00 = max(0, y00-int(0.1*h00)) 
                    human_box = ['human', x00, y00, x11, y11]
                
            if have_good_human_flag == 0:
                No_human_detection_list.append(line)
                print('No good human',line)
                   
            # if test_visual_num==test_visual_num_max:
            #     break  
            save_name =  os.path.join(self.root_path, 'human_box', type, '{}.json'.format(line) ) 
            with open(save_name, 'w') as f:
                json.dump(human_box, f) 
        print(type, 'finished')            
    

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
        is_train = True
        if is_train:
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
        total_frame = 0
        neck_num = 0
        for line in lines:
            total_frame+=1
            line = line.strip()
            # [cate, frame_index] = line.split('-') 
            [cate, video_base_name, frame_index] = line.strip().split('-')
            frame_index = int(frame_index)
            human_sketon_ann_name = os.path.join(self.human_skeleton_path,cate,video_base_name,'{}_{:012d}_keypoints.json'.format(video_base_name, frame_index))
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
                # if body[0,2]>0.1 and body[1,2]>0.1:
                #     d = euclideanDistance([body[0,0],body[0,1]],[body[1,0],body[1,1]])
                #     d_statis.append(d)
                # else:
                #     nose_neck_not_exist+=1
                " neck joint"
                if body[1,2]>0.1:
                    neck_num += 1
                    
                "joint statis"
                for i in range(18):
                    if body[i,2]>0.1:
                        if i not in joint_num:
                            joint_num[i] = 1
                        else:
                            joint_num[i]+=1
                
                "nose, left hio, right hip"
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
        # neck_nose_d_mean = statistics.mean(d_statis)
        neck_rlip_d_statis_mean = statistics.mean(neck_rlip_d_statis)
        neck_lrip_d_statis_mean = statistics.mean(neck_lrip_d_statis)
        
        print(f"total_frame_num == {total_frame}") 
        print(f"have_person_frame_num == {have_person_frame_num}")
        print(f"have_neck_frame_num == {neck_num}")
        # print(f"nose_neck_not_exist_num == {nose_neck_not_exist}\n")
        print(f"have head center num =={have_head_center_num}\n")   
        # print(f"zero_egiht_eleven_co_not_exist_num=={zero_egiht_eleven_co_not_exist_num}")
        print(f"head_center_x_statis == {head_center_x_mean}\n")
        print(f"head_center_y_statis == {head_center_y_mean}\n")
        
        # print(f"neck_nose_d_mean == {neck_nose_d_mean}\n") 
        print(f"neck_rlip_d_statis_mean == {neck_rlip_d_statis_mean}\n")   
        print(f"neck_lrip_d_statis_mean == {neck_lrip_d_statis_mean}\n")    
        print(f"joint num == {joint_num}")  
                    
                            
    def human_body_part_extraction(self,):
        dis_threhold = 0.9
        is_train = True
        if is_train:
            with open(os.path.join(self.tain_list), 'r') as file:
                lines = file.readlines()
            mkdirs([self.root_path, 'human_body_part_box', 'train'])
            save_type = 'train'
        else:
            with open(os.path.join(self.test_list), 'r') as file:
                lines = file.readlines()
            mkdirs([self.root_path, 'human_body_part_box', 'test'])
            save_type = 'test'
        if self.pattern == 'CAD':
            self.l_wrist_distance = 25
            self.r_wrist_distance = 25
        else:
            self.l_wrist_distance = 40
            self.r_wrist_distance = 40
            self.head_distance_min = 50
            
        test_visual_num = 0
        test_visual_num_max = 30
        test_flag = 0
        for line in lines:
            line = line.strip()
            # [cate, frame_index] = line.split('-') 
            # human_sketon_ann_name = os.path.join(self.human_skeleton_path,cate,'{}_{:012d}_keypoints.json'.format(cate, int(frame_index)))
            [cate, video_base_name, frame_index] = line.strip().split('-')
            frame_index = int(frame_index)
            human_sketon_ann_name = os.path.join(self.human_skeleton_path,cate,video_base_name,'{}_{:012d}_keypoints.json'.format(video_base_name, frame_index))
            
            if np.random.random_sample() <= 0.0005 and test_visual_num<test_visual_num_max :
                test_visual_num+=1
                test_flag = 1
                # raw image
                raw_images = sorted(os.listdir(os.path.join(self.raw_image_path, cate, video_base_name)))
                image_name = os.path.join(self.raw_image_path, cate, video_base_name, raw_images[frame_index])
                print(image_name)
                self.image = cv2.imread(image_name)
            else:
                test_flag = 0
            
            with open (human_sketon_ann_name, 'r') as file:
                skeletons = json.load(file)  # it is a list
            persons = skeletons['people']  # person is a list
            num = len(persons)
            
            body_part = {}
            if num >0:        
                    
                person = self.chooseONEperson(persons)
                body = np.reshape(person['pose_keypoints_2d'],(18, 3))  # body is a array
                # body = self.confidence_filter(body, 0.1)"
                "right wrist"
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
                    if test_visual_num<test_visual_num_max and test_flag == 1:
                        cv2.rectangle(self.image, (int(x1), int(y1)), (int(x2), int(y2)),(0, 255, 0), thickness=2)
                else:
                    body_part['r_wrist_box'] = []
                    body_part['r_wrist_center'] = []
                
                "left wrist"
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
                    if test_visual_num<test_visual_num_max and test_flag == 1:
                        cv2.rectangle(self.image, (int(x1), int(y1)), (int(x2), int(y2)),(0, 255, 0), thickness=2)
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
                    
                    "head center and neck distance"
                    self.hean_neck_d = euclideanDistance([self.head_center_x,self.head_center_y],[body[1,0],body[1,1]])
                    self.hean_neck_d *= dis_threhold
                    self.hean_neck_d = max(self.hean_neck_d, self.head_distance_min)
                    
                    x1 = self.head_center_x-self.hean_neck_d
                    x2 = self.head_center_x+self.hean_neck_d
                    y1 = self.head_center_y-self.hean_neck_d
                    y2 = self.head_center_y+self.hean_neck_d
                    
                    body_part['head_center'] = [int(self.head_center_x), int(self.head_center_y)]
                    body_part['head_box'] = [int(x1), int(y1), int(x2), int(y2)]
    
                    "visual to check"
                    if test_visual_num<test_visual_num_max and test_flag == 1:
                        cv2.rectangle(self.image, (int(x1), int(y1)), (int(x2), int(y2)),(0, 0, 255), thickness=2)              
                    # cv2.rectangle(raw_image, (int(x1), int(y1)), (int(x2), int(y2)),(255, 0, 0), thickness=2)
                    # self.draw_skeleton(raw_image,body)
                    # cv2.circle(raw_image, center=(int(self.head_center_x), int(self.head_center_y)), radius=3,color=(255,0,0),thickness=3,lineType=8)
                  
                else:
                    body_part['head_box']=[]
                    body_part['head_center']  = []
                    print(f"head box failed frame: {line}")
                
                "visual to check"
                if test_visual_num<test_visual_num_max and test_flag == 1:
                    mkdirs([self.root_path, 'human_body_part_box_visualization',save_type])
                    imgae_save_name = os.path.join(self.root_path, 'human_body_part_box_visualization',save_type, f"{line}.png")
                    cv2.imwrite(imgae_save_name,self.image)
                
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
    
    def human_skeleton_points_filter(self, data, con_threshold):
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
    
def main():
    xx = DenseInteraction_Preparation()
    # xx.generate_dense_train_test_list()
    # xx.extract_training_index()
    # xx.extract_human_box_mask()
    # xx.extract_object_box_mask()
    # xx.human_skeleton_statis()
    # xx.human_body_part_extraction()
    # xx.get_human_box('train')
    xx.get_human_box('test')

if __name__ == '__main__':
    main()