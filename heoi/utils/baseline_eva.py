import os,json
import os,json,cv2,statistics
from matplotlib import image
import numpy as np
from util import mkdir, mkdirs, euclideanDistance
from utils_metric import L2_dist,angle_compute
import time

### evaluate NIPS'2015 distance metric and angle metric
class WhereAreTheyMetric():
    def __init__(self):
        self.pattern = 'TIA'
        self.save = 0
        self.resize_H = 240
        self.resize_W = 320
        
        if self.pattern == 'CAD':
            self.H = 480
            self.W = 640
            self.root_path = '/home/cqu/nzx/CAD'
            self.raw_image_path = '/home/cqu/nzx/CAD/raw_image'
            self.test_list = '/home/cqu/nzx/CAD/test_list.txt'
            self.human_skeleton_path = '/home/cqu/nzx/CAD/skeleton_openpose_json'
            self.gaze_point_path = '/home/cqu/nzx/check_points/baseline_2015_NIPS_where are they looking'
            self.attention_path  =  self.root_path + '/attention_annotation_name'
  
            mkdirs([self.gaze_point_path,'vis_exp_CAD'])
            self.exp_result = os.path.join(self.gaze_point_path,'vis_exp_CAD','CAD_angle_dist_exp.txt')
        
        if self.pattern == 'TIA':
            self.H = 1080
            self.W = 1920
            self.root_path = '/home/cqu/nzx/TIA'
            self.raw_image_path = '/home/cqu/nzx/TIA/raw_image'
            self.test_list = '/home/cqu/nzx/TIA/test_list.txt'
            self.human_skeleton_path = '/home/cqu/nzx/TIA/skeleton'
            self.gaze_point_path = '/home/cqu/nzx/check_points/baseline_2015_NIPS_where are they looking'
            self.attention_path  =  self.root_path + '/attention_annotation_name'
            
            mkdirs([self.gaze_point_path,'vis_exp_TIA'])
            self.exp_result = os.path.join(self.gaze_point_path,'vis_exp_TIA','TIA_angle_dist_exp.txt')
            
    def metric_eva(self,):
        with open(os.path.join(self.test_list), 'r') as file:
            lines = file.readlines()
        if self.pattern == 'CAD':
            with open(os.path.join(self.gaze_point_path,'CAD_gaze_result.txt'), 'r') as file:
                gaze_lines = file.readlines()
        if self.pattern == 'TIA':
            with open(os.path.join(self.gaze_point_path,'TIA_gaze_result.txt'), 'r') as file:
                gaze_lines = file.readlines()
                
        print(len(lines))
        dist = []
        angle = []
        for idx, line in enumerate(lines):          
            line = line.strip()
            self.line = line
            
            if self.pattern == 'CAD':
                [cate, frame_index] = line.split('-') 
                if idx%100 == 0:
                    print(line)
                
                # attention point
                name = os.path.join(self.attention_path, cate,'attention-{}.json'.format(frame_index))
                with open(name, 'r') as file:
                    attentions = json.load(file)
                att_point = self.attention_point_compute(attentions)
                att_point = np.array(att_point)
                
                # gaze point
                gaze_point = gaze_lines[idx].strip().split(',')
                gaze_point = [float(gaze_point[0]),float(gaze_point[1])]
                gaze_point = np.array(gaze_point)

                # head point
                human_sketon_ann_name = os.path.join(self.human_skeleton_path,cate,'{}_{:012d}_keypoints.json'.format(cate, int(frame_index)))
                with open (human_sketon_ann_name, 'r') as file:
                    skeletons = json.load(file)  # it is a list
                persons = skeletons['people']  # person is a list
                head_point = self.head_point_compute(persons)
                head_point = np.array(head_point)
                
                # dist compute
                ecu_dist = L2_dist(att_point, gaze_point)
                dist.append(ecu_dist)
                
                # angle compute
                gaze_direction = gaze_point-head_point
                gt_direction = att_point-head_point
                if np.sqrt(gaze_direction.dot(gaze_direction)) == 0 or np.sqrt(gt_direction.dot(gt_direction)) == 0:
                    # print(gaze_direction,gt_direction, gaze_point, att_point, head_point )
                    continue
                each_angle = angle_compute(gaze_direction,gt_direction)
                angle.append(each_angle)
                
                # visualization
                if self.save:
                    image_name = os.path.join(self.raw_image_path, cate,'{:04d}.png'.format(int(frame_index)))
                    raw_image = cv2.imread(image_name) 
                    head_x, head_y = int(head_point[0]), int(head_point[1])
                    gaze_x, gaze_y = int(gaze_point[0]), int(gaze_point[1])
                    gt_x, gt_y = int(att_point[0]), int(att_point[1])
                    
                    cv2.line(raw_image, (head_x, head_y), (gt_x,gt_y),color=(0,0,255),thickness=4,lineType=8)               
                    cv2.line(raw_image, (head_x, head_y), (gaze_x,gaze_y),color=(0,255,0),thickness=4,lineType=8)
                    
                    cv2.circle(raw_image, center=(head_x, head_y), radius=3,color=(255,255,255),thickness=6,lineType=8)
                    cv2.circle(raw_image, center=(gt_x, gt_y), radius=5,color=(0,0,255),thickness=10,lineType=8)
                    cv2.circle(raw_image, center=(gaze_x, gaze_y), radius=5,color=(0,255,0),thickness=10,lineType=8)
                    
                    raw_image = cv2.resize(raw_image, (self.resize_W, self.resize_H)) 
                    save_name = os.path.join(self.gaze_point_path,'vis_exp',f'{line}.png')
                    cv2.imwrite(save_name,raw_image)
            
            if self.pattern == 'TIA':
                [task, video_base_name, frame_index] = line.split("-")
                if idx%100 == 0:
                    print(line)
                
                # attention point
                name = os.path.join(self.attention_path, task, video_base_name, f'AttentionBoxes_{frame_index.zfill(4)}.json')
                with open(name, 'r') as file:
                    attentions = json.load(file)
                att_point = self.attention_point_compute(attentions)
                att_point = np.array(att_point)
                
                # gaze point
                gaze_point = gaze_lines[idx].strip().split(',')
                gaze_point = [float(gaze_point[0]),float(gaze_point[1])]
                gaze_point = np.array(gaze_point)

                # head point
                human_sketon_ann_name = os.path.join(self.human_skeleton_path, task, video_base_name,'{}_{:012d}_keypoints.json'.format(video_base_name, int(frame_index)))
                with open (human_sketon_ann_name, 'r') as file:
                    skeletons = json.load(file)  # it is a list
                persons = skeletons['people']  # person is a list
                head_point = self.head_point_compute(persons)
                head_point = np.array(head_point)
                
                # dist compute
                ecu_dist = L2_dist(att_point, gaze_point)
                dist.append(ecu_dist)
                
                # angle compute
                gaze_direction = gaze_point-head_point
                gt_direction = att_point-head_point
                if np.sqrt(gaze_direction.dot(gaze_direction)) == 0 or np.sqrt(gt_direction.dot(gt_direction)) == 0:
                    print(gaze_direction,gt_direction, gaze_point, att_point, head_point )
                    continue
                each_angle = angle_compute(gaze_direction,gt_direction)
                angle.append(each_angle)
                
                # visualization
                if self.save:
                    raw_images = sorted(os.listdir(os.path.join(self.raw_image_path, task, video_base_name)))
                    image_name = os.path.join(self.raw_image_path, task,video_base_name, raw_images[int(frame_index)])
                    raw_image = cv2.imread(image_name) 
                    head_x, head_y = int(head_point[0]), int(head_point[1])
                    gaze_x, gaze_y = int(gaze_point[0]), int(gaze_point[1])
                    gt_x, gt_y = int(att_point[0]), int(att_point[1])
                    
                    cv2.line(raw_image, (head_x, head_y), (gt_x,gt_y),color=(0,0,255),thickness=10,lineType=8)               
                    cv2.line(raw_image, (head_x, head_y), (gaze_x,gaze_y),color=(0,255,0),thickness=10,lineType=8)
                    
                    cv2.circle(raw_image, center=(head_x, head_y), radius=3,color=(255,255,255),thickness=15,lineType=8)
                    cv2.circle(raw_image, center=(gt_x, gt_y), radius=5,color=(0,0,255),thickness=20,lineType=8)
                    cv2.circle(raw_image, center=(gaze_x, gaze_y), radius=5,color=(0,255,0),thickness=20,lineType=8)
                    
                    raw_image = cv2.resize(raw_image, (self.resize_W, self.resize_H)) 
                    if not os.path.exists(os.path.join(self.gaze_point_path,'vis_exp_TIA',task)):
                        mkdirs([self.gaze_point_path,'vis_exp_TIA',task])
                    save_name = os.path.join(self.gaze_point_path,'vis_exp_TIA',task,f'{line}.png')
                    cv2.imwrite(save_name,raw_image)
            
            # visual check examples    
            # if idx == 20:
            #     break    
              
        avg_dist = statistics.mean(dist)      
        avg_angle = statistics.mean(angle)   
          
        self.f_result = open(self.exp_result, 'a')
        system_t = str(time.ctime())
        self.f_result.write(f"{system_t}\n")
        self.f_result.write(f'avg_dist: {avg_dist}, avg_angle: {avg_angle}\n')
        self.f_result.close() 
        
    
    def attention_point_compute(self,attentions):
        attention = attentions[0]
        [x_min, y_min, x_max, y_max] = attention[1:]
        [x_min, y_min, x_max, y_max] = [int(x_min), int(y_min), int(x_max), int(y_max)]
        [x_min, y_min, x_max, y_max] = [max(0,x_min),y_min,x_max,min(self.H,y_max)] # specially for CAD dataset
        if not 0<=x_min<=x_max<=self.W:
            print(attention,self.line)
        if not 0<=y_min<=y_max<=self.H:
            print(attention,self.line)
        return [0.5*(x_min+x_max),0.5*(y_min+y_max)]
            
    def head_point_compute(self,persons):
        num = len(persons)
        if num == 0:
            return [0,0]
        if num > 0:
            person = self.chooseONEperson(persons) 
            head_list = [0, 14, 15, 16, 17]  
            body = np.reshape(person['pose_keypoints_2d'],(18, 3))  # body is a array
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
                    return [0,0]
                else:
                    head_x = statistics.mean(head_center_x)
                    head_y = statistics.mean(head_center_y)
                    return[head_x,head_y]
            
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
     
def main():
    xx = WhereAreTheyMetric()
    # xx.extract_test_index()
    # xx.extract_training_index()
    # xx.extract_human_box_mask()
    # xx.extract_object_box_mask()
    # xx.human_skeleton_statis()
    # xx.human_body_part_extraction()
    xx.metric_eva()

if __name__ == '__main__':
    main()