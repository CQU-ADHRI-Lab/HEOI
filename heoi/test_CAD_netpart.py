import os
import json
import cv2
from cv2 import mean
import numpy as np
import torch
import statistics
from utils.util import mkdirs, tensor2im
from utils.util import IOU_compute
from options.test_opt import TOptions
from data_n.factory_dataloader import CustomDatasetDataLoader
from model.factory_models import ModelsFactory
import matplotlib.pyplot as plt
from statistics import mean
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class Test:
    def __init__(self):
        self._opt = TOptions().parse()

        self._save_path = os.path.join(self._opt.checkpoints_dir,self._opt.name)
        
        self.image_size = self._opt.image_size
        test_loader = CustomDatasetDataLoader(self._opt, is_for_train=False)
        self.data_test = test_loader.load_data()
        self.data_test_size = len(test_loader)
        self.plot_step = int(self.data_test_size // (self._opt.batch_size*30))
        self.data_name = self._opt.dataset_mode

        if self.data_name.startswith('CAD'):
            self.pattern = 'CAD'
            self.data_path = self._opt.data_root + 'CAD/'
            self.H = 480
            self.W = 640
        else:
            pass
        
        " start epoch"
        self.start_epoch = 1

        " save or not"
        self.is_save = 1
        self.best_epoch_id = 1
        self.img_path = '/home/cqu/nzx/CAD/raw_image_224/'
       
        
        " cross metric "
        self.is_test_using_cross_criterion = 1  #自己提出来的metric, 直线与物体gt相交
        
        "object detection or GT"
        self.focus_loss_IOU_threshold = 0.6
        self.is_test_using_object_detection = 1

        " loss"
        self.global_loss = []
        
        " experiment path"
        self.exp_path = os.path.join(self._save_path, 'exp_result')
        self.img_name = None
        
        " run"
        self.run()
        
    def run(self):
        
        self.file_init()
        
        self.epoch_index = 7 # 第 个epoch的结果是最好的
        
        max_score = 0.001
        min_score = 0
        
        best_acc = 0
        best_iepoch = 0
        acc_list = []
        
        angle_avg_list=[]; dist_avg_list=[]
        
        match_key = {}
                        
        # for i_epoch in range(self.start_epoch,self.count_checkpoint_num()+1):
        # 
        for i_epoch in range(self.epoch_index,self.epoch_index+1):
            " load checkpoint "
            self._opt.load_epoch = i_epoch
            load_filename = 'net_epoch_%d_id_G.pth' % (i_epoch)
            load_path = os.path.join(self._save_path, load_filename)
            if not os.path.exists(load_path):
                print('No trained network !!!')
            model = ModelsFactory.get_by_name(self._opt.model_name, self._opt)
        
            " test"
            angle_list=[];dist_list=[]
            
            i = 1
            self.pre_result = None
            
            for i_val_batch, val_batch in enumerate(self.data_test):
                print(f'Epoch: {i_epoch} batch:{i_val_batch}/{len(self.data_test)}')
                "test"
                task_total_frames = {}
                task_correct_frames = {}
                
                " input data"
                model.set_input(val_batch)      
            
                "Forward, output and loss"
                batch_output = model.G_forward_and_loss_compute()
                self.pre_result = batch_output['result'] # dict key = frame_name, dict valule = [ score-x1-y1-x2-y2, , ]
               
                k_list = list(sorted(self.pre_result.keys()))
                i = i-1
                
                for key in k_list[i:]:
                    key = key.strip()
                    [cate, frame_index] = key.split('-') 
                    task_name = '_'.join(cate.split('_')[3:5])
                    "Subject3_rgbd_images_arranging_objects_0510143426-0000"
                    
                    box_pre_list = self.pre_result[key]
                    
                    best_score = -100
                    
                    for box in box_pre_list:
                        [x1,y1,x2,y2] = box.strip().split('_')[1].split("-")
                        
                        x1 = int(float(x1))
                        x2 = int(float(x2))
                        y1 = int(float(y1))
                        y2 = int(float(y2))
                      
                        score =float(box.strip().split('_')[0])
                        score = format(score,".3f")
                        score = float(score)
                    
                        if score>max_score:
                            max_score = score
                        if score<min_score:
                            min_score = score
                        
                        if score > best_score:
                            best_score = score
                            best_obj = [best_score,x1,y1,x2,y2]   # best_obj就是最后预测得到的一个box
                                
                    match_key[key] = []
                    match_key[key].append(best_obj)
                    i += 1
                    
            keys = list(match_key.keys())
           
            for i in range(len(match_key)):
                key = keys[i]
                best_obj = match_key[key]
                [cate, frame_index] = key.split('-') 
                task_name = '_'.join(cate.split('_')[3:5])
                image_name = os.path.join(self.data_path,'raw_image' ,cate,'{:04d}.png'.format(int(frame_index)))
                self.raw_image_640 = cv2.imread(image_name)     
                is_exist_head, head_point = self.find_head_location(key)   # 获取头部的box
              
                is_correct = self.gaze_direction_box_cross_metric('CAD', key, best_obj,is_exist_head, head_point)  # 判断是否相交
                is_exist_metric,dist, angle = self.angle_dist_metric('CAD', key, best_obj, is_exist_head, head_point) # 计算metric:角度和距离
                
                " total frames"
                if task_name not in task_total_frames:
                    task_total_frames[task_name] = 1
                else:
                    task_total_frames[task_name] += 1
                    
                " correct frames"
                if is_correct:
                    if task_name not in task_correct_frames:
                        task_correct_frames[task_name] = 1
                    else:                                   
                        task_correct_frames[task_name] += 1
                
                if is_exist_metric:
                    angle_list.append(angle)
                    dist_list.append(dist)
                
                if self.is_save:  
                    save_task_name = 'arranging_objects'
                    vis_path = '/home/cqu/jly/2023_FGHOIAttention_v2_1_CAD/vis/arranging_objects'  # path to save visualization images

                "visualization, head to best-obj line"
                if self.is_save == 1 and is_exist_head and task_name == save_task_name:
                    attentions = self.load_attentions(self.pattern, key) 
                    for attention in attentions:
                        [att_name, att_x1, att_y1, att_x2, att_y2] = attention
                        x1,y1,x2,y2 = int(att_x1), int(att_y1), int(att_x2), int(att_y2)
                        gt_x = int((x1 + x2)/2)
                        gt_y = int((y1 + y2)/2)
                        cv2.rectangle(self.raw_image_640, (x1,y1), (x2, y2), (0, 255, 0), thickness = 10)
                        cv2.line(self.raw_image_640, (int(head_point[0]),int(head_point[1])), (gt_x,gt_y), (0, 255, 0), thickness=7) 
                        cv2.circle(self.raw_image_640, center=(gt_x, gt_y), radius=3,color=(60, 174, 65),thickness=9,lineType=9)
                    
                    center_x = int(0.5*(best_obj[0][1]+best_obj[0][3]))
                    center_y = int(0.5*(best_obj[0][2]+best_obj[0][4])) 
                    cv2.line(self.raw_image_640, (int(head_point[0]),int(head_point[1])), (center_x,center_y),  (23, 210 ,252), thickness=7) 
                    cv2.circle(self.raw_image_640, center=(center_x, center_y), radius=3,color=(20, 164, 251),thickness=9,lineType=9)
                    [x1, y1, x2, y2] = best_obj[0][1:]
                    cv2.rectangle(self.raw_image_640, (x1, y1), (x2, y2), (23, 210 ,252), thickness = 7)
                    
                    save_name = os.path.join(vis_path,f"{key}.png")
                    cv2.imwrite(save_name,self.raw_image_640)
            
            
            total_correct = 0
            total_total = 0
            
            for key in sorted(task_total_frames.keys()):
                self.f_result = open(self.result_save_name, 'a')
                self.f_result.write('precision = {:.3f}, correct num = {}, total_number = {}, {} \n'.format(float(task_correct_frames[key] / task_total_frames[key]),task_correct_frames[key], task_total_frames[key],key))
                total_total += task_total_frames[key]
                total_correct += task_correct_frames[key]
                
            self.f_result.write('total_correct: {} total_total: {} overall precision:{:.3f}\n'.format(total_correct, total_total, float(total_correct / total_total)))
            self.f_result.write('avg_dist: {:.3f}  avg_angle:{:.3f}\n'.format(float(sum(dist_list) / total_total),float(sum(angle_list) / total_total)))
            self.f_result.close()

            " best epoch and its accuracy"
            acc = float(total_correct / total_total)
            acc_list.append(acc)
            angle_avg_list.append(float(sum(angle_list) / total_total))
            dist_avg_list.append(float(sum(dist_list) / total_total))
            if acc>best_acc:
                best_acc = acc
                best_iepoch = i_epoch
                
        print('best accuracy = {:.3f} \nbest epoch = {}\n'.format( best_acc, best_iepoch))
        print('best_angle = {}\nbest_dist = {}\n'.format(angle_avg_list[0], dist_avg_list[0]))
        # best print 
        mean_acc = statistics.mean(acc_list)
        mean_acc = 0
        self.f_result = open(self.result_save_name, 'a')
        self.f_result.write('\naverage accuracy =  {:.3f} \nbest accuracy = {:.3f} \nbest epoch = {}\n'.format(mean_acc, best_acc, best_iepoch))
        self.f_result.close()
        
   
    def file_init(self,):
            
        self.f_statistics = None
        self.no_detection_num = 0
        self.error_by_detection_num = 0
        self.no_head_num = 0
        self.failture_statistics = None
        
        " loss file "
        self.plot_loss_init()
        self.loss_save_name = os.path.join(self.exp_path,'test_loss.pdf')
    

        " result file, statistics file, using cross metric"
        if self.is_test_using_cross_criterion:
            if self.is_test_using_object_detection:
                self.result_save_name = os.path.join(self.exp_path, 'result_CrossMetric_Object_Detection.txt')
            
            else:
                self.result_save_name = os.path.join(self.exp_path, 'result_CrossMetric_Object_Annotation.txt')
                self.experiment_statistics_name = os.path.join(self.exp_path, 'statistics_CrossMetric_Object_Annotation.txt')
                self.failture_statistics_name = os.path.join(self.exp_path, 'failture_statistics_CrossMetric_Object_Annotation_0804.txt')
            
        " for continue experiment"
        self.f_result = open(self.result_save_name, 'a')
        self.f_result.write('******************  new experiments *******************,   batch size =  {}\n'.format(self._opt.batch_size))
        self.f_result.close()
            
    def count_checkpoint_num(self):
      
        all_networks = os.listdir(os.path.join(self._save_path))
        max_number = 0
        for each_file in all_networks:
            if each_file.startswith('net'):
                idx = int(each_file.strip().split('_')[2])
                if idx > max_number:
                    max_number = idx
        return max_number
    
    def gaze_direction_box_cross_metric(self, pattern, frame_name, best_obj, is_exist_head, head_point):
        is_correct = False
        # load object detection
        error_by_object_detetcion_flag = False
        error_by_head_detection_flag = False
        
        # best object center
        best_obj_center_point = [int(0.5 * (best_obj[0][1] + best_obj[0][3]) ),int(0.5 * (best_obj[0][2] + best_obj[0][4]) )]
       
        # load attentions，gt attention
        attentions = self.load_attentions(pattern, frame_name)  
        att_num = len(attentions)
        assert att_num == 1 or att_num == 2
        
        if is_exist_head:
            self.point_on_bottom_edge = self.two_point_to_downedge_point(best_obj_center_point, head_point, self.H, self.W)
            gaze_line = [self.point_on_bottom_edge, head_point]
            for attention in attentions:
                four_lines = self.four_lines_of_box(attention[1:])
                for each_line in four_lines:
                    is_cross = self.IsIntersec(gaze_line[0], gaze_line[1],
                                                each_line[0], each_line[1])
                    if is_cross:
                        is_correct = True
                        break
                
                if is_correct == True:
                    break
        
        return is_correct
    
    def load_attentions(self, pattern, frame_name):
        if pattern == 'CAD':
            path = self.data_path + 'attention_annotation_name'
            [video_base_name, frame_index] = frame_name.split("-")
            name = os.path.join(path, video_base_name,
                                'attention-{}.json'.format(frame_index))
            with open(name, 'r') as file:
                attentions = json.load(file)
            return attentions
        
    def find_head_location(self, frame_name):
        is_exist_head = 1
        X = 0
        Y = 0
        head_list = [0, 14, 15, 16, 17]
        if self.data_name.startswith('CAD'):
            [cate, frame_index] = frame_name.split('-')
            skeleton_path = self.data_path + 'skeleton_openpose_json'
            openpose_file = os.path.join(skeleton_path, cate,
                                         '{}_{:012d}_keypoints.json'
                                         .format(cate, int(frame_index)))

        # heat location
        head_points = []
        with open(openpose_file, 'r') as output:
            skeletons = json.load(output)  # it is a list
            persons = skeletons['people']  # person is a list
        num = len(persons)
        if num == 0:
            is_exist_head = 0
        else:
            person = persons[0]
            body = np.reshape(person['pose_keypoints_2d'],
                              (18, 3))  # body is a array
            for i in head_list:
                x = body[i, 0]
                y = body[i, 1]
                if x != 0 and y != 0:
                    head_points.append((x, y))

            head_points_num = len(head_points)
            if head_points_num == 0:
                is_exist_head = 0
            else:
                for (x, y) in head_points:
                    X += x
                    Y += y
                X = int(float(X / head_points_num))
                Y = int(float(Y / head_points_num))
        point = [X, Y]

        return is_exist_head, point
        
    @staticmethod
    def two_point_to_downedge_point(p1, p2, h, w):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]

        deta_y = abs(y2 - y1)
        if deta_y < 5:
            return_p = (w, min(y1, y2))
        else:
            if y1 > y2:
                p_upper = [x2, y2]
                p_down = [x1, y1]
            else:
                p_upper = [x1, y1]
                p_down = [x2, y2]

            X = p_down[0] - (p_upper[0] - p_down[0]) * (p_down[1] - h) / (
                    p_upper[1] - p_down[1])
            X = int(X)
            return_p = (X, h)

        return return_p
    
    @staticmethod
    def four_lines_of_box(attention):
        x1, y1, x2, y2 = int(attention[0]), int(attention[1]), int(
            attention[2]), int(attention[3])
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

    def IsIntersec(self, p1, p2, p3, p4):

        if (max(p1[0], p2[0]) >= min(p3[0], p4[0])
                and max(p3[0], p4[0]) >= min(p1[0], p2[0])
                and max(p1[1], p2[1]) >= min(p3[1], p4[1])
                and max(p3[1], p4[1]) >= min(p1[1], p2[1])):

            if (self.cross(p1, p2, p3) * self.cross(p1, p2, p4) <= 0
                    and self.cross(p3, p4, p1) * self.cross(p3, p4, p2) <= 0):
                D = 1
            else:
                D = 0
        else:
            D = 0
        return D
    
    @staticmethod
    def cross(p1, p2, p3):
        x1 = p2[0] - p1[0]
        y1 = p2[1] - p1[1]
        x2 = p3[0] - p1[0]
        y2 = p3[1] - p1[1]
        return x1 * y2 - x2 * y1

    def angle_dist_metric(self, pattern, frame_name, best_obj, is_exist_head, head_point):       
        is_exist_metric = False
        angle_error = 10000
        dist_error = 10000
    
        # load attentions，gt attention  
        attentions = self.load_attentions(pattern, frame_name)
        att_num = len(attentions)
        assert att_num == 1 or att_num == 2
        
        if is_exist_head:
            is_exist_metric = True
       
        if is_exist_metric:
            for att in attentions:
                [x1,y1,x2,y2] = [att[1],att[2],att[3],att[4]]
                [att_x_c, att_y_c] = [0.5*(x1+x2), 0.5*(y1+y2)]
                gaze_point = [int(0.5 * (best_obj[0][1] + best_obj[0][3]) ),int(0.5 * (best_obj[0][2] + best_obj[0][4]) )]
                gt_angle = [att_x_c - head_point[0], att_y_c - head_point[1]]
                pre_angle = [gaze_point[0] - head_point[0], gaze_point[1] - head_point[1]]
                angle = self.angle_compute(gt_angle,pre_angle)
                if angle < angle_error:
                    angle_error = angle
                
                gt_position = [att_x_c/640, att_y_c/480]
                gaze_point[0] /= 640
                gaze_point[1] /= 480
                dist = self.L2_dist(gt_position,gaze_point)
                if dist < dist_error:
                    dist_error = dist
        
        return is_exist_metric,dist_error, angle_error


    def L2_dist(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


    def angle_compute(self,v1,v2):
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
            angle_value = 0
        else:
            # 求得夹角（弧度制）：
            angle_radian = np.arccos(cos_theta)

            # 转换为角度值：
            angle_value = angle_radian * 180 / np.pi
        return angle_value


    def plot_loss_init(self):
        plt.ion()
        fig = plt.figure()

        self.ax1 = fig.add_subplot(211)
        # self.ax2 = fig.add_subplot(212)
        # self.ax1 = fig.add_subplot(411)
        # self.ax2 = fig.add_subplot(412)
        # self.ax3 = fig.add_subplot(413)
        # self.ax4 = fig.add_subplot(414)
        # plt.subplots_adjust(hspace=0.7, wspace=0)

        # title
        # self.ax1.set_title('Global Loss')
        # self.ax2.set_title('Local Loss')
        # self.ax3.set_title('Task Prediction Loss')
        # self.ax4.set_title('Task decoding Loss')

        self.ax1.set_title('Global Loss')
        # self.ax2.set_title('Local Loss')
        # self.ax3.set_title('Task Loss')
        # self.ax4.set_title('Vae Loss')
        
    def plot_loss(self, loss_dict, save_flag):
        for k, v in loss_dict.items():
            if k == 'total_loss':
                self.global_loss.append(v)
        if len(self.global_loss) > 0:
            self.ax1.plot(range(1, len(self.global_loss) + 1), self.global_loss)
      
        if save_flag:
            plt.savefig(self.loss_save_name, dpi=30)
    
    def plot_loss_seg(self, ):
        
        if len(self.global_loss) > 0:
            self.ax1.axvline( len(self.global_loss))
        
        save_flag = 1
        if save_flag:
            plt.savefig(self.loss_save_name, dpi=30)


def main():
    xx = Test()


if __name__ == '__main__':
    main()       