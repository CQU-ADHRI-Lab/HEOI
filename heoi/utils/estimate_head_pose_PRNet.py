###########################################################
### author: Zhixiong Nan
### date: 05/09 2018
##############################################################


"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
from multiprocessing import Process, Queue
import numpy as np
import cv2, os, sys,json
from utils.util import IOU_NMS_Objs_by_Att

class PRNet_baseline():
    def __init__(self):
        # TIA
        self.test_image_txt_path = '/home/nan/dataset/TIA/train_and_test/test_list.txt'
        self.raw_image_path = '/home/nan/dataset/TIA_misc/raw_image'
        self.head_position_path = '/home/nan/dataset/TIA_misc/TIA_where_are_they'
        self.attention_ann_path = '/home/nan/dataset/TIA/annotation/attention_box_gt'
        self.save_name_path = '/home/nan/dataset/TIA/PRNet_result'

        # VR
        # self.test_image_txt_path = '/home/nan/dataset/VR/test_list.txt'
        # self.raw_image_path = '/home/nan/dataset/VR_misc/raw_image'
        # self.head_position_path = '/home/nan/dataset/VR_misc/VR_where_are_they_new'
        # self.attention_ann_path = '/home/nan/dataset/VR/attention_annotation'
        # self.attention_ann_path = '/home/nan/dataset/VR/attention_annotation'

        # CAD
        # self.test_image_txt_path = '/home/nan/dataset/CAD120/CAD120/test_list.txt'
        # self.raw_image_path = '/home/nan/dataset/CAD120/CAD120/raw_image'
        # self.head_position_path = '/home/nan/dataset/CAD120/CAD120/cad_where_are_they_02'
        # self.attention_ann_path = '/home/nan/dataset/CAD120/CAD120/attention_annotation'
        # self.save_name_path = '/home/nan/dataset/CAD120/CAD120/PRNet_result'
        # self.attention_ann_path = '/home/nan/dataset/CAD120/CAD120/attention_annotation'
    def draw_prn(self, image,head_8_points,attentions,H,W):
        p1 = (int(head_8_points[1]), int(head_8_points[2]))
        p2 = (int(head_8_points[3]), int(head_8_points[4]))
        p3 = (int(head_8_points[5]), int(head_8_points[6]))
        p4 = (int(head_8_points[7]), int(head_8_points[8]))
        p5 = (int(head_8_points[9]), int(head_8_points[10]))
        p6 = (int(head_8_points[11]), int(head_8_points[12]))
        p7 = (int(head_8_points[13]), int(head_8_points[14]))
        p8 = (int(head_8_points[15]), int(head_8_points[16]))
        p_Rear_center = self.center_of_points([p1, p2, p3, p4])
        p_Front_center = self.center_of_points([p5, p6, p7, p8])
        p_center_front_edge = self.two_point_to_downedge_point(p_Rear_center, p_Front_center, H, W)
        p_FrontDownCenter = self.center_of_points([p6, p7])
        p_FrontDownCenter_edge = self.two_point_to_downedge_point(p_Rear_center, p_FrontDownCenter, H, W)
        p_FrontDownAndFrontCenter = self.center_of_points([p_Front_center, p_FrontDownCenter])
        p_FrontDownAndFrontCenter_edge = self.two_point_to_downedge_point(p_Rear_center, p_FrontDownAndFrontCenter, H, W)

        for attention in attentions:
            x1, y1, x2, y2 = int(attention[0]), int(attention[1]), int(attention[2]), int(attention[3])
            cv2.rectangle(image, (x1, y1), (x2, y2), [0, 0, 255], 5)

        color = [0, 255, 0]
        line_width = 3
        cv2.line(image, p1, p2, color, line_width, cv2.LINE_AA)
        cv2.line(image, p2, p3, color, line_width, cv2.LINE_AA)
        cv2.line(image, p3, p4, color, line_width, cv2.LINE_AA)
        cv2.line(image, p1, p4, color, line_width, cv2.LINE_AA)
        cv2.line(image, p5, p6, color, line_width, cv2.LINE_AA)
        cv2.line(image, p6, p7, color, line_width, cv2.LINE_AA)
        cv2.line(image, p7, p8, color, line_width, cv2.LINE_AA)
        cv2.line(image, p5, p8, color, line_width, cv2.LINE_AA)
        cv2.line(image, p4, p5, color, line_width, cv2.LINE_AA)
        cv2.line(image, p3, p8, color, line_width, cv2.LINE_AA)
        cv2.line(image, p2, p7, color, line_width, cv2.LINE_AA)
        cv2.line(image, p1, p6, color, line_width, cv2.LINE_AA)
        cv2.line(image, p_Rear_center, p_FrontDownAndFrontCenter_edge, color, line_width, cv2.LINE_AA)

        Thickness = 3
        radis = 3
        cv2.circle(image, p1, Thickness, (255, 255, 255), radis)
        cv2.circle(image, p2, Thickness, (255, 255, 255), radis)
        cv2.circle(image, p3, Thickness, (255, 255, 255), radis)
        cv2.circle(image, p4, Thickness, (255, 255, 255), radis)
        cv2.circle(image, p5, Thickness, (0, 0, 0), radis)
        cv2.circle(image, p6, Thickness, (0, 0, 0), radis)
        cv2.circle(image, p7, Thickness, (0, 0, 0), radis)
        cv2.circle(image, p8, Thickness, (0, 0, 0), radis)

    def evaluate(self):


        # TIA
        self.root_path = '/home/nan/dataset/TIA/PRNet_result/'
        self.head_box_name = self.root_path + 'head_box_8_points_35.txt'
        self.face_box_name = self.root_path + 'face_35.txt'
        self.visual_save_path = self.root_path + 'head_attention_visual'
        self.attention_ann_path = '/home/nan/dataset/TIA/annotation/attention_box_gt'
        self.objects_ann_path = '/home/nan/dataset/TIA/annotation/rcnn_and_org_objects'


        # VR
        # self.root_path = '/home/nan/dataset/VR/PRNet_result/'
        # self.head_box_name = self.root_path + 'head_box_8_points.txt'
        # self.face_box_name = self.root_path + 'face.txt'
        # self.visual_save_path = self.root_path + 'head_attention_visual'
        # self.attention_ann_path = '/home/nan/dataset/VR/attention_annotation'
        # self.objects_ann_path = '/home/nan/dataset/VR/object_annotation'

        # CAD
        # self.root_path = '/home/nan/dataset/CAD120/CAD120/PRNet_result'
        # self.head_box_name = self.root_path + '/head_box_8_points.txt'
        # self.face_box_name = self.root_path + '/face.txt'
        # self.visual_save_path = self.root_path + '/head_attention_visual'
        # self.attention_ann_path = '/home/nan/dataset/CAD120/CAD120/attention_annotation'


        right_name = self.root_path+'/right35.txt'
        f_right = open(right_name,'w')
        wrong_name = self.root_path + '/wrong35.txt'
        f_wrong = open(wrong_name, 'w')
        with open(self.face_box_name, 'r') as file:
            face_lines = file.readlines()
        with open(self.head_box_name, 'r') as file:
            head_lines = file.readlines()
        right_num = 0
        total_num = len(head_lines)
        for index, line in enumerate(head_lines):
            self.KEY = line
            key = line.strip().split(' ')[0]
            head_line = line
            head_line = head_line.strip().split(' ')
            face_line = face_lines[index]
            face_line = face_line.strip().split(' ')
            head_8_points = head_line
            # print(head_8_points)
            face_2_points = face_line[1:]
            # print(face_2_points)
            face_2_points = [int(face_2_points[0]),int(face_2_points[1]),int(face_2_points[2]),int(face_2_points[3])]
            # print(face_2_points)

            # TIA
            [cate, video_base_name, frame_index] = key.split('-')
            frame_index = int(frame_index)
            raw_images = sorted(os.listdir(os.path.join(self.raw_image_path, cate, video_base_name)))
            image_name = os.path.join(self.raw_image_path, cate, video_base_name, raw_images[frame_index])
            image = cv2.imread(image_name)
            H = image.shape[0]
            W = image.shape[1]
            attention_ann_name = os.path.join(self.attention_ann_path, cate, '{}_AttentionBox.txt'.format(video_base_name))
            with open(attention_ann_name, 'r') as att_obj_gt_txt:
                att_lines = att_obj_gt_txt.readlines()
            att_line = att_lines[frame_index]  # line is like "1 dustpan 450 719 632 841 broom 576 705 664 813"
            # print(att_line)
            att_line = att_line.strip().split(' ')
            att_line = att_line[1:]
            num_att = int(len(att_line) / 5)
            att_list = []
            for i in range(num_att):
                b_lt = (int(att_line[i * 5 + 1]), int(att_line[i * 5 + 2]))  # (X1,Y1)
                b_rb = (int(att_line[i * 5 + 3]), int(att_line[i * 5 + 4]))
                att_list.append([b_lt[0],b_lt[1],b_rb[0],b_rb[1]])
            json_name = '{}-allobjects.json'.format(video_base_name)
            objs_load_name = os.path.join(self.objects_ann_path, cate, json_name)
            with open(objs_load_name, 'r') as output:
                objects_video = json.load(output) # dict
            objects_list = objects_video[str(frame_index)]
            filtered_objects = IOU_NMS_Objs_by_Att(objects_list, att_list, 0.9)
            for each_obj in filtered_objects:
                x1, y1, x2, y2 = int(each_obj[0]), int(each_obj[1]), int(each_obj[2]), int(each_obj[3])
                cv2.rectangle(image, (x1,y1), (x2,y2), [255, 0, 0], 2)

            if not os.path.exists(self.visual_save_path):
                os.mkdir(self.visual_save_path)
            if not os.path.exists(os.path.join(self.visual_save_path, cate)):
                os.mkdir(os.path.join(self.visual_save_path, cate))
            if not os.path.exists(os.path.join(self.visual_save_path, cate, video_base_name)):
                os.mkdir(os.path.join(self.visual_save_path, cate, video_base_name))
            save_name = os.path.join(self.visual_save_path, cate, video_base_name, "{:04d}.png".format(frame_index))

            # VR
            # [cate, video_base_name, frame_index] = key.split('/')
            # frame_index = int(frame_index)
            # image_name = os.path.join(self.raw_image_path, cate, video_base_name, "sequence_{:04d}.png".format(frame_index))
            # image = cv2.imread(image_name)
            # H = image.shape[0]
            # W = image.shape[1]
            #     # objects
            # json_name = 'objects-{}-{}-{:04d}.json'.format(cate,video_base_name,frame_index)
            # objs_load_name = os.path.join(self.objects_ann_path, cate, video_base_name, json_name)
            # with open(objs_load_name, 'r') as output:
            #     objects = json.load(output)  # list of [x1, y1, x2, y2] ,int
            # for object in objects:
            #     x1, y1, x2, y2 = int(object[0]), int(object[1]), int(object[2]), int(object[3])
            #     cv2.rectangle(image, (x1,y1), (x2,y2), [255, 0, 0], 2)

            # CAD
            # print(key)
            # [video_base_name, frame_index] = key.split('-') # CAD, Subject1_rgbd_images_arranging_objects_0510175411-0000
            # frame_index = int(frame_index)
            # image_name = os.path.join(self.raw_image_path, video_base_name, "{:04d}.png".format(frame_index))
            # image = cv2.imread(image_name)
            # H = image.shape[0]
            # W = image.shape[1]



            # parse head box 8 points
            p1 = (int(head_8_points[1]), int(head_8_points[2]))
            p2 = (int(head_8_points[3]), int(head_8_points[4]))
            p3 = (int(head_8_points[5]), int(head_8_points[6]))
            p4 = (int(head_8_points[7]), int(head_8_points[8]))
            p5 = (int(head_8_points[9]), int(head_8_points[10]))
            p6 = (int(head_8_points[11]), int(head_8_points[12]))
            p7 = (int(head_8_points[13]), int(head_8_points[14]))
            p8 = (int(head_8_points[15]), int(head_8_points[16]))
            p_Rear_center = self.center_of_points([p1,p2,p3,p4])
            p_Front_center = self.center_of_points([p5,p6,p7,p8])
            p_center_front_edge = self.two_point_to_downedge_point(p_Rear_center,p_Front_center,H,W)
            p_FrontDownCenter = self.center_of_points([p6,p7])
            p_FrontDownCenter_edge = self.two_point_to_downedge_point(p_Rear_center,p_FrontDownCenter,H,W)
            p_FrontDownAndFrontCenter = self.center_of_points([p_Front_center,p_FrontDownCenter])
            p_FrontDownAndFrontCenter_edge = self.two_point_to_downedge_point(p_Rear_center,p_FrontDownAndFrontCenter,H,W)

            is_right = 0
            gaze_line_p1 = p_Rear_center
            gaze_line_p2 = p_FrontDownAndFrontCenter_edge

            # right or wrong CAD
            # attention_ann_name = os.path.join(self.attention_ann_path, video_base_name,
            #                                   '{}-AttentionBox-{:04d}.txt'.format(video_base_name, frame_index))
            # with open(attention_ann_name, 'r') as att_obj_gt_txt:
            #     att_line = att_obj_gt_txt.readline()
            #     att_line = att_line.strip().split(' ')
            # num_att = int(len(att_line) / 4)
            #
            # for i in range(num_att):
                # b_lt = (int(att_line[i * 4]), int(att_line[i * 4 + 1]))  # (X1,Y1)
                # b_rb = (int(att_line[i * 4 + 2]), int(att_line[i * 4 + 3]))
                # cv2.rectangle(image, b_lt, b_rb, [0, 0, 255], 2)
                # lines = []
                # p11 = (b_lt[0],b_lt[1])
                # p22 = (b_lt[0],b_rb[1])
                # lines.append((p11,p22))
                # p11 = (b_lt[0], b_lt[1])
                # p22 = (b_rb[0], b_lt[1])
                # lines.append((p11, p22))
                # p11 = (b_rb[0], b_lt[1])
                # p22 = (b_rb[0], b_rb[1])
                # lines.append((p11, p22))
                # p11 = (b_lt[0], b_rb[1])
                # p22 = (b_rb[0], b_rb[1])
                # lines.append((p11, p22))
                # for line in lines:
                #     p111 = line[0]
                #     p222 = line[1]
                #     is_cross = self.IsIntersec(p111, p222, gaze_line_p1, gaze_line_p2)
                #     if is_cross == 1:
                #         right_num += 1
                #         is_right = 1
                #         f_right.write(self.KEY)
                #         print(self.KEY)
                #         break
                # if is_right:
                #     break

            # right or wrong VR
            # json_name = "attention-{}-{}-{:04d}.json".format(cate, video_base_name, frame_index)
            # att_load_name = os.path.join(self.attention_ann_path, cate, video_base_name, json_name)
            # with open(att_load_name, 'r') as output:
            #     attentions = json.load(output)  # list of [x1, y1, x2, y2] ,int
            # for attention in attentions:
            #     x1, y1, x2, y2 = int(attention[0]), int(attention[1]), int(attention[2]), int(attention[3])
            #     cv2.rectangle(image, (x1,y1), (x2,y2), [0, 0, 255], 2)
            #     lines = []
            #     p11 = (x1,y1)
            #     p22 = (x1,y2)
            #     lines.append((p11,p22)) # left
            #     p11 = (x1, y1)
            #     p22 = (x2, y1)
            #     lines.append((p11, p22)) # top
            #     p11 = (x2, y2)
            #     p22 = (x2, y1)
            #     lines.append((p11, p22)) # right
            #     p11 = (x1, y2)
            #     p22 = (x2, y2)
            #     lines.append((p11, p22)) # down
            #     for line in lines:
            #         p111 = line[0]
            #         p222 = line[1]
            #         is_cross = self.IsIntersec(p111, p222, gaze_line_p1, gaze_line_p2)
            #         if is_cross == 1:
            #             right_num += 1
            #             is_right = 1
            #             f_right.write(self.KEY)
            #             print(self.KEY)
            #             break
            #     if is_right:
            #         break

            # right and wrong, TIA
            for attention in att_list:
                x1, y1, x2, y2 = int(attention[0]), int(attention[1]), int(attention[2]), int(attention[3])
                cv2.rectangle(image, (x1, y1), (x2, y2), [0, 0, 255], 2)
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
                for line in lines:
                    p111 = line[0]
                    p222 = line[1]
                    is_cross = self.IsIntersec(p111, p222, gaze_line_p1, gaze_line_p2)
                    if is_cross == 1:
                        right_num += 1
                        is_right = 1
                        f_right.write(self.KEY)
                        print(self.KEY)
                        break
                if is_right:
                    break


            if is_right==0:
                f_wrong.write(self.KEY)
            # draw
            color = [0,255,0]
            red = [0,0,255]
            line_width =2
            cv2.line(image, p1, p2, color, line_width, cv2.LINE_AA)
            cv2.line(image, p2, p3, color, line_width, cv2.LINE_AA)
            cv2.line(image, p3, p4, color, line_width, cv2.LINE_AA)
            cv2.line(image, p1, p4, color, line_width, cv2.LINE_AA)
            cv2.line(image, p5, p6, color, line_width, cv2.LINE_AA)
            cv2.line(image, p6, p7, color, line_width, cv2.LINE_AA)
            cv2.line(image, p7, p8, color, line_width, cv2.LINE_AA)
            cv2.line(image, p5, p8, color, line_width, cv2.LINE_AA)
            cv2.line(image, p4, p5, color, line_width, cv2.LINE_AA)
            cv2.line(image, p3, p8, color, line_width, cv2.LINE_AA)
            cv2.line(image, p2, p7, color, line_width, cv2.LINE_AA)
            cv2.line(image, p1, p6, color, line_width, cv2.LINE_AA)
            # cv2.line(image, p_Rear_center, p_FrontDownCenter_edge, color, line_width, cv2.LINE_AA)
            cv2.line(image, p_Rear_center, p_FrontDownAndFrontCenter_edge, color, line_width, cv2.LINE_AA)
            # cv2.line(image, p_Rear_center, p_center_front_edge, red, line_width, cv2.LINE_AA)
            # cv2.line(image, p_Rear_center, p_center_front_edge, color, line_width, cv2.LINE_AA)
            # cv2.rectangle(image,(face_2_points[0],face_2_points[1]),(face_2_points[2],face_2_points[3]),[0,0,255],2)


            Thickness = 2
            radis = 2
            cv2.circle(image, p1, Thickness, (255, 255, 255), radis)
            cv2.circle(image, p2, Thickness, (255, 255, 255), radis)
            cv2.circle(image, p3, Thickness, (255, 255, 255), radis)
            cv2.circle(image, p4, Thickness, (255, 255, 255), radis)
            cv2.circle(image, p5, Thickness, (0, 0, 0), radis)
            cv2.circle(image, p6, Thickness, (0, 0, 0), radis)
            cv2.circle(image, p7, Thickness, (0, 0, 0), radis)
            cv2.circle(image, p8, Thickness, (0, 0, 0), radis)

            cv2.imshow('gaze_object_pose', image)
            if cv2.waitKey(33) == 27:
                break

            dim = (960, 540)
            resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(save_name,resized_image)

        print('right_num', right_num)
        f_right.write('total num = {}'.format(total_num))
        f_right.write('right num = {}'.format(right_num))
        f_right.write('accuracy = {}'.format(right_num/total_num))
        f_right.close()
        f_wrong.close()
    def center_of_points(self, points):
        X = 0
        Y = 0
        for point in points:
            x = point[0]
            y = point[1]
            X+=x
            Y+=y
        num = len(points)
        X/=num
        Y/=num
        X = int(X)
        Y = int(Y)
        return (X,Y)

    def two_point_to_downedge_point(self, p1, p2, h,w):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]

        deta_y = abs(y2-y1)
        if deta_y<5:
            return_p = (w, min(y1,y2))
        else:
            if y1>y2:
                p_upper = [x2,y2]
                p_down = [x1,y1]
            else:
                p_upper = [x1,y1]
                p_down = [x2,y2]

            X = p_down[0]-(p_upper[0]-p_down[0])*(p_down[1]-h)/(p_upper[1]-p_down[1])
            X = int(X)
            return_p = (X,h)

        return return_p
    def distance_point_to_line(self, point1,point2, P):
        p1 = np.array([point1[0], point1[1]])
        p2 = np.array([point2[0], point2[1]])
        p3 = np.array([P[0],P[1]])
        d = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)
        d = int(d)
        return d

    def distance_betweent_two_point(self, p1, p2):
        x_deta = abs(p1[0]-p2[0])
        y_deta = abs(p1[1]-p2[1])
        d = pow(x_deta,2) + pow(y_deta,2)
        d= pow(d,0.5)
        d=int(d)
        return d

    def islineintersect(self,line1, line2):

        is_corss = 1
        i1 = [min(line1[0][0], line1[1][0]), max(line1[0][0], line1[1][0])]
        i2 = [min(line2[0][0], line2[1][0]), max(line2[0][0], line2[1][0])]
        ia = [max(i1[0], i2[0]), min(i1[1], i2[1])]
        if max(line1[0][0], line1[1][0]) < min(line2[0][0], line2[1][0]):
            is_corss = 0
            # return False
        # if
        m1 = (line1[1][1] - line1[0][1]) * 1. / max((line1[1][0] - line1[0][0]),1) * 1.
        m2 = (line2[1][1] - line2[0][1]) * 1. / max((line2[1][0] - line2[0][0]),1) * 1.
        if m1 == m2:
            is_corss = 0
            # return False
        b1 = line1[0][1] - m1 * line1[0][0]
        b2 = line2[0][1] - m2 * line2[0][0]
        x1 = (b2 - b1) / (m1 - m2)
        if (x1 < max(i1[0], i2[0])) or (x1 > min(i1[1], i2[1])):
            is_corss = 0
            # return False
        # return True
        return is_corss

    def cross(self,p1, p2, p3):
        x1 = p2[0] - p1[0]
        y1 = p2[1] - p1[1]
        x2 = p3[0] - p1[0]
        y2 = p3[1] - p1[1]
        return x1 * y2 - x2 * y1

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


def main():
    xx = PRNet_baseline()
    xx.evaluate()





if __name__ == '__main__':
    main()
