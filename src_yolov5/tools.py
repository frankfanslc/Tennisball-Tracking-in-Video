#! /home/drcl_yang/anaconda3/envs/py36/bin/python

from pathlib import Path
import sys
import os

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])
sys.path.insert(0, './yolov5')
sys.path.insert(0, './yolov5')


import numpy as np
import time
import cv2

import torch
import torch.backends.cudnn as cudnn

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

from yolov5.utils.augmentations import letterbox

from kalman_utils.KFilter import *


# ball_tracking setup
fgbg = cv2.createBackgroundSubtractorMOG2(1, 10, False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

kernel_dilation_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))

kernel_erosion_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))


class Ball_Pos_Estimation():

    def __init__(self):
        self.pre_ball_cen_left_list = []
        self.pre_ball_cen_right_list = []

        self.ball_pos_list = [[np.nan, np.nan, np.nan]]

        self.ball_detect_flag = 0

        self.ball_cand_pos = []
        self.ball_trajectory = []

        self.kf = np.nan
        self.kf_flag = False


        self.disappear_cnt = 0
        
        self.dT = 1 / 25

    def check_ball_flying(self, ball_cen_left_list, ball_cen_right_list):

        self.ball_cen_left_list = ball_cen_left_list
        self.ball_cen_right_list = ball_cen_right_list

        left_flag = False
        right_flag = False

        if len(self.pre_ball_cen_left_list) and len(self.pre_ball_cen_right_list):
                    
            for i in range(len(self.ball_cen_left_list)):

                if left_flag: break   

                x_cen = self.ball_cen_left_list[i][0]
                
                for j in range(len(self.pre_ball_cen_left_list)):

                    pre_x_cen = self.pre_ball_cen_left_list[j][0]
                    
                    if x_cen > pre_x_cen:

                        self.pre_ball_cen_left_list = self.ball_cen_left_list
                        left_flag = True
                        break

            for i in range(len(self.ball_cen_right_list)):

                if right_flag: break   

                x_cen = self.ball_cen_right_list[i][0]
                
                for j in range(len(self.pre_ball_cen_right_list)):

                    pre_x_cen = self.pre_ball_cen_right_list[j][0]

                    if x_cen < pre_x_cen:

                        self.pre_ball_cen_right_list = self.ball_cen_right_list
                        right_flag = True
                        
                        break

            if (left_flag == False or right_flag == False): 
                #self.reset_ball()
                return False

            return 1

        else:
            self.pre_ball_cen_left_list = self.ball_cen_left_list
            self.pre_ball_cen_right_list = self.ball_cen_right_list

            return 3

    def reset_ball(self):
        self.pre_ball_cen_left_list = []
        self.pre_ball_cen_right_list = []

        self.kf_flag = False
        self.kf = np.nan

        self.ball_trajectory = []

    def get_ball_pos(self):
        
        ball_pos = []

        cx = 360
        cy = 204
        focal_length = 320.754

        net_length = 13.11

        post_hegith_left = 1.13
        post_hegith_right = 1.12 

        post_hegith_avg = (post_hegith_left + post_hegith_right) / 2

        L_pos = self.ball_cen_left_list
        R_pos =  self.ball_cen_right_list

        for i in range(len(L_pos)):
            x_L, y_L = L_pos[i][0] - cx, L_pos[i][1] - cy

            for j in range(len(R_pos)):
                x_R, y_R = R_pos[j][0] - cx, R_pos[j][1] - cy


                c_L = np.sqrt(focal_length ** 2 + x_L ** 2 + y_L ** 2)
                a_L = np.sqrt(focal_length ** 2 + x_L ** 2)

                if x_L < 0:
                    th_L = 0.785398 + np.arccos(focal_length / a_L)

                else :
                    th_L = 0.785398 - np.arccos(focal_length / a_L)


                b_L = a_L * np.cos(th_L)

                c_R = np.sqrt(focal_length ** 2 + x_R ** 2 + y_R ** 2)
                a_R = np.sqrt(focal_length ** 2 + x_R ** 2)

                if x_R > 0:
                    th_R = 0.785398 + np.arccos(focal_length / a_R)

                else :
                    th_R = 0.785398 - np.arccos(focal_length / a_R)

                b_R = a_R * np.cos(th_R)

                theta_L = np.arccos(b_L/c_L)
                theta_R = np.arccos(b_R/c_R)


                D_L = net_length * np.sin(theta_R) / np.sin(3.14 - (theta_L + theta_R))
                D_R = net_length * np.sin(theta_L) / np.sin(3.14 - (theta_L + theta_R))

                height_L = abs(D_L * np.sin(np.arcsin(y_L/c_L)))
                height_R = abs(D_R * np.sin(np.arcsin(y_R/c_R)))

                #height_L = abs(D_L * np.sin(np.arctan(y_L/a_L)))
                #height_R = abs(D_R * np.sin(np.arctan(y_R/a_R)))

                if y_L < 0:
                    height_L += post_hegith_left

                else:
                    height_L -= post_hegith_left  


                if y_R < 0:
                    height_R += post_hegith_right

                else:
                    height_R -= post_hegith_right  

                ball_height_list = [height_L, height_R]
                ball_distance_list = [D_L, D_R]

                height = sum(ball_height_list) / 2 - post_hegith_avg

                ball2net_length_x_L = ball_distance_list[0] * np.sin(theta_L)
                ball_position_y_L = ball_distance_list[0] * np.cos(theta_L)

                ball_plate_angle_L = np.arcsin(height / ball2net_length_x_L)

                ball_position_x_L = ball2net_length_x_L * np.cos(ball_plate_angle_L)

                ball2net_length_x_R = ball_distance_list[1] * np.sin(theta_R)
                ball_position_y_R = ball_distance_list[1] * np.cos(theta_R)

                ball_plate_angle_R = np.arcsin(height / ball2net_length_x_R)

                ball_position_x_R = ball2net_length_x_R * np.cos(ball_plate_angle_R)

                if theta_L > theta_R:
                    ball_position_y = ball_position_y_L - (net_length / 2)

                else :
                    ball_position_y = (net_length / 2) - ball_position_y_R


                #print(L_pos[i],R_pos[j])
                #print([D_L, D_R, height_L, height_R])
                #print([-ball_position_x_L, ball_position_y, height + post_hegith_avg])



                ball_pos.append([-ball_position_x_L, ball_position_y, height + post_hegith_avg])

        return ball_pos


        
        if self.swing_check:
            
            if self.ball_detect_flag == 0:
                self.ball_detect_flag = 1

                self.ball_cand_pos = self.ball_pos_list

                self.disappear_cnt = 0

                x_pos, y_pos, z_pos = self.ball_pos_list[np.array(self.ball_pos_list)[:,0].argmin()]

                self.kf = Kalman_filiter(x_pos, y_pos, z_pos, self.dT)

                self.kf_flag = 1

            else:
                self.ball_cand_pos = self.ball_pos_list

                self.disappear_cnt = 0


        else:
            self.disappear_cnt += 1

            if self.disappear_cnt > 3: # 공 궤적 초기화

                self.ball_cand_pos.clear()
                self.ball_trajectory.clear()
                self.disappear_cnt = 0
                self.ball_detect_flag = 0

                self.kf = np.nan
                self.kf_flag = False

    def ball_vel_check(self, ball_pos):
        
        if len(self.ball_trajectory) > 3:
            
            ball_trajectory = np.array(self.ball_trajectory).reshape([-1,3])

            x_pos_list = ball_trajectory[-3:,0]
            y_pos_list = ball_trajectory[-3:,1]

            mean_x_vel = np.mean(np.diff(x_pos_list))/ self.dT
            mean_y_vel = np.mean(np.diff(y_pos_list))/ self.dT

            x_vel = ( ball_pos[0] - ball_trajectory[-1][0]) / self.dT
            y_vel = ( ball_pos[1] - ball_trajectory[-1][1]) / self.dT

            # print("mean_vel", mean_vel)
            # print("y_vel", y_vel)


            if x_pos_list[-1] > ball_pos[0]:
                ball_pos[0] = ball_trajectory[-1][0] + mean_x_vel * self.dT

            if abs(abs(mean_y_vel) - abs(y_vel)) > 2:
                ball_pos[1] = ball_trajectory[-1][1] + mean_y_vel * self.dT

        return ball_pos


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def img_preprocessing(img0, imgsz, stride, pt):
    img = letterbox(img0, new_shape = imgsz, stride= stride, auto= pt)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return img, img0

def ball_tracking(image):

        ball_cand_box_left = []
        ball_cand_box_right = []

        image_ori = image.copy()

        blur = cv2.GaussianBlur(image_ori, (3, 3), 0)

        fgmask = fgbg.apply(blur, None, 0.1)

        fgmask_erode = cv2.erode(fgmask, kernel_erosion_1, iterations = 1) #오픈 연산이아니라 침식으로 바꾸자

        fgmask_dila = cv2.dilate(fgmask_erode,kernel_dilation_2,iterations = 1)

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask_dila, connectivity = 8)

        for i in range(len(stats)):
            x, y, w, h, area = stats[i]

            if area > 3000 or area > 500 : # or area < 500 or aspect > 1.2 or aspect < 0.97 : 
                continue
            #cv2.rectangle(image_ori, (x, y), (x + w, y + h), (255,0,0), 3)

            x0, y0, x1, y1 = x, y, x+w, y+h

            if y0 < image_ori.shape[0] / 2 :

                if (x0 + x1) / 2 < 690:

                    ball_cand_box_left.append([x0, y0, x1, y1])

            else :
                if (x0 + x1) / 2 > 30:
                    ball_cand_box_right.append([x0, y0, x1, y1])

        MOG2_img = cv2.hconcat([fgmask,fgmask_erode,fgmask_dila])


        #cv2.imshow("MOG2_img",MOG2_img)

        #cv2.imshow("fgmask",fgmask)
        #cv2.imshow("fgmask_erode",fgmask_erode)
        #cv2.imshow("fgmask_dila",fgmask_dila)

        
        return image_ori, ball_cand_box_left, ball_cand_box_right

def check_iou(person_box, ball_cand_box):
        no_ball_box = []

        if len(person_box) < 1:
            ball_box = ball_cand_box
            return ball_box

        for i in range(len(person_box)):
            
            for j in range(len(ball_cand_box)):
                if iou(person_box[i], ball_cand_box[j]):

                    if ball_cand_box[j] in no_ball_box: #중복되는 no ball box 제거
                        continue

                    no_ball_box.append(ball_cand_box[j])



        for i in no_ball_box:
            del ball_cand_box[ball_cand_box.index(i)]
        
        ball_box = ball_cand_box

        return ball_box

def iou(box_0, box_1):
    b0x_0, b0y_0, b0x_1 ,b0y_1 = box_0
    b1x_0, b1y_0, b1x_1 ,b1y_1 = box_1

    min_x = np.argmin([b0x_0,b1x_0])
    min_y = np.argmin([b0y_0,b1y_0])

    if min_x == 0 and min_y == 0:
        if ((b0x_0 <= b1x_0 <= b0x_1) or (b0x_0 <= b1x_1 <= b0x_1)) and ((b0y_0 <= b1y_0 <= b0y_1) or (b0y_0 <= b1y_1 <= b0y_1)):
            return True
    if min_x == 0 and min_y == 1:
        if ((b0x_0 <= b1x_0 <= b0x_1) or (b0x_0 <= b1x_1 <= b0x_1)) and ((b1y_0 <= b0y_0 <= b1y_1) or (b1y_0 <= b0y_1 <= b1y_1)):
            return True
    if min_x == 1 and min_y == 0:
        if ((b1x_0 <= b0x_0 <= b1x_1) or (b1x_0 <= b0x_1 <= b1x_1)) and ((b0y_0 <= b1y_0 <= b0y_1) or (b0y_0 <= b1y_1 <= b0y_1)):
            return True
    if min_x == 1 and min_y == 1:
        if ((b1x_0 <= b0x_0 <= b1x_1) or (b1x_0 <= b0x_1 <= b1x_1) ) and ((b1y_0 <= b0y_0 <= b1y_1) or (b1y_0 <= b0y_1 <= b1y_1) ):
            return True

    return False


def trans_point(img_ori, point_list):

    new_point = []

    for i in range(len(point_list)):
        x0, y0, x1, y1 = point_list[i]

        if y0 > (img_ori.shape[0] / 2):
            y0 -= (img_ori.shape[0] / 2)
            y1 -= (img_ori.shape[0] / 2)

        x_cen, y_cen = int((x0 + x1)/2), int((y0 +y1)/2)

        new_point.append([x_cen, y_cen])

    return new_point


def draw_point_court(img, camera_predict_point_list):

    tennis_court_img = img


    predict_pix_point_list = []

    x_pred = camera_predict_point_list[0]
    y_pred = camera_predict_point_list[1]

    if np.isnan(x_pred):
        return tennis_court_img

    y_pix_length, x_pix_length = tennis_court_img.shape[0], tennis_court_img.shape[1]

    x_meter2pix = 23.77 / x_pix_length
    y_meter2pix = 10.97 / y_pix_length

    predict_pix_point_list.append(int(np.round((11.885 + x_pred) / x_meter2pix)))
    predict_pix_point_list.append(int(np.round((5.485 - y_pred) / y_meter2pix)))

    predict_pix_point = predict_pix_point_list[0:2]

    cv2.circle(tennis_court_img,predict_pix_point, 4, [0, 255, 0], -1)

    return tennis_court_img



def get_prior_ball_pos(ball_cand_trajectory, pre_ball_pos):

    ball_cand_trajectory_list = ball_cand_trajectory

    distance_list = []


    for i in range(len(ball_cand_trajectory_list)):

        distance = get_distance(ball_cand_trajectory_list[i],pre_ball_pos)

        distance_list.append(distance)

    #print("distance_list = ",distance_list)

    return ball_cand_trajectory[np.array(distance_list).argmin()]

def get_distance(point_1, point_2):

    #print("point_1 : ",point_1)
    #print("point_2 : ",point_2)


    return (np.sqrt((point_2[0]-point_1[0])**2 + (point_2[1]-point_1[1])**2 + (point_2[2]-point_1[2])**2))

