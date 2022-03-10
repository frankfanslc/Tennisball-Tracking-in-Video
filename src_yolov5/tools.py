#! /home/drcl_yang/anaconda3/envs/py36/bin/python

from pathlib import Path
import sys
import os

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])
sys.path.insert(0, './yolov5')


import numpy as np
import time
import cv2

from yolov5.utils.augmentations import letterbox

from kalman_utils.KFilter import *


# ball_tracking setup
fgbg = cv2.createBackgroundSubtractorMOG2(20, 25, False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

kernel_erosion_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
kernel_erosion_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))

kernel_dilation_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
kernel_dilation_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))



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

       
    def ball_vel_check(self, ball_pos):
        
        if len(self.ball_trajectory) > 3:
            
            ball_trajectory = np.array(self.ball_trajectory).reshape([-1,3])

            x_pos_list = ball_trajectory[-3:,0]
            y_pos_list = ball_trajectory[-3:,1]

            mean_x_vel = np.mean(np.diff(x_pos_list))/ self.dT
            mean_y_vel = np.mean(np.diff(y_pos_list))/ self.dT

            x_vel = ( ball_pos[0] - ball_trajectory[-1][0]) / self.dT
            y_vel = ( ball_pos[1] - ball_trajectory[-1][1]) / self.dT

            """print("mean_x_vel", mean_x_vel)
            print("x_vel", x_vel)

            print("mean_y_vel", mean_y_vel)
            print("y_vel", y_vel)"""

            if abs(abs(mean_x_vel) - abs(x_vel)) > 5: # 이전 x축 속도평균 보다 현재 속도가 5이상 더 빠를때
                ball_pos[0] = ball_trajectory[-1][0] + mean_x_vel * self.dT

            if abs(abs(mean_y_vel) - abs(y_vel)) > 1: # 이전 y축 속도평균 보다 현재 속도가 1이상 더 빠를때

                ball_pos[1] = ball_trajectory[-1][1] + mean_y_vel * self.dT

            if x_pos_list[-1] >= ball_pos[0]: #공이 뒤로 움직일때 이전 x 위치에서 평균 속도로 현재 위치 추정

                ball_pos[0] = ball_trajectory[-1][0] + mean_x_vel * self.dT


            

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

def ball_tracking(image, debug = 0):

        ball_cand_box_left = []
        ball_cand_box_right = []

        image_ori = image.copy()

        blur = cv2.GaussianBlur(image_ori, (3, 3), 0)

        fgmask = fgbg.apply(blur, None, 0.3)

        fgmask_erode_1 = cv2.erode(fgmask, kernel_erosion_1, iterations = 1) 
        fgmask_dila_1 = cv2.dilate(fgmask_erode_1, kernel_dilation_2,iterations = 1)
        
        nlabels, labels, stats_after, centroids = cv2.connectedComponentsWithStats(fgmask_dila_1, connectivity = 8)

        if len(stats_after) > 30 :
            return image_ori, ball_cand_box_left, ball_cand_box_right

        for i in range(len(stats_after)):
            x, y, w, h, area = stats_after[i]

            if area > 3000 : #or area > 600 : # or area < 500 or aspect > 1.2 or aspect < 0.97 : 
               continue
            cv2.rectangle(image_ori, (x, y), (x + w, y + h), (255,0,0), 3)
            
            x0, y0, x1, y1 = x, y, x+w, y+h

            if y0 < image_ori.shape[0] / 2 :

                if (x0 + x1) / 2 < image_ori.shape[1] - 30:

                    ball_cand_box_left.append([x0, y0, x1, y1])

            else :
                if   50 < (x0 + x1) / 2 < image_ori.shape[1] - (image_ori.shape[1] / 10) :
                    ball_cand_box_right.append([x0, y0, x1, y1])

        """fgmask_erode_2 = cv2.erode(fgmask, kernel_erosion_1, iterations = 1) 
        fgmask_dila_2 = cv2.dilate(fgmask_erode_2, kernel_dilation_2,iterations = 1)

        nlabels, labels, stats_before, centroids = cv2.connectedComponentsWithStats(fgmask_dila_2, connectivity = 8)

        for i in range(len(stats_before)):
            x, y, w, h, area = stats_before[i]

            if area > 3000 : #or area > 600 : # or area < 500 or aspect > 1.2 or aspect < 0.97 : 
               continue
            #cv2.rectangle(image_ori, (x, y), (x + w, y + h), (0,0,255), 3)
        MOG2_img_before = cv2.hconcat([fgmask,fgmask_erode_2,fgmask_dila_2])


        cv2.imshow("MOG2_img_before",MOG2_img_before)
             """

        if debug == True:
            MOG2_img_after = cv2.hconcat([fgmask,fgmask_erode_1,fgmask_dila_1])

            cv2.imshow("MOG2_img_after",MOG2_img_after)

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

def draw_point_court(img, camera_predict_point_list, padding_x, padding_y):

    tennis_court_img = img


    predict_pix_point_list = []

    x_pred = camera_predict_point_list[0]
    y_pred = camera_predict_point_list[1]

    if np.isnan(x_pred):
        return tennis_court_img

    #y_pix_length, x_pix_length = tennis_court_img.shape[0], tennis_court_img.shape[1]
    y_pix_length, x_pix_length = 600, 1276

    x_meter2pix = 23.77 / x_pix_length
    y_meter2pix = 10.97 / y_pix_length

    predict_pix_point_list.append(int(np.round((11.885 + x_pred) / x_meter2pix)))
    predict_pix_point_list.append(int(np.round((5.485 - y_pred) / y_meter2pix)))

    predict_pix_point = predict_pix_point_list[0:2]

    cv2.circle(tennis_court_img,(predict_pix_point[0] + padding_x, predict_pix_point[1] + padding_y), 4, [0, 255, 0], -1)

    return tennis_court_img

def draw_landing_point_court(img, pos_list, padding_x, padding_y):
    tennis_court_img = img


    predict_pix_point_list = []

    x_pred = pos_list[0]
    y_pred = pos_list[1]

    if np.isnan(x_pred):
        return tennis_court_img

    #y_pix_length, x_pix_length = tennis_court_img.shape[0], tennis_court_img.shape[1]
    y_pix_length, x_pix_length = 600, 1276

    x_meter2pix = 23.77 / x_pix_length
    y_meter2pix = 10.97 / y_pix_length

    predict_pix_point_list.append(int(np.round((11.885 + x_pred) / x_meter2pix)))
    predict_pix_point_list.append(int(np.round((5.485 - y_pred) / y_meter2pix)))

    predict_pix_point = predict_pix_point_list[0:2]

    cv2.circle(tennis_court_img,(predict_pix_point[0] + padding_x, predict_pix_point[1] + padding_y), 4, [0, 0, 255], -1)

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

def cal_landing_point(pos_list):

    t_list = []

    if len(pos_list) < 2 : return [np.nan, np.nan, np.nan]

    pos = pos_list[-1]

    x0, y0, z0 = pos[0], pos[1], pos[2]

    vx, vy, vz = get_velocity(pos_list)

    a = -((0.5 * 0.507 * 1.2041 * np.pi * (0.033 ** 2) * vz ** 2 ) / 0.057 + 9.8 / 2 )
    b = vz
    c = z0

    t_list.append((-b + np.sqrt(b ** 2 - 4 * a * c))/(2 * a))
    t_list.append((-b - np.sqrt(b ** 2 - 4 * a * c))/(2 * a))

    t = max(t_list)

    drag_x = (0.5 * 0.507 * 1.2041 * np.pi * (0.033 ** 2) * vx ** 2 )
    drag_y = (0.5 * 0.507 * 1.2041 * np.pi * (0.033 ** 2) * vy ** 2 )
    drag_z = (0.5 * 0.507 * 1.2041 * np.pi * (0.033 ** 2) * vz ** 2 )

    #drag_x = 0
    #drag_y = 0
    #drag_z = 0

    x = np.array(x0 + vx * t - (drag_x * (t ** 2) / 0.057 / 2)  ,float)
    y = np.array(y0 + vy * t - (drag_y * (t ** 2) / 0.057 / 2) ,float)
    z = np.array(z0 + vz * t - ((drag_z / 0.057 + 9.8) * (t ** 2) / 2) ,float)

    #print("x0, y0, z0 : ",x0, y0, z0)
    #print("vx, vy, vz : ",vx, vy, vz)
    
    return [np.round(x,3), np.round(y,3), np.round(z,3)]


def get_velocity(pos_list):

    dT = 1 / 25
    
    np_pos_list = np.array(pos_list)

    x_pos_list = np_pos_list[:,0]
    y_pos_list = np_pos_list[:,1]
    z_pos_list = np_pos_list[:,2]  

    vel_x_list = np.diff(x_pos_list) / dT
    vel_y_list = np.diff(y_pos_list) / dT
    vel_z_list = np.diff(z_pos_list) / dT


    return vel_x_list[-1], vel_y_list[-1], vel_z_list[-1]



def clear_tennis_court_img():
    tennis_court_img = cv2.imread(path + "/images/tennis_court.png")

    tennis_court_img = cv2.resize(tennis_court_img,(0,0), fx=2, fy=2, interpolation = cv2.INTER_AREA) # 1276,600,0

    padding_y = int((810 - tennis_court_img.shape[0]) /2 )
    padding_x = int((1500 - tennis_court_img.shape[1]) /3)

    WHITE = [255,255,255]
    tennis_court_img= cv2.copyMakeBorder(tennis_court_img.copy(),padding_y,padding_y,padding_x,padding_x,cv2.BORDER_CONSTANT,value=WHITE)

    return tennis_court_img