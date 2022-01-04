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

import torch
import torch.backends.cudnn as cudnn

from tools import *
#from tools_backup import *

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

from yolov5.utils.augmentations import letterbox

from kalman_utils.KFilter import *

device = 0
weights = path + "/yolov5/weights/yolov5m6.pt"
imgsz = 640
conf_thres = 0.25
iou_thres = 0.45
classes = [0, 38]
agnostic_nms = False
max_det = 1000
half=False
dnn = False

device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size(imgsz, s=stride)  # check image size


half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA

if pt:
    model.model.half() if half else model.model.float()

cudnn.benchmark = True  # set True to speed up constant image size inference


color = tuple(np.random.randint(low=200, high = 255, size = 3).tolist())
color = tuple([0,125,255])

recode = False
start_frame = 800
video_path = "videos/tennis_video_2/2.mov"

# 2번 frame 1250



def person_tracking(model, img, img_ori, device):

        person_box_left = []
        person_box_right = []

        img_in = torch.from_numpy(img).to(device)
        img_in = img_in.float()
        img_in /= 255.0

        if img_in.ndimension() == 3:
            img_in = img_in.unsqueeze(0)
        

        pred = model(img_in, augment=False, visualize=False)

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):  # detections per image
            
            im0 = img_ori.copy()

            if len(det):
                det[:, :4] = scale_coords(img_in.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s = f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class

                    label = names[c] #None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                    x0, y0, x1, y1 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    x0, y0, x1, y1 = x0 - 10, y0 - 10, x1 + 10, y1

                    plot_one_box([x0, y0, x1, y1], im0, label=label, color=colors(c, True), line_thickness=3)

                    if y0 < (img_ori.shape[0] / 2) :
                        person_box_left.append([x0, y0, x1, y1])

                    else : 
                        person_box_right.append([x0, y0, x1, y1])
            
        return im0, person_box_left, person_box_right


def main(input_video):

    ball_esti_pos = []
    dT = 1 / 25
    

    tennis_court_img = cv2.imread(path + "/images/tennis_court.png")

    tennis_court_img = cv2.resize(tennis_court_img,(0,0), fx=2, fy=2, interpolation = cv2.INTER_AREA) # 1276,600,0

    padding_y = int((810 - tennis_court_img.shape[0]) /2 )
    padding_x = int((1500 - tennis_court_img.shape[1]) /3)


    WHITE = [255,255,255]
    tennis_court_img= cv2.copyMakeBorder(tennis_court_img.copy(),padding_y,padding_y,padding_x,padding_x,cv2.BORDER_CONSTANT,value=WHITE)



    cap_main = cv2.VideoCapture(input_video)

    fps = int(cap_main.get(cv2.CAP_PROP_FPS))
    point_image = np.zeros([408 * 2,720,3], np.uint8) + 255

    if recode:
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("ball_landing_point.mp4", codec, fps, (2144,810))

    estimation_ball = Ball_Pos_Estimation()

    disappear_cnt = 0
    ball_pos_jrajectory = []

    total_frmae = int(cap_main.get(cv2.CAP_PROP_FRAME_COUNT))

    print("total_frmae : ",total_frmae)

    cap_main.set(cv2.CAP_PROP_POS_FRAMES, start_frame) #set start frame number

    while cap_main.isOpened():

        print("-----------------------------------------------------------------")
        t1 = time.time()

        frame_count = int(cap_main.get(cv2.CAP_PROP_POS_FRAMES))

        print("frame_count : ",frame_count)

        ball_box_left = []
        ball_box_right = []

        ball_cen_left = []
        ball_cen_right = []

        ball_pos = []

        ret, frame = cap_main.read()
        
        """frame = frame_main[:,320:960,:]
        frame_left = frame_main[0:360,:590,:]
        frame_right = frame_main[360:,50:,:]"""

        frame = cv2.resize(frame, dsize=(720, 405 * 2), interpolation=cv2.INTER_LINEAR)
        #frame = cv2.resize(frame, dsize=(1072, 603 * 2), interpolation=cv2.INTER_LINEAR)

        frame_left = frame[0 : int(frame.shape[0]/2), : , : ]
        frame_right = frame[int(frame.shape[0]/2): , : , : ]

                
        frame_main = cv2.vconcat([frame_left,frame_right])

        frame_recode = cv2.vconcat([frame_left,frame_right])

        frame_mog2 = frame_main.copy()
        frame_yolo_main = frame_main.copy()

        img, img_ori = img_preprocessing(frame_yolo_main, imgsz, stride, pt)

        person_tracking_img, person_box_left_list, person_box_right_list = person_tracking(model, img, img_ori, device)

        ball_detect_img, ball_cand_box_list_left, ball_cand_box_list_right = ball_tracking(frame_mog2)  #get ball cand bbox list

        if ball_cand_box_list_left:
            ball_box_left = check_iou(person_box_left_list, ball_cand_box_list_left) # get left camera ball bbox list

        if ball_cand_box_list_right:
            ball_box_right = check_iou(person_box_right_list, ball_cand_box_list_right) # get right camera ball bbox list

        ball_box = [ball_box_left, ball_box_right]

        if ball_box:  #draw ball bbox 
            
            total_ball_box = ball_box[0] + ball_box[1]

            for i in range(len(total_ball_box)):
                x0, y0, x1, y1 = total_ball_box[i]

                ball_x_pos, ball_y_pos = int((x0 + x1)/2), int((y0 +y1)/2)

                cv2.rectangle(frame_main, (x0, y0), (x1, y1), color, 3)

                if recode == True :
                    cv2.rectangle(frame_recode, (x0, y0), (x1, y1), color, 3)

                #cv2.circle(point_image,(ball_x_pos, ball_y_pos), 4, color, -1)

                #ball_list.append([ball_x_pos, ball_y_pos])

        ball_cen_left = trans_point(frame_main, ball_box_left)
        ball_cen_right = trans_point(frame_main, ball_box_right)

            

        print("ball_cen_left = ",ball_cen_left)
        print("ball_cen_right = ",ball_cen_right)

        print("KF_flag : ",estimation_ball.kf_flag)

        if len(ball_cen_left) and len(ball_cen_right): #2개의 카메라에서 ball이 검출 되었는가?
            fly_check = estimation_ball.check_ball_flying(ball_cen_left, ball_cen_right)
            if (fly_check) == 1:

                

                ball_cand_pos = estimation_ball.get_ball_pos()

                print("check_ball_fly")

                if estimation_ball.kf_flag:
                    print("ball_detect_next")


                    pred_ball_pos = estimation_ball.kf.get_predict()
                    ball_pos = get_prior_ball_pos(ball_cand_pos, pred_ball_pos)

                    ball_pos = estimation_ball.ball_vel_check(ball_pos)

                    ball_pos_jrajectory.append(ball_pos)

                    estimation_ball.kf.update(ball_pos[0], ball_pos[1], ball_pos[2], dT)

                    estimation_ball.ball_trajectory.append([ball_pos])

                else:
                    print("ball_detect_frist")

                    if len(ball_cand_pos) > 1:
                        pass
                        #***************사람 위치와 공 위치 평가 함수******************
                        #임시

                        del_list = []

                        for i in range(len(ball_cand_pos)) : #임시

                            if abs(ball_cand_pos[i][1]) > 13.1 / 2 :

                                del_list.append(i)

                        ball_cand_pos = np.delete(np.array(ball_cand_pos),del_list,axis = 0).tolist()

                        ball_pos = ball_cand_pos[(9 - abs(np.array(ball_cand_pos)[:,0])).argmin()]

                        ball_pos_jrajectory.append(ball_pos)

                        estimation_ball.kf = Kalman_filiter(ball_pos[0], ball_pos[1], ball_pos[2], dT)
                        estimation_ball.kf_flag = True
                        estimation_ball.ball_trajectory.append([ball_pos])

                    else:
                        ball_pos = ball_cand_pos[0]
                        ball_pos_jrajectory.append(ball_pos)

                        estimation_ball.kf = Kalman_filiter(ball_pos[0], ball_pos[1], ball_pos[2], dT)
                        estimation_ball.kf_flag = True
                        estimation_ball.ball_trajectory.append([ball_pos])

            elif (fly_check) == 3:
                print("setup_ball_fly")
                #estimation_ball.reset_ball()

            else : 
                print("not_detect_fly_ball")

                if estimation_ball.kf_flag == True:
                    print("ball_predict_next_KF")
                    
                    estimation_ball.kf.predict(dT)

                    ball_pos = estimation_ball.kf.get_predict()

                    ball_pos = estimation_ball.ball_vel_check(ball_pos)

                    ball_pos_jrajectory.append(ball_pos.tolist())


                    estimation_ball.ball_trajectory.append([ball_pos])

                    print("pred_ball_pos = ",ball_pos)



                else : 
                    print("reset_ALL")
                    estimation_ball.reset_ball()
                    ball_pos_jrajectory.clear()

                
        else:
            print("not ball_detect")

            if estimation_ball.kf_flag: #칼만 필터가 있는가?
                print("ball_predict_next_KF")

                estimation_ball.kf.predict(dT)

                ball_pos = estimation_ball.kf.get_predict()

                ball_pos = estimation_ball.ball_vel_check(ball_pos)

                ball_pos_jrajectory.append(ball_pos.tolist())

                estimation_ball.ball_trajectory.append([ball_pos])

                print("pred_ball_pos = ",ball_pos)

                disappear_cnt += 1

                if ball_pos[2] < 0 or disappear_cnt > 4 or  ball_pos[0] > 0 :
    
                    print("reset_ALL")

                    estimation_ball.reset_ball()
                    ball_pos_jrajectory.clear()
                    disappear_cnt = 0


            else:
                print("reset_ALL")
                estimation_ball.reset_ball()
                ball_pos_jrajectory.clear()

        if len(ball_pos):
            #print("ball_pos_jrajectory = ",ball_pos_jrajectory)

            ball_landing_point = cal_landing_point(ball_pos_jrajectory)

            draw_point_court(tennis_court_img, ball_pos, padding_x, padding_y)
            draw_landing_point_court(tennis_court_img, ball_landing_point, padding_x, padding_y)

            print("ball_pos = ",ball_pos)
            print("ball_landing_point = ",ball_landing_point)


        t2 = time.time()


        cv2.imshow('person_tracking_img',person_tracking_img)
        cv2.imshow('ball_detect_img',ball_detect_img)

        #cv2.imshow('tennis_court_img',tennis_court_img)
        #cv2.imshow('frame_main',frame_main)


        frame_recode = cv2.hconcat([frame_main,tennis_court_img])

        cv2.imshow('frame_recode',frame_recode)

        if recode:

            print(frame_recode.shape)
            out.write(frame_recode)


        print("FPS : " , 1/(t2-t1))

        key = cv2.waitKey(0)

        if key == ord("c") : 
            tennis_court_img = cv2.imread(path + "/images/tennis_court.png")

            tennis_court_img = cv2.resize(tennis_court_img,(0,0), fx=2, fy=2, interpolation = cv2.INTER_AREA) # 1276,600,0

            padding_y = int((810 - tennis_court_img.shape[0]) /2 )
            padding_x = int((1500 - tennis_court_img.shape[1]) /3)


            WHITE = [255,255,255]
            tennis_court_img= cv2.copyMakeBorder(tennis_court_img.copy(),padding_y,padding_y,padding_x,padding_x,cv2.BORDER_CONSTANT,value=WHITE)

            #print(tennis_court_img.shape)

        if key == 27 : 
            cap_main.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":

    main(video_path)