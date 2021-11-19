#! /home/drcl_yang/anaconda3/envs/py36/bin/python

from pathlib import Path
import sys
import os

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path


import numpy as np
import time
import cv2


windowName = 'image'

def ball_tracking(image):

        ball_cand_box = []

        image_ori = image.copy()

        blur = cv2.GaussianBlur(image_ori, (13, 13), 0)

        fgmask = fgbg.apply(blur, None, 0.1)

        fgmask_erode = cv2.erode(fgmask, kernel_erosion_1, iterations = 1) #오픈 연산이아니라 침식으로 바꾸자

        fgmask_dila = cv2.dilate(fgmask_erode,kernel_dilation_2,iterations = 1)

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask_dila, connectivity = 8)

        for i in range(len(stats)):
            x, y, w, h, area = stats[i]
            
            if area > 3000 : # or area < 500 or aspect > 1.2 or aspect < 0.97 : 
                continue
            cv2.rectangle(image_ori, (x, y), (x + w, y + h), (255,0,0), 3)

            x0, y0, x1, y1 = x, y, x+w, y+h

            ball_cand_box.append([x0, y0, x1, y1])

        MOG2_img = cv2.hconcat([fgmask,fgmask_erode,fgmask_dila])


        cv2.imshow("MOG2_img",MOG2_img)

        #cv2.imshow("fgmask",fgmask)
        #cv2.imshow("fgmask_erode",fgmask_erode)
        #cv2.imshow("fgmask_dila",fgmask_dila)

        
        return image_ori, ball_cand_box

def main(input_video):


    check_id = 0
    ball_id = 0

    cap_main = cv2.VideoCapture(input_video)

    fps = int(cap_main.get(cv2.CAP_PROP_FPS))

    while cap_main.isOpened():
        t1 = time.time()

        ball_list = []

        ret, frame_main = cap_main.read()
        
        """frame = frame_main[:,320:960,:]
        frame_left = frame_main[0:360,:590,:]
        frame_right = frame_main[360:,50:,:]"""

        frame = cv2.resize(frame_main, dsize=(720, 408 * 2), interpolation=cv2.INTER_LINEAR)
        frame_left = frame_main[0:408,:,:]
        frame_right = frame_main[408:,:,:]
                
        frame_main = cv2.vconcat([frame_left,frame_right])

        frame_mog2 = frame_main.copy()


        ball_detect_img, ball_cand_box_list = ball_tracking(frame_mog2)  #get ball cand bbox list


        t2 = time.time()


        #cv2.imshow('frame_recode',frame_recode)
        #cv2.imshow('person_tracking_img',person_tracking_img)
        # cv2.imshow('ball_detect_img',ball_detect_img)
        # cv2.imshow('point_image',point_image)

        cv2.imshow('frame_main',frame_main)



        print("FPS : " , 1/(t2-t1))
        key = cv2.waitKey(1)

        if key == 27 : 
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":

    video_path = "videos/tennis_video_1.mov"
    main(video_path)