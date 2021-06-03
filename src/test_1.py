import cv2
import numpy as np
import os
import math
import time
from kalman_utils.sub_utils import *
from kalman_utils.KFilter import *
from kalman_utils.UKFilter import *


fgbg = cv2.createBackgroundSubtractorMOG2(500, 30, False)
kernel_dilation_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))


def main(video_name):
    cap = cv2.VideoCapture(video_name)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    fps = cap.get(cv2.CAP_PROP_FPS)


    tj = Trajectory_kf()
    frame_count = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("test_kf.avi", fourcc, 60, (int(width),int(height)))

    while cap.isOpened():

        ret, frame = cap.read()
        
        blur = cv2.GaussianBlur(frame, (13, 13), 0)

        fgmask_1 = fgbg.apply(blur, None, 0.01)

        fgmask_dila_1 = cv2.dilate(fgmask_1,kernel_dilation_2,iterations = 1)

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask_dila_1, connectivity = 8)

        center_points, stats_points = get_center_point(centroids,stats)

        #draw_rectangle(frame,stats_points,(255,255,0))

        print("frame :",frame_count)
        objects_dict = tj.update(center_points)

        if objects_dict:
            for (objectID, centroid) in objects_dict.items():
                
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                np.random.seed(objectID)
                
                color = tuple(np.random.randint(low=50, high = 255, size = 3).tolist())
                text = "ID {}".format(objectID)
                x, y = centroid[-1]
                cv2.putText(frame, text, (x -10, y -10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (x, y), 4, color, -1)
                point = tj.kf_pred_dict[objectID]

                #print(objects_dict)

                x_pred, y_pred = point[0], point[1]

                cv2.putText(frame, text, (int(x_pred) - 10, int(y_pred) -15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (int(x_pred) - 10, int(y_pred) -10), (int(x_pred) + 10, int(y_pred) + 10), color, 3)


        frame_count += 1
        
        #cv2.imshow("fask",fgmask_dila_1)
        cv2.imshow('frame',frame)


        out.write(frame)

        key = cv2.waitKey(10)

        if key == 27:
            break

        elif key == ord('c'):
            cv2.imwrite("images/main_frame/main_{}.png".format(frame_count),frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #main('video/test_video.mp4')
    main('video/unkf-2021-05-24_13.37.47.mp4')