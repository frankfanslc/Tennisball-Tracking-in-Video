import cv2
import numpy as np
import os
import math
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import *
import numpy as np
from kalman_utils.sub_utils import *
from kalman_utils.KFilter import *
from kalman_utils.UKFilter import *

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weight/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './video/main.mov',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', './video/recode.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


FLAGS.yolo_iou_threshold = 0.3
FLAGS.yolo_score_threshold = 0.5


fgbg = cv2.createBackgroundSubtractorMOG2(500, 30, False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

kernel_dilation_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))

kernel_erosion_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))


def main(_argv):

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    cap_main = cv2.VideoCapture(FLAGS.video)

    width = int(cap_main.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_main.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_main.get(cv2.CAP_PROP_FPS))
    
    point_image = np.zeros([360,1180,3], np.uint8) + 255

    trajectroy_image = point_image
    
    frame_count = 0
    
    tj_ukf = Trajectory_ukf()
    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    #out = cv2.VideoWriter("main.mp4", codec, fps, (1180,360))
    #out_1 = cv2.VideoWriter("trajectroy.mp4", codec, fps, (1180,360))

    check_id = 0
    ball_id = 0

    while cap_main.isOpened():

        ret, frame = cap_main.read()
        
        frame = frame[:,320:960,:]

        frame_left = frame[0:360,:590,:]
        frame_right = frame[360:,50:,:]

        frame_main = cv2.vconcat([frame_left,frame_right])

        #frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        frame_mog2 = frame_main.copy()
        frame_yolo_main = frame_main.copy()

        time_1 = time.time()
        
        in_main = cv2.cvtColor(frame_yolo_main, cv2.COLOR_BGR2RGB) 
        in_main = tf.expand_dims(in_main, 0)
        in_main = transform_images(in_main, FLAGS.size)
        boxes, scores, classes, nums = yolo(in_main)

        #frame_yolo, location = draw_outputs(frame_main, (boxes, scores, classes, nums), class_names) 


        time_2 = time.time()

        """cv2.putText(frame_main, "Time: {:.2f}fps".format(1/(time_2 - time_1)), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)"""
        
    
        person_boxes = get_person(frame_yolo_main, (boxes, scores, classes, nums))

        for i in range(len(person_boxes)):
            x1, y1, x2, y2 = person_boxes[i]
            cv2.rectangle(frame_main, (x1, y1), (x2, y2), (255,0,0),4)




        # 가우시안 블러 적용
        blur = cv2.GaussianBlur(frame_mog2, (13, 13), 0)

        # Background 마스크 생성 background x

        fgmask_1 = fgbg.apply(blur, None, 0.01)


        fgmask_erode = cv2.erode(fgmask_1, kernel_erosion_1, iterations = 1) #오픈 연산이아니라 침식으로 바꾸자

        fgmask_dila = cv2.dilate(fgmask_erode,kernel_dilation_2,iterations = 1)
        #fgmask_dila_2 = cv2.dilate(fgmask_open_1,kernel_dilation_2,iterations = 2)

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask_dila, connectivity = 8)
        center_points, stats_points = get_center_point(centroids,stats)

        _, new_stats_points = sum_box(center_points,stats_points) ##***************

        center_list, stats_list = get_ball_list(new_stats_points)

        #draw_rectangle(frame_test,ball_stats_list,(255,255,0))
        #draw_rectangle(frame_main,stats_list,(255,0,0))

        ball_list = get_ball(center_list, person_boxes)  ## change IOU tennis  

        ball_list = trans_xy(ball_list)

        frame_recode = cv2.hconcat([frame_left,frame_right])
         
        objects_dict = tj_ukf.update(ball_list)
        
        if objects_dict:
            for (objectID, centroid) in objects_dict.items():

                ball_id = objectID

                np.random.seed(objectID)
                
                color = (0,0,255)
                text = "ID {}".format(objectID)
                x, y = centroid[-1]
                cv2.putText(frame_recode, text, (int(x) -10, int(y) -10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame_recode, (int(x), int(y)), 4, color, -1)
                cv2.circle(trajectroy_image,(int(x), int(y)), 4, color, -1)
                point = tj_ukf.kf_pred_dict[objectID]

                x_pred, y_pred = point[0], point[1]
                

                #cv2.rectangle(frame_recode, (int(x_pred) - 10, int(y_pred) -10), (int(x_pred) + 10, int(y_pred) + 10), color, 3)




        #frame_recode = cv2.vconcat([frame_test,point_image])

        #cv2.imshow('fgmask_erode',fgmask_erode)
        #cv2.imshow('fgmask_dila',fgmask_dila)
        cv2.imshow('frame_main',frame_main)

        cv2.imshow('trajectroy_image',trajectroy_image)
        

        cv2.imshow('frame_recode',frame_recode)
        

        #out.write(frame_recode)
        #out_1.write(trajectroy_image)

    
        frame_count += 1
        print(frame_count)

        key = cv2.waitKey(1)

        if key == 27 : break

        elif key == ord('c') or True:
            cv2.imwrite("images/main/main_{}.png".format(frame_count),frame_recode)
            cv2.imwrite("images/trajectory/trajectory_{}.png".format(frame_count),trajectroy_image)



        if check_id != ball_id:
            trajectroy_image = np.zeros([360,1180,3], np.uint8) + 255
            check_id = ball_id

    cap_main.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #main(["video/tennis_left.mp4","video/tennis_right.mp4"])
    app.run(main)


