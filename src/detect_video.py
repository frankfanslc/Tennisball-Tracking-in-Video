import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils_pre import draw_outputs_ori, image_preprocess
import numpy as np
import serial

#python detect_video.py --classes ./data/custom_1.names --weights ./weight/waste_classification_3_60.tf --video ./video/detect_video/test_video.mp4 --num_classes 2

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weight/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './video/in and out_1.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', './video/in and out_1.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


FLAGS.yolo_iou_threshold = 0.3
FLAGS.yolo_score_threshold = 0.6


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

    

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)
        
    out = None

    #if FLAGS.output:
    if True:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        #out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue
        
        img = cv2.resize(img,(0,0),fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        img,location = draw_outputs_ori(img, (boxes, scores, classes, nums), class_names) 

        cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        #out.write(img)

        cv2.imshow('output', img)

        if cv2.waitKey(1) == 27 : break
            


    vid.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


