# Tennisball-Tracking-in-Video
Tennisball Tracking in Video




Convert pre-trained Darknet weights
'''bash
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf
'''



'''bash
python src/main.py
'''

