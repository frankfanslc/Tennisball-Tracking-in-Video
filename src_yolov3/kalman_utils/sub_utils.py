import cv2
import numpy as np
import os
import math
from scipy.spatial import distance as dist
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints


lower_green = np.array([89, 250, 250])
upper_green = np.array([90, 255, 255])

lower_rgb = np.array([10,180,120])
upper_rgb = np.array([40,255,250])


class Bouncing_point():

    def __init__(self):

        self.ball_left_center_list = []
        self.ball_right_center_list = []

        self.left_gradient_list = []
        self.right_gradient_list = []

        self.bouncing_point = []

    def gradient_check(self,num):

        if num >= 0:
            return True
        
        elif num < 0:
            return False
     

    def gradient_cal(self,ball_center_list):

        x_pre,y_pre = ball_center_list[0][0], ball_center_list[0][1]
        x,y = ball_center_list[1][0], ball_center_list[1][1]
        

        delta_x = x-x_pre
        delta_y = y-y_pre
 
        #if (abs(delta_x) or abs(delta_y))> 50 or delta_x == 0:
        if (abs(delta_x) or abs(delta_y))> 200 or delta_x == 0:
        #if delta_x == 0:
            #self.ball_center_list = []
            #self.ball_state_list = []
            #self.gradient_list = []

            return False

        gradient = delta_y/delta_x

        if ((x_pre + x)/2) < 640 and x_pre > x:
            self.left_gradient_list.append(gradient)

        elif ((x_pre + x)/2) > 640 and x_pre < x :
            self.right_gradient_list.append(gradient)
    
        if len(self.left_gradient_list) == 2: #left view
            if self.gradient_check(self.left_gradient_list[0]) == False and self.gradient_check(self.left_gradient_list[1]) == True:
                self.bouncing_point.append([(x_pre + x)/2, (y_pre + y)/2])
                
            del self.left_gradient_list[0]

        if len(self.right_gradient_list) == 2: #right view
            if self.gradient_check(self.right_gradient_list[0]) == True and self.gradient_check(self.right_gradient_list[1]) == False:
                self.bouncing_point.append([(x_pre + x)/2, (y_pre + y)/2])

            del self.right_gradient_list[0]
                
            


    def append_ball(self,ball_center_list,):

        if ball_center_list[0] < 640:
            self.ball_left_center_list.append(ball_center_list)

            if len(self.ball_left_center_list) > 2:
                del self.ball_left_center_list[0]

            if len(self.ball_left_center_list) == 2:
                self.gradient_cal(self.ball_left_center_list)
        else :
            self.ball_right_center_list.append(ball_center_list)

            if len(self.ball_right_center_list) > 2:
                del self.ball_right_center_list[0]

            if len(self.ball_right_center_list) == 2:
                self.gradient_cal(self.ball_right_center_list)
                

class Trajectory_center:

    def __init__(self, maxDisappeared = 10):

        self.nextObjectID = 0
        self.point_dict = OrderedDict()
        self.disappeared_dict = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.point_dict[self.nextObjectID] = [centroid]
        self.disappeared_dict[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.point_dict[objectID]
        del self.disappeared_dict[objectID]

    def update(self, next_centroid_list):
        
        if len(next_centroid_list) == 0:
       
            for ID in list(self.disappeared_dict.keys()):
                self.disappeared_dict[ID] += 1
                
                if self.disappeared_dict[ID] >= self.maxDisappeared:
                    self.deregister(ID)

            return self.point_dict
        
        if len(self.point_dict) == 0:
            for i in range(len(next_centroid_list)):
                self.register(next_centroid_list[i])
                
        else:
            objectIDs = list(self.point_dict.keys())     
            pre_point = list()
            
            for ID in list(self.point_dict.keys()):
                
                pre_point.append(((self.point_dict[ID])[-1]))

            
            distan = dist.cdist(np.array(pre_point), next_centroid_list)
            ID_list, indexes = linear_sum_assignment(distan)

            used_ID = set()
            used_next_pts = set()
            
            for i in (ID_list):
                
                objectID = objectIDs[i]
                
                min_index = (ID_list.tolist()).index(i)
                
                if distan[i][indexes[min_index]] > 100:
                    continue
                            
                self.point_dict[objectID].append(next_centroid_list[indexes[min_index]])

                self.disappeared_dict[objectID] = 0
                
                used_ID.add(objectID)
                used_next_pts.add(indexes[min_index])
                
            unused_ID = set(objectIDs).difference(used_ID)
            unused_next_pts = set(range(len(next_centroid_list))).difference(used_next_pts)

            if unused_ID:
                for ID in unused_ID:
                    self.disappeared_dict[ID] += 1
                
                    if self.disappeared_dict[ID] > self.maxDisappeared:
                        self.deregister(ID)

            if unused_next_pts:
                for index in unused_next_pts:
                    self.register(next_centroid_list[index])
            
                    
            return self.point_dict
            


            
       
            
"""class Kalman_filter():
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas, init_x, init_y):  
        
        dt = sample time 
        u_x = x 방향 가속도
        u_y = y 방향 가속도
        std_acc = process noise magnitude
        x_std_meas = x 방향 측정 표준편차
        y_std_meas = y 방향 측정 표준편차
        
        self.init_x = init_x 
        self.init_y = init_y
        self.dt = dt

        self.u = np.matrix([[u_x],[u_y]])
        self.x = np.matrix([[self.init_x],[self.init_y],[0],[0]]) #초기 위치


        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        self.R = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])

        self.P = np.eye(self.A.shape[1])
        
        self.update([self.init_x, self.init_y])

    def predict(self):

        #x_k =Ax_(k-1) + Bu_(k-1)    
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # P= A*P*A' + Q              
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        point = self.x[0:2].tolist()

        return point[-1]

    def update(self, z):
        # S = H*P*H'+R

        #Z = np.matrix(z)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) 
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
        I = np.eye(self.H.shape[1])

        self.P = (I - (K * self.H)) * self.P

        point = self.x[0:2].tolist()

        return point[-1]"""

      



def img2court(M,point):

    div = M[2][0]*point[0] + M[2][1]*point[1] + 1
    new_x = (M[0][0]*point[0] + M[0][1]*point[1] + M[0][2]) / div
    new_y = (M[1][0]*point[0] + M[1][1]*point[1] + M[1][2]) / div

    return new_x, new_y



def sum_box(center_list,stats_list): #두점 사이 거리를 계산하여 일정 거리 미만일 경우 합친다. 
    
    if len(center_list) < 2:
        return center_list, stats_list

    center_points = center_list
    stats = stats_list
    i , j = 0 , 1

    while True:
        
        if i + 1 == len(center_points):
            break

        #score = int(math.sqrt((center_points[i][0] - center_points[j][0])**2 + (center_points[i][1] - center_points[j][1])**2))

        x_length = center_points[i][0] - center_points[j][0]
        y_length = center_points[i][1] - center_points[j][1]
        #print(i,j,score)

        #if score < 130:
        if abs(x_length) < 30 and abs(y_length) < 130:
            
            #stats_list 변경 stats_points([x, y, width, height, area])
            min_x = int(min(stats[i][0],stats[j][0]))
            min_y = int(min(stats[i][1],stats[j][1]))
            new_width = int(max(stats[i][0] + stats[i][2],stats[j][0] + stats[j][2])) - min_x
            new_height = int((max(stats[i][1] + stats[i][3],stats[j][1] + stats[j][3]))) - min_y
            new_area = new_width * new_height
            stats.append([min_x, min_y, new_width, new_height, new_area])

            new_cen_x = int(new_width + min_x)
            new_cen_y = int(new_height + min_y)
            center_points.append([new_cen_x, new_cen_y])

            del center_points[j]
            del center_points[i]
            del stats[j]
            del stats[i]

            j = i + 1
        
        else:
            j += 1

            if j == len(center_points):
                i += 1
                j = i+1
                
    return center_points, stats
    

def bounce_point2top_view(point,M_letf,M_right):

    x, y = point

    if x < 640:
        new_x , new_y = img2court(M_letf,[x,y])

    else:
        new_x , new_y = img2court(M_right,[x,y])

    return int(new_x), int(new_y)


def check_player_box(center_points_list, stats_points_list, stats_points_open_list, pre_stats_points_open_L,pre_stats_points_open_R):

    i = 0

    player_stats_points_list = []

    stats_points_open_L, stats_points_open_R = stats_points_open_list[0], stats_points_open_list[1]

    if len(center_points_list) == 0:
        return center_points_list, stats_points_list, player_stats_points_list

    if len(stats_points_open_L) == 0:
        stats_points_open_L = pre_stats_points_open_L
    
    if len(stats_points_open_R) == 0:
        stats_points_open_R = pre_stats_points_open_R

        
    while i < len(center_points_list):

        x_center, y_center = center_points_list[i]
        x, y, width, height, area = stats_points_list[i]

        if x_center < 640 :
            if len(stats_points_open_L) == 0 : 
                i += 1 
                continue

            for j in range(len(stats_points_open_L)):
                #x_open, y_open, width_open, height_open, area_open = stats_points_open_L[j]

                #if x_open < x_center < (width_open + x_open):
                if nms(stats_points_list[i], stats_points_open_L[j]):

                    player_stats_points_list.append(stats_points_list[i])
                    del center_points_list[i]
                    del stats_points_list[i]
                    break

                if j == (len(stats_points_open_L)-1):
                    i += 1

        else:
            if len(stats_points_open_R) == 0 : 
                i += 1 
                continue

            for j in range(len(stats_points_open_R)):
                #x_open, y_open, width_open, height_open, area_open = stats_points_open_R[j]

                if nms(stats_points_list[i], stats_points_open_R[j]):

                    player_stats_points_list.append(stats_points_list[i])
                    del center_points_list[i]
                    del stats_points_list[i]
                    break

                if j == (len(stats_points_open_R)-1):
                    i += 1

    return center_points_list, stats_points_list, player_stats_points_list

def get_center_point(centroids_list, stats_list):

    center_points = []
    stats_points = []
    stats = stats_list
    centroids = centroids_list

    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[index]
        centerX, centerY = int(x + width/2), int(y + height/2)

        center_points.append([centerX,centerY])
        stats_points.append([x, y, width, height, area])

    return center_points, stats_points

def split_stats_point(stats_list):

    stats_points_open_L = []
    stats_points_open_R = []

    for index, stats in enumerate(stats_list):

        if (stats[0] <= 0 or stats[1] <= 0) or stats[0] > 2000:
            continue
        if np.any(np.isnan(stats)):
            continue

        x, y, width, height, area = stats

        if x < 640 :
            stats_points_open_L.append([x, y, width, height, area])
        else: 
            stats_points_open_R.append([x, y, width, height, area])

    return stats_points_open_L, stats_points_open_R    


def draw_rectangle(image, points, color = (0,255,0), info = False):

    if len(points) > 0:
        for i in range(len(points)):


            if len(points[i])>3:
                x, y, width, height, area = points[i]

                if x < 0: continue

                if info:
                    cv2.putText(image, str(width/(height+1)), (x - 1, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA) 
                cv2.rectangle(image, (x, y), (x + width, y + height), color,3)
            
            elif len(points[i]) == 2:

                x, y = points[i]

                if x < 1: continue

                if info:
                    cv2.putText(image, str(width/(height+1)), (x - 1, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA) 
                cv2.rectangle(image, (int(x) - 10, int(y) - 10), (int(x) + 10, int(y) + 10), color,3)


def nms(box_1, box_2):

    x_1, y_1, width_1, height_1, are_1 = box_1 
    x_2, y_2, width_2, height_2, are_2 = box_2

    max_x = np.argmax([x_1,x_2])

    if max_x == 0:
        if (x_2 <= x_1 <= (x_2 + width_2)) or (x_2 <= (x_1 + width_1) <= (x_2 + width_2)):
            return True

        else:
            return False
    
    else:
        if (x_1 <= x_2 <= (x_1 + width_1)) or (x_1 <= (x_2 + width_2) <= (x_1 + width_1)):
            return True
        
        else:
            return False


def match_player_box(pre_stats_points_open,stats_points):

    for j in range(len(pre_stats_points_open)):
        x_open, y_open, width_open, height_open, area_open = pre_stats_points_open[j]
  
        for i in range(len(stats_points)):
            x, y, width, height, area = stats_points[i]

            if nms(pre_stats_points_open[j], stats_points[i]):
                pre_stats_points_open[j] = stats_points[i]
                break

    return pre_stats_points_open


def get_player_postion(stats_points,M_letf,M_right):
    
    
    player_postion_list = []

    for i in range(len(stats_points)):
        x, y, width, height, area = stats_points[i]

        player_x, player_y = bounce_point2top_view([x+(width/2),y+height-(height/5)],M_letf,M_right)

        if player_x < 25:
            player_x = 25
        
        elif player_x > 937:
            player_x = 957

        player_postion_list.append([player_x, player_y])

    return player_postion_list

def get_ball_point(ball_candidate_list):

    if len(ball_candidate_list) == 0 or len(box_list) == 0:
        return ball_candidate_list
    
    i = 0
    while i < len(ball_candidate_list):
        ball_x, ball_y = ball_candidate_list[i]
        #print(i, len(ball_candidate_list))

        for j in range(len(box_list)):
            x, y, width, height, area = box_list[j]

            if (x < ball_x < (x + width)) and (y < ball_y < (y + height)):
                del ball_candidate_list[i]
                break

            if j == (len(box_list)-1):
                i += 1
    
    return ball_candidate_list

def check_ball(stats_points):

    x, y, width, height, area = stats_points
    aspect_ratio = width / (height+1)

    if 0.94 < aspect_ratio < 1.11 and area < 2500:
        return True

    else:
        return False


def get_ball_list(stats_list):

    ball_center_list = []
    ball_stats_list = []

    if len(stats_list) == 0:
        return ball_center_list, ball_stats_list

    for i in range(len(stats_list)):

        x, y, width, height, area = stats_list[i]
        ball_center_list.append([(x + width/2), (y + height/2)])
        ball_stats_list.append(stats_list[i])
        
    return ball_center_list, ball_stats_list



def get_person(img, outputs):

    wh = np.flip(img.shape[0:2])
    boxes, scores, classes, nums = outputs

    boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]

    person_boxe = []

    for i in range(nums):

        if int(classes[i]) == 0 or int(classes[i]) == 38:

            x1, y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2, y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))

            person_boxe.append([x1, y1, x2, y2])


    return person_boxe
        
    


def get_ball(point_list, person_box):

    j = 0

    for i in range(len(person_box)):


        x1, y1, x2, y2 = person_box[i]

        while True:

            if len(point_list) <= j:
                j = 0
                break

            x_cen, y_cen = point_list[j]

            if (x1 <= x_cen <= x2) and (y1 <= y_cen <= y2):
                del point_list[j]
            
            else:
                j += 1

    return point_list
        
    
def trans_xy(point_list):

    for i in range(len(point_list)):
        x_cen, y_cen = point_list[i]

        if 360 < y_cen:
            point_list[i][0] = x_cen + 590
            point_list[i][1] = y_cen - 360

    return point_list