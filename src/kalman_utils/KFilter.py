import cv2
import numpy as np
import os
import math
from scipy.spatial import distance as dist
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

  
class Kalman_filter():

    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas, init_x, init_y):  
        
        self.init_x = init_x 
        self.init_y = init_y
        self.dt = dt

        self.f = KalmanFilter(dim_x=4, dim_z=2)
        self.f.x = np.array([[self.init_x],[self.init_y],[0],[0]])
        self.f.F = np.array([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.f.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        self.f.P = np.eye(self.f.F.shape[1])

        self.f.Q *= np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        self.f.R = np.array([[x_std_meas**2,0],
                           [0, y_std_meas**2]])

        self.f.predict()



class Trajectory_kf:
    def __init__(self, maxDisappeared = 10):

        self.nextObjectID = 0
        self.point_dict = OrderedDict()
        self.disappeared_dict = OrderedDict()
        self.kf_dict = OrderedDict()
        self.kf_pred_dict = OrderedDict()
        self.maxDisappeared = maxDisappeared
        #self.kf_esti_dict = OrderedDict()


    def register(self, centroid):
        self.point_dict[self.nextObjectID] = [centroid]
        self.disappeared_dict[self.nextObjectID] = 0
        self.kf_dict[self.nextObjectID] = Kalman_filter(dt = 0.3, 
                                                        u_x = 0, 
                                                        u_y = 0.1, 
                                                        std_acc = 0.01, 
                                                        x_std_meas = 0.01, 
                                                        y_std_meas = 0.01,
                                                        init_x = centroid[0],
                                                        init_y = centroid[1])
        
        self.kf_pred_dict[self.nextObjectID] = centroid
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.point_dict[objectID]
        del self.disappeared_dict[objectID]
        del self.kf_dict[objectID]
        del self.kf_pred_dict[objectID]

    def update(self, next_centroid_list):
        
        if len(next_centroid_list) == 0:
       
            for ID in list(self.disappeared_dict.keys()):
                self.disappeared_dict[ID] += 1

                pred_point = self.kf_dict[ID].f.x  
                x, y = int(pred_point[0]), int(pred_point[1])
                self.kf_pred_dict[ID] = [x, y]
                
                if self.disappeared_dict[ID] >= self.maxDisappeared:
                    self.deregister(ID)
                    
            return self.point_dict
        
        if len(self.point_dict) == 0:
            for i in range(len(next_centroid_list)):
                self.register(next_centroid_list[i])
                
        else:
            objectIDs = list(self.point_dict.keys())     
            #pre_point = list()
            self.kf_predict_list = list()
            
            for ID in objectIDs:
                
                #pre_point.append(((self.point_dict[ID])[-1]))
                
                self.kf_dict[ID].f.predict()
                pred_point = self.kf_dict[ID].f.x
                x, y = int(pred_point[0]), int(pred_point[1])
                self.kf_pred_dict[ID] = [x, y]
                self.kf_predict_list.append([x, y])


            distan = dist.cdist(np.array(self.kf_predict_list), next_centroid_list)
            

            ID_list, indexes = linear_sum_assignment(distan)
            
                
            used_ID = set()
            used_next_pts = set()

            for i in (ID_list):
                
                objectID = objectIDs[i]
                
                min_index = (ID_list.tolist()).index(i)
                
                """if distan[i][indexes[min_index]] > 100:
                    continue"""
                            
                self.point_dict[objectID].append(next_centroid_list[indexes[min_index]])

                self.disappeared_dict[objectID] = 0
                self.kf_dict[ID].f.update(next_centroid_list[indexes[min_index]])

                pred_point = self.kf_dict[ID].f.x
                x, y = int(pred_point[0]), int(pred_point[1])
                self.kf_pred_dict[ID] = [x, y]
                
                used_ID.add(objectID)
                used_next_pts.add(indexes[min_index])
                
            unused_ID = set(objectIDs).difference(used_ID)
            unused_next_pts = set(range(len(next_centroid_list))).difference(used_next_pts)

            if unused_ID:
                for ID in unused_ID:
                    self.disappeared_dict[ID] += 1

                    #pred_point = self.kf_dict[ID].update(self.point_dict[ID][-1])
                    #x, y = int(pred_point[0]), int(pred_point[1])
                    #self.kf_pred_dict[ID] = [x, y]
                
                    if self.disappeared_dict[ID] > self.maxDisappeared:
                        self.deregister(ID)

            if unused_next_pts:
                for index in unused_next_pts:
                    #print("register_pts",next_centroid_list[index])
                    self.register(next_centroid_list[index])
            
                    
            return self.point_dict
            
        
            