import cv2
import numpy as np
import os
import math
from scipy.spatial import distance as dist
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise



def fx(x, dt):
    # state transition function - predict next state based
    # on constant velocity model x = vt + x_0

    u_x, u_y = 0, 1

    F = np.array([[1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])


    B = np.array([[(dt**2)/2, 0],
                    [0, (dt**2)/2],
                    [dt,0],
                    [0,dt]])

    u = np.array([[u_x],[u_y]]) 
    
    a = np.dot(B,u).reshape(1,4)

    return np.dot(F, x) + a

def hx(x):
    # measurement function - convert state into a measurement
    # where measurements are [x_pos, y_pos]

    return x[[0, 1]]

class UK_filter():

    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas, init_x, init_y):  
    
        self.init_x = init_x 
        self.init_y = init_y
        self.dt = dt
        self.z_std = 0.1

        self.points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)
        self.f = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=self.dt, fx=fx, hx=hx, points=self.points)

        self.f.x = np.array([self.init_x,0,self.init_y,0])


        self.f.P = np.eye(4)

        """self.f.Q *= np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2"""

        self.f.Q = Q_discrete_white_noise(2, dt = self.dt, var = 0.01**2, block_size = 2)

        self.f.R = np.array([[x_std_meas**2,0],
                            [0, y_std_meas**2]])

        self.f.predict()



class Trajectory_ukf:
    def __init__(self, maxDisappeared = 10):

        self.nextObjectID = 0
        self.point_dict = OrderedDict()
        self.disappeared_dict = OrderedDict()
        self.kf_dict = OrderedDict()
        self.kf_pred_dict = OrderedDict()
        self.maxDisappeared = maxDisappeared


    def register(self, centroid):
        self.point_dict[self.nextObjectID] = [centroid]
        self.disappeared_dict[self.nextObjectID] = 0
        self.kf_dict[self.nextObjectID] = UK_filter(dt = 1, 
                                                        u_x = 0, 
                                                        u_y = 0, 
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

                self.kf_dict[ID].f.predict()
                
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
                
                pred_point = self.kf_dict[ID].f.x

                x, y = int(pred_point[0]), int(pred_point[1])
                self.kf_pred_dict[ID] = [x, y]
                self.kf_predict_list.append([x, y])


            distan = dist.cdist(np.array(self.kf_predict_list), next_centroid_list)
            

            ID_list, indexes = linear_sum_assignment(distan)
            
            next_ball_point = next_centroid_list[indexes[0]]


            self.point_dict[objectIDs[0]].append(next_ball_point)
            self.kf_dict[objectIDs[0]].f.update(next_ball_point)
            self.kf_dict[objectIDs[0]].f.predict()



        return self.point_dict


        


