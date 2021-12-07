import pandas as pd
import numpy as np
from gtda.time_series import SlidingWindow
import matplotlib.pyplot as plt
from math import atan2, pi, sqrt, atan2, sin, cos, radians
import h5py
import json
import glob
from scipy.signal import savgol_filter
from geographiclib.geodesic import Geodesic
from tqdm import tqdm
import os
import quaternion
import math
import geometry_helpers
from pydometer import Pedometer


def import_gundog_dataset(type_flag = 1,useStepCounter = True, dataset_folder = 'Gundog/', AugmentationCopies = 0,sampling_rate = 40, window_size = 10, stride = 2, verbose=False):
        
    x0_list = []
    y0_list = []
    size_of_each = []
    X_orig = np.empty([0,6])
    X = np.empty([0, window_size, 6])
    Y_disp = np.empty([0])
    Y_head = np.empty([0])
    Y_pos = np.empty([0,window_size, 2])
    x_vel = np.empty([0])
    y_vel = np.empty([0])
    head_s = np.empty([0])
    head_c = np.empty([0])
    
    if(useStepCounter):
        loc_3D_mat = np.empty([0,window_size])
        
    if(type_flag==1):
        z = pd.read_csv(dataset_folder+'Train/40317_2021_245_MOESM6_ESM.txt',delimiter='\t')
    else:
        z = pd.read_csv(dataset_folder+'Test/Test.Data.P10A_selection_split#1_0.txt',delimiter='\t')
    acc = z[['Acc_x','Acc_y','Acc_z']].to_numpy()
    mag = z[['Mag_x','Mag_y','Mag_z']].to_numpy()
    cur_train = np.concatenate((acc,mag),axis=1)
    
    if(useStepCounter):
        p = Pedometer(gx=z['Acc_x'].to_numpy(), gy=z['Acc_y'].to_numpy(), gz=z['Acc_z'].to_numpy(), sr=sampling_rate)
        step_count, step_locations = p.get_steps()
        loc = np.zeros(cur_train.shape[0])
        loc[step_locations] = 1
   
    GPS = z[['GPS Longitude','GPS Latitude']].to_numpy()
    mis = z['GPS Latitude'].to_numpy().nonzero()
    GPS = GPS[mis]
    gt_file = long_lat_to_x_y(GPS)        
        
    m = np.zeros((len(cur_train),3))
    j = 0
    for i in tqdm(range(len(cur_train))):
        if i in mis[0]:
            m[i,:] = gt_file[j,:]
            j = j+1
        else:
            m[i,:] = np.nan
        
    cur_GT = np.zeros((len(cur_train),3))
    for i in range(0,m.shape[1]):
        time_series = pd.Series(m[:,i])
        cur_GT[:,i] = savgol_filter(time_series.interpolate(method="linear").ffill().bfill(), 
                                           window_length=101, polyorder=2)
    
    windows = SlidingWindow(size=window_size, stride=stride)
    cur_train_3D = windows.fit_transform(cur_train[:,0])
    for i in range(1,cur_train.shape[1]):
        X_windows = windows.fit_transform(cur_train[:,i])
        cur_train_3D = np.dstack((cur_train_3D,X_windows))  

    if(useStepCounter):   
        loc_3D = windows.fit_transform(loc)
        
    cur_GT_3D = windows.fit_transform(cur_GT[:,0])
    for i in range(1,cur_GT.shape[1]):
        X_windows = windows.fit_transform(cur_GT[:,i])
        cur_GT_3D = np.dstack((cur_GT_3D,X_windows))
        
    heading_s = np.zeros((cur_GT_3D.shape[0]))
    heading_c = np.zeros((cur_GT_3D.shape[0]))
    for i in range(cur_GT_3D.shape[0]):    
        s,c = abs_heading_sin_cos(cur_GT_3D[i,-1,0],cur_GT_3D[i,-1,1],cur_GT_3D[i,0,0],cur_GT_3D[i,0,1])
        heading_s[i] = s
        heading_c[i] = c

    displacement_GT_abs = np.zeros(cur_GT_3D.shape[0])
    heading_GT = np.zeros((cur_GT_3D.shape[0]))
    vx = np.zeros((cur_GT_3D.shape[0]))
    vy = np.zeros((cur_GT_3D.shape[0]))
    prev = 0
    for i in range(cur_GT_3D.shape[0]):  
        Xdisp = (cur_GT_3D[i,-1,0]-cur_GT_3D[i,0,0])
        vx[i] = Xdisp
        Ydisp = (cur_GT_3D[i,-1,1]-cur_GT_3D[i,0,1])
        vy[i] = Ydisp
        displacement_GT_abs[i] = sqrt((Xdisp**2) + (Ydisp**2))   
        theta = abs_heading(cur_GT_3D[i,-1,0],cur_GT_3D[i,-1,1],cur_GT_3D[i,0,0],cur_GT_3D[i,0,1])
        if theta<180:
            theta = theta + 180

        heading_GT[i] = theta - prev
        if(heading_GT[i]>100 or heading_GT[i]<-100):
            theta2 = theta
            prev2 = prev
            if theta<prev:
                theta2 = theta + 360
            else:
                prev2 =  prev + 360
            heading_GT[i] = theta2 - prev2 
        prev = theta
                
    size_of_each.append(cur_GT_3D.shape[0])
    X = np.vstack((X, cur_train_3D))
    x0_list.append(cur_GT[0,0])
    y0_list.append(cur_GT[0,1])
    X_orig = np.concatenate((X_orig,cur_train))
    Y_disp = np.concatenate((Y_disp, displacement_GT_abs))
    Y_head = np.concatenate((Y_head, heading_GT))
    Y_pos = np.vstack((Y_pos, cur_GT_3D[:,:,0:2]))
    x_vel = np.concatenate((x_vel, vx))
    y_vel = np.concatenate((y_vel, vy))
    head_s = np.concatenate((head_s,heading_s))
    head_c = np.concatenate((head_c,heading_c))
    

    if(useStepCounter):
        loc_3D_mat = np.vstack((loc_3D_mat,loc_3D))
        
    if(AugmentationCopies>0):
        for i in range(AugmentationCopies):
            out = random_rotate(cur_train_3D, useMagnetometer=False)
            X = np.vstack((X, out))
            X_orig = np.concatenate((X_orig,cur_train))
            Y_disp = np.concatenate((Y_disp, displacement_GT_abs))
            Y_head = np.concatenate((Y_head, heading_GT))
            if(timestamp==True):
                ts_list = np.concatenate((ts_list, ts_windowed))
            Y_pos = np.vstack((Y_pos, cur_GT_3D[:,:,0:2]))
            x0_list.append(cur_GT[0,0])
            y0_list.append(cur_GT[0,1])
            size_of_each.append(cur_GT_3D.shape[0])
            x_vel = np.concatenate((x_vel, vx))
            y_vel = np.concatenate((y_vel, vy))
            head_s = np.concatenate((head_s,heading_s))
            head_c = np.concatenate((head_c,heading_c))
            if(useStepCounter):
                loc_3D_mat = np.vstack((loc_3D_mat,loc_3D)) 

    if(useStepCounter):
        X = np.concatenate((X,loc_3D_mat.reshape(loc_3D_mat.shape[0],loc_3D_mat.shape[1],1)),axis=2)
    
    #returns 1. training set from IMU 2. ground truth displacements 3. ground truth heading rates 4. ground truth position
    # 5. list of initial x positions 6. list of initial y positions 7. size of each file in windowed form
    # 8. ground truth x velocity 9. ground truth y velocity 10. heading rate in terms of sin 11. heading rate in terms of cos
    # 12. unwindowed training set from IMU
    
    return  X, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, head_s, head_c, X_orig 
    
    
    
def long_lat_to_x_y(long_lat_mat):
    x_y_z_mat = np.zeros((long_lat_mat.shape[0],3))
    geod = Geodesic.WGS84
    lat_init = long_lat_mat[0,1]
    long_init = long_lat_mat[0,0]
    for i in tqdm(range(1,long_lat_mat.shape[0])):
        g = geod.Inverse(lat_init,long_init,long_lat_mat[i,1],long_lat_mat[i,0])
        x_y_z_mat[i,0] = g['s12']*np.cos(np.abs(radians(g['azi1'])))
        x_y_z_mat[i,1] = g['s12']*np.sin(np.abs(radians(g['azi1'])))
    return x_y_z_mat


def random_rotate(input,useMagnetometer=True):
    output = np.copy(input)
    euler = np.random.uniform(0, np.pi, size=3)
    for i in range(0, input.shape[0]):
        input_acc = input[i,:,0:3]
        input_rot = input[i,:,3:6]
        if(useMagnetometer):
            input_mag = input[i,:,6:9]  
        Rot = geometry_helpers.euler2mat(euler[0],euler[1],euler[2])
        output_acc = np.dot(Rot, input_acc.T).T
        output_rot = np.dot(Rot, input_rot.T).T
        if(useMagnetometer):
            output_mag = np.dot(Rot, input_mag.T).T
            output[i,:,:] = np.hstack((output_acc, output_rot, output_mag))  
        else:
            output[i,:,:] = np.hstack((output_acc, output_rot))    
    return output

def orientation_to_angles(ori):
    if ori.dtype != quaternion.quaternion:
        ori = quaternion.from_float_array(ori)

    rm = quaternion.as_rotation_matrix(ori)
    angles = np.zeros([ori.shape[0], 3])
    angles[:, 0] = adjust_angle_array(np.arctan2(rm[:, 0, 1], rm[:, 1, 1]))
    angles[:, 1] = adjust_angle_array(np.arcsin(-rm[:, 2, 1]))
    angles[:, 2] = adjust_angle_array(np.arctan2(-rm[:, 2, 0], rm[:, 2, 2]))

    return angles


def adjust_angle_array(angles):
    new_angle = np.copy(angles)
    angle_diff = angles[1:] - angles[:-1]

    diff_cand = angle_diff[:, None] - np.array([-math.pi * 4, -math.pi * 2, 0, math.pi * 2, math.pi * 4])
    min_id = np.argmin(np.abs(diff_cand), axis=1)

    diffs = np.choose(min_id, diff_cand.T)
    new_angle[1:] = np.cumsum(diffs) + new_angle[0]
    return new_angle


def abs_heading(cur_x, cur_y, prev_x, prev_y):
        dely = (cur_y - prev_y)
        delx = (cur_x - prev_x)
        delh= atan2(delx,dely)*57.2958
        return delh
    
def abs_heading_sin_cos(cur_x, cur_y, prev_x, prev_y):
        dely = (cur_y - prev_y)
        delx = (cur_x - prev_x)
        sqr = np.sqrt(dely*dely + delx*delx)
        s = dely/sqr
        c = delx/sqr
        return s,c
    
def Cal_TE(Gvx, Gvy, Pvx, Pvy, sampling_rate=40,window_size=10,stride=2,length=None):
    
    if length==None:
        length = len(Gvx)
        
    distance = []
    
    for i in range(length):
        d = ((Gvx[i]-Pvx[i])*(Gvx[i]-Pvx[i])) + ((Gvy[i]-Pvy[i])*(Gvy[i]-Pvy[i]))
        d = math.sqrt(d)
        distance.append(d)
    
    mean_distance = sum(distance)/len(distance)
    ate = mean_distance
    at_all = distance
    
    n_windows_one_min= int(((sampling_rate*60)-window_size)/stride)
    distance = []
    if(n_windows_one_min < length):
        for i in range(n_windows_one_min):
            d = ((Gvx[i]-Pvx[i])*(Gvx[i]-Pvx[i])) + ((Gvy[i]-Pvy[i])*(Gvy[i]-Pvy[i]))
            d = math.sqrt(d)
            distance.append(d)
        rte = sum(distance)/len(distance)
    else:
        rte=ate*(n_windows_one_min/length)
    
    rt_all = distance
    return ate, rte, at_all, rt_all

def Cal_len_meters(Gvx, Gvy, length=None):
    if length==None:
        length = len(Gvx)
        
    distance = []
    
    for i in range(1, length):
        d = ((Gvx[i]-Gvx[i-1])*(Gvx[i]-Gvx[i-1])) + ((Gvy[i]-Gvy[i-1])*(Gvy[i]-Gvy[i-1]))
        d = math.sqrt(d)
        distance.append(d)
    
    sum_distance = sum(distance)
    
    return sum_distance    
    