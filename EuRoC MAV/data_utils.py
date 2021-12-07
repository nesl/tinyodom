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
from scipy.fft import fft

def import_euroc_mav_dataset(type_flag = 1, timestamp = True, addNoise = True, usePhysics=True, AugmentationCopies = 0, dataset_folder = 'euroc_mav/', sampling_rate = 200, window_size = 400, stride = 20, verbose=False):
    if(type_flag==1):
        type_file = 'train_full.txt'
    elif(type_flag==2):
        type_file = 'train.txt'
    elif(type_flag==3):
        type_file = 'val.txt'
    elif(type_flag==4):
        type_file = 'test.txt'
        
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
    if(usePhysics):
        Physics_Vec = np.empty([0])
    if(timestamp==True):
        ts_list = np.empty([0, window_size])
    
    with open(dataset_folder+type_file, 'r') as f:
            list_of_files = [line.strip() for line in f]
    for line in tqdm(list_of_files):
        if(verbose==True): 
            print('Processing for (file and ground truth): '+line)
            
        cur_file = pd.read_csv(dataset_folder+line+'/mav0/state_groundtruth_estimate0/data.csv')
        cur_train = cur_file[[' b_a_RS_S_x [m s^-2]',' b_a_RS_S_y [m s^-2]',' b_a_RS_S_z [m s^-2]',
           ' b_w_RS_S_x [rad s^-1]',' b_w_RS_S_y [rad s^-1]',' b_w_RS_S_z [rad s^-1]']].to_numpy()
        
        if(addNoise==True):
            cur_train = add_gaussian_noise(cur_train)
        
        if(timestamp == True):
            cur_timestamp = cur_file[['#timestamp']].to_numpy()
            ts = np.array([y-x for x,y in zip(cur_timestamp,cur_timestamp[1:])]).flatten()/1e9
            ts = np.insert(ts,0,0)
    
        cur_GT = cur_file[[' p_RS_R_x [m]',' p_RS_R_y [m]', ' p_RS_R_z [m]']].to_numpy()
    
        windows = SlidingWindow(size=window_size, stride=stride)
        cur_train_3D = windows.fit_transform(cur_train[:,0])
        for i in range(1,cur_train.shape[1]):
            X_windows = windows.fit_transform(cur_train[:,i])
            cur_train_3D = np.dstack((cur_train_3D,X_windows))
       
        if(timestamp == True):
            ts_windowed = windows.fit_transform(ts)

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

        if(usePhysics):
            loc_mat = np.empty((cur_train_3D.shape[0]))
            for i in range(cur_train_3D.shape[0]):
                acc_x =  cur_train_3D[i,:,0]
                acc_y =  cur_train_3D[i,:,1]
                acc_z =  cur_train_3D[i,:,2]
                VecSum = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
                VecSum = VecSum - np.mean(VecSum)
                FFT_VS = fft(VecSum)
                P2 = np.abs(FFT_VS/acc_x.shape[0])
                P1 = P2[0:math.ceil(acc_x.shape[0]/2)]
                P1[1:-1-2] = 2*P1[1:-1-2]
                loc_mat[i] = np.mean(P1)   
                
        size_of_each.append(cur_GT_3D.shape[0])
        X = np.vstack((X, cur_train_3D))
        x0_list.append(cur_GT[0,0])
        y0_list.append(cur_GT[0,1])
        X_orig = np.concatenate((X_orig,cur_train))
        if(usePhysics):
            Physics_Vec = np.concatenate((Physics_Vec,loc_mat))
        Y_disp = np.concatenate((Y_disp, displacement_GT_abs))
        Y_head = np.concatenate((Y_head, heading_GT))
        if(timestamp==True):
            ts_list = np.concatenate((ts_list, ts_windowed))
        Y_pos = np.vstack((Y_pos, cur_GT_3D[:,:,0:2]))
        x_vel = np.concatenate((x_vel, vx))
        y_vel = np.concatenate((y_vel, vy))
        head_s = np.concatenate((head_s,heading_s))
        head_c = np.concatenate((head_c,heading_c))
        
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
                if(usePhysics):
                    Physics_Vec = np.concatenate((Physics_Vec,loc_mat))
    if(timestamp == True and usePhysics==False):
        return  X, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, head_s, head_c, X_orig, ts_list
    elif(timestamp == False and usePhysics==True):
        return  X, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, head_s, head_c, X_orig, Physics_Vec
    elif(timestamp == True and usePhysics==True):
        return  X, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, head_s, head_c, X_orig, Physics_Vec, ts_list
    else:
        return  X, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, head_s, head_c, X_orig

def add_gaussian_noise(imu,acc_noise_std=sqrt(0.2),gyro_noise_std=sqrt(0.02)):
    acc_noise = np.random.normal(0,sqrt(0.2),imu.shape[0])
    imu[:,0] = imu[:,0] + acc_noise
    imu[:,1] = imu[:,1] + acc_noise
    imu[:,2] = imu[:,2] + acc_noise
    gyro_noise = np.random.normal(0,sqrt(0.02),imu.shape[0])
    imu[:,3] = imu[:,3] + gyro_noise
    imu[:,4] = imu[:,4] + gyro_noise
    imu[:,5] = imu[:,5] + gyro_noise  
    return imu

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
        
def long_lat_to_x_y(long_lat_mat):
    x_y_z_mat = np.zeros((long_lat_mat.shape[0],3))
    geod = Geodesic.WGS84
    lat_init = long_lat_mat[0,1]
    long_init = long_lat_mat[0,0]
    for i in range(1,long_lat_mat.shape[0]):
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
    
    
def Cal_TE(Gvx, Gvy, Pvx, Pvy, sampling_rate=100,window_size=200,stride=10,length=None):
    
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
