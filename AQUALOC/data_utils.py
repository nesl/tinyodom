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
from ahrs.filters import Madgwick
from ahrs import Quaternion
from scipy.signal import resample
from scipy.fft import fft

def import_aqualoc_dataset(type_flag = 1, usePhysics=True, dataset_folder = 'aqualoc/', useMagnetometer = True, returnOrientation = True, emulateDVL=True, AugmentationCopies = 0, sampling_rate = 200, resampling_rate = 10, window_size = 400, stride = 20, verbose=False):
    if(type_flag==1):
        type_file = 'train_full.txt'
    elif(type_flag==2):
        type_file = 'train.txt'
    elif(type_flag==3):
        type_file = 'valid.txt'
    elif(type_flag==4):
        type_file = 'test.txt'
        
    if(useMagnetometer):
        channel_count = 9
    else:
        channel_count = 6  
        
    X_orig = np.empty([0,channel_count])
    x0_list = []
    y0_list = []
    if(returnOrientation):
        X_Orientation = np.empty([0, window_size, 3])
        Y_Orientation = np.empty([0, window_size, 3])
    if(emulateDVL):
        DVL_X = np.empty([0])
        DVL_Y = np.empty([0])
        DVL_Z = np.empty([0])
    if(usePhysics):
        Physics_Vec = np.empty([0])
    size_of_each = []
    X = np.empty([0, window_size, channel_count])
    Y_disp = np.empty([0])
    Y_head = np.empty([0])
    Y_pos = np.empty([0,window_size, 2])
    x_vel = np.empty([0])
    y_vel = np.empty([0])
    z_vel = np.empty([0])
    head_s = np.empty([0])
    head_c = np.empty([0])
    
    with open(dataset_folder+type_file, 'r') as f:
            list_of_files = [line.strip() for line in f]
    for line in tqdm(list_of_files):
        if(verbose==True): 
            print('Processing for (file and ground truth): '+line)
            
        cur_file = pd.read_csv(glob.glob(dataset_folder+line+'/raw_data/*imu*.csv')[0])
        cur_train = cur_file[['a_RS_S_x [m s^-2]','a_RS_S_y [m s^-2]','a_RS_S_z [m s^-2]',
                      'w_RS_S_x [rad s^-1]','w_RS_S_y [rad s^-1]','w_RS_S_z [rad s^-1]']].to_numpy()
                               
        if(useMagnetometer):
            cur_mag = pd.read_csv(glob.glob(dataset_folder+line+'/raw_data/*mag*.csv')[0])
            cur_mag_train = cur_mag[[' b_x',' b_y',' b_z']].to_numpy()
            if(cur_mag_train.shape[0] > cur_train.shape[0]):
                cur_mag_train = cur_mag_train[0:cur_train.shape[0],:]
            elif(cur_mag_train.shape[0] < cur_train.shape[0]):
                extra = np.repeat([cur_mag_train[-1,0],cur_mag_train[-1,1],cur_mag_train[-1,2]],cur_train.shape[0]-cur_mag_train.shape[0]).reshape(cur_train.shape[0]-cur_mag_train.shape[0],3)
                cur_mag_train = np.concatenate((cur_mag_train,extra))
            cur_train = np.concatenate((cur_train,cur_mag_train),axis=1)
            
        gt_file = pd.read_csv(glob.glob(dataset_folder+line+'/*colmap*.txt')[0],header=None,delimiter=' ')
        cur_img = pd.read_csv(glob.glob(dataset_folder+line+'/raw_data/*img*.csv')[0])
        idx = []
        for item in cur_img.iloc[np.int_(np.array(gt_file[0])),0].to_numpy():
            idx.append((np.abs(cur_file['#timestamp [ns]'].to_numpy()-item)).argmin())
        
        m = np.zeros((len(cur_file),3))
        j = 0
        for i in tqdm(range(len(cur_file)),position=0,leave=True):
            if i in idx:
                m[i,:] = gt_file.iloc[j,1:4]
                j = j+1
            else:
                m[i,:] = np.nan
        
        cur_GT = np.zeros((len(cur_file),3))
        for i in range(0,m.shape[1]):
            time_series = pd.Series(m[:,i])
            cur_GT[:,i] = savgol_filter(time_series.interpolate(method="linear").ffill().bfill(), 
                                               window_length=5001, polyorder=2)
        
        if(returnOrientation):
            X_Ori = Madgwick(gyr = cur_train[:,3:6],acc=cur_train[:,0:3]).Q
            Euler_ang = np.zeros((X_Ori.shape[0],3))
            for i in range(X_Ori.shape[0]):
                cur_quat = Quaternion(X_Ori[i,:])
                Euler_ang[i,:] = cur_quat.to_angles()

            m = np.zeros((len(cur_file),4))
            j = 0
            for i in tqdm(range(len(cur_file)),position=0,leave=True):
                if i in idx:
                    m[i,:] = gt_file.iloc[j,4:8]
                    j = j+1
                else:
                    m[i,:] = np.nan
        
            cur_GT_Ori = np.zeros((len(cur_file),4))
            for i in range(0,m.shape[1]):
                time_series = pd.Series(m[:,i])
                cur_GT_Ori[:,i] = savgol_filter(time_series.interpolate(method="linear").ffill().bfill(), 
                                               window_length=5001, polyorder=2)
        
        
            Roll, Pitch, Yaw = quaternion_to_euler_angle(cur_GT_Ori[:,3],cur_GT_Ori[:,0],cur_GT_Ori[:,1],cur_GT_Ori[:,2])
            cur_GT_Ori =np.concatenate((Roll.reshape(Roll.shape[0],1),Pitch.reshape(Yaw.shape[0],1),Yaw.reshape(Yaw.shape[0],1)),axis=1)
        
        if(resampling_rate>0):
            cur_train_new = np.zeros((math.ceil((cur_train.shape[0]/sampling_rate)*resampling_rate),cur_train.shape[1]))
            cur_GT_new = np.zeros((math.ceil((cur_GT.shape[0]/sampling_rate)*resampling_rate),cur_GT.shape[1]))
            cur_GT_Ori_new = np.zeros((math.ceil((cur_GT_Ori.shape[0]/sampling_rate)*resampling_rate),cur_GT_Ori.shape[1]))
            Euler_ang_new = np.zeros((math.ceil((Euler_ang.shape[0]/sampling_rate)*resampling_rate),Euler_ang.shape[1]))
            
            for i in range(cur_train.shape[1]):
                cur_train_new[:,i] = resample(cur_train[:,i],math.ceil((cur_train.shape[0]/sampling_rate)*resampling_rate))
            for i in range(cur_GT.shape[1]):
                cur_GT_new[:,i] = resample(cur_GT[:,i],math.ceil((cur_GT.shape[0]/sampling_rate)*resampling_rate))
                cur_GT_Ori_new[:,i] = resample(cur_GT_Ori[:,i],math.ceil((cur_GT_Ori.shape[0]/sampling_rate)*resampling_rate))
                Euler_ang_new[:,i]= resample(Euler_ang[:,i],math.ceil((Euler_ang.shape[0]/sampling_rate)*resampling_rate))
                
            cur_train = cur_train_new
            cur_GT = cur_GT_new
            cur_GT_Ori = cur_GT_Ori_new
            Euler_ang = Euler_ang_new
            
        windows = SlidingWindow(size=window_size, stride=stride)
        cur_train_3D = windows.fit_transform(cur_train[:,0])
        for i in range(1,cur_train.shape[1]):
            X_windows = windows.fit_transform(cur_train[:,i])
            cur_train_3D = np.dstack((cur_train_3D,X_windows))
                    
        cur_GT_3D = windows.fit_transform(cur_GT[:,0])
        for i in range(1,cur_GT.shape[1]):
            X_windows = windows.fit_transform(cur_GT[:,i])
            cur_GT_3D = np.dstack((cur_GT_3D,X_windows))
            
        if(returnOrientation):    
            cur_GT_Ori_3D = windows.fit_transform(cur_GT_Ori[:,0])
            for i in range(1,cur_GT_Ori.shape[1]):
                X_windows = windows.fit_transform(cur_GT_Ori[:,i])
                cur_GT_Ori_3D = np.dstack((cur_GT_Ori_3D,X_windows))
            
            Euler_ang_3D = windows.fit_transform(Euler_ang[:,0])
            for i in range(1,cur_GT_Ori.shape[1]):
                X_windows = windows.fit_transform(Euler_ang[:,i])
                Euler_ang_3D = np.dstack((Euler_ang_3D,X_windows))
            
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
        vz = np.zeros((cur_GT_3D.shape[0]))
        prev = 0
        for i in range(cur_GT_3D.shape[0]): 
            Xdisp = (cur_GT_3D[i,-1,0]-cur_GT_3D[i,0,0])
            vx[i] = Xdisp
            Ydisp = (cur_GT_3D[i,-1,1]-cur_GT_3D[i,0,1])
            vy[i] = Ydisp
            Zdisp = (cur_GT_3D[i,-1,2]-cur_GT_3D[i,0,2])
            vz[i] = Zdisp
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
        
        if(emulateDVL):
            X_DVL = vx + np.random.normal(0,sqrt(0.2),vx.shape[0])
            Y_DVL = vy + np.random.normal(0,sqrt(0.2),vy.shape[0])
            Z_DVL = vz + np.random.normal(0,sqrt(0.2),vz.shape[0])

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
                
        X = np.vstack((X, cur_train_3D))
        if(returnOrientation):
            X_Orientation = np.vstack((X_Orientation, Euler_ang_3D))
            Y_Orientation = np.vstack((Y_Orientation, cur_GT_Ori_3D))    
        X_orig = np.concatenate((X_orig,cur_train))
        if(usePhysics):
            Physics_Vec = np.concatenate((Physics_Vec,loc_mat))
        Y_disp = np.concatenate((Y_disp, displacement_GT_abs))
        Y_head = np.concatenate((Y_head, heading_GT))
        Y_pos = np.vstack((Y_pos, cur_GT_3D[:,:,0:2]))
        x0_list.append(cur_GT[0,0])
        y0_list.append(cur_GT[0,1])
        size_of_each.append(cur_GT_3D.shape[0])
        x_vel = np.concatenate((x_vel, vx))
        y_vel = np.concatenate((y_vel, vy))
        z_vel = np.concatenate((z_vel, vz))
        if(emulateDVL):
            DVL_X = np.concatenate((DVL_X, X_DVL))
            DVL_Y = np.concatenate((DVL_Y, Y_DVL))
            DVL_Z = np.concatenate((DVL_Z, Z_DVL))            
        head_s = np.concatenate((head_s,heading_s))
        head_c = np.concatenate((head_c,heading_c)) 
        if(AugmentationCopies>0):
            for i in range(AugmentationCopies):
                out = random_rotate(cur_train_3D, useMagnetometer)
                X = np.vstack((X, out))
                if(returnOrientation):
                    X_Orientation = np.vstack((X_Orientation, Euler_ang_3D))
                    Y_Orientation = np.vstack((Y_Orientation, cur_GT_Ori_3D))
                if(emulateDVL):
                    DVL_X = np.concatenate((DVL_X, X_DVL))
                    DVL_Y = np.concatenate((DVL_Y, Y_DVL))
                    DVL_Z = np.concatenate((DVL_Z, Z_DVL)) 
                X_orig = np.concatenate((X_orig,cur_train))
                Y_disp = np.concatenate((Y_disp, displacement_GT_abs))
                Y_head = np.concatenate((Y_head, heading_GT))
                Y_pos = np.vstack((Y_pos, cur_GT_3D[:,:,0:2]))
                x0_list.append(cur_GT[0,0])
                y0_list.append(cur_GT[0,1])
                size_of_each.append(cur_GT_3D.shape[0])
                x_vel = np.concatenate((x_vel, vx))
                y_vel = np.concatenate((y_vel, vy))
                z_vel = np.concatenate((z_vel, vz))
                head_s = np.concatenate((head_s,heading_s))
                head_c = np.concatenate((head_c,heading_c))
                if(usePhysics):
                    Physics_Vec = np.concatenate((Physics_Vec,loc_mat))
    
    #can return:
    #1. Windowed IMU Data 2. X Orientation (Windowed) 3. Y Orientation (Windowed) 4, 5, 6. Doppler Velocity Log Emulation X,Y,Z
    #7. ground truth displacements 8. ground truth heading rates 9. ground truth position
    # 10. list of initial x positions 11. list of initial y positions 12. size of each file in windowed form
    # 13. ground truth x velocity 14. ground truth y velocity 15. heading rate in terms of sin 16. heading rate in terms of cos
    # 17. unwindowed training set
    if(returnOrientation==True and emulateDVL==True and usePhysics==False):
        return  X, X_Orientation, Y_Orientation, DVL_X, DVL_Y, DVL_Z, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, z_vel, head_s, head_c, X_orig
    elif(returnOrientation==True and emulateDVL==False and usePhysics==False):
        return  X, X_Orientation, Y_Orientation, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, z_vel, head_s, head_c, X_orig  
    elif(returnOrientation==False and emulateDVL==True and usePhysics==False):
        return  X, DVL_X, DVL_Y, DVL_Z, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, z_vel, head_s, head_c, X_orig 
    
    if(returnOrientation==True and emulateDVL==True and usePhysics==True):
        return  X, X_Orientation, Y_Orientation, DVL_X, DVL_Y, DVL_Z, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, z_vel, head_s, head_c, X_orig, Physics_Vec
    elif(returnOrientation==True and emulateDVL==False and usePhysics==True):
        return  X, X_Orientation, Y_Orientation, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, z_vel, head_s, head_c, X_orig, Physics_Vec
    elif(returnOrientation==False and emulateDVL==True and usePhysics==True):
        return  X, DVL_X, DVL_Y, DVL_Z, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, z_vel, head_s, head_c, X_orig, Physics_Vec 
    elif(returnOrientation==False and emulateDVL==False and usePhysics==True):
        return  X, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, z_vel, head_s, head_c, X_orig, Physics_Vec
    else:
        return  X, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, z_vel, head_s, head_c, X_orig

    
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
 
def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z #roll, pitch, yaw
    
    
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
