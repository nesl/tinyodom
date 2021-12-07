import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import savgol_filter
from tqdm import tqdm
import os
import math


def import_aqualoc_pressure(dataset_folder = 'aqualoc/',verbose=False,file_idx=1):
    
    type_file = 'cur_file.txt'
    with open(dataset_folder+type_file, 'r') as f:
            list_of_files = [line.strip() for line in f]
    for line in list_of_files[file_idx:file_idx+1]:
        if(verbose==True): 
            print('Processing for (file and ground truth): '+line)
            
        cur_file = pd.read_csv(glob.glob(dataset_folder+line+'/raw_data/*depth*.csv')[0])
        print(cur_file.columns[0],cur_file.columns[1])
        cur_train = cur_file[[cur_file.columns[0], cur_file.columns[1]]].to_numpy()   
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
                                               window_length=501, polyorder=2)
        
        
        #returns raw pressure readings and ground truth depth
      
        return  cur_train, cur_GT[:,2]

