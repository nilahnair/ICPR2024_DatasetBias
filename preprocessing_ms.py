# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:23:29 2022
@author: fmoya
modified and reused: nilah
"""

import numpy as np
import pandas as pd
import os
from sliding_window import sliding_window
import pickle
import sys

location = "../A_DeviceMotion_data/A_DeviceMotion_data/"
NB_SENSOR_CHANNELS = 9
NUM_ACT_CLASSES= 6 #activity class
NUM_CLASSES = 24 #subject identity classes

'''sliding window size - ws
   sliding step size - ss'''

ws = 200
ss = 25

def norm_ms(data):
    """
    Normalizes all sensor channels
    Zero mean and unit variance

    @param data: numpy integer matrix
    @return data_norm: Normalized sensor data
    """

    mean_values = np.array([ 0.04213359,  0.75472223, -0.13882479,  0.00532117,  0.01458119,  0.01276031,
                            -0.00391064,  0.0442438,   0.03927177])
    mean_values = np.reshape(mean_values, [1, 9])
    std_values = np.array([0.33882991, 0.33326483, 0.42832299, 1.29291558, 1.22646988, 0.80804086,
                           0.32820886, 0.52756613, 0.37621195])
    std_values = np.reshape(std_values, [1, 9])
    
    mean_array = np.repeat(mean_values, data.shape[0], axis=0)
    std_array = np.repeat(std_values, data.shape[0], axis=0)

    max_values = mean_array + 2 * std_array
    min_values = mean_array - 2 * std_array

    data_norm = (data - min_values) / (max_values - min_values)

    data_norm[data_norm > 1] = 1
    data_norm[data_norm < 0] = 0

    return data_norm

def opp_sliding_window(data_x, data_y, label_pos_end=True):

    print("Sliding window: Creating windows {} with step {}".format(ws, ss))

    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    print(data_x.shape)
    count_l = 0
    idy=0
    # Label from the end
    if label_pos_end:
        print("check 1")
        data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    else:
        if False:
            # Label from the middle
            # not used in experiments
            print("check 2")
            data_y_labels = np.asarray(
                [[i[i.shape[0] // 2]] for i in sliding_window(data_y, ws, ss)])
        else:
            # Label according to mode
            try:
                print("check 3")
                data_y_labels = []
                for sw in sliding_window(data_y, ws, ss):
                    count_l = np.bincount(sw.astype(int), minlength=NUM_ACT_CLASSES)
                    idy = np.argmax(count_l)
                    data_y_labels.append(idy)
                data_y_labels = np.asarray(data_y_labels)
        

            except:
                print("Sliding window: error with the counting {}".format(count_l))
                return np.Inf

            # All labels per window
            data_y_all = np.asarray([i[:] for i in sliding_window(data_y, ws, ss)])
            print(data_y_all.shape)
    return data_x.astype(np.float32), data_y_labels.astype(np.uint8), data_y_all.astype(np.uint8)
    
def get_ds_infos():
    ## 0:Code, 1:Weight, 2:Height, 3:Age, 4:Gender
    dss = np.genfromtxt("../motion-sense-master/data/data_subjects_info.csv", delimiter=',') #the subject info is as provided by the dataset
    dss = dss[1:]
    print("----> Data subjects information is imported.")
    return dss

def creat_time_series(dt_list, act_labels, trial_codes, base_directory, subjects, mode="mag", labeled=True, usage_modus='trainval'):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.

    Returns:
        It returns a time-series of sensor data.
    
    """
    sel_sub=subjects
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

    if labeled:
        dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 
    else:
        dataset = np.zeros((0,num_data_cols))
        
    ds_list = get_ds_infos()
    
    if usage_modus=='trainval':
        X_train = np.empty((0, NB_SENSOR_CHANNELS))
        act_train = np.empty((0))
        id_train = np.empty((0))
    
        X_val = np.empty((0, NB_SENSOR_CHANNELS))
        act_val = np.empty((0))
        id_val = np.empty((0))
    
    elif usage_modus=='test':
        X_test = np.empty((0, NB_SENSOR_CHANNELS))
        act_test = np.empty((0))
        id_test = np.empty((0))
         
     
    print("[INFO] -- Creating Time-Series")
    for sub_id in ds_list[:,0]:
        
        if sub_id in sel_sub:
           
            for act_id, act in enumerate(act_labels):
                for trial in trial_codes[act_id]:
                    fname = location+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                    raw_data = pd.read_csv(fname)
                    raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                    vals = np.zeros((len(raw_data), num_data_cols))
                    for x_id, axes in enumerate(dt_list):
                        if mode == "mag":
                            vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        
                        else:
                            vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                        vals = vals[:,:num_data_cols]
                        
                        #normalisation of the data
                        vals = norm_ms(vals)
                    
                        frames=vals.shape[0]
                        
                        if labeled:
                            lbls = np.array([[act_id,
                            sub_id-1,
                            ds_list[int(sub_id-1)][1],
                            ds_list[int(sub_id-1)][2],
                            ds_list[int(sub_id-1)][3],
                            ds_list[int(sub_id-1)][4],
                            trial          
                           ]]*len(raw_data))
                        
                        if frames != 0 and usage_modus=='trainval':
                            train_no=round(0.70*frames)
                            val_no=round(0.15*frames)
                            tv= train_no+val_no
            
                            X_train = np.vstack((X_train, vals[0:train_no,:]))
                            act_train = np.append(act_train, [lbls[0:train_no,0]])
                            id_train = np.append(id_train, [lbls[0:train_no,1]])
                            print('done train split')
                            
                            X_val = np.vstack((X_val, vals[train_no:tv,:]))
                            act_val = np.append(act_val, [lbls[train_no:tv,0]])
                            id_val = np.append(id_val, [lbls[train_no:tv,1]])
                            print('done val split')
                        elif frames != 0 and usage_modus=='test':
                            X_test = np.vstack((X_test, vals))
                            act_test = np.append(act_test, [lbls[:,0]])
                            id_test = np.append(id_test, [lbls[:,1]])
                            print('done test split')
                    
    try: 
        if usage_modus=='trainval':
            data_train, act_train, act_all_train = opp_sliding_window(X_train, act_train, label_pos_end = False)
            data_val, act_val, act_all_val = opp_sliding_window(X_val, act_val, label_pos_end = False)
        elif usage_modus=='test':
            data_test, act_test, act_all_test = opp_sliding_window(X_test, act_test, label_pos_end = False)
    except:
        print("error in sliding window")
    
    try:
        print("training data save")
        if usage_modus=='trainval':
            print("target file name")
            print(data_dir_train)
            counter_seq = 0
            for f in range(data_train.shape[0]):
                try:
                    sys.stdout.write('\r' + 'Creating sequence file '
                                     'number {} with id {}'.format(f, counter_seq))
                    sys.stdout.flush()
                    seq = np.reshape(data_train[f], newshape = (1, data_train.shape[1], data_train.shape[2]))
                    seq = np.require(seq, dtype=np.float)                    
                    obj = {"data": seq, "label": act_train[f], "labels": act_all_train[f]}
                    f = open(os.path.join(data_dir_train, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()

                    counter_seq += 1
                except:
                    raise('\nError adding the seq')
                
            print("val data save")
            counter_seq = 0
            for f in range(data_val.shape[0]):
                try:
                    sys.stdout.write('\r' + 'Creating sequence file '
                                     'number {} with id {}'.format(f, counter_seq))
                    sys.stdout.flush()
                    seq = np.reshape(data_val[f], newshape = (1, data_val.shape[1], data_val.shape[2]))
                    seq = np.require(seq, dtype=np.float)
                    obj = {"data": seq, "label": act_val[f], "labels": act_all_val[f]}
                
                    f = open(os.path.join(data_dir_val, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()

                    counter_seq += 1
                except: 
                     raise('\nError adding the seq')
        elif usage_modus=='test':         
            print("test data save")
            counter_seq = 0
            for f in range(data_test.shape[0]):
                try:
                    sys.stdout.write('\r' + 'Creating sequence file '
                                     'number {} with id {}'.format(f, counter_seq))
                    sys.stdout.flush()
                    seq = np.reshape(data_test[f], newshape = (1, data_test.shape[1], data_test.shape[2]))
                    seq = np.require(seq, dtype=np.float)
                    obj = {"data": seq, "label": act_test[f], "labels": act_all_test[f]}
                
                    f = open(os.path.join(data_dir_test, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()

                    counter_seq += 1
                except:
                    raise('\nError adding the seq')
    except:
        print("error in saving")     
    
    return #dataset

def generate_CSV(csv_dir, data_dir):
    '''
    Generate CSV file with path to all (Training) of the segmented sequences
    This is done for the DATALoader for Torch, using a CSV file with all the paths from the extracted
    sequences.

    @param csv_dir: Path to the dataset
    @param data_dir: Path of the training data
    '''
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for n in range(len(filenames)):
            f.append(data_dir + 'seq_{0:06}.pkl'.format(n))

    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')
    
    return

def generate_CSV_final(csv_dir, data_dir1, data_dir2):
    '''
    Generate CSV file with path to all (Training and Validation) of the segmented sequences
    This is done for the DATALoader for Torch, using a CSV file with all the paths from the extracted
    sequences.

    @param csv_dir: Path to the dataset
    @param data_dir1: Path of the training data
    @param data_dir2: Path of the validation data
    '''
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir1):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:06}.pkl'.format(n))

    for dirpath, dirnames, filenames in os.walk(data_dir2):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:06}.pkl'.format(n))

    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')

    return

if __name__ == '__main__':
    datatype = {0:"A_DeviceMotion_data", 1:"B_Accelerometer_data", 2:"C_Gyroscope_data"}
    ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
    TRIAL_CODES = {
        ACT_LABELS[0]:[1,2,11],
        ACT_LABELS[1]:[3,4,12],
        ACT_LABELS[2]:[7,8,15],
        ACT_LABELS[3]:[9,16],
        ACT_LABELS[4]:[6,14],
        ACT_LABELS[5]:[5,13]}
    dt_list = []
    sensors=['gravity', 'rotationRate', 'userAcceleration']
    for t in sensors:
        dt_list.append([t+".x",t+".y",t+".z"])
        
    act_labels = ACT_LABELS [0:6]
    
    print("[INFO] -- Selected activites: "+str(act_labels))    
    trial_codes = [TRIAL_CODES[act] for act in act_labels]
    
    '''base directory is where you would like to save the preprocessed files. 
    note that within the base directory, one must create folders with the names:
    -sequences_train
    -sequences_test
    -sequences_val'''

    base_directory = '/prepros/exp5/'
    
    #sel_subjects_train=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    sel_subjects_train=[5,10,19,23]
    sel_subjects_test=[1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 24]
    
    
    creat_time_series(dt_list, act_labels, trial_codes, base_directory=base_directory, subjects=sel_subjects_train, mode="raw", labeled=True, usage_modus='trainval')
    creat_time_series(dt_list, act_labels, trial_codes, base_directory=base_directory, subjects=sel_subjects_test, mode="raw", labeled=True, usage_modus='test')
    
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    
    generate_CSV(base_directory + "train.csv", data_dir_train)
    generate_CSV(base_directory + "val.csv", data_dir_val)
    generate_CSV(base_directory + "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)
  
    print("Done")
   
    
    
    
    



