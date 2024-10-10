import os
import re
import sys

import time

# data processing packages
import hdf5storage
import pandas as pd
import numpy as np
import neo
# signal processing packages

from scipy import signal
from scipy.signal import butter, filtfilt, decimate
from scipy.io import loadmat
from scipy.stats import zscore

# future engeneering packages

import pyfftw
from scipy.stats import kurtosis       # kurtosis function
from scipy.stats import skew           # skewness function

# config file has all the processing parameters for the data files
import config
from IPython.display import display
import gc
import matplotlib.pyplot as plt


def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    # imbr
    # https://stackoverflow.com/questions/3160699/python-progress-bar/26761413#26761413
    count = len(it)
    start = time.time() # time estimate start
    def show(j):
        x = int(size*j/count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)        
        mins, sec = divmod(remaining, 60) # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        dh_progress.update(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}")
    show(0.1) # avoid div/0 
    for i, item in enumerate(it):
        yield item
        show(i+1)

# This function goes through nested file folders and return all the files as a list
# THe list is formatted as [complete_path, annotation type folder name, file_name]
def return_file_list_from_server (server):
    file_list=[]
    folderlist_server = os.listdir(server)  # get cohort folders
    
    # get all subfolders for annotation types in '/NoNo/, /NoSz/, /SlNo/, /SlSl/ format from Spike2 export'
    for folder in folderlist_server[::-1]:
        # get the type level folder
        types=os.listdir(server+"/"+folder)
        for annotation_type in types[::-1]:
            type_folder=server+"/"+folder+"/"+annotation_type
            type_folder_file_list=os.listdir(type_folder)
            for mat_file in type_folder_file_list:
                # print  (mat_file)
                file_list.append([server+"/"+folder+"/"+annotation_type+"/"+mat_file, annotation_type, mat_file])
    # total count
    return file_list        
    
 # Optional Funciton that prints out names of the folders and files found:
def describe_the_files(files_found):
    print('  ')
    print("Total files found {}".format(len(files_found)))
    print("   ")
    recording_formats=set()
    recording_names=[]
    for recording in files_found:
        if recording[1] not in recording_formats:
            recording_formats.add(recording[1])
        recording_names.append(recording[2])

    print('Recording formats found {0}: {1}, '.format(len(recording_formats),recording_formats))
    print("    ")
    #print('Complete list of the files found: \n{}'.format("\n".join(recording_names)))

# This function uploads matlab files to numpy array using hdf5storage package
# Unused channelas are removed and a dicitonary of all channels and scoring are returned
def download_new_file(file_name):
    file_dict=hdf5storage.loadmat(file_name)
    keys_to_remove = ['Keyboard','Racine','file']
    new_dict = {k: v for k, v in file_dict.items() if k not in keys_to_remove}
    return new_dict

# This function takes name of the recorded channel and compressees the data 10 times using 
# forward-backward filtering wit parameters  a, b stored in the config file
# And normalizes the data using z-score (if ScoreAtDecimation is labeled as True in the config file)
def decimate_channel(k, dict):
    channel_data=np.array(dict[k]['values'][0],dtype=np.float32)
    channel_data=signal.filtfilt(config.b,config.a,channel_data,axis=0)
    if config.zScoreAtDecimation==True:
        channel_data=(channel_data-np.mean(channel_data))/np.std(channel_data)

    if 'ECog' in k:
        order=2
    else:
        order=8
    
    dec_record=np.array(signal.decimate(channel_data, 10, n=order, ftype='iir',axis=0,zero_phase=True),dtype=np.float32)

    return dec_record

# This function goes through all the values in the dicitonary, applies decimate function
# And returns a numpy array for all channels in orfder specified by "channels" list in conifg file

def decimate_all_channels(new_array):
    dec_dict={}
    dh_keys.update(f'Keys: {new_array.keys()}')

    for k in new_array.keys():
        for c in config.channels:
            if  c in k: 
                new_key="dec"+c
                dec_dict[new_key]=decimate_channel(k, new_array)
                
    min_channel_samples=len(min(dec_dict.values(), key=len))

    flattened_arrays = [np.array(value).flatten()[:min_channel_samples] for value in dec_dict.values()]
    
    
    l=np.array(flattened_arrays)
    return l

#Ths function reshapes the aray creating 2160 epochs with 4000 recordings (20s) each
def create_epochs(dec_data):
    
    axis=-1
    l=np.array(dec_data)
    # Calculate the number samples in whole epochs of epoch_length in the given file
    complete_epochs=config.epoch_samples_dec*np.floor(l.shape[1]/(config.epoch_samples_dec)).astype(int)
    # Trim any samples over this whole number
    strict_epochs=l[:, :complete_epochs]
    # Z-score the decimated data
    strict_epochs=zscore(strict_epochs,axis=-1)

    # Calculate the number of epochs of epoch_length in this file
    n_epochs=int(complete_epochs/config.epoch_samples_dec)

    # reshape the array to Epochs, Channels, Samples order
    c_arr = strict_epochs[:, :n_epochs*config.epoch_samples_dec].reshape(4, n_epochs, config.epoch_samples_dec).transpose(1, 0, 2)
    dh_progress.update(f'{c_arr.shape}')
    return c_arr

# This function looks for EMG channel (specified as second column in the array)
# And creates RMS function in 1 s intervals
def calculate_EMG_RMS(decimated_array_data):
    
    
# This is the function that creates RMS values for EMG dataframe
    EMG_epochs=decimated_array_data[:,1, :]
    bins=config.epoch_length                               # number of 1 second bins to use for RMS
    EMG_window=int(config.epoch_samples_dec/config.epoch_length )  # 4000 is total number of samples/epoch - results in 1 sec rms bins at 200 Hz 
    # iterate over all epochs
    EMG_rms=[]
    
    for epoch in range(EMG_epochs.shape[0]):
        for bin in range(bins):
            win1=int(bin*(EMG_window))                                 # beginning of window = bin * length of bin
            win2=int(win1 + (EMG_window-1))  # beginning of window + length of bin - 1
            emg_subset=EMG_epochs[epoch][win1:win2]     # extract given EMG from window
            EMG_rms=np.append(EMG_rms, np.sqrt((emg_subset).mean()**2))    # perform root-mean-square on EMG from window
    

    EMG_rms=EMG_rms.reshape(EMG_epochs.shape[0],config.epoch_length)

    
    return EMG_rms


#This function generates full set of features described in the paper 
#for all 4 channels
# NOTE: in the original version of the file 108 colums instead of 100 are created

def feature_generation(decimated_array_data):
    sig=decimated_array_data
    # Calculate all the datapoints per epoch per channel
    sig_len=sig.shape[-1]

    #Perform Fourier transformation
    # get broadband FFT magnitude in bin 2-55 Hz for normalization
    fourier_space = pyfftw.builders.fft(sig, axis=-1)
    mag=abs(fourier_space()[:,:,0:sig_len])/sig_len*2
    broadband=np.mean(mag[:,:,2:55], axis=-1)

    #Perform PSD transformation
    # get broadband PSD magnitude in bin 2-55 Hz for normalization
    f, psd=signal.welch(sig, config.sampling_freq_dec, axis=-1)
    broadband_psd=np.mean(psd[:, :, 2:55], axis=-1)


    #Generate Features array based on transformations data and concatenate it with 
    # The array with EMG RMS data

    # features_array= np.array([])
    features_array =np.array([np.mean(sig, axis=-1), 
                        np.median(sig, axis=-1), 
                        np.std(sig, axis=-1), 
                        np.var(sig, axis=-1), 
                        skew(sig, axis=-1), 
                        kurtosis(sig, axis=-1), 
                        np.mean(mag[:, :, 2:4], axis=-1)/broadband,
                        np.mean(mag[:, :, 4:7], axis=-1)/broadband,
                        np.mean(mag[:, :, 7:13], axis=-1)/broadband,
                        np.mean(mag[:, :, 13:30], axis=-1)/broadband,
                        np.mean(mag[:, :, 30:55], axis=-1)/broadband,
                        np.mean(mag[:, :,65:100], axis=-1)/broadband,
                            np.mean(mag[:, :, 2:4], axis=-1)/np.mean(mag[:, :,4:7], axis=-1),
                            # broadband,
                            np.mean(psd[:, :, 2:4], axis=-1)/broadband_psd,
                            np.mean(psd[:, :, 4:7], axis=-1)/broadband_psd,
                            np.mean(psd[:, :, 7:13], axis=-1)/broadband_psd,
                            np.mean(psd[:, :, 13:30], axis=-1)/broadband_psd,
                            np.mean(psd[:, :, 30:55], axis=-1)/broadband_psd,
                            np.mean(psd[:, :, 65:100], axis=-1)/broadband_psd,
                            np.mean(psd[:, :, 2:4], axis=-1)/np.mean(psd[:, :,4:7],axis=-1),
                            # broadband_psd
                            ])
    # CHange the shape of the array, adding the columns of all four channels as rows
    f=features_array.transpose(1,2, 0).reshape(sig.shape[0],-1)
    # Get RMS data
    e=calculate_EMG_RMS(sig)

    # If EMG is sampled at <20 seconds, repeat values such that there are 20 total values to fit indices 20 final indices
    emg = np.repeat(e, int(20/e.shape[-1]), axis=-1)
    #Concatenate both features arrays
    x = np.concatenate((f, emg), axis=1)

    return x

# This function looks for scoring data in the dictionary created from uploaded matlab data and reshape it to 
# concatenate it with the rest of decimated data

def find_scores(new_dict): # This is code that looks for scoring array
    a_reshaped=None
    for k in new_dict.keys():
        pattern = r'(?i)sl'
        match = re.search(pattern, k)
        if match:
            # print("Found:", match.group())
            # print(np.array(new_dict[k]['codes'][0]).shape)
            Sleep_codes=np.array(new_dict[k]['codes'][0])
            epoch_ratio=int(round(config.target_epoch_count/len(Sleep_codes)))
            sleepEpochsRound=int(config.target_epoch_count/epoch_ratio)        
    
            a_reshaped = np.repeat(Sleep_codes[:sleepEpochsRound,0],epoch_ratio).reshape(config.target_epoch_count, 1)
            
            
    return a_reshaped
    

# This function saves fully processed data into the provided path
def save_processed_data(decimated_folder_path,file_name, result,scores):


    if type(scores)==np.ndarray:
        folder_path = config.decimated_folder_path+'npy_newest_scored/'+'Feats_Fourier_and_PSD/'
    else:
        folder_path = config.decimated_folder_path+'npy_newest_unscored/'+'Feats_Fourier_and_PSD/'

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name=file_name.replace(".smrx", "").replace(".mat", "")
    word_list = file_name.split(" ")
    
    name_x=f'x_ffnorm {word_list[4]} {word_list[1]} {word_list[2]} {word_list[0]}'
    name_y=f'y_ffnorm {word_list[4]} {word_list[1]} {word_list[2]} {word_list[0]}'

    # Full path for the file
    full_path_x = os.path.join(folder_path, name_x)
    full_path_y = os.path.join(folder_path, name_y)


    # Save the data to the specified file within the new folder
    if type(scores)==np.ndarray:

        joined_array=np.concatenate((result,scores),axis=-1)
        np.save(full_path_x, joined_array)

    else:
        np.save(full_path_x, result)

    dh_name.update(print(f"{file_name} processed and saved"))

def process_and_save(path):# Here the names of files to process are uploaded
# THe list is formatted as [complete_path, annotation type folder name, file_name]
    
    # Globals for display handlers to monitor progress and output updating progress bars.
    global dh_name, dh_keys,dh_error, dh_progress

    # Display handlers for file, keys, errors, and progress bars
    dh_name = display(f'Item: ',display_id=True)
    dh_keys = display(f'Keys: ',display_id=True)
    dh_error = display(f'',display_id=True)
    dh_progress = display(f'',display_id=True)

    # Get files from designated path
    file_list = return_file_list_from_server(config.path)
    file_names = file_list
    for file in progressbar(file_names):

        # Establish names for Fourier transformed files
        scored_folder_path = config.decimated_folder_path+'npy_newest_scored/'+'Feats_Fourier_and_PSD/'
        unscored_folder_path = config.decimated_folder_path+'npy_newest_unscored/'+'Feats_Fourier_and_PSD/'

        # Strip existing filetypes from file name, display it
        file_name=file[2].replace(".smrx", "").replace(".mat", "")
        dh_name.update(f'processing {file_name}')

        # Use existing file name to get fourier transformed file name
        word_list = file_name.split(" ")
        name_x=f'x_ffnorm {word_list[5]} {word_list[2]} {word_list[3]} {word_list[0]}'

        # If file is already scored with designated name, skip
        full_path_scored = os.path.join(unscored_folder_path, name_x+'.npy')
        full_path_unscored = os.path.join(scored_folder_path, name_x+'.npy')
        if os.path.isfile(full_path_unscored) or os.path.isfile(full_path_scored):
            continue

        new_array=download_new_file(file[0])
    


        """Here the uploaded file is ordered and decimated and converted to numpy array"""
        dec_data=decimate_all_channels(new_array)

        # For annotated data y is found and data is transformed into epochs
        
        array_epochs=create_epochs(dec_data)
        result= feature_generation(array_epochs)
        scaled_result=StandardScaler().fit_transform(result)
        scores=find_scores(new_array)
        save_processed_data(config.decimated_folder_path,file[2], result, scores)
        gc.collect()
        time.sleep(.1)
   

    
    print ("All Done!")