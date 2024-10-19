import os
import re
import sys

import time

# data processing packages
import hdf5storage
import pandas as pd
import numpy as np
#import neo
# signal processing packages

from scipy import signal
from scipy.signal import butter, filtfilt, decimate
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from sklearn.pipeline import Pipeline

# future engeneering packages

import pyfftw
from scipy.stats import kurtosis       # kurtosis function
from scipy.stats import skew           # skewness function

# config file has all the processing parameters for the data files
import config
from IPython.display import display
import gc
import matplotlib.pyplot as plt



import os
from glob import glob

def return_file_list_from_server(input_path):
    all_file_paths = []

    
    # If it's a single path
    if os.path.isfile(input_path):
        # Single file, add it to the list
        all_file_paths.append(input_path)
    
    elif os.path.isdir(input_path):
        # It's a folder, get all files recursively
        for root, dirs, files in os.walk(input_path):
            for file in files:
                all_file_paths.append(os.path.join(root, file))
    
    
    return all_file_paths

def extract_metadata(file_name):
    file_name = file_name.replace(".smrx", "").replace(".mat", "")
    file_parts = file_name.split('/')

    # Step 2: Extract the needed parts
    # Cohort is in the part before the  folder (or type folder), so we split by spaces and slashes to extract
    cohort = file_parts[-3] # NPM661-664
    annotation=file_parts[-2]  #'SlSz'
        
    # Extracting type, mouse, and other parts from the final part of the filename
    last_part = file_parts[-1].split()

    # Step 3: Extract Metadata
    metadata = {
        'Cohort': cohort,                  # 'NPM661-664'
        'Mouse': last_part[3],             # 'm1'
        'Annotation Type': annotation,              # 'SlSz'
        'Date': last_part[1],              # '210712'
        'Time': last_part[2],              # '191029_056'
    }

    # Printing the metadata dictionary
    return metadata

 # Optional Funciton that prints out names of the folders and files found:
# def describe_the_files(files_found):
#     print('  ')
#     print("Total files found {}".format(len(files_found)))
#     print("   ")
#     recording_formats=set()
#     recording_names=[]
#     for recording in files_found:
#         if recording[1] not in recording_formats:
#             recording_formats.add(recording[1])
#         recording_names.append(recording[2])

#     print('Recording formats found {0}: {1}, '.format(len(recording_formats),recording_formats))
#     print("    ")
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
def decimate_channel(k, dict, a, b):
    channel_data=np.array(dict[k]['values'][0],dtype=np.float32)
    channel_data=signal.filtfilt(b,a,channel_data,axis=0)


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
    b,a=signal.butter(1,[1],'high', fs=config.sampling_freq)
    #dh_keys.update(f'Keys: {new_array.keys()}')

    for index, input, target in zip( range(len(config.input_channels) ),config.input_channels,config.target_channels):
        
        match_key=list(k for k in new_array.keys() if input in k)[0]
        dec_dict[target]=decimate_channel(match_key, new_array, a, b)

                
    min_channel_samples=len(min(dec_dict.values(), key=len))
    min_full_epochs=(min_channel_samples//config.epoch_samples_dec)
    min_samples_full_epochs=config.epoch_samples_dec*min_full_epochs
    
    dec_dict={key: np.array(value).flatten()[:min_samples_full_epochs] for key, value in dec_dict.items()}
    

    return  dec_dict, min_full_epochs

def zscore_all_channels(dec_dict):
     dec_dict={key: zscore(value,axis=-1)  for key, value in dec_dict.items()}
     return dec_dict

#Ths function reshapes the aray creating 2160 epochs with 4000 recordings (20s) each
def create_epochs(dec_data):
    channel_number=len(config.target_channels)
    ordered_rows = np.array([dec_data[row] for row in config.target_channels])

# Transpose the list of lists to get the data in the correct shape
    n_epochs=ordered_rows.shape[1]//config.epoch_samples_dec
    ordered_array=ordered_rows.reshape(channel_number, n_epochs, config.epoch_samples_dec).transpose(1, 0, 2)
    #dh_progress.update(f'{c_arr.shape}')
    
    return ordered_array

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
    b = np.repeat(e, int(20/e.shape[-1]), axis=-1)
    #Concatenate both features arrays
    x = np.concatenate((f, b), axis=1)
    
    return x

# This function looks for scoring data in the dictionary created from uploaded matlab data and reshape it to 
# concatenate it with the rest of decimated data

def find_scores(new_dict): # This is code that looks for scoring array

    # If scores are not indexed as they are in SWISC, subtract so lowest value is 0
    reindex=min(config.input_scores)-min(config.target_scores)
        
    for k in new_dict.keys():
        # Find channel which is designated as sleep channel
        pattern = r'(?i)sl'
        match = re.search(pattern, k)
        if match:
            print("Found:", match.group())
            # Grab sleep codes from dictionary
            Sleep_codes=np.array(new_dict[k]['codes'][0])
            # Use only sleep codes which evenly match the desired recording length and original scoring epoch length
            Sleep_codes_match=Sleep_codes[:config.original_scoring_epoch_total,0]
            # If epoch length for evaluation is smaller than the scoring epoch length, upsample the original scores
            epoch_ratio=config.target_epoch_count//config.original_scoring_epoch_total
            # If epoch length the same, repeat will be 1, and it will not repeat
            scores = np.repeat(Sleep_codes_match,epoch_ratio).reshape(config.target_epoch_count, 1)
            # Reindex the scores to SWISC's 0-4 output
            scores_reindex = scores-reindex
            return scores_reindex

    return
            
    


def save_processed_data(file_name, folder, x_data, y_data=None ):
    """
    Save the processed data to the specified folder.
    """
    folder_path=config.processed_data_folder_path + folder
    
    if y_data is not None:
        results=np.concatenate((x_data,y_data),axis=-1)      
    else:
        results=x_data
    
    os.makedirs(folder_path, exist_ok=True)
    np.save(os.path.join(folder_path, file_name), results)

def make_filename(meta):
    return f" {meta['Cohort']}_{meta['Mouse']}_{meta['Date']}_{meta['Time']}"
        # 'Cohort': cohort,                  # 'NPM661-664'
        # 'Mouse': last_part[3],             # 'm1'
        # 'Annotation Type': last_part[4],              # 'SlSz'
        # 'Date': last_part[1],              # '210712'
        # 'Time': last_part[2],              # '191029_056'
   


# This function saves fully processed data into the provided path

    #dh_name.update(print(f"{file_name} processed and saved"))
# def get_final_file_name(file):
#     scored_folder_path = config.decimated_folder_path+'npy_newest_scored/'+'Feats_Fourier_and_PSD/'
#     unscored_folder_path = config.decimated_folder_path+'npy_newest_unscored/'+'Feats_Fourier_and_PSD/'

#     file_name=file[2].replace(".smrx", "").replace(".mat", "")
#     #dh_name.update(f'processing {file_name}')
#     word_list = file_name.split(" ")
#     name_x=f'x_ffnorm {word_list[5]} {word_list[2]} {word_list[3]} {word_list[0]}'
#     full_path_scored = os.path.join(unscored_folder_path, name_x+'.npy')
#     full_path_unscored = os.path.join(scored_folder_path, name_x+'.npy')
#     if os.path.isfile(full_path_unscored) or os.path.isfile(full_path_scored):
#         continue    
#     return config.decimated_folder_path,file[2]    

def process_and_save(path):

    file_list=return_file_list_from_server(path)
    file_names = file_list
    i=1
    for file in file_names:

        print(f"Starting file {i} from {len(file_names)} files")    
    # all the file metadata is extracted into meta_dict
        meta_dict=extract_metadata(file)
        file_name=make_filename(meta_dict)
    # These are cosequtive steps to process data
        new_array=download_new_file(file)
        dec_data, min_full_epochs=decimate_all_channels(new_array)
        z_scored_data=zscore_all_channels(dec_data)
        array_epochs=create_epochs(z_scored_data)
        result= feature_generation(array_epochs)    
        scaled_result=StandardScaler().fit_transform(result)
    # try to find scores
        try:
            scores=find_scores(new_array, min_full_epochs)
            folder="Processed_Scored/"
            save_processed_data(file_name, folder, result,scores)
            if config.save_decimated==True:
                folder="Z_Decimated_Scored/"
                dec_flat=np.array([dec_data[row] for row in config.target_channels])
                save_processed_data(file_name, folder, dec_flat,np.repeat(scores, config.epoch_samples_dec) )
        
        except Exception as e:
            print(f"No score found for file {file}, error {e}")
            folder="Processed_Unscored/"
            save_processed_data(file_name, folder, result)
            if config.save_decimated==True:
                folder="Z_Decimated_Unscored/"
                dec_flat=np.array([dec_data[row] for row in config.target_channels])
                save_processed_data(file_name, folder, dec_flat)
        
        print(f"Finished file {i} from {len(file_names)} files")   
        i+=1
        

 #scores_data= np.repeat(scores, 4000, axis=2)
        

            
        gc.collect()
        time.sleep(.1)
   

    

    print("All Done!")