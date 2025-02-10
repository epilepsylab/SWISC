# InteractiveShell.cache_size=0
# 10, n=2, ftype='iir',axis=0,zero_phase=True
# PC_name = socket.gethostname()

from scipy import signal

#Recording Data Parameters
recording_length_hours=12
sampling_freq=2000
epoch_length=4


recording_length_seconds=int(recording_length_hours*3600)
target_epoch_count=int(recording_length_seconds/epoch_length)
# sampling_freq_dec=int(sampling_freq/10)    
sampling_freq_dec=200
resampling_factor=sampling_freq/sampling_freq_dec
epoch_samples_dec=sampling_freq_dec*epoch_length 
samples_expected=recording_length_seconds*sampling_freq_dec

# Original time base at which the data was scored
original_scoring_epoch_length=4
original_scoring_epoch_total=int(recording_length_seconds/original_scoring_epoch_length)

#File procressing parameters
# Partial channel designators found in the input data, must match order of target_channels
input_channels=['ECog','EMG','HPC_L','HPC_R']
# input_channels=['ECog','HPC_L','HPC_R']
# input_channels=['HPC_L','HPC_R']
input_channels=['ECog','EMG']


# Equivalent channels in the SWISC paradigm
target_channels=['dec_ECog','dec_EMG','dec_HPC_L','dec_HPC_R']
# target_channels=['dec_ECog','dec_HPC_L','dec_HPC_R']
# target_channels=['dec_HPC_L','dec_HPC_R']
target_channels=['dec_ECog','dec_EMG']

# Designation for the above, in order, found in input data
input_scores=[1,2,3,4,5]

# SWISC is designed to score these states
states=['Wake','NREM','REM','Seizure','Post Ictal']
target_scores=[0,1,2,3,4]

# Filter to applied prior to decimation
b,a=signal.butter(1,[1],'high', fs=sampling_freq)

decimated_folder_path="/Users/Olga/CS_projects/SWISC/package/data"
metadata_folder_path="/Users/Olga/CS_projects/SWISC/package/metadata"
processed_data_folder_path='/Users/Olga/CS_projects/SWISC/package/data'

parent_folder='D:/Pilocarpine/'
input_path=f"{parent_folder}Mats"
decimated_folder_path=f"{parent_folder}Export/Decimated"
metadata_folder_path=f"{parent_folder}Export/Metadata"
processed_data_folder_path=f"{parent_folder}Export/Py/"
scoring_output_path=f"{parent_folder}Export/ScoredData/"

model_path="./Models/"

#Feature Generation:

full_features_length=60


#Data Saving Path:

types = ['/NoNo/', '/NoSz/', '/SlNo/', '/SlSl/' ]

save_decimated=False

# b,a=(1,[1],'high', fs==sampling_freq)
