# InteractiveShell.cache_size=0
# 10, n=2, ftype='iir',axis=0,zero_phase=True
# PC_name = socket.gethostname()

from scipy import signal

#Recording Data Parameters
recording_length_hours=12
sampling_freq=2000
epoch_length=20

recording_length_seconds=int(recording_length_hours*3600)
target_epoch_count=int(recording_length_seconds/epoch_length)
sampling_freq_dec=int(sampling_freq/10)    
epoch_samples_dec=sampling_freq_dec*epoch_length 

# Original time base at which the data was scored
original_scoring_epoch_length=20
original_scoring_epoch_total=int(recording_length_seconds/original_scoring_epoch_length)

#File procressing parameters
# Partial channel designators found in the input data, must match order of target_channels
input_channels=['ECog','EMG','HPC_L','HPC_R']
# Equivalent channels in the SWISC paradigm
target_channels=['dec_ECog','dec_EMG','dec_HPC_L','dec_HPC_R']

# SWISC is designed to score these states
states=['Wake','NREM','REM','Seizure','Post Ictal']
target_scores=[0,1,2,3,4]

# Designation for the above, in order, found in input data
input_scores=[1,2,3,4,5]

# Filter to applied prior to decimation
b,a=signal.butter(1,[1],'high', fs=sampling_freq)

decimated_folder_path="/Users/Olga/CS_projects/SWISC/package/data"
metadata_folder_path="/Users/Olga/CS_projects/SWISC/package/metadata"
processed_data_folder_path='/Users/Olga/CS_projects/SWISC/package/data'

#Feature Generation:

full_features_length=100

#Data Saving Path:

basepath = f'D:/npy_no_z_/'
types = ['/NoNo/', '/NoSz/', '/SlNo/', '/SlSl/' ]


save_decimated=False

basepath = f'D:/npy_no_z_{epoch_length}/'
# b,a=(1,[1],'high', fs==sampling_freq)
