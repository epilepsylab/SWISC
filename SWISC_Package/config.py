from scipy import signal

# Input folder containing .mat files
parent_folder='D:/Paper Revision 2025/'
input_path=f"{parent_folder}Mats"

# Designations for subfolders in Mat folder.
# Coding represents presence of sleep or seizure scores 
types = ['/NoNo/', '/NoSz/', '/SlNo/', '/SlSz/' ]

# Desired output path for decimated .npy files, scored files, metadata
parent_folder='Y:/Brandon/Spike2_Pipeline/'
export_folder_path=f"{parent_folder}Export"
decimated_folder_path=f"{export_folder_path}/Decimated"
metadata_folder_path=f"{export_folder_path}/Metadata"
processed_data_folder_path=f"{export_folder_path}/Py/"
scoring_output_path=f"{export_folder_path}/ScoredData/"
log_file_path=f"{export_folder_path}/export_log.txt"

folders=[export_folder_path,decimated_folder_path,metadata_folder_path,processed_data_folder_path,scoring_output_path]

# Flag for saving decimated, epoched data to .npy, in addtion to feature-extracted preprocessed data.
save_decimated=False

# Path containing trained model files
model_path="./Models/"

#Recording Data Parameters
# Length of each individual file in Hours, Seconds
recording_length_hours=12
recording_length_seconds=int(recording_length_hours*3600)

# Recording sampling frequency
sampling_freq=2000

# Original time base at which the data was scored
original_scoring_epoch_length=20
original_scoring_epoch_total=int(recording_length_seconds/original_scoring_epoch_length)

# Desired epoch length for analysis of signal
epoch_length=20

# Number of Epochs in of epoch_length in signal of recording_length_seconds
target_epoch_count=int(recording_length_seconds/epoch_length)


# Target sampling frequency 
sampling_freq_dec=200
resampling_factor=int(sampling_freq/sampling_freq_dec)
epoch_samples_dec=sampling_freq_dec*epoch_length 
samples_expected=recording_length_seconds*sampling_freq_dec


#File procressing parameters
# Partial channel designators found in the input data, must match order of target_channels
# Uncomment correct line equivalent to desired input channels
input_channels=['ECog','EMG','HPC_L','HPC_R']
# input_channels=['ECog','HPC_L','HPC_R']
# input_channels=['HPC_L','HPC_R']
# input_channels=['ECog','EMG']


# Equivalent channels in the SWISC paradigm
# Uncomment correct line equivalent to desired target channels
target_channels, full_features_length =[['dec_ECog','dec_EMG','dec_HPC_L','dec_HPC_R'],100]
#target_channels, full_features_length =[['dec_ECog','dec_HPC_L','dec_HPC_R'],60]
#target_channels, full_features_length =[['dec_HPC_L','dec_HPC_R'], 40]
#target_channels, full_features_length =[['dec_ECog','dec_EMG'],60]

# SWISC is designed to score these states
states=['Wake','NREM','REM','Seizure','Post Ictal']

# Numeric designation for the above, in order, found in input data
input_scores=[1,2,3,4,5]

# SWISC coding for the five states, do not change
target_scores=[0,1,2,3,4]

# Filter to applied prior to decimation
b,a=signal.butter(1,[1],'high', fs=sampling_freq)

    