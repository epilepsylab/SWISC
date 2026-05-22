import scipy.signal as signal 
recording_length_hours = 12
sampling_freq = 2000
epoch_length = 20
recording_length_seconds = 43200
target_epoch_count= 2160
sampling_freq_dec = 200
epoch_samples_dec = 4000
resampling_factor=10
input_channels = ['ECog', 'EMG', 'HPC_L', 'HPC_R']
target_channels = ['dec_ECog', 'dec_EMG', 'dec_HPC_L', 'dec_HPC_R']
states = ['Wake', 'NREM', 'REM', 'Seizure', 'Post Ictal']
input_scores = [1, 2, 3, 4, 5]
target_scores = [0, 1, 2, 3, 4]
save_decimated = False
input_path = "D:/Paper Revision 2025/Mats/"
types = ['/NoNo/', '/NoSz/', '/SlNo/', '/SlSz/'] 
export_folder_path = "Y:/Brandon/Spike2_Pipeline/Test/Export"
decimated_folder_path = "Y:/Brandon/Spike2_Pipeline/Test/Export/Decimated"
metadata_folder_path = "Y:/Brandon/Spike2_Pipeline/Test/Export/Metadata"
processed_data_folder_path = "Y:/Brandon/Spike2_Pipeline/Test/Export/Py"
scoring_output_path = "Y:/Brandon/Spike2_Pipeline/Test/Export/ScoredData"
log_file_path = "Y:/Brandon/Spike2_Pipeline/Test/Export/export_log.txt"
model_path = "./Models/"
folders=[export_folder_path,decimated_folder_path,metadata_folder_path,processed_data_folder_path,scoring_output_path]
b,a = signal.butter(1,[1],'high', fs=sampling_freq)

full_features_length = 100
