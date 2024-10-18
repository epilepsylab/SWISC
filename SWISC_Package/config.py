# InteractiveShell.cache_size=0
# 10, n=2, ftype='iir',axis=0,zero_phase=True
# PC_name = socket.gethostname()


#Recording Data

recording_length_hours=12
sampling_freq=2000
epoch_length=20

recording_length_seconds=int(recording_length_hours*3600)
target_epoch_count=int(recording_length_seconds/epoch_length)
sampling_freq_dec=int(sampling_freq/10)    
epoch_samples_dec=sampling_freq_dec*epoch_length 

#File procressing parameters



#Decimation parameters:
channels=['ECog', 'EMG', 'HPC_L', 'HPC_R']

zScoreAtDecimation=True

decimated_folder_path="/Users/Olga/CS_projects/SWISC/package/data"
metadata_folder_path="/Users/Olga/CS_projects/SWISC/package/metadata"
processed_data_folder_path='/Users/Olga/CS_projects/SWISC/package/data'

#Feauture Generation:

full_features_length=100

#Data Saving Path:

basepath = f'D:/npy_no_z_/'
types = ['/NoNo/', '/NoSz/', '/SlNo/', '/SlSl/' ]

norm_flag=0



save_decimated=False

basepath = f'D:/npy_no_z_{epoch_length}/'
# b,a=(1,[1],'high', fs==sampling_freq)

