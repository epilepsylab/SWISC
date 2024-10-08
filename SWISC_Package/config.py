
recording_length_hours=12
sampling_freq=2000
sampling_freq_dec=int(sampling_freq/10)
epoch_length=4
recording_length_seconds=int(recording_length_hours*3600)
target_epoch_count=int(recording_length_seconds/epoch_length)
epoch_samples_dec=int(epoch_length*sampling_freq/10)
channels=channels=['ECog','EMG','HPC_L','HPC_R']
zScoreAtDecimation=False
decimated_folder_path=f'./Processed'
epoch_samples_dec=sampling_freq_dec*epoch_length

b,a=signal.butter(1,[1],'high', fs=sampling_freq)
