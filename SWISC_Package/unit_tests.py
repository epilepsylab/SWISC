import os
import numpy as np
import config

# (C) Github.com/vassardog77

# Define the path to log.txt in the root directory
log_file_path = os.path.join(os.getcwd(), 'log.txt')

def log_message(message, log_file_path):
    """Appends a message as a new line to the log file."""
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')


def signal_length_test(decimated_channel):


    channel_length = decimated_channel.shape[-1]

    if channel_length > config.samples_expected:
        excess_samples = channel_length - config.samples_expected
        log_message(f'Channel truncated by {excess_samples} samples.',log_file_path)
        truncated_channel = decimated_channel[:config.samples_expected] 
        return truncated_channel

    elif channel_length < config.samples_expected:
        # Log the error without raising an exception
        log_message(
            f'Channel is not {config.samples_expected} samples long; '
            f'actual length: {channel_length}. Length rounded to nearest epoch.', log_file_path
        )
        # Return False to indicate an issue with the channel length
        return False

    # If channel_length == config.samples_expected, return True
    return True



def invalid_data_test(decimated_channel):

    # check decimated_channel for NaN

    if np.any(np.isnan(decimated_channel))==True:
        # pipe output to log.txt: NaN value at point {point #} of file {file #}
        log_message(
            f'Nan value located in ',log_file_path)
        return False

    else: return True


def run_all_tests(dec_data):
       for decimated_channel in dec_data[:4]:
            correct_length = signal_length_test(decimated_channel)
            valid_data = invalid_data_test(decimated_channel)

            if correct_length==False or valid_data==False:
                return True