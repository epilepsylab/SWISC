import os
import numpy as np
import config

# Define the path to log.txt in the root directory
log_file_path = config.log_file_path

def log_message(message, log_file_path):
    """Appends a message as a new line to the log file."""
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')