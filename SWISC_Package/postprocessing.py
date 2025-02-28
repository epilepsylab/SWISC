from keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib as mpl
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, log_loss, cohen_kappa_score, matthews_corrcoef
import pandas as pd
import os
import config
from sklearn.preprocessing import StandardScaler
from keras.models import clone_model

processed_data_folder_path=config.processed_data_folder_path
channels_available=" ".join(sorted(config.target_channels))
model_channel_map= {
    'SWISC EMG HPCL HPCR RMS.h5': ['dec_EMG', 'dec_HPC_L', 'dec_HPC_R'],
    'SWISC ECoG EMG RMS.h5': ['dec_ECog', 'dec_EMG'],
    'SWISC EMG RMS.h5': ['dec_EMG'],
    'SWISC ECoG.h5': ['dec_ECog'],
    'SWISC 4 Channel and RMS.h5': ['dec_ECog', 'dec_EMG', 'dec_HPC_L', 'dec_HPC_R'],
    'SWISC HPCL.h5': ['dec_HPC_L'],
    'SWISC HPCL HPCR.h5': ['dec_HPC_L', 'dec_HPC_R']
}

# Function to choose the correct model based on recorded channels from config file
def choose_model(channels_available, model_channel_map):
    
    
    # Iterate over the model to channel mapping
    for model_name, required_channels in model_channel_map.items():
        # Convert required channels to a sorted string
        required_string = " ".join(sorted(required_channels))
        
        # Check if the available channels match the required channels
        if channels_available == required_string:
            file_path = os.path.join(config.model_path, model_name)
            print (f"Model selected: {model_name}, file path {file_path}")
            
            return file_path  # Return the model if there's an exact match
    
    # If no match is found, return this message
    return "No model available for this configuration"

def generate_x_sequences(input_array, window):
    
    #scaler = StandardScaler()

# Fit the scaler to the data and transform it
    #input_array = scaler.fit_transform(input_array)
    shift = window * 2  # Shift based on window size for sliding window
    
    # Set max_feats if not provided
    max_feats = input_array.shape[1] 
    
    # Preallocate output array for X
    output_array = np.zeros((len(input_array) - shift, window * 2 + 1, max_feats))
    
    # Create sliding window sequences
    for i in range(window, len(input_array) - window):
        output_array[i - window] = input_array[i - window:i + window + 1, :max_feats]
    return output_array

def generate_y_sequences (input_array, windows):

    shift=windows*2
    output_array = np.zeros((len(input_array) - shift))
    for i in range(windows, len(input_array) - windows):
            output_array[i - windows] = input_array[i]
    output_array=output_array.reshape(-1, 1)
    encoded_features=to_categorical(output_array, num_classes=5) 
    
    return encoded_features

def load_data(path):
    data = np.load(path, allow_pickle=True)
    print(path)
    X = data[:, 0:config.full_features_length]  # Assuming 100 is the column with labels
    y = None
    y_predict = None
    
    if data.shape[1]>config.full_features_length:
        
        y = data[:, config.full_features_length]
        
        minscore=min(y)
        if minscore==1:
            y=y-1
        y_predict=generate_y_sequences(y, 3)

    X_predict=generate_x_sequences(X, 3)
    return X_predict, y_predict

def predict_data(path, model):
    X_predict, y_predict = load_data(path)
    predictions = model.predict(X_predict,verbose=False)
    
    y_pred = np.argmax(predictions, axis=1)
    y_pred_corrected = rechtshaffen(y_pred)
    
    if y_predict is not None:
        y_true = np.argmax(y_predict, axis=1)
    
        # for state in range(5):
        #     display(len(np.where(y_true==state)[0])-len(np.where(y_pred==state)[0]))
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        print('Acc: ', accuracy)
        print('With RK: ',accuracy_score(y_true, y_pred_corrected))
        print('Num wrong: ', len(np.where(y_true!=y_pred)[0]))
        overview={
            'File Name': path.split('/')[-1],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Cohen Kappa': kappa,
            'MCC': mcc,
            # 'Log Loss': loss
            }

    else:
        overview={
            'File Name': path.split('/')[-1],
            }
        
    file_name=path.split('/')[-1].replace('.npy','.csv')
    file_name=file_name.strip()
    print(os.path.join(config.scoring_output_path, file_name))
    y_pred_corrected.tofile(os.path.join(config.scoring_output_path, file_name), sep = ',')

    

    # Return as a dictionary
    
    return overview, y_pred

def model_predict(processed_data_folder_path):
    print("This program fits processed data to selected model and saves model output it as a csv file")
    results=[]
    model_file_path=choose_model(channels_available, model_channel_map)
    model=load_model(model_file_path)
    for file_name in sorted(os.listdir(processed_data_folder_path)):
        file_path = os.path.join(processed_data_folder_path, file_name)
        
        # Check if it's a file (and not a directory)
        if os.path.isfile(file_path):
            results.append(predict_data(file_path, model))
    results_df=pd.DataFrame(results)
    results_df.to_csv(os.path.join(config.scoring_output_path, "results.csv"))
    return results_df

def model_clone_fit(X_train, y_train, use_weights=True):
    print("You can train suggested models on your data:")
    results=[]
    model_file_path=choose_model(channels_available, model_channel_map)
    original_model=load_model(model_file_path)
    model=clone_model(original_model)

    if use_weights==True:
        model.set_weights(original_model.get_weights())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train,              # Your input features
        y_train,              # Your target labels
        validation_split=0.2, # Split 20% of the training data for validation
        epochs=60,            # Number of epochs to train
        batch_size=2160         # Batch size
    )

    return model, history

def rechtshaffen(scores):
    for idx in range(0,len(scores)-1):
        if idx>0:
            # REM Must Follow NREM
            # "If REM is preceded by Wake, score as Wake"
            if scores[idx]==2 and scores[idx-1]==0:
                scores[idx]=0

            # In order to score NREM, there must be 2 consecutive epochs
            # or lone NREM is scored as the previous epoch
            if scores[idx]==1:
                if scores[idx-1]!=1 and scores[idx+1]!=1:
                    scores[idx]=scores[idx-1]

            # In order to score Seizure, there must be 2 consecutive epochs
            # # or lone Seizure is scored as the previous epoch
            # if scores[idx]==3:
            #     if scores[idx-1]!=3 and scores[idx+1]!=1:
            #         scores[idx]=scores[idx-1]
    return scores


 