import numpy as np
# import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D
from datetime import datetime
import pandas as pd
import os
import glob
import config
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from time import gmtime, strftime
import config

def getDatasetSummary(xlsx_path):
    file_path=Path(xlsx_path)
    datasetSummary = pd.read_excel(file_path)
    datasetSummary['Start'] = pd.to_datetime(datasetSummary['Start'])
    datasetSummary['End'] = pd.to_datetime(datasetSummary['End'])
    datasetSummary['Condition Start DateTime'] = pd.to_datetime(datasetSummary['Condition Start Date'].astype(str) + ' ' + datasetSummary['Condition Start Time'].astype(str),errors='coerce')
    # datasetSummary=datasetSummary[datasetSummary['Condition']!='X']

    
    return datasetSummary
    
def get_time_label(row, start_times):
    
    subject = row['subject']
    dt = row['datetime']
    start_time = start_times.get(subject)
    
    if pd.isnull(start_time):
        return 'Unknown'
    elif dt < start_time:
        return 'Pre'
    else:
        return 'Post'

def get_condition_label(row, start_times):
    
    subject = row['subject']
    condition = start_times.get(subject)
    
    if pd.isnull(condition):
        return 'X'
    elif condition=='KA':
        return 'KA'
    else:
        return 'NaCl'

def get_condition_label(row, start_times):
    
    subject = row['subject']
    condition = start_times.get(subject)
    
    if pd.isnull(condition):
        return 'Include'
    elif condition=='X':
        return 'X'
    else:
        return 'X'

def parse_start_times(combined_df,datasetSummary):
    condition_start_times = dict(zip(datasetSummary['Name'], 
                                        pd.to_datetime(datasetSummary['Condition Start DateTime'])))
    
    combined_df['prepost'] = combined_df.apply(lambda row: get_time_label(row, condition_start_times), axis=1)

    recording_start_times = dict(zip(datasetSummary['Name'], 
                                    pd.to_datetime(datasetSummary['Start'])))

    combined_df['recorded'] = combined_df.apply(lambda row: get_time_label(row, recording_start_times), axis=1)

    conditions = dict(zip(datasetSummary['Name'], 
                                    datasetSummary['Condition']))

    combined_df['condition'] = combined_df.apply(lambda row: get_condition_label(row, conditions), axis=1)
    
    conditions = dict(zip(datasetSummary['Name'], 
                                    datasetSummary['Include']))

    combined_df['include'] = combined_df.apply(lambda row: get_condition_label(row, conditions), axis=1)

    return combined_df
 
def parse_file(file_path):
    """Parse individual csv file to extract data."""
    # Extract subject, date, and time from file name
    file_name = os.path.basename(file_path)
    cohort, mouse_num, subject, date, start_time, sequence_num, annotation = file_name.replace('.csv', '').split('_')
    
    # Read CSV file into a DataFrame
    scores = pd.read_csv(file_path, sep=',', header=None)
    
    start_datetime=pd.to_datetime(f'{date}, {start_time}', errors='raise', yearfirst=True)
    windowed_scoring_start=start_datetime+pd.to_timedelta(3*config.epoch_length,unit='s')
    df=pd.DataFrame()
    
    df['datetime']= pd.date_range(start=windowed_scoring_start, periods=len(scores.transpose()), freq=f'{config.epoch_length}s',   unit='s')
    df['subject']=[subject for i in range(len(df))]
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['scores']=scores.transpose()
    df['epoch_length'] = pd.to_timedelta(config.epoch_length, unit='s')  
        
    return df
    
def load_all_files(data_folder):
    """Load all CSV files from the folder into a single DataFrame."""
    all_files = glob.glob(os.path.join(data_folder, '*.csv'))
    df_list = [parse_file(file) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def state_by_hour(df):
    """Calculate total seconds per label for each hour."""
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    return df.groupby(['condition','subject', 'prepost','scores', 'year','month','day','hour']).agg(seconds_per_hour=('epoch_length','sum'),datetime=('datetime','min'))

def state_by_circadian(df):
    """Calculate total seconds per label for each hour."""
    df['hour'] = df['datetime'].dt.hour
    df['day']=df['datetime'].dt.day

    df=df.groupby(['condition','subject', 'prepost', 'scores', 'hour']).agg(seconds_per_bin=('epoch_length','sum'),epochs=('day','nunique'),datetime=('datetime','min'))
    df['mean_seconds_per_circadian_hour']=(df['seconds_per_bin']/df['epochs']).astype('int64')/1e9
    return df

def state_by_day(df):
    """Calculate daily averages of scores."""
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day

    return df.groupby(['condition','subject', 'prepost', 'scores','year', 'month','day']).agg(seconds_per_day=('epoch_length','sum'),datetime=('datetime','min'))

def state_by_week(df):
    """Calculate weekly averages of scores."""
    df['week'] = df['datetime'].dt.isocalendar().week
    df_weekly = df.groupby(['subject', 'week', 'scores']).size().reset_index(name='seconds')
    return df_weekly.groupby(['subject','intervention', 'scores']).mean().reset_index()

def compute_bouts(df):
    """Compute bout lengths and occurrences."""
    df['bout_id'] = (df['scores'] != df['scores'].shift()).cumsum()  # Create unique id for each bout
    # Calculate bout length: account for the final epoch by adding epoch_length to the bout duration
    # bouts = df.groupby(['subject','datetime','bout_id']).agg(
    #     datetime=('datetime', 'min'),
    #     scores=('scores', 'first')
    # )
    
    # Bout length = (bout_end + epoch_length) - bout_start
    df['epoch_length'] = pd.to_timedelta(config.epoch_length, unit='s')    
    
    # by_bout_id=df.groupby(['condition','subject', 'prepost','bout_id','scores']).agg(bout_length=('epoch_length','sum'),year=('year','min'),month=('month','min'),day=('day','min'),hour=('hour','min'),datetime=('datetime','min'))
    by_bout_id=df.groupby(['condition','subject', 'prepost','bout_id','scores','year','month','day','hour']).agg(bout_length=('epoch_length','sum'),datetime=('datetime','min'))
    return df, by_bout_id

def average_bout_length_by_day(bouts):
    """Calculate average bout length by day."""
    bout_length_daily = bouts.groupby(['condition','subject','scores','year','month', 'day']).agg(average_bout_length=('bout_length','mean'),datetime=('datetime','min'))
    return bout_length_daily
    
def average_bout_length_by_hour(bouts):
    """Calculate average bout length by day."""
    # bout_length_daily = bouts.groupby(['condition','subject', 'prepost','scores','year','month', 'day','hour']).agg(average_bout_length=('bout_length','mean'))
    bout_length_hourly = bouts.groupby(['condition','subject','scores','year','month', 'day','hour']).agg(average_bout_length=('bout_length','mean'),datetime=('datetime','min'))
    return bout_length_hourly

def average_bout_length_by_week(bouts):
    """Calculate average bout length by week."""
    bouts['week'] = bouts['datetime'].dt.isocalendar().week
    return bouts.groupby(['condition','subject', 'scores','year','month','week']).agg(average_bout_length=('bout_length','mean'),datetime=('datetime','min'))

def compute_bout_occurrences(bouts):
    """Compute how many times a bout starts."""
    bouts.reset_index()
    return bouts.groupby(['condition','subject', 'prepost','scores']).agg(bout_occurrences=('bout_id','nunique'))
    
def bout_occurrences_by_day(bouts):
    """Compute how many times a bout starts."""
    bouts.reset_index()
    return bouts.groupby(['condition','subject', 'prepost','scores','year','month','day']).agg(bout_occurrences=('bout_id','nunique'))
    
def bout_occurrences_by_week(bouts):
    """Calculate average bout occurrences by week."""
    bouts['week'] = bouts['datetime'].dt.isocalendar().week
    bout_occurrences_weekly = bouts.groupby(['condition','subject','prepost', 'scores', 'year','month', 'week']).agg(bout_occurrences=('bout_id','nunique'))
    
    return bout_occurrences_weekly

def average_by_score(df,measure):
    return df.groupby(['subject','scores'][measure].mean())

# Example usage:
# Assuming files are in 'data_folder'

# Load data
# df = load_all_files(data_folder)

from concurrent.futures import ThreadPoolExecutor
from functools import partial

def parse_file_optimized(file_path, config):
    """Optimized version of parse_file with vectorized operations."""
    # Extract metadata from filename more efficiently
    file_name = os.path.basename(file_path)
    # _, subject, date, start_time = file_name.replace('.csv', '').split('_')[1].split(' ')
    cohort, mouse_num, subject, date, start_time, sequence_num, annotation = file_name.replace('.csv', '').split('_')

    
    # Read CSV more efficiently by specifying dtypes
    # scores = pd.read_csv(file_path, sep=',', header=None).T  # Transpose during read
    scores = pd.DataFrame(np.loadtxt(file_path, delimiter=",",dtype='int'))

    
    # Calculate start time once
    start_datetime = pd.to_datetime(f'{date}, {start_time}', yearfirst=True)
    windowed_scoring_start = start_datetime + pd.Timedelta(seconds=3*config.epoch_length)
    
    # Create datetime index efficiently
    n_periods = len(scores)
    datetime_index = pd.date_range(
        start=windowed_scoring_start, 
        periods=n_periods, 
        freq=f'{config.epoch_length}s'
    )
    
    # Create DataFrame efficiently using a dictionary
    df = pd.DataFrame({
        'datetime': datetime_index,
        'subject': np.repeat(subject, n_periods),
        'scores': scores[0].values,  # Access first column directly
        'epoch_length': np.repeat(pd.Timedelta(seconds=config.epoch_length), n_periods)
    })
    
    
    return df

def process_folder(folder, data_folder, config):
    """Process all files in a folder."""
    all_files = glob.glob(os.path.join(data_folder, folder, '*classifier*.csv'))
    if not all_files:
        return pd.DataFrame()
    
    # Process files in the folder
    dfs = [parse_file_optimized(file, config) for file in all_files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def process_all_data(data_folder, config, num_folders=None):
    """Process all folders with parallel execution."""
    folders = os.listdir(data_folder)[:num_folders] if num_folders else os.listdir(data_folder)
    
    # Use ThreadPoolExecutor for I/O-bound operations
    with ThreadPoolExecutor(max_workers=None) as executor:
        # Create a partial function with fixed arguments
        process_func = partial(process_folder, data_folder=data_folder, config=config)
        # Process folders in parallel
        df_list = list(executor.map(process_func, folders))
    
    # Concatenate all results
    final_df = pd.concat(df_list, ignore_index=True)
    
    # Get dataset summary and parse start times
    datasetSummary = getDatasetSummary()
    return parse_start_times(final_df, datasetSummary)
    
    
def parse_start_times_fast(combined_df, datasetSummary):
    # Create a mapping Series for each condition
    start_time_series = pd.Series(pd.to_datetime(datasetSummary['Condition Start DateTime'].values),
                                 index=datasetSummary['Name'],name='Condition Start DateTime')

    recording_time_series = pd.Series(pd.to_datetime(datasetSummary['Start'].values),
                                    index=datasetSummary['Name'],name='Start')
    recording_end_series = pd.Series(pd.to_datetime(datasetSummary['End'].values),
                                    index=datasetSummary['Name'],name='End')
    condition_series = pd.Series(datasetSummary['Condition'].values,
                               index=datasetSummary['Name'],name='Condition')
    include_series = pd.Series(datasetSummary['Include'].values,
                             index=datasetSummary['Name'], name='Include')
    
    # Use merge instead of map/apply
    merged_df = combined_df.merge(
        start_time_series.reset_index().rename(
            columns={'Name': 'subject', 'Condition Start DateTime': 'condition_start'}),
        on='subject',
        how='left'
    )
    
    # Vectorized operations for all conditions at once
    merged_df['prepost'] = np.select(
        [merged_df['condition_start'].isna(),
         merged_df['datetime'] < merged_df['condition_start']],
        ['X', 'Ante'],
        default='Post'
    )
    
    # Add recorded times
    merged_df = merged_df.merge(
        recording_time_series.reset_index().rename(
            columns={'Name': 'subject', 'Start': 'recording_start'}),
        on='subject',
        how='left'
    )
    merged_df['recorded'] = np.select(
        [merged_df['recording_start'].isna(),
         merged_df['datetime'] < merged_df['recording_start']],
        ['Unknown', 'Ante'],
        default='Post'
    )
    
    merged_df = merged_df.merge(
        recording_end_series.reset_index().rename(
            columns={'Name': 'subject', 'End': 'recording_end'}),
        on='subject',
        how='left'
    )
    merged_df['end'] = np.select(
        [merged_df['recording_end'].isna(),
         merged_df['datetime'] >= merged_df['recording_end']],
        ['Unknown', 'End'],
        default=''
    )
    
    # Add conditions
    merged_df = merged_df.merge(
        condition_series.reset_index().rename(
            columns={'Name': 'subject', 'Condition': 'condition_type'}),
        on='subject',
        how='left'
    )
    merged_df['condition'] = np.select(
        [merged_df['condition_type'].isna(),
         merged_df['condition_type'] == 'KA',
         merged_df['condition_type'] == 'Pilo'],
        ['X', 'KA','Pilo'],
        default='NaCl'
    )
    
    # Add includes
    merged_df = merged_df.merge(
        include_series.reset_index().rename(
            columns={'Name': 'subject', 'Include': 'include_type'}),
        on='subject',
        how='left'
    )
    merged_df['include'] = np.select(
        [merged_df['include_type'].isna(),
         merged_df['include_type'] == 'X'],
        ['Include', 'X'],
        default='X'
    )
    
    # Drop temporary columns
    result_df = merged_df.drop(columns=['condition_start', 'recording_start', 
                                      'condition_type', 'include_type'])
    
    return result_df

def strdtYMD(input):
    return datetime.strptime(input,'%Y-%m-%d')


def strdtYMDH(input):
    return datetime.strptime(input,'%Y-%m-%d-%H')


def output_excel(data, filename):
    
    data=filename.astype('str')
    with pd.ExcelWriter(
        f'{outpath+outfile}',
        date_format="YYYY-MM-DD",
        datetime_format="YYYY-MM-DD HH:MM:SS") as writer:
        output.to_excel(writer)  