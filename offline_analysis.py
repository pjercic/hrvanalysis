'''
Created on Dec 5, 2019

@author: petar
'''

from typing import List
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_jamzone_time_domain_features
from machinelearning import classify_features_supervised_knn, classify_features_supervised_reg, classify_features_supervised_linreg
import json
import pandas as pd
import os

def transform_to_snapshot_statistics(rr_list: List[float], timestamp_list: List[str]) -> dict:

    time_domain_features = transform_to_hrv_statistics(rr_list, timestamp_list, '60s')
  
    return time_domain_features

def transform_to_snapshot_statistics_ipc(path_named_pipe: str):
    
    try:
        # get file descriptor of the input pipe without blocking
        fd = os.open(path_named_pipe, os.O_RDONLY | os.O_NONBLOCK)
        
        # read input pipe
        with os.fdopen(fd) as input_pipe:
            message = input_pipe.read()
            if not message:
                raise ValueError('Named pipe is empty')
                return -1
    except:
        raise ConnectionError('Error reading data from the PIPE')      
        return -1
    
    try:    
        data_temp = json.loads(message)
    except:
        raise SyntaxError('Error parsing JSON data from the PIPE')      
        return -1
        
    data_temp = pd.read_json(json.dumps(data_temp['rrs'], ensure_ascii=False))
    df = pd.DataFrame(data_temp)
    
    rr_list = df['values'].to_numpy()
    timestamp_list = df['index'].tolist()
    
    time_domain_features = transform_to_hrv_statistics(rr_list, timestamp_list, '60s')
    
    try:
        # write generated data to the pipe
        with open(path_named_pipe, 'wt') as output_pipe:
            #json.dump(json_example, output_pipe)
            output_pipe.write(time_domain_features + '\n')
    
    except:
        raise ConnectionError('Error writing data from the PIPE')      
        return -1
        
    return 0

def transform_to_snapshot_statistics_ipc_echo(path_named_pipe: str):
    
    try:
        # get file descriptor of the input pipe without blocking
        fd = os.open(path_named_pipe, os.O_RDONLY | os.O_NONBLOCK)
        
        # read input pipe
        with os.fdopen(fd) as input_pipe:
            message = input_pipe.read()
            json_input_list = '[successful] Hello from python: ' + message
            if not message:
                json_input_list = '[failed] Named pipe is empty'
                raise ValueError('Named pipe is empty')
    except:
        json_input_list = '[failed] Error reading data from the PIPE'
    
    try:
        # write generated data to the pipe
        with open(path_named_pipe, 'wt') as output_pipe:
            #json.dump(json_example, output_pipe)
            output_pipe.write(json_input_list + '\n')
    
    except:
        return 'END transform_to_snapshot_statistics_ipc_echo: [failed] Error writing data from the PIPE ' + json_input_list

    return 'END transform_to_snapshot_statistics_ipc_echo: ' + json_input_list

def transform_to_snapshot_statistics_ipc_error(path_named_pipe: str):
    
    try:
        # get file descriptor of the input pipe without blocking
        fd = os.open(path_named_pipe, os.O_RDONLY | os.O_NONBLOCK)
        
        # read input pipe
        with os.fdopen(fd) as input_pipe:
            message = input_pipe.read()
            if not message:
                json_input_list = '[failed] Named pipe is empty'
                raise ValueError('Named pipe is empty')
    except:
        json_input_list = '[failed] Error reading data from the PIPE'
        
    try:
        data_temp = pd.read_json(message)
        json_input_list = json.dumps(json.loads('{"errorCode":202}'), ensure_ascii=False)
    except:
        json_input_list = json.dumps(json.loads('{"errorCode":404}'), ensure_ascii=False)
    
    try:
        # write generated data to the pipe
        with open(path_named_pipe, 'wt') as output_pipe:
            #json.dump(json_example, output_pipe)
            output_pipe.write(json_input_list + '\n')
    
    except:
        return 'END transform_to_snapshot_statistics_ipc_error: [failed] Error writing data from the PIPE ' + json_input_list

    return 'END transform_to_snapshot_statistics_ipc_error: ' + json_input_list
        
def transform_to_3dayme_statistics(rr_list: List[float], timestamp_list: List[str]) -> dict:

    starting_value = 0
    ending_value = 0
    nn_timestamps = pd.to_datetime(timestamp_list)
    
    # Cut data on 3 days with 24h recording data
    for x in range(3):
  
        time = nn_timestamps[nn_timestamps < nn_timestamps[0] + pd.to_timedelta('1 day')]
        starting_value = ending_value
        ending_value = ending_value + time.size
    
        # Call the method three times and collect three return objects before sending them back
        time_domain_features = transform_to_hrv_statistics(rr_list[starting_value:ending_value], timestamp_list[starting_value:ending_value], '5min')
  
        nn_timestamps = nn_timestamps[time.size:]
        
    return time_domain_features

def transform_to_morning_snapshots_statistics(rr_list: List[float], timestamp_list: List[str]) -> dict:
    
    nn_timestamps = pd.to_datetime(timestamp_list)
    
    # snapshot for two minutes
    if nn_timestamps[-1] - nn_timestamps[0] < pd.to_timedelta('2 minutes'):
        return json.dumps(json.loads('{"errorCode":202}'), ensure_ascii=False)
    
    # first 15 seconds of data are discarded as preparation time
    trimmed_data = nn_timestamps[nn_timestamps > nn_timestamps[0] + pd.to_timedelta('15 seconds')]
    
    time_domain_features = transform_to_hrv_statistics(rr_list[len(rr_list) - trimmed_data.size:], trimmed_data, '60s')
    
    return time_domain_features
    
def transform_to_hrv_statistics(rr_list: List[float], timestamp_list: List[str], window_duration: str) -> dict:
    
    error_code = 0
    
    #Remove the outliers in the signal
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_list, low_rri=300, high_rri=3000, verbose = False)
    
    #Check for data errors
    if pd.isna(rr_intervals_without_outliers).all():
        error_code = 201
    elif sum(pd.notnull(rr_intervals_without_outliers)) / len(rr_intervals_without_outliers) < 0.75:
        error_code = 301
    
    # This replace outliers nan values with linear interpolation
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers, interpolation_method="linear")
    
    # This remove ectopic beats from signal
    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="karlsson", verbose = False)
    
    #Check for data errors
    if pd.isna(rr_intervals_without_outliers).all():
        error_code = 201
    elif sum(pd.notnull(rr_intervals_without_outliers)) / len(rr_intervals_without_outliers) < 0.75:
        error_code = 301
        
    # This replace ectopic beats nan values with linear interpolation
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
    
    # Remove leading NaN values after filtering
    s = pd.Series(interpolated_nn_intervals)
    interpolated_nn_intervals = s.loc[s.first_valid_index():].tolist()
    timestamp_list = timestamp_list[-len(interpolated_nn_intervals):]

    # Get time-domain and frequency domain features from our signal
    time_domain_features = get_jamzone_time_domain_features(interpolated_nn_intervals, timestamp_list, window_duration)
    
    time_domain_features = json.loads(time_domain_features)
    if error_code != 0:
        time_domain_features['errorCode'] = error_code
    time_domain_features['version'] = '1.0.0'
    
    return json.dumps(time_domain_features, ensure_ascii=False)

def classify_hrv_statistics(nn_intervals_train: List[float], timestamp_list_train: List[str], labels_list_train: List[str], nn_intervals: List[float], timestamp_list: List[str]) -> dict:
    
    return classify_features_supervised_knn(nn_intervals_train, timestamp_list_train, labels_list_train, nn_intervals, timestamp_list)

def regression_hrv_statistics(nn_intervals_train: List[float], timestamp_list_train: List[str], labels_list_train: List[str], nn_intervals: List[float], timestamp_list: List[str]) -> dict:
    
    #return classify_features_supervised_reg(nn_intervals_train, timestamp_list_train, labels_list_train, nn_intervals, timestamp_list)
    return classify_features_supervised_linreg(nn_intervals_train, timestamp_list_train, labels_list_train, nn_intervals, timestamp_list)
