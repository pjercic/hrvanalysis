'''
Created on Dec 5, 2019

@author: petar
'''

from typing import List
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_jamzone_time_domain_features, get_jamzone_frequency_domain_features
from machinelearning import classify_features_supervised_knn, classify_features_supervised_reg, classify_features_supervised_linreg
from dataanalysis import compare
import json
import pandas as pd
import os

def transform_to_snapshot_statistics(rr_list: List[float], timestamp_list: List[str]) -> dict:

    time_domain_features = transform_to_hrv_statistics(rr_list, timestamp_list, '60s')
  
    return time_domain_features

def transform_to_snapshot_statistics_ipc(snapshotInfo: str, path_named_pipe: str) ->  int:
    
    try:
        data = json.loads(snapshotInfo);

    except:
        raise SyntaxError('Error parsing snapshot info JSON data')
        answer = json.dumps(json.loads('{"errorCode":401}'), ensure_ascii=False)

    # Get RR values
    #data_temp = pd.read_json(json.dumps(data_temp['rrs'], ensure_ascii=False))
    df = pd.read_json('[{"index":"2020-07-18T10:17:33.253Z","values":577},{"index":"2020-07-18T10:17:33.838Z","values":569},{"index":"2020-07-18T10:17:34.391Z","values":569},{"index":"2020-07-18T10:17:34.963Z","values":553},{"index":"2020-07-18T10:17:35.503Z","values":553},{"index":"2020-07-18T10:17:36.042Z","values":545},{"index":"2020-07-18T10:17:36.583Z","values":530},{"index":"2020-07-18T10:17:37.124Z","values":530},{"index":"2020-07-18T10:17:37.617Z","values":530},{"index":"2020-07-18T10:17:38.16Z","values":514},{"index":"2020-07-18T10:17:38.652Z","values":514},{"index":"2020-07-18T10:17:39.196Z","values":506},{"index":"2020-07-18T10:17:39.687Z","values":506},{"index":"2020-07-18T10:17:40.229Z","values":514},{"index":"2020-07-18T10:17:40.768Z","values":530},{"index":"2020-07-18T10:17:41.309Z","values":553},{"index":"2020-07-18T10:17:41.937Z","values":608},{"index":"2020-07-18T10:17:42.527Z","values":616},{"index":"2020-07-18T10:17:43.198Z","values":655},{"index":"2020-07-18T10:17:43.873Z","values":709},{"index":"2020-07-18T10:17:44.599Z","values":709},{"index":"2020-07-18T10:17:45.313Z","values":725},{"index":"2020-07-18T10:17:46.036Z","values":756},{"index":"2020-07-18T10:17:46.798Z","values":764},{"index":"2020-07-18T10:17:47.564Z","values":733},{"index":"2020-07-18T10:17:48.284Z","values":694},{"index":"2020-07-18T10:17:48.968Z","values":678},{"index":"2020-07-18T10:17:49.543Z","values":584},{"index":"2020-07-18T10:17:50.133Z","values":561},{"index":"2020-07-18T10:17:50.713Z","values":592},{"index":"2020-07-18T10:17:51.255Z","values":577},{"index":"2020-07-18T10:17:51.838Z","values":577},{"index":"2020-07-18T10:17:52.423Z","values":577},{"index":"2020-07-18T10:17:53.01Z","values":569},{"index":"2020-07-18T10:17:53.594Z","values":577},{"index":"2020-07-18T10:17:54.139Z","values":569},{"index":"2020-07-18T10:17:54.719Z","values":577},{"index":"2020-07-18T10:17:55.307Z","values":569},{"index":"2020-07-18T10:17:55.89Z","values":592},{"index":"2020-07-18T10:17:56.474Z","values":592},{"index":"2020-07-18T10:17:57.058Z","values":569},{"index":"2020-07-18T10:17:57.643Z","values":600},{"index":"2020-07-18T10:17:58.235Z","values":577},{"index":"2020-07-18T10:17:58.819Z","values":592},{"index":"2020-07-18T10:17:59.443Z","values":592},{"index":"2020-07-18T10:18:00.027Z","values":592},{"index":"2020-07-18T10:18:00.613Z","values":577},{"index":"2020-07-18T10:18:01.246Z","values":631},{"index":"2020-07-18T10:18:01.874Z","values":663},{"index":"2020-07-18T10:18:02.503Z","values":625},{"index":"2020-07-18T10:18:03.133Z","values":608},{"index":"2020-07-18T10:18:03.72Z","values":608},{"index":"2020-07-18T10:18:04.348Z","values":592},{"index":"2020-07-18T10:18:04.888Z","values":577},{"index":"2020-07-18T10:18:05.473Z","values":577},{"index":"2020-07-18T10:18:06.058Z","values":577},{"index":"2020-07-18T10:18:06.689Z","values":569},{"index":"2020-07-18T10:18:07.184Z","values":569},{"index":"2020-07-18T10:18:07.769Z","values":561},{"index":"2020-07-18T10:18:08.308Z","values":553},{"index":"2020-07-18T10:18:08.892Z","values":561},{"index":"2020-07-18T10:18:09.433Z","values":561},{"index":"2020-07-18T10:18:10.018Z","values":561},{"index":"2020-07-18T10:18:10.559Z","values":553},{"index":"2020-07-18T10:18:11.099Z","values":553},{"index":"2020-07-18T10:18:11.683Z","values":530},{"index":"2020-07-18T10:18:12.224Z","values":553},{"index":"2020-07-18T10:18:12.831Z","values":600},{"index":"2020-07-18T10:18:13.394Z","values":584},{"index":"2020-07-18T10:18:14.022Z","values":625},{"index":"2020-07-18T10:18:14.698Z","values":663},{"index":"2020-07-18T10:18:15.329Z","values":655},{"index":"2020-07-18T10:18:15.961Z","values":625},{"index":"2020-07-18T10:18:16.599Z","values":616},{"index":"2020-07-18T10:18:17.208Z","values":616},{"index":"2020-07-18T10:18:17.763Z","values":616},{"index":"2020-07-18T10:18:18.394Z","values":625},{"index":"2020-07-18T10:18:19.062Z","values":625},{"index":"2020-07-18T10:18:19.695Z","values":625},{"index":"2020-07-18T10:18:20.283Z","values":608},{"index":"2020-07-18T10:18:20.912Z","values":600},{"index":"2020-07-18T10:18:21.495Z","values":592},{"index":"2020-07-18T10:18:22.124Z","values":592},{"index":"2020-07-18T10:18:22.75Z","values":647},{"index":"2020-07-18T10:18:23.388Z","values":663},{"index":"2020-07-18T10:18:24.016Z","values":663},{"index":"2020-07-18T10:18:24.693Z","values":655},{"index":"2020-07-18T10:18:25.369Z","values":647},{"index":"2020-07-18T10:18:25.948Z","values":608},{"index":"2020-07-18T10:18:26.578Z","values":592},{"index":"2020-07-18T10:18:27.119Z","values":577},{"index":"2020-07-18T10:18:27.707Z","values":561},{"index":"2020-07-18T10:18:28.289Z","values":561},{"index":"2020-07-18T10:18:28.832Z","values":553},{"index":"2020-07-18T10:18:29.413Z","values":545},{"index":"2020-07-18T10:18:30.003Z","values":592},{"index":"2020-07-18T10:18:30.629Z","values":639},{"index":"2020-07-18T10:18:31.306Z","values":670},{"index":"2020-07-18T10:18:32.025Z","values":725},{"index":"2020-07-18T10:18:32.699Z","values":702},{"index":"2020-07-18T10:18:33.417Z","values":702},{"index":"2020-07-18T10:18:34.094Z","values":663},{"index":"2020-07-18T10:18:34.68Z","values":616},{"index":"2020-07-18T10:18:35.314Z","values":616},{"index":"2020-07-18T10:18:35.942Z","values":608},{"index":"2020-07-18T10:18:36.526Z","values":600},{"index":"2020-07-18T10:18:37.156Z","values":608},{"index":"2020-07-18T10:18:37.739Z","values":616},{"index":"2020-07-18T10:18:38.325Z","values":561},{"index":"2020-07-18T10:18:38.872Z","values":600},{"index":"2020-07-18T10:18:39.474Z","values":584},{"index":"2020-07-18T10:18:40.127Z","values":600},{"index":"2020-07-18T10:18:40.709Z","values":577},{"index":"2020-07-18T10:18:41.364Z","values":647},{"index":"2020-07-18T10:18:41.979Z","values":616},{"index":"2020-07-18T10:18:42.577Z","values":592},{"index":"2020-07-18T10:18:43.139Z","values":577},{"index":"2020-07-18T10:18:43.782Z","values":608},{"index":"2020-07-18T10:18:44.362Z","values":616},{"index":"2020-07-18T10:18:45Z","values":647},{"index":"2020-07-18T10:18:45.632Z","values":631},{"index":"2020-07-18T10:18:46.296Z","values":639},{"index":"2020-07-18T10:18:46.876Z","values":625},{"index":"2020-07-18T10:18:47.507Z","values":592},{"index":"2020-07-18T10:18:48.106Z","values":625},{"index":"2020-07-18T10:18:48.732Z","values":600},{"index":"2020-07-18T10:18:49.325Z","values":592},{"index":"2020-07-18T10:18:49.893Z","values":584},{"index":"2020-07-18T10:18:50.476Z","values":577},{"index":"2020-07-18T10:18:51.016Z","values":569},{"index":"2020-07-18T10:18:51.628Z","values":553},{"index":"2020-07-18T10:18:52.153Z","values":545},{"index":"2020-07-18T10:18:52.681Z","values":522},{"index":"2020-07-18T10:18:53.21Z","values":569},{"index":"2020-07-18T10:18:53.793Z","values":561},{"index":"2020-07-18T10:18:54.357Z","values":561},{"index":"2020-07-18T10:18:54.952Z","values":561},{"index":"2020-07-18T10:18:55.573Z","values":647},{"index":"2020-07-18T10:18:56.243Z","values":639},{"index":"2020-07-18T10:18:56.909Z","values":655},{"index":"2020-07-18T10:18:57.559Z","values":686},{"index":"2020-07-18T10:18:58.233Z","values":647},{"index":"2020-07-18T10:18:58.866Z","values":600},{"index":"2020-07-18T10:18:59.45Z","values":608},{"index":"2020-07-18T10:19:00.067Z","values":616},{"index":"2020-07-18T10:19:00.689Z","values":639},{"index":"2020-07-18T10:19:01.333Z","values":625},{"index":"2020-07-18T10:19:01.905Z","values":569},{"index":"2020-07-18T10:19:02.491Z","values":616},{"index":"2020-07-18T10:19:03.081Z","values":561},{"index":"2020-07-18T10:19:03.676Z","values":569}]')
    rr_list = df['values'].to_numpy()
    timestamp_list = df['index'].tolist()

    if data['analysis'] == 'recording':
        answer = transform_to_hrv_statistics(rr_list, timestamp_list, '60s')

    elif data['analysis'] == 'wakeup':
        answer = transform_to_morning_snapshots_statistics(rr_list, timestamp_list)

    elif data['analysis'] == '3dayme':
        answer = transform_to_3dayme_statistics(rr_list, timestamp_list)
    
    try:
        # write generated data to the pipe
        with open(path_named_pipe, 'wt') as output_pipe:
            #json.dump(json_example, output_pipe)
            output_pipe.write(answer + '\n')
    
    except:
        raise ConnectionError('Error writing data to the PIPE')      
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
    freq_domain_features = get_jamzone_frequency_domain_features(interpolated_nn_intervals)
    
    hrv_domain_features = json.loads(time_domain_features)
    freq_domain_features = json.loads(freq_domain_features)
    if error_code != 0:
        hrv_domain_features['errorCode'] = error_code
    hrv_domain_features['lfHfRatio'] = freq_domain_features['lf_hf_ratio']
    hrv_domain_features['balanceGroup'] = freq_domain_features['balance_group']
    hrv_domain_features['balanceScore'] = freq_domain_features['balance_score']
    hrv_domain_features['vitalityScore'] = 0.833 * hrv_domain_features['balanceScore'] + 0.167 * hrv_domain_features['restScore'] + 0.0000000000000001457
    hrv_domain_features['vitalityGroup'] = hrv_domain_features['vitalityScore']
    hrv_domain_features['version'] = '1.0.1'
    
    return json.dumps(hrv_domain_features, ensure_ascii=False)

def classify_hrv_statistics(nn_intervals_train: List[float], timestamp_list_train: List[str], labels_list_train: List[str], nn_intervals: List[float], timestamp_list: List[str]) -> dict:
    
    return classify_features_supervised_knn(nn_intervals_train, timestamp_list_train, labels_list_train, nn_intervals, timestamp_list)

def regression_hrv_statistics(nn_intervals_train: List[float], timestamp_list_train: List[str], labels_list_train: List[str], nn_intervals: List[float], timestamp_list: List[str]) -> dict:
    
    #return classify_features_supervised_reg(nn_intervals_train, timestamp_list_train, labels_list_train, nn_intervals, timestamp_list)
    return classify_features_supervised_linreg(nn_intervals_train, timestamp_list_train, labels_list_train, nn_intervals, timestamp_list)

def compare_snapshots (snapshotGroups: str, path_named_pipe: str) ->  int:
    
    answer = compare(snapshotGroups)
    
    try:
        # write generated data to the pipe
        with open(path_named_pipe, 'wt') as output_pipe:
            #json.dump(json_example, output_pipe)
            output_pipe.write(answer + '\n')
    
    except:
        raise ConnectionError('Error writing data to the PIPE')      
        return -1
    
    return 0
