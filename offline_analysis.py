'''
Created on Dec 5, 2019

@author: petar
'''

from typing import List
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_jamzone_time_domain_features
import json
import pandas as pd
from _ast import If

def transform_to_snapshot_statistics(rr_list: List[float], timestamp_list: List[str]) -> dict:

    time_domain_features = transform_to_hrv_statistics(rr_list, timestamp_list, '60s')
  
    return time_domain_features

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
        return json.loads('{"errorCode":202}')
    
    # first 15 seconds of data are discarded as preparation time
    trimmed_data = nn_timestamps[nn_timestamps > nn_timestamps[0] + pd.to_timedelta('15 seconds')]
    
    time_domain_features = transform_to_hrv_statistics(rr_list[rr_list.size - trimmed_data.size:], trimmed_data, '60s')
    
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
    
    # Get time-domain and frequency domain features from our signal
    time_domain_features = get_jamzone_time_domain_features(interpolated_nn_intervals, timestamp_list, window_duration)
    
    time_domain_features = json.loads(time_domain_features)
    time_domain_features['errorCode'] = error_code
    
    return time_domain_features
