'''
Created on Dec 5, 2019

@author: petar
'''

from typing import List
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_jamzone_time_domain_features

def transform_to_snapshot_statistics(rr_list: List[float], timestamp_list: List[str]) -> dict:

    time_domain_features = transform_to_hrv_statistics(rr_list, timestamp_list, '60s')
  
    return time_domain_features

def transform_to_3dayme_statistics(rr_list: List[float], timestamp_list: List[str]) -> dict:

    # Cut data on 3 days with 24h recording data

    # Call the method three times and collect three return objects before sending them back
    time_domain_features = transform_to_hrv_statistics(rr_list, timestamp_list, '5min')
  
    return time_domain_features

def transform_to_hrv_statistics(rr_list: List[float], timestamp_list: List[str], window_duration: str) -> dict:
    
    #Remove the outliers in the signal
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_list, low_rri=300, high_rri=3000, verbose = False)
    
    # This replace outliers nan values with linear interpolation
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers, interpolation_method="linear")
    
    # This remove ectopic beats from signal
    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="karlsson", verbose = False)
    
    # This replace ectopic beats nan values with linear interpolation
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
    
    # Get time-domain and frequency domain features from our signal
    time_domain_features = get_jamzone_time_domain_features(interpolated_nn_intervals, timestamp_list, window_duration)
  
    return time_domain_features
