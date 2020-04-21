'''
Created on Dec 5, 2019

@author: petar
'''

from typing import List
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_jamzone_time_domain_features

def transform_to_rmssd_statistics(rr_list: List[float]) -> dict:
    
    #Remove the outliers in the signal
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_list, low_rri=300, high_rri=3000, verbose = False)
    
    # This replace outliers nan values with linear interpolation
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers, interpolation_method="linear")
    
    # This remove ectopic beats from signal
    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="karlsson", verbose = False)
    
    # This replace ectopic beats nan values with linear interpolation
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
    
    # Get time-domain and frequency domain features from our signal
    time_domain_features = get_jamzone_time_domain_features(interpolated_nn_intervals)
  
    return time_domain_features
