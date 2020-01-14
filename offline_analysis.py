'''
Created on Dec 5, 2019

@author: petar
'''

import os
from typing import List
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_jamzone_time_domain_features
    
TEST_DATA_FILENAME_10 = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_10.txt')
TEST_DATA_FILENAME_20 = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_20.txt')
TEST_DATA_FILENAME_60 = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_60.txt')
TEST_DATA_FILENAME_LARGE = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_large.txt')

def load_test_data(path):
    # Load test rr_intervals data
    with open(path, "r") as text_file:
        lines = text_file.readlines()
    nn_intervals = list(map(lambda x: int(x.strip()), lines))
    return nn_intervals

def transform_to_rmssd_statistics(rr_list: List[float]) -> dict:
    
    #Remove the outliers in the signal
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_list, low_rri=300, high_rri=3000, verbose = False)
    
    # This replace outliers nan values with linear interpolation
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers, interpolation_method="linear")
    
    # This remove ectopic beats from signal
    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik", verbose = False)
    
    # This replace ectopic beats nan values with linear interpolation
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
    
    # Get time-domain and frequency domain features from our signal
    time_domain_features = get_jamzone_time_domain_features(interpolated_nn_intervals)
  
    return time_domain_features
  
def test_transform_to_rmssd_statistics():
    
    # rr_intervals_list contains integer values of RR-interval
    rr_test_intervals = load_test_data(TEST_DATA_FILENAME_LARGE)
    
    # standardized test values
    #nn_intervals_10 = load_test_data(TEST_DATA_FILENAME_10)
    #nn_intervals_20 = load_test_data(TEST_DATA_FILENAME_20)
    #nn_intervals_60 = load_test_data(TEST_DATA_FILENAME_60)
    
    time_domain_features = transform_to_rmssd_statistics(rr_test_intervals)
    
    print(time_domain_features)
    
    # test standardized features
    #time_domain_features = transform_to_rmssd_statistics(nn_intervals_large)
    #print('large: [' + time_domain_features + ']')
    
    #time_domain_features = transform_to_rmssd_statistics(nn_intervals_10)
    #print('10: [' + time_domain_features + ']')
    
    #time_domain_features = transform_to_rmssd_statistics(nn_intervals_20)
    #print('20: [' + time_domain_features + ']')
    
    #time_domain_features = transform_to_rmssd_statistics(nn_intervals_60)
    #print('60: [' + time_domain_features + ']')
