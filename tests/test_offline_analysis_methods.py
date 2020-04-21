import os
import random
import json
import numpy as np
from offline_analysis import transform_to_rmssd_statistics
from hrvanalysis.plot import plot_timeseries

TEST_DATA_FILENAME_10 = os.path.join(os.path.dirname(__file__), './test_nn_intervals_10.txt')
TEST_DATA_FILENAME_20 = os.path.join(os.path.dirname(__file__), './test_nn_intervals_20.txt')
TEST_DATA_FILENAME_60 = os.path.join(os.path.dirname(__file__), './test_nn_intervals_60.txt')
TEST_DATA_FILENAME_LARGE = os.path.join(os.path.dirname(__file__), './test_nn_intervals_large.txt')
TEST_DATA_FILENAME_BUG = os.path.join(os.path.dirname(__file__), './bug20200408_test_nn_intervals.txt')

def load_test_data(path):
    # Load test rr_intervals data
    with open(path, "r") as text_file:
        lines = text_file.readlines()
    nn_intervals = list(map(lambda x: int(x.strip()), lines))
    return nn_intervals

def test_transform_to_rmssd_statistics(noElements):
    
    # rr_intervals_list contains integer values of RR-interval
    if noElements <= 1000:
        rr_test_intervals = np.array(load_test_data(TEST_DATA_FILENAME_LARGE))
        rr_test_intervals = rr_test_intervals[:noElements]
    else:
        rr_test_intervals = np.array([random.normalvariate(600, 60) for _ in range(noElements)])
        rr_test_intervals = rr_test_intervals.astype(int)
    
    time_domain_features = transform_to_rmssd_statistics(rr_test_intervals)
    
    print(time_domain_features)
    
def test_bugs():
    
    # rr_intervals_list contains integer values of RR-interval for the bug
    rr_test_intervals = np.array(load_test_data(TEST_DATA_FILENAME_BUG))
        
    time_domain_features = transform_to_rmssd_statistics(rr_test_intervals)
    
    print(time_domain_features)
    
    # Plot others
    plot_timeseries(rr_test_intervals);
    
    jdata = json.loads(time_domain_features)
    plot_timeseries(jdata['rmssdArray']);
