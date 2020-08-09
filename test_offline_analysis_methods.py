import os
import random
import json
import numpy as np
import pandas as pd
import json
from offline_analysis import transform_to_snapshot_statistics, transform_to_3dayme_statistics, transform_to_morning_snapshots_statistics, classify_hrv_statistics, regression_hrv_statistics
from machinelearning import classify_models_evaluation_knn, classify_models_evaluation_reg, classify_models_evaluation_linreg
from hrvanalysis.plot import plot_timeseries
from numpy import int

TEST_DATA_FILENAME_10 = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_10.txt')
TEST_DATA_FILENAME_20 = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_20.txt')
TEST_DATA_FILENAME_60 = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_60.txt')
TEST_DATA_FILENAME_LARGE = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_large.txt')
TEST_DATA_FILENAME_BUG = os.path.join(os.path.dirname(__file__), './tests/bug20200603_test_nn_intervals.txt')
TEST_TIMESTAMPS_FILENAME_BUG = os.path.join(os.path.dirname(__file__), './tests/bug20200603_test_timestamps.txt')

def load_test_data(path):
    # Load test rr_intervals data
    with open(path, "r") as text_file:
        lines = text_file.readlines()
    nn_intervals = list(map(lambda x: int(x.strip()), lines))
    return nn_intervals

def load_test_timestamps(path):
    # Load test rr_intervals data
    with open(path, "r") as text_file:
        lines = text_file.readlines()
    nn_timestamps = list(lines)
    return nn_timestamps

def test_transform_to_snapshot_statistics(noElements):
    
    # rr_intervals_list contains integer values of RR-interval
    if noElements <= 1000:
        rr_test_intervals = np.array(load_test_data(TEST_DATA_FILENAME_BUG))
        rr_test_intervals = rr_test_intervals[:noElements]
        rr_test_timestamps = load_test_timestamps(TEST_TIMESTAMPS_FILENAME_BUG)
        rr_test_timestamps = rr_test_timestamps[:noElements]
    else:
        rr_test_intervals = np.array([random.normalvariate(600, 60) for _ in range(noElements)])
        rr_test_intervals = rr_test_intervals.astype(int)
        rr_test_timestamps = pd.date_range(start=pd.datetime.now(), periods=noElements, freq = '600ms')
        rr_test_timestamps = rr_test_timestamps.strftime("%Y-%m-%d %H:%M:%S.%f")
    
    time_domain_features = transform_to_snapshot_statistics(rr_test_intervals, rr_test_timestamps)
    
    print(time_domain_features)
    
def test_transform_to_3dayme_statistics():
    
    rr_test_intervals = np.array([random.normalvariate(600, 60) for _ in range(500000)])
    rr_test_intervals = rr_test_intervals.astype(int)
    rr_test_timestamps = pd.date_range(start=pd.datetime.now(), periods=500000, freq = '600ms')
    rr_test_timestamps = rr_test_timestamps.strftime("%Y-%m-%d %H:%M:%S.%f")

    time_domain_features = transform_to_3dayme_statistics(rr_test_intervals, rr_test_timestamps)
    
    print(time_domain_features)

def test_transform_to_morning_snapshots_statistics():
    
    rr_test_intervals = np.array([random.normalvariate(600, 60) for _ in range(250)])
    rr_test_intervals = rr_test_intervals.astype(int)
    rr_test_timestamps = pd.date_range(start=pd.datetime.now(), periods=250, freq = '600ms')
    rr_test_timestamps = rr_test_timestamps.strftime("%Y-%m-%d %H:%M:%S.%f")

    time_domain_features = transform_to_morning_snapshots_statistics(rr_test_intervals, rr_test_timestamps)
    
    print(time_domain_features)
    
def test_bugs():
    
    # rr_intervals_list contains integer values of RR-interval for the bug
    rr_test_intervals = np.array(load_test_data(TEST_DATA_FILENAME_BUG))
    
    # rr_intervals_list contains integer values of RR-interval for the bug
    rr_test_timestamps = load_test_timestamps(TEST_TIMESTAMPS_FILENAME_BUG)
        
    time_domain_features = transform_to_morning_snapshots_statistics(rr_test_intervals, rr_test_timestamps)
    
    print(time_domain_features)
    
    # Plot others
    plot_timeseries(rr_test_intervals);
    
    jdata = json.loads(time_domain_features)
    df = pd.read_json(jdata['rmssdArray'], typ='series')
    plot_timeseries(df);

def test_classify_hrv_statistics(noElements):
    
    # rr_intervals_list contains integer values of RR-interval
    if noElements <= 1000:
        rr_test_intervals = np.array(load_test_data(TEST_DATA_FILENAME_BUG))
        rr_test_intervals = rr_test_intervals[:noElements]
        rr_test_timestamps = load_test_timestamps(TEST_TIMESTAMPS_FILENAME_BUG)
        rr_test_timestamps = rr_test_timestamps[:noElements]
    else:
        rr_test_intervals = np.array([random.normalvariate(600, 60) for _ in range(noElements)])
        rr_test_intervals = rr_test_intervals.astype(int)
        rr_test_timestamps = pd.date_range(start=pd.datetime.now(), periods=noElements, freq = '600ms')
        rr_test_timestamps = rr_test_timestamps.strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # Use https://www.snorkel.org/ in te future to programatically label the data
        labels_list_train = rr_test_intervals < 600
        labels_list_train = labels_list_train.astype(str)
        
        rr_predict_intervals = np.array([random.normalvariate(600, 60) for _ in range(noElements)])
        rr_predict_intervals = rr_predict_intervals.astype(int)
        rr_predict_timestamps = pd.date_range(start=pd.datetime.now(), periods=noElements, freq = '600ms')
        rr_predict_timestamps = rr_predict_timestamps.strftime("%Y-%m-%d %H:%M:%S.%f")
        classify_features = classify_hrv_statistics(rr_test_intervals, rr_test_timestamps, labels_list_train, rr_predict_intervals, rr_predict_timestamps)
        print(classify_features)

def test_regression_hrv_statistics(noElements):
    
    # rr_intervals_list contains integer values of RR-interval
    if noElements <= 1000:
        rr_test_intervals = np.array(load_test_data(TEST_DATA_FILENAME_BUG))
        rr_test_intervals = rr_test_intervals[:noElements]
        rr_test_timestamps = load_test_timestamps(TEST_TIMESTAMPS_FILENAME_BUG)
        rr_test_timestamps = rr_test_timestamps[:noElements]
    else:
        rr_test_intervals = np.array([random.normalvariate(600, 60) for _ in range(noElements)])
        rr_test_intervals = rr_test_intervals.astype(int)
        rr_test_timestamps = pd.date_range(start=pd.datetime.now(), periods=noElements, freq = '600ms')
        rr_test_timestamps = rr_test_timestamps.strftime("%Y-%m-%d %H:%M:%S.%f")
        
        labels_list_train = []
        time_domain_features = transform_to_snapshot_statistics(rr_test_intervals, rr_test_timestamps)
        time_domain_features = json.loads(time_domain_features)
        for obj in time_domain_features['hrArray']:
            labels_list_train.append(obj['values'])
        
        rr_predict_intervals = np.array([random.normalvariate(600, 60) for _ in range(noElements)])
        rr_predict_intervals = rr_predict_intervals.astype(int)
        rr_predict_timestamps = pd.date_range(start=pd.datetime.now(), periods=noElements, freq = '600ms')
        rr_predict_timestamps = rr_predict_timestamps.strftime("%Y-%m-%d %H:%M:%S.%f")
        classify_features = regression_hrv_statistics(rr_test_intervals[1:], rr_test_timestamps, np.array(labels_list_train), rr_predict_intervals, rr_predict_timestamps)
        print(classify_features)

def test_classifier_model_hrv_statistics(noElements):

    # rr_intervals_list contains integer values of RR-interval
    if noElements <= 1000:
        rr_test_intervals = np.array(load_test_data(TEST_DATA_FILENAME_BUG))
        rr_test_intervals = rr_test_intervals[:noElements]
        rr_test_timestamps = load_test_timestamps(TEST_TIMESTAMPS_FILENAME_BUG)
        rr_test_timestamps = rr_test_timestamps[:noElements]
    else:
        rr_test_intervals = np.array([random.normalvariate(600, 60) for _ in range(noElements)])
        rr_test_intervals = rr_test_intervals.astype(int)
        rr_test_timestamps = pd.date_range(start=pd.datetime.now(), periods=noElements, freq = '600ms')
        rr_test_timestamps = rr_test_timestamps.strftime("%Y-%m-%d %H:%M:%S.%f")
        
        labels_list_train = rr_test_intervals < 600
        labels_list_train = labels_list_train.astype(str)
        classify_features = classify_models_evaluation_knn(rr_test_intervals, rr_test_timestamps, np.array(labels_list_train))

def test_regression_model_hrv_statistics(noElements):

    # rr_intervals_list contains integer values of RR-interval
    if noElements <= 1000:
        rr_test_intervals = np.array(load_test_data(TEST_DATA_FILENAME_BUG))
        rr_test_intervals = rr_test_intervals[:noElements]
        rr_test_timestamps = load_test_timestamps(TEST_TIMESTAMPS_FILENAME_BUG)
        rr_test_timestamps = rr_test_timestamps[:noElements]
    else:
        rr_test_intervals = np.array([random.normalvariate(600, 60) for _ in range(noElements)])
        rr_test_intervals = rr_test_intervals.astype(int)
        rr_test_timestamps = pd.date_range(start=pd.datetime.now(), periods=noElements, freq = '600ms')
        rr_test_timestamps = rr_test_timestamps.strftime("%Y-%m-%d %H:%M:%S.%f")

        labels_list_train = []
        time_domain_features = transform_to_snapshot_statistics(rr_test_intervals, rr_test_timestamps)
        time_domain_features = json.loads(time_domain_features)
        for obj in time_domain_features['hrArray']:
            labels_list_train.append(obj['values'])
        #classify_features = classify_models_evaluation_reg(rr_test_intervals[1:], rr_test_timestamps[1:], np.array(labels_list_train))
        classify_features = classify_models_evaluation_linreg(rr_test_intervals[1:], rr_test_timestamps[1:], np.array(labels_list_train))
