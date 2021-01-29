import os
import random
import multiprocessing
import json
import numpy as np
import pandas as pd
import json
from offline_analysis import transform_to_snapshot_statistics, transform_to_3dayme_statistics, transform_to_morning_snapshots_statistics, classify_hrv_statistics, regression_hrv_statistics, compare_snapshots
from offline_analysis import transform_to_snapshot_statistics_ipc, transform_to_snapshot_statistics_ipc_echo, transform_to_snapshot_statistics_ipc_error
from machinelearning import classify_models_evaluation_knn, classify_models_evaluation_reg, classify_models_evaluation_linreg
from hrvanalysis.plot import plot_timeseries
from numpy import int

TEST_DATA_FILENAME_10 = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_10.txt')
TEST_DATA_FILENAME_20 = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_20.txt')
TEST_DATA_FILENAME_60 = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_60.txt')
TEST_DATA_FILENAME_LARGE = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_large.txt')
TEST_DATA_FILENAME_BUG = os.path.join(os.path.dirname(__file__), './tests/bug20200729_test_nn_intervals.txt')
TEST_TIMESTAMPS_FILENAME_BUG = os.path.join(os.path.dirname(__file__), './tests/bug20200729_test_timestamps.txt')
TEST_DATA_FILENAME_GOODFLOW = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_goodflow.txt')
TEST_TIMESTAMPS_FILENAME_GOODFLOW = os.path.join(os.path.dirname(__file__), './tests/test_nn_timestamp_goodflow.txt')
TEST_DATA_FILENAME_BADFLOW = os.path.join(os.path.dirname(__file__), './tests/test_nn_intervals_badflow.txt')
TEST_TIMESTAMPS_FILENAME_BADFLOW = os.path.join(os.path.dirname(__file__), './tests/test_nn_timestamp_badflow.txt')

path = "./mypipe"

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
        rr_test_intervals = np.array(load_test_data(TEST_DATA_FILENAME_BADFLOW))
        rr_test_timestamps = load_test_timestamps(TEST_TIMESTAMPS_FILENAME_BADFLOW)
    else:
        rr_test_intervals = np.array([random.normalvariate(600, 60) for _ in range(noElements)])
        rr_test_intervals = rr_test_intervals.astype(int)
        rr_test_timestamps = pd.date_range(start=pd.datetime.now(), periods=noElements, freq = '600ms')
        rr_test_timestamps = rr_test_timestamps.strftime("%Y-%m-%d %H:%M:%S.%f")
    
    time_domain_features = transform_to_snapshot_statistics(rr_test_intervals, rr_test_timestamps)
    
    print(time_domain_features)

def test_transform_to_snapshot_statistics_ipc():
    
    input = '{"snapshotID":101,"analysis":"recording","answer":{"mean":true,"range":false,"difference":true}}'
    
    try:
        os.mkfifo(path)
    except FileExistsError:
        os.remove(path)
        os.mkfifo(path)
        pass
        
    multiprocessing.Process(target=transform_to_snapshot_statistics_ipc, args=(input, path,)).start()
    
    with open(path, 'rt') as p:
        print(p.read())

def test_transform_to_snapshot_statistics_db(isServer = True):

    from sqlalchemy import create_engine
    from sshtunnel import SSHTunnelForwarder

    remote_postgres = 'jz-pg1.cg18srohk0ph.eu-west-1.rds.amazonaws.com'
    remote_postgres_port = 5432

    if isServer:
        conn = create_engine('copyUriFromEvernote').connect()
        #df = pd.read_sql_table('rr_data', conn, schema='gb')
        df = pd.read_sql_query("SELECT * FROM gb.rr_data WHERE snapshot_id = 866;", conn, parse_dates = ['datum_tijd'])
        conn.dispose()
    else:
        remote_user = 'copyUserFromEvernote'
        remote_host = 'copyUriFromEvernote'
        remote_port = 22
        local_host = '127.0.0.1'
        local_port = 5000

        try:
            with SSHTunnelForwarder(
                (remote_host, remote_port),
                ssh_private_key = "G:/My Drive/Development/jamzone_aws.pem",
                ssh_username = remote_user, 
                remote_bind_address=(remote_postgres, remote_postgres_port),
                local_bind_address=(local_host, local_port)) as server:

                server.start()
                print ("server connected")

                conn = create_engine('postgresql+psycopg2://python_dev3:t6CkGDwIbjXjVwyWI4rq@localhost:5000/stressjam_dev3')
                with conn.connect():
                    print ("database connected")
                    #df = pd.read_sql_table('rr_data', conn, schema='gb')
                    df = pd.read_sql_query("SELECT * FROM gb.rr_data WHERE snapshot_id=866;", conn, index_col = 'id')
                    print(df.head())

                conn.dispose()
                server.stop()
        except:
            print ("Connection Failed")
        finally:
            print("Done!")
    
    #df = df[~dfzoom.index.duplicated(keep='first')]
    #df3.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')

    # sort based on index old > new
    df.reindex(index=df.index[::-1])

    #df = df.drop_duplicates(subset='datum_tijd', keep="first")

    time_domain_features = transform_to_snapshot_statistics(df['waarde'], df['datum_tijd'].tolist())
    
    print(time_domain_features)

def test_transform_to_3dayme_statistics():
    
    rr_test_intervals = np.array([random.normalvariate(600, 60) for _ in range(500000)])
    rr_test_intervals = rr_test_intervals.astype(int)
    rr_test_timestamps = pd.date_range(start=pd.datetime.now(), periods=500000, freq = '600ms')
    rr_test_timestamps = rr_test_timestamps.strftime("%Y-%m-%d %H:%M:%S.%f")

    time_domain_features = transform_to_3dayme_statistics(rr_test_intervals, rr_test_timestamps)
    
    print(time_domain_features)

def test_transform_to_morning_snapshots_statistics():
    
    rr_test_intervals = np.array([random.normalvariate(600, 60) for _ in range(350)])
    rr_test_intervals = rr_test_intervals.astype(int)
    rr_test_timestamps = pd.date_range(start=pd.datetime.now(), periods=350, freq = '600ms')
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

def test_compare_snapshots():
    
    input = '{"grouping":[{"id":0,"snapshotsID":[101,301,200]},{"id":1,"snapshotsID":[102,302,202]},{"id":2,"snapshotsID":[666]}],"answer":{"mean":true,"range":false,"difference":true}}';
    
    try:
        os.mkfifo(path)
    except FileExistsError:
        os.remove(path)
        os.mkfifo(path)
        pass
        
    multiprocessing.Process(target=compare_snapshots, args=(input, path,)).start()
    
    with open(path, 'rt') as p:
        print(p.read())
