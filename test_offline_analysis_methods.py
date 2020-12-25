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

def test_transform_to_snapshot_statistics_ipc(noElements):
    
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
    
    json_rr_test = pd.Series(rr_test_intervals)
    json_rr_test.index = rr_test_timestamps
    rrArray_json = json.loads(json_rr_test.fillna(0).to_json(date_format='iso', orient='table'))
    input_json = rrArray_json['data']
    #input_jso2 = json.loads('[{"index":2020-07-18T10:17:33.253Z,"values":577},{"index":2020-07-18T10:17:33.838Z,"values":569},{"index":2020-07-18T10:17:34.391Z,"values":569},{"index":2020-07-18T10:17:34.963Z,"values":553},{"index":2020-07-18T10:17:35.503Z,"values":553},{"index":2020-07-18T10:17:36.042Z,"values":545},{"index":2020-07-18T10:17:36.583Z,"values":530},{"index":2020-07-18T10:17:37.124Z,"values":530},{"index":2020-07-18T10:17:37.617Z,"values":530},{"index":2020-07-18T10:17:38.16Z,"values":514},{"index":2020-07-18T10:17:38.652Z,"values":514},{"index":2020-07-18T10:17:39.196Z,"values":506},{"index":2020-07-18T10:17:39.687Z,"values":506},{"index":2020-07-18T10:17:40.229Z,"values":514},{"index":2020-07-18T10:17:40.768Z,"values":530},{"index":2020-07-18T10:17:41.309Z,"values":553},{"index":2020-07-18T10:17:41.937Z,"values":608},{"index":2020-07-18T10:17:42.527Z,"values":616},{"index":2020-07-18T10:17:43.198Z,"values":655},{"index":2020-07-18T10:17:43.873Z,"values":709},{"index":2020-07-18T10:17:44.599Z,"values":709},{"index":2020-07-18T10:17:45.313Z,"values":725},{"index":2020-07-18T10:17:46.036Z,"values":756},{"index":2020-07-18T10:17:46.798Z,"values":764},{"index":2020-07-18T10:17:47.564Z,"values":733},{"index":2020-07-18T10:17:48.284Z,"values":694},{"index":2020-07-18T10:17:48.968Z,"values":678},{"index":2020-07-18T10:17:49.543Z,"values":584},{"index":2020-07-18T10:17:50.133Z,"values":561},{"index":2020-07-18T10:17:50.713Z,"values":592},{"index":2020-07-18T10:17:51.255Z,"values":577},{"index":2020-07-18T10:17:51.838Z,"values":577},{"index":2020-07-18T10:17:52.423Z,"values":577},{"index":2020-07-18T10:17:53.01Z,"values":569},{"index":2020-07-18T10:17:53.594Z,"values":577},{"index":2020-07-18T10:17:54.139Z,"values":569},{"index":2020-07-18T10:17:54.719Z,"values":577},{"index":2020-07-18T10:17:55.307Z,"values":569},{"index":2020-07-18T10:17:55.89Z,"values":592},{"index":2020-07-18T10:17:56.474Z,"values":592},{"index":2020-07-18T10:17:57.058Z,"values":569},{"index":2020-07-18T10:17:57.643Z,"values":600},{"index":2020-07-18T10:17:58.235Z,"values":577},{"index":2020-07-18T10:17:58.819Z,"values":592},{"index":2020-07-18T10:17:59.443Z,"values":592},{"index":2020-07-18T10:18:00.027Z,"values":592},{"index":2020-07-18T10:18:00.613Z,"values":577},{"index":2020-07-18T10:18:01.246Z,"values":631},{"index":2020-07-18T10:18:01.874Z,"values":663},{"index":2020-07-18T10:18:02.503Z,"values":625},{"index":2020-07-18T10:18:03.133Z,"values":608},{"index":2020-07-18T10:18:03.72Z,"values":608},{"index":2020-07-18T10:18:04.348Z,"values":592},{"index":2020-07-18T10:18:04.888Z,"values":577},{"index":2020-07-18T10:18:05.473Z,"values":577},{"index":2020-07-18T10:18:06.058Z,"values":577},{"index":2020-07-18T10:18:06.689Z,"values":569},{"index":2020-07-18T10:18:07.184Z,"values":569},{"index":2020-07-18T10:18:07.769Z,"values":561},{"index":2020-07-18T10:18:08.308Z,"values":553},{"index":2020-07-18T10:18:08.892Z,"values":561},{"index":2020-07-18T10:18:09.433Z,"values":561},{"index":2020-07-18T10:18:10.018Z,"values":561},{"index":2020-07-18T10:18:10.559Z,"values":553},{"index":2020-07-18T10:18:11.099Z,"values":553},{"index":2020-07-18T10:18:11.683Z,"values":530},{"index":2020-07-18T10:18:12.224Z,"values":553},{"index":2020-07-18T10:18:12.831Z,"values":600},{"index":2020-07-18T10:18:13.394Z,"values":584},{"index":2020-07-18T10:18:14.022Z,"values":625},{"index":2020-07-18T10:18:14.698Z,"values":663},{"index":2020-07-18T10:18:15.329Z,"values":655},{"index":2020-07-18T10:18:15.961Z,"values":625},{"index":2020-07-18T10:18:16.599Z,"values":616},{"index":2020-07-18T10:18:17.208Z,"values":616},{"index":2020-07-18T10:18:17.763Z,"values":616},{"index":2020-07-18T10:18:18.394Z,"values":625},{"index":2020-07-18T10:18:19.062Z,"values":625},{"index":2020-07-18T10:18:19.695Z,"values":625},{"index":2020-07-18T10:18:20.283Z,"values":608},{"index":2020-07-18T10:18:20.912Z,"values":600},{"index":2020-07-18T10:18:21.495Z,"values":592},{"index":2020-07-18T10:18:22.124Z,"values":592},{"index":2020-07-18T10:18:22.75Z,"values":647},{"index":2020-07-18T10:18:23.388Z,"values":663},{"index":2020-07-18T10:18:24.016Z,"values":663},{"index":2020-07-18T10:18:24.693Z,"values":655},{"index":2020-07-18T10:18:25.369Z,"values":647},{"index":2020-07-18T10:18:25.948Z,"values":608},{"index":2020-07-18T10:18:26.578Z,"values":592},{"index":2020-07-18T10:18:27.119Z,"values":577},{"index":2020-07-18T10:18:27.707Z,"values":561},{"index":2020-07-18T10:18:28.289Z,"values":561},{"index":2020-07-18T10:18:28.832Z,"values":553},{"index":2020-07-18T10:18:29.413Z,"values":545},{"index":2020-07-18T10:18:30.003Z,"values":592},{"index":2020-07-18T10:18:30.629Z,"values":639},{"index":2020-07-18T10:18:31.306Z,"values":670},{"index":2020-07-18T10:18:32.025Z,"values":725},{"index":2020-07-18T10:18:32.699Z,"values":702},{"index":2020-07-18T10:18:33.417Z,"values":702},{"index":2020-07-18T10:18:34.094Z,"values":663},{"index":2020-07-18T10:18:34.68Z,"values":616},{"index":2020-07-18T10:18:35.314Z,"values":616},{"index":2020-07-18T10:18:35.942Z,"values":608},{"index":2020-07-18T10:18:36.526Z,"values":600},{"index":2020-07-18T10:18:37.156Z,"values":608},{"index":2020-07-18T10:18:37.739Z,"values":616},{"index":2020-07-18T10:18:38.325Z,"values":561},{"index":2020-07-18T10:18:38.872Z,"values":600},{"index":2020-07-18T10:18:39.474Z,"values":584},{"index":2020-07-18T10:18:40.127Z,"values":600},{"index":2020-07-18T10:18:40.709Z,"values":577},{"index":2020-07-18T10:18:41.364Z,"values":647},{"index":2020-07-18T10:18:41.979Z,"values":616},{"index":2020-07-18T10:18:42.577Z,"values":592},{"index":2020-07-18T10:18:43.139Z,"values":577},{"index":2020-07-18T10:18:43.782Z,"values":608},{"index":2020-07-18T10:18:44.362Z,"values":616},{"index":2020-07-18T10:18:45Z,"values":647},{"index":2020-07-18T10:18:45.632Z,"values":631},{"index":2020-07-18T10:18:46.296Z,"values":639},{"index":2020-07-18T10:18:46.876Z,"values":625},{"index":2020-07-18T10:18:47.507Z,"values":592},{"index":2020-07-18T10:18:48.106Z,"values":625},{"index":2020-07-18T10:18:48.732Z,"values":600},{"index":2020-07-18T10:18:49.325Z,"values":592},{"index":2020-07-18T10:18:49.893Z,"values":584},{"index":2020-07-18T10:18:50.476Z,"values":577},{"index":2020-07-18T10:18:51.016Z,"values":569},{"index":2020-07-18T10:18:51.628Z,"values":553},{"index":2020-07-18T10:18:52.153Z,"values":545},{"index":2020-07-18T10:18:52.681Z,"values":522},{"index":2020-07-18T10:18:53.21Z,"values":569},{"index":2020-07-18T10:18:53.793Z,"values":561},{"index":2020-07-18T10:18:54.357Z,"values":561},{"index":2020-07-18T10:18:54.952Z,"values":561},{"index":2020-07-18T10:18:55.573Z,"values":647},{"index":2020-07-18T10:18:56.243Z,"values":639},{"index":2020-07-18T10:18:56.909Z,"values":655},{"index":2020-07-18T10:18:57.559Z,"values":686},{"index":2020-07-18T10:18:58.233Z,"values":647},{"index":2020-07-18T10:18:58.866Z,"values":600},{"index":2020-07-18T10:18:59.45Z,"values":608},{"index":2020-07-18T10:19:00.067Z,"values":616},{"index":2020-07-18T10:19:00.689Z,"values":639},{"index":2020-07-18T10:19:01.333Z,"values":625},{"index":2020-07-18T10:19:01.905Z,"values":569},{"index":2020-07-18T10:19:02.491Z,"values":616},{"index":2020-07-18T10:19:03.081Z,"values":561},{"index":2020-07-18T10:19:03.676Z,"values":569}]')
    #input_jso3 = json.loads('[{"index": "2020-07-27T22:09:54.983Z\n", "values": 256}, {"index": "2020-07-27T22:09:56.85Z\n", "values": 1530}, {"index": "2020-07-27T22:09:58.825Z\n", "values": 2225}, {"index": "2020-07-27T22:10:07.471Z\n", "values": 7366}, {"index": "2020-07-27T22:10:08.546Z\n", "values": 2256}, {"index": "2020-07-27T22:10:09.178Z\n", "values": 522}, {"index": "2020-07-27T22:10:14.94Z\n", "values": 5663}, {"index": "2020-07-27T22:12:58.153Z\n", "values": 834}, {"index": "2020-07-27T22:12:59.004Z", "values": 827}]')
        
    try:
        os.mkfifo(path)
    except FileExistsError:
        os.remove(path)
        os.mkfifo(path)
        pass
        
    multiprocessing.Process(target=transform_to_snapshot_statistics_ipc, args=(path,)).start()
    
    with open(path, 'wt') as p:
        p.write('{"rrs":' + json.dumps(input_json, ensure_ascii=False) + '}\n')
    
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
