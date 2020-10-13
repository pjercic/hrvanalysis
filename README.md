# Heart Rate Variability analysis

## Installation / Prerequisites

You can clone the repository and run setup:

```python3
git clone https://<your-bitbucket-account>@bitbucket.org/jamzone/composer.git
python3 setup.py install
```

or install hrv-analysis using pip :

```python3
pip3 install hrv-analysis
```

# Machine Learning

## Installation / Prerequisites

Make sure the 64bit version of Python 3 is installed

Then run:

```python3
pip3 install -U scikit-learn
```

In order to check your installation you can use

```python3
python3 -m pip show scikit-learn # to see which version and where scikit-learn is installed
python3 -m pip freeze # to see all packages installed in the active virtualenv
python3 -c "import sklearn; sklearn.show_versions()"
```

## Getting started

When the new data [Datetime, RRs] from the DB has been received, call the method **transform_to_snapshot_statistics** from **offline_analysis**.
In the case of a bigger quantities of data, use the *NAMED PIPES* and provide the string path to the *_ipc* version of the methods.
In the case of IPC, the data is expected in JSON format in the following way:

```python3
{"rrs": [{'index': '2020-07-27T22:09:54.983Z\n', 'values': 256}, {'index': '2020-07-27T22:09:56.85Z\n', 'values': 1530}, {'index': '2020-07-27T22:09:58.825Z\n', 'values': 2225}, {'index': '2020-07-27T22:10:07.471Z\n', 'values': 7366}, {'index': '2020-07-27T22:10:08.546Z\n', 'values': 2256}, {'index': '2020-07-27T22:10:09.178Z\n', 'values': 522}, {'index': '2020-07-27T22:10:14.94Z\n', 'values': 5663}, ... {'index': '2020-07-27T22:12:58.153Z\n', 'values': 834}, {'index': '2020-07-27T22:12:59.004Z', 'values': 827}]}
```

```python3
transform_to_snapshot_statistics(rr_intervals, rr_timestamps)
transform_to_snapshot_statistics_ipc(path_named_pipe)
```

When the morning snapshot data [Datetime, RRs] from the DB has been received, call the method **transform_to_morning_snapshots_statistics** from **offline_analysis**.

```python3
transform_to_morning_snapshots_statistics(rr_intervals, rr_timestamps)
```

_Input:_ The method expects a list of RR values, and a list of associated datetime values

_Output:_ The JSON object with the following fields below

Explanation of the mobile app values returning from the library

- HRV RMSSD avg: 'rmssd', **RMSSD for a given period (use only when medically grade sensor, since prone to outliers)**
- HRV SDNN avg: 'sdnn', **SDNN is used for longer periods of known and similar length, which allows the comparison**
- HRV graph: 'rmssdArray', **A list indexed with datetime values received as input**
- HRV avg: 'rmssdAvg', **Median RMSSD for a given period, corrected according to min, max and range values**
- HRV min: 'rmssdMin',
- HRV max: 'rmssdMax',
- HRV range: 'rmssdRange', **(rmssdRange / 200) * 100**
- Calm percentage: 'rmssdRangeRatioCalm', **rmssdRangeRatioCalm * 100**
- Focus percentage: 'rmssdRangeRatioFocus', **rmssdRangeRatioFocus * 100**
- Focus speed: 'rmssdMaxSpeedStress', **this value is RMSSD per second [Rps], like kilometers per hour [kmh]**
- Calm speed: 'rmssdMaxSpeedRelax', **this value is RMSSD per second [Rps], like kilometers per hour [kmh]**
- Rest score: 'restScore', **this values is index score [1-10] specifying how good the average HRV is**
- Balance score: 'balanceScore', **this values is index score [1-10] specifying how balanced is user to keep the same level of HRV**
- HR graph: 'hrArray', **A list indexed with datetime values received as input**
- HR avg: 'hrMean': mean_hr,
- HR max: 'hrMax': max_hr,
- HR min: 'hrMin': min_hr,
- HR standard deviation: 'hrStd': std_hr,
- Error code: 'errorCode': error_code
- LF/HF ratio: 'lfHfRatio'
- Version number: 'version'

_Code for testing the snapshot functionality on the server > python3 -c 'from test_offline_analysis_methods import *; test_transform_to_snapshot_statistics(2000)'_

_Code for testing the morning snapshot functionality on the server > python3 -c 'from test_offline_analysis_methods import *; test_transform_to_morning_snapshots_statistics()'_

_Code for testing the 3dayme functionality on the server > python3 -c 'from test_offline_analysis_methods import *; test_transform_to_3dayme_statistics()'_

# Git Setup for remote repositories

- Initialize the existing repo on BitBucket
- Clone the repo from BitBucket to your local machine
- Set URLs

```console
git remote rename origin upstream
git remote add origin URL_TO_GITHUB_REPO
git push origin master
```

- Check to control the sources setup

```console
git remote -v
```
- Now you can work with it just like any other github repo. To pull in patches from origin, and push to upstream

```console
git pull origin master
git push upstream master 
```

Update version number

```console
git tag -a v1.0.0 -m 'Version 1.0.0'
git push --tags
```
