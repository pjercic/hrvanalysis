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

# Heart Rate Variability analysis

## Installation / Prerequisites

The easiest way to install hrv-analysis is using pip :

```python3
pip3 install hrv-analysis
```

you can also clone the repository:

```python3
git clone https://github.com/aura-healthcare/hrvanalysis.git
python3 setup.py install
```

## Getting started

When the new data [Datetime, RRs] from the DB has been received, call the method **transform_to_snapshot_statistics** from **offline_analysis**.

```python3
transform_to_snapshot_statistics(rr_intervals, rr_timestamps)
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
- HRV graph: 'rmssdArray', **A list indexed with datetime values received as input
- HRV avg: 'rmssdAvg', **Median RMSSD for a given period, corrected according to min, max and range values**
- HRV min: 'rmssdMin',
- HRV max: 'rmssdMax',
- HRV range: 'rmssdRange', **(rmssdRange / 200) * 100
- Calm percentage: 'rmssdRangeRatioCalm', **rmssdRangeRatioCalm * 100
- Focus percentage: 'rmssdRangeRatioFocus', **rmssdRangeRatioFocus * 100
- Focus speed: 'rmssdMaxSpeedStress', **this value is RMSSD per second [Rps], like kilometers per hour [kmh]
- Calm speed: 'rmssdMaxSpeedRelax', **this value is RMSSD per second [Rps], like kilometers per hour [kmh]
- HR graph: 'hrArray', **A list indexed with datetime values received as input
- HR avg: 'hrMean': mean_hr,
- HR max: 'hrMax': max_hr,
- HR min: 'hrMin': min_hr,
- HR standard deviation: 'hrStd': std_hr,
- Error code: 'errorCode': error_code

_Code for testing the snapshot functionality on the server > python3 -c 'from test_offline_analysis_methods import *; test_transform_to_snapshot_statistics(2000)'_

_Code for testing the morning snapshot functionality on the server > python3 -c 'from test_offline_analysis_methods import *; test_transform_to_morning_snapshots_statistics()'_

_Code for testing the 3dayme functionality on the server > python3 -c 'from test_offline_analysis_methods import *; test_transform_to_3dayme_statistics()'_
