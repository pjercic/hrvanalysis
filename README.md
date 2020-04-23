# Heart Rate Variability analysis

When the new data [Datetime, RRs] from the DB has been received, call the method **transform_to_rmssd_statistics** from **offline_analysis**.

```python3
transform_to_rmssd_statistics(rr_intervals, rr_timestamps)
```

_Input:_ The method expects a list of RR values, and a list of associated datetime values

_Output:_ The JSON object with the following fields below

Explanation of the mobile app values returning from the library

- HRV avg: 'rmssd', **RMSSD for a given period (use only when medically grade sensor, since prone to outliers)**
- HRV graph: 'rmssdArray',
- HRV avg: 'rmssdAvg', **Median RMSSD for a given period, corrected according to min, max and range values**
- HRV min: 'rmssdMin',
- HRV max: 'rmssdMax',
- HRV range: 'rmssdRange', * (rmssdRange / 200) * 100
- Calm percentage: 'rmssdRangeRatioCalm', * rmssdRangeRatioCalm * 100
- Focus percentage: 'rmssdRangeRatioFocus', * rmssdRangeRatioFocus * 100
- Focus speed: 'rmssdMaxSpeedStress', * this value is RMSSD per second [Rps], like kilometers per hour [kmh]
- Calm speed: 'rmssdMaxSpeedRelax', * this value is RMSSD per second [Rps], like kilometers per hour [kmh]
- HR avg: 'hrMean': mean_hr,
- HR max: 'hrMax': max_hr,
- HR min: 'hrMin': min_hr,
- HR standard deviation: 'hrStd': std_hr,

_Code for testing on the server > python3 -c 'from test_offline_analysis_methods import *; test_transform_to_rmssd_statistics(2000)'_
