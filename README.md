# Heart Rate Variability analysis

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

_Code for testing on the server > python3 -c 'from offline_analysis import *; test_transform_to_rmssd_statistics(1000)'
