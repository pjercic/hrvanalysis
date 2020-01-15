# Heart Rate Variability analysis

Explanation of the mobile app values returning from the library

- HRV avg: 'rmssd',
- HRV grpah: 'rmssdArray', * It would be best to have numbers start form beginning, not jump from 0 to a value
- HRV min: 'rmssdMin',
- HRV max: 'rmssdMax',
- HRV range: 'rmssdRange', * (rmssdRange / 200) * 100
- Calm percentage: 'rmssdRangeRatioCalm', * rmssdRangeRatioCalm * 100
- Focus percentage: 'rmssdRangeRatioFocus', * rmssdRangeRatioFocus * 100
- Focus speed: 'rmssdMaxSpeedStress', * RMSSD / per second
- Calm speed: 'rmssdMaxSpeedRelax', * RMSSD / per second
- HR avg: 'hrMean': mean_hr,
- HR max: 'hrMax': max_hr,
- HR min: 'hrMin': min_hr,
- HR standard deviation: 'hrStd': std_hr,

_The improvement would be to make the graphing start from the beginning, not jump from 0 to a value, and to do the calculation in the backend_