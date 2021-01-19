#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Dec 17, 2020

@author: petar
'''

"""This script provides several statistical methods to compare data between each other
 for heart rate variability analysis."""

from typing import List
from sklearn import linear_model
import numpy as np
import pandas as pd
import json

def compare(snapshotGroups: str) -> dict:
    
    answer = {'grouping':[]}
    
    try:
        data = json.loads(snapshotGroups);
    except:
        answer['errorCode'] = 402
        raise SyntaxError('Error parsing snapshot grouping JSON data')      
    
    for group in data['grouping']:
        groupingResults = {'id':group['id']}
        
        if data['answer']['mean']:
            groupingResults['mean'] = 1.23
    
        if data['answer']['difference']:
            groupingResults['difference'] = 'no'
        
        answer['grouping'].append({'comparison':groupingResults})

    answer['errorCode'] = 0
    
    import time
    time.sleep(1);
    
    return json.dumps(answer, ensure_ascii=False)

def linear_regression(nn_intervals: List[float], timestamp_list: List[str]) -> dict:

    answer = {}

    # X will have to be a column vector later for the analysis
    X = np.array(nn_intervals)
    
    # Extract milliseconds from timestamps as X axis
    Y = np.array(pd.to_datetime(timestamp_list).astype(np.int64) // 10**6, dtype=float)
    # Convert to seconds as a SI unit
    Y = (Y - Y[0]) / 1000

    linear_regressor = linear_model.LinearRegression()  # create object for the class
    linear_regressor.fit(X[:, np.newaxis], Y)  # perform linear regression

    import math 
    answer['linregSlope'] = math.atan(linear_regressor.coef_/1) # unit of time

    return json.dumps(answer, ensure_ascii=False)

    ## ---

    ### Without a constant

    #import statsmodels.api as sm

    ## x has columns
    #X = df["RM"]

    ## y has rows
    #y = target["MEDV"]

    ## Note the difference in argument order
    #model = sm.OLS(y, X)
    #results = model.fit()


    #results.adjustedrsquared

    ## array with b0, b1, and b2 respectively
    #results.params

    #predictions = model.predict(X) # make the predictions by the model

    ## Print out the statistics
    #model.summary()


    ### With a constant

    #import statsmodels.api as sm # import statsmodels 

    #X = df["RM"] ## X usually means our input variables (or independent variables)
    #y = target["MEDV"] ## Y usually means our output/dependent variable
    #X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

    ## Note the difference in argument order
    #model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
    #predictions = model.predict(X)

    ## Print out the statistics
    #model.summary()

    ## ---

    #import scipy.stats as sp

    #y=np.array(df['OW2 As(mg/L)'].dropna().values, dtype=float)
    #x=np.array(pd.to_datetime(df['OW2 As(mg/L)'].dropna()).index.values, dtype=float)
    #slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
