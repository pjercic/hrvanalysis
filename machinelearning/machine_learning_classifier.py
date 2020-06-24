#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Jun 24, 2020

@author: petar
'''

"""This script provides several methods to extract features from Normal to Normal Intervals
 for heart rate variability analysis."""

from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def classify_features_supervised(nn_intervals_train: List[float], timestamp_list_train: List[str], labels_list_train: List[str], nn_intervals: List[float], timestamp_list: List[str]) -> dict:
    print('machine learning')
    
    # create a mapping from fruit label value to fruit name to make results easier to interpret
    labels = pd.Series(labels_list_train).unique()
    lookup_label_name = dict(zip([0, 1], labels))
    
    X = np.array([nn_intervals_train]).T
    y = np.array([labels_list_train]).T
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    knn = KNeighborsClassifier(n_neighbors = 5)
    
    fit = knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    
    label_prediction = knn.predict(np.array([nn_intervals]).T)
    
    # How sensitive is k-NN classification accuracy to the choice of the 'k' parameter
    k_range = range(1,20)
    scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    
    print(scores)
    
    # How sensitive is k-NN classification accuracy to the train/test split proportion?
    t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    
    knn = KNeighborsClassifier(n_neighbors = 5)
    
    scores = []
    for s in t:
    
        mean_scores = []
        for i in range(1,10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
            knn.fit(X_train, y_train)
            mean_scores.append(knn.score(X_test, y_test))
        scores.append(np.mean(mean_scores))
        
    print(scores)