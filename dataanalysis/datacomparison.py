#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Dec 17, 2020

@author: petar
'''

"""This script provides several statistical methods to compare data between each other
 for heart rate variability analysis."""

import json

def compare(snapshotGroups: str, configAnswer: str) -> dict:
    
    answer = {
    
       'test': 1,
       'errorCode': 0
    }
    
    data = json.loads(snapshotGroups);
    config = json.loads(configAnswer);
    
    for (k, v) in data.items():
        answer[k] = v;
    
    import time
    time.sleep(1);
    
    if config['mean'].lower() == 'true':
        answer['mean'] = 1.23
    
    return json.dumps(answer, ensure_ascii=False)
