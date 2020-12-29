#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Dec 17, 2020

@author: petar
'''

"""This script provides several statistical methods to compare data between each other
 for heart rate variability analysis."""

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
