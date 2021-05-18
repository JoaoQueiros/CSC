#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:37:38 2021

@author: luis
"""

"""
*********************************************************************
1) Import Packages

*********************************************************************
"""

import pandas as pd


"""
*********************************************************************
2) Import Dataset

*********************************************************************
"""

df = pd.read_csv('Traffic_Incidents_Braga_Until_20191231.csv')


"""
*********************************************************************
3) Remove Duplicate

*********************************************************************
"""
df = df.drop_duplicates(subset=None, keep='first', inplace=False)

"""
*********************************************************************
4) Insert Date and count

*********************************************************************
"""
df['date'] = df['incident_date'].astype(str).str[0:13]
df['n_incidents'] = 1

"""
*********************************************************************
5) Group By and Order by Date

*********************************************************************
"""

df_pre = df.groupby(['date'], as_index= False).agg({'n_incidents':'sum','length_in_meters':'mean',
                                                  'delay_in_seconds':'mean','latitude':'mean','longitude':'mean'}).sort_values(by="date")


"""
*********************************************************************
6) Create a Dim Calendar   **** automatization max date and first hour***

*********************************************************************
"""
min_date, max_date = min(df['incident_date'].astype(str).str[0:10]), max(df['incident_date'].astype(str).str[0:10])

rng = pd.DataFrame(pd.date_range(min_date, '2020-01-01', freq='T'))

rng.columns=['date']

rng['date'] = rng['date'].astype(str).str[0:13]

rng = rng.groupby('date', as_index = False).agg({'date':'max'})

rng = rng.iloc[19:-1,:]


"""
*********************************************************************
7) Left Join with calendar

*********************************************************************
"""
df_pre = rng.merge(df_pre, how='left').fillna(0)


"""
*********************************************************************
8) write dataset

*********************************************************************
"""
N = len(df_pre)

df_pre[:-50].to_csv('training_set.csv', index = False)
df_pre.iloc[N-50:].to_csv('test_set.csv', index = False)


