#coding=utf-8
'''
@author: DMao
@time: 05.07.20    20:23
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

counts = pd.read_csv('FremontBridge.csv', index_col= 'Date', parse_dates= True)
print(type(counts))
daily = counts.resample('d').sum()

daily['Total'] = daily.sum(axis = 1)
daily = daily[['Total']]

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i ).astype(float)

def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """Compute the hours of daylight for the given date"""
    days = (date - pd.datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
daily[['daylight_hrs']].plot()
plt.ylim(8, 17)
plt.show()
