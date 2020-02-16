#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:11:04 2020

@author: piedpiper
"""

import requests
import urllib
import pandas as pd

# USGS Elevation Point Query Service
url = r'https://nationalmap.gov/epqs/pqs.php?'

df_1 = pd.read_csv('trackdata_c1_clean.csv')

# coordinates with known elevation 
#lat = [48.633, 48.733, 45.1947, 45.1962]
#lon = [-93.9667, -94.6167, -93.3257, -93.2755]

# create data frame
df = df_1.filter(['lat', 'lon'], axis = 1)

def elevation_function(df, lat_column, lon_column):
    """Query service using lat, lon. add the elevation values as a new column."""
    elevations = []
    for lat, lon in zip(df[lat_column], df[lon_column]):

        # define rest query params
        params = {
            'output': 'json',
            'x': lon,
            'y': lat,
            'units': 'Meters'
        }

        # format query string and return query value
        result = requests.get((url + urllib.parse.urlencode(params)))
        elevations.append(result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])

    df['elev_meters'] = elevations


elevation_function(df, 'lat', 'lon')
df.head()