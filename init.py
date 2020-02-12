# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def plot_on_map(df):

    # How much to zoom from coordinates (in degrees)
    zoom_scale = 1
    
    # Setup the bounding box for the zoom and bounds of the map
    bbox = [df['lat'].min()-zoom_scale,df['lat'].max()+zoom_scale,\
            df['lon'].min()-zoom_scale,df['lon'].max()+zoom_scale]
    
    plt.figure(figsize=(12,6))
    # Define the projection, scale, the corners of the map, and the resolution.
    m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\
    llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')

    # m.drawmapboundary(fill_color='#46bcec') # Make your map into any style you like
    # m.fillcontinents(color='#f2f2f2',lake_color='#46bcec') # Make your map into any style you like
    m.drawcoastlines()
    # m.drawrivers() # Default colour is black but it can be customised
    m.drawcountries()
    
    # df['lat_lon'] = list(zip(df.Easting, df.Northing)) # Creating tuples
    # df_2000 = df[df['AADFYear']==2000]
    
    for x,y in zip(df['lat'],df['lon']):
        m.plot(x, y, marker = 'o', c='r', markersize=1, alpha=0.8)
    
    plt.savefig('test.png')
    return

#Feature Engineering (add column)
def route_distance():
    return    


if __name__ == "__main__":
    df_1 = pd.read_csv('trackdata_c1.csv')
    df_2 = pd.read_csv('trackdata_c2.csv')
    df_x = pd.read_csv('trackdata_cx.csv')
    plot_on_map(df_1)