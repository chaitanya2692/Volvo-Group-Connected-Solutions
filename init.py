# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
#import herepy
#from herepy import RoutingApi
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from geopy import distance
from multiprocessing import Process, Manager
import pyproj as proj
import json

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
    
    x,y=m(df['lon'].tolist(),df['lat'].tolist())
    m.plot(x,y,'r*',markersize=5)
#    plt.savefig('test.png')

    return

#Feature Engineering (Works)
def route_distance(df, uid):
    lon = df_1.loc[df['uid']==uid, 'lon'].to_numpy()
    lat = df_1.loc[df['uid']==uid, 'lat'].to_numpy()
    total_dist = 0
    for i in range(len(lon)-1):
        dist = distance.distance((lat[i],lon[i]),(lat[i+1],lon[i+1])).km
        total_dist = dist + total_dist
    
#        response = RoutingApi.truck_route(lat,
#                                      lon,
#                                      [herepy.RouteMode.truck, herepy.RouteMode.fastest])
    return_dict[uid] = total_dist

#def check_time_period(df):
#    periodlist = []
#    for i in uid:
#        timestamps = max(df_1.loc[df_1['uid']==i, 'row_number'].to_numpy())
#        timeperiod = route_distance/timestamps 
#        periodlist.append(timeperiod)
    
#Shit doesn't work
def cood_trans(lat,lon):
    # setup your projections
    crs_wgs = proj.Proj(init='epsg:4326') # assuming you're using WGS84 geographic
    crs_bng = proj.Proj(init='epsg:27700') # use a locally appropriate projected CRS
    
    # then cast your geographic coordinate pair to the projected system
    x, y = proj.transform(crs_wgs, crs_bng, lon, lat)
    return x,y

#Shit doesn't work
def norm_route_distance(df, uid):
    lon = df_1.loc[df['uid']==uid, 'lat'].to_numpy()
    lat = df_1.loc[df['uid']==uid, 'lon'].to_numpy()
    points = []
    for la,lo in zip(lat,lon):
        lat_x,lon_y = cood_trans(la,lo)
        points.append((lat_x,lon_y))
    d = np.diff(points, axis=0)
    segdists = np.hypot(d[:,0], d[:,1])
#    segdists = np.sqrt((d ** 2).sum(axis=1))
    return segdists
    
def get_uid(df):
    uid = set(df['uid'].tolist())
    return list(uid)

if __name__ == "__main__":
    df_1 = pd.read_csv('trackdata_c1.csv')
#    df_2 = pd.read_csv('trackdata_c2.csv')
    
    # Needs sorting according to row_number (currently coods scrambled)
    df_x = pd.read_csv('trackdata_cx.csv')
#    lon = df_1.loc[df_1['uid']==5, 'lat'].to_numpy()
#    lat = df_1.loc[df_1['uid']==5, 'lon'].to_numpy()
    
#    uid = route_distance(df, uid)
    manager = Manager()
    return_dict = manager.dict()
    jobs=[]
    
    '''Run route_distance in Colab'''
    trucks = sorted(set(df_1['uid']))
    
    #Parallel Implementation
    for truck in trucks[:2]:
        m = Process(target=route_distance, name=truck, args=(df_x, truck))
        m.daemon = True
        jobs.append(m)
        m.start()
    
    for proc in jobs:
        proc.join()
    
    distances = return_dict.copy()
    
#    distance_1 = {}
#    for truck in trucks[:30]:
#        distance_1[truck] = np.sum(norm_route_distance(df_x, truck))
    
    with open('geodesic_distances_c1.txt') as json_file:
        data = json.load(json_file)
#    plot_on_map(df_1)
