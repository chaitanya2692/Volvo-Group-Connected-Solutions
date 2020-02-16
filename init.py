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
import geo_heatmap as geo
from argparse import ArgumentParser, RawTextHelpFormatter
import urllib, time
from tqdm import tqdm

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
def route_distance(df, uid, return_dict):
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

def make_heatmap(df):
    output_file = "heatmap.html"
    settings = {
        "tiles": "OpenStreetMap",
        "zoom_start": 6,
        "radius": 7,
        "blur": 4,
        "min_opacity": 0.2,
        "max_zoom": 4
    }
    
    generator = geo.Generator()
    generator.run(df, output_file, settings)
    return

def get_elevations(df, filename):
    lon = df['lon'].to_numpy()
    lat = df['lat'].to_numpy()
    cood_list = np.stack((lat,lon), axis = -1)
#    cood_list_f = cood_list.flatten()
    elevation = []
#    for i in range(cood_list.shape[0]):
    for i in range(0,100,10):
        request = 'https://api.airmap.com/elevation/v1/ele/?points=' +str(list(cood_list[i:i+10]))[1:-1]
        response = urllib.request.urlopen(request.replace(" ", "")).read()
        response_str = response.decode('utf8').replace("'", '"')
        data = json.loads(response_str)
        elevation.append(data['data'])
    
#    with open(filename, 'w') as json_file:
#        json.dump(elevation, json_file)
    return elevation

def make_parallel(df,process):
    manager = Manager()
    return_dict = manager.dict()
    jobs=[]
    
    '''Run route_distance in Colab'''
    trucks = sorted(set(df['uid']))
    
    #Parallel Implementation
    for truck in trucks:
        m = Process(target=process, name=truck, args=(df,truck, return_dict))
        m.daemon = True
        jobs.append(m)
        m.start()
    
    for proc in jobs:
        proc.join()
    
    result = return_dict.copy()  
    return result

def get_avg_lats(df_1,df_2):
    all_lats = df_1.append(df_2).filter(['uid', 'lat'])
    
    lat_class = dict()
    trucks = get_uid(all_lats) 
    for truck in tqdm(trucks):
        lat_class[truck] = all_lats.loc[all_lats['uid']==truck, 'lat'].mean()
    lat_df = pd.DataFrame(lat_class.items(),  columns = ['uid','lat'])
   
    lat_df.to_csv('all_lats.csv')
    
    return lat_df

def get_delta_lats(df):
    trucks = get_uid(df)
    lat_delta = dict()
    for truck in tqdm(trucks):
        lat = df.loc[df['uid']==truck, 'lat']
        min_lat = min(lat)
        max_lat = max(lat)
        if min_lat > 0 and max_lat > 0:
            lat_delta[truck] = min(lat) - max(lat)
        if min_lat < 0 and max_lat < 0:
            lat_delta[truck] = max(lat) - min(lat)
        if min_lat < 0 and max_lat > 0:
            lat_delta[truck] = max(lat) - min(lat)
    lat_delta_df = pd.DataFrame(lat_delta.items(),  columns = ['uid','lat_delta'])
    lat_delta_df = lat_delta_df.round({"lat_delta":5})
    return lat_delta_df
    
def start_end_route(df):
    route_coods = pd.DataFrame()
    start_lats = df.groupby('uid')['lat'].first().to_frame().reset_index().rename(columns={'lat':'start_lat'})
    end_lats = df.groupby('uid')['lat'].last().to_frame().reset_index().rename(columns={'lat':'end_lat'})
    start_lons = df.groupby('uid')['lon'].first().to_frame().reset_index().rename(columns={'lon':'start_lon'})
    end_lons = df.groupby('uid')['lon'].last().to_frame().reset_index().rename(columns={'lon':'end_lon'})
    
    start_lats = start_lats.merge(end_lats, on='uid')
    start_lats = start_lats.merge(start_lons, on='uid')
    start_lats = start_lats.merge(end_lons, on='uid')
    route_coods = start_lats
    return route_coods

def start_end_route_p(df,truck,return_dict):
    current_coods = {}
#    current_coods['uid'] = truck 
    current_coods['start_lat'] = df.loc[df['uid']==truck, 'lat'].iloc[0]
    current_coods['end_lat'] = df.loc[df['uid']==truck, 'lat'].iloc[-1]
    current_coods['start_lon'] = df.loc[df['uid']==truck, 'lon'].iloc[0]
    current_coods['end_lon'] = df.loc[df['uid']==truck, 'lon'].iloc[-1]
    return_dict[truck]=current_coods

def build_features(df_1_agg, df_2_agg):
#    lat_df = get_lats(df_1, df_2)
    lat_df = pd.read_csv('all_lats.csv', index_col=0)
    df_train = df_1_agg.append(df_2_agg)
    df_train['uid'] = df_train['uid'].astype(int) 
    df_train = df_train.merge(lat_df, on='uid').rename(columns={'lat':'lat_class'})
    df_train['lat_class'] = (df_train['lat_class']/10).round(0).astype(int)
    
    return df_train
    
if __name__ == "__main__":
    
#    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
#    
#    parser.add_argument("-o", "--output", dest="output", type=str, required=False,
#                        help="Path of heatmap HTML output file.", default="heatmap.html")
#    parser.add_argument("--map", "-m", dest="map", metavar="MAP", type=str, required=False, default="OpenStreetMap",
#                        help="The name of the map tiles you want to use.\n" \
#                        "(e.g. 'OpenStreetMap', 'StamenTerrain', 'StamenToner', 'StamenWatercolor')")
#    parser.add_argument("-z", "--zoom-start", dest="zoom_start", type=int, required=False,
#                        help="The initial zoom level for the map. (default: %(default)s)", default=6)
#    parser.add_argument("-r", "--radius", type=int, required=False,
#                        help="The radius of each location point. (default: %(default)s)", default=7)
#    parser.add_argument("-b", "--blur", type=int, required=False,
#                        help="The amount of blur. (default: %(default)s)", default=4)
#    parser.add_argument("-mo", "--min-opacity", dest="min_opacity", type=float, required=False,
#                        help="The minimum opacity of the heatmap. (default: %(default)s)", default=0.2)
#    parser.add_argument("-mz", "--max-zoom", dest="max_zoom", type=int, required=False,
#                        help="The maximum zoom of the heatmap. (default: %(default)s)", default=4)
    
    df_1 = pd.read_csv('trackdata_c1.csv')
    df_2 = pd.read_csv('trackdata_c2.csv')
    
    # Needs sorting according to row_number (currently coods scrambled)
#    df_x = pd.read_csv('trackdata_cx.csv')
 
#    distances = get_distance(df_1)
#    with open('geodesic_distances_c1.txt', 'w') as json_file:
#      json.dump(distances, json_file)
    
#    make_heatmap(df_1)
        
    with open('geodesic_distances_c1.txt') as json_file:
        data_1 = json.load(json_file)
    
    with open('geodesic_distances_c2.txt') as json_file:
        data_2 = json.load(json_file)
        
    df_1_agg = pd.DataFrame(data_1.items(),  columns = ['uid','distance'])
    df_2_agg = pd.DataFrame(data_2.items(),  columns = ['uid','distance'])
       
#    df_train = build_features(df_1_agg, df_2_agg)

#    lat_delta_df_1 = get_delta_lats(df_1)
#    lat_delta_df_2 = get_delta_lats(df_2)
#    all_lat_delta = lat_delta_df_1.append(lat_delta_df_2)
    all_lat_delta = pd.read_csv('all_lat_delta.csv', index_col=0)
    
    route_coods_1 = start_end_route(df_1)
    route_coods_2 = start_end_route(df_2)
    all_routes = route_coods_1.append(route_coods_2, ignore_index=True)
    
    df_train = df_1_agg.append(df_2_agg, ignore_index=True)
    df_train['uid'] = df_train['uid'].astype(int)
    df_train = df_train.merge(all_lat_delta, on='uid')
    df_train = df_train.merge(all_routes, on='uid')
    
    df_train.to_csv('df_train.csv')