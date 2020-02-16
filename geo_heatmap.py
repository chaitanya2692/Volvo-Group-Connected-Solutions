#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:59:00 2020

@author: piedpiper
"""

#!/usr/bin/env python3

from argparse import ArgumentParser, RawTextHelpFormatter
import collections
#import fnmatch
import folium
from folium.plugins import HeatMap
#import ijson
#import json
#import os
#from progressbar import ProgressBar, Bar, ETA, Percentage
from utils import *
#import webbrowser
#from xml.etree import ElementTree
#from xml.dom import minidom
#import zipfile
import numpy as np
import pandas as pd

class Generator:
    def __init__(self):
        self.coordinates = collections.defaultdict(int)
        self.max_coordinates = (0, 0)
        self.max_magnitude = 0

    def load_dataframe(self, df):
        """Loads the Google location data from the given json file.

        Arguments:
            json_file {file} -- An open file-like object with JSON-encoded
                Google location data.
            date_range {tuple} -- A tuple containing the min-date and max-date.
                e.g.: (None, None), (None, '2019-01-01'), ('2017-02-11'), ('2019-01-01')
        """
        
        lats = df["lat"].to_numpy()
        lons = df["lon"].to_numpy()
        coods = np.stack((lats, lons), axis=-1)
#        w = [Bar(), Percentage(), " ", ETA()]
#        with ProgressBar(max_value=coods.shape[0], widgets=w) as pb:
        for i in coods:
            self.updateCoord(tuple(i))
#                pb.update(i)
            
    def updateCoord(self, coords):
        self.coordinates[coords] += 1
        if self.coordinates[coords] > self.max_magnitude:
            self.max_coordinates = coords
            self.max_magnitude = self.coordinates[coords]

    def generateMap(self, settings):
        """Generates the heatmap.
        
        Arguments:
            settings {dict} -- The settings for the heatmap.
        
        Returns:
            Map -- The Heatmap.
        """
        tiles = settings["tiles"]
        zoom_start = settings["zoom_start"]
        radius = settings["radius"]
        blur = settings["blur"]
        min_opacity = settings["min_opacity"]
        max_zoom = settings["max_zoom"]
        
        map_data = [(coords[0], coords[1], magnitude)
                    for coords, magnitude in self.coordinates.items()]

        # Generate map
        m = folium.Map(location=self.max_coordinates,
                       zoom_start=zoom_start,
                       tiles=tiles)

        # Generate heat map
        heatmap = HeatMap(map_data,
                          max_val=self.max_magnitude,
                          min_opacity=min_opacity,
                          radius=radius,
                          blur=blur,
                          max_zoom=max_zoom)

        m.add_child(heatmap)
        return m

    def run(self, data_file, output_file, settings):
        """Load the data, generate the heatmap and save it.

        Arguments:
            data_files {list} -- List of names of the data files with the Google
                location data or the Google takeout ZIP archive.
            output_file {string} -- The name of the output file.
            date_range {tuple} -- A tuple containing the min-date and max-date.
                e.g.: (None, None), (None, '2019-01-01'), ('2017-02-11'), ('2019-01-01')
            stream_data {bool} -- Stream option.
            settings {dict} -- The settings for the heatmap.
        """
        
        print("Loading data")
        self.load_dataframe(data_file)
        
        print("GenerateMapGenerating heatmap")
        m = self.generateMap(settings)
        
        print("Saving map to {}\n".format(output_file))
#        m.save(output_file)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    
    parser.add_argument("-o", "--output", dest="output", type=str, required=False,
                        help="Path of heatmap HTML output file.", default="heatmap.html")
    parser.add_argument("--map", "-m", dest="map", metavar="MAP", type=str, required=False, default="OpenStreetMap",
                        help="The name of the map tiles you want to use.\n" \
                        "(e.g. 'OpenStreetMap', 'StamenTerrain', 'StamenToner', 'StamenWatercolor')")
    parser.add_argument("-z", "--zoom-start", dest="zoom_start", type=int, required=False,
                        help="The initial zoom level for the map. (default: %(default)s)", default=6)
    parser.add_argument("-r", "--radius", type=int, required=False,
                        help="The radius of each location point. (default: %(default)s)", default=7)
    parser.add_argument("-b", "--blur", type=int, required=False,
                        help="The amount of blur. (default: %(default)s)", default=4)
    parser.add_argument("-mo", "--min-opacity", dest="min_opacity", type=float, required=False,
                        help="The minimum opacity of the heatmap. (default: %(default)s)", default=0.2)
    parser.add_argument("-mz", "--max-zoom", dest="max_zoom", type=int, required=False,
                        help="The maximum zoom of the heatmap. (default: %(default)s)", default=4)
    

    args = parser.parse_args()
    df = pd.read_csv('trackdata_cx.csv')
    output_file = args.output
    settings = {
        "tiles": args.map,
        "zoom_start": args.zoom_start,
        "radius": args.radius,
        "blur": args.blur,
        "min_opacity": args.min_opacity,
        "max_zoom": args.max_zoom
    }

    generator = Generator()
    generator.run(df, output_file, settings)