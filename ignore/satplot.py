# -----------------------------------------------------------------------------
#                               IMPORT LIBRARIES
# -----------------------------------------------------------------------------
import tkinter
import matplotlib

matplotlib.use('TkAgg')

# File Manipulation
import glob
import json
import os
from os import listdir
from os.path import isfile, join
import pathlib
import pandas as pd

import webbrowser
from datetime import datetime
from natsort import natsorted

# GIS
import cartopy.crs as ccrs
import cartopy.feature as cf
import geopandas as gpd
import rasterio as rio
from osgeo import gdal, osr
import pyproj
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box

# Read GSI Data
import xarray as xr  # To map to .NC metadata file
import netCDF4  # To get data of .NC file with metadata variables
import spectral.io.envi as envi
# from sentinelsat.sentinel import SentinelAPI

# Plot and Image Gen
import folium
from matplotlib import colors
import matplotlib as mpl
from matplotlib import ticker as mticker
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['figure.facecolor'] = 'white'

from matplotlib import pyplot as plt
from mpl_toolkits import axes_grid1
from matplotlib import ticker
from matplotlib import colors

# Numerical
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull, Delaunay


class SatPlot:
    plotZoomFactor = 1.0  # 1.05 for normal and 1.0 for debug
    color_bar_pos = 'bottom'  # 'bottom' or 'right'
    # Plotting Reference
    dst_crs = CRS.from_user_input(4326)  # 'EPSG:4326'

    def __init__(self, satellite):
        self.satellite = satellite

    def myLogFormat(self, y, pos):
        # Find the number of decimal places required
        decimalplaces = int(np.maximum(-np.log10(y), 0))  # =0 for numbers >=1
        if decimalplaces == 0:
            # Insert that number into a format string
            formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
            # Return the formatted tick label
            return formatstring.format(y)
        else:
            formatstring = '{:2.1e}'.format(y)
            return formatstring

    def self_chl(self, plotTitle='Chlorophyll Estimation OCX',
                 chl_key='ocx',
                 satName=None,
                 interpMethodUsed=None,
                 special_extent=None,
                 overlapSatImg=False):

        # Get Chl and lat data
        chl_array = []
        if satName == 'hypso':
            chl_external_key = 'hypso_' + chl_key
        elif satName == 'hypso_acolite' or satName == 'hypso_py6s':
            chl_external_key = satName + '_' + chl_key
        elif satName is not None:
            if interpMethodUsed is not None:
                chl_external_key = interpMethodUsed + '_' + satName + '_' + chl_key
            else:
                chl_external_key = chl_key
        else:
            return

        print("Plotting " + chl_external_key)
        # If coming from a complex method like OCX, the chl dict will be another
        # if coming from just interpolation in lon/lat, will be an array
        if isinstance(self.satellite.chl[chl_external_key], dict):
            chl_array = self.satellite.chl[chl_external_key]['chl_algorithm']
        else:
            chl_array = self.satellite.chl[chl_external_key]

        lat = self.satellite.lat
        lon = self.satellite.lon

        # Create meshgrid Sparse if Array is a vector
        if len(lon.shape) == 1:
            lon, lat = np.mesgrid(
                lon, lat, sparse=True)

        extent_lon_min = 0
        extent_lon_max = 0
        extent_lat_min = 0
        extent_lat_max = 0

        def image_extent_lon_lat(inproj_value):
            # Convert WKT projection information into a cartopy projection
            projcs = inproj_value.GetAuthorityCode('PROJCS')
            projection_img = ccrs.epsg(projcs)

            # Transform Current lon and lat limits to another
            new_max_lon = np.max(lon)
            new_max_lat = np.max(lat)
            new_min_lon = np.min(lon)
            new_min_lat = np.min(lat)

            # Convert lat and lon to the image CRS so we create Projection with Dataset CRS
            dataset_proj = pyproj.Proj(projection_img)  # your data crs

            # Transform Coordinates to Image CRS
            transformed_min_lon, transformed_min_lat = dataset_proj(
                new_min_lon, new_min_lat, inverse=False)
            transformed_max_lon, transformed_max_lat = dataset_proj(
                new_max_lon, new_max_lat, inverse=False)

            transformed_img_extent = (transformed_min_lon, transformed_max_lon,
                                      transformed_min_lat, transformed_max_lat)

            return transformed_img_extent, projection_img

        def extent_lon_lat(inproj_value):
            # Convert WKT projection information into a cartopy projection
            projcs = inproj_value.GetAuthorityCode('PROJCS')
            projection_img = ccrs.epsg(projcs)

            # Transform Current lon and lat limits to another
            new_max_lon = np.max(lon)
            new_max_lat = np.max(lat)
            new_min_lon = np.min(lon)
            new_min_lat = np.min(lat)

            # # Convert lat and lon to the image CRS so we create Projection with Dataset CRS
            # dataset_proj = pyproj.Proj(projection_img)  # your data crs
            #
            # # Transform Coordinates to Image CRS
            # transformed_min_lon, transformed_min_lat = dataset_proj(
            #     new_min_lon, new_min_lat, inverse=False)
            # transformed_max_lon, transformed_max_lat = dataset_proj(
            #     new_max_lon, new_max_lat, inverse=False)
            #
            # transformed_img_extent = (transformed_min_lon, transformed_max_lon,
            #                           transformed_min_lat, transformed_max_lat)

            extent_lon_min = new_min_lon
            extent_lon_max = new_max_lon

            extent_lat_min = new_min_lat
            extent_lat_max = new_max_lat

            return [extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max]

        # If projection is not None, is Hypso and needs reprojection
        projection_img = None
        transformed_img_extent = None
        if hasattr(self.satellite, "projection_metadata"):
            inproj = self.satellite.projection_metadata['inproj']
            extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max = extent_lon_lat(inproj)
            if overlapSatImg:
                transformed_img_extent, projection_img = image_extent_lon_lat(inproj)
        elif hasattr(self.satellite, "projection_metadata_from_Hypso"):
            if self.satellite.projection_metadata_from_Hypso is None:
                if special_extent is None:
                    extent_lon_min = lon.min()
                    extent_lon_max = lon.max()

                    extent_lat_min = lat.min()
                    extent_lat_max = lat.max()
                else:
                    extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max = special_extent

        print(
            f"Extent Coordinates\nmin_lon {extent_lon_min}, min_lat {extent_lat_min},\nmax_lon {extent_lon_max}, max_lat {extent_lat_max}")

        projection = ccrs.Mercator()
        # crs is PlateCarree -> we are explicitly telling axes, that we are creating bounds that are in degrees
        crs = ccrs.PlateCarree()
        # Now we will create axes object having specific projection

        fig = plt.figure(figsize=(10, 10), dpi=450)
        fig.patch.set_alpha(1)
        ax = fig.add_subplot(projection=projection, frameon=True)

        # Draw gridlines in degrees over Mercator map
        gl = ax.gridlines(draw_labels=True,
                          linewidth=.6, color='gray', alpha=0.5, linestyle='-.')
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}

        # To plot borders and coastlines, we can use cartopy feature
        ax.add_feature(cf.COASTLINE.with_scale("10m"), lw=0.5)
        ax.add_feature(cf.BORDERS.with_scale("10m"), lw=0.3)
        # ax.add_feature(cf.LAND, zorder=100, edgecolor='k')  # Covers Data in land

        ax.set_extent([extent_lon_min / self.plotZoomFactor, extent_lon_max * self.plotZoomFactor,
                       extent_lat_min / self.plotZoomFactor, extent_lat_max * self.plotZoomFactor], crs=crs)

        # Clipping to simulate Alpha channel
        # rgb_array[np.where(rgb_array == 0)] = np.nan

        if overlapSatImg:
            rgb_array = self.satellite.projection_metadata['data'][:3, :, :]
            masked_data = np.ma.masked_where(rgb_array == 0, rgb_array)
            ax.imshow(np.rot90(masked_data.transpose((1, 2, 0)), k=2), origin='upper', extent=transformed_img_extent,
                      transform=projection_img, zorder=1)

        # * -----------------------          CHLOROPHYLL FROM HYPSO   ------------------------------
        # TODO: Change this chl_display range to match modis and sentinel

        min_chlr_val = np.nanmin(chl_array)
        lower_limit_chl = 0.01 if min_chlr_val < 0.01 else min_chlr_val

        max_chlr_val = np.nanmax(chl_array)
        upper_limit_chl = 100 if max_chlr_val > 100 else max_chlr_val

        # Set Range to next full log
        for i in range(-2, 3):
            full_log = 10 ** i
            if full_log < lower_limit_chl:
                lower_limit_chl = full_log
        for i in range(2, -3, -1):
            full_log = 10 ** i
            if full_log > upper_limit_chl:
                upper_limit_chl = full_log

        chl_range = [lower_limit_chl, upper_limit_chl]  # old: [0.01, 100] [0.3, 1]
        # chl_range = [0.01, 100]  # OLD
        # chl_range = [0.3, 1] # OLD

        print('Chl Range: ', chl_range)

        # Log Normalize Color Bar colors
        norm = colors.LogNorm(chl_range[0], chl_range[1])
        im = ax.pcolormesh(lon, lat, chl_array,
                           cmap=plt.cm.jet, transform=ccrs.PlateCarree(), norm=norm, zorder=0)

        # Colourmap with axes to match figure size
        cbar = plt.colorbar(im, location=self.color_bar_pos, shrink=1, ax=ax, pad=0.05)

        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.myLogFormat))
        cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.myLogFormat))

        # cbar.ax.yaxis.set_minor_formatter(mticker.ScalarFormatter(useMathText=False, useOffset=False))
        # cbar.ax.xaxis.set_minor_formatter(mticker.ScalarFormatter(useMathText=False, useOffset=False))

        cbar.set_label(f" Chlorophyll Concentration [mg m^-3]")

        plt.title(plotTitle)
        plt.show()
