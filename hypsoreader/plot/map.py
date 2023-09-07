# File Manipulation
import numpy as np
import pyproj
from matplotlib import colors
import warnings

import matplotlib.pyplot as plt

# GIS
import cartopy.crs as ccrs
import cartopy.feature as cf
import geopandas as gpd
import rasterio as rio

PLOTZOOM = 1.0


def axis_extent(lat, lon):
    # Transform Current lon and lat limits to another
    extent_lon_min = np.nanmin(lon)
    extent_lon_max = np.nanmax(lon)

    extent_lat_min = np.nanmin(lat)
    extent_lat_max = np.nanmax(lat)

    return [extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max]


def image_extent(inproj_value, lat, lon):
    # Convert WKT projection information into a cartopy projection
    projcs = inproj_value.GetAuthorityCode("PROJCS")
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
        new_min_lon, new_min_lat, inverse=False
    )
    transformed_max_lon, transformed_max_lat = dataset_proj(
        new_max_lon, new_max_lat, inverse=False
    )

    transformed_img_extent = (
        transformed_min_lon,
        transformed_max_lon,
        transformed_min_lat,
        transformed_max_lat,
    )

    return transformed_img_extent, projection_img


def show_rgb(satellite_obj, plotTitle="RGB Image"):

    lat = satellite_obj.info["lat"]
    lon = satellite_obj.info["lon"]

    # Create Axis Transformation
    inproj = satellite_obj.projection_metadata["inproj"]
    extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max = axis_extent(
        lat, lon)
    transformed_img_extent, projection_img = image_extent(inproj, lat, lon)

    # crs is PlateCarree -> we are explicitly telling axes, that we are creating bounds that are in degrees
    projection = ccrs.Mercator()
    crs = ccrs.PlateCarree()

    # Now we will create axes object having specific projection
    fig = plt.figure(figsize=(10, 10), dpi=450)
    fig.patch.set_alpha(1)
    ax = fig.add_subplot(projection=projection, frameon=True)

    ax.set_extent(
        [
            extent_lon_min / PLOTZOOM,
            extent_lon_max * PLOTZOOM,
            extent_lat_min / PLOTZOOM,
            extent_lat_max * PLOTZOOM,
        ],
        crs=crs,
    )

    # Draw gridlines in degrees over Mercator map
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.6, color="gray", alpha=0.5, linestyle="--"
    )
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    # TODO: Warnings are disabled as a rounding error with shapely causes an "no intersection warning". New version of GDAL might solve it
    # To plot borders and coastlines, we can use cartopy feature
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')

    ax.add_feature(cf.COASTLINE.with_scale("10m"), lw=0.5)
    ax.add_feature(cf.BORDERS.with_scale("10m"), lw=0.3)
    # ax.add_feature(cf.LAND, zorder=100, edgecolor='k')  # Covers Data in land

    warnings.resetwarnings()

    # Plot Image
    rgb_array = satellite_obj.projection_metadata["data"][:3, :, :]

    masked_data = np.ma.masked_where(rgb_array == 0, rgb_array)
    ax.imshow(
        np.rot90(masked_data.transpose((1, 2, 0)), k=2),
        origin="upper",
        extent=transformed_img_extent,
        transform=projection_img,
        zorder=1,
    )

    plt.title(plotTitle)
    plt.show()


def plot_chlorophyll(satellite_obj, chl_array, plotTitle="Chlorophyll Concentration"):
    MAXCHL = 100
    MINCHL = 0.01

    lat = satellite_obj.info["lat"]
    lon = satellite_obj.info["lon"]

    # Create Axis Transformation
    inproj = satellite_obj.projection_metadata["inproj"]
    extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max = axis_extent(
        lat, lon)
    transformed_img_extent, projection_img = image_extent(inproj, lat, lon)

    # crs is PlateCarree -> we are explicitly telling axes, that we are creating bounds that are in degrees
    projection = ccrs.Mercator()
    crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10, 10), dpi=450)
    fig.patch.set_alpha(1)
    ax = fig.add_subplot(projection=projection, frameon=True)

    # Draw gridlines in degrees over Mercator map
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.6, color="gray", alpha=0.5, linestyle="-."
    )
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    # To plot borders and coastlines, we can use cartopy feature
    ax.add_feature(cf.COASTLINE.with_scale("10m"), lw=0.5)
    ax.add_feature(cf.BORDERS.with_scale("10m"), lw=0.3)
    # ax.add_feature(cf.LAND, zorder=100, edgecolor='k')  # Covers Data in land

    ax.set_extent(
        [
            extent_lon_min / PLOTZOOM,
            extent_lon_max * PLOTZOOM,
            extent_lat_min / PLOTZOOM,
            extent_lat_max * PLOTZOOM,
        ],
        crs=crs,
    )

    min_chlr_val = np.nanmin(chl_array)
    lower_limit_chl = MINCHL if min_chlr_val < MINCHL else min_chlr_val

    max_chlr_val = np.nanmax(chl_array)
    upper_limit_chl = MAXCHL if max_chlr_val > MAXCHL else max_chlr_val

    # Set Range to next full log
    for i in range(-2, 3):
        full_log = 10**i
        if full_log < lower_limit_chl:
            lower_limit_chl = full_log
    for i in range(2, -3, -1):
        full_log = 10**i
        if full_log > upper_limit_chl:
            upper_limit_chl = full_log

    # old: [0.01, 100] [0.3, 1]
    chl_range = [lower_limit_chl, upper_limit_chl]

    print("Chl Range: ", chl_range)

    # Log Normalize Color Bar colors
    norm = colors.LogNorm(chl_range[0], chl_range[1])
    im = ax.pcolormesh(
        lon,
        lat,
        chl_array,
        cmap=plt.cm.jet,
        transform=ccrs.PlateCarree(),
        norm=norm,
        zorder=0,
    )

    # Colourmap with axes to match figure size
    cbar = plt.colorbar(im, location="bottom", shrink=1, ax=ax, pad=0.05)

    # cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.myLogFormat))
    # cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.myLogFormat))

    # cbar.ax.yaxis.set_minor_formatter(mticker.ScalarFormatter(useMathText=False, useOffset=False))
    # cbar.ax.xaxis.set_minor_formatter(mticker.ScalarFormatter(useMathText=False, useOffset=False))

    cbar.set_label(f" Chlorophyll Concentration [mg m^-3]")
    plt.title(plotTitle)
    plt.show()
