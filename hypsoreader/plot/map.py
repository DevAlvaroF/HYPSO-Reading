# File Manipulation
import glob
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from matplotlib import colors

from datetime import datetime
import matplotlib.pyplot as plt

# GIS
import cartopy.crs as ccrs
import cartopy.feature as cf
import geopandas as gpd
import rasterio as rio
from osgeo import gdal, osr
import pyproj



plotZoomFactor = 1.0


class SatelliteClass:
    def __init__(self, top_folder_name) -> None:
        self.DEBUG = False
        self.spatialDim = (956, 684)

        self.info = self.get_metainfo(top_folder_name)
        self.cube = self.get_raw_cube(top_folder_name)

        # Get Latitude and Longitude .dat files
        dat_files = glob.glob(top_folder_name + "/*.dat")
        longitude_dataPath = [f for f in dat_files if "longitudes" in f][0]
        latitude_dataPath = [f for f in dat_files if "latitudes" in f][0]

        # Get .geotiff file from geotiff folder
        geotiff_dir = [
            f.path
            for f in os.scandir(top_folder_name)
            if (f.is_dir() and ("geotiff" in os.path.basename(os.path.normpath(f))))
        ][0]

        self.geotiffFilePath = [
            join(geotiff_dir, f)
            for f in listdir(geotiff_dir)
            if (isfile(join(geotiff_dir, f)) and ("8bit" in f))
        ][0]

        # Load GeoTiff Metadata with gdal
        ds = gdal.Open(self.geotiffFilePath)
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        inproj = osr.SpatialReference()
        inproj.ImportFromWkt(proj)

        self.projection_metadata = {
            "data": data,
            "gt": gt,
            "proj": proj,
            "inproj": inproj,
        }

        # Load Latitude
        self.lat = np.fromfile(latitude_dataPath, dtype="float32")
        # Load Longitude
        self.lon = np.fromfile(longitude_dataPath, dtype="float32")

    def get_raw_cube(self, top_folder_name) -> np.ndarray:
        # find file ending in .bip
        path_to_bip = None
        for file in os.listdir(top_folder_name):
            if file.endswith(".bip"):
                path_to_bip = os.path.join(top_folder_name, file)
                break

        cube = np.fromfile(path_to_bip, dtype="uint16")
        if self.DEBUG:
            print(path_to_bip)
            print(cube.shape)
        cube = cube.reshape((-1, self.info["image_height"], self.info["image_width"]))

        # reverse the order of the third dimension
        cube = cube[:, :, ::-1]

        return cube

    def get_metainfo(self, top_folder_name: str) -> dict:
        """Get the metadata from the top folder of the data.

        Args:
            top_folder_name (str): The name of the top folder of the data.

        Returns:
            dict: The metadata.
        """
        info = {}
        info["top_folder_name"] = top_folder_name
        info["folder_name"] = top_folder_name.split("/")[-1]

        # find folder with substring "hsi0" or throw error
        for folder in os.listdir(top_folder_name):
            if "hsi0" in folder:
                raw_folder = folder
                break
        else:
            raise ValueError("No folder with metadata found.")

        # combine top_folder_name and raw_folder to get the path to the raw
        # data
        config_file_path = os.path.join(
            top_folder_name, raw_folder, "capture_config.ini"
        )

        def is_integer_num(n) -> bool:
            if isinstance(n, int):
                return True
            if isinstance(n, float):
                return n.is_integer()
            return False

        # read all lines in the config file
        with open(config_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # split the line at the equal sign
                line = line.split("=")
                # if the line has two elements, add the key and value to the
                # info dict
                if len(line) == 2:
                    key = line[0].strip()
                    value = line[1].strip()
                    try:
                        if is_integer_num(float(value)):
                            info[key] = int(value)
                        else:
                            info[key] = float(value)
                    except BaseException:
                        info[key] = value

        timetamp_file = os.path.join(top_folder_name, raw_folder, "timestamps.txt")

        with open(timetamp_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "." not in line:
                    continue
                s_part = line.split(".")[0]
                if s_part.strip().isnumeric():
                    info["unixtime"] = int(s_part) + 20
                    info["iso_time"] = datetime.utcfromtimestamp(
                        info["unixtime"]
                    ).isoformat()
                    break

        # find local_angle_csv file with substring "local-angles.csv" or throw error
        for file in os.listdir(top_folder_name):
            if "local-angles.csv" in file:
                local_angle_csv = file
                break
        else:
            raise ValueError("No local-angles.csv file found.")

        local_angle_df = pd.read_csv(top_folder_name + "/" + local_angle_csv)

        solar_za = local_angle_df["Solar Zenith Angle [degrees]"].tolist()
        solar_aa = local_angle_df["Solar Azimuth Angle [degrees]"].tolist()
        sat_za = local_angle_df["Satellite Zenith Angle [degrees]"].tolist()
        sat_aa = local_angle_df["Satellite Azimuth Angle [degrees]"].tolist()

        # Calculates the average solar/sat azimuth/zenith angle.
        average_solar_za = np.round(np.average(solar_za), 5)
        average_solar_aa = np.round(np.average(solar_aa), 5)
        average_sat_za = np.round((np.average(sat_za)), 5)
        average_sat_aa = np.round(np.average(sat_aa), 5)

        info["solar_zenith_angle"] = average_solar_za
        info["solar_azimuth_angle"] = average_solar_aa
        info["sat_zenith_angle"] = average_sat_za
        info["sat_azimuth_angle"] = average_sat_aa

        info["background_value"] = 8 * info["bin_factor"]

        info["x_start"] = info["aoi_x"]
        info["x_stop"] = info["aoi_x"] + info["column_count"]
        info["y_start"] = info["aoi_y"]
        info["y_stop"] = info["aoi_y"] + info["row_count"]
        info["exp"] = info["exposure"] / 1000  # in seconds

        info["image_height"] = info["row_count"]
        info["image_width"] = int(info["column_count"] / info["bin_factor"])
        info["im_size"] = info["image_height"] * info["image_width"]

        pos_file = ""
        foldername = info["top_folder_name"]
        for file in os.listdir(foldername):
            if file.endswith("geometric-meta-info.txt"):
                pos_file = os.path.join(foldername, file)
                break

        if pos_file == "":
            raise ValueError(f"Could not find position file in {foldername}")

        found_pos = False
        with open(pos_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "lat lon" in line:
                    info["lat"] = float(line.split("lat lon")[1].split(" ")[1])
                    info["lon"] = float(line.split("lat lon")[1].split(" ")[2])
                    found_pos = True
                    break

        if not found_pos:
            raise ValueError(f"Could not find position in {pos_file}")

        if self.DEBUG:
            print(info)

        return info


def show_image(
    satellite_obj, plotTitle="Chlorophyll Estimation OCX", overlapSatImg=False
):
    chl_array = np.random.randint(1, 100, size=satellite_obj.spatialDim, dtype="uint8")

    lat = satellite_obj.lat
    lon = satellite_obj.lon

    # Create meshgrid Sparse if Array is a vector
    if len(lon.shape) == 1:
        lon, lat = np.meshgrid(lon, lat, sparse=True)

    extent_lon_min = 0
    extent_lon_max = 0
    extent_lat_min = 0
    extent_lat_max = 0

    def image_extent_lon_lat(inproj_value):
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

    def extent_lon_lat(inproj_value):
        # Convert WKT projection information into a cartopy projection
        projcs = inproj_value.GetAuthorityCode("PROJCS")
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
    if hasattr(satellite_obj, "projection_metadata"):
        inproj = satellite_obj.projection_metadata["inproj"]
        extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max = extent_lon_lat(
            inproj
        )
        if overlapSatImg:
            transformed_img_extent, projection_img = image_extent_lon_lat(inproj)

    print(
        f"Extent Coordinates\nmin_lon {extent_lon_min}, min_lat {extent_lat_min},\nmax_lon {extent_lon_max}, max_lat {extent_lat_max}"
    )

    projection = ccrs.Mercator()
    # crs is PlateCarree -> we are explicitly telling axes, that we are creating bounds that are in degrees
    crs = ccrs.PlateCarree()
    # Now we will create axes object having specific projection

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
            extent_lon_min / plotZoomFactor,
            extent_lon_max * plotZoomFactor,
            extent_lat_min / plotZoomFactor,
            extent_lat_max * plotZoomFactor,
        ],
        crs=crs,
    )

    # Clipping to simulate Alpha channel
    # rgb_array[np.where(rgb_array == 0)] = np.nan

    if overlapSatImg:
        rgb_array = satellite_obj.projection_metadata["data"][:3, :, :]

        masked_data = np.ma.masked_where(rgb_array == 0, rgb_array)
        ax.imshow(
            np.rot90(masked_data.transpose((1, 2, 0)), k=2),
            origin="upper",
            extent=transformed_img_extent,
            transform=projection_img,
            zorder=1,
        )

    else:
        # * -----------------------          CHLOROPHYLL FROM HYPSO   ------------------------------

        min_chlr_val = np.nanmin(chl_array)
        lower_limit_chl = 0.01 if min_chlr_val < 0.01 else min_chlr_val

        max_chlr_val = np.nanmax(chl_array)
        upper_limit_chl = 100 if max_chlr_val > 100 else max_chlr_val

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
        # chl_range = [0.01, 100]  # OLD
        # chl_range = [0.3, 1] # OLD

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
