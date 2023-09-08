import glob
import os
from os import listdir
from os.path import isfile, join
from osgeo import gdal, osr
import numpy as np
import pandas as pd
from datetime import datetime
from importlib.resources import files
import netCDF4 as nc

from .radiometric import calibrate_cube, get_coefficients_from_file


class Satellite:
    def __init__(self, top_folder_name) -> None:
        self.DEBUG = False
        self.spatialDim = (956, 684)

        self.info = self.get_metainfo(top_folder_name)

        self.rawcube = self.get_raw_cube(top_folder_name)

        self.projection_metadata = self.get_projection_metadata(
            top_folder_name)

        # Radiometric Coefficients
        self.radiometric_coeff_file = self.get_radiometric_coefficients_path()
        self.radiometric_coefficients = get_coefficients_from_file(
            self.radiometric_coeff_file)

        # Wavelengths
        self.spectral_coeff_file = self.get_spectral_coefficients_path()
        self.spectral_coefficients = get_coefficients_from_file(
            self.spectral_coeff_file)
        self.wavelengths = self.spectral_coefficients

        self.l1b_cube = calibrate_cube(
            self.info, self.rawcube, self.radiometric_coefficients)

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
        cube = cube.reshape(
            (-1, self.info["image_height"], self.info["image_width"]))

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

        timetamp_file = os.path.join(
            top_folder_name, raw_folder, "timestamps.txt")

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

        # Find Coordinates of the Center of the Image
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
                    info["latc"] = float(line.split(
                        "lat lon")[1].split(" ")[1])
                    info["lonc"] = float(line.split(
                        "lat lon")[1].split(" ")[2])
                    found_pos = True
                    break

        if not found_pos:
            raise ValueError(f"Could not find position in {pos_file}")

        # Find 2D Coordinate
        dat_files = glob.glob(top_folder_name + "/*.dat")
        longitude_dataPath = [f for f in dat_files if "longitudes" in f][0]
        latitude_dataPath = [f for f in dat_files if "latitudes" in f][0]

        # Load Latitude
        info["lat"] = np.fromfile(latitude_dataPath, dtype="float32")
        info["lat"] = info["lat"].reshape(self.spatialDim)
        # Load Longitude
        info["lon"] = np.fromfile(longitude_dataPath, dtype="float32")
        info["lon"] = info["lon"].reshape(self.spatialDim)

        # info["lon"], info["lat"] = np.meshgrid(info["lon"], info["lat"], sparse=True)
        if self.DEBUG:
            print(info)

        return info

    def get_projection_metadata(self, top_folder_name: str) -> dict:
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
        # Not hyperspectral, fewer bands
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        inproj = osr.SpatialReference()
        inproj.ImportFromWkt(proj)

        return {
            "data": data,
            "gt": gt,
            "proj": proj,
            "inproj": inproj,
        }

    def get_radiometric_coefficients_path(self) -> str:
        coeff_file = files(
            'hypsoreader.radiometric').joinpath('data/rad_coeffs_FM_binx9_2022_08_06_Finnmark_recal_a.csv')
        return coeff_file

    def get_spectral_coefficients_path(self) -> str:
        wl_file = files(
            'hypsoreader.radiometric').joinpath('data/spectral_bands_HYPSO-1_120bands.csv')
        return wl_file

    def georeference_image(self, top_folder_name: str):
        # gcpPath = r"C:\Users\alvar\OneDrive\Desktop\karachi_2023-02-06_0531Z-bin3.png.points"225
        # lat_coeff, lon_coeff = reference_correction.geotiff_correction(gcpPath, self.projection_metadata)
        point_file = glob.glob(top_folder_name + '/*.points')

        if len(point_file) == 0:
            print("Points File Was Not Found")
        else:
            self.info["lat"], self.info["lon"] = coordinate_correction(
                point_file[0], self.projection_metadata,
                self.info["lat"], self.info["lon"])
