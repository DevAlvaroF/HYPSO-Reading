import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import json
from datetime import datetime


def write_h1data_as_geojson(sat_object, path_to_save: str) -> None:
    # check if file ends with .geojson
    if not path_to_save.endswith('.geojson'):
        path_to_save = path_to_save + '.geojson'

    with open(path_to_save, 'w') as f:
        f.write(get_geojson_str(sat_object))


def write_h1data_as_NetCDF4(sat_object, path_to_save: str) -> None:
    DEBUG = True
    """
    Write the HYPSO-1 data as a NetCDF4 file.

    Args:
        path_to_save (str): The path to save the NetCDF4 file.
        path_to_h1data (str): The path to the HYPSO-1 data.

    Raises:
        ValueError: If the NetCDF4 file already exists.
    """

    # check if file ends with .nc
    if not path_to_save.endswith('.nc'):
        path_to_save = path_to_save + '.nc'

    if os.path.exists(path_to_save):
        os.remove(path_to_save)

    h1 = sat_object
    path_to_h1data = sat_object.info["top_folder_name"]

    temp = h1.info["folder_name"].split("CaptureDL_")[-1]
    name = temp.split("_")[0]
    frames = h1.info["frame_count"]
    lines = h1.info["image_height"]
    bands = h1.info["image_width"]

    if DEBUG:
        print(h1.info)

    is_nav_data_available = False
    for file in os.listdir(path_to_h1data):
        if "sat-azimuth.dat" in file:
            sata = np.fromfile(os.path.join(
                path_to_h1data, file), dtype=np.float32)
            is_nav_data_available = True
        elif "sat-zenith.dat" in file:
            satz = np.fromfile(os.path.join(
                path_to_h1data, file), dtype=np.float32)
        elif "sun-azimuth.dat" in file:
            suna = np.fromfile(os.path.join(
                path_to_h1data, file), dtype=np.float32)
        elif "sun-zenith.dat" in file:
            sunz = np.fromfile(os.path.join(
                path_to_h1data, file), dtype=np.float32)
        elif "latitudes.dat" in file:
            lat = np.fromfile(os.path.join(
                path_to_h1data, file), dtype=np.float32)
        elif "longitudes.dat" in file:
            lon = np.fromfile(os.path.join(
                path_to_h1data, file), dtype=np.float32)

    if not is_nav_data_available:
        raise ValueError("Navigation data is not available.")

    # get unixtimes
    posetime_file = None
    for file in os.listdir(path_to_h1data):
        if "frametime-pose" in file:
            posetime_file = os.path.join(path_to_h1data, file)
            break
    if posetime_file is None:
        raise ValueError("frametime-pose file is not available.")

    df = pd.read_csv(posetime_file)
    # df.columns = df.iloc[0]
    # df = df[1:]

    if DEBUG:
        print(df)

    # Create a new NetCDF file
    with nc.Dataset(path_to_save, 'w', format='NETCDF4') as f:

        f.instrument = "HYPSO-1 Hyperspectral Imager"
        f.institution = "Norwegian University of Science and Technology"
        f.location_description = name
        f.license = "TBD"
        f.naming_authority = "NTNU SmallSat Lab"
        f.date_processed = datetime.utcnow().isoformat() + "Z"
        f.date_aquired = h1.info["iso_time"] + "Z"
        f.publisher_name = "NTNU SmallSat Lab"
        f.publisher_url = "https://hypso.space"
        # f.publisher_contact = "smallsat@ntnu.no"
        f.processing_level = "L1B"
        f.radiometric_file = os.path.basename(
            h1.radiometric_coeff_file)
        f.spectral_file = os.path.basename(
            h1.spectral_coeff_file)
        # Create dimensions
        f.createDimension('frames', frames)
        f.createDimension('lines', lines)
        f.createDimension('bands', bands)

        # create groups
        navigation_group = f.createGroup('navigation')

        navigation_group.iso8601time = h1.info["iso_time"] + "Z"

        # Create variables

        time = f.createVariable('navigation/unixtime', 'u8', ('frames',))
        time[:] = df["timestamp"].values

        sensor_z = f.createVariable(
            'navigation/sensor_zenith', 'f4', ('frames', 'lines'))
        sensor_z[:] = satz.reshape(frames, lines)
        sensor_z.long_name = "Sensor Zenith Angle"
        sensor_z.units = "degrees"
        sensor_z.valid_range = [-180, 180]

        sensor_a = f.createVariable(
            'navigation/sensor_azimuth', 'f4', ('frames', 'lines'))
        sensor_a[:] = sata.reshape(frames, lines)
        sensor_a.long_name = "Sensor Azimuth Angle"
        sensor_a.units = "degrees"
        sensor_a.valid_range = [-180, 180]

        solar_z = f.createVariable(
            'navigation/solar_zenith', 'f4', ('frames', 'lines'))
        solar_z[:] = sunz.reshape(frames, lines)
        solar_z.long_name = "Solar Zenith Angle"
        solar_z.units = "degrees"
        solar_z.valid_range = [-180, 180]

        solar_a = f.createVariable(
            'navigation/solar_azimuth', 'f4', ('frames', 'lines'))
        solar_a[:] = suna.reshape(frames, lines)
        solar_a.long_name = "Solar Azimuth Angle"
        solar_a.units = "degrees"
        solar_a.valid_range = [-180, 180]

        latitude = f.createVariable(
            'navigation/latitude', 'f4', ('frames', 'lines'))
        latitude[:] = lat.reshape(frames, lines)
        latitude.long_name = "Latitude"
        latitude.units = "degrees"
        latitude.valid_range = [-180, 180]

        longitude = f.createVariable(
            'navigation/longitude', 'f4', ('frames', 'lines'))
        longitude[:] = lon.reshape(frames, lines)
        longitude.long_name = "Longitude"
        longitude.units = "degrees"
        longitude.valid_range = [-180, 180]

        f.createGroup('products')
        Lt = f.createVariable('products/Lt', 'f4',
                              ('frames', 'lines', 'bands'))
        Lt.units = "W/m^2/micrometer/sr"
        Lt.long_name = "Top of Atmosphere Measured Radiance"
        Lt.wavelength_units = "nanometers"
        Lt.fwhm = [5.5] * bands
        Lt.wavelengths = np.around(h1.spectral_coefficients, 1)
        Lt[:] = h1.l1b_cube


def print_nc(nc_file, path='', depth=0):

    indent = ''
    for i in range(depth):
        indent += '  '

    print(indent, '--- GROUP: "', path + nc_file.name, '" ---', sep='')

    print(indent, 'DIMENSIONS: ', sep='', end='')
    for d in nc_file.dimensions.keys():
        print(d, end=', ')
    print('')
    print(indent, 'VARIABLES: ', sep='', end='')
    for v in nc_file.variables.keys():
        print(v, end=', ')
    print('')

    print(indent, 'ATTRIBUTES: ', sep='', end='')
    for a in nc_file.ncattrs():
        print(a, end=', ')
    print('')

    print(indent, 'SUB-GROUPS: ', sep='', end='')
    for g in nc_file.groups.keys():
        print(g, end=', ')
    print('')
    print('')

    for g in nc_file.groups.keys():
        if nc_file.name == '/':
            newname = path + nc_file.name
        else:
            newname = path + nc_file.name + '/'
        print_nc(nc_file.groups[g], path=newname, depth=depth + 1)


# For Exporting on different Format

def get_geojson_str(sat_object) -> str:
    """Write the geojson metadata file.

    Args:
        writingmode (str, optional): The writing mode. Defaults to "w".

    Raises:
        ValueError: If the position file could not be found.
    """
    geojsondict = get_geojson_dict(sat_object)
    geojsonstr = json.dumps(geojsondict)

    return geojsonstr


def get_geojson_dict(sat_object) -> dict:
    """
    Get the data as a geojson dictionary.

    Returns:
        dict: The geojson dictionary.
    """

    # convert dictionary to json
    geojsondict = {}

    geojsondict["type"] = "Feature"

    geojsondict["geometry"] = {}
    geojsondict["geometry"]["type"] = "Point"
    geojsondict["geometry"]["coordinates"] = [
        sat_object.info["lon"], sat_object.info["lat"]]

    geojsondict["properties"] = {}
    name = sat_object.info["folder_name"].split("CaptureDL_")[-1].split("_")[0]
    geojsondict["properties"]["name"] = name
    geojsondict["properties"]["path"] = sat_object.info["top_folder_name"]

    geojsondict["metadata"] = {}
    date = sat_object.info["folder_name"].split(
        "CaptureDL_")[-1].split("20")[1].split("T")[0]
    date = f"20{date}"

    try:
        timestamp = datetime.strptime(date, "%Y-%m-%d_%H%MZ").isoformat()
    except BaseException:
        timestamp = datetime.strptime(date, "%Y-%m-%d").isoformat()

    geojsondict["metadata"]["timestamp"] = timestamp + "Z"
    geojsondict["metadata"]["frames"] = sat_object.info["frame_count"]
    geojsondict["metadata"]["bands"] = sat_object.info["image_width"]
    geojsondict["metadata"]["lines"] = sat_object.info["image_height"]
    geojsondict["metadata"]["satellite"] = "HYPSO-1"

    geojsondict["metadata"]["rad_coeff"] = os.path.basename(
        sat_object.radiometric_coeff_file)
    geojsondict["metadata"]["spec_coeff"] = os.path.basename(
        sat_object.spectral_coeff_file)

    geojsondict["metadata"]["solar_zenith_angle"] = sat_object.info["solar_zenith_angle"]
    geojsondict["metadata"]["solar_azimuth_angle"] = sat_object.info["solar_azimuth_angle"]
    geojsondict["metadata"]["sat_zenith_angle"] = sat_object.info["sat_zenith_angle"]
    geojsondict["metadata"]["sat_azimuth_angle"] = sat_object.info["sat_azimuth_angle"]

    return geojsondict
