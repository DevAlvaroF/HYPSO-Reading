import json
import os
from datetime import datetime

import netCDF4 as nc
import numpy as np
import pandas as pd

DEBUG = False


class h1data:
    def __init__(self, top_folder_name, load_cubes=True):
        

        if load_cubes:
            
    def set_radiometric_coefficients(self, path: str) -> np.ndarray:
        """Set the radiometric coefficients from the csv file.

        Args:
            path (str, optional): Path to the radiometric coefficients csv file. Defaults to None.
            sets the rad_file attribute to the path.
            if no path is given, the rad_file path is used.

        Returns:
            np.ndarray: The radiometric coefficients.

        """
        if path is None:
            coeff_path = self.get_path_to_coefficients()
            radiometric_coeff_csv_name = self.rad_file
            radiometric_coeff_file = os.path.join(
                coeff_path, radiometric_coeff_csv_name)
        else:
            radiometric_coeff_file = path
            self.rad_file = path

        try:
            radiometric_coeffs = np.genfromtxt(
                radiometric_coeff_file, delimiter=',')
        except BaseException:
            radiometric_coeffs = None

        self.radiometric_coefficients = radiometric_coeffs
        self.calibrate_cube()
        return radiometric_coeffs

    def get_raw_cube(self) -> np.ndarray:
        """Get the raw data from the top folder of the data.

        Returns:
            np.ndarray: The raw data.
        """
        # find file ending in .bip
        for file in os.listdir(self.info["top_folder_name"]):
            if file.endswith(".bip"):
                path_to_bip = os.path.join(
                    self.info["top_folder_name"], file)
                break

        cube = np.fromfile(path_to_bip, dtype='uint16')
        if DEBUG:
            print(path_to_bip)
            print(cube.shape)
        cube = cube.reshape(
            (-1, self.info["image_height"], self.info["image_width"]))

        # reverse the order of the third dimension
        cube = cube[:, :, ::-1]

        # ---------------------------------------------------------
        # Do Second Order Effect Correction
        # ----------------------------------------------------------
        # import cv2
        # # # this function will be called whenever the mouse is right-clicked
        # def mouse_callback(event, x, y, flags, params):
        #
        #     # lef-click event value is 1
        #     if event == 1:
        #         # store the coordinates of the right-click event
        #         self.left_clicks.append([x, y])
        #
        #         # this just verifies that the mouse data is being collected
        #         print(self.left_clicks)
        #
        # # Read raw image and normalize to 1
        # cube_for_image = cube.copy()
        # cube_for_image = (255.0 * cube_for_image / np.nanmax(cube_for_image)).astype(np.uint8)
        # img = cube_for_image[:, :, [80, 60, 30]]
        #
        # # Get window size based on scaled image size
        # scale_width = 640 / img.shape[1]
        # scale_height = 480 / img.shape[0]
        # scale = min(scale_width, scale_height)
        # window_width = int(img.shape[1] * scale)
        # window_height = int(img.shape[0] * scale)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', window_width, window_height)
        #
        # # set mouse callback function for window
        # cv2.setMouseCallback('image', mouse_callback)
        # cv2.imshow('image', img)
        # while True:
        #     k = cv2.waitKey(0)  # change the value from the original 0 (wait forever) to something appropriate
        #     if k == 27:
        #         break
        # print("Out of loop")
        # cv2.destroyAllWindows()
        # #
        # # # Get points: Odd numbers are shallow pair numbers are deep
        # deep_points = []
        # shallow_points = []
        # for i in range(len(self.left_clicks)):
        #     # If odd, shallow
        #     if i % 2 != 0:
        #         shallow_points.append(self.left_clicks[i])
        #     # if even, deep
        #     elif i % 2 == 0:
        #         deep_points.append(self.left_clicks[i])
        #
        # # Get line equation to model the value to interpolate to
        # mid_idx = int(len(self.wavelengths) / 2)
        #
        # delta_y = np.ceil(self.wavelengths[mid_idx]) - np.ceil(self.wavelengths[0])
        # delta_x = self.wavelengths[-1] - self.wavelengths[mid_idx + 1]
        # m = delta_y / delta_x
        #
        # b = self.wavelengths[mid_idx] - self.wavelengths[-1] * m
        #
        # all_points_correction = []
        # correction_lambda = []
        # corrected_index = []
        # for p in range(len(shallow_points)):
        #     pair_correction = []
        #     correction_lambda = []
        #     corrected_index = []
        #
        #     def window_size_optimal_spectra(raw_cube, index_row, index_col):
        #         min_std_idx = 0
        #         min_std = np.inf
        #         for i in range(10):
        #             if i > 0:
        #                 pixel_spectra = raw_cube[index_row - i:index_row + i + 1, index_col - i:index_col + i + 1, :]
        #                 mean_spectra = np.mean(pixel_spectra, axis=(0, 1))
        #                 std_area = np.std(mean_spectra)
        #                 if std_area < min_std:
        #                     min_std = std_area
        #                     min_std_idx = i
        #         pixel_spectra = raw_cube[index_row - min_std_idx:index_row + min_std_idx + 1,
        #                         index_col - min_std_idx:index_col + min_std_idx + 1, :]
        #         mean_spectra_final = np.mean(pixel_spectra, axis=(0, 1))
        #         return mean_spectra_final
        #
        #     # Get Raw cube data
        #     shallow_row = shallow_points[p][1]  # opencv has X and Y
        #     shallow_col = shallow_points[p][0]
        #     shallow_pixel_spectra = window_size_optimal_spectra(cube, shallow_row, shallow_col)
        #     # shallow_pixel_spectra = cube[shallow_row - 2:shallow_row + 2, shallow_col - 2:shallow_col + 2, :]
        #     # shallow_pixel_spectra = np.mean(shallow_pixel_spectra, axis=(0, 1))
        #
        #     deep_row = deep_points[p][1]  # opencv has X and Y
        #     deep_col = deep_points[p][0]
        #     deep_pixel_spectra = window_size_optimal_spectra(cube, deep_row, deep_col)
        #     # deep_pixel_spectra = cube[deep_row - 2:deep_row + 2, deep_col - 2:deep_col + 2, :]
        #     # deep_pixel_spectra = np.mean(deep_pixel_spectra, axis=(0, 1))
        #
        #     for i, w in enumerate(self.wavelengths):
        #         if i > mid_idx:
        #             # save lambda used for correction
        #             correction_lambda.append(w)
        #             corrected_index.append(i)
        #             # Get lambda for correction based on current lambda and found line equation
        #             lambda_for_correction = (m * w) + b
        #
        #             # Interpolate radiance based on lambda and lambda for correction
        #             shallow_interp_radiance = np.interp(lambda_for_correction, self.wavelengths, shallow_pixel_spectra)
        #             deep_interp_radiance = np.interp(lambda_for_correction, self.wavelengths, deep_pixel_spectra)
        #
        #             # Get scale factor for the current lambda
        #             numerator = shallow_pixel_spectra[i] - deep_pixel_spectra[i]
        #             denominator = shallow_interp_radiance - deep_interp_radiance
        #             scale_factor = numerator / denominator
        #
        #             # store the correction
        #             pair_correction.append(scale_factor)
        #
        #     # Store
        #     if len(all_points_correction) == 0:
        #         all_points_correction = pair_correction
        #     else:
        #         all_points_correction = np.row_stack((all_points_correction, pair_correction))
        #
        # # Estimate mean values
        # mean_correction = np.mean(all_points_correction, axis=0)
        #
        # # plot
        # import matplotlib.pyplot as plt
        # plt.rcParams['axes.facecolor'] = 'white'
        # plt.rcParams['figure.facecolor'] = 'white'
        # fig = plt.figure()
        # lgnd = []
        # for data_idx in range(all_points_correction.shape[0]):
        #     plt.plot(correction_lambda, all_points_correction[data_idx, :])
        #     lgnd.append(f"Pair {data_idx + 1}")
        # plt.plot(correction_lambda, mean_correction, 'k')
        # lgnd.append("Mean")
        # plt.xlabel("Correction Factor")
        # plt.ylabel("DN")
        # plt.title("Mean correction factor")
        # plt.legend(lgnd)
        # plt.show()
        #
        # output_cube = cube.copy()
        # # correcting cube
        # from tqdm import tqdm
        # for idx in tqdm(range(len(corrected_index)), desc="Second Order Correction"):
        #     idx_to_correct = corrected_index[idx]
        #     lambda_for_correction = (m * correction_lambda[idx]) + b
        #     for row in range(cube.shape[0]):
        #         for col in range(cube.shape[1]):
        #             current_spectra = cube[row, col, :]
        #             interp_value = np.interp(lambda_for_correction, self.wavelengths, current_spectra)
        #
        #             # Correct Values
        #             output_cube[row, col, idx_to_correct] = cube[row, col, idx_to_correct] - (mean_correction[
        #                                                                                           idx] * interp_value)
        # Update Cube
        # cube = output_cube
        return cube

    def apply_radiometric_calibration(
            self,
            frame,
            exp,
            background_value,
            radiometric_calibration_coefficients):
        ''' Assumes input is 12-bit values, and that the radiometric calibration
        coefficients are the same size as the input image.

        Note: radiometric calibration coefficients have original size (684,1080),
        matching the "normal" AOI of the HYPSO-1 data (with no binning).'''

        frame = frame - background_value
        frame_calibrated = frame * radiometric_calibration_coefficients / exp

        return frame_calibrated

    def calibrate_cube(self) -> np.ndarray:
        """Calibrate the raw data cube."""

        background_value = self.info['background_value']
        exp = self.info['exp']
        image_height = self.info['image_height']
        image_width = self.info['image_width']

        # Radiometric calibration
        num_frames = self.info["frame_count"]
        cube_calibrated = np.zeros([num_frames, image_height, image_width])
        for i in range(num_frames):
            frame = self.raw_cube[i, :, :]
            frame_calibrated = self.apply_radiometric_calibration(
                frame, exp, background_value, self.radiometric_coefficients)
            cube_calibrated[i, :, :] = frame_calibrated

        self.l1b_cube = cube_calibrated
        # self.wavelengths = self.spec_coefficients

        return cube_calibrated

    def get_geojson_str(self) -> str:
        """Write the geojson metadata file.

        Args:
            writingmode (str, optional): The writing mode. Defaults to "w".

        Raises:
            ValueError: If the position file could not be found.
        """
        geojsondict = self.get_geojson_dict()
        geojsonstr = json.dumps(geojsondict)

        return geojsonstr

    def get_geojson_dict(self) -> dict:
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
            self.info["lon"], self.info["lat"]]

        geojsondict["properties"] = {}
        name = self.info["folder_name"].split("CaptureDL_")[-1].split("_")[0]
        geojsondict["properties"]["name"] = name
        geojsondict["properties"]["path"] = self.info["top_folder_name"]

        geojsondict["metadata"] = {}
        date = self.info["folder_name"].split(
            "CaptureDL_")[-1].split("20")[1].split("T")[0]
        date = f"20{date}"

        try:
            timestamp = datetime.strptime(date, "%Y-%m-%d_%H%MZ").isoformat()
        except BaseException:
            timestamp = datetime.strptime(date, "%Y-%m-%d").isoformat()

        geojsondict["metadata"]["timestamp"] = timestamp + "Z"
        geojsondict["metadata"]["frames"] = self.info["frame_count"]
        geojsondict["metadata"]["bands"] = self.info["image_width"]
        geojsondict["metadata"]["lines"] = self.info["image_height"]
        geojsondict["metadata"]["satellite"] = "HYPSO-1"

        geojsondict["metadata"]["rad_coeff"] = self.rad_file
        geojsondict["metadata"]["spec_coeff"] = self.spec_file

        geojsondict["metadata"]["solar_zenith_angle"] = self.info["solar_zenith_angle"]
        geojsondict["metadata"]["solar_azimuth_angle"] = self.info["solar_azimuth_angle"]
        geojsondict["metadata"]["sat_zenith_angle"] = self.info["sat_zenith_angle"]
        geojsondict["metadata"]["sat_azimuth_angle"] = self.info["sat_azimuth_angle"]

        return geojsondict

    def write_rgb(
            self,
            path_to_save: str,
            R_wl: float = 650,
            G_wl: float = 550,
            B_wl: float = 450) -> None:
        """
        Write the RGB image.

        Args:
            path_to_save (str): The path to save the RGB image.
            R_wl (float, optional): The wavelength for the red channel. Defaults to 650.
            G_wl (float, optional): The wavelength for the green channel. Defaults to 550.
            B_wl (float, optional): The wavelength for the blue channel. Defaults to 450.
        """
        import plotly.express as px

        # check if file ends with .jpg
        if not path_to_save.endswith('.png'):
            path_to_save = path_to_save + '.png'

        R = np.argmin(abs(self.spec_coefficients - R_wl))
        G = np.argmin(abs(self.spec_coefficients - G_wl))
        B = np.argmin(abs(self.spec_coefficients - B_wl))

        # get the rgb image
        rgb = self.l1b_cube[:, :, [R, G, B]]

        fig = px.imshow(rgb)
        fig.update_layout(
            autosize=False,
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0}
        )
        fig.write_image(path_to_save)

        return


def write_h1data_as_geojson(path_to_save: str, path_to_h1data: str) -> None:
    # check if file ends with .geojson
    if not path_to_save.endswith('.geojson'):
        path_to_save = path_to_save + '.geojson'

    h1 = h1data(path_to_h1data)

    with open(path_to_save, 'w') as f:
        f.write(h1.get_geojson_str())


def write_h1data_as_NetCDF4(path_to_save: str, path_to_h1data: str, h1dataInstance=None) -> None:
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
    if h1dataInstance is None:
        h1 = h1data(path_to_h1data)
    else:
        h1 = h1dataInstance

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
        f.radiometric_file = h1.rad_file
        f.spectral_file = h1.spec_file
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
        Lt.wavelengths = np.around(h1.spec_coefficients, 1)
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


# if __name__ == "__main__":
def main(path_to_h1data, path_to_save, format='nc'):
    print("Writing HYPSO-1 data as NetCDF4 or geojson, as specified by the user.")
    # import argparse
    #
    # parser = argparse.ArgumentParser(
    #     description="Writes HYPSO-1 data as NetCDF4 or geojon, as specified by the user.")
    # parser.add_argument(
    #     "-i",
    #     "--path_to_h1data",
    #     help="Path to H1 data folder or .nc file to read.",
    #     required=True)
    # parser.add_argument(
    #     "-o", "--path_to_save", help="Relative path to save file")
    # parser.add_argument(
    #     "-f",
    #     "--format",
    #     help="Format to save file as. Either 'nc' or 'geojson'.")
    # parser.add_argument("--DEBUG", help="Debug mode", action="store_true")
    # parser.add_argument(
    #     "-r",
    #     "--read",
    #     help="Reads a NetCDF4 file and prints its contents",
    #     action="store_true")
    # args = parser.parse_args()
    #
    # DEBUG = args.DEBUG

    # if args.read:
    #     print_nc(nc.Dataset(args.path_to_h1data))
    # elif args.format == "nc" and args.path_to_save is not None:
    #     write_h1data_as_NetCDF4(args.path_to_save, args.path_to_h1data)
    # elif args.format == "geojson" and args.path_to_save is not None:
    #     write_h1data_as_geojson(args.path_to_save, args.path_to_h1data)
    # elif args.format is None or args.path_to_save is None:
    #     raise ValueError(
    #         "Please specify a path to save the file and a format.")
    # else:
    #     raise ValueError("Supported formats are 'nc' or 'geojson'.")

    if format == 'nc' and path_to_save is not None:
        write_h1data_as_NetCDF4(path_to_save, path_to_h1data)
    elif format == 'geojson' and path_to_save is not None:
        write_h1data_as_geojson(path_to_save, path_to_h1data)
