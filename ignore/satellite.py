# -----------------------------------------------------------------------------
#                               IMPORT LIBRARIESn# -----------------------------------------------------------------------------
# File Manipulation
# Modified from pip to have as local directory
from .WaterDetect import waterdetect as wd
from .Utilities import SatPlot, geotiff_correction
from .Chlorophyll import Chlorophyll
import numpy as np


# GIS
import cartopy.crs as ccrs

from osgeo import gdal, osr
import pyproj

# Read GSI Data
import xarray as xr  # To map to .NC metadata file

# Plot and Image Gen
import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'

# Numerical

# -----------------------------------------------------------------------------
#                          IMPORT CUSTOM LIBRARIES
# -----------------------------------------------------------------------------

# from chlorophyll import Chlorophyllrgood to have crunchm


class Satellite:
    # 10:1Paths

    def export_dataset(self):

        other_sensors_dic = {}
        for key in self.other_sensors:
            other_sensors_dic[key] = {
                "hypercube": self.other_sensors[key].hypercube,
                "wavelengths": self.other_sensors[key].wavelengths,
                'chl': self.other_sensors[key].chl,
            }

        # Exclude OCX Visualizations
        chl_dict = {}
        for key in self.chl:
            split_key = key.split('_')
            if not any(x in self.chl_algorithms for x in split_key):
                chl_dict[key] = self.chl[key]

        dict_export = {
            'chl': chl_dict,
            'hypercube': self.hypercube,
            'waterMask': self.waterMask,
            'lat': self.lat,
            'lon': self.lon,
            'wavelengths': self.wavelengths
        }

        file_name = os.path.basename(
            os.path.normpath(self.datasetDir)) + '.npy'

        np.save(os.path.join(self.outputDir, file_name), dict_export)

        # To Load data=np.load("file.npy")
        # data.item().get('chl')
        # data.item().get('hypercube')

    def add_acolite(self, nc_path, overlapSatImg=False):
        print("\n\n-------  Loading L2 Acolite Cube  ----------")
        # Extract corrected hypercube
        self.hypercube['L2'] = np.empty_like(self.hypercube['L1'])

        with xr.open_dataset(nc_path) as ncdat:
            keys = [i for i in ncdat.data_vars]
            try:
                keys.remove('lat')
                keys.remove('lon')
            except:
                print("Couldn't find lat and lon keys on Acolite .nc")

            toa_keys = [k for k in keys if 'rhos' not in k]
            surface_keys = [kk for kk in keys if 'rhot' not in kk]

            # Add Cube
            for i, k in enumerate(surface_keys):
                self.hypercube['L2'][:, :, i] = ncdat.variables[k].values

        # Get Chlorophyll
        self.chlEstimator.ocx_estimation(
            satName='hypsoacolite', overlapSatImg=overlapSatImg)

        print("Done Importing Acolite L2 Data")

        return self.hypercube['L2']

    def getWaterMask(self):
        print("\n\n-------  Naive-Bayes Water Mask Detector  ----------")
        water_config_file = './hypso/WaterDetect/WaterDetect.ini'
        water_config_file = os.path.abspath(water_config_file)

        config = wd.DWConfig(config_file=water_config_file)
        print(config.clustering_bands)
        print(config.detect_water_cluster)

        # Band 3 in Sentinel-2 is centered at 560nm with a FWHM: 34.798nm
        # Band 560.6854258053552 at index 49 is the closes Hypso equivalent
        b3 = self.hypercube['L2'][:, :, 49]

        # Band 8 in Sentinel-2 is NIR centered at 835nm with a FWHM: 104.784n8
        # Hypso last 2 bands (maybe noisy) are at 799.10546171nm & 802.51814719nm at index 118 and 119
        nir = self.hypercube['L2'][:, :, 118]

        # Division by 10000 may not be needed
        bands = {'Green': b3, 'Nir': nir}
        wmask = wd.DWImageClustering(bands=bands, bands_keys=[
                                     'Nir', 'ndwi'], invalid_mask=None, config=config)

        mask = wmask.run_detect_water()

        mask = wmask.water_mask

        # Mask Adjustment
        boolMask = mask.copy()
        boolMask[boolMask != 1] = 0
        boolMask = boolMask.astype(bool)

        # # # TODO: Improve Water Map Selection (Non threshold dependant)
        # def get_is_water_map(cube, wl, binary_threshold=2.9):
        #     C1 = np.argmin(abs(wl - 460))
        #     C2 = np.argmin(abs(wl - 650))
        #     ret = np.zeros([cube.shape[0], cube.shape[1]])
        #     for xx in range(cube.shape[0]):
        #         for yy in range(cube.shape[1]):
        #             spectra = cube[xx, yy, :]
        #             ret[xx, yy] = spectra[C1] / spectra[C2]
        #
        #     return ret > binary_threshold
        #
        # boolMask = get_is_water_map(self.l1a_cube, self.wavelengths)

        self.waterMask = boolMask
