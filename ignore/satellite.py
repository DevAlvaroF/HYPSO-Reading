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

    def print_summary(self):
        print("\n---------------- SUMMARY ----------------")
        print("Sensors Added:\n")
        print(list(self.other_sensors.keys()), "\n")

        print("Chlorophyll Estimations:\n")
        print(list(self.chl.keys()), "\n")

    def load_paths(self, gcpPath):
        # Find file with extension ".hdr"
        self.hdrFilePath = glob.glob(self.datasetDir + '/*.hdr')[0]

        # Get .geotiff file from geotiff folder
        geotiff_dir = [f.path for f in os.scandir(self.datasetDir) if (
            f.is_dir() and ('geotiff' in os.path.basename(os.path.normpath(f))))][0]

        self.geotiffFilePath = [join(geotiff_dir, f) for f in listdir(geotiff_dir) if (
            isfile(join(geotiff_dir, f)) and ('8bit' in f))][0]

        # Get Latitude and Longitude .dat files
        dat_files = glob.glob(self.datasetDir + '/*.dat')
        longitude_dataPath = [f for f in dat_files if 'longitudes' in f][0]
        latitude_dataPath = [f for f in dat_files if 'latitudes' in f][0]

        # Load Latitude
        self.lat = np.fromfile(latitude_dataPath, dtype='float32')
        self.lat.shape = self.spatialDim

        # Load Longitude
        self.lon = np.fromfile(longitude_dataPath, dtype='float32')
        self.lon.shape = self.spatialDim

        # Load GeoTiff Metadata with gdal
        ds = gdal.Open(self.geotiffFilePath)
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        inproj = osr.SpatialReference()
        inproj.ImportFromWkt(proj)

        self.projection_metadata = {
            'data': data,
            'gt': gt,
            'proj': proj,
            'inproj': inproj,
        }

        if gcpPath is not None:
            # gcpPath = r"C:\Users\alvar\OneDrive\Desktop\karachi_2023-02-06_0531Z-bin3.png.points"
            # lat_coeff, lon_coeff = reference_correction.geotiff_correction(gcpPath, self.projection_metadata)
            M = geotiff_correction(gcpPath, self.projection_metadata)

            for i in range(self.lat.shape[0]):
                for j in range(self.lat.shape[1]):
                    X = self.lon[i, j]
                    Y = self.lat[i, j]
                    # Affine Matrix
                    # current_coord = np.array([[X], [Y], [1]])
                    # res_mult = np.matmul(M, current_coord)
                    # newLon = res_mult[0]
                    # newLat = res_mult[1]

                    # estimateAffinePartial2D

                    # current_coord = np.array([[X], [Y], [1]])
                    # res_mult = np.matmul(M, current_coord)
                    # newLon = res_mult[0]
                    # newLat = res_mult[1]

                    # Second Degree Polynomial (Scikit)
                    # lon_coeff = M.params[0]
                    # lat_coeff = M.params[1]
                    # newLat, newLon = reference_correction.calculate_poly_geo_coords_skimage(X, Y, lon_coeff, lat_coeff)

                    # Np lin alg
                    LonM = M[0]
                    newLon = LonM[0] * X + LonM[1]

                    LatM = M[1]
                    newLat = LatM[0] * Y + LatM[1]

                    self.lat[i, j] = newLat
                    self.lon[i, j] = newLon

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

    def hypso_extent_lon_lat(self, inproj_value):
        # Convert WKT projection information into a cartopy projection
        projcs = inproj_value.GetAuthorityCode('PROJCS')
        projection_img = ccrs.epsg(projcs)

        # Transform Current lon and lat limits to another
        new_max_lon = np.max(self.lon)
        new_max_lat = np.max(self.lat)
        new_min_lon = np.min(self.lon)
        new_min_lat = np.min(self.lat)

        # Convert lat and lon to the image CRS so we create Projection with Dataset CRS
        dataset_proj = pyproj.Proj(projection_img)  # your data crs

        # Transform Coordinates to Image CRS
        transformed_min_lon, transformed_min_lat = dataset_proj(
            new_min_lon, new_min_lat, inverse=False)
        transformed_max_lon, transformed_max_lat = dataset_proj(
            new_max_lon, new_max_lat, inverse=False)

        transformed_img_extent = (transformed_min_lon, transformed_max_lon,
                                  transformed_min_lat, transformed_max_lat)

        extent_lon_min = new_min_lon
        extent_lon_max = new_max_lon

        extent_lat_min = new_min_lat
        extent_lat_max = new_max_lat

        return [extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max]
    # def add_colorbar(self, im, aspect=15, pad_fraction=0.5, **kwargs):
    #     """Add a vertical color bar to an image plot."""
    #     divider = axes_grid1.make_axes_locatable(im.axes)
    #     width = axes_grid1.axes_size.AxesY(im.axes, aspect=1. / aspect)
    #     pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    #     current_ax = plt.gca()
    #     cax = divider.append_axes(self.color_bar_pos, size=width, pad=pad)
    #     plt.sca(current_ax)
    #     cbar = im.axes.figure.colorbar(im, cax=cax, **kwargs)
    #     cbar.ax.yaxis.set_major_locator(ticker.AutoLocator())
    #     cbar.ax.yaxis.set_minor_locator(ticker.AutoLocator())
    #     cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False, useOffset=False))
    #     cbar.ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    #     cbar.ax.ticklabel_format(style='plain', scilimits=(-2, 3))
    #     return cbar

    # def self_chl_cartopy(self, plotTitle='Chlorophyll Estimation', dataTrim=[[0, -1], [0, -1]]):
    #
    #     # Plot 2D array with lon and lat coordinates
    #     # Input:
    #     #       chl_2d_array: 2D array with chlorophyll values
    #     #       lon: 2D array with coordinates per pixel
    #     #       lat: 2D array with coordinates per pixel
    #     #       color_bar_ppos: Position of the color bar -> 'right' or 'bottom'
    #     #       chl_range: Known range of Chlorophyll values to normalize colors [min, max]
    #     #
    #     # Output:
    #     #       None (Just visualize plot)
    #
    #     chl_extractor = Chlorophyll(satellite=self)
    #     self.chl_ocx = chl_extractor.ocx_estimation()
    #     # self.chl_ocx = chl_extractor.siver_TOA_chl()
    #
    #     chl_array, chl_lat_array, chl_lon_array = self.chl_ocx, self.lat, self.lon
    #
    #     # Read GeoTiff with gdal
    #     ds = gdal.Open(self.geotiffFilePath)
    #     data = ds.ReadAsArray()
    #     gt = ds.GetGeoTransform()
    #     proj = ds.GetProjection()
    #     inproj = osr.SpatialReference()
    #     inproj.ImportFromWkt(proj)
    #
    #     # Convert WKT projection information into a cartopy projection
    #     projcs = inproj.GetAuthorityCode('PROJCS')
    #     projection_img = ccrs.epsg(projcs)
    #
    #     projection = ccrs.Mercator()
    #     # Specify CRS, that will be used to tell the code, where should our data be plotted
    #     crs = ccrs.PlateCarree()
    #     # Now we will create axes object having specific projection
    #
    #     fig = plt.figure(figsize=(10, 10), dpi=450)
    #     fig.patch.set_alpha(1)
    #     # ax = fig.add_subplot(projection=projection,
    #     #                      frameon=True)
    #     ax = fig.add_subplot(projection=projection,
    #                          frameon=True)
    #
    #     # Draw gridlines in degrees over Mercator map
    #     gl = ax.gridlines(draw_labels=True,
    #                       linewidth=.6, color='gray', alpha=0.5, linestyle='-.')
    #     gl.xlabel_style = {"size": 7}
    #     gl.ylabel_style = {"size": 7}
    #
    #     # To plot borders and coastlines, we can use cartopy feature
    #     ax.add_feature(cf.COASTLINE.with_scale("10m"), lw=0.5)
    #     ax.add_feature(cf.BORDERS.with_scale("10m"), lw=0.3)
    #     # ax.add_feature(cf.LAND, zorder=100, edgecolor='k')  # Covers Data in land
    #
    #     lat = self.lat
    #     lon = self.lon
    #
    #     # min_lon, min_lat, max_lon, max_lat
    #     print(
    #         f"From .dat file\nmin_lon {np.min(lon)}, min_lat {np.min(lat)},\nmax_lon {np.max(lon)}, max_lat {np.max(lat)}")
    #
    #     new_max_lon = np.max(lon)
    #     new_max_lat = np.max(lat)
    #     new_min_lon = np.min(lon)
    #     new_min_lat = np.min(lat)
    #     geom = box(new_min_lon, new_min_lat,
    #                new_max_lon, new_max_lat)
    #
    #     img_extent = (new_min_lon, new_max_lon,
    #                   new_min_lat, new_max_lat)
    #
    #     # Conbert lat and lon to the image CRS so we create Projection with Dataset CRS
    #     dataset_proj = pyproj.Proj(projection_img)  # your data crs
    #     # Transform Coordinates to Image CRS
    #     transformed_min_lon, transformed_min_lat = dataset_proj(
    #         new_min_lon, new_min_lat, inverse=False)
    #     transformed_max_lon, transformed_max_lat = dataset_proj(
    #         new_max_lon, new_max_lat, inverse=False)
    #
    #     transformed_img_extent = (transformed_min_lon, transformed_max_lon,
    #                               transformed_min_lat, transformed_max_lat)
    #
    #     # img = plt.imread(r"D:\4th Semester\Code\Hypso\hypso\tmp.jpg")
    #     # crs is PlateCarree -> we are explicitly telling axes, that we are creating bounds that are in degrees
    #
    #     ax.set_extent([new_min_lon / self.plotZoomFactor, new_max_lon * self.plotZoomFactor,
    #                    new_min_lat / self.plotZoomFactor, new_max_lat * self.plotZoomFactor], crs=crs)
    #
    #     # ax.set_extent([lon_min, lon_max, lat_min, lat_max],crs=crs)
    #
    #     # img = Image.open(r"D:\4th Semester\Code\Hypso\hypso\tmp.jpg")
    #     # add the image. Because this image was a tif, the "origin" of the image is in the
    #     # upper left corner
    #     # ax.imshow(data[:3, :, :].transpose((1, 2, 0)), origin='upper', extent=img_extent,
    #     #           transform=crs, resample=False, interpolation="nearest")
    #     rgb_array = data[:3, :, :]
    #     masked_data = np.ma.masked_where(rgb_array == 0, rgb_array)
    #     # Clipping to simulate Alpha channel
    #     # rgb_array[np.where(rgb_array == 0)] = np.nan
    #
    #     # ax.imshow(np.rot90(masked_data.transpose((1, 2, 0)), k=2), origin='upper', extent=transformed_img_extent,
    #     #           transform=projection_img, zorder=1)
    #
    #     # * -----------------------          CHLOROPHYLL FROM HYPSO   ------------------------------
    #     # TODO: Change this chl_display range to match modis and sentinel
    #     max_chlr_val = np.nanmax(chl_array)
    #     upper_limit_chl = 100 if max_chlr_val > 100 else max_chlr_val
    #
    #     chl_range = [np.nanmin(chl_array), upper_limit_chl]  # old: [0.01, 100] [0.3, 1]
    #
    #     # chl_range = [0.01, 100]
    #     print(chl_range)
    #     # Create meshgrid Sparse
    #     if len(chl_lon_array.shape) == 1:
    #         chl_lon_array, chl_lat_array = np.mesgrid(
    #             chl_lon_array, chl_lat_array, sparse=True)
    #
    #     # Now, we will specify extent of our map in minimum/maximum longitude/latitude
    #     lon_min = chl_lon_array.min()
    #     lon_max = chl_lon_array.max()
    #     lat_min = chl_lat_array.min()
    #     lat_max = chl_lat_array.max()
    #     # Cholorphyll Range (Common)
    #     cmin, cmax = chl_range
    #     norm = colors.LogNorm(cmin, cmax)
    #     im = ax.pcolormesh(chl_lon_array, chl_lat_array, chl_array,
    #                        cmap=plt.cm.jet, transform=ccrs.PlateCarree(), norm=norm, zorder=0)
    #
    #     # Colourmap with axes to match figure size
    #     cbar = plt.colorbar(im, location=self.color_bar_pos, shrink=1, ax=ax, pad=0.05)
    #
    #     def myLogFormat(y, pos):
    #         # Find the number of decimal places required
    #         decimalplaces = int(np.maximum(-np.log10(y), 0))  # =0 for numbers >=1
    #         if (decimalplaces == 0):
    #             # Insert that number into a format string
    #             formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    #             # Return the formatted tick label
    #             return formatstring.format(y)
    #         else:
    #             formatstring = '{:2.1e}'.format(y)
    #             return formatstring
    #
    #     cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
    #     cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
    #
    #     cbar.ax.yaxis.set_minor_formatter(mticker.ScalarFormatter(useMathText=False, useOffset=False))
    #     cbar.ax.xaxis.set_minor_formatter(mticker.ScalarFormatter(useMathText=False, useOffset=False))
    #
    #     cbar.set_label(f" Chlorophyll Concentration [mg m^-3]")
    #
    #     # --------------------------------------------------------------------------------------------
    #     # ax.add_geometries([geom], crs=crs, alpha=0.3)
    #
    #     plt.title(plotTitle)
    #     plt.show()
