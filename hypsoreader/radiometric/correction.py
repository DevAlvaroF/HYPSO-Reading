import numpy as np
import os


def get_coefficients_from_file(coeff_path: str) -> np.ndarray:
    """Get the coefficients from the csv file.

    Args:
        path (str, optional): Path to the radiometric coefficients csv file. Defaults to None.
        sets the rad_file attribute to the path.
        if no path is given, the rad_file path is used.

    Returns:
        np.ndarray: The coefficients.

    """

    try:
        coeffs = np.genfromtxt(
            coeff_path, delimiter=',')
    except BaseException:
        coeffs = None
        raise ValueError("Could not read coefficients file.")

    return coeffs


def calibrate_cube(info_sat: dict, raw_cube: np.ndarray, radiometric_coefficients: np.ndarray) -> np.ndarray:
    """Calibrate the raw data cube."""
    DEBUG = False

    background_value = info_sat['background_value']
    exp = info_sat['exp']
    image_height = info_sat['image_height']
    image_width = info_sat['image_width']

    # Radiometric calibration
    num_frames = info_sat["frame_count"]
    cube_calibrated = np.zeros([num_frames, image_height, image_width])

    if DEBUG:
        print("F:", num_frames, "H:", image_height, "W:", image_width)
        print("Radioshape: ", radiometric_coefficients.shape)

    f_standard = num_frames
    h_standard = radiometric_coefficients.shape[0]
    if (num_frames == f_standard and image_height == h_standard):
        for i in range(num_frames):
            frame = raw_cube[i, :, :]
            frame_calibrated = apply_radiometric_calibration(
                frame, exp, background_value, radiometric_coefficients)
            cube_calibrated[i, :, :] = frame_calibrated
    elif True:
        print("DEV No Standard Dimensions. Image height: ", image_height)
        # TODO: Delete this
        for i in range(num_frames):
            frame = raw_cube[i, :, :]
            frame_calibrated = apply_radiometric_calibration(
                frame, exp, background_value, np.ones_like(frame))
            cube_calibrated[i, :, :] = frame_calibrated
    else:
        print("Format w/o Standard Dimensions for rad coeffs detected. Image height: ", image_height)

        # 1-pad overflowing and cut underflowing radiometric coeffs
        # before generating a calibrated cube.
        h_diff = image_height - h_standard  # difference from standard
        f_diff = num_frames - f_standard
        hparam = int(h_diff/2)
        fparam = int(f_diff/2)
        if (h_diff > 0):
            padded_coeffs = np.pad(radiometric_coefficients, ((
                hparam, hparam), (0, 0)), constant_values=1)
            radiometric_coefficients = padded_coeffs
        elif (h_diff < 0):
            trimmed_coeffs = radiometric_coefficients
            for i in range(abs(hparam)):
                trimmed_coeffs = np.delete(trimmed_coeffs, -1, 0)
                trimmed_coeffs = np.delete(trimmed_coeffs, 0, 0)
            radiometric_coefficients = trimmed_coeffs
        if (f_diff > 0):
            padded_coeffs = np.pad(radiometric_coefficients, ((
                0, 0), (fparam, fparam)), constant_values=1)
            radiometric_coefficients = padded_coeffs
        elif (f_diff < 0):
            trimmed_coeffs = radiometric_coefficients
            for i in range(abs(fparam)):
                trimmed_coeffs = np.delete(trimmed_coeffs, -1, 1)
                trimmed_coeffs = np.delete(trimmed_coeffs, 0, 1)
            radiometric_coefficients = trimmed_coeffs

        # "calibrate" as normal (data will effectively be uncalibrated for much of the capture)
        for i in range(num_frames):
            frame = raw_cube[i, :, :]
            frame_calibrated = apply_radiometric_calibration(
                frame, exp, background_value, radiometric_coefficients)
            cube_calibrated[i, :, :] = frame_calibrated

    l1b_cube = cube_calibrated
    # self.wavelengths = self.spec_coefficients

    return l1b_cube


def apply_radiometric_calibration(
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
