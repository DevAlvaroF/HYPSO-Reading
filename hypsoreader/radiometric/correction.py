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

    background_value = info_sat['background_value']
    exp = info_sat['exp']
    image_height = info_sat['image_height']
    image_width = info_sat['image_width']

    # Radiometric calibration
    num_frames = info_sat["frame_count"]
    cube_calibrated = np.zeros([num_frames, image_height, image_width])
    for i in range(num_frames):
        frame = raw_cube[i, :, :]
        frame_calibrated = apply_radiometric_calibration(
            frame, exp, background_value, radiometric_coefficients)
        cube_calibrated[i, :, :] = frame_calibrated

    l1b_cube = cube_calibrated

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
