import numpy as np
import os


def get_coefficients_from_file(coeff_path: str) -> np.ndarray:
    coefficients = None
    try:
        coefficients = np.genfromtxt(
            coeff_path, delimiter=',')
    except BaseException:
        coefficients = None
        raise ValueError("Could not read coefficients file.")

    return coefficients


def get_coefficients_from_dict(coeff_dict: str) -> np.ndarray:
    """Get the coefficients from the csv file.

    Args:
        path (str, optional): Path to the radiometric coefficients csv file. Defaults to None.
        sets the rad_file attribute to the path.
        if no path is given, the rad_file path is used.

    Returns:
        np.ndarray: The coefficients.

    """
    coeffs = coeff_dict.copy()
    for k in coeff_dict:
        coeffs[k] = get_coefficients_from_file(coeff_dict[k])

    return coeffs


def calibrate_cube(info_sat: dict, raw_cube: np.ndarray, correction_coefficients_dict: dict) -> np.ndarray:
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
        print("Radioshape: ",
              correction_coefficients_dict["radiometric"].shape)

    for i in range(num_frames):
        frame = raw_cube[i, :, :]
        # Radiometric Calibration
        frame_calibrated = apply_radiometric_calibration(
            frame, exp, background_value, correction_coefficients_dict["radiometric"])
        # Smile Correction

        # Destriping Correction

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
