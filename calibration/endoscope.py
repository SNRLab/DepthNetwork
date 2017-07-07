import os

import cv2
import numpy as np

_crop_diameter = 504

_calibration_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'camera_parameters.xml')

_calibration = cv2.FileStorage(_calibration_file, flags=cv2.FILE_STORAGE_READ)
if not _calibration.isOpened():
    raise IOError('Could not open calibration file.')
_camera_matrix = _calibration.getNode('cameraMatrix').mat()
_dist_coeffs = _calibration.getNode('dist_coeffs').mat()
_mask = np.zeros((_crop_diameter, _crop_diameter, 1), np.uint8)
cv2.circle(_mask, (252, 252), 252, 1, thickness=-1)


def camera_matrix():
    return _camera_matrix


def dist_coeffs():
    return _dist_coeffs


def preprocess_endoscope_image(image):
    image = np.pad(image, ((0, 60), (0, 0), (0, 0)), mode='edge')
    image = cv2.undistort(image, _camera_matrix, _dist_coeffs)
    image = image[10:10 + _crop_diameter, 60:60 + _crop_diameter]
    # cv2.circle(image, (252, 252), 252, 0, -1)
    image = cv2.bitwise_and(image, image, mask=_mask)
    return image
