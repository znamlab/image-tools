from math import cos, radians, sin

import numpy as np
import numpy.typing as npt
from skimage.transform import SimilarityTransform, warp


def transform_image(
    im: npt.NDArray,
    scale: float = 1,
    angle: float = 0,
    shift: tuple = (0, 0),
    cval: float = 0.0,
):
    """
    Transform an image using provided scale, rotation angle, and shift.

    Args:
        im (npt.NDArray): image to transform
        scale (float): scale factor
        angle (float): rotation angle in degrees
        shift (tuple): shift in x and y
        cval (float): value to fill in for pixels outside of the image

    Returns:
        npt.NDArray: transformed image

    """
    tform = SimilarityTransform(matrix=make_transform(scale, angle, shift, im.shape))
    return warp(im, tform.inverse, preserve_range=True, cval=cval)


def make_transform(s: float, angle: float, shift: tuple, shape: tuple):
    """
    Make a transformation matrix using provided scale, rotation angle, and shift.

    Args:
        s (float): scale factor
        angle (float): rotation angle in degrees
        shift (tuple): shift in x and y
        shape (tuple): shape of the image

    Returns:
        numpy.ndarray: transformation matrix

    """
    angle = -radians(angle)
    center_x = float(shape[1]) / 2 - 0.5
    center_y = float(shape[0]) / 2 - 0.5
    shift_x = shift[1]
    shift_y = shift[0]
    tform = [
        [
            cos(angle) * s,
            -sin(angle) * s,
            shift_x + (center_x - s * (center_x * cos(angle) - center_y * sin(angle))),
        ],
        [
            sin(angle) * s,
            cos(angle) * s,
            shift_y + (center_y - s * (center_x * sin(angle) + center_y * cos(angle))),
        ],
        [0, 0, 1],
    ]
    return np.array(tform)
