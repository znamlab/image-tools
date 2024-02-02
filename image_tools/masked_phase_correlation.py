"""Masked phase correlation

As described in:
D. Padfield. Masked object registration in the Fourier domain.
IEEE Transactions on Image Processing, 21(5):2706–2718, 2012.
https://ieeexplore.ieee.org/document/6111478

Translated from matlab code by D. Padfield using copilot.
"""

from typing import Optional

import numpy as np
from scipy.fft import fft2, fftshift, ifft2


def masked_translation_registration(
    fixed_image: np.array,
    moving_image: np.array,
    fixed_mask: np.array,
    moving_mask: np.array,
    overlap_ratio: float = 0.3,
    max_shift: Optional[int] = None,
    min_shift: Optional[int] = None,
):
    """Perform masked translation registration.

    Based on the masked phase correlation method described in:
    D. Padfield. Masked object registration in the Fourier domain.
    IEEE Transactions on Image Processing, 21(5):2706–2718, 2012.
    https://ieeexplore.ieee.org/document/6111478

    Args:
        fixed_image (np.array): The fixed image.
        moving_image (np.array): The moving image.
        fixed_mask (np.array): The fixed mask.
        moving_mask (np.array): The moving mask.
        overlap_ratio (float): The overlap ratio.
        max_shift (int): Maximum allowed shift.
        min_shift (int): Minimum allowed shift.

    Returns:
        tuple: The translation corresponding to the maximum correlation.
        float: The maximum correlation
        np.array: the cross-correlation
        np.array: the number of overlap masked pixels at each shift.
    """
    xcorr, number_of_overlap_masked_pixels = normxcorr2_masked(
        fixed_image, moving_image, fixed_mask, moving_mask
    )

    image_size = moving_image.shape

    # Mask the borders
    number_of_pixels_threshold = overlap_ratio * np.max(number_of_overlap_masked_pixels)
    xcorr[number_of_overlap_masked_pixels < number_of_pixels_threshold] = 0

    if max_shift:
        xcorr[max_shift:-max_shift, :] = 0
        xcorr[:, max_shift:-max_shift] = 0
    if min_shift:
        xcorr[:min_shift, :min_shift] = 0
        xcorr[-min_shift:, -min_shift:] = 0
        xcorr[-min_shift:, :min_shift] = 0
        xcorr[:min_shift, -min_shift:] = 0

    max_xcorr = np.max(xcorr)
    xcorr = fftshift(xcorr)
    number_of_overlap_masked_pixels = fftshift(number_of_overlap_masked_pixels)

    shift = np.unravel_index(np.argmax(xcorr), image_size) - np.array(image_size) / 2

    return shift, max_xcorr, xcorr, number_of_overlap_masked_pixels


def normxcorr2_masked(fixed_image, moving_image, fixed_mask, moving_mask):
    """Perform masked normalized cross-correlation.

    This inner function calculates the masked normalized cross-correlation between two
    images.

    Args:
        fixed_image (np.array): The fixed image.
        moving_image (np.array): The moving image.
        fixed_mask (np.array): The fixed mask.
        moving_mask (np.array): The moving mask.

    Returns:
        np.array: The cross-correlation.
        np.array: The number of overlap masked pixels at each shift.
    """
    float_dtype = moving_image.dtype
    if not np.issubdtype(float_dtype, np.floating):
        # we probably have a boolean array. Use float32 for the FFTs.
        float_dtype = np.float32

    fixed_image = fixed_image.astype(float_dtype)
    moving_image = moving_image.astype(float_dtype)
    fixed_mask = fixed_mask.astype(float_dtype)
    moving_mask = moving_mask.astype(float_dtype)

    fixed_mask = np.where(fixed_mask <= 0, 0, 1)
    moving_mask = np.where(moving_mask <= 0, 0, 1)

    fixed_image = np.where(fixed_mask == 0, 0, fixed_image)
    moving_image = np.where(moving_mask == 0, 0, moving_image)

    rotated_moving_image = np.rot90(moving_image, 2)
    rotated_moving_mask = np.rot90(moving_mask, 2)

    fixed_image_size = fixed_image.shape
    moving_image_size = rotated_moving_image.shape
    combined_size = np.add(fixed_image_size, moving_image_size) - 1
    fixed_fft = fft2(fixed_image)
    rotated_moving_fft = fft2(rotated_moving_image)
    fixed_mask_fft = fft2(fixed_mask)
    rotated_moving_mask_fft = fft2(rotated_moving_mask)

    number_of_overlap_masked_pixels = ifft2(
        rotated_moving_mask_fft * fixed_mask_fft
    ).real

    number_of_overlap_masked_pixels = np.round(number_of_overlap_masked_pixels)
    number_of_overlap_masked_pixels = np.maximum(
        number_of_overlap_masked_pixels, np.finfo(float).eps
    )

    mask_correlated_fixed_fft = np.real(ifft2(rotated_moving_mask_fft * fixed_fft))
    mask_correlated_rotated_moving_fft = np.real(
        ifft2(fixed_mask_fft * rotated_moving_fft)
    )

    numerator = (
        np.real(ifft2(rotated_moving_fft * fixed_fft))
        - mask_correlated_fixed_fft
        * mask_correlated_rotated_moving_fft
        / number_of_overlap_masked_pixels
    )

    fixed_squared_fft = fft2(fixed_image * fixed_image)
    fixed_denom = (
        np.real(ifft2(rotated_moving_mask_fft * fixed_squared_fft))
        - np.power(mask_correlated_fixed_fft, 2) / number_of_overlap_masked_pixels
    )
    fixed_denom = np.maximum(fixed_denom, 0)

    rotated_moving_squared_fft = fft2(rotated_moving_image * rotated_moving_image)
    moving_denom = (
        np.real(ifft2(fixed_mask_fft * rotated_moving_squared_fft))
        - np.power(mask_correlated_rotated_moving_fft, 2)
        / number_of_overlap_masked_pixels
    )
    moving_denom = np.maximum(moving_denom, 0)

    denom = np.sqrt(fixed_denom * moving_denom)

    xcorr = np.zeros(numerator.shape)
    tol = 1000 * np.finfo(float).eps * np.max(np.abs(denom))
    i_nonzero = np.where(denom > tol)
    xcorr[i_nonzero] = numerator[i_nonzero] / denom[i_nonzero]
    xcorr = np.where(xcorr < -1, -1, xcorr)
    xcorr = np.where(xcorr > 1, 1, xcorr)

    xcorr = xcorr[0 : combined_size[0], 0 : combined_size[1]]
    number_of_overlap_masked_pixels = number_of_overlap_masked_pixels[
        0 : combined_size[0], 0 : combined_size[1]
    ]

    return xcorr, number_of_overlap_masked_pixels
