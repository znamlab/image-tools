"""Masked phase correlation

As described in:
D. Padfield. Masked object registration in the Fourier domain.
IEEE Transactions on Image Processing, 21(5):2706–2718, 2012.
https://ieeexplore.ieee.org/document/6111478

Translated from matlab code by D. Padfield using copilot.
"""


import numpy as np
from scipy.fftpack import fft2, ifft2


def normxcorr2_masked(fixed_image, moving_image, fixed_mask, moving_mask):
    fixed_mask = np.where(fixed_mask <= 0, 0, 1)
    moving_mask = np.where(moving_mask <= 0, 0, 1)

    fixed_image = np.where(fixed_mask == 0, 0, fixed_image)
    moving_image = np.where(moving_mask == 0, 0, moving_image)

    rotated_moving_image = np.rot90(moving_image, 2)
    rotated_moving_mask = np.rot90(moving_mask, 2)

    fixed_image_size = fixed_image.shape
    moving_image_size = rotated_moving_image.shape
    combined_size = np.add(fixed_image_size, moving_image_size) - 1

    optimal_size = [
        find_closest_valid_dimension(combined_size[0]),
        find_closest_valid_dimension(combined_size[1]),
    ]

    fixed_fft = fft2(fixed_image, optimal_size)
    rotated_moving_fft = fft2(rotated_moving_image, optimal_size)
    fixed_mask_fft = fft2(fixed_mask, optimal_size)
    rotated_moving_mask_fft = fft2(rotated_moving_mask, optimal_size)

    number_of_overlap_masked_pixels = np.real(
        ifft2(rotated_moving_mask_fft * fixed_mask_fft)
    )
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

    fixed_squared_fft = fft2(fixed_image * fixed_image, optimal_size)
    fixed_denom = (
        np.real(ifft2(rotated_moving_mask_fft * fixed_squared_fft))
        - np.power(mask_correlated_fixed_fft, 2) / number_of_overlap_masked_pixels
    )
    fixed_denom = np.maximum(fixed_denom, 0)

    rotated_moving_squared_fft = fft2(
        rotated_moving_image * rotated_moving_image, optimal_size
    )
    moving_denom = (
        np.real(ifft2(fixed_mask_fft * rotated_moving_squared_fft))
        - np.power(mask_correlated_rotated_moving_fft, 2)
        / number_of_overlap_masked_pixels
    )
    moving_denom = np.maximum(moving_denom, 0)

    denom = np.sqrt(fixed_denom * moving_denom)

    C = np.zeros(numerator.shape)
    tol = 1000 * np.finfo(float).eps * np.max(np.abs(denom))
    i_nonzero = np.where(denom > tol)
    C[i_nonzero] = numerator[i_nonzero] / denom[i_nonzero]
    C = np.where(C < -1, -1, C)
    C = np.where(C > 1, 1, C)

    C = C[0 : combined_size[0], 0 : combined_size[1]]
    number_of_overlap_masked_pixels = number_of_overlap_masked_pixels[
        0 : combined_size[0], 0 : combined_size[1]
    ]

    return C, number_of_overlap_masked_pixels


def find_closest_valid_dimension(n):
    new_number = n
    result = 0
    new_number -= 1
    while result != 1:
        new_number += 1
        result = factorize_number(new_number)
    return new_number


def factorize_number(n):
    for ifac in [2, 3, 5]:
        while n % ifac == 0:
            n /= ifac
    return n
