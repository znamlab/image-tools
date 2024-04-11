"""Masked phase correlation

As described in:
D. Padfield. Masked object registration in the Fourier domain.
IEEE Transactions on Image Processing, 21(5):2706–2718, 2012.
https://ieeexplore.ieee.org/document/6111478

Translated from matlab code by D. Padfield using copilot.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.fft import fft2, fftshift, ifft2  # type: ignore


def phase_correlation(
    fixed_image: npt.NDArray,
    moving_image: npt.NDArray,
    fixed_mask: Optional[npt.NDArray] = None,
    moving_mask: Optional[npt.NDArray] = None,
    overlap_ratio: float = 0.3,
    max_shift: Optional[int] = None,
    min_shift: Optional[int] = None,
    whiten: bool = True,
    fixed_image_is_fft: bool = False,
    fixed_mask_is_fft: bool = False,
    fixed_squared_fft: Optional[npt.NDArray] = None,
) -> tuple[tuple, float, npt.NDArray, npt.NDArray]:
    """Perform masked translation registration.

    Based on the masked phase correlation method described in:
    D. Padfield. Masked object registration in the Fourier domain.
    IEEE Transactions on Image Processing, 21(5):2706–2718, 2012.
    https://ieeexplore.ieee.org/document/6111478

    Args:
        fixed_image (npt.NDArray): The fixed image.
        moving_image (npt.NDArray): The moving image.
        fixed_mask (npt.NDArray, optional): The fixed mask. Defaults to None.
        moving_mask (npt.NDArray, optional): The moving mask. Defaults to None.
        overlap_ratio (float, optional): The overlap ratio. Defaults to 0.3.
        max_shift (int, optional): Maximum allowed shift. Defaults to None.
        min_shift (int, optional): Minimum allowed shift. Defaults to None.
        whiten (bool, optional): Whether to whiten the FFTs of the images. If True, the
            method performs phase correlation, otherwise cross correlation is performed.
            Used only for non-masked phase correlation. Defaults to True.
        fixed_image_is_fft (bool, optional): Whether the fixed image is already in the
            Fourier domain. Defaults to False.
        fixed_mask_is_fft (bool, optional): Whether the fixed mask is already in the
            Fourier domain. Defaults to False.
        fixed_image_squared_fft (npt.NDArray, optional): The squared Fourier transform
            of the fixed image. Defaults to None.

    Returns:
        tuple: The translation corresponding to the maximum correlation.
        float: The maximum correlation
        npt.NDArray: the full cross-correlation
        npt.NDArray: the number of overlap pixels at each shift.
    """
    float_dtype = moving_image.dtype
    if not np.issubdtype(float_dtype, np.floating):
        # we probably have a boolean array. Use float32 for the FFTs.
        float_dtype = np.float32

    if fixed_mask is None:
        # perform non-masked phase correlation
        xcorr = _simple_phase_corr(
            fixed_image,
            moving_image,
            whiten,
            not fixed_image_is_fft,
            float_dtype,
        )
        number_of_overlap_masked_pixels = np.ones_like(xcorr, dtype=int) * xcorr.size
    else:
        assert moving_mask is not None, "moving_mask must be provided"
        xcorr, number_of_overlap_masked_pixels = _normxcorr2_masked(
            fixed_image,
            moving_image,
            fixed_mask,
            moving_mask,
            fixed_image_is_fft,
            fixed_mask_is_fft,
            fixed_squared_fft,
            float_dtype,
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


def _normxcorr2_masked(
    fixed_image: npt.NDArray,
    moving_image: npt.NDArray,
    fixed_mask: npt.NDArray,
    moving_mask: npt.NDArray,
    fixed_image_is_fft: bool = False,
    fixed_mask_is_fft: bool = False,
    fixed_squared_fft: Optional[npt.NDArray] = None,
    float_dtype: Optional[np.dtype] = np.float32,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Perform masked normalized cross-correlation.

    This inner function calculates the masked normalized cross-correlation between two
    images.

    Args:
        fixed_image (np.array): The fixed image.
        moving_image (np.array): The moving image.
        fixed_mask (np.array): The fixed mask.
        moving_mask (np.array): The moving mask.
        fixed_image_is_fft (bool): Whether the fixed image is already in the Fourier
            domain.
        fixed_mask_is_fft (bool): Whether the fixed mask is already in the Fourier
            domain.
        fixed_squared_fft (np.array): The squared Fourier transform of the fixed image.
        float_dtype (np.dtype, optional): The dtype to use for the FFTs. Defaults to
            np.float32.

    Returns:
        np.array: The cross-correlation.
        np.array: The number of overlap masked pixels at each shift.
    """

    if fixed_mask_is_fft:
        assert (
            fixed_image_is_fft
        ), "If fixed_mask_is_fft is True, fixed_image_is_fft must also be True"
        fixed_mask_fft = fixed_mask
    else:
        fixed_mask = fixed_mask.astype(float_dtype)
        fixed_mask = np.where(fixed_mask <= 0, 0, 1)
        fixed_mask_fft = fft2(fixed_mask)

    if fixed_image_is_fft:
        assert (
            fixed_squared_fft is not None
        ), "If fixed_image_is_fft is True, fixed_image_squared_fft must be provided"
        fixed_fft = fixed_image
    else:
        fixed_image = fixed_image.astype(float_dtype)
        fixed_image = np.where(fixed_mask == 0, 0, fixed_image)
        fixed_fft = fft2(fixed_image)

    if fixed_squared_fft is None:
        fixed_squared_fft = fft2(fixed_image * fixed_image)

    rotated_moving_image = np.rot90(moving_image, 2)
    rotated_moving_mask = np.rot90(moving_mask, 2)
    (
        rotated_moving_fft,
        rotated_moving_squared_fft,
        rotated_moving_mask_fft,
    ) = get_mask_and_ffts(
        image=rotated_moving_image,
        mask=rotated_moving_mask,
        float_dtype=float_dtype,
    )

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

    fixed_denom = (
        np.real(ifft2(rotated_moving_mask_fft * fixed_squared_fft))
        - np.power(mask_correlated_fixed_fft, 2) / number_of_overlap_masked_pixels
    )
    fixed_denom = np.maximum(fixed_denom, 0)

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

    return xcorr, number_of_overlap_masked_pixels


def _simple_phase_corr(
    reference: npt.NDArray,
    target: npt.NDArray,
    whiten: Optional[int] = True,
    fft_ref: Optional[int] = True,
    float_dtype: Optional[np.dtype] = np.float32,
) -> npt.NDArray:
    """
    Compute phase correlation of two images.

    Args:
        reference (numpy.ndarray): reference image
        target (numpy.ndarray): target image
        max_shift (int, optional): the range over which to search for the maximum of the
            cross-correlogram. Defaults to None.
        min_shift (int, optional): the range over which to search for the minimum of the
            cross-correlogram. Defaults to None.
        whiten (bool, optional): whether or not to whiten the FFTs of the images.
            If True, the method performs phase correlation, otherwise cross correlation
            is performed. Defaults to True.
        fft_ref (bool, optional): whether to compute the FFT transform of the reference
            image. Defaults to True.
        float_dtype (np.dtype, optional): The dtype to use for the FFTs. Defaults to
            np.float32.

    Returns:
        xcorr: numpy.ndarray of the cross-correlagram itself.

    """
    if fft_ref:
        f1 = fft2(reference.astype(float_dtype))
    else:
        assert np.iscomplexobj(reference), (
            "`reference` must be complex if fft_ref is False",
        )
        f1 = reference
    f2 = fft2(target.astype(float_dtype))
    if whiten:
        # add the smallest number to avoid division by zero
        precision = np.finfo(float_dtype).eps
        f1 = f1 / (np.abs(f1) + precision)
        f2 = f2 / (np.abs(f2) + precision)
    xcorr = np.abs(ifft2(f1 * np.conj(f2)))

    return xcorr


def get_mask_and_ffts(image, mask=None, float_dtype=None):
    """Create a mask from image and return FFTs required for masked_phase_correlation.

    This function creates a mask from the image and returns the FFTs required for
    masked_phase_correlation. The mask is created by thresholding the image at 0.


    Args:
        image (np.array): The image to create the mask from.
    """
    if float_dtype is None:
        float_dtype = np.float32

    if mask is None:
        mask = image.astype(float_dtype)
        mask = np.where(mask <= 0, 0, 1)
    else:
        mask = mask.astype(float_dtype)
    mask_fft = fft2(mask)

    fft = fft2(image)
    image = image.astype(float_dtype)
    image = np.where(mask == 0, 0, image)
    fft = fft2(image)
    fft_img_squared = fft2(image * image)

    return fft, fft_img_squared, mask_fft
