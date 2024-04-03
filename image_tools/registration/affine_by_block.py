from typing import Optional

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import HuberRegressor

from . import phase_correlation as _pc


def find_affine_by_block(
    reference: npt.NDArray,
    target: npt.NDArray,
    block_size: int = 256,
    overlap: float = 0.5,
    correlation_threshold: Optional[float] = None,
):
    """Find affine transformation between two images by dividing them into blocks and
    estimating translation for each block.

    We fit:
    x_coords = a_x * x + b_x * y + c_x
    y_coords = a_y * x + b_y * y + c_y

    Args:
        reference (np.array): reference image
        target (np.array): target image
        block_size (int, optional): size of the blocks, defaults to 256
        overlap (float, optional): fraction of overlap between blocks, defaults to 0.5
        correlation_threshold (float, optional): minimum correlation threshold, defaults
            to None

    Returns:
        affine_transform (function): function that gives the affine transformation for
            any point
        inverse_map (function): function that gives the inverse affine transformation
            for any point
        x_params (np.array): parameters (a_x, b_x, c_x) of the affine transformation in
            x direction
        y_params (np.array): parameters (a_y, b_y, c_y) of the affine transformation in
            y direction
    """

    # first perform phase correlation by block
    shifts, corr, centers = phase_correlation_by_block(
        reference, target, block_size=block_size, overlap=overlap
    )
    # then fit affine transformation to the shifts
    shifts = shifts.reshape(-1, 2)
    hom_centers = centers.reshape(-1, 2)

    if correlation_threshold is not None:
        corr = corr.reshape(-1)
        shifts = shifts[corr > correlation_threshold]
        hom_centers = hom_centers[corr > correlation_threshold]

    # minor annoyance, if shifts are all exactly the same, the HuberRegressor will
    # sometimes fail to converge, in that case we add a small amount of noise
    if np.sum(shifts - shifts[0]) < 1:
        shifts += np.random.normal(0, 0.01, shifts.shape)

    huber_x = HuberRegressor(fit_intercept=True).fit(
        hom_centers, shifts[:, 0] + hom_centers[:, 0]
    )
    huber_y = HuberRegressor(fit_intercept=True).fit(
        hom_centers, shifts[:, 1] + hom_centers[:, 1]
    )

    # finally make a function that gives the affine transformation for any point
    def affine_transform(point: npt.NDArray):
        """Affine transformation for a point or array of points.

        Transforms coordinates of point in the reference image to the coordinates in the
        target image.

        Args:
            point (np.array): point or array of points

        Returns:
            new_point (np.array): transformed point or array of points
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        if point.ndim == 1:
            point = point.reshape(1, -1)
        new_point = np.hstack(
            [
                huber_x.predict(point).reshape(-1, 1),
                huber_y.predict(point).reshape(-1, 1),
            ]
        )
        return new_point

    def inverse_map(point: npt.NDArray):
        """Inverse affine transformation for a point or array of points.

        Transforms coordinates of point in the target image to the coordinates in the
        reference image.

        Args:
            point (np.array): point or array of points

        Returns:
            new_point (np.array): transformed point or array of points
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        if point.ndim == 1:
            point = point.reshape(1, -1)
        # inverse the affine transformation
        """
        We fit:
        x_coords = a_x * x + b_x * y + c_x
        y_coords = a_y * x + b_y * y + c_y

        The inverse is:
        x = (x_coords - c_x)/a_x - b_x /a_x * y
        y = (y_coords - c_y)/b_y - a_y / b_y * x

        with Ax = (x_coords - c_x)/a_x and Bx = - b_x /a_x
        and Ay = (y_coords - c_y)/b_y and By = - a_y / b_y
        x = Ax + Bx . y
        y = Ay + By . x

        so we can solve for x:
        x = Ax + Bx(Ay + By * x)
        x = (Ax + Bx . Ay) / (1 - Bx.By)
        and for y:
        y = Ay + By(Ax + Bx * y)
        y = (Ay + By . Ax) / (1 - Bx.By)

        """
        Ax = (point[:, 0] - huber_x.intercept_) / huber_x.coef_[0]
        Bx = -huber_x.coef_[1] / huber_x.coef_[0]
        Ay = (point[:, 1] - huber_y.intercept_) / huber_y.coef_[1]
        By = -huber_y.coef_[0] / huber_y.coef_[1]
        x = (Ax + Bx * Ay) / (1 - Bx * By)
        y = (Ay + By * Ax) / (1 - Bx * By)

        return np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])

    x_params = np.hstack([huber_x.coef_, huber_x.intercept_])
    y_params = np.hstack([huber_y.coef_, huber_y.intercept_])
    return affine_transform, inverse_map, x_params, y_params


def phase_correlation_by_block(
    reference: npt.NDArray,
    target: npt.NDArray,
    block_size: int = 256,
    overlap: float = 0.1,
):
    """Estimate translation between two images by dividing them into blocks and
    estimating translation for each block.

    Args:
        reference (np.array): reference image
        target (np.array): target image
        block_size (int, optional): size of the blocks, defaults to 256
        overlap (float, optional): fraction of overlap between blocks, defaults to 0.1

    Returns:
        shifts (np.array): array of shifts for each block
        corrs (np.array): array of correlation coefficients for each block
        block_centers (np.array): array of centers of each block

    """
    step_size = int(block_size * (1 - overlap))
    nblocks = np.array(reference.shape) // step_size
    block_centers = np.zeros((*nblocks, 2))
    shifts = np.zeros((*nblocks, 2))
    corrs = np.zeros(nblocks)
    for row in range(nblocks[0]):
        for col in range(nblocks[1]):
            ref_block = reference[
                row * step_size : row * step_size + block_size,
                col * step_size : col * step_size + block_size,
            ]
            target_block = target[
                row * step_size : row * step_size + block_size,
                col * step_size : col * step_size + block_size,
            ]
            shift, corr = _pc.phase_correlation(
                ref_block,
                target_block,
            )[:2]
            shifts[row, col] = shift
            corrs[row, col] = corr
            block_centers[row, col] = np.array(
                [row * step_size + block_size // 2, col * step_size + block_size // 2]
            )
    return shifts, corrs, block_centers
