from functools import partial

import numpy as np
from skimage import data

from image_tools.registration import affine_by_block as abb


def test_phase_corr_by_block():
    true_shifts = (10, 20)

    # Make test data
    cat = data.cat()
    fixed_image = cat[:256, :256, 0]
    moving_image = cat[
        true_shifts[0] : 256 + true_shifts[0], true_shifts[1] : 256 + true_shifts[1], 0
    ]

    shifts, corrs, centers = abb.phase_correlation_by_block(
        fixed_image, moving_image, block_size=156, overlap=0.8
    )
    assert np.mean(corrs) > 0.5
    assert corrs.shape == shifts.shape[:2]
    assert centers.shape == shifts.shape
    shift = np.nanmedian(shifts, axis=(0, 1))
    assert np.allclose(shift, true_shifts, atol=1)
    shifts, corrs, centers = abb.phase_correlation_by_block(
        fixed_image, moving_image, block_size=156, overlap=0.8, max_shift=11
    )
    assert np.all(np.abs(shifts) < 11)
    shifts, corrs, centers = abb.phase_correlation_by_block(
        fixed_image, moving_image, block_size=156, overlap=0.8, min_shift=5
    )
    assert np.all(np.abs(shifts) >= 5)


def test_affine_by_block():
    true_shifts = (10, 15)  # row columns
    # Make test data
    cat = data.cat()
    fixed_image = cat[:256, :256, 0]
    moving_image = cat[
        true_shifts[0] : 256 + true_shifts[0], true_shifts[1] : 256 + true_shifts[1], 0
    ]

    # it works with no threshold
    params = abb.find_affine_by_block(
        fixed_image,
        moving_image,
        block_size=156,
        overlap=0.8,
        correlation_threshold=None,
    )
    true_params = np.array([1, 0, true_shifts[1], 0, 1, true_shifts[0]])
    assert np.all(np.abs(params - true_params) < 0.5)

    # but is better with
    params = abb.find_affine_by_block(
        fixed_image,
        moving_image,
        block_size=156,
        overlap=0.8,
        correlation_threshold=0.4,
    )
    true_params = np.array([1, 0, true_shifts[1], 0, 1, true_shifts[0]])
    assert np.all(np.abs(params - true_params) < 0.1)

    # Test the affine transformation
    affin = partial(abb.affine_transform, params=params)
    inv_map = partial(abb.inverse_map, params=params)
    point = np.array([10, 10])

    ts_xy = np.array([true_shifts[1], true_shifts[0]])
    assert np.allclose(abb.affine_transform(point, params)[0], point + ts_xy, atol=1)
    # it should also work with a list
    assert np.allclose(affin(point.tolist())[0], point + ts_xy, atol=1)
    # and multiple points
    points = np.array([[0, 0], [20, 30]])
    assert np.allclose(affin(points), points + ts_xy, atol=1)

    # and the inverse
    assert np.allclose(inv_map(point)[0], point - ts_xy, atol=1)
    # it should also work with a list
    assert np.allclose(inv_map(point.tolist())[0], point - ts_xy, atol=1)
    # and multiple points
    assert np.allclose(inv_map(points), points - ts_xy, atol=1)

    # inverse should invert the affine
    assert np.allclose(inv_map(affin(point)), point, atol=1)
    assert np.allclose(affin(inv_map(point)), point, atol=1)

    # we can contrain min/max shifts. (the fit output can still be outside the limits)
    params = abb.find_affine_by_block(
        fixed_image,
        moving_image,
        block_size=156,
        overlap=0.8,
        max_shift=11,
        correlation_threshold=0.1,
    )
    assert np.all(np.abs(params[[2, 5]]) <= 11)
    assert np.all(np.abs(params - true_params) < 0.1)

    # check debug mode can run
    params, db = abb.find_affine_by_block(
        fixed_image,
        moving_image,
        block_size=156,
        overlap=0.8,
        correlation_threshold=0.4,
        debug=True,
    )
    assert "huber_x" in db


def test_transform_image(do_plot=False):
    true_shifts = (10, 50)
    np.array([true_shifts[1], true_shifts[0]])
    # Make test data
    cat = data.cat()
    fixed_image = cat[:256, :256, 0]
    moving_image = cat[
        true_shifts[0] : 256 + true_shifts[0], true_shifts[1] : 256 + true_shifts[1], 0
    ]
    true_params = np.array([1, 0, true_shifts[1], 0, 1, true_shifts[0]])
    transformed = abb.transform_image(moving_image, true_params)

    # the overlap part should be nicely registered
    overlap = (
        fixed_image[true_shifts[0] :, true_shifts[1] :]
        - transformed[true_shifts[0] :, true_shifts[1] :]
    )
    assert np.sum(overlap) < 1
    if do_plot:
        import matplotlib.pyplot as plt

        vmin = fixed_image.min()
        vmax = fixed_image.max()
        plt.subplot(2, 2, 1)
        plt.imshow(fixed_image, cmap="Greens_r", vmin=vmin, vmax=vmax)
        plt.subplot(2, 2, 2)
        plt.imshow(moving_image, cmap="Reds_r", vmin=vmin, vmax=vmax)
        plt.subplot(2, 2, 3)
        plt.imshow(fixed_image, cmap="Greens_r", vmin=vmin, vmax=vmax)
        plt.imshow(moving_image, cmap="Reds_r", alpha=0.5, vmin=vmin, vmax=vmax)
        plt.subplot(2, 2, 4)
        plt.imshow(fixed_image, cmap="Greens_r", alpha=0.5, vmin=vmin, vmax=vmax)
        plt.imshow(transformed, cmap="Reds_r", alpha=0.5, vmin=vmin, vmax=vmax)

        plt.show()


if __name__ == "__main__":
    test_phase_corr_by_block()
    test_affine_by_block()
    test_transform_image()

    print("Everything passed")
