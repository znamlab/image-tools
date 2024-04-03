from functools import partial

import numpy as np
from skimage import data

from image_tools.registration import affine_by_block as abb


def test_phase_corr_by_block():
    true_shifts = (10, 15)

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


def test_affine_by_block():
    true_shifts = (10, 15)
    # Make test data
    cat = data.cat()
    fixed_image = cat[:256, :256, 0]
    moving_image = cat[
        true_shifts[0] : 256 + true_shifts[0], true_shifts[1] : 256 + true_shifts[1], 0
    ]

    params = abb.find_affine_by_block(
        fixed_image,
        moving_image,
        block_size=156,
        overlap=0.8,
        correlation_threshold=0.4,
    )
    true_params = np.array([1, 0, true_shifts[0], 0, 1, true_shifts[1]])
    assert np.all(np.abs(params - true_params) < 0.1)

    # Test the affine transformation
    affin = partial(abb.affine_transform, params=params)
    inv_map = partial(abb.inverse_map, params=params)
    point = np.array([10, 10])

    assert np.allclose(
        abb.affine_transform(point, params)[0], point + true_shifts, atol=1
    )
    # it should also work with a list
    assert np.allclose(affin(point.tolist())[0], point + true_shifts, atol=1)
    # and multiple points
    points = np.array([[0, 0], [20, 30]])
    assert np.allclose(affin(points), points + true_shifts, atol=1)

    # and the inverse
    assert np.allclose(inv_map(point)[0], point - true_shifts, atol=1)
    # it should also work with a list
    assert np.allclose(inv_map(point.tolist())[0], point - true_shifts, atol=1)
    # and multiple points
    assert np.allclose(inv_map(points), points - true_shifts, atol=1)

    # inverse should invert the affine
    assert np.allclose(inv_map(affin(point)), point, atol=1)
    assert np.allclose(affin(inv_map(point)), point, atol=1)


if __name__ == "__main__":
    test_affine_by_block()
    test_phase_corr_by_block()
    print("Everything passed")
