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

    affin, param_x, param_y = abb.find_affine_by_block(
        fixed_image,
        moving_image,
        block_size=156,
        overlap=0.8,
        correlation_threshold=0.4,
    )
    # a_x, a_y, b_x, and b_y should be very small
    assert np.all(np.abs(param_x[:2]) < 0.01)
    assert np.all(np.abs(param_y[:2]) < 0.01)
    # c_x and c_y should be close to the true shifts
    assert np.allclose(param_x[2], true_shifts[0], atol=1)
    assert np.allclose(param_y[2], true_shifts[1], atol=1)
    # Test the affine transformation
    point = np.array([10, 10])
    assert np.allclose(affin(point)[0], point + true_shifts, atol=1)


if __name__ == "__main__":
    test_affine_by_block()
    test_phase_corr_by_block()
    print("Everything passed")
