import numpy as np
from skimage import data

from image_tools.registration import affine_by_block as abb


def test_phase_corr_by_block():
    true_shifts = (10, 15)
    
    # Make test data
    cat = data.cat()
    fixed_image = cat[:256, :256, 0]
    moving_image = cat[true_shifts[0]:256 + true_shifts[0], true_shifts[1]:256 + true_shifts[1], 0]
    shifts, corrs, centers = abb.phase_correlation_by_block(fixed_image, moving_image, block_size=156, overlap=0.8)
    assert np.mean(corrs) > 0.5
    assert centers.shape == shifts.shape
    shift = np.nanmedian(shifts, axis=(0, 1))
    assert np.allclose(shift, true_shifts, atol=1)
