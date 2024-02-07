from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from image_tools.registration import phase_correlation as mpc
from image_tools.similarity_transforms import transform_image


def test_phase_correlation():
    # test the simple, non-masked phase correlation
    # create a simple image and a shifted version of it
    big_image = np.zeros((200, 200), dtype="uint8")
    square = [[100, 80], [50, 50], [100, 150], [120, 120]]
    for row, col in square:
        big_image[row : row + 10, col : col + 10] = 255
    # add gaussian noise to the image
    big_image = big_image + np.abs(np.random.normal(0, 2, big_image.shape)).astype(
        "uint8"
    )
    image = big_image[50:, 20:]
    shifted_image = big_image[:-50, :-20]

    # calculate the phase correlation
    translation, max_corr, xcorr, novlp = mpc.phase_correlation(image, shifted_image)
    # check the result
    assert np.allclose(translation, [-50, -20])
    assert max_corr > 0.4
    assert np.all(novlp == image.size)
    assert np.isnan(xcorr).any() is False


def test_masked_phase_correlation(do_plot=False):
    # Perform the registration on several sets of images.
    x = [75, -130, 130]
    y = [75, 130, 130]
    overlap_ratio = 0.1
    test_dir = Path(__file__).resolve().parent.parent
    for i in range(3):
        fixed_image = imageio.imread(
            test_dir / f"test_data/OriginalX{x[i]:02}Y{y[i]:02}.png"
        )
        moving_image = imageio.imread(
            test_dir / f"test_data/TransformedX{x[i]:02}Y{y[i]:02}.png"
        )
        fixed_mask = fixed_image != 0
        moving_mask = moving_image != 0

        (
            translation,
            max_corr,
            xcorr,
            novlp,
        ) = mpc.phase_correlation(
            fixed_image, moving_image, fixed_mask, moving_mask, overlap_ratio
        )
        transformed_moving_image = transform_image(moving_image, shift=translation)

        # Calculate the difference between the two images in the overlap region.
        overlap = fixed_mask & moving_mask
        np.mean(
            np.abs(
                fixed_image[overlap].astype(float) - moving_image[overlap].astype(float)
            )
        )
        overlap = (transformed_moving_image != 0) & fixed_mask
        difference = np.mean(
            np.abs(
                transformed_moving_image[overlap].astype(float)
                - fixed_image[overlap].astype(float)
            )
        )
        assert difference < 20, f"Test {i+1}: Difference between images is too large."

        assert max_corr > 0.9, f"Test {i+1}: Correlation score is too low."
        true_translation = np.array([y[i], -x[i]])
        translation_error = np.abs(np.array(translation) - true_translation)
        assert np.sum(translation_error) < 4, "Translation error is too large."

        if do_plot:
            import matplotlib.pyplot as plt

            # Given the transform, transform the moving image.
            overlay_image = overlay_registration(fixed_image, transformed_moving_image)
            plt.figure()
            plt.imshow(overlay_image)
            plt.title(f"Test {i+1}: Registered Overlay Image")
            plt.savefig(test_dir / f"test_data/RegisteredOverlayImage{i+1}.png")

            print(f"Test {i+1}:")
            print(f"Computed translation: {translation[0]} {-translation[1]}")
            print(f"Correlation score: {max_corr}")
            print(
                f"Transformation error: {translation_error[0]} {translation_error[1]}"
            )
            print(" ")

    # test that precomputed ffts give the same result
    (
        translation,
        max_corr,
        xcorr,
        novlp,
    ) = mpc.phase_correlation(
        fixed_image, moving_image, fixed_mask, moving_mask, overlap_ratio
    )
    fixed_fft = mpc.fft2(fixed_image.astype(float))
    fixed_mask_fft = mpc.fft2(fixed_mask.astype(float))
    fixed_squared_fft = mpc.fft2(fixed_image.astype(float) * fixed_image.astype(float))
    (
        translation_prefft,
        max_corr_prefft,
        xcorr_prefft,
        novlp_prefft,
    ) = mpc.phase_correlation(
        fixed_fft,
        moving_image,
        fixed_mask_fft,
        moving_mask,
        overlap_ratio,
        fixed_image_is_fft=True,
        fixed_mask_is_fft=True,
        fixed_squared_fft=fixed_squared_fft,
    )
    assert np.allclose(translation, translation_prefft)
    assert np.allclose(max_corr, max_corr_prefft)
    assert (xcorr - xcorr_prefft).max() < 1e-4
    assert np.allclose(novlp, novlp_prefft)

    # works if we give only the fft of the fixed image
    (
        translation_prefft,
        max_corr_prefft,
        xcorr_prefft,
        novlp_prefft,
    ) = mpc.phase_correlation(
        fixed_fft,
        moving_image,
        fixed_mask,
        moving_mask,
        overlap_ratio,
        fixed_image_is_fft=True,
        fixed_squared_fft=fixed_squared_fft,
    )
    assert np.allclose(translation, translation_prefft)
    assert np.allclose(max_corr, max_corr_prefft)
    assert (xcorr - xcorr_prefft).max() < 1e-4
    assert np.allclose(novlp, novlp_prefft)

    # cannot run if fixed_image_is_fft is True and fixed_squared_fft is not provided
    with pytest.raises(AssertionError):
        _ = mpc.phase_correlation(
            fixed_fft,
            moving_image,
            fixed_mask_fft,
            moving_mask,
            overlap_ratio,
            fixed_image_is_fft=True,
            fixed_mask_is_fft=True,
        )
    # cannot run if fixed_mask_is_fft is True and fixed_image_is_fft is False
    with pytest.raises(AssertionError):
        _ = mpc.phase_correlation(
            fixed_image,
            moving_image,
            fixed_mask_fft,
            moving_mask,
            overlap_ratio,
            fixed_mask_is_fft=True,
        )


def overlay_registration(fixed_image, transformed_moving_image):
    fixed_image = fixed_image.astype(float)
    transformed_moving_image = transformed_moving_image.astype(float)

    fixed_image_size = fixed_image.shape
    moving_image_size = transformed_moving_image.shape

    if fixed_image_size[0] > moving_image_size[0]:
        transformed_moving_image = np.pad(
            transformed_moving_image,
            ((0, fixed_image_size[0] - moving_image_size[0]), (0, 0)),
        )
    elif fixed_image_size[0] < moving_image_size[0]:
        fixed_image = np.pad(
            fixed_image, ((0, moving_image_size[0] - fixed_image_size[0]), (0, 0))
        )

    if fixed_image_size[1] > moving_image_size[1]:
        transformed_moving_image = np.pad(
            transformed_moving_image,
            ((0, 0), (0, fixed_image_size[1] - moving_image_size[1])),
        )
    elif fixed_image_size[1] < moving_image_size[1]:
        fixed_image = np.pad(
            fixed_image, ((0, 0), (0, moving_image_size[1] - fixed_image_size[1]))
        )

    out_image = np.zeros(
        (fixed_image.shape[0], fixed_image.shape[1], 3), dtype=np.uint8
    )

    fixed_image = (
        255
        * (fixed_image - np.min(fixed_image))
        / (np.max(fixed_image) - np.min(fixed_image))
    )
    transformed_moving_image = (
        255
        * (transformed_moving_image - np.min(transformed_moving_image))
        / (np.max(transformed_moving_image) - np.min(transformed_moving_image))
    )

    out_image[:, :, 0] = fixed_image.astype(np.uint8)
    out_image[:, :, 1] = transformed_moving_image.astype(np.uint8)
    out_image[:, :, 2] = np.zeros_like(fixed_image, dtype=np.uint8)

    return out_image


if __name__ == "__main__":
    test_phase_correlation()
    test_masked_phase_correlation(do_plot=True)
