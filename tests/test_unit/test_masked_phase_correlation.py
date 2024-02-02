import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from image_tools import masked_phase_correlation as mpc


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

        (translation, max_corr, _, _,) = mpc.masked_translation_registration(
            fixed_image, moving_image, fixed_mask, moving_mask, overlap_ratio
        )
        transformed_moving_image = mpc.transform_image_translation(
            moving_image, translation
        )

        # Calculate the difference between the two images in the overlap region.
        overlap = fixed_mask & moving_mask
        initial_diff = np.mean(
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
        assert (
            difference < initial_diff / 5
        ), f"Test {i+1}: Difference between images is too large."

        assert max_corr > 0.9, f"Test {i+1}: Correlation score is too low."
        true_translation = np.array([x[i], -y[i]])
        translation_error = np.array(translation) - true_translation
        assert np.sum(translation_error) < 3, "Translation error is too large."

        if do_plot:
            # Given the transform, transform the moving image.
            overlay_image = overlay_registration(fixed_image, transformed_moving_image)
            plt.figure()
            plt.imshow(overlay_image)
            plt.title(f"Test {i+1}: Registered Overlay Image")
            plt.show()

            print(f"Test {i+1}:")
            print(f"Computed translation: {translation[0]} {-translation[1]}")
            print(f"Correlation score: {max_corr}")
            print(
                f"Transformation error: {translation_error[0]} {translation_error[1]}"
            )
            print(" ")


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
    test_masked_phase_correlation()
