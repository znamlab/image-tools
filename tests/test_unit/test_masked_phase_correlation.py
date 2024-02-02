import imageio
import matplotlib.pyplot as plt
import numpy as np

from image_tools import masked_phase_correlation as mpc


def test_masked_phase_correlation():
    # Perform the registration on several sets of images.
    x = [75, -130, 130]
    y = [75, 130, 130]
    overlap_ratio = 0.1

    for i in range(3):
        fixed_image = imageio.imread(
            f"../test_data/dataOriginalX{x[i]:02}Y{y[i]:02}.png"
        )
        moving_image = imageio.imread(
            f"../test_data/TransformedX{x[i]:02}Y{y[i]:02}.png"
        )
        fixed_mask = fixed_image != 0
        moving_mask = moving_image != 0

        translation, maxC, _, _ = mpc.masked_translation_registration(
            fixed_image, moving_image, fixed_mask, moving_mask, overlap_ratio
        )
        transformed_moving_image = mpc.transform_image_translation(
            moving_image, translation
        )

        def overlay_registration(fixed_image, transformed_moving_image):
            raise NotImplementedError

        # Given the transform, transform the moving image.
        overlay_image = overlay_registration(fixed_image, transformed_moving_image)
        plt.figure()
        plt.imshow(overlay_image)
        plt.title(f"Test {i+1}: Registered Overlay Image")
        plt.show()

        print(f"Test {i+1}:")
        print(f"Computed translation: {translation[0]} {-translation[1]}")
        print(f"Correlation score: {maxC}")
        true_translation = np.array([x[i], -y[i]])
        print(f"Transformation error: {np.array(translation) - true_translation}")
        print(" ")
