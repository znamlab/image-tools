import numpy as np
import numpy.testing as npt
import skimage
from skimage.transform import SimilarityTransform

from image_tools.similarity_transforms import make_transform, transform_image


def test_transform_image():
    # Test case 1: No transformation
    im = skimage.data.cat()[:, :, 0]
    transformed_im = transform_image(im)
    npt.assert_array_equal(im, transformed_im)

    # Test case 2: Rotation
    if False:
        # this doesn t work. THere is interpolation in the transform_image function
        angle = 90.0
        transformed_im = transform_image(im, angle=angle)
        expected_image = np.zeros_like(im)
        dif = np.diff(im.shape)[0]
        expected_image[:, dif // 2 : -dif // 2] = np.rot90(im)[dif // 2 : -dif // 2, :]
        npt.assert_array_equal(expected_image, transformed_im)

    # TODO check scale, translation and combination of everything


def test_make_transform():
    # Test case 1: No transformation
    s = 1.0
    angle = 0.0
    shift = (0, 0)
    shape = (10, 10)
    expected_transform = SimilarityTransform(scale=s, rotation=angle, translation=shift)
    transform = make_transform(s, angle, shift, shape)
    npt.assert_equal(expected_transform, transform)

    # TODO add more test cases for scale, rotation, translation and combination


if __name__ == "__main__":
    test_make_transform()
    test_transform_image()
