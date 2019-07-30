import unittest

import numpy as np

from datarogue import image_tools


class ImageResizeTests(unittest.TestCase):
    def test_padding_2D(self):
        in_img = np.array(
            [[0, 1, 0],
             [0, 1, 1]]
        )
        target = np.array(
            [[0, 1, 0],
             [0, 1, 1],
             [0, 0, 0],
             [0, 0, 0]]
        )
        out_img = image_tools.resize_image(img=in_img, new_shape=(4, 3))
        np.testing.assert_array_equal(out_img, target)

    def test_scaling_2D(self):
        in_img = np.array(
            [[1., 0],
             [0, 1.]]
        )
        a, b = 0.736328125, 0.263672
        target = np.array(
            [[a, a, b, b],
             [a, a, b, b],
             [b, b, a, a],
             [b, b, a, a]]
        )
        out_img = image_tools.resize_image(img=in_img, new_shape=(4, 4))
        np.testing.assert_almost_equal(out_img, target)
