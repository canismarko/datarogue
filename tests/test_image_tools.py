import unittest
import imageio
import os

import numpy as np
import matplotlib.pyplot as plt

from datarogue import image_tools


comic_fname = os.path.join(os.path.dirname(__file__), '../', 'training_data', 'images', 'xkcd2180.png')

class ImageResizeTests(unittest.TestCase):
    def test_padding_2D(self):
        in_img = np.array(
            [[0., 1., 0.],
             [0., 0., 1.]]
        )
        target = np.array(
            [[0., 1., 0.],
             [0., 0., 1.],
             [0., 0., 0.],
             [0., 0., 0.]]
        )
        out_img = image_tools.resize_image(img=in_img, new_shape=(4, 3))
        np.testing.assert_array_equal(out_img, target)
    
    def test_scaling_2D(self):
        in_img = np.array(
            [[1., 0],
             [0, 1.]]
        )
        # a, b = 0.736328125, 0.263672
        a, b = 1., 0.
        target = np.array(
            [[a, a, b, b],
             [a, a, b, b],
             [b, b, a, a],
             [b, b, a, a]]
        )
        out_img = image_tools.resize_image(img=in_img, new_shape=(4, 4))
        np.testing.assert_almost_equal(out_img, target)
    
    def test_real_image_rgb(self):
        comic = imageio.imread(comic_fname)
        to_shape = (1024, 1024, 3)
        out_img = image_tools.resize_image(img=comic, new_shape=to_shape)
        self.assertEqual(out_img.shape, to_shape)
