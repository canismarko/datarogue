import unittest
import os

import imageio

from datarogue import axes_positions, num_axes

comic_fname = os.path.join(os.path.dirname(__file__), '../', 'training_data', 'images', 'xkcd2180.png')

class AxesNumberTests(unittest.TestCase):
    def test_no_axes(self):
        comic = imageio.imread(comic_fname)
        n_axs = num_axes(comic)
        print(n_axs)
        self.assertEqual(n_axs, 0)


class AxesPositionTests(unittest.TestCase):
    def test_no_axes(self):
        comic = imageio.imread(comic_fname)
        axs = axes_positions(comic)
        self.assertEqual(len(axs), 0)
