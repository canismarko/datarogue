import unittest
import os

import imageio

from datarogue import axes_positions, num_axes


class AxesNumberTests(unittest.TestCase):
    def test_no_axes(self):
        comic = imageio.imread(os.path.join(os.path.dirname(__file__), 'xkcd2180.png'))
        n_axs = num_axes(comic)
        self.assertEqual(n_axs, 0)


class AxesPositionTests(unittest.TestCase):
    def test_no_axes(self):
        comic = imageio.imread(os.path.join(os.path.dirname(__file__), 'xkcd2180.png'))
        axs = axes_positions(comic)
        self.assertEqual(len(axs), 0)
