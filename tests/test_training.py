import unittest

from datarogue.training import training_images, train_all_networks


class TrainingDataTests(unittest.TestCase):
    def test_training_images(self):
        imgs = training_images()
        img, metadata = next(imgs)
        self.assertEqual(metadata.filename, 'xkcd2180.png')
        self.assertEqual(metadata.n_axes, 0)
    
    def test_train_all_networks(self):
        train_all_networks()
