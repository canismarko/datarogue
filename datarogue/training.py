import os

import imageio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .image_tools import resize_image, training_dir
from .axes_positions import NumAxesCNN

def training_images(img_size=None):
    """Generator of images with metadata.
    
    Each entry is a 2-tuple of the image data and associated
    meta-data.
    
    """
    # Open the file of metadata
    df = pd.read_csv(os.path.join(training_dir, 'images-metadata.tsv'), sep='\t')
    for metadata in df.itertuples():
        fname = os.path.join(training_dir, 'images', metadata.filename)
        img = imageio.imread(fname)
        # Change image size
        if img_size is not None:
            img = resize_image(img, img_size)
        yield (img, metadata)


def train_all_networks():
    """Load metadata and train all the convolution neural networks."""
    # Number of axes
    n_axes_cnn = NumAxesCNN()
    training_data = training_images(img_size=n_axes_cnn.image_shape)
    xdata = []
    ydata = []
    for img, metadata in training_data:
        xdata.append(img)
        ydata.append([metadata.n_axes])
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    n_axes_cnn.fit(x=xdata, y=ydata, batch_size=8)
    print('done')
    # plt.imshow(next(training_data)[0])
    # plt.show()
