import os

import imageio
import pandas as pd


training_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')


def training_images():
    """Genator of images with metadata.
    
    Each entry is a 2-tuple of the image data and associated
    meta-data.
    
    """
    # Open the file of metadata
    df = pd.read_csv(os.path.join(training_dir, 'images-metadata.tsv'), sep='\t')
    for metadata in df.itertuples():
        fname = os.path.join(training_dir, 'images', metadata.filename)
        img = imageio.imread(fname)
        yield (img, metadata)
