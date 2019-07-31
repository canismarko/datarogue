import os
from typing import Tuple

import numpy as np
from skimage import transform, util, exposure


training_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

def resize_image(img: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    """Take an image array and scale/pad it to the desired dimensions.
    
    The resulting image will contain all the original data with the
    same aspect ratio. For example, a (32, 28) input image with a
    *new_shape* of (64, 76) will be scaled to (64, 56), then padded to
    (64, 76) with the median value of the input image.
    
    Parameters
    ==========
    img 
        The 2D or 3D array holding image data.
    new_shape
        The desired 2D shape of the new image.
    
    Returns
    =======
    new_img
        The scaled and padded image matching *new_shape* (possibly
        with RGB dimension).
    
    """
    new_img = img[()]
    is_rgb = (img.ndim == 3)
    # Determine scaling factor
    old_shape = new_img.shape
    scale_factor = np.min(np.divide(new_shape, old_shape))
    # Apply scaling
    if scale_factor != 1:
        new_img = transform.rescale(new_img, scale=scale_factor,
                                    order=1, multichannel=is_rgb, preserve_range=True)
    # Determine the degree of padding
    old_shape = new_img.shape
    pads = tuple((0, p) for p in np.subtract(new_shape, old_shape))
    pad_val = np.median(img)
    # if is_rgb:
    #     pads += ((0, 0),)
    # Apply padding
    new_img = np.pad(new_img, pad_width=pads, mode='constant', constant_values=pad_val)
    # Rescale to get good exposure
    new_img = exposure.rescale_intensity(new_img)
    return new_img
