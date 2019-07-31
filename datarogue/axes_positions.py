from typing import List, Tuple

import numpy as np
from keras import Sequential, losses, optimizers
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense

from .image_tools import resize_image

class NumAxesCNN(Sequential):
    """A convolution neural network for determining the number of axes."""
    image_shape = (256, 256, 3)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initially taken from tutorial
        # https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
        self.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=self.image_shape))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(Conv2D(64, (5, 5), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Flatten())
        self.add(Dense(1000, activation='relu'))
        self.add(Dense(1, activation='softmax'))
        # Compile the neural network
        self.compile(loss=losses.mean_squared_error,
                      optimizer=optimizers.SGD(lr=0.01),
                      metrics=['accuracy'])


def num_axes(figure: np.ndarray) -> int:
    """Extract the number of axes in a figure.
    
    Parameters
    ==========
    figure :
      The axes image.
    
    Returns
    =======
    n_axes :
      How many axes were detected within the figure image.
    
    """
    cnn = NumAxesCNN()
    # Resample the image to match the CNN
    x = resize_image(figure, new_shape=cnn.image_shape)
    x = np.array([x])
    # Use the neural network to predict the number of axes
    cnn_out = cnn.predict(x)
    n_axes = int(cnn_out[0,0])
    return n_axes


def axes_positions(figure: np.ndarray) -> List[Tuple[int]]:
    """Extract the pixel positions for axes in a figures.
    
    Parameters
    ==========
    figure :
      The axes image.
    
    Returns
    =======
    positions :
      A list of tuples, each with the (row, col) pixel position of a
      unique axis.
    
    """
    pass
