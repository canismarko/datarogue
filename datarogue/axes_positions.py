from typing import List, Tuple

import numpy as np
from keras import Sequential
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense

def num_axes_cnn():
    # initially taken from tutorial
    input_shape = (64, 64, 3)
    num_classes = 10
    # https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


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
    cnn = num_axes_cnn()
    pass


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
