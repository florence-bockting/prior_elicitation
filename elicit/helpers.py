# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import pickle
import os
import tensorflow as tf
import numpy as np

from pythonjsonlogger import jsonlogger # noqa


def save_as_pkl(variable, path_to_file):
    """
    Helper functions to save a file as pickle.

    Parameters
    ----------
    variable : any
        file that needs to be saved.
    path_to_file : str
        path indicating the file location.

    Returns
    -------
    None.

    """
    # if directory does not exists, create it
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    # save file to location as pickle
    with open(path_to_file, "wb") as df_file:
        pickle.dump(variable, file=df_file)


def identity(x):
    return x

class DoubleBound:
    def __init__(self, lower, upper):
        self.lower=lower
        self.upper=upper

    def logit(self, x):
        return tf.cast(np.log(x) - np.log(1 - x), dtype=tf.float32)

    def inv_logit(self, x):
        return tf.cast(np.exp(x) / (1 + np.exp(x)), dtype=tf.float32)

    def forward(self, x):
        return tf.cast(self.logit((x-self.lower)/(self.upper-self.lower)), dtype=tf.float32)

    def inverse(self, x):
        return tf.cast(self.lower+(self.upper-self.lower)*self.inv_logit(x), dtype=tf.float32)

class LowerBound:
    def __init__(self, lower):
        self.lower=lower
    def forward(self, x):
        return tf.cast(np.log(x-self.lower), dtype=tf.float32)
    def inverse(self, x):
        return tf.cast(np.exp(x)+self.lower, dtype=tf.float32)
    
class UpperBound:
    def __init__(self, upper):
        self.upper=upper
    def forward(self, x):
        return tf.cast(np.log(self.upper-x), dtype=tf.float32)
    def inverse(self, x):
        return tf.cast(self.upper-np.exp(x), dtype=tf.float32)