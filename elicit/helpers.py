# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import pickle
import os
import pandas as pd
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


def marginal_prior_moments(prior_samples, epoch, output_path):
    """
    Used for summarizing learned prior distributions in the case of
    method='deep_prior'.
    Computes mean and standard deviation of the sampled marginal prior
    distributions for each epoch.

    Parameters
    ----------
    prior_samples : dict
        Samples from prior distributions.
    epoch : int
        current epoch.
    global_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    res_dict : dict
        returns mean (key:'means') and standard deviation (key:'stds')
        for each sampled marginal prior distribution; for each epoch.

    """
    if output_path is not None:
        saving_path = output_path
    else:
        saving_path = "elicit_temp"

    if epoch == 0:
        res_dict = {"means": [], "stds": []}
    else:
        path_res_dict = saving_path + "/res_dict.pkl"
        res_dict = pd.read_pickle(rf"{path_res_dict}")

    means = tf.reduce_mean(prior_samples, (0, 1))
    sds = tf.reduce_mean(tf.math.reduce_std(prior_samples, 1), 0)

    for val, name in zip([means, sds], ["means", "stds"]):
        res_dict[name].append(val)
    # save result dictionary

    path_res_dict = saving_path + "/res_dict.pkl"
    save_as_pkl(res_dict, path_res_dict)
    return res_dict


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