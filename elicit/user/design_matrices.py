# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import numpy as np
import tensorflow as tf
import pandas as pd
import patsy as pa
import tensorflow_probability as tfp

tfd = tfp.distributions



def load_design_matrix(N):
    X = tf.range(1.,N,1.)
    x_sd = tf.math.reduce_std(X)
    X_std = X/x_sd
    X_selected = tfp.stats.percentile(X_std,[25, 75])
    d_final = tf.stack([[1.0] * len(X_selected), X_selected.numpy()], axis=-1)
    return d_final


def load_design_matrix_normal(N_group):
    # construct design matrix with a 3-level factor
    df = pa.dmatrix("a", pa.balanced(a=3, repeat=N_group),
                    return_type="dataframe")
    # save in correct format
    d_final = tf.cast(df, dtype=tf.float32)
    return d_final


def load_design_matrix_binomial2():
    X = tf.cast(list(range(0, 31)), tf.float32)
    X_mean = tf.reduce_mean(X)
    X_sd = tf.math.reduce_std(X)
    X_std = (X - X_mean) / X_sd
    d_final = tf.stack([[1.0] * len(X_std), X_std], axis=-1)
    return d_final


def load_design_matrix_poisson():
    """
    Loads the equality index dataset from BayesRule!, preprocess the data, and
    creates a design matrix as used in the poisson model.
    source: https://www.bayesrulesbook.com/chapter-12

    Parameters
    ----------
    scaling : str or None
        whether the continuous predictor should be scaled;
        possible values = ['divide_by_std', 'standardize']
    selected_obs : list of integers or None
        whether only specific observations shall be selected from the design
        matrix.

    Returns
    -------
    design_matrix : tf.Tensor
        design matrix.

    """
    # load dataset from repo
    url = "https://github.com/bayes-rules/bayesrules/blob/\
        404fbdbae2957976820f9249e9cc663a72141463/data-raw/\
            equality_index/equality_index.csv?raw=true"
    df = pd.read_csv(url)
    # exclude california from analysis as extreme outlier
    df_filtered = df.loc[df["state"] != "california"]
    # select predictors
    df_prep = df_filtered.loc[:, ["historical", "percent_urban"]]
    # reorder historical predictor
    df_reordered = df_prep.sort_values(["historical", "percent_urban"])
    # add dummy coded predictors
    df_reordered["gop"] = np.where(df_reordered["historical"] == "gop", 1, 0)
    df_reordered["swing"] = np.where(df_reordered["historical"] == "swing", 1,
                                     0)
    df_reordered["intercept"] = 1
    # select only required columns
    data_reordered = df_reordered[["intercept", "percent_urban", "gop",
                                   "swing"]]
    # scale predictor if specified
    sd = np.std(np.array(data_reordered["percent_urban"]))
    mean = np.mean(np.array(data_reordered["percent_urban"]))
    d_scaled = data_reordered.assign(
        percent_urban_scaled=(
            np.array(data_reordered["percent_urban"]) - mean
            ) / sd
    )
    d_final = d_scaled.loc[:, ["intercept", "percent_urban_scaled", "gop",
                               "swing"]]
    # cast to appropriate data type
    array = tf.cast(d_final, tf.float32)
    return array
