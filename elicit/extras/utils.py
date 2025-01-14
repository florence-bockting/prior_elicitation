# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp


def log_R2(ypred, epred):
    var_epred = tf.math.reduce_variance(epred, -1)
    # variance of difference between ypred and epred
    var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
    var_total = var_epred + var_diff
    # variance of linear predictor divided by total variance
    log_R2 = tf.subtract(tf.math.log(var_epred), tf.math.log(var_total))
    return log_R2


def pearson_correlation(prior_samples):
    corM = tfp.stats.correlation(prior_samples, sample_axis=1, event_axis=-1)
    tensor = tf.experimental.numpy.triu(corM, 1)
    tensor_mask = tf.experimental.numpy.triu(corM, 1) != 0.0

    cor = tf.boolean_mask(tensor, tensor_mask, axis=0)
    diag_elements = int((tensor.shape[-1] * (tensor.shape[-1] - 1)) / 2)
    return tf.reshape(cor, (prior_samples.shape[0], diag_elements))