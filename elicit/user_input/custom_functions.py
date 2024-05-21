import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd

tfd = tfp.distributions

from functions.helper_functions import get_lower_triangular


## custom distributions
class Normal_log():
    def __call__(self, loc, scale):
        return tfd.Normal(loc, tf.exp(scale))
    
class TruncNormal():
    def __init__(self, low, high):
        self.low = low
        self.high = high
    def __call__(self, loc, scale):
        return tfd.TruncatedNormal(loc, scale, low=self.low, high=self.high)

## % custom target quantities
def custom_R2(ypred, epred):
    var_epred = tf.math.reduce_variance(epred, -1) 
    var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
    r2 = var_epred/(var_epred + var_diff)
    return r2

def custom_cor(prior_samples):
    cor_M = tfp.stats.correlation(prior_samples, sample_axis = 1, event_axis = 2)
    cor_val = get_lower_triangular(cor_M)
    return cor_val

def custom_group_means(ypred, design_matrix, factor_indices):
    # exclude cont. predictor
    dmatrix_fct = tf.gather(design_matrix, factor_indices, axis = 1)
    # create contrast matrix
    cmatrix = tf.cast(pd.DataFrame(dmatrix_fct).drop_duplicates(), tf.float32)
    # compute group means (shape = B,rep,N_obs,N_gr)
    groups = tf.stack([tf.boolean_mask(ypred,
                              tf.math.reduce_all(cmatrix[i,:] == dmatrix_fct, axis = 1),
                              axis = 2) for i in range(cmatrix.shape[0])], -1)
    group_means = tf.reduce_mean(groups, 2) 
    return group_means

## % custom distribution families
def custom_normal_log(loc, scale):
    unconstrained_normal = tfd.Normal(loc=loc, scale=tf.exp(scale))
    return unconstrained_normal


