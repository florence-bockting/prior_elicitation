import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd

tfd = tfp.distributions
tfb = tfp.bijectors

#%% custom distribution families

class Normal_log():
    def __init__(self):
        self.name = "Normal_log_scale"
        self.parameters = ["loc","log_scale"]
    def __call__(self, loc, scale):
        """
        Instantiation of normal distribution with sigma being learned on the 
        log-scale.

        Parameters
        ----------
        loc : int
            location parameter of normal distribution.
        scale : int
            scale parameter of normal distribution on the original scale.

        Returns
        -------
        tfp.distribution object
            normal distribution with sigma being on the log scale.

        """
        return tfd.Normal(loc, tf.exp(scale))

class Normal_log_log():
    def __init__(self):
        self.name = "Normal_log"
        self.parameters = ["log_loc","log_scale"]
    def __call__(self, loc, scale):
        """
        Instantiation of normal distribution with both mu and sigma being 
        learned on the log-scale.

        Parameters
        ----------
        loc : int
            location parameter of normal distribution on the original scale.
        scale : int
            scale parameter of normal distribution on the original scale.

        Returns
        -------
        tfp.distribution object
            normal distribution with mu and sigma being on the log scale.

        """
        return tfd.Normal(tf.exp(loc), tf.exp(scale))

class TruncNormal_log():
    def __init__(self, loc, low, high):
        self.name = "TruncatedNormal_log"
        self.parameters = ["log_scale"]
        self.low = low
        self.high = high
        self.loc = loc
    def __call__(self, scale):
        """
        Instantiation of truncated normal distribution with loc=0. and scale 
        being learned on the log scale. 

        Parameters
        ----------
        scale : int
            scale parameter of truncated normal on the original scale.

        Returns
        -------
        tfp.distribution object
            truncated normal distribution with mu=0. and sigma being learned
            on the log scale.

        """
        return tfd.TruncatedNormal(self.loc, tf.exp(scale), 
                                   low=self.low, high=self.high)

class Gamma_log():
    def __init__(self):
        self.name = "Gamma_log"
        self.parameters = ["log_concentration","log_rate"]
    def __call__(self, concentration, rate):
        """
        Instantiation of gamma distribution with both concentration and rate 
        being learned on the log scale. 

        Parameters
        ----------
        concentration : int
            concentration parameter of gamma distribution on the original scale.
        rate : int
            rate parameter of gamma distribution on the original scale.

        Returns
        -------
        tfp.distribution object
            gamma distribution with both concentration and rate being learned
            on the log scale.

        """
        return tfd.Gamma(concentration=tf.exp(concentration), 
                         rate=tf.exp(rate))

class Beta_log():
    def __init__(self):
        self.name = "Beta_log"
        self.parameters = ["log_concentration1","log_concentration0"]
    def __call__(self, concentration1, concentration0):
        """
        Instantiation of Beta distribution with both concentration1 and 
        concentration2 being learned on the log scale. 

        Parameters
        ----------
        concentration1 : int
            concentration1 parameter of beta distribution on the original scale.
        concentration0 : int
            concentration0 parameter of beta distribution on the original scale.

        Returns
        -------
        tfp.distribution object
            beta distribution with both concentration1 and concentration2 being 
            learned on the log scale.

        """
        return tfd.Beta(concentration1=tf.exp(concentration1), 
                        concentration0=tf.exp(concentration0))

class Normal0_log():
    def __init__(self):
        self.name = "Normal_log_scale_loc0"
        self.parameters = ["log_scale"]
        self.loc = 0.
    def __call__(self, scale):
        return tfd.Normal(self.loc, tf.exp(scale))

class HalfNormal_log():
    def __init__(self):
        self.name = "HalfNormal_log_scale"
        self.parameters = ["log_scale"]
    def __call__(self, scale):
        return tfd.HalfNormal(tf.exp(scale))

class InvGamma_log():
    def __init__(self):
        self.name = "InvGamma_log_nuslab"
        self.parameters = ["log_nuslab"]
    def __call__(self, nu_slab):
        return tfd.InverseGamma(tf.exp(nu_slab)/2, tf.exp(nu_slab)/2)

class Student0():
    def __init__(self, sigma, nu_global):
        self.name = "Student_0"
        self.sigma = sigma
        self.nu_global = nu_global
        self.parameters = ["log_sigmaglobal"]
    def __call__(self, sigma_global):
        return tfd.HalfStudentT(self.nu_global, 0., tf.exp(sigma_global)*self.sigma)

class InvSoftplus(tfb.Bijector):
    def __init__(self, validate_args=False, name='inv_softplus'):
      super(InvSoftplus, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          name=name)

    def _forward(self, x):
      return tfp.math.softplus_inverse(x)

class Gamma_inv_softplus():
    def __init__(self):
        self.name = "Gamma_inv_softplus"
        self.parameters = ["concentration_softplus","rate_softplus"]
        
    def __call__(self, concentration, rate):
        transformed_dist = tfd.TransformedDistribution(
            distribution=tfd.Gamma(concentration, rate),
            bijector = InvSoftplus())
        return transformed_dist

#%% custom target quantities
def custom_R2(ypred, epred):
    """
    Defines R2 such that it is guaranteed to lie within zero and one.
    https://avehtari.github.io/bayes_R2/bayes_R2.html#2_Functions_for_Bayesian_R-squared_for_stan_glm_models

    Parameters
    ----------
    ypred : tf.Tensor
        simulated prior predictions from generative model.
    epred : tf.Tensor
        simulated linear predictor from generative model.

    Returns
    -------
    r2 : tf.Tensor
        computed R2 value.

    """
    # variance of linear predictor 
    var_epred = tf.math.reduce_variance(epred, -1) 
    # variance of difference between ypred and epred
    var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
    # variance of linear predictor divided by total variance
    r2 = var_epred/(var_epred + var_diff)
    return r2


def custom_R2_beta(ypred, epred):
    """
    Defines R2 such that it is guaranteed to lie within zero and one.
    https://avehtari.github.io/bayes_R2/bayes_R2.html#2_Functions_for_Bayesian_R-squared_for_stan_glm_models

    Parameters
    ----------
    ypred : tf.Tensor
        simulated prior predictions from generative model.
    epred : tf.Tensor
        simulated linear predictor from generative model.

    Returns
    -------
    r2 : tf.Tensor
        computed R2 value.

    """
    # variance of linear predictor 
    var_epred = tf.math.reduce_variance(tf.exp(epred), -1) 
    # variance of difference between ypred and epred
    var_diff = tf.math.reduce_variance(tf.subtract(ypred, tf.exp(epred)), -1)
    # variance of linear predictor divided by total variance
    r2 = var_epred/(var_epred + var_diff)
    return r2

def custom_group_means(ypred, design_matrix, factor_indices):
    """
    Computes group means from prior predictions with N observations.

    Parameters
    ----------
    ypred : tf.Tensor
        prior predictions as simulated from the generative model.
    design_matrix : tf.Tensor
        design matrix.
    factor_indices : list of integers
        indices referring to factors in design matrix. First columns has index = 0.

    Returns
    -------
    group_means : tf.Tensor
        group means computed from the model predictions.

    """
    # exclude cont. predictor
    dmatrix_fct = tf.gather(design_matrix, factor_indices, axis = 1)
    # create contrast matrix
    cmatrix = tf.cast(pd.DataFrame(dmatrix_fct).drop_duplicates(), tf.float32)
    # compute group means (shape = B,rep,N_obs,N_gr)
    group_means = tf.stack([tf.reduce_mean(tf.boolean_mask(ypred,
                              tf.math.reduce_all(cmatrix[i,:] == dmatrix_fct, axis = 1),
                              axis = 2), -1) for i in range(cmatrix.shape[0])], -1)
    return group_means

def custom_group(ypred, design_matrix, factor_indices):
    # exclude cont. predictor
    dmatrix_fct = tf.gather(design_matrix, factor_indices, axis = 1)
    # create contrast matrix
    cmatrix = tf.cast(pd.DataFrame(dmatrix_fct).drop_duplicates(), tf.float32)
    # compute group means (shape = B,rep,N_obs,N_gr)
    groups = [tf.boolean_mask(ypred,tf.math.reduce_all(cmatrix[i,:] == dmatrix_fct, 
                                                       axis = 1), axis = 2) for i in range(cmatrix.shape[0])]
    groups_reshaped = [groups[i][:,:,0] for i in range(len(groups))]
    return tf.stack(groups_reshaped, -1)

def custom_std_comparison(prior_samples):
    sd_b0 = tf.math.reduce_std(prior_samples[:,:,0], 1)
    sd_sigma = tf.math.reduce_std(prior_samples[:,:,-1], 1)
    
    log_diff = tf.subtract(tf.math.log(sd_b0), tf.math.log(sd_sigma))

    return tf.expand_dims(log_diff,0)

def custom_group_range(ypred, design_matrix, factor_indices):
    # exclude cont. predictor
    dmatrix_fct = tf.gather(design_matrix, factor_indices, axis = 1)
    # create contrast matrix
    cmatrix = tf.cast(pd.DataFrame(dmatrix_fct).drop_duplicates(), tf.float32)
    # compute group means (shape = B,rep,N_obs,N_gr)
    groups = tf.stack([tf.boolean_mask(ypred,
                              tf.math.reduce_all(cmatrix[i,:] == dmatrix_fct, axis = 1),
                              axis = 2) for i in range(cmatrix.shape[0])], -1)

    groups_reshaped = tf.reshape(groups, (groups.shape[0], groups.shape[1]*groups.shape[2],groups.shape[3]))

    group_ranges = tf.einsum("ij...->ji...", tfp.stats.percentile(groups_reshaped, [1., 99.], axis=1))
    
    return group_ranges

def custom_mu0_sd(ypred, selected_days, R2day0, from_simulated_truth = ["R2day0"]):
    """
    Computes the standard deviation of the linear predictor as the squared
    product of R2 with the variance of ypred.

    Parameters
    ----------
    ypred : tf.Tensor
        model predictions as generated from the generative model
    selected_days : list of integers
        indices of days for which the expert has to indicate prior predictions
    R2day0 : tf.Tensor
        R2 for day 0 as predicted by the expert or as simulated from a pre-
        defined ground truth.
    from_simulated_truth : list of strings, optional
        indicates that the argument "R2day0" should be used from the expert data / 
        simulated ground truth (and not searched for in the model simulations). 

    Returns
    -------
    sdmu : tf.Tensor
        standard deviation of linear predictor for day 0.

    """
    day = selected_days[0]
    len_days = len(selected_days)
    sdmu = tf.sqrt(tf.multiply(R2day0, tf.math.reduce_variance(
                                   ypred[:,:,day::len_days], 
                                   axis=-1)))
    return sdmu

def custom_mu9_sd(ypred, selected_days, R2day9, from_simulated_truth = ["R2day9"]):
    """
    Computes the standard deviation of the linear predictor as the squared
    product of R2 with the variance of ypred.

    Parameters
    ----------
    ypred : tf.Tensor
        model predictions as generated from the generative model
    selected_days : list of integers
        indices of days for which the expert has to indicate prior predictions
    R2day9 : tf.Tensor
        R2 for day 9 as predicted by the expert or as simulated from a pre-
        defined ground truth.
    from_simulated_truth : list of strings, optional
        indicates that the argument "R2day9" should be used from the expert data / 
        simulated ground truth (and not searched for in the model simulations). 

    Returns
    -------
    sdmu : tf.Tensor
        standard deviation of linear predictor for day 9.

    """
    day = selected_days[-1]
    len_days = len(selected_days)
    sdmu = tf.sqrt(tf.multiply(R2day9, tf.math.reduce_variance(
                                   ypred[:,:,day::len_days], 
                                   axis=-1)))
    return sdmu

def custom_avg_obs(ypred):
    y_mean = tf.reduce_mean(ypred, 0)
    return y_mean

def CustomMixture():
    return tfd.Mixture(
        cat=tfd.Categorical(probs=[0.5, 0.5]),
        components=[
          tfd.Normal(loc=-4., scale=0.3),
          tfd.Normal(loc=+4., scale=0.3),
      ])

def custom_correlation(prior_samples):
    corM = tfp.stats.correlation(prior_samples, sample_axis=1, event_axis=-1)
    tensor = tf.experimental.numpy.triu(corM, 1)
    tensor_mask = tf.experimental.numpy.triu(corM, 1) != 0.

    cor = tf.boolean_mask(tensor, tensor_mask, axis=0)
    diag_elements = int((tensor.shape[-1]*(tensor.shape[-1]-1))/2)
    return tf.reshape(cor,(prior_samples.shape[0],diag_elements))

def group_quantiles(y_group):
   quants = tfp.stats.percentile(y_group, [5,25,50,75,95], axis=-1)
   return tf.transpose(quants,perm=(1,2,0))

def custom_target_R2(ypred, epred):
    epred_reshaped = tf.reshape(epred, (1,epred.shape[1]*epred.shape[2]))
    ypred_reshaped = tf.reshape(ypred, (1,ypred.shape[1]*ypred.shape[2]))
    
    # variance of linear predictor 
    var_epred = tf.math.reduce_variance(epred_reshaped, -1) 
    # variance of difference between ypred and epred
    var_diff = tf.math.reduce_variance(tf.subtract(ypred_reshaped, epred_reshaped), -1)
    var_total = var_epred + var_diff
    # variance of linear predictor divided by total variance
    log_R2 = tf.subtract(tf.math.log(var_epred), tf.math.log(var_total))
    
    return log_R2