# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class ToyModel:
    def __call__(self, ground_truth, prior_samples, N):
        # number of observations (intercept-only)
        X = tf.ones((1, N))
        # linear predictor (= mu)
        epred = tf.expand_dims(prior_samples[:, :, 0], -1) @ X
        # data-generating model
        likelihood = tfd.Normal(
            loc=epred, scale=tf.expand_dims(prior_samples[:, :, 1], -1)
        )
        # prior predictive distribution (=height)
        ypred = likelihood.sample()

        return dict(
            likelihood=likelihood, ypred=ypred, epred=epred,
            prior_samples=prior_samples
        )


class BinomialModel(tf.Module):
    def __call__(self, ground_truth, prior_samples, design_matrix,
                 total_count):

        epred = prior_samples @ tf.transpose(design_matrix)

        probs = tf.sigmoid(epred)

        likelihood = tfd.Binomial(total_count=total_count,
                                  probs=probs[:, :, :, None])

        return dict(
            likelihood=likelihood,
            ypred=None,
            epred=epred,
            prior_samples=prior_samples
        )


class NormalModelSimple(tf.Module):
    def __call__(self, ground_truth, prior_samples, design_matrix, sigma):

        epred = prior_samples @ tf.transpose(design_matrix)
        likelihood = tfd.Normal(loc=epred, scale=sigma)
        ypred = likelihood.sample()

        X0 = ypred[:,:,0]
        X1 = ypred[:,:,1]

        # R2
        var_epred = tf.math.reduce_variance(epred, -1)
        # variance of difference between ypred and epred
        var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
        var_total = var_epred + var_diff
        # variance of linear predictor divided by total variance
        log_R2 = tf.subtract(tf.math.log(var_epred), tf.math.log(var_total))

        return dict(
            likelihood=likelihood,
            ypred=ypred,
            epred=epred,
            prior_samples=prior_samples,
            logR2 = log_R2,
            X0=X0,
            X1=X1
            )


class NormalModelComplex(tf.Module):
    def __call__(self, ground_truth, prior_samples, design_matrix):

        epred = prior_samples[:, :, :-1] @ tf.transpose(design_matrix)
        sigma = tf.abs(prior_samples[:, :, -1][:, :, None])

        likelihood = tfd.Normal(loc=epred, scale=sigma)
        ypred = likelihood.sample()

        X0 = ypred[:,:,0]
        X1 = ypred[:,:,1]

        # R2
        var_epred = tf.math.reduce_variance(epred, -1)
        # variance of difference between ypred and epred
        var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
        var_total = var_epred + var_diff
        # variance of linear predictor divided by total variance
        log_R2 = tf.subtract(tf.math.log(var_epred), tf.math.log(var_total))

        prior_samples = tf.concat(
            [prior_samples[:, :, :-1],
             tf.abs(prior_samples[:, :, -1][:, :, None])],
            axis=-1,
        )

        return dict(
            likelihood=likelihood,
            ypred=ypred,
            epred=epred,
            prior_samples=prior_samples,
            R2 = tf.exp(log_R2),
            X0=X0,
            X1=X1
            )


class NormalModel(tf.Module):
    def __call__(self, ground_truth, prior_samples, design_matrix):

        epred = prior_samples[:, :, :-1] @ tf.transpose(design_matrix)
        sigma = tf.abs(prior_samples[:, :, -1][:, :, None])

        likelihood = tfd.Normal(loc=epred, scale=sigma)

        ypred = likelihood.sample()

        group1 = ypred[:, :, 0::3]
        group2 = ypred[:, :, 1::3]
        group3 = ypred[:, :, 2::3]
        
        # effects
        group2vs1 = tf.reduce_mean(group2,-1)-tf.reduce_mean(group1,-1)
        group3vs1 = tf.reduce_mean(group3,-1)-tf.reduce_mean(group1,-1)
        
        # R2
        var_epred = tf.math.reduce_variance(epred, -1)
        # variance of difference between ypred and epred
        var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
        var_total = var_epred + var_diff
        # variance of linear predictor divided by total variance
        log_R2 = tf.subtract(tf.math.log(var_epred), tf.math.log(var_total))

        prior_samples = tf.concat(
            [prior_samples[:, :, :-1],
             tf.abs(prior_samples[:, :, -1][:, :, None])],
            axis=-1,
        )

        return dict(
            likelihood=likelihood,
            ypred=ypred,
            epred=epred,
            prior_samples=prior_samples,
            logR2=log_R2,
            group1=group1,
            group2=group2,
            group3=group3,
            group2vs1=group2vs1,
            group3vs1=group3vs1
        )


class BinomialModel2(tf.Module):
    def __call__(
        self, ground_truth, prior_samples, design_matrix, total_count,
        **kwargs
    ):
        """
        Binomial model with one continuous predictor.

        Parameters
        ----------
        ground_truth: bool
            argument for internal usage
        prior_samples : dict
            samples from prior distributions.
        design_matrix : tf.Tensor
            design matrix.
        total_count : int
            total counts of Binomial model.
        **kwargs : keyword argument, optional
            additional keyword arguments

        Returns
        -------
        dictionary with the following keys:

        - likelihood: model likelihood
        - ypred: model predictions; for discrete likelihoods ypred=None as it
          will be approximated via the softmax-gumble trick
        - epred: predictions of linear predictor
        - prior samples: samples from prior distributions

        """

        # linear predictor
        theta = design_matrix @ tf.expand_dims(prior_samples, axis=-1)

        # map linear predictor to theta
        epred = tf.sigmoid(theta)

        # define likelihood
        likelihood = tfd.Binomial(total_count=total_count, probs=epred)

        return dict(
            likelihood=likelihood, ypred=None, epred=epred,
            prior_samples=prior_samples
        )


class PoissonModel(tf.Module):
    def __call__(self, ground_truth, prior_samples, design_matrix, **kwargs):
        """
        Poisson model with one continuous predictor and one categorical
        predictor with three levels.

        Parameters
        ----------
        prior_samples : dict
            samples from prior distributions.
        design_matrix : tf.Tensor
            design matrix.
        **kwargs : keyword argument, optional
            additional keyword arguments

        Returns
        -------
        dictionary with the following keys:

        - likelihood: model likelihood
        - ypred: model predictions; for discrete likelihoods ypred=None as it
          will be approximated via the softmax-gumble trick
        - epred: predictions of linear predictor
        - prior samples: samples from prior distributions

        """
        # linear predictor
        theta = design_matrix @ tf.expand_dims(prior_samples, -1)

        # map linear predictor to theta
        epred = tf.exp(theta)

        # define likelihood
        likelihood = tfd.Poisson(rate=epred)

        return dict(
            likelihood=likelihood,
            ypred=None,
            epred=epred[:, :, :, 0],
            prior_samples=prior_samples,
        )
