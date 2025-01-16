# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import pickle
import os
import tensorflow as tf

from pythonjsonlogger import jsonlogger  # noqa


def save_as_pkl(obj: any, save_dir: str):
    """
    Helper functions to save a file as pickle.

    Parameters
    ----------
    obj : any
        variable that needs to be saved.
    save_dir : str
        path indicating the file location.

    Returns
    -------
    None.

    Examples
    --------
    >>> save_as_pkl(obj, "results/file.pkl")

    """
    # if directory does not exists, create it
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    # save obj to location as pickle
    with open(save_dir, "wb") as file:
        pickle.dump(obj, file=file)


def identity(x):
    return x


class DoubleBound:
    def __init__(self, lower: float, upper: float):
        """
        A variable constrained to be in the open interval
        (``lower``, ``upper``) is transformed to an unconstrained variable Y
        via a scaled and translated log-odds transform.

        Parameters
        ----------
        lower : float
            lower bound of variable x.
        upper : float
            upper bound of variable x.

        """
        self.lower = lower
        self.upper = upper

    def logit(self, u: float):
        r"""
        Helper function that implements the logit transformation for
        :math:`u \in (0,1)`:

        .. math::

            logit(u) = \log\left(\frac{u}{1-u}\right)

        Parameters
        ----------
        u : float
            variable in open unit interval.

        Returns
        -------
        v : float
            log-odds of u.

        """
        # log-odds definition
        v = tf.math.log(u) - tf.math.log(1 - u)
        # cast v into correct dtype
        v = tf.cast(v, dtype=tf.float32)
        return v

    def inv_logit(self, v: float):
        r"""
        Helper function that implements the inverse-logit transformation (i.e.,
        the logistic sigmoid for :math:`v \in (-\infty,+\infty)`:

        .. math::

            logit^{-1}(v) = \frac{1}{1+\exp(-v)}

        Parameters
        ----------
        v : float
            unconstrained variable

        Returns
        -------
        u : float
            logistic sigmoid of the unconstrained variable

        """
        # logistic sigmoid transform
        u = tf.divide(1.0, (1.0 + tf.exp(v)))
        # cast v to correct dtype
        u = tf.cast(u, dtype=tf.float32)
        return u

    def forward(self, x: float):
        r"""
        scaled and translated logit transform of variable x with ``lower`` and
        ``upper`` bound into an unconstrained variable y.

        .. math::

            Y = logit\left(\frac{X - lower}{upper - lower}\right)

        Parameters
        ----------
        x : float
            variable with lower and upper bound.

        Returns
        -------
        y : float
            unconstrained variable.

        """
        # scaled and translated logit transform
        y = self.logit(tf.divide((x - self.lower), (self.upper - self.lower)))
        # cast y to correct dtype
        y = tf.cast(y, dtype=tf.float32)
        return y

    def inverse(self, y: float):
        r"""
        inverse of the log-odds transform applied to the unconstrained
        variable y in order to transform it into a constrained variable x
        with ``lower`` and ``upper`` bound.

        .. math::

            X = lower + (upper - lower) \cdot logit^{-1}(Y)

        Parameters
        ----------
        y : float
            unconstrained variable

        Returns
        -------
        x : float
        constrained variable with lower and upper bound

        """
        # inverse of log-odds transform
        x = self.lower + (self.upper - self.lower) * self.inv_logit(y)
        # cast x to correct dtype
        x = tf.cast(x, dtype=tf.float32)
        return x


class LowerBound:
    def __init__(self, lower: float):
        """
        A variable with a ``lower`` bound is transformed to an
        unconstrained variable Y via a logarithmic transform.

        Parameters
        ----------
        lower : float
            lower bound of variable X.

        """
        self.lower = lower

    def forward(self, x: float):
        r"""
        logarithmic transform of variable x with ``lower`` bound into an
        unconstrained variable y.

        .. math::

            Y = \log(X - lower)

        Parameters
        ----------
        x : float
            variable with a lower bound.

        Returns
        -------
        y : float
            unconstrained variable.

        """
        # logarithmic transform
        y = tf.math.log(x - self.lower)
        # cast y into correct type
        y = tf.cast(y, dtype=tf.float32)
        return y

    def inverse(self, y: float):
        r"""
        exponential transform of unconstrained variable y into a constrained
        variable x with ``lower`` bound.

        .. math::

            X = \exp(Y) + lower

        Parameters
        ----------
        y : float
            unconstrained variable.

        Returns
        -------
        x : float
            variable with a lower bound.

        """
        # exponential transform
        x = tf.exp(y) + self.lower
        # cast x into correct dtype
        x = tf.cast(x, dtype=tf.float32)
        return x


class UpperBound:
    def __init__(self, upper: float):
        """
        A variable with an ``upper`` bound is transformed into an
        unconstrained variable Y via a logarithmic transform.

        Parameters
        ----------
        upper : float
            upper bound of variable X.

        """
        self.upper = upper

    def forward(self, x: float):
        r"""
        logarithmic transform of variable x with ``upper`` bound into an
        unconstrained variable y.

        .. math::

            Y = \log(upper - X)

        Parameters
        ----------
        x : float
            variable with an upper bound.

        Returns
        -------
        y : float
            unconstrained variable.

        """
        # logarithmic transform
        y = tf.math.log(self.upper - x)
        # cast y into correct dtype
        y = tf.cast(y, dtype=tf.float32)
        return y

    def inverse(self, y: float):
        r"""
        exponential transform of unconstrained variable y into a constrained
        variable x with ``upper`` bound.

        .. math::

            X = upper - \exp(Y)

        Parameters
        ----------
        y : float
            unconstrained variable.

        Returns
        -------
        x : float
            variable with an upper bound.

        """
        # exponential transform
        x = self.upper - tf.exp(y)
        # cast x into correct dtype
        x = tf.cast(x, dtype=tf.float32)
        return x
