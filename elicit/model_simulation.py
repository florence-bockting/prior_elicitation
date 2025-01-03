# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import logging
import elicit as el

from elicit.configs import *  # noqa

tfd = tfp.distributions


def softmax_gumbel_trick(epred: float, likelihood: callable,
                         upper_thres: float, temp: float, seed: int):
    """
    The softmax-gumbel trick computes a continuous approximation of ypred from
    a discrete likelihood and thus allows for the computation of gradients for
    discrete random variables.

    Currently this approach is only implemented for models without upper
    boundary (e.g., Poisson model).

    Corresponding literature:

    - Maddison, C. J., Mnih, A. & Teh, Y. W. The concrete distribution:
        A continuous relaxation of
      discrete random variables in International Conference on Learning
      Representations (2017). https://doi.org/10.48550/arXiv.1611.00712
    - Jang, E., Gu, S. & Poole, B. Categorical reparameterization with
    gumbel-softmax in International Conference on Learning Representations
    (2017). https://openreview.net/forum?id=rkE3y85ee.
    - Joo, W., Kim, D., Shin, S. & Moon, I.-C. Generalized gumbel-softmax
    gradient estimator for generic discrete random variables.
      Preprint at https://doi.org/10.48550/arXiv.2003.01847 (2020).

    Parameters
    ----------
    model_simulations : dict
        dictionary containing all simulated output variables from the
        generative model.
    global_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    ypred : tf.Tensor
        continuously approximated ypred from the discrete likelihood.

    """
    # set seed
    tf.random.set_seed(seed)
    # get batch size
    B = epred.shape[0]
    # get number of simulations from priors
    S = epred.shape[1]
    # get number of observations
    number_obs = epred.shape[2]
    # constant outcome vector (including zero outcome)
    thres = upper_thres
    c = tf.range(thres + 1, delta=1, dtype=tf.float32)
    # broadcast to shape (B, rep, outcome-length)
    c_brct = tf.broadcast_to(c[None, None, None, :], shape=(B, S, number_obs,
                                                            len(c)))
    # compute pmf value
    pi = likelihood.prob(c_brct)
    # prevent underflow
    pi = tf.where(pi < 1.8 * 10 ** (-30), 1.8 * 10 ** (-30), pi)
    # sample from uniform
    u = tfd.Uniform(0, 1).sample((B, S, number_obs, len(c)))
    # generate a gumbel sample from uniform sample
    g = -tf.math.log(-tf.math.log(u))
    # softmax gumbel trick
    w = tf.nn.softmax(
        tf.math.divide(
            tf.math.add(tf.math.log(pi), g), temp,
        )
    )
    # reparameterization/linear transformation
    ypred = tf.reduce_sum(tf.multiply(w, c), axis=-1)
    return ypred


def simulate_from_generator(prior_samples, ground_truth, global_dict):
    """
    Simulates data from the specified generative model.

    Parameters
    ----------
    prior_samples : dict
        samples from prior distributions.
    ground_truth : bool
        if simulation is based on true hyperparameter vector. Mainly for
        saving results in a specific "expert" folder for later analysis.
    global_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    model_simulations : dict
        simulated data from generative model.

    """
    logger = logging.getLogger(__name__)
    if ground_truth:
        logger.info("simulate from true generative model")
    else:
        logger.info("simulate from generative model")

    # set seed
    seed=global_dict["trainer"]["seed"]
    tf.random.set_seed(seed)
    # create subdictionaries for better readability
    dict_generator = global_dict["model"]
    # get model and initialize generative model
    GenerativeModel = dict_generator["obj"]
    generative_model = GenerativeModel()
    # get model specific arguments (that are not prior samples)
    add_model_args = dict_generator.copy()
    add_model_args.pop("obj")
    add_model_args["seed"]=seed
    # simulate from generator
    if add_model_args is not None:
        model_simulations = generative_model(prior_samples, **add_model_args)
    else:
        model_simulations = generative_model(ground_truth, prior_samples)
    # save file in object
    saving_path = global_dict["trainer"]["output_path"]
    if saving_path is not None:
        if ground_truth:
            saving_path = saving_path + "/expert"
        path = saving_path + "/model_simulations.pkl"
        el.save_as_pkl(model_simulations, path)

    return model_simulations
