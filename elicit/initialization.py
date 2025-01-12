# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import logging
import elicit as el

from scipy.stats import qmc
from tqdm import tqdm

tfd = tfp.distributions


def init_method(seed, hyppar, n_samples, method, mean, radius, parameters):
    """
    Initialize multivariate normal prior over hyperparameter values

    Parameters
    ----------
    n_hypparam : int
        Number of hyperparameters.
    n_samples : int
        number of warmup iterations.

    Returns
    -------
    mvdist : tf.tensor
        samples from the multivariate prior
        (shape=(n_warm_up, n_hyperparameters).

    """
    # set seed
    tf.random.set_seed(seed)

    # Validate n_samples
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    
    # Validate method
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if method not in ["sobol", "lhs", "random"]:
        raise ValueError(
            "Unsupported method. Choose from 'sobol', 'lhs', or 'random'.")

    # counter number of hyperparameters
    n_hypparam=0
    name_hyper=list()
    res_dict=dict()
    for i in range(len(parameters)):
        for hyperparam in parameters[i]["hyperparams"]:
            dim = parameters[i]["hyperparams"][hyperparam]["dim"]
            name = parameters[i]["hyperparams"][hyperparam]["name"]
            n_hypparam+=dim
            for j in range(dim):
                name_hyper.append(name)

            if hyppar is None:
                # make sure type is correct
                mean = tf.cast(mean, tf.float32)
                radius = tf.cast(radius, tf.float32)

                # Generate samples based on the chosen method
                if method == "sobol":
                    sampler = qmc.Sobol(d=dim, seed=seed.numpy())
                    sample_data = sampler.random(n=n_samples)
                elif method == "lhs":
                    sampler = qmc.LatinHypercube(d=dim, seed=seed.numpy())
                    sample_data = sampler.random(n=n_samples)
                elif method == "random":
                    uniform_samples = tfd.Uniform(
                        tf.subtract(mean,radius), tf.add(mean,radius)
                        ).sample((n_samples, dim))
                # Inverse transform
                if method == "sobol" or method == "lhs":
                    sample_dat = tf.cast(tf.convert_to_tensor(sample_data),
                                         tf.float32)
                    uniform_samples = tfd.Uniform(tf.subtract(mean,radius),
                                                    tf.add(mean,radius)
                                                  ).quantile(sample_dat)
                res_dict[name]=uniform_samples
            else:
                uniform_samples=[]
                for i,j,n in zip(mean, radius, hyppar):
                    # make sure type is correct
                    i = tf.cast(i, tf.float32)
                    j = tf.cast(j, tf.float32)

                    if method == "random":
                        uniform_samples = tfd.Uniform(
                            tf.subtract(i,j), tf.add(i,j)).sample((n_samples,1))
                    elif method == "sobol":
                        sampler = qmc.Sobol(d=1)
                        sample_data = sampler.random(n=n_samples)
                    elif method == "lhs":
                        sampler = qmc.LatinHypercube(d=1)
                        sample_data = sampler.random(n=n_samples)

                    # Inverse transform
                    if method == "sobol" or method == "lhs":
                        sample_dat = tf.cast(tf.convert_to_tensor(sample_data),
                                             tf.float32)
                        uniform_samples = tfd.Uniform(
                            tf.subtract(i,j), tf.add(i,j)).quantile(
                                tf.squeeze(sample_dat, -1))

                    res_dict[n] = tf.stack(uniform_samples, axis=-1)

    return res_dict


def initialization_phase(
    expert_elicited_statistics, initializer, parameters, trainer, model,
    targets, network, expert):
    """
    For the method "parametric_prior" it might be helpful to run different
    initializations before the actual training starts in order to find a
    'good' set of initial values. For this purpose the burnin phase can be
    used. It rans multiple initializations and computes for each the
    respective loss value. At the end that set of initial values is chosen
    which leads to the smallest loss.

    Parameters
    ----------
    expert_elicited_statistics : dict
        dictionary with expert elicited statistics.
    one_forward_simulation : callable
        one forward simulation from prior samples to model-simulated elicited
        statistics.
    compute_loss : callable
        wrapper for loss computation from loss components to (weighted) total
        loss.
    global_dict : dict
        global dictionary with all user input specifications.

    Returns
    -------
    loss_list : list
        list containing the loss values for each set of initial values.
    init_var_list : list
        set of initial values for each run.

    """

    seed=tf.identity(trainer["seed"])
    tf.random.set_seed(seed)

    loss_list = []
    init_var_list = []
    save_prior = []

    # create initializations
    init_matrix = init_method(
        seed,
        initializer["distribution"]["hyper"],
        initializer["iterations"],
        initializer["method"],
        initializer["distribution"]["mean"],
        initializer["distribution"]["radius"],
        parameters
        )

    if trainer["output_path"] is not None:
        path = trainer["output_path"] + "/initialization_matrix.pkl"
        el.helpers.save_as_pkl(init_matrix, path)

    print("Initialization")

    for i in tqdm(range(initializer["iterations"])):
        seed = seed+1
        # create init-matrix-slice
        init_matrix_slice = {f"{key}": init_matrix[key][i] for key in init_matrix}

        # prepare generative model
        prior_model = el.simulations.Priors(
            ground_truth=False,
            init_matrix_slice=init_matrix_slice,
            trainer=trainer, 
            parameters=parameters,
            network=network,
            expert=expert,
            seed=seed
            )

        # generate simulations from model
        (training_elicited_statistics,
         *_) = el.utils.one_forward_simulation(prior_model, trainer, model,
                                               targets)

        # compute loss for each set of initial values
        weighted_total_loss = el.losses.compute_loss(
            training_elicited_statistics,
            expert_elicited_statistics,
            epoch=0,
            targets=targets,
            output_path=trainer["output_path"]
        )

        init_var_list.append(prior_model)
        save_prior.append(prior_model.trainable_variables)
        loss_list.append(weighted_total_loss.numpy())

    if trainer["output_path"] is not None:
        path = trainer["output_path"] + "/pre_training_results.pkl"
        el.helpers.save_as_pkl((loss_list, save_prior), path)

    print(" ")
    return loss_list, init_var_list, init_matrix


def pre_training(expert_elicited_statistics, initializer, parameters, trainer,
                 model, targets, network, expert):
    logger = logging.getLogger(__name__)

    if trainer["method"] == "parametric_prior":

        logger.info("Pre-training phase (only first run)")

        loss_list, init_prior, init_matrix = initialization_phase(
            expert_elicited_statistics, initializer, parameters, trainer,
            model, targets, network, expert
        )

        # extract pre-specified quantile loss out of all runs
        # get corresponding set of initial values
        loss_quantile = initializer["loss_quantile"]
        index = tf.squeeze(tf.where(loss_list == tfp.stats.percentile(
            loss_list, [loss_quantile])))

        init_prior_model = init_prior[int(index)]
    else:
        # prepare generative model
        init_prior_model = el.simulations.Priors(
            ground_truth=False,
            init_matrix_slice=None,
            trainer=trainer, 
            parameters=parameters,
            network=network,
            expert=expert)

    return init_prior_model, loss_list, init_prior, init_matrix


def uniform(radius: list or float=1., mean: list or float=0.,
            hyper: list or None=None):
    """
    Specification of uniform distribution used for drawing initial values for
    each hyperparameter. Initial values are drawn from a uniform distribution
    ranging from ``mean-radius`` to ``mean+radius``.

    Parameters
    ----------
    radius : float or list
        Initial values are drawn from a uniform distribution
        ranging from ``mean-radius`` to ``mean+radius``.
        If a float is provided the same setting will be used for all hyperparameters.
        If different settings per hyperparameter are required a list of length
        equal to the number of hyperparameters should be provided.
        The order of values should be equivalent to the order of hyperparameter
        names provided in **hyper**.
        The default is ``1.``.
    mean : float or list
        Initial values are drawn from a uniform distribution
        ranging from ``mean-radius`` to ``mean+radius``.
        If a float is provided the same setting will be used for all hyperparameters.
        If different settings per hyperparameter are required a list of length
        equal to the number of hyperparameters should be provided.
        The order of values should be equivalent to the order of hyperparameter
        names provided in **hyper**.
        The default is ``0.``.
    hyper : None or list, optional
        List of hyperparameter names as specified in :func:`hyper`. The values
        provided in **radius** and **mean** should follow the order
        of hyperparameters indicated in this list.
        If a float is passed to **radius** and **mean** this argument is not
        necessary.
        The default is ``None``.

    Returns
    -------
    init_dict : dict
        Dictionary with all seetings of the uniform distribution used for
        initializing the hyperparameter values.

    """
    init_dict=dict(
        radius=radius,
        mean=mean,
        hyper=hyper
        )

    return init_dict