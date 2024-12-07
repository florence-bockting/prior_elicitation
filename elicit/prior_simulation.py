# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import logging
import tensorflow as tf
import tensorflow_probability as tfp
import elicit.logs_config # noqa

from bayesflow import networks
from elicit.helper_functions import save_as_pkl

tfd = tfp.distributions


# initalize generator model
class Priors(tf.Module):
    """
    Initializes the hyperparameters of the prior distributions.

    Attributes
    ----------
    ground_truth : bool
        whether samples are drawn from a true prior ('oracle')
    global_dict : dict
        dictionary containing all user and default input settings
    logger : logging method
        retrieves module name for passing it to the logger
    init_priors : dict
        initialized hyperparameters (i.e., trainable variables);
        None if ground_truth = True
    """

    def __init__(self, ground_truth, global_dict, init_matrix_slice):
        """
        Initializes the hyperparameters (i.e., trainable variables)

        Parameters
        ----------
        ground_truth : bool
            whether samples are drawn from a true prior ('oracle')
        global_dict : dict
            dictionary containing all user and default input settings.
        """

        self.global_dict = global_dict
        self.ground_truth = ground_truth
        self.init_matrix_slice = init_matrix_slice
        self.logger = logging.getLogger(__name__)
        # set seed
        tf.random.set_seed(global_dict["training_settings"]["seed"])
        # initialize hyperparameter for learning (if true hyperparameter
        # are given, no initialization is needed)
        if not self.ground_truth:
            self.logger.info("Initialize prior hyperparameters")
            self.init_priors = intialize_priors(self.global_dict,
                                                self.init_matrix_slice)
        else:
            self.logger.info("Set true prior hyperparameters")
            self.init_priors = None

    def __call__(self):
        """
        Samples from the initialized prior distribution(s).

        Returns
        -------
        prior_samples : dict
            Samples from prior distribution(s).

        """
        if self.ground_truth:
            self.logger.info("Sample from true prior(s)")
        else:
            self.logger.info("Sample from prior(s)")
        prior_samples = sample_from_priors(
            self.init_priors, self.ground_truth, self.global_dict
        )
        return prior_samples


def intialize_priors(global_dict, init_matrix_slice):
    """
    Initialize prior distributions.

    Parameters
    ----------
    global_dict : dict
        dictionary including all user-and default input settings.

    Returns
    -------
    init_prior : dict
        returns initialized prior distributions ready for sampling.

    """
    # set seed
    tf.random.set_seed(global_dict["training_settings"]["seed"])

    if global_dict["training_settings"]["method"] == "parametric_prior":
        # list for saving initialize hyperparameter values
        init_hyperparam_list = []
        # initialize j counting number of hyperparameters
        j = 0

        # loop over model parameter and initialize each hyperparameter
        for model_param in sorted(
            list(
                set(global_dict["model_parameters"].keys()).difference(
                    set(["independence", "no_params"])
                )
            )
        ):
            get_hyp_dict = global_dict["model_parameters"][model_param][
                "hyperparams_dict"
            ]

            initialized_hyperparam = dict()
            for name in get_hyp_dict:
                if global_dict["initialization_settings"][
                        "method"] is not None:
                    initial_value = init_matrix_slice[j]
                    # increase j
                    j += 1
                else:
                    initial_value = get_hyp_dict[name]

                # initialize hyperparameter
                initialized_hyperparam[f"{name}"] = tf.Variable(
                    initial_value=initial_value,
                    trainable=True,
                    name=f"{name}",
                )
            init_hyperparam_list.append(initialized_hyperparam)

        # save initialized priors
        init_prior = init_hyperparam_list
        # save file in object
        output_path = global_dict["training_settings"]["output_path"]
        path = output_path + "/init_hyperparameters.pkl"
        save_as_pkl(init_prior, path)

    if global_dict["training_settings"]["method"] == "deep_prior":
        # for more information see BayesFlow documentation
        # https://bayesflow.org/api/bayesflow.inference_networks.html
        input_INN = global_dict["normalizing_flow"]["coupling_flow"]

        invertible_neural_network = networks.InvertibleNetwork(
            num_params=global_dict["model_parameters"]["no_params"],
            **input_INN
        )
        # save initialized priors
        init_prior = invertible_neural_network
        # save file in object
        output_path = global_dict["training_settings"]["output_path"]
        path = output_path + "/init_hyperparameters.pkl"
        save_as_pkl(init_prior.trainable_variables, path)

    return init_prior


def sample_from_priors(initialized_priors, ground_truth, global_dict):
    """
    Samples from initialized prior distributions.

    Parameters
    ----------
    initialized_priors : dict
        initialized prior distributions ready for sampling.
    ground_truth : bool
        whether simulations are based on ground truth
        (then sampling is performed from true distribution).
    global_dict : dict
        dictionary including all user-input settings..

    Returns
    -------
    prior_samples : dict
        Samples from prior distributions.

    """
    # extract variables from dict
    S = global_dict["training_settings"]["samples_from_prior"]
    B = global_dict["training_settings"]["B"]
    # set seed
    tf.random.set_seed(global_dict["training_settings"]["seed"])

    if ground_truth:
        # number of samples for ground truth
        rep_true = global_dict["expert_data"]["samples_from_prior"]
        priors = []

        for prior in list(
                global_dict["expert_data"]["simulator_specs"].values()
                ):
            # sample from the prior distribution
            priors.append(prior.sample((1, rep_true)))

        # this is a workaround for the changed shape when a
        # multivariate prior is used
        if type(priors[0]) is list:
            prior_samples = tf.concat(priors[0], axis=-1)
        else:
            prior_samples = tf.stack(priors, axis=-1)

    if (global_dict["training_settings"]["method"] == "parametric_prior") and (
        not ground_truth
    ):

        priors = []
        for i, param in enumerate(
            sorted(
                list(
                    set(global_dict["model_parameters"].keys()).difference(
                        set(["independence", "no_params"])
                    )
                )
            )
        ):
            # get the prior distribution family as specified by the user
            prior_family = global_dict["model_parameters"][param]["family"]

            # sample from the prior distribution
            priors.append(
                prior_family(*initialized_priors[i].values()).sample((B, S))
                )
        # stack all prior distributions into one tf.Tensor of
        # shape (B, S, num_parameters)
        prior_samples = tf.stack(priors, axis=-1)

    if (global_dict["training_settings"]["method"] == "deep_prior") and (
        not ground_truth
    ):

        # initialize base distribution
        base_dist = global_dict["normalizing_flow"]["base_distribution"]
        # sample from base distribution
        u = base_dist.sample((B, S))
        # apply transformation function to samples from base distr.
        prior_samples, _ = initialized_priors(u, condition=None, inverse=False)

    # scaling of prior distributions according to param_scaling
    if not ground_truth:
        scaled_priors = []
        for i, param in enumerate(
            sorted(
                list(
                    set(global_dict["model_parameters"].keys()).difference(
                        set(["independence", "no_params"])
                    )
                )
            )
        ):
            factor = global_dict["model_parameters"][param]["param_scaling"]
            scaled_priors.append(prior_samples[:, :, i] * factor)

        prior_samples = tf.stack(scaled_priors, -1)

    # save results
    saving_path = global_dict["training_settings"]["output_path"]

    if ground_truth:
        save_as_pkl(prior_samples, saving_path + "/expert/prior_samples.pkl")
    else:
        save_as_pkl(prior_samples, saving_path + "/prior_samples.pkl")

    return prior_samples
