# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import logging
import tensorflow as tf
import tensorflow_probability as tfp
import elicit as el

from elicit.configs import *  # noqa

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
        tf.random.set_seed(global_dict["trainer"]["seed"])
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
    tf.random.set_seed(global_dict["trainer"]["seed"])

    if global_dict["trainer"]["method"] == "parametric_prior":

        # create dict with all hyperparameters
        hyp_dict = dict()
        hp_keys=list()
        param_names=list()
        hp_names=list()
        initialized_hyperparam = dict()
        for i in range(len(global_dict["parameters"])):
            hyp_dict[f"param{i}"] = global_dict["parameters"][i]["hyperparams"]
            param_names += [global_dict["parameters"][i]["name"]]*len(global_dict["parameters"][i]["hyperparams"])
            hp_keys += list(global_dict["parameters"][i]["hyperparams"].keys())
            for j in range(len(global_dict["parameters"][i]["hyperparams"])):
                current_key = list(global_dict["parameters"][i]["hyperparams"].keys())[j]
                hp_names.append(global_dict["parameters"][i]["hyperparams"][current_key]["name"])

        checked_params=list()
        for j, (i, hp_n, hp_k) in enumerate(zip(tf.unique(param_names).idx,
                                 hp_names, hp_keys)):

            hp_dict = global_dict["parameters"][i]["hyperparams"][hp_k]
            
            if hp_dict["shared"] and hp_dict["name"] in checked_params:
                pass
                #initialized_hyperparam[f"{hp_k}"] = initialized_hyperparam[f"{hp_k}"]
            else:
                # get initial value
                initial_value = init_matrix_slice[hp_n]
                # initialize hyperparameter
                initialized_hyperparam[f"{hp_k}_{hp_n}"] = tf.Variable(
                    initial_value=hp_dict["constraint"](initial_value),
                    trainable=True,
                    name=f"{hp_n}",
                )

                # save initialized priors
                init_prior = initialized_hyperparam
                
            if hp_dict["shared"]:
                checked_params.append(hp_n)

            # save file in object
            if global_dict["trainer"]["output_path"] is not None:
                output_path = global_dict["trainer"]["output_path"]
                path = output_path + "/init_hyperparameters.pkl"
                el.save_as_pkl(init_prior, path)

    if global_dict["trainer"]["method"] == "deep_prior":
        # for more information see BayesFlow documentation
        # https://bayesflow.org/api/bayesflow.inference_networks.html
        INN = global_dict["normalizing_flow"]["inference_network"]

        invertible_neural_network = INN(
            **global_dict["normalizing_flow"]["network_specs"]
        )

        # save initialized priors
        init_prior = invertible_neural_network

        # save file in object
        if global_dict["trainer"]["output_path"] is not None:
            output_path = global_dict["trainer"]["output_path"]
            path = output_path + "/init_hyperparameters.pkl"
            el.save_as_pkl(init_prior.trainable_variables, path)

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
    S = global_dict["trainer"]["num_samples"]
    B = global_dict["trainer"]["B"]
    # set seed
    tf.random.set_seed(global_dict["trainer"]["seed"])

    if ground_truth:

        # number of samples for ground truth
        rep_true = global_dict["expert"]["num_samples"]
        priors = []

        for prior in list(
                global_dict["expert"]["ground_truth"].values()
                ):
            # sample from the prior distribution
            priors.append(prior.sample((1, rep_true)))

        # this is a workaround for the changed shape when a
        # multivariate prior is used
        if type(priors[0]) is list:
            prior_samples = tf.concat(priors[0], axis=-1)
        elif tf.rank(priors[0]) > 2:
            prior_samples = tf.concat(priors, axis=-1)
        else:
            prior_samples = tf.stack(priors, axis=-1)

    if (global_dict["trainer"]["method"] == "parametric_prior") and (
        not ground_truth
    ):

        priors = []
        for i in range(len(global_dict["parameters"])):
            # get the prior distribution family as specified by the user
            prior_family = global_dict["parameters"][i]["family"]

            hp_k=list(global_dict["parameters"][i]["hyperparams"].keys())
            init_dict={}
            for k in hp_k:
                hp_n=global_dict["parameters"][i]["hyperparams"][k]["name"]
                init_key = f"{k}_{hp_n}"
                init_dict[f"{k}"]=initialized_priors[init_key]
                
            # sample from the prior distribution
            priors.append(
                prior_family(**init_dict).sample((B, S))
                )
        # stack all prior distributions into one tf.Tensor of
        # shape (B, S, num_parameters)
        prior_samples = tf.stack(priors, axis=-1)

    if (global_dict["trainer"]["method"] == "deep_prior") and (
        not ground_truth
    ):

        # initialize base distribution
        base_dist = global_dict["normalizing_flow"]["base_distribution"]
        # sample from base distribution
        u = base_dist.sample((B, S))
        # apply transformation function to samples from base distr.
        prior_samples, _ = initialized_priors(u, condition=None, inverse=False)

    # save results
    saving_path = global_dict["trainer"]["output_path"]
    if saving_path is not None:
        if ground_truth:
            el.save_as_pkl(prior_samples, saving_path + "/expert/prior_samples.pkl")
        else:
            el.save_as_pkl(prior_samples, saving_path + "/prior_samples.pkl")

    return prior_samples
