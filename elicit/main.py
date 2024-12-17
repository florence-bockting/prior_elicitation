# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import logging

import elicit.save_config
import elicit.logs_config # noqa

from elicit.prior_simulation import Priors
from elicit.loss_computation import compute_total_loss
from elicit.optimization_process import sgd_training
from elicit.initialization_methods import initialization_phase
from elicit.expert_data import get_expert_data
from elicit.helper_functions import remove_unneeded_files
from elicit.model_simulation import simulate_from_generator
from elicit.target_quantities import computation_target_quantities
from elicit.elicitation_techniques import computation_elicited_statistics

tfd = tfp.distributions


def one_forward_simulation(prior_model, global_dict, ground_truth=False):
    """
    One forward simulation from prior samples to elicited statistics.

    Parameters
    ----------
    prior_model : instance of Priors class objects
        initialized prior distributions which can be used for sampling.
    global_dict : dict
        global dictionary with all user input specifications.
    ground_truth : bool, optional
        Is true if model should be learned with simulated data that
        represent a pre-defined ground truth. The default is False.

    Returns
    -------
    elicited_statistics : dict
        dictionary containing the elicited statistics that can be used to
        compute the loss components

    """
    # set seed
    tf.random.set_seed(global_dict["training_settings"]["seed"])
    # generate samples from initialized prior
    prior_samples = prior_model()
    # simulate prior predictive distribution based on prior samples
    # and generative model
    model_simulations = simulate_from_generator(
        prior_samples, ground_truth, global_dict,
    )
    # compute the target quantities
    target_quantities = computation_target_quantities(
        model_simulations, ground_truth, global_dict
    )
    # compute the elicited statistics by applying a specific elicitation
    # method on the target quantities
    elicited_statistics = computation_elicited_statistics(
        target_quantities, ground_truth, global_dict
    )
    return elicited_statistics


def pre_training(global_dict, expert_elicited_statistics):
    logger = logging.getLogger(__name__)

    if global_dict["training_settings"]["method"] == "parametric_prior":

        logger.info("Pre-training phase (only first run)")

        loss_list, init_prior = initialization_phase(
            expert_elicited_statistics,
            one_forward_simulation,
            compute_total_loss,
            global_dict,
        )

        # extract pre-specified quantile loss out of all runs
        # get corresponding set of initial values
        loss_quantile = global_dict["initialization_settings"][
            "loss_quantile"]
        index = tf.squeeze(tf.where(loss_list == tfp.stats.percentile(
            loss_list, [loss_quantile])))
        init_prior_model = init_prior[int(index)]
    else:
        # prepare generative model
        init_prior_model = Priors(global_dict=global_dict,
                                  ground_truth=False,
                                  init_matrix_slice=None)

    return init_prior_model


def run(global_dict: dict):

    logger = logging.getLogger(__name__)

    # create saving path
    global_dict["training_settings"][
        "output_path"
    ] = f"./elicit/{global_dict['training_settings']['output_path']}/{global_dict['training_settings']['method']}/{global_dict['training_settings']['sim_id']}_{global_dict['training_settings']['seed']}"  # noqa

    # set seed
    tf.random.set_seed(global_dict["training_settings"]["seed"])

    # get expert data
    expert_elicited_statistics = get_expert_data(
        global_dict,
        one_forward_simulation)

    init_prior_model = pre_training(global_dict, expert_elicited_statistics)

    # run dag with optimal set of initial values
    logger.info("Training Phase (only first epoch)")
    sgd_training(
        expert_elicited_statistics,
        init_prior_model,
        one_forward_simulation,
        compute_total_loss,
        global_dict
    )

    # remove saved files that are not of interest for follow-up analysis
    remove_unneeded_files(global_dict, elicit.save_config.save_results)
