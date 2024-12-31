# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import bayesflow as bf
import inspect
import pandas as pd
import elicit as el

from elicit.configs import *
from elicit.user.custom_functions import custom_correlation

tfd = tfp.distributions
bfn = bf.networks

# TODO: Update Custom Target Function
def use_custom_functions(custom_function, model_simulations, global_dict):
    """
    Helper function that prepares custom functions if specified by checking
    all inputs and extracting the argument from different sources.

    Parameters
    ----------
    custom_function : callable
        custom function as specified by the user.
    model_simulations : dict
        simulations from the generative model.
    global_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    custom_quantity : tf.Tensor
        returns the evaluated custom function.

    """
    # get function
    custom_func = custom_function["function"]
    # create a dict with arguments from model simulations and custom args
    # for custom func
    args_dict = dict()
    if custom_function["additional_args"] is not None:
        additional_args_dict = {
            f"{key}": custom_function["additional_args"][key]
            for key in list(custom_function["additional_args"].keys())
        }
    else:
        additional_args_dict = {}
    # select only relevant keys from args_dict
    custom_args_keys = inspect.getfullargspec(custom_func)[0]
    # check whether expert-specific input has been specified
    if "from_simulated_truth" in custom_args_keys:
        for i in range(len(inspect.getfullargspec(custom_func)[3][0])):
            quantity = inspect.getfullargspec(custom_func)[3][i][0]
            true_model_simulations = pd.read_pickle(
                global_dict["output_path"] + "/expert/model_simulations.pkl"
            )
            for key in custom_args_keys:
                if f"{key}" == quantity:
                    args_dict[key] = true_model_simulations[quantity]
                    custom_args_keys.remove(quantity)
        custom_args_keys.remove("from_simulated_truth")
    # TODO: check that all args needed for custom function were detected
    for key in list(set(custom_args_keys) - set(additional_args_dict)):
        args_dict[key] = model_simulations[key]
    for key in additional_args_dict:
        args_dict.update(additional_args_dict)
    # evaluate custom function
    custom_quantity = custom_func(**args_dict)
    return custom_quantity


def computation_elicited_statistics(
        target_quantities: dict, ground_truth: bool, global_dict: dict):
    """
    Computes the elicited statistics from the target quantities by applying a
    prespecified elicitation technique.

    Parameters
    ----------
    target_quantities : dict
        simulated target quantities.
    ground_truth : bool
        True if target quantities are used for oracle.
    global_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    elicits_res : dict
        simulated elicited statistics.

    """
    logger = logging.getLogger(__name__)
    if ground_truth:
        logger.info("Compute true elicited statistics")
    else:
        logger.info("Compute elicited statistics")

    # create sub-dictionaries for readability
    target_dict = global_dict["target_quantities"]
    # initialize dict for storing results
    elicits_res = dict()
    # loop over elicitation techniques
    for i in range(len(target_dict)):
        # use custom method if specified otherwise use built-in methods
        if target_dict[i]["elicitation_method"]["name"] == "custom":
            elicited_statistic = use_custom_functions(
                target_dict[i]["elicitation_method"]["value"],
                target_quantities,
                global_dict,
            )
            elicits_res[f"custom_{target_dict[i]['name']}"] = elicited_statistic

        if target_dict[i]["elicitation_method"]["name"] == "identity":
            elicits_res[f"identity_{target_dict[i]['name']}"
                        ] = target_quantities[target_dict[i]['name']]

        if target_dict[i]["elicitation_method"]["name"] == "pearson_correlation":
            # compute correlation between model parameters (used for
            # learning correlation structure of joint prior)
            elicited_statistic = custom_correlation(
                target_quantities[target_dict[i]['name']])
            # save correlation in result dictionary
            elicits_res[f"pearson_{target_dict[i]['name']}"
                        ] = elicited_statistic

        if target_dict[i]["elicitation_method"]["name"] == "quantiles":
            quantiles = target_dict[i]["elicitation_method"]["value"]

            # reshape target quantity
            if tf.rank(target_quantities[target_dict[i]['name']]) == 3:
                quan_reshaped = tf.reshape(
                    target_quantities[target_dict[i]['name']],
                    shape=(
                        target_quantities[target_dict[i]['name']].shape[0],
                        target_quantities[target_dict[i]['name']].shape[1]
                        * target_quantities[target_dict[i]['name']].shape[2],
                    ),
                )
            if tf.rank(target_quantities[target_dict[i]['name']]) == 2:
                quan_reshaped = target_quantities[target_dict[i]['name']]

            # compute quantiles
            computed_quantiles = tfp.stats.percentile(
                quan_reshaped, q=quantiles, axis=-1
            )
            # bring quantiles to the last dimension
            elicited_statistic = tf.einsum("ij...->ji...",
                                           computed_quantiles)
            elicits_res[f"quantiles_{target_dict[i]['name']}"] = elicited_statistic

    # save file in object
    saving_path = global_dict["training_settings"]["output_path"]
    if saving_path is not None:
        if ground_truth:
            saving_path = saving_path + "/expert"
        path = saving_path + "/elicited_statistics.pkl"
        el.save_as_pkl(elicits_res, path)

    # return results
    return elicits_res
