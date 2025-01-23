# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import bayesflow as bf
import inspect
import pandas as pd

from elicit.extras import utils

tfd = tfp.distributions
bfn = bf.networks


# TODO: Update Custom Target Function
def use_custom_functions(custom_function, model_simulations):
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
                output_path + "/expert/model_simulations.pkl"
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
        target_quantities: dict[str, tf.Tensor],  # shape=[B, num_samples, num_obs]
        targets: list[dict]
        ) -> dict[str, tf.Tensor]:  # shape=[B, num_stats]
    """
    Computes the elicited statistics from the target quantities by applying a
    prespecified elicitation technique.

    Parameters
    ----------
    target_quantities : dict[str, tf.Tensor], shape: [B,num_samples,num_obs]
        simulated target quantities.
    targets : list[dict]
        list of target quantities specified with :func:`target`.

    Returns
    -------
    elicits_res : dict[res, tf.Tensor], shape: [B, num_stats]
        simulated elicited statistics.

    """
    # initialize dict for storing results
    elicits_res = dict()
    # loop over elicitation techniques
    for i in range(len(targets)):
        # use custom method if specified otherwise use built-in methods
        if targets[i]["query"]["name"] == "custom":
            elicited_statistic = use_custom_functions(
                targets[i]["elicitation_method"]["value"],
                target_quantities
            )
            elicits_res[f"custom_{targets[i]['name']}"] = elicited_statistic

        if targets[i]["query"]["name"] == "identity":
            elicits_res[f"identity_{targets[i]['name']}"
                        ] = target_quantities[targets[i]['name']]

        if targets[i]["query"]["name"] == "pearson_correlation":
            # compute correlation between model parameters (used for
            # learning correlation structure of joint prior)
            elicited_statistic = utils.pearson_correlation(
                target_quantities[targets[i]['name']])
            # save correlation in result dictionary
            elicits_res[f"pearson_{targets[i]['name']}"
                        ] = elicited_statistic

        if targets[i]["query"]["name"] == "quantiles":
            quantiles = targets[i]["query"]["value"]

            # reshape target quantity
            if tf.rank(target_quantities[targets[i]['name']]) == 3:
                quan_reshaped = tf.reshape(
                    target_quantities[targets[i]['name']],
                    shape=(
                        target_quantities[targets[i]['name']].shape[0],
                        target_quantities[targets[i]['name']].shape[1]
                        * target_quantities[targets[i]['name']].shape[2],
                    ),
                )
            if tf.rank(target_quantities[targets[i]['name']]) == 2:
                quan_reshaped = target_quantities[targets[i]['name']]

            # compute quantiles
            computed_quantiles = tfp.stats.percentile(
                quan_reshaped, q=quantiles, axis=-1
            )
            # bring quantiles to the last dimension
            elicited_statistic = tf.einsum("ij...->ji...",
                                           computed_quantiles)
            elicits_res[f"quantiles_{targets[i]['name']}"] = elicited_statistic

    # return results
    return elicits_res


def computation_target_quantities(model_simulations: dict[str,tf.Tensor],
                                  targets: dict) -> dict[str, tf.Tensor]:
    """
    Computes target quantities from model simulations.

    Parameters
    ----------
    model_simulations : dict[str, tf.Tensor]
        simulations from generative model.
    targets : list[dict]
        list of target quantities specified with :func:`target`.

    Returns
    -------
    targets_res : dict[str, tf.Tensor]
        computed target quantities.
    """
    # initialize dict for storing results
    targets_res = dict()
    # loop over target quantities
    for i in range(len(targets)):
        tar = targets[i]
        # use custom function for target quantity if it has been defined
        if tar["name"] == "correlation":
            target_quantity = model_simulations["prior_samples"]
        elif (
            tar["target_method"] is not None
        ):
            target_quantity = use_custom_functions(
                tar["target_method"],
                model_simulations
            )
        else:
            target_quantity = model_simulations[tar["name"]]

        # save target quantities
        targets_res[tar["name"]] = target_quantity

    return targets_res