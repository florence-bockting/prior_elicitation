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
def use_custom_functions():
    """
    ToDo: Function for using custom target_method

    """
    raise NotImplementedError(
        "Tue usage of custom target_method is not implemented yet.")


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
        list of target quantities specified with :func:`elicit.elicit.target`.

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
            elicits_res[f"cor_{targets[i]['name']}"
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


def computation_target_quantities(
        model_simulations: dict[str,tf.Tensor],
        prior_samples: tf.Tensor, # shape=[B,rep,num_param]
        targets: dict) -> dict[str, tf.Tensor]:
    """
    Computes target quantities from model simulations.

    Parameters
    ----------
    model_simulations : dict[str, tf.Tensor]
        simulations from generative model.
    prior_samples : tf.Tensor; shape = [B, rep, num_params]
        samples from prior distributions of model parameters. Currently only
        needed if correlations between model parameters is used as elicitation
        technique.
    targets : list[dict]
        list of target quantities specified with :func:`elicit.elicit.target`.

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

        if tar["query"]["name"] == "pearson_correlation":
            target_quantity = prior_samples
        elif (
            tar["target_method"] is not None
        ):
            target_quantity = use_custom_functions()
        else:
            target_quantity = model_simulations[tar["name"]]

        # save target quantities
        targets_res[tar["name"]] = target_quantity

    return targets_res