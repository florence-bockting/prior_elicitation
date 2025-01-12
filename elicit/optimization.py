# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import time
import numpy as np
import pandas as pd
import logging
import os
import elicit as el

from tqdm import tqdm

tfd = tfp.distributions


def sgd_training(
    expert_elicited_statistics,
    prior_model_init,
    trainer,
    optimizer,
    model,
    targets
):
    """
    Wrapper that runs the optimization algorithms for E epochs.

    Parameters
    ----------
    expert_elicited_statistics : dict
        expert data or simulated data representing a prespecified ground truth.
    prior_model_init : class instance
        instance of a class that initializes and samples from the prior
        distributions.
    one_forward_simulation : callable
        one forward simulation cycle including: sampling from priors,
        simulating model
        predictions, computing target quantities and elicited statistics.
    compute_loss : callable
        sub-dag to compute the loss value including: compute loss components
        of model simulations and expert data, compute loss per component,
        compute total loss.
    global_dict : dict
        dictionary including all user-input settings.

    """
    # set seed
    tf.random.set_seed(trainer["seed"])

    # prepare generative model
    prior_model = prior_model_init
    total_loss = []
    component_losses = []
    gradients_ep = []
    time_per_epoch = []

    # save files in folder or in temporary location
    if trainer["output_path"] is not None:
        saving_path = trainer["output_path"]
    else:
        saving_path = "elicit_temp"

    # initialize the adam optimizer
    optimizer_copy = optimizer.copy()
    init_sgd_optimizer = optimizer["optimizer"]
    optimizer_copy.pop("optimizer")
    sgd_optimizer = init_sgd_optimizer(**optimizer_copy)

    # start training loop
    print("Training")
    for epoch in tqdm(tf.range(trainer["epochs"])):
        if epoch > 0:
            logging.disable(logging.INFO)
        # runtime of one epoch
        epoch_time_start = time.time()

        with tf.GradientTape() as tape:
            # generate simulations from model
            (train_elicits, prior_sim,
             model_sim, target_quants) = el.utils.one_forward_simulation(
                prior_model, trainer, model, targets
            )
            # compute total loss as weighted sum
            weighted_total_loss = el.losses.compute_loss(
                train_elicits,
                expert_elicited_statistics,
                epoch,
                targets,
                trainer["output_path"]
            )

            # compute gradient of loss wrt trainable_variables
            gradients = tape.gradient(
                weighted_total_loss, prior_model.trainable_variables
            )
            # update trainable_variables using gradient info with adam
            # optimizer
            sgd_optimizer.apply_gradients(
                zip(gradients, prior_model.trainable_variables)
                )

        # time end of epoch
        epoch_time_end = time.time()
        epoch_time = epoch_time_end - epoch_time_start

        # break for loop if loss is NAN and inform about cause
        if tf.math.is_nan(weighted_total_loss):
            print("Loss is NAN. The training process has been stopped.")
            break

        if trainer["method"] == "parametric_prior":
            if trainer["output_path"] is not None:
                path = saving_path + "/gradients.pkl"
                el.helpers.save_as_pkl(gradients, path)
            # save for each epoch
            gradients_ep.append(gradients)

        # savings per epoch
        time_per_epoch.append(epoch_time)
        total_loss.append(weighted_total_loss)
        component_losses.append(
            pd.read_pickle(saving_path + "/loss_per_component.pkl")
            )

        if trainer["method"] == "parametric_prior":
            # save single learned hyperparameter values for each prior and
            # epoch
            res_dict = el.helpers.save_hyperparameters(
                prior_model, epoch, trainer["output_path"]
                )
        else:
            # save mean and std for each sampled marginal prior; for each epoch
            res_dict = el.helpers.marginal_prior_moments(
                model_sim["prior_samples"], epoch, trainer["output_path"]
            )

    res_ep = {
        "loss": total_loss,
        "loss_component": component_losses,
        "time": time_per_epoch,
        "hyperparameter": res_dict
    }

    output_res = {
        "target_quantities": target_quants,
        "elicited_statistics": train_elicits,
        "prior_samples": prior_sim,
        "model_samples": model_sim,
        "model": prior_model
        }

    if trainer["method"] == "parametric_prior":
        res_ep["hyperparameter_gradient"] = gradients_ep
    path = saving_path + "/final_results.pkl"
    el.helpers.save_as_pkl(res_ep, path)

    if trainer["output_path"] is None:
        os.remove(saving_path + "/final_results.pkl")
        os.remove(saving_path + "/loss_per_component.pkl")
        os.remove(saving_path + "/res_dict.pkl")
        os.rmdir(saving_path)

    return res_ep, output_res
