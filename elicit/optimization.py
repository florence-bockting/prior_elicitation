# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import elicit as el
import time

from typing import Tuple
from tqdm import tqdm

tfd = tfp.distributions


def sgd_training(
    expert_elicited_statistics: dict[str,tf.Tensor],
    prior_model_init: callable,
    trainer: dict,
    optimizer: dict,
    model: dict,
    targets: dict,
    parameters: list[dict],
    seed: int
) -> Tuple[dict, dict]:
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
    parameters : list[dict]
        list of model parameters specified with
        :func:`elicit.elicit.parameter`.
    seed : int
        internally used seed for reproducible results

    Raises
    ------
    ValueError
        Training has been stopped because loss value is NAN.

    """
    # set seed
    tf.random.set_seed(seed)

    # prepare generative model
    prior_model = prior_model_init
    total_loss = []
    component_losses = []
    gradients_ep = []
    time_per_epoch = []
    
    if trainer["method"] == "parametric_prior":
        # save initialized trainable variables of epoch=0 (before first update)
        init_vars_values = [
            prior_model.trainable_variables[i].numpy().copy() for i in range(
                len(prior_model.trainable_variables))
            ]
        init_vars_names = [
            prior_model.trainable_variables[i].name[:-2].split(".")[1] for i 
            in range(len(prior_model.trainable_variables))
            ]

    # initialize the adam optimizer
    optimizer_copy = optimizer.copy()
    init_sgd_optimizer = optimizer["optimizer"]
    optimizer_copy.pop("optimizer")
    sgd_optimizer = init_sgd_optimizer(**optimizer_copy)

    # start training loop
    print("Training")
    for epoch in tqdm(tf.range(trainer["epochs"])):
        # runtime of one epoch
        epoch_time_start = time.time()

        with tf.GradientTape() as tape:
            # generate simulations from model
            (train_elicits, prior_sim, model_sim, target_quants
             ) = el.utils.one_forward_simulation(
                prior_model, trainer, model, targets, seed
            )
            # compute total loss as weighted sum
            (loss, indiv_losses, loss_components_expert,
             loss_components_training) = el.losses.total_loss(
                 train_elicits,
                 expert_elicited_statistics,
                 epoch,
                 targets
                 )

            # compute gradient of loss wrt trainable_variables
            gradients = tape.gradient(loss, prior_model.trainable_variables)

            # update trainable_variables using gradient info with adam
            # optimizer
            sgd_optimizer.apply_gradients(
                zip(gradients, prior_model.trainable_variables)
                )

        # time end of epoch
        epoch_time_end = time.time()
        epoch_time = epoch_time_end - epoch_time_start

        # break for loop if loss is NAN and inform about cause
        if tf.math.is_nan(loss):
            print("Loss is NAN and therefore training stops.")
            break
        #     raise ValueError(
        #         "Loss is NAN. The training process has been stopped."
        #         )

        # %% Saving of results
        if trainer["method"] == "parametric_prior":
            # create a list with constraints for re-transforming hyperparameter
            # before saving them
            constraint_dict=dict()

            for i in range(len(parameters)):
                hyp_dict = parameters[i]["hyperparams"]
                for hyp in hyp_dict:
                    constraint_dict[hyp_dict[hyp]["name"]]= hyp_dict[hyp]["constraint"]  # noqa

            # save gradients per epoch
            gradients_ep.append(gradients)

            # save learned hyperparameter values for each prior and epoch
            # extract learned hyperparameter values
            hyperparams = prior_model.trainable_variables
            if epoch == 0:
                # prepare list for saving hyperparameter values
                hyp_list = []
                for i in range(len(hyperparams)):
                    hyp_list.append(hyperparams[i].name[:-2].split(".")[1])
                # create a dict with empty list for each hyperparameter
                res_dict = {f"{k}": [] for k in hyp_list}
                # create final dict with initial train. variables
                for val, name in zip(init_vars_values, init_vars_names):
                    res_dict[name].append(val)
            # save names and values of hyperparameters
            vars_values = [
                hyperparams[i].numpy().copy() for i in range(len(hyperparams))
                ]
            vars_names = [
                hyperparams[i].name[:-2].split(".")[1] for i in range(
                    len(hyperparams))
                ]
            # create a final dict of hyperparameter values
            for val, name in zip(vars_values, vars_names):
                res_dict[name].append(float(constraint_dict[name](val)))

        if trainer["method"] == "deep_prior":
            # save mean and std for each sampled marginal prior for each epoch

            if epoch == 0:
                res_dict = {"means": [], "stds": []}

            means = tf.reduce_mean(prior_sim, (0, 1))
            sds = tf.reduce_mean(tf.math.reduce_std(prior_sim, 1), 0)

            for val, name in zip([means, sds], ["means", "stds"]):
                res_dict[name].append(val)

        # savings per epoch (independent from chosen method)
        time_per_epoch.append(epoch_time)
        total_loss.append(loss)
        component_losses.append(indiv_losses)

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
        "loss_tensor_expert": loss_components_expert,
        "loss_tensor_model": loss_components_training,
        }

    if trainer["method"] == "parametric_prior":
        res_ep["hyperparameter_gradient"] = gradients_ep

    return res_ep, output_res
