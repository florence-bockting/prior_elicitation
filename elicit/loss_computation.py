# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import bayesflow as bf
import elicit as el
import logging

from elicit.configs import *  # noqa

tfd = tfp.distributions
bfn = bf.networks


def compute_loss_components(elicited_statistics, glob_dict, expert):
    """
    Computes the single loss components used for computing the discrepancy
    between the elicited statistics. This computation depends on the
    method as specified in the 'combine-loss' argument.

    Parameters
    ----------
    elicited_statistics : dict
        dictionary including the elicited statistics.
    glob_dict : dict
        dictionary including all user-input settings.
    expert : bool
        if workflow is run to simulate a pre-specified ground truth; expert is
        set as 'True'. As consequence the files are saved in a special 'expert'
        folder.

    Returns
    -------
    loss_comp_res : dict
        dictionary including all loss components which will be used to compute
        the discrepancy.

    """
    logger = logging.getLogger(__name__)
    if expert:
        logger.info("preprocess expert elicited statistics")
    else:
        logger.info("preprocess simulated statistics")

    # extract names from elicited statistics
    name_elicits = list(elicited_statistics.keys())


    # prepare dictionary for storing results
    loss_comp_res = dict()
    # initialize some helpers for keeping track of target quantity
    target_control = []
    i_target = 0
    eval_target = True
    # loop over elicited statistics
    for i, name in enumerate(name_elicits):
        # get name of target quantity
        target = name.split(sep="_")[-1]
        if i != 0:
            # check whether elicited statistic correspond to same target
            # quantity
            eval_target = target_control[-1] == target
        # append current target quantity
        target_control.append(target)
        # if target quantity changes go with index one up
        if not eval_target:
            i_target += 1
        # extract loss component
        loss_comp = elicited_statistics[name]

        assert tf.rank(loss_comp) <= 2, "elicited statistics can only have 2 dimensions."  # noqa

        if tf.rank(loss_comp) == 1:
            # add a last axis for loss computation
            final_loss_comp = tf.expand_dims(loss_comp, axis=-1)
            # store result
            loss_comp_res[f"{name}_loss"] = final_loss_comp
        else:
            loss_comp_res[f"{name}_loss_{i_target}"] = loss_comp

    # save file in object
    saving_path = glob_dict["training_settings"]["output_path"]
    if saving_path is not None:
        if expert:
            saving_path = saving_path + "/expert"
        path = saving_path + "/loss_components.pkl"
        el.save_as_pkl(loss_comp_res, path)
        # return results
    return loss_comp_res


def dynamic_weight_averaging(
    epoch,
    loss_per_component_current,
    loss_per_component_initial,
    task_balance_factor,
    saving_path,
):
    """DWA determines the weights based on the learning speed of each component

    The Dynamic Weight Averaging (DWA) method proposed by
    Liu, Johns, & Davison (2019) determines the weights based on the learning
    speed of each component, aiming to achieve a more balanced learning
    process. Specifically, the weight of a component exhibiting a slower
    learning speed is increased, while it is decreased for faster learning
    components.

    Liu, S., Johns, E., & Davison, A. J. (2019). End-To-End Multi-Task \
        Learning With Attention. In IEEE/CVF Conference on Computer Vision \
            and Pattern Recognition (CVPR) (pp. 1871–1880). \
                doi: https://doi.org/10.1109/CVPR.2019.00197

    Parameters
    ----------
    epoch : int
        How often should the hyperparameter values be updated?
    loss_per_component_current : list of floats
        List of loss values per loss component for the current epoch.
    loss_per_component_initial : list of floats
        List of loss values per loss component for the initial epoch
        (epoch = 0).
    task_balance_factor : float
        temperature parameter that controls the softness of the loss weighting
        in the softmax operator. Setting the temperature 𝑎 to a large value
        results in the weights approaching unity.

    Returns
    -------
    total_loss : float
        Weighted sum of all loss components. Loss used for gradient
        computation.
    weight_loss_comp : list of floats
        List of computed weight values per loss component for current epoch.

    """
    # get number of loss components
    num_loss_comps = len(loss_per_component_current)

    # initialize weights
    if epoch < 2:
        rel_weight_descent = tf.ones(num_loss_comps)
    # w_t (epoch-1) = L_t (epoch-1) / L_t (epoch_0)
    else:
        rel_weight_descent = tf.math.divide(
            loss_per_component_current, loss_per_component_initial
        )

    # T*exp(w_t(epoch-1)/a)
    numerator = tf.math.multiply(
        tf.cast(num_loss_comps, dtype=tf.float32),
        tf.exp(tf.math.divide(rel_weight_descent, task_balance_factor)),
    )

    # softmax operator
    weight_loss_comp = tf.math.divide(numerator, tf.math.reduce_sum(numerator))

    # total loss: L = sum_t lambda_t*L_t
    weighted_total_loss = tf.math.reduce_sum(
        tf.math.multiply(weight_loss_comp, loss_per_component_current)
    )
    # save file in object
    path = saving_path + "/weighted_total_loss.pkl"
    el.save_as_pkl(weighted_total_loss, path)
    return weighted_total_loss


def compute_discrepancy(loss_components_expert, loss_components_training,
                        global_dict):
    """
    Computes the discrepancy between all loss components using a specified
    discrepancy measure and returns a list with all loss values.

    Parameters
    ----------
    loss_components_expert : dict
        dictionary including all loss components derived from the
        expert-elicited statistics.
    loss_components_training : dict
        dictionary including all loss components derived from the model
        simulations. (The names (keys) between loss_components_expert and \
                      loss_components_training must match)
    glob_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    loss_per_component : list
        list of loss value for each loss component

    """
    logger = logging.getLogger(__name__)
    logger.info("compute discrepancy")
    # create dictionary for storing results
    loss_per_component = []
    # extract expert loss components by name
    keys_loss_comps = list(loss_components_expert.keys())
    # compute discrepancy
    for i, name in enumerate(keys_loss_comps):
        # import loss function
        loss_function = global_dict["target_quantities"][i]["loss"]
        # broadcast expert loss to training-shape
        loss_comp_expert = tf.broadcast_to(
            loss_components_expert[name],
            shape=(
                loss_components_training[name].shape[0],
                loss_components_expert[name].shape[1],
            ),
        )
        # compute loss
        loss = loss_function(loss_comp_expert, loss_components_training[name])
        loss_per_component.append(loss)

    # save file in object
    if global_dict["training_settings"]["output_path"] is not None:
        saving_path = global_dict["training_settings"]["output_path"]
    else:
        saving_path = "elicit_temp"
    path = saving_path + "/loss_per_component.pkl"
    el.save_as_pkl(loss_per_component, path)
    return loss_per_component


def compute_total_loss(
    training_elicited_statistics, expert_elicited_statistics, global_dict,
    epoch
):
    """
    Wrapper around the loss computation from elicited statistics to final
    loss value.

    Parameters
    ----------
    training_elicited_statistics : dict
        dictionary containing the expert elicited statistics.
    expert_elicited_statistics : dict
        dictionary containing the model-simulated elicited statistics.
    global_dict : dict
        global dictionary with all user input specifications.
    epoch : int
        epoch .

    Returns
    -------
    total_loss : float
        total loss value.

    """
    # regularization term for preventing degenerated solutions in var
    # collapse to zero used from Manderson and Goudie (2024)
    def regulariser(prior_samples):
        """
        Regularizer term for loss function: minus log sd of each prior
        distribution (priors with larger sds should be prefered)

        Parameters
        ----------
        prior_samples : tf.Tensor
            samples from prior distributions.

        Returns
        -------
        float
            the negative mean log std across all prior distributions.

        """
        log_sd = tf.math.log(tf.math.reduce_std(prior_samples, 1))
        mean_log_sd = tf.reduce_mean(log_sd)
        return -mean_log_sd

    def compute_total_loss(epoch, loss_per_component, global_dict):
        """
        applies dynamic weight averaging for multi-objective loss function
        if specified. If loss_weighting has been set to None, all weights
        get an equal weight of 1.

        Parameters
        ----------
        epoch : int
            curernt epoch.
        loss_per_component : list of floats
            list of loss values per loss component.
        global_dict : dict
            global dictionary with all user input specifications.

        Returns
        -------
        total_loss : float
            total loss value (either weighted or unweighted).

        """
        logger = logging.getLogger(__name__)
        logger.info("compute total loss")
        # loss_per_component_current = loss_per_component
        # TODO: check whether order of loss_per_component and target quantities
        # is equivalent!
        total_loss=0
        # create subdictionary for better readability
        for i in range(len(global_dict["target_quantities"])):
            total_loss += tf.multiply(
                loss_per_component[i],
                global_dict["target_quantities"][i]["loss_weight"]
                )

        # TODO: include loss_balancing
        # # apply selected loss weighting method
        # # TODO: Tests and Checks
        # if dict_loss["loss_weighting"]["method"] == "dwa":
        #     # dwa needs information about the initial loss per component
        #     if epoch == 0:
        #         dict_loss["loss_weighting"]["method_specs"][
        #             "loss_per_component_initial"
        #         ] = loss_per_component
        #     # apply method
        #     total_loss = dynamic_weight_averaging(
        #         epoch,
        #         loss_per_component_current,
        #         dict_loss["loss_weighting"]["method_specs"][
        #             "loss_per_component_initial"
        #         ],
        #         dict_loss["loss_weighting"]["method_specs"]["temperature"],
        #         global_dict["output_path"],
        #     )

        # if dict_loss["loss_weighting"]["method"] == "custom":
        #     total_loss = tf.math.reduce_sum(
        #         tf.multiply(
        #             loss_per_component,
        #             dict_loss["loss_weighting"]["weights"],
        #         )
        #     )

        return total_loss

    loss_components_expert = compute_loss_components(
        expert_elicited_statistics, global_dict, expert=True
    )
    loss_components_training = compute_loss_components(
        training_elicited_statistics, global_dict, expert=False
    )
    loss_per_component = compute_discrepancy(
        loss_components_expert, loss_components_training, global_dict
    )
    weighted_total_loss = compute_total_loss(epoch, loss_per_component,
                                             global_dict)

    return weighted_total_loss
