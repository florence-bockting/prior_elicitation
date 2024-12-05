# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import bayesflow as bf
import pandas as pd
import logging
import elicit.logs_config # noqa

from elicit.helper_functions import save_as_pkl
from elicit.loss_functions import norm_diff

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
    # extract the target quantities section from the glob dict
    target_glob_dict = glob_dict["target_quantities"]

    # extract names from elicited statistics
    name_elicits = list(elicited_statistics.keys())
    # for later check in if statement
    name_elicits_copy = name_elicits.copy()

    if glob_dict["model_parameters"]["independence"] is not None:
        if "correlation" in name_elicits:
            name_elicits.remove("correlation")
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

        if tf.rank(loss_comp) == 1:
            assert (
                target_glob_dict[target]["loss_components"] == "all"
            ), f"the elicited statistic {name} has rank=1 and can therefore \
                support only combine_loss = 'all'"
            # add a last axis for loss computation
            final_loss_comp = tf.expand_dims(loss_comp, axis=-1)
            # store result
            loss_comp_res[f"{name}_loss"] = final_loss_comp

        else:
            if target_glob_dict[target]["loss_components"] == "all":
                assert (
                    tf.rank(loss_comp) <= 3
                ), f"the elicited statistic {name} has more than 3 dimensions;\
                    combine_loss = all is therefore not possible. \
                        Consider using combine_loss = 'by-group'"
                if tf.rank(loss_comp) == 3:
                    loss_comp_res[f"{name}_loss_{i_target}"] = tf.reshape(
                        loss_comp,
                        (
                            loss_comp.shape[0],
                            loss_comp.shape[1] * loss_comp.shape[2],
                        ),
                    )
                if tf.rank(loss_comp) <= 2:
                    loss_comp_res[f"{name}_loss_{i_target}"] = loss_comp

            if target_glob_dict[target]["loss_components"] == "by-stats":
                assert (
                    target_glob_dict[target][
                        "elicitation_method"] == "quantiles"
                ), "loss combination method 'by-stats' is currently only \
                    possible for elicitation techniques: 'quantiles'."
                for j in range(loss_comp.shape[1]):
                    if tf.rank(loss_comp) == 2:
                        loss_comp_res[f"{name}_loss_{j}"] = loss_comp[:, j]
                    if tf.rank(loss_comp) == 3:
                        loss_comp_res[f"{name}_loss_{j}"] = loss_comp[:, j, :]

            if target_glob_dict[target]["loss_components"] == "by-group":
                for j in range(loss_comp.shape[-1]):
                    final_loss_comp = loss_comp[..., j]
                    if tf.rank(final_loss_comp) == 1:
                        final_loss_comp = tf.expand_dims(
                            final_loss_comp, axis=-1
                        )

                    loss_comp_res[f"{name}_loss_{j}"] = final_loss_comp

    if glob_dict["model_parameters"]["independence"] is not None:
        if "correlation" in name_elicits_copy:
            loss_comp = elicited_statistics["correlation"]
            for j in range(loss_comp.shape[-1]):
                correl_loss_comp = loss_comp[:, j]
                if tf.rank(correl_loss_comp) == 1:
                    correl_loss_comp = tf.expand_dims(
                        correl_loss_comp, axis=-1
                    )
                loss_comp_res[f"correlation_loss_{j}"] = correl_loss_comp

    # save file in object
    saving_path = glob_dict["training_settings"]["output_path"]
    if expert:
        saving_path = saving_path + "/expert"
    path = saving_path + "/loss_components.pkl"
    save_as_pkl(loss_comp_res, path)
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
    save_as_pkl(weighted_total_loss, path)
    return weighted_total_loss


def compute_discrepancy(loss_components_expert, loss_components_training,
                        glob_dict):
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
    # import loss function
    loss_function = glob_dict["loss_function"]["loss"]
    # create dictionary for storing results
    loss_per_component = []
    # extract expert loss components by name
    keys_loss_comps = list(loss_components_expert.keys())
    if glob_dict["model_parameters"]["independence"] is not None:
        keys_loss_comps = [
            x for x in keys_loss_comps if not x.startswith("correlation_loss")
        ]
    # compute discrepancy
    for name in keys_loss_comps:
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

    if glob_dict["model_parameters"]["independence"] is not None:
        keys_loss_comps = [
            x
            for x in list(loss_components_training.keys())
            if x.startswith("correlation_loss")
        ]
        for key_loss in keys_loss_comps:
            loss_per_component.append(
                norm_diff(loss_components_training[key_loss])
                * glob_dict["model_parameters"]["independence"]["corr_scaling"]
            )

    # save file in object
    saving_path = glob_dict["training_settings"]["output_path"]
    path = saving_path + "/loss_per_component.pkl"
    save_as_pkl(loss_per_component, path)
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
        loss_per_component_current = loss_per_component
        # TODO: check whether indicated loss weighting input is implemented
        # create subdictionary for better readability
        dict_loss = global_dict["loss_function"]
        if dict_loss["loss_weighting"] is None:
            total_loss = tf.math.reduce_sum(loss_per_component)

            if dict_loss["use_regularization"]:
                # read from results (if not yet existing, create list)
                try:
                    pd.read_pickle(
                        global_dict["training_settings"]["output_path"]
                        + "/regularizer.pkl"
                    )
                except FileNotFoundError:
                    regularizer_term = []
                else:
                    regularizer_term = pd.read_pickle(
                        global_dict["training_settings"]["output_path"]
                        + "/regularizer.pkl"
                    )

                # get prior samples
                priorsamples = pd.read_pickle(
                    global_dict["training_settings"]["output_path"]
                    + "/model_simulations.pkl"
                )["prior_samples"]
                # compute regularization
                regul_term = regulariser(priorsamples)
                # save regularization
                regularizer_term.append(regul_term)

                path = (
                    global_dict["training_settings"][
                        "output_path"] + "/regularizer.pkl"
                )
                save_as_pkl(regularizer_term, path)

                total_loss = total_loss + regul_term
        else:
            # apply selected loss weighting method
            # TODO: Tests and Checks
            if dict_loss["loss_weighting"]["method"] == "dwa":
                # dwa needs information about the initial loss per component
                if epoch == 0:
                    dict_loss["loss_weighting"]["method_specs"][
                        "loss_per_component_initial"
                    ] = loss_per_component
                # apply method
                total_loss = dynamic_weight_averaging(
                    epoch,
                    loss_per_component_current,
                    dict_loss["loss_weighting"]["method_specs"][
                        "loss_per_component_initial"
                    ],
                    dict_loss["loss_weighting"]["method_specs"]["temperature"],
                    global_dict["output_path"],
                )

            if dict_loss["loss_weighting"]["method"] == "custom":
                total_loss = tf.math.reduce_sum(
                    tf.multiply(
                        loss_per_component,
                        dict_loss["loss_weighting"]["weights"],
                    )
                )

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
