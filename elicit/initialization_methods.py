# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp

from elicit.helper_functions import save_as_pkl
from elicit.prior_simulation import Priors

tfd = tfp.distributions


def initialization_phase(
    expert_elicited_statistics, one_forward_simulation, compute_loss,
    global_dict,
):
    """
    For the method "parametric_prior" it might be helpful to run different
    initializations before the actual training starts in order to find a
    'good' set of initial values. For this purpose the burnin phase can be
    used. It rans multiple initializations and computes for each the
    respective loss value. At the end that set of initial values is chosen
    which leads to the smallest loss.

    Parameters
    ----------
    expert_elicited_statistics : dict
        dictionary with expert elicited statistics.
    one_forward_simulation : callable
        one forward simulation from prior samples to model-simulated elicited
        statistics.
    compute_loss : callable
        wrapper for loss computation from loss components to (weighted) total
        loss.
    global_dict : dict
        global dictionary with all user input specifications.

    Returns
    -------
    loss_list : list
        list containing the loss values for each set of initial values.
    init_var_list : list
        set of initial values for each run.

    """

    loss_list = []
    init_var_list = []
    save_prior = []
    dict_copy = dict(global_dict)

    def init_method(n_hypparam, n_warm_up):
        mvdist = tfd.MultivariateNormalDiag(
            tf.zeros(n_hypparam), tf.ones(n_hypparam)
        ).sample(n_warm_up)
        return mvdist

    # get number of hyperparameters
    n_hypparam = 0
    param_names = set(global_dict["model_parameters"]).difference(
        ["independence", "no_params"]
    )
    for param in param_names:
        n_hypparam += len(
            global_dict["model_parameters"][param]["hyperparams_dict"].keys()
        )
    # create initializations
    init_matrix = init_method(
        n_hypparam, dict_copy["training_settings"]["warmup_initializations"]
    )

    path = dict_copy["training_settings"][
        "output_path"] + "/initialization_matrix.pkl"
    save_as_pkl(init_matrix, path)

    for i in range(dict_copy["training_settings"]["warmup_initializations"]):
        dict_copy["training_settings"]["seed"] = (
            dict_copy["training_settings"]["seed"] + i
        )
        # prepare generative model
        prior_model = Priors(global_dict=dict_copy,
                             ground_truth=False)
        # generate simulations from model
        training_elicited_statistics = one_forward_simulation(prior_model,
                                                              dict_copy)
        # compute loss for each set of initial values
        weighted_total_loss = compute_loss(
            training_elicited_statistics,
            expert_elicited_statistics,
            dict_copy,
            epoch=0,
        )
        print(f"({i}) {weighted_total_loss.numpy():.1f} ", end="")

        init_var_list.append(prior_model)
        save_prior.append(prior_model.trainable_variables)
        loss_list.append(weighted_total_loss.numpy())

    path = dict_copy["training_settings"][
        "output_path"] + "/initialization_phase.pkl"
    save_as_pkl((loss_list, save_prior), path)

    print(" ")
    return loss_list, init_var_list
