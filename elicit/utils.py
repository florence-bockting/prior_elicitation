# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import elicit as el
import pandas as pd
import os


#%% wrapper for workflow
def one_forward_simulation(prior_model, trainer, model, targets):
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
    tf.random.set_seed(trainer["seed"])
    # generate samples from initialized prior
    prior_samples = prior_model()
    # simulate prior predictive distribution based on prior samples
    # and generative model
    model_simulations = el.simulations.simulate_from_generator(
        prior_samples, trainer["seed"], model
    )
    # compute the target quantities
    target_quantities = el.targets.computation_target_quantities(
        model_simulations, targets
    )
    # compute the elicited statistics by applying a specific elicitation
    # method on the target quantities
    elicited_statistics = el.targets.computation_elicited_statistics(
        target_quantities, targets
    )
    return (elicited_statistics, prior_samples, model_simulations,
            target_quantities)


#%% simulate expert data or get input data
def get_expert_data(trainer, model, targets, expert, parameters, network):
    """
    Wrapper for loading the training data which can be expert data or
    data simulations using a pre-defined ground truth.

    Parameters
    ----------
    global_dict : dict
        global dictionary with all user input specifications.
    path_to_expert_data : str, optional
        path to file location where expert data has been saved

    Returns
    -------
    expert_data : dict
        dictionary containing the training data. Must have same form as the
        model-simulated elicited statistics.

    """

    try:
        expert["data"]
    except KeyError:
        oracle=True
    else:
        oracle=False

    if oracle:
        # set seed
        tf.random.set_seed(trainer["seed"])
        # sample from true priors
        prior_model = el.simulations.Priors(
            ground_truth=True, 
            init_matrix_slice=None,
            trainer=trainer, parameters=parameters, network=network,
            expert=expert,
            seed=trainer["seed"])
        # compute elicited statistics and target quantities
        expert_data, expert_prior, *_ = one_forward_simulation(
            prior_model, trainer, model, targets
        )
        return expert_data, expert_prior

    else:
        # load expert data from file
        # TODO Expert data must have same name and structure as sim-based
        # elicited statistics
        expert_data = expert["data"]
        return expert_data, None


def save_elicit(elicit_obj, save_dir, force_overwrite=False):
    # check whether saving path is already used
    if os.path.isfile(save_dir) and not force_overwrite:
        user_ans = input("In provided directory exists already a file with"+
                         " identical name. Do you want to overwrite it?"+
                         " Press 'y' for overwriting and 'n' for abording.")
        while user_ans not in ["n", "y"]:
            user_ans = input("Please press either 'y' for overwriting or 'n'"+
                             "for abording the process.")

        if user_ans == "n":
            return("Process aborded. File is not overwritten.")

    storage = dict()
    # user inputs
    storage["model"] = elicit_obj.model
    storage["parameters"] = elicit_obj.parameters
    storage["targets"] = elicit_obj.targets
    storage["expert"] = elicit_obj.expert
    storage["optimizer"] = elicit_obj.optimizer
    storage["trainer"] = elicit_obj.trainer
    storage["initializer"] = elicit_obj.initializer
    storage["network"] = elicit_obj.network
    # results
    storage["results"] = elicit_obj.results
    storage["history"] = elicit_obj.history

    el.helpers.save_as_pkl(storage, save_dir)
    print(f"saved elicit as: {save_dir}")


def load_elicit(save_dir):
    obj = pd.read_pickle(save_dir)

    elicit = el.Elicit(
        model = obj["model"],
        parameters = obj["parameters"],
        targets = obj["targets"],
        expert = obj["expert"],
        optimizer = obj["optimizer"],
        trainer = obj["trainer"],
        initializer = obj["initializer"],
        network = obj["network"]
        )

    # add results if already fitted
    elicit.history = obj["history"]
    elicit.results = obj["results"]

    return elicit
