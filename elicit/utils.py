# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import elicit as el
import logging
import logging.config


#%% configuration for logging information
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
        }
    },
    "handlers": {
        "json_file": {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'logs.json',
            'formatter': 'json',
            }
    },
    "loggers": {"": {"handlers": ["json_file"], "level": "INFO"}},
}

logging.config.dictConfig(LOGGING)


#%% configuration for saving results
save_results = dict(
    # all generated initializations during pre-training
    initialization_matrix=True,
    # tuple: loss values corresp. to each set of generated initial values
    pre_training_results=True,
    # initialized hyperparameter values
    init_hyperparameters=True,
    # prior samples of last epoch
    prior_samples=False,
    # elicited statistics of last epoch
    elicited_statistics=True,
)


#%% wrapper for workflow
def one_forward_simulation(prior_model, trainer, model, targets,
                           ground_truth=False):
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
        prior_samples, ground_truth, trainer["seed"], trainer["output_path"],
        model
    )
    # compute the target quantities
    target_quantities = el.targets.computation_target_quantities(
        model_simulations, ground_truth, targets, trainer["output_path"]
    )
    # compute the elicited statistics by applying a specific elicitation
    # method on the target quantities
    elicited_statistics = el.targets.computation_elicited_statistics(
        target_quantities, ground_truth, targets, trainer["output_path"]
    )
    return elicited_statistics, prior_samples, model_simulations, target_quantities


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
    logger = logging.getLogger(__name__)

    try:
        expert["data"]
    except KeyError:
        oracle=True
    else:
        oracle=False

    if oracle:
        logger.info("Simulate from oracle")
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
            prior_model, trainer, model, targets, ground_truth=True
        )
        return expert_data, expert_prior

    else:
        logger.info("Read expert data")
        # load expert data from file
        # TODO Expert data must have same name and structure as sim-based
        # elicited statistics
        expert_data = expert["data"]
        return expert_data, None