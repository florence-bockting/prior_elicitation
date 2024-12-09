# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import logging
import tensorflow as tf
import elicit.logs_config # noqa

from elicit.prior_simulation import Priors


def get_expert_data(global_dict, one_forward_simulation,
                    path_to_expert_data=None):
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

    if global_dict["expert_data"]["from_ground_truth"]:
        logger.info("Simulate from oracle")
        # set seed
        tf.random.set_seed(global_dict["training_settings"]["seed"])
        # sample from true priors
        prior_model = Priors(global_dict=global_dict,
                             ground_truth=True,
                             init_matrix_slice=None)
        expert_data = one_forward_simulation(
            prior_model, global_dict, ground_truth=True
        )
    else:
        logger.info("Read expert data")
        # load expert data from file
        # TODO Expert data must have same name and structure as sim-based
        # elicited statistics
        expert_data = global_dict["expert_data"]["data"]
    return expert_data
