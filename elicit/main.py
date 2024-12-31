# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import logging
import elicit as el

from elicit.configs import *  # noqa

tfd = tfp.distributions


def one_forward_simulation(prior_model, global_dict, ground_truth=False):
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
    tf.random.set_seed(global_dict["training_settings"]["seed"])
    # generate samples from initialized prior
    prior_samples = prior_model()
    # simulate prior predictive distribution based on prior samples
    # and generative model
    model_simulations = el.simulate_from_generator(
        prior_samples, ground_truth, global_dict,
    )
    # compute the target quantities
    target_quantities = el.computation_target_quantities(
        model_simulations, ground_truth, global_dict
    )
    # compute the elicited statistics by applying a specific elicitation
    # method on the target quantities
    elicited_statistics = el.computation_elicited_statistics(
        target_quantities, ground_truth, global_dict
    )
    return elicited_statistics, prior_samples, model_simulations, target_quantities


def pre_training(global_dict, expert_elicited_statistics):
    logger = logging.getLogger(__name__)

    if global_dict["training_settings"]["method"] == "parametric_prior":

        logger.info("Pre-training phase (only first run)")

        loss_list, init_prior = el.initialization_phase(
            expert_elicited_statistics,
            one_forward_simulation,
            el.compute_total_loss,
            global_dict,
        )

        # extract pre-specified quantile loss out of all runs
        # get corresponding set of initial values
        loss_quantile = global_dict["initialization_settings"][
            "loss_quantile"]
        index = tf.squeeze(tf.where(loss_list == tfp.stats.percentile(
            loss_list, [loss_quantile])))
        init_prior_model = init_prior[int(index)]
    else:
        # prepare generative model
        init_prior_model = el.Priors(global_dict=global_dict,
                                  ground_truth=False,
                                  init_matrix_slice=None)

    return init_prior_model


class Elicit:

   def __init__(self,
        model: callable,
        parameters: list,
        target_quantities: list,
        expert: callable,
        training_settings: callable,
        optimization_settings: callable,
        normalizing_flow: callable or None=None,
        initialization_settings: callable or None=None,
    ):
        """
        Parameters
        ----------
        model : callable
            specification of generative model using
            :func:`elicit.prior_elicitation.generator`.
        parameter : list
            list of model parameters specified with
            :func:`elicit.prior_elicitation.par`.
        target_quantities : list
            list of target quantities specified with
            :func:`elicit.prior_elicitation.tar`.
        expert : callable
            specification of input data from expert or oracle using
            :func:`el.expert.data` or func:`el.expert.simulate`
        training_settings : callable
            specification of training settings for learning prior distribution(s)
            using :func:`elicit.prior_elicitation.train`
        optimization_settings : callable
            specification of optimizer using
            :func:`elicit.prior_elicitation.optimizer`.
        normalizing_flow : callable or None
            specification of normalizing flow using :func:`elicit.prior_elicitation.nf`.
            Only required for ``deep_prior`` method is used. If 
            ``parametric_prior`` is used this argument should be ``None``. Default
            value is None.
        initialization_settings : callable
            specification of initialization settings using
            :func:`elicit.prior_elicitation.initializer`. For method
            'parametric_prior' the argument should be None. Default value is None.

        Returns
        -------
        global_dict : dict
            specification of all settings to run the optimization procedure.

        """
        self.inputs = dict(
            model=model,
            parameters=parameters,
            target_quantities=target_quantities,
            expert=expert,
            training_settings=training_settings,
            optimization_settings=optimization_settings,
            normalizing_flow=normalizing_flow,
            initialization_settings=initialization_settings,
            )


   def train(self, save_file: str or None=None):
        logger = logging.getLogger(__name__)

        if save_file is not None:
            # create saving path
            self.inputs["training_settings"][
                "output_path"
            ] = f"./elicit/{save_file}/{self.inputs['training_settings']['method']}/{self.inputs['training_settings']['name']}_{self.inputs['training_settings']['seed']}"  # noqa
        else:
            self.inputs["training_settings"]["output_path"] = None

        # set seed
        tf.random.set_seed(self.inputs["training_settings"]["seed"])

        # get expert data
        try:
            self.inputs["expert"]["ground_truth"]
        except KeyError:
            expert_elicits = self.inputs["expert"]["data"]
        else:
            expert_elicits, expert_prior = el.get_expert_data(
                self.inputs,
                one_forward_simulation)

        # initialization of hyperparameter
        init_prior_model = pre_training(self.inputs, expert_elicits)

        # run dag with optimal set of initial values
        logger.info("Training Phase (only first epoch)")
        res_ep, res = el.sgd_training(
            expert_elicits,
            init_prior_model,
            one_forward_simulation,
            el.compute_total_loss,
            self.inputs
        )

        res["expert_elicited_statistics"] = expert_elicits
        try:
            self.inputs["expert"]["ground_truth"]
        except KeyError:
            pass
        else:
            res["expert_prior_samples"] = expert_prior

        # remove saved files that are not of interest for follow-up analysis
        if save_file is not None:
            el.remove_unneeded_files(self.inputs, save_results)  # noqa

        return res_ep, res