# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import logging

import elicit.save_config
import elicit.logs_config # noqa

from elicit.prior_simulation import Priors
from elicit.loss_computation import compute_total_loss
from elicit.loss_functions import MMD_energy
from elicit.optimization_process import sgd_training
from elicit.initialization_methods import initialization_phase
from elicit.expert_data import get_expert_data
from elicit.helper_functions import save_as_pkl, remove_unneeded_files
from elicit.checks import check_run
from elicit.model_simulation import simulate_from_generator
from elicit.target_quantities import computation_target_quantities
from elicit.elicitation_techniques import computation_elicited_statistics

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
    model_simulations = simulate_from_generator(
        prior_samples, ground_truth, global_dict,
    )
    # compute the target quantities
    target_quantities = computation_target_quantities(
        model_simulations, ground_truth, global_dict
    )
    # compute the elicited statistics by applying a specific elicitation
    # method on the target quantities
    elicited_statistics = computation_elicited_statistics(
        target_quantities, ground_truth, global_dict
    )
    return elicited_statistics


def prior_elicitation(
    model_parameters: dict,
    expert_data: dict,
    generative_model: dict,
    target_quantities: dict,
    training_settings: dict,
    normalizing_flow: dict or bool = False,
    loss_function: dict or None = None,
    initialization_settings: dict or None = None,
    optimization_settings: dict or None = None,
):
    """
    Performes prior learning based on expert knowledge

    Parameters
    ----------
    model_parameters : dict

    Specify a dictionary for each model parameter

        * if method=parametric_prior: dictionary per model parameter
        (here: param1, param2) with

            * family: tfd.distribution, specification of the prior distribution
            family per parameter

            * hyperparams_dict: dict, keys: name of hyperparameter of parametric # noqa
            prior distribution family and values representing the initialization # noqa
            which can be done via distribution or integer

            * param_scaling: float, scaling of samples after being sampled from
            prior distributions

            * independence: include correlation between priors in loss computation # noqa
            with specified scaling; or ignore correlation when None

        **Code example**::

            {
                param1 = dict(
                    family=tfd.distribution,
                    hyperparams_dict={
                        hyperparam1=tfd.distribution or int,
                        hyperparam2=tfd.distribution or int,
                        # ...
                        },
                     param_scaling=1.),
                param2 = dict(
                    family=tfd.distribution,
                    # ... (see above)
                    ),
                independence={cor_scaling=0.1} or None
            }

        * if method=deep_prior: dictionary per model parameter
        (here: param1, param2) with

            * param_scaling: float, scaling of samples after being sampled from
            prior distributions

            * independence: include correlation between priors in loss computation # noqa
            with specified scaling; or ignore correlation when None

        **Code example**::

            {
                param1 = dict(param_scaling=1.),
                param2 = dict(param_sacling=1.),
                independence={corr_scaling=0.1} or None
            }

    expert_data : dict
        Specify the expert information. There are two possibilities: Either
        you pass elicited expert data or you simulate from a pre-specified
        ground truth.
        If you have expert data::

            {
                from_ground_truth=False,
                expert_data=read-file-with-expert-data

            }

        If you simulate from a pre-specified ground truth::

            {
                from_ground_truth=True,
                simulator_specs={
                    param1 = tfd.Normal(0.1, 0.5),
                    param2 = tfd.Normal(0.5, 0.8)
                    },
                samples_from_prior=10_000
            }

        The keywords have the following interpretation:
            * from_ground_truth: bool, True if simulating from ground truth
            otherwise False

            * expert_data: read file with expert data; must have the same
            format as the model-simulated elicited statistics

            * simulator_specs: dict, specifies the true prior distributions
            of the model parameters

            * samples_from_prior: int, number of samples drawn from each prior
            distribution

    generative_model : dict
        Specification of the generative model.

        **Code example**::

            {
                model=ExampleModel,
                additional_model_args={
                    design_matrix=design-matrix-file,
                    #...
                    }
            }

        With the following definitions:
            * model: class, class definition of the generative model
            (see example below)

            * additional_model_args: all input arguments of the generative
            model

        **Code example** of an example model::

            class ExampleModel:
                def __call__(self,
                             ground_truth,  # required argument
                             prior_samples, # required argument
                             X,             # additional model args
                             N              # additional model args
                             ):
                    epred = tf.expand_dims(prior_samples[:,:,0],-1) @ X
                    likelihood = tfd.Normal(
                        loc=epred,
                        scale=tf.expand_dims(prior_samples[:,:,1],-1))
                    ypred = likelihood.sample()

                    return dict(likelihood = likelihood,      # required
                                ypred = ypred,                # required
                                epred = epred                 # required
                                prior_samples = prior_samples # required
                                )

    target_quantities : dict
        **Code example**::

            {
                ypred=dict(
                        elicitation_method="quantiles",       \
                            # "quantiles", "moments", "histogram"
                        quantiles_specs=(5, 25, 50, 75, 95),  \
                            # only if elicitation_method="quantiles"
                        moments_specs=("mean","sd"),          \
                            # only if elicitation_method="moments"
                        loss_components = "all",              \
                            # "all","by-group"
                        custom_target_function=None,          \
                            # optional if user-specific target quantity should
                            be used
                        custom_elicitation_method=None
                        )
            }

    training_settings : dict
        **Code example**::

            {
                method="parametric_prior",     \
                    # "parametric_prior", "deep_prior"
                sim_id="toy_example",          \
                    # individual id
                warmup_initializations=50,     \
                    # only for method="parametric_prior"; search for best initialization # noqa
                seed=0,
                view_ep=50,                    \
                    # how often should the progress_info be printed during training # noqa
                epochs=500,
                B=128,                         \
                    # number of batches
                samples_from_prior=200         \
                    # number of samples from the prior distributions
            }

    normalizing_flow : dict or bool, optional
        Architecture of the normalizing flow. The default is False.

        **Code Example**::

            {
                 num_coupling_layers=3,
                 coupling_design="affine",
                 coupling_settings={
                     "dropout": False,
                     "dense_args": {
                         "units": 128,
                         "activation": "relu",
                         "kernel_regularizer": None,
                     },
                     "num_dense": 2,
                 },
                 permutation="fixed",
                 base_distribution=tfd.MultivariateNormalTriL(
                         loc=tf.zeros(num_params),
                         scale_tril=tf.linalg.cholesky(tf.eye(num_params))
                         )
            }

    loss_function : dict or None, optional
        Specification of the loss function. The default is None.

        **Code Example**::

            {
                  loss=MMD_energy,            \
                      # default discrepancy measure MMD
                  loss_weighting=None,        \
                      # if loss-balancing method is applied
                  use_regularization=False    \
                      # if regularizer is added to the loss function
            }

    optimization_settings : dict or None, optional
        Specification of the optimizer used for SGD training.
        The default is None.

        **Code Example**::

            {
                optimizer=tf.keras.optimizers.Adam,
                optimizer_specs={
                    "learning_rate": 0.0001,
                    "clipnorm": 1.0
                    }
            }

    Returns
    -------
    Learns the prior distributions

    """

    logger = logging.getLogger(__name__)
    # %% HELPER VALUES
    num_params = len(
        sorted(list(set(
            model_parameters.keys()).difference(set(["independence"]))))
    )

    # %% CHECKS
    check_run.check_model_parameters(training_settings, model_parameters,
                                     num_params)
    check_run.check_expert_data(expert_data)
    check_run.check_generative_model(generative_model)
    check_run.check_target_quantities(target_quantities)
    check_run.check_loss_function(loss_function)
    check_run.check_optimization_settings(optimization_settings)
    check_run.check_training_settings(training_settings)

    # %% BUILD DICTIONARIES
    _default_dict_parameter = dict(
        param_scaling=1.0, family=None, hyperparams_dict=None
    )

    _default_dict_independence = dict(corr_scaling=0.1)

    _default_dict_normalizing_flow = dict(
        coupling_flow=dict(
            num_coupling_layers=3,
            coupling_design="affine",
            coupling_settings={
                "dropout": False,
                "dense_args": {
                    "units": 128,
                    "activation": "relu",
                    "kernel_regularizer": None,
                },
                "num_dense": 2,
            },
            permutation="fixed",
        ),
        base_distribution=tfd.MultivariateNormalTriL(
            loc=tf.zeros(num_params),
            scale_tril=tf.linalg.cholesky(tf.eye(num_params))
        ),
    )

    _default_dict_expert = dict(
        data=None, simulator_specs=None, samples_from_prior=None
    )

    _default_dict_model = dict(
        additional_model_args=None, discrete_likelihood=False,
        softmax_gumble_specs=None
    )

    # TODO: custom_functions have a specific dictionary setting which needs to
    # be checked
    _default_dict_targets = dict(
        elicitation_method=None,
        quantiles_specs=None,
        moments_specs=None,
        loss_components="all",
        custom_target_function=None,
        custom_elicitation_method=None,
    )

    _default_dict_loss = dict(
        loss=MMD_energy, loss_weighting=None, use_regularization=False
    )

    _default_dict_optimizer = dict(
        optimizer=tf.keras.optimizers.Adam,
        optimizer_specs={"learning_rate": 0.0001, "clipnorm": 1.0},
    )

    _default_dict_initialization = dict(
        method="random",
        loss_quantile=0,
        number_of_iterations=200
    )

    _default_dict_training = dict(
        method=None,
        sim_id=None,
        B=128,
        samples_from_prior=200,
        epochs=None,
        output_path="results",
        progress_info=1,
        view_ep=1,
        print_log=True,
        save_log=False
    )

    # %% CREATE GLOBAL DICTIONARY
    global_dict = dict()

    # Section: model_parameters
    global_dict["model_parameters"] = dict()
    for param_name in sorted(
        list(set(model_parameters.keys()).difference(set(["independence"])))
    ):
        global_dict["model_parameters"][
            param_name] = _default_dict_parameter.copy()
        global_dict["model_parameters"][
            param_name].update(model_parameters[param_name])

    # TODO-TEST: include test with independence = True, False, user-dict
    if model_parameters["independence"] is not None:
        global_dict["model_parameters"][
            "independence"
        ] = _default_dict_independence.copy()
        global_dict["model_parameters"]["independence"].update(
            model_parameters["independence"]
        )
    else:
        global_dict["model_parameters"]["independence"] = None

    # Section: normalizing_flow
    # TODO-TEST: include test with normalizing_flow = True, False, user-dict
    global_dict["normalizing_flow"] = dict()

    if type(normalizing_flow) is dict:
        global_dict["normalizing_flow"] = _default_dict_normalizing_flow.copy()
        try:
            global_dict["normalizing_flow"].update(normalizing_flow)
        except AttributeError:
            pass
    elif normalizing_flow is True:
        global_dict["normalizing_flow"] = _default_dict_normalizing_flow.copy()
    else:
        global_dict["normalizing_flow"] = False

    # Section: expert_data
    global_dict["expert_data"] = dict()
    global_dict["expert_data"][
        "from_ground_truth"] = expert_data["from_ground_truth"]
    global_dict["expert_data"].update(_default_dict_expert.copy())
    global_dict["expert_data"].update(expert_data)

    # Section: generative_model
    global_dict["generative_model"] = dict()
    global_dict["generative_model"]["model"] = generative_model["model"]
    global_dict["generative_model"].update(_default_dict_model.copy())
    global_dict["generative_model"].update(generative_model)

    # Section: target_quantities
    global_dict["target_quantities"] = dict()
    for target_quant in target_quantities.keys():
        global_dict["target_quantities"][
            target_quant] = _default_dict_targets.copy()
        global_dict["target_quantities"][target_quant].update(
            target_quantities[target_quant]
        )

    # Section: loss_function
    # TODO-Test: include test loss_function is None
    global_dict["loss_function"] = dict()
    global_dict["loss_function"] = _default_dict_loss.copy()
    if loss_function is not None:
        global_dict["loss_function"].update(loss_function)

    # Section: optimization_settings
    # TODO-Test: include test optimization_setting is None
    global_dict["optimization_settings"] = dict()
    global_dict["optimization_settings"] = _default_dict_optimizer.copy()
    if optimization_settings is not None:
        global_dict["optimization_settings"].update(optimization_settings)

    # Section: initialization settings
    global_dict["initialization_settings"] = dict()
    global_dict["initialization_settings"] = _default_dict_initialization.copy() # noqa
    global_dict["initialization_settings"].update(initialization_settings)

    # Section: training_settings
    # TODO-Test: include test optimization_setting is None
    global_dict["training_settings"] = dict()
    global_dict["training_settings"] = _default_dict_training.copy()
    global_dict["training_settings"].update(training_settings)

    # include helper value about parameter number to global dict
    global_dict["model_parameters"]["no_params"] = num_params

    # %% SAVE GLOBAL DICT
    global_dict["training_settings"][
        "output_path"
    ] = f"./elicit/{global_dict['training_settings']['output_path']}/{training_settings['method']}/{training_settings['sim_id']}_{training_settings['seed']}"  # noqa
    path = global_dict["training_settings"]["output_path"] + "/global_dict.pkl"
    save_as_pkl(global_dict, path)

    # %% RUN DAG
    # set seed
    tf.random.set_seed(training_settings["seed"])

    # get expert data
    expert_elicited_statistics = get_expert_data(global_dict,
                                                 one_forward_simulation)

    def pre_training(global_dict):
        logger = logging.getLogger(__name__)

        if global_dict["training_settings"] == "parametric_prior":
            logger.info("Pre-training phase (only first run)")
            loss_list, init_prior = initialization_phase(
                expert_elicited_statistics,
                one_forward_simulation,
                compute_total_loss,
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
            init_prior_model = Priors(global_dict=global_dict,
                                      ground_truth=False,
                                      init_matrix_slice=None)

        return init_prior_model

    init_prior_model = pre_training(global_dict)

    # run dag with optimal set of initial values
    logger.info("Training Phase (only first epoch)")
    sgd_training(
        expert_elicited_statistics,
        init_prior_model,
        one_forward_simulation,
        compute_total_loss,
        global_dict,
        training_settings["seed"],
    )

    # remove saved files that are not of interest for follow-up analysis
    remove_unneeded_files(global_dict, elicit.save_config.save_results)
