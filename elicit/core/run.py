# SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import inspect
import pandas as pd

from elicit.functions.prior_simulation import Priors
from elicit.functions.model_simulation import simulate_from_generator
from elicit.functions.targets_elicits_computation import (
    computation_target_quantities,
    computation_elicited_statistics,
)
from elicit.functions.loss_computation import (
    compute_loss_components,
    compute_discrepancy,
    dynamic_weight_averaging,
)
from elicit.functions.helper_functions import save_as_pkl
from elicit.functions.loss_functions import MMD_energy
from elicit.functions.training import training_loop

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
    # generate samples from initialized prior
    prior_samples = prior_model()
    # simulate prior predictive distribution based on prior samples
    # and generative model
    model_simulations = simulate_from_generator(
        prior_samples, ground_truth, global_dict
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


def load_expert_data(global_dict, path_to_expert_data=None):
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
    if global_dict["expert_data"]["from_ground_truth"]:

        prior_model = Priors(global_dict=global_dict, ground_truth=True)
        expert_data = one_forward_simulation(
            prior_model, global_dict, ground_truth=True
        )
    else:
        # load expert data from file
        expert_data = global_dict["expert_data"]["data"]
    return expert_data


def compute_loss(
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
    expert_elicited_statistics : TYPE
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
                        global_dict["output_path"] + "/regularizer.pkl"
                        )
                except FileNotFoundError:
                    regularizer_term = []
                else:
                    regularizer_term = pd.read_pickle(
                        global_dict["output_path"] + "/regularizer.pkl"
                    )

                # get prior samples
                priorsamples = pd.read_pickle(
                    global_dict["output_path"] + "/model_simulations.pkl"
                )["prior_samples"]
                # compute regularization
                regul_term = regulariser(priorsamples)
                # save regularization
                regularizer_term.append(regul_term)

                path = global_dict["output_path"] + "/regularizer.pkl"
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
                    dict_loss["loss_weighting"]["method_specs"][
                        "temperature"
                    ],
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


def burnin_phase(
    expert_elicited_statistics, one_forward_simulation, compute_loss,
    global_dict
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
    # create subdictionary for better readability
    dict_training = global_dict["training_settings"]
    for i in range(dict_training["warmup_initializations"]):
        print("|", end="")
        # prepare generative model
        prior_model = Priors(global_dict=global_dict, ground_truth=False)
        # generate simulations from model
        training_elicited_statistics = one_forward_simulation(prior_model,
                                                              global_dict)
        # compute loss for each set of initial values
        weighted_total_loss = compute_loss(
            training_elicited_statistics,
            expert_elicited_statistics,
            global_dict,
            epoch=0,
        )

        init_var_list.append(prior_model)
        save_prior.append(prior_model.trainable_variables)
        loss_list.append(weighted_total_loss.numpy())

    path = global_dict["output_path"] + "/burnin_phase.pkl"
    save_as_pkl((loss_list, save_prior), path)

    print(" ")
    return loss_list, init_var_list


def prior_elicitation(
    model_parameters: dict,
    expert_data: dict,
    generative_model: dict,
    target_quantities: dict,
    training_settings: dict,
    normalizing_flow: dict or bool = False,
    loss_function: dict or None = None,
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
        * hyperparams_dict: dict, keys: name of hyperparameter of parametric
        prior distribution family and values representing the initialization
        which can be done via distribution or integer
        * param_scaling: float, scaling of samples after being sampled from
        prior distributions
        * independence: include correlation between priors in loss computation
        with specified scaling; or ignore correlation when None

        Code example::

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
        * independence: include correlation between priors in loss computation
        with specified scaling; or ignore correlation when None

        Code example::

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

        Code example::

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

        Code example of an example model::

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
        Code example::

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
        Code example::

            {
                method="parametric_prior",     \
                    # "parametric_prior", "deep_prior"
                sim_id="toy_example",          \
                    # individual id
                warmup_initializations=50,     \
                    # only for method="parametric_prior"; search for best \
                        initialization
                seed=0,
                view_ep=50,                    \
                    # how often should the progress_info be printed during \
                        training
                epochs=500,
                B=128,                         \
                    # number of batches
                samples_from_prior=200         \
                    # number of samples from the prior distributions
            }

    normalizing_flow : dict or bool, optional
        Architecture of the normalizing flow. The default is False.

        Code Example::

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

        Code Example::

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

        Code Example::

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
    # %% HELPER VALUES
    num_params = len(
        sorted(
            list(
                set(model_parameters.keys()).difference(set(["independence"]))
                )
            )
    )

    # %% CHECKS
    # section: model_parameters

    # check spelling of arguments
    # TODO-TEST: include test with invalid spelling of arguments; with bool
    # argument
    for arg_name in model_parameters.keys():
        try:
            model_parameters[arg_name].keys()
        except AttributeError:
            pass
        else:
            for k in model_parameters[arg_name].keys():
                assert k in [
                    "family",
                    "hyperparams_dict",
                    "param_scaling",
                    "corr_scaling",
                ], f"Have you misspelled '{k}' in the parameter settings? \
                    Only ['family', 'hyperparam_dict', 'param_scaling'] are \
                        valid argument names."

    # check whether non-optional arguments are specified
    # TODO-Test: include test with missing non-optional argument
    if training_settings["method"] == "parametric_prior":
        for param_name in sorted(
            list(
                set(model_parameters.keys()).difference(set(["independence"]))
                )
        ):
            assert set(["family", "hyperparams_dict"]) <= set(
                model_parameters[param_name].keys()
            ), f"For parameter {param_name} one of the non-optinal arguments \
                ['family', 'hyperparams_dict'] is missing."

    if training_settings["method"] == "deep_prior":
        assert (
            num_params > 1
        ), "When using the method 'deep_prior' the minimum number of \
            parameters must be 2."

    # section: normalizing_flow

    # TODO: check whether spelling of arguments is correct?
    # TODO: additional checks for normalizing flows?

    # section: expert_data
    # check whether non-optional arguments are specified
    assert (
        "from_ground_truth" in expert_data.keys()
    ), "The argument 'from_ground_truth' needs to be specified."

    # check whether combination of arguments is consistent
    # TODO-TEST: include test with (1) data=path, ground_truth=True,
    # (2) data=None, ground_truth=False, (3) missing non-optional keywords,
    # (4) misspelled keywords
    if not expert_data["from_ground_truth"]:
        assert (
            expert_data["data"] is not None
        ), "The 'data' argument needs to be specified if not simulating from\
            ground truth."

    else:
        assert set(["simulator_specs", "samples_from_prior"]) <= set(
            expert_data.keys()
        ), "At least one of the non-optional arguments 'simulator_specs', \
            'samples_from_prior' is missing."

    # section: generative_model
    # check correct spelling of arguments
    # TODO-TEST: include test with incorrect spelling
    for k in generative_model.keys():
        assert k in [
            "model",
            "additional_model_args",
            "discrete_likelihood",
            "softmax_gumble_specs",
        ], f"Have you misspelled '{k}' in the parameter settings? Only \
            ['model', 'additional_model_args' ,'discrete_likelihood', \
             'softmax_gumble_specs'] are valid argument names."

    # asure that provided model is a class (and not an instance of the class)
    assert inspect.isclass(
        generative_model["model"]
    ), "The model must be an object of type class."

    # check specification of non-optional arguments
    # TODO-TEST: include case with missing model argument
    assert "model" in generative_model.keys(), "The argument 'model' is \
        missing."

    # section: target_quantities
    # check correct spelling of arguments
    # TODO: check for duplicate naming
    # TODO-TEST: include test with (1) missing required arguments,
    # (2) quantiles without quantiles_specs, (3) moments without moments_specs
    # TODO-TEST: include test with incorrect spelling
    for k in target_quantities.keys():
        for k2 in target_quantities[k].keys():
            assert k2 in [
                "elicitation_method",
                "quantiles_specs",
                "moment_specs",
                "loss_components",
                "custom_target_function",
                "custom_elicitation_method",
            ], f'Have you misspelled "{k}" in the parameter settings? Only \
                ["elicitation_method", "quantiles_specs" , "moment_specs", \
                 "loss_components", "custom_target_function", \
                     "custom_elicitation_method"] are valid argument names.'

        try:
            target_quantities[k]["elicitation_method"]
        except KeyError:
            print(
                "The non-optional argument 'elicitation method' is missing. \
                    If you want to use a custom_elicitation_method use None."
            )
        else:
            if target_quantities[k]["elicitation_method"] == "quantiles":
                # check whether quantiles_specs is specified
                assert (
                    "quantiles_specs" in target_quantities[k].keys()
                ), "The method 'quantiles' requires the additional argument \
                    'quantiles_specs'"
                # check whether quantiles are provided in the correct format
                assert (
                    target_quantities[k]["quantiles_specs"][-1] > 1
                ), "quantiles must be specified as values between [0, 100]"
            if target_quantities[k]["elicitation_method"] == "moments":
                assert (
                    "moments_specs" in target_quantities[k].keys()
                ), "The method 'moments' requires the additional argument \
                    'moments_specs'"
            if target_quantities[k]["elicitation_method"] is None:
                assert (
                    target_quantities[k]["custom_elicitation_method"] is not
                    None
                ), "Both custom_elicitation_method and elicitation_method \
                    can't be simultaneously None."
        try:
            target_quantities[k]["loss_components"]
        except KeyError:
            print("The non-optional argument 'loss_components' is missing.")

        try:
            target_quantities[k]["custom_elicitation_method"]
        except AttributeError:
            pass
        else:
            if target_quantities[k]["custom_elicitation_method"] is not None:
                assert (
                    target_quantities[k]["elicitation_method"] is None
                ), "If custom_elicitation_method is specified, \
                    elicitation_method has to be None."

    # section: loss_function
    if loss_function is not None:
        # check correct spelling of arguments
        for k in loss_function.keys():
            assert k in [
                "loss",
                "loss_weighting",
                "use_regularization",
            ], f"Have you misspelled '{k}' in the parameter settings? Only \
                ['loss', 'loss_weighting' ,'use_regularization'] are valid \
                    argument names."

    # section: optimization_settings
    # check correct spelling of arguments
    try:
        optimization_settings.keys()
    except AttributeError:
        pass
    else:
        for k in optimization_settings.keys():
            assert k in [
                "optimizer",
                "optimizer_specs",
            ], f"Have you misspelled '{k}' in the parameter settings? Only \
                ['optimizer', 'optimizer_specs'] are valid argument names."

    # section: training_settings
    # TODO-Test: include test with incorrect spelling and missing non-optional
    # arguments check correct spelling of arguments
    for k in training_settings.keys():
        assert k in [
            "method",
            "sim_id",
            "B",
            "samples_from_prior",
            "seed",
            "warmup_initializations",
            "epochs",
            "output_path",
            "progress_info",
            "view_ep",
        ], f'Have you misspelled "{k}" in the parameter settings? Only \
            ["method", "sim_id","B","samples_from_prior","seed",\
             "warmup_initializations","epochs","output_path","progress_info"] \
                are valid argument names.'

    for k in training_settings.keys():
        assert set(["method", "sim_id", "seed", "epochs"]) <= set(
            training_settings.keys()
        ), 'At least one of the non-optional arguments "method","sim_id",\
            "seed","epochs"] is missing.'

    if training_settings["method"] == "parametric_prior":
        assert "warmup_initializations" in list(
            training_settings.keys()
        ), "If method is parametric_prior, warmup_initializations has to be \
            specified."

    # %% BUILD DICTIONARIES
    _default_dict_parameter = dict(
        param_scaling=1.0, family=None, hyperparams_dict=None
    )

    _default_dict_independence = dict(corr_scaling=0.1)

    _default_dict_normalizing_flow = dict(
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

    _default_dict_training = dict(
        method=None,
        sim_id=None,
        B=128,
        samples_from_prior=200,
        warmup_initializations=None,
        epochs=None,
        output_path="results",
        progress_info=1,
        view_ep=1,
    )

    # %% CREATE GLOBAL DICTIONARY
    global_dict = dict()

    # Section: model_parameters
    global_dict["model_parameters"] = dict()
    for param_name in sorted(
        list(set(model_parameters.keys()).difference(set(["independence"])))
    ):
        global_dict["model_parameters"
                    ][param_name] = _default_dict_parameter.copy()
        global_dict["model_parameters"
                    ][param_name].update(model_parameters[param_name])

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

    if normalizing_flow is not False:
        global_dict["normalizing_flow"] = _default_dict_normalizing_flow.copy()
        try:
            global_dict["normalizing_flow"].update(normalizing_flow)
        except AttributeError:
            pass
    else:
        global_dict["normalizing_flow"] = False

    # Section: expert_data
    global_dict["expert_data"] = dict()
    global_dict["expert_data"][
        "from_ground_truth"
        ] = expert_data["from_ground_truth"]
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
        global_dict[
            "target_quantities"
            ][target_quant] = _default_dict_targets.copy()
        global_dict["target_quantities"][target_quant].update(
            target_quantities[target_quant]
        )

    # Section: loss_function
    # TODO-Test: include test loss_function is None
    dict_loss = dict()
    dict_loss = _default_dict_loss.copy()
    if loss_function is not None:
        dict_loss.update(loss_function)

    # Section: optimization_settings
    # TODO-Test: include test optimization_setting is None
    global_dict["optimization_settings"] = dict()
    global_dict["optimization_settings"] = _default_dict_optimizer.copy()
    if optimization_settings is not None:
        global_dict["optimization_settings"].update(optimization_settings)

    # Section: training_settings
    # TODO-Test: include test optimization_setting is None
    dict_training = dict()
    dict_training = _default_dict_training.copy()
    dict_training.update(training_settings)

    # include helper value about parameter number to global dict
    global_dict["model_parameters"]["no_params"] = num_params

    # %% SAVE GLOBAL DICT
    dict_training[
        "output_path"
    ] = f"./elicit/{global_dict['training_settings']['output_path']}/\
        {training_settings['method']}/{training_settings['sim_id']}\
            _{training_settings['seed']}"
    path = dict_training["output_path"] + "/global_dict.pkl"
    save_as_pkl(global_dict, path)

    # %% RUN DAG
    # set seed
    tf.random.set_seed(training_settings["seed"])

    # get expert data
    expert_elicited_statistics = load_expert_data(global_dict)

    if dict_training["warmup_initializations"] is None:
        # prepare generative model
        init_prior_model = Priors(global_dict=global_dict, ground_truth=False)

    else:
        loss_list, init_prior = burnin_phase(
            expert_elicited_statistics,
            one_forward_simulation,
            compute_loss,
            global_dict,
        )

        # extract minimum loss out of all runs and corresponding set of
        # initial values
        min_index = tf.argmin(loss_list)
        init_prior_model = init_prior[min_index]

    # run dag with optimal set of initial values
    training_loop(
        expert_elicited_statistics,
        init_prior_model,
        one_forward_simulation,
        compute_loss,
        global_dict,
        training_settings["seed"],
    )
