# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import inspect

# %% CHECKS
# section: model_parameters

# check spelling of arguments
# TODO-TEST: include test with invalid spelling of arguments; with bool
# argument
def check_model_parameters(training_settings, model_parameters, num_params):
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
def check_expert_data(expert_data):
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
def check_generative_model(generative_model):
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
def check_target_quantities(target_quantities):
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
        except KeyError:
            pass
        else:
            if target_quantities[k]["custom_elicitation_method"] is not None:
                assert (
                    target_quantities[k]["elicitation_method"] is None
                ), "If custom_elicitation_method is specified, \
                    elicitation_method has to be None."

def check_loss_function(loss_function):
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
def check_optimization_settings(optimization_settings):
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
def check_training_settings(training_settings):
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