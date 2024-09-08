import tensorflow as tf
import tensorflow_probability as tfp
import bayesflow as bf
import inspect
import pandas as pd

tfd = tfp.distributions
bfn = bf.networks

from functions.helper_functions import save_as_pkl
from user_input.custom_functions import custom_correlation


def use_custom_functions(custom_function, model_simulations, global_dict):
    """
    Helper function that prepares custom functions if specified by checking
    all inputs and extracting the argument from different sources.

    Parameters
    ----------
    custom_function : callable
        custom function as specified by the user.
    model_simulations : dict
        simulations from the generative model.
    global_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    custom_quantity : tf.Tensor
        returns the evaluated custom function.

    """
    # get function
    custom_func = custom_function["function"]
    # create a dict with arguments from model simulations and custom args for custom func
    args_dict = dict()
    if custom_function["additional_args"] is not None:
        additional_args_dict = {
            f"{key}": custom_function["additional_args"][key]
            for key in list(custom_function["additional_args"].keys())
        }
    else:
        additional_args_dict = {}
    # select only relevant keys from args_dict
    custom_args_keys = inspect.getfullargspec(custom_func)[0]
    # check whether expert-specific input has been specified
    if "from_simulated_truth" in custom_args_keys:
        for i in range(len(inspect.getfullargspec(custom_func)[3][0])):
            quantity = inspect.getfullargspec(custom_func)[3][i][0]
            true_model_simulations = pd.read_pickle(
                global_dict["output_path"]["data"] + "/expert/model_simulations.pkl"
            )
            for key in custom_args_keys:
                if f"{key}" == quantity:
                    args_dict[key] = true_model_simulations[quantity]
                    custom_args_keys.remove(quantity)
        custom_args_keys.remove("from_simulated_truth")
    # TODO: check that all args needed for custom function were detected
    for key in list(set(custom_args_keys) - set(additional_args_dict)):
        args_dict[key] = model_simulations[key]
    for key in additional_args_dict:
        args_dict.update(additional_args_dict)
    # evaluate custom function
    custom_quantity = custom_func(**args_dict)
    return custom_quantity


def computation_target_quantities(model_simulations, ground_truth, global_dict):
    """
    Computes target quantities from model simulations.

    Parameters
    ----------
    model_simulations : dict
        simulations from generative model.
    ground_truth : bool
        whether simulations are based on ground truth. Mainly used for saving
        results in extra folder "expert" for later analysis.
    global_dict : dict
        dictionary including all user-input settings..

    Returns
    -------
    targets_res : dict
        computed target quantities.
    """
    # extract target quantities section from global dict
    targets_global_dict = global_dict["target_quantities"]

    # names of target quantities
    name_targets = targets_global_dict["name"]
    # TODO: check for duplicate naming
    # initialize dict for storing results
    targets_res = dict()
    # loop over target quantities
    for i, target in enumerate(name_targets):

        # use custom function for target quantity if it has been defined
        if targets_global_dict["custom_target_function"][i] is not None:
            target_quantity = use_custom_functions(
                targets_global_dict["custom_target_function"][i],
                model_simulations,
                global_dict,
            )
        else:
            target_quantity = model_simulations[target]

        # select indicated observations from design matrix
        # TODO: I suppose that makes only sense for ypred as target quantity; there should be a warning?, error?
        if targets_global_dict["select_obs"][i] is not None:
            target_quantity = tf.gather(
                target_quantity, list(targets_global_dict["select_obs"][i]), axis=-1
            )
        # save target quantities
        targets_res[target] = target_quantity

    if global_dict["param_independence"]["independent"]:
        target_quantity = use_custom_functions(
            {"function": custom_correlation, "additional_args": None},
            model_simulations,
            global_dict,
        )

        targets_res["correlation"] = target_quantity
    # save file in object
    saving_path = global_dict["output_path"]["data"]
    if ground_truth:
        saving_path = saving_path + "/expert"
    path = saving_path + "/target_quantities.pkl"
    save_as_pkl(targets_res, path)
    # return results
    return targets_res


def computation_elicited_statistics(target_quantities, ground_truth, global_dict):
    """
    Computes the elicited statistics from the target quantities by applying a
    prespecified elicitation technique.

    Parameters
    ----------
    target_quantities : dict
        simulated target quantities.
    ground_truth : bool
        whether simulations are based on ground truth. Mainly used for saving
        results in extra folder "expert" for later analysis..
    global_dict : dict
        dictionary including all user-input settings..

    Returns
    -------
    elicits_res : dict
        simulated elicited statistics.

    """
    # extract target quantities section from global dict
    targets_global_dict = global_dict["target_quantities"]

    # names of elicitation techniques
    name_elicits = targets_global_dict["elicitation_method"]
    # names of target quantities
    name_targets = list(target_quantities.keys())
    # initialize dict for storing results
    elicits_res = dict()
    # loop over elicitation techniques
    for i, (target, elicit) in enumerate(zip(name_targets, name_elicits)):
        # check for support of elicitation technique
        assert elicit in set(
            ["quantiles", "histogram", "moments", "identity"]
        ), "Name error of elicitation techniques. Currently supported elicitation techniques are quantiles, histogram, moments."

        # use custom function for target quantity if it has been defined
        # TODO: this functionality has to be checked!
        try:
            targets_global_dict["custom_elicitation_function"][i]
        except:
            pass
        else:
            if targets_global_dict["custom_elicitation_function"][i] is not None:
                elicited_statistic = use_custom_functions(
                    targets_global_dict["custom_elicitation_function"][i],
                    target_quantities,
                )

        if elicit == "identity":
            elicited_statistic = target_quantities[target]

        if elicit == "histogram":
            quantiles_hist = list(tf.range(2, 100, 2))
            target_hist = tfp.stats.percentile(
                target_quantities[target], q=quantiles_hist, axis=-1
            )
            elicited_statistic = tf.einsum("ij...->ji...", target_hist)

        if elicit == "quantiles":
            quantiles = targets_global_dict["quantiles_specs"][i]
            # check whether user-provided quantiles exist and have the correct format
            assert (
                quantiles[-1] > 1
            ), "quantiles must be specified as values between [0, 100]"
            assert (
                quantiles is not None
            ), "no quantiles in the argument quantiles_specs have been defined"
            # compute quantiles
            quan = target_quantities[target]
            if tf.rank(quan) == 3:
                quan_reshaped = tf.reshape(
                    quan, (quan.shape[0], quan.shape[1] * quan.shape[2])
                )
            if tf.rank(quan) == 2:
                quan_reshaped = quan
            computed_quantiles = tfp.stats.percentile(
                quan_reshaped, q=quantiles, axis=-1
            )
            # bring quantiles to the last dimension
            elicited_statistic = tf.einsum("ij...->ji...", computed_quantiles)

        if elicit == "moments":
            moments = targets_global_dict["moments_specs"][i]
            assert (
                moments is not None
            ), "no moments in the argument moments_specs have been defined"
            # for each moment
            # TODO: implement feature for custom moment functions
            for moment in moments:
                # check whether moment is supported
                assert moment in [
                    "sd",
                    "mean",
                ], "currently only 'mean' and 'sd' are supported as moments"

                if moment == "mean":
                    computed_mean = tf.reduce_mean(target_quantities[target], axis=-1)
                    elicited_statistic = computed_mean
                if moment == "sd":
                    computed_sd = tf.math.reduce_std(target_quantities[target], axis=-1)
                    elicited_statistic = computed_sd
                # save all moments in one tensor
                elicits_res[f"{elicit}.{moment}_{target}"] = elicited_statistic

        if elicit != "moments":
            elicits_res[f"{elicit}_{target}"] = elicited_statistic

    if global_dict["param_independence"]["independent"]:
        elicits_res["correlation"] = target_quantities["correlation"]

    # save file in object
    saving_path = global_dict["output_path"]["data"]
    if ground_truth:
        saving_path = saving_path + "/expert"
    path = saving_path + "/elicited_statistics.pkl"
    save_as_pkl(elicits_res, path)
    # return results
    return elicits_res
