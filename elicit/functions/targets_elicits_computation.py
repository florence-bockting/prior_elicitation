import tensorflow as tf
import tensorflow_probability as tfp
import bayesflow as bf
import inspect
import pandas as pd

from elicit.functions.helper_functions import save_as_pkl
from elicit.user.custom_functions import custom_correlation

tfd = tfp.distributions
bfn = bf.networks


# TODO: Update Custom Target Function
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
    # create a dict with arguments from model simulations and custom args
    # for custom func
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
                global_dict["output_path"] + "/expert/model_simulations.pkl"
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


def computation_target_quantities(model_simulations, ground_truth,
                                  global_dict):
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
    # create sub-dictionaries for readability
    target_dict = global_dict["target_quantities"]
    # initialize dict for storing results
    targets_res = dict()
    # loop over target quantities
    for i, tar in enumerate(target_dict):
        # use custom function for target quantity if it has been defined
        if (
            target_dict[tar]["custom_target_function"]
            is not None
        ):
            target_quantity = use_custom_functions(
                target_dict[tar]["custom_target_function"],
                model_simulations,
                global_dict,
            )
        else:
            target_quantity = model_simulations[tar]

        # save target quantities
        targets_res[tar] = target_quantity

    if global_dict["model_parameters"]["independence"] is not None:
        target_quantity = use_custom_functions(
            {"function": custom_correlation, "additional_args": None},
            model_simulations,
            global_dict,
        )
        targets_res["correlation"] = target_quantity
    # save file in object
    saving_path = global_dict["output_path"]
    if ground_truth:
        saving_path = saving_path + "/expert"
    path = saving_path + "/target_quantities.pkl"
    save_as_pkl(targets_res, path)
    # return results
    return targets_res


def computation_elicited_statistics(target_quantities, ground_truth,
                                    global_dict):
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
    # create sub-dictionaries for readability
    target_dict = global_dict["target_quantities"]
    # initialize dict for storing results
    elicits_res = dict()
    # loop over elicitation techniques
    for tar in sorted(
            list(set(target_quantities).difference(set(["correlation"])))
            ):
        # use custom method if specified otherwise use built-in methods
        if (
            target_dict[tar]["custom_elicitation_method"]
            is not None
        ):
            elicited_statistic = use_custom_functions(
                target_dict[tar]["custom_elicitation_method"],
                target_quantities,
            )
            elicits_res[f"custom_{tar}"] = elicited_statistic

        else:
            if (
                target_dict[tar]["elicitation_method"]
                == "histogram"
            ):
                quantiles_hist = list(tf.range(2, 100, 2))
                target_hist = tfp.stats.percentile(
                    target_quantities[tar], q=quantiles_hist, axis=-1
                )
                elicited_statistic = tf.einsum("ij...->ji...", target_hist)
                elicits_res[f"histogram_{tar}"] = elicited_statistic

            if (
                target_dict[tar]["elicitation_method"]
                == "quantiles"
            ):
                quantiles = target_dict[tar]["quantiles_specs"]

                # reshape target quantity
                if tf.rank(target_quantities[tar]) == 3:
                    quan_reshaped = tf.reshape(
                        target_quantities[tar],
                        shape=(
                            target_quantities[tar].shape[0],
                            target_quantities[tar].shape[1]
                            * target_quantities[tar].shape[2],
                        ),
                    )
                if tf.rank(target_quantities[tar]) == 2:
                    quan_reshaped = target_quantities[tar]

                # compute quantiles
                computed_quantiles = tfp.stats.percentile(
                    quan_reshaped, q=quantiles, axis=-1
                )
                # bring quantiles to the last dimension
                elicited_statistic = tf.einsum("ij...->ji...",
                                               computed_quantiles)
                elicits_res[f"quantiles_{tar}"] = elicited_statistic

            if (
                target_dict[tar]["elicitation_method"]
                == "moments"
            ):
                moments = target_dict[tar]["moments_specs"]

                # for each moment
                # TODO: implement feature for custom moment functions
                for mom in moments:
                    # check whether moment is supported
                    assert mom in [
                        "sd",
                        "mean",
                    ], "currently only 'mean' and 'sd' are supported as \
                        moments"

                    if mom == "mean":
                        computed_mean = tf.reduce_mean(
                            target_quantities[tar], axis=-1
                        )
                        elicited_statistic = computed_mean
                    if mom == "sd":
                        computed_sd = tf.math.reduce_std(
                            target_quantities[tar], axis=-1
                        )
                        elicited_statistic = computed_sd
                    # save all moments in one tensor
                    elicit = target_dict[tar][
                        "elicitation_method"
                    ]
                    elicits_res[f"{elicit}.{mom}_{tar}"] = elicited_statistic

    if global_dict["model_parameters"]["independence"] is not None:
        elicits_res["correlation"] = target_quantities["correlation"]

    # save file in object
    saving_path = global_dict["output_path"]
    if ground_truth:
        saving_path = saving_path + "/expert"
    path = saving_path + "/elicited_statistics.pkl"
    save_as_pkl(elicits_res, path)

    # return results
    return elicits_res
