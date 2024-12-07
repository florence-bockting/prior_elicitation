# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import pickle
import os
import pandas as pd
import tensorflow as tf

from pythonjsonlogger import jsonlogger # noqa


def save_as_pkl(variable, path_to_file):
    """
    Helper functions to save a file as pickle.

    Parameters
    ----------
    variable : any
        file that needs to be saved.
    path_to_file : str
        path indicating the file location.

    Returns
    -------
    None.

    """
    # if directory does not exists, create it
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    # save file to location as pickle
    with open(path_to_file, "wb") as df_file:
        pickle.dump(variable, file=df_file)


def remove_unneeded_files(global_dict, save_results):
    path = global_dict["training_settings"]["output_path"]

    if not save_results["init_hyperparameters"]:
        os.remove(path + "/init_hyperparameters.pkl")
    if not save_results["init_hyperparameters"]:
        os.remove(path + "/prior_samples.pkl")


def save_hyperparameters(generator, epoch, global_dict):
    """
    extracts the learnable hyperparameter values from the model and saves them
    in appropriate form for post-analysis

    Parameters
    ----------
    generator : trainable tf.model
        initialized prior model used for training.
    epoch : int
        Current epoch.
    global_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    res_dict : dict
        learned values for each hyperparameter and epoch.

    """
    saving_path = global_dict["training_settings"]["output_path"]
    # extract learned hyperparameter values
    hyperparams = generator.trainable_variables
    if epoch == 0:
        # prepare list for saving hyperparameter values
        hyp_list = []
        for i in range(len(hyperparams)):
            hyp_list.append(hyperparams[i].name[:-2])
        # create a dict with empty list for each hyperparameter
        res_dict = {f"{k}": [] for k in hyp_list}
    # read saved list to add new values
    else:
        path_res_dict = saving_path + "/res_dict.pkl"
        res_dict = pd.read_pickle(rf"{path_res_dict}")
    # save names and values of hyperparameters
    vars_values = [
        hyperparams[i].numpy().copy() for i in range(len(hyperparams))
        ]
    vars_names = [
        hyperparams[i].name[:-2] for i in range(len(hyperparams))
        ]
    # create a final dict of hyperparameter values
    for val, name in zip(vars_values, vars_names):
        res_dict[name].append(val)
    # save result dictionary
    path_res_dict = saving_path + "/res_dict.pkl"
    save_as_pkl(res_dict, path_res_dict)
    return res_dict


def marginal_prior_moments(prior_samples, epoch, global_dict):
    """
    Used for summarizing learned prior distributions in the case of
    method='deep_prior'.
    Computes mean and standard deviation of the sampled marginal prior
    distributions for each epoch.

    Parameters
    ----------
    prior_samples : dict
        Samples from prior distributions.
    epoch : int
        current epoch.
    global_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    res_dict : dict
        returns mean (key:'means') and standard deviation (key:'stds')
        for each sampled marginal prior distribution; for each epoch.

    """
    saving_path = global_dict["training_settings"]["output_path"]
    if epoch == 0:
        res_dict = {"means": [], "stds": []}
    else:
        path_res_dict = saving_path + "/res_dict.pkl"
        res_dict = pd.read_pickle(rf"{path_res_dict}")

    means = tf.reduce_mean(prior_samples, (0, 1))
    sds = tf.reduce_mean(tf.math.reduce_std(prior_samples, 1), 0)
    for val, name in zip([means, sds], ["means", "stds"]):
        res_dict[name].append(val)
    # save result dictionary
    path_res_dict = saving_path + "/res_dict.pkl"
    save_as_pkl(res_dict, path_res_dict)
    return res_dict


def create_output_summary(path_res, global_dict):
    """
    Creates a text summary of all user inputs and some information about the
    learning process

    Parameters
    ----------
    path_res : str
        path to folder in which the results are saved.
    global_dict : dict
        global dictionary incl. all user specifications as created when call
        the prior_elicitation function.

    Returns
    -------
    txt file
        returns a text file summarizing all input information and some
        information about the learning process (e.g., wall time).

    """

    def summary_targets():
        df = pd.DataFrame()
        df["target quantities"] = []
        df["elicitation technique"] = []
        df["combine-loss"] = []
        for k in global_dict["target_quantities"]:
            df["target quantities"].append(k)
            df["elicitation technique"].append(
                global_dict["target_quantities"][k]["elicitation_method"]
                )
            df["combine-loss"].append(
                global_dict["target_quantities"][k]["loss_components"]
                )
        return df

    loss_comp = pd.read_pickle(path_res + "/loss_components.pkl")

    df2 = pd.DataFrame()
    df2["loss components"] = list(loss_comp.keys())
    df2["shape"] = [
        list(loss_comp[key].shape) for key in list(loss_comp.keys())
        ]

    time = (
        tf.reduce_sum(
            pd.read_pickle(path_res + "/final_results.pkl")["time_epoch"]
            ) / 60.0)
    min, sec = tuple(f"{time:.2f}".split("."))

    optimizer_dict = {}
    if (
        type(global_dict["optimization_settings"][
            "optimizer_specs"
            ]["learning_rate"]) is float
    ):
        optimizer_dict["init_lr"] = global_dict["optimization_settings"][
            "optimizer_specs"
        ]["learning_rate"]
    elif (
        global_dict["optimization_settings"]["optimizer_specs"][
            "learning_rate"
        ]._keras_api_names[0]
        == "keras.optimizers.schedules.CosineDecayRestarts"
    ):
        optimizer_dict["lr_scheduler"] = global_dict["optimization_settings"][
            "optimizer_specs"
        ]["learning_rate"]._keras_api_names[0]
        optimizer_dict["init_lr"] = global_dict["optimization_settings"][
            "optimizer_specs"
        ]["learning_rate"].initial_learning_rate
        optimizer_dict["decay_steps"] = global_dict["optimization_settings"][
            "optimizer_specs"
        ]["learning_rate"].first_decay_steps
    elif (
        global_dict["optimization_settings"]["optimizer_specs"][
            "learning_rate"
        ]._keras_api_names[0]
        == "keras.optimizers.schedules.ExponentialDecay"
    ):
        optimizer_dict["lr_scheduler"] = global_dict["optimization_settings"][
            "optimizer_specs"
        ]["learning_rate"]._keras_api_names[0]
        optimizer_dict["init_lr"] = global_dict["optimization_settings"][
            "optimizer_specs"
        ]["learning_rate"].initial_learning_rate
        optimizer_dict["decay_rate"] = global_dict["optimization_settings"][
            "optimizer_specs"
        ]["learning_rate"].decay_rate
        optimizer_dict["decay_steps"] = global_dict["optimization_settings"][
            "optimizer_specs"
        ]["learning_rate"].decay_steps
    else:
        optimizer_dict["lr_scheduler"] = global_dict["optimization_settings"][
            "optimizer_specs"
        ]["learning_rate"]
    if global_dict["training_settings"]["method"] == "deep_prior":
        # create sub(-sub)dict for better readability
        dict_nf = global_dict['normalizing_flow']
        dict_nf_copl = dict_nf['coupling_settings']

        def method_settings():

            return str(
                "\nNormalizing Flow"
                + "\n---------------- \n"
                + f"number coupling layers={dict_nf['num_coupling_layers']}\n"
                + f"coupling design={dict_nf['coupling_design']}\n"
                + f"units in dense layer=\
                    {dict_nf_copl['dense_args']['units']}\n"
                + f"activation function=\
                    {dict_nf_copl['dense_args']['activation']}\n"
                + f"number dense layers={dict_nf_copl['num_dense']}\n"
                + f"permutation={dict_nf['permutation']}\n"
                + f"base distribution family=\
                    {dict_nf['base_distribution']._name}\n"
            )

    else:
        # create subdict for better readability
        dict_param = global_dict['model_parameters']

        def method_settings():
            param_names = sorted(
                list(
                    set(
                        dict_param.keys()
                        ).difference(set(["independence", "no_params"]))
                    )
                )
            family_dict = {
                f"{param}": dict_param[param]["family"].__name__
                for param in param_names
            }
            init_dict = {}
            for param in param_names:
                for k in dict_param[param]["hyperparams_dict"]:
                    init_info = dict_param[param]["hyperparams_dict"][k]
                    init_dict[k] = init_info

            return str(
                "\nParametric Prior"
                + "\n---------------- \n"
                + f"distribution family={family_dict}\n"
                + f"initialization={init_dict}\n"
            )

    if global_dict["expert_data"]["from_ground_truth"]:
        param_list = global_dict["expert_data"]["simulator_specs"].keys()
        true_info2 = {}
        for key in param_list:
            true_info = global_dict["expert_data"]["simulator_specs"][
                key
            ].parameters.copy()
            true_info.pop("validate_args", None)
            true_info.pop("allow_nan_stats", None)
            true_info.pop("force_probs_to_zero_outside_support", None)
            true_info2[key] = true_info

        expert_info = [
            str(f"{key}={true_info2[key]}") for key in true_info2.keys()
            ]
    else:
        expert_info = ["see file (ToDo)"]

    output_summary = str(
        "General summary"
        + "\n---------------- \n"
        + f"method={global_dict['training_settings']['method']}\n"
        + f"sim_id={global_dict['training_settings']['sim_id']}\n"
        + f"seed={global_dict['seed']}\n"
        + f"B={global_dict['training_settings']['B']}\n"
        + f"rep={global_dict['training_settings']['simulations_from_prior']}\n"
        + f"epochs={global_dict['training_settings']['epochs']}\n"
        + f"wall time={min}:{sec} (min:sec)\n"
        + f"optimizer={global_dict['optimization_settings']['optimizer']}\n"
        + f"learning rate={optimizer_dict}\n"
        + f"use_regularizer={global_dict['use_regularizer']}\n"
        + "\nModel info"
        + "\n---------------- \n"
        + f"model name={global_dict['generative_model']['model_function']}\n"
        + f"model parameters={list(dict_param.keys())}\n"
        + f"model parameter scaling=\
            {[dict_param[k]['param_scaling'] for k in dict_param]}\n"
        + f"model parameter independent={dict_param['independence']}\n"
        + "\nExpert info"
        + "\n---------------- \n"
        + f"{expert_info}\n"
        + method_settings()
        + "\nTarget quantities and elicitation techniques"
        + "\n--------------------- \n"
        + f"\n{summary_targets()}\n"
        + "\nLoss components"
        + "\n--------------------- \n"
        + f"\n{df2}"
    )
    return output_summary


def write_res_summary(path_res, global_dict):
    """
    saves the summary of the user inputs in a text file

    Parameters
    ----------
    path_res : str
        path to location where results are saved.
    global_dict : dict
        dictionary containing all user specifications.

    """
    f = open(path_res + "/overview.txt", "w")
    output_summary = create_output_summary(path_res, global_dict)
    f.write(output_summary)
    f.close()


def print_res_summary(path_res, global_dict):
    """
    Prints the summary output without saving it to a particular location.

    Parameters
    ----------
    path_res : str
        path to location where results are saved.
    global_dict : dict
        global dictionary containing all user specifications.

    """
    print(create_output_summary(path_res, global_dict))
