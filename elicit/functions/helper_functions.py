import pickle
import os
import pandas as pd
import tensorflow as tf


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
    saving_path = global_dict["output_path"]
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
    saving_path = global_dict["output_path"]
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
