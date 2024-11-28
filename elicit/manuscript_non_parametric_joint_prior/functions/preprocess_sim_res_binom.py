import os
import pandas as pd
import tensorflow as tf


def prep_sim_res_binom(path_sim):
    all_files = os.listdir(path_sim)

    # preprocessing
    prior_res_list = []
    cor_res_list = []
    elicit_res_list = []
    mean_res_list = []
    sd_res_list = []
    for i in range(len(all_files)):
        if (
            len(
                pd.read_pickle(
                    path_sim + f"/{all_files[i]}" + "/final_results.pkl")[
                    "loss"
                ]
            )
            == 600
        ):
            prior_res = pd.read_pickle(
                path_sim + f"/{all_files[i]}" + "/prior_samples.pkl"
            )
            means_res = tf.stack(
                pd.read_pickle(
                    path_sim + f"/{all_files[i]}" + "/final_results.pkl")[
                    "hyperparameter"
                ]["means"],
                0,
            )
            sds_res = tf.stack(
                pd.read_pickle(
                    path_sim + f"/{all_files[i]}" + "/final_results.pkl")[
                    "hyperparameter"
                ]["stds"],
                0,
            )
            elicit_res = pd.read_pickle(
                path_sim + f"/{all_files[i]}" + "/elicited_statistics.pkl"
            )["custom_ypred"]
            cor_res = pd.read_pickle(
                path_sim + f"/{all_files[i]}" + "/elicited_statistics.pkl"
            )["correlation"]

            prior_res_list.append(prior_res[0, :, :])
            cor_res_list.append(cor_res[0, :])
            elicit_res_list.append(elicit_res[0, :])
            mean_res_list.append(means_res)
            sd_res_list.append(sds_res)

    cor_res_agg = tf.stack(cor_res_list, -1)
    prior_res_agg = tf.stack(prior_res_list, -1)
    elicit_res_agg = tf.stack(elicit_res_list, -1)
    mean_res_agg = tf.stack(mean_res_list, -1)
    sd_res_agg = tf.stack(sd_res_list, -1)

    return (cor_res_agg, prior_res_agg, elicit_res_agg, mean_res_agg,
            sd_res_agg)
