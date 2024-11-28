import os
import pandas as pd
import tensorflow as tf


def prep_sim_res(path_sim, scenario):
    all_files = os.listdir(path_sim)

    prior_res_list = []
    cor_res_list = []
    elicits_gr1_list = []
    elicits_gr2_list = []
    elicits_gr3_list = []
    elicits_r2_list = []
    for i in range(len(all_files)):
        if (
            len(
                pd.read_pickle(
                    path_sim + f"/{all_files[i]}" + "/final_results.pkl")[
                    "loss"
                ]
            )
            == 1500
        ):
            prior_res = pd.read_pickle(
                path_sim + f"/{all_files[i]}" + "/model_simulations.pkl"
            )["prior_samples"]
            elicits_gr1 = pd.read_pickle(
                path_sim + f"/{all_files[i]}" + "/elicited_statistics.pkl"
            )["quantiles_group1"]
            elicits_gr2 = pd.read_pickle(
                path_sim + f"/{all_files[i]}" + "/elicited_statistics.pkl"
            )["quantiles_group2"]
            elicits_gr3 = pd.read_pickle(
                path_sim + f"/{all_files[i]}" + "/elicited_statistics.pkl"
            )["quantiles_group3"]
            elicits_r2 = tf.exp(
                pd.read_pickle(
                    path_sim + f"/{all_files[i]}" + "/elicited_statistics.pkl"
                )["quantiles_logR2"]
            )
            if scenario == "correlated":
                cor_res = pd.read_pickle(
                    path_sim + f"/{all_files[i]}" + "/elicited_statistics.pkl"
                )["identity_correl"]
            else:
                cor_res = pd.read_pickle(
                    path_sim + f"/{all_files[i]}" + "/elicited_statistics.pkl"
                )[
                    "correlation"
                ]  # histogram_correl
            prior_res_list.append(prior_res[0, :, :])
            cor_res_list.append(cor_res[0, :])
            elicits_gr1_list.append(elicits_gr1[0, :])
            elicits_gr2_list.append(elicits_gr2[0, :])
            elicits_gr3_list.append(elicits_gr3[0, :])
            elicits_r2_list.append(elicits_r2[0, :])

    cor_res_agg = tf.stack(cor_res_list, -1)
    prior_res_agg = tf.stack(prior_res_list, -1)
    elicits_gr1_agg = tf.stack(elicits_gr1_list, -1)
    elicits_gr2_agg = tf.stack(elicits_gr2_list, -1)
    elicits_gr3_agg = tf.stack(elicits_gr3_list, -1)
    elicits_r2_agg = tf.stack(elicits_r2_list, -1)

    return (
        cor_res_agg,
        prior_res_agg,
        elicits_gr1_agg,
        elicits_gr2_agg,
        elicits_gr3_agg,
        elicits_r2_agg,
    )
