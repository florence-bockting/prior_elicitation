import os
import pandas as pd
import tensorflow as tf

from elicit.plotting.sensitivity_func import (
    binomial_sensitivity,
    binomial_diagnostics,
    binomial_convergence,
    normals_convergence,
    normal_sensitivity,
)

# plot results of deep-binomial model
path_sim = "elicit/simulations/LiDO_cluster/sim_results/deep_prior/binomial"
path_expert = "elicit/simulations/LiDO_cluster/experts/deep_binomial"

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
                path_sim + f"/{all_files[i]}" + "/final_results.pkl")["loss"]
        )
        == 600
    ):
        prior_res = pd.read_pickle(
            path_sim + f"/{all_files[i]}" + "/prior_samples.pkl")
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

prior_expert = pd.read_pickle(path_expert + "/prior_samples.pkl")

# plot results
binomial_sensitivity(
    prior_expert,
    path_expert,
    path_sim,
    elicit_res_agg,
    prior_res_agg,
    cor_res_agg,
    save_fig=False,
)
binomial_diagnostics(path_sim, path_expert, f"/{all_files[3]}", save_fig=False)
binomial_convergence(path_sim, path_expert, f"/{all_files[3]}", save_fig=False)


# %% Independent, skewed Normal
scenario = "correlated"
path_sim = f"elicit/simulations/LiDO_cluster/sim_results/deep_prior/normal_{scenario}2" # noqa

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
                path_sim + f"/{all_files[i]}" + "/final_results.pkl")["loss"]
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
                path_sim + f"/{all_files[i]}" + "/elicited_statistics.pkl")[
                "quantiles_logR2"
            ]
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

prior_expert = pd.read_pickle(
    f"elicit/simulations/LiDO_cluster/experts/deep_{scenario}_normal/prior_samples.pkl" # noqa
)

path_expert = f"elicit/simulations/LiDO_cluster/experts/deep_{scenario}_normal"

normals_convergence(
    path_sim, path_expert, f"/normal_{scenario}_6", model=scenario,
    save_fig=False
)

normal_sensitivity(
    prior_expert,
    path_expert,
    prior_res_agg,
    elicits_gr1_agg,
    elicits_gr2_agg,
    elicits_gr3_agg,
    elicits_r2_agg,
    cor_res_agg,
    scenario,
    save_fig=False,
)
