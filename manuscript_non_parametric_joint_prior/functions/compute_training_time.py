import pandas as pd
import tensorflow as tf
import numpy as np


def min_sec(seconds):
    whole_min, secs = np.divmod(seconds, 60)
    return print(int(whole_min), ":", int(secs), " (min:sec)", sep="")


# %% Binomial model

# path to simulation results
path_sim_res = "elicit/simulations/LiDO_cluster/simulation_results/binomial"

# get total training time per replication
time_total = [
    tf.reduce_sum(
        pd.read_pickle(path_sim_res + f"/binomial_{i}/final_results.pkl")[
            "time_epoch"]
    )
    for i in range(30)
]

# compute mean and sd across all 30 replications
time_res = dict(
    mean_time=tf.reduce_mean(time_total), sd_time=tf.math.reduce_std(
        time_total)
)

# compute average min and secs required for training
min_sec(time_res["mean_time"].numpy())
min_sec(time_res["sd_time"].numpy())


# %% Normal models
# select normal scenario
scenario = "correlated"

# path to simulation results
path_sim_res = f"elicit/simulations/LiDO_cluster/simulation_results/normal_{scenario}" # noqa

# get total training time per replication
time_total = [
    tf.reduce_sum(
        pd.read_pickle(
            path_sim_res + f"/normal_{scenario}_{i}/final_results.pkl")[
            "time_epoch"
        ]
    )
    for i in range(30)
]

# compute mean and sd across all 30 replications
time_res = dict(
    mean_time=tf.reduce_mean(time_total), sd_time=tf.math.reduce_std(
        time_total)
)

# compute average min and secs required for training
min_sec(time_res["mean_time"].numpy())
min_sec(time_res["sd_time"].numpy())
