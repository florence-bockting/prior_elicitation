import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calc_slope(res, start, end):
    from scipy.stats import linregress

    Y = res[start:end]
    X = range(start, end)
    slope = linregress(X, Y)[0]

    return slope


def plot_conv_diagnostics(
    path_sim,
    start,
    end,
    last_vals,
    scenario=None,
    max_seeds=4,
    factor=100,
    num_seeds=30,
):

    all_files = os.listdir(path_sim)

    seed = []
    slopes = []
    slopes_orig = []
    for i in range(num_seeds):
        res_loss = tf.stack(
            pd.read_pickle(
                path_sim + f"/{all_files[i]}" + "/final_results.pkl")["loss"]
        )
        slope_abs = abs(calc_slope(res_loss, start, end)) * factor
        slope = calc_slope(res_loss, start, end) * factor
        slopes.append(slope_abs)
        slopes_orig.append(slope)
        seed.append(int(f"/{all_files[i]}".split("_")[-1]))

    res_dict = {"seed": seed, "slope": slopes, "slope_o": slopes_orig}

    res_sorted = pd.DataFrame(res_dict).sort_values(by="slope")
    res_highest = res_sorted[-max_seeds:]
    res_lowest = res_sorted[0:0 + 1]
    res_single = pd.concat([res_lowest, res_highest])

    single_losses = []
    for i in res_single["seed"].index:
        res_loss = tf.stack(
            pd.read_pickle(
                path_sim + f"/{all_files[i]}" + "/final_results.pkl")["loss"]
        )
        single_losses.append(res_loss)

    fig = plt.figure(layout="constrained", figsize=(6, 3.5))
    subfigs = fig.subfigures(2, 1, wspace=0.07)
    subfig0 = subfigs[0].subplots(1, 1)
    subfig1 = subfigs[1].subplots(1, 5, sharey=True, sharex=True)

    subfig0.axhline(0, linestyle="dashed", color="black", lw=1)
    subfig0.plot(range(num_seeds), res_sorted["slope"], "o", color="grey")
    subfig0.plot(
        range(num_seeds - max_seeds, num_seeds),
        res_highest["slope"],
        "o",
        color="#21284f",
    )
    subfig0.plot([0], res_lowest["slope"], "o", color="#2e5c2c")
    subfig0.set_xticks(range(num_seeds), res_sorted["seed"], fontsize="small")
    subfig0.spines[["right", "top"]].set_visible(False)
    subfig0.set_ylabel("|slope|*100", fontsize="small")
    subfig0.set_xlabel("seed", fontsize="small")
    subfig0.yaxis.set_tick_params(labelsize=7)
    subfig0.xaxis.set_tick_params(labelsize=7)
    subfig0.set_ylim(-0.001, 0.02)

    for i, c in enumerate(["#2e5c2c"] + ["#21284f"] * 4):
        subfig1[i].plot(single_losses[i][-last_vals:], color="grey")
        subfig1[i].plot(
            range(last_vals - 100, last_vals), single_losses[i][-100:], color=c
        )
        subfig1[i].plot(
            [last_vals - 100, last_vals],
            [
                np.mean(single_losses[i][-105:-95]),
                (res_single.iloc[i]["slope_o"] / 100) * 100
                + np.mean(single_losses[i][-105:-95]),
            ],
            "-",
            color="red",
        )
        subfig1[i].set_title(f"seed: {res_single['seed'].iloc[i]}",
                             fontsize="medium")
        subfig1[i].set_xticks(
            np.linspace(0, last_vals, 3),
            np.linspace(end - last_vals, end, 3, dtype=int),
            fontsize="x-small",
        )
        subfig1[i].spines[["right", "top"]].set_visible(False)
        subfig1[i].set_xlabel("epochs", fontsize="small")
    subfig1[0].yaxis.set_tick_params(labelsize=7)
    subfig1[0].set_ylabel(r"$L(\lambda)$", fontsize="small")
