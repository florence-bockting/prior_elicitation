import os
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def run_model_averaging(path, B, sim_from_prior, num_param):
    all_files = os.listdir(path)

    prior_list = []
    loss_dict = {"loss": []}
    for i in range(len(all_files)):
        res = pd.read_pickle(path + f"/{all_files[i]}" + "/final_results.pkl")
        prior_list.append(
            pd.read_pickle(path + f"/{all_files[i]}" + "/prior_samples.pkl")
        )
        loss_dict["loss"].append(tf.stack(res["loss"])[-100:].numpy())

    # compute final loss per run by averaging over last x values
    mean_losses = np.stack(
        [np.mean(loss_dict["loss"][i]) for i in range(len(all_files))]
    )
    # retrieve min MMD
    min_loss = min(mean_losses)
    # compute Delta_i MMD
    delta_MMD = mean_losses - min_loss
    # relative likelihood
    rel_likeli = np.exp(1 * delta_MMD)
    # compute Akaike weights
    w_MMD = rel_likeli / np.sum(rel_likeli)
    # get minimum weight (for plotting)
    min_weight = np.argmin(w_MMD)

    # model averaging
    # extract prior samples; shape = (num_sims, B*sim_prior, num_param)
    prior_samples = np.stack(
        tf.reshape(prior_list, (len(w_MMD), B * sim_from_prior, num_param))
    )

    average_prior = []

    for i in range(10_000):
        # sample component
        sampled_component = np.random.choice(
            np.arange(len(w_MMD)), size=1, replace=True, p=w_MMD
        )
        # sample observation index
        sampled_obs = np.random.choice(np.arange(sim_from_prior), size=1,
                                       replace=True)
        # select prior
        sampled_prior = prior_samples[sampled_component, sampled_obs, :]
        # store results
        average_prior.append(sampled_prior)

    averaged_priors = np.concatenate(average_prior)
    return (w_MMD, prior_samples, averaged_priors, min_weight)


def plot_learned_priors(
    w_MMD, prior_samples, averaged_priors, min_weight, prior_samples_expert
):
    # plot average and single priors
    fig = plt.figure(layout="constrained", figsize=(7, 3.5))
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=0.16)
    subfig0 = subfigs[0].subplots(1, 1)
    subfig1 = subfigs[1].subplots(1, 2)

    sns.barplot(w_MMD, ax=subfig0, color="darkgrey")
    subfig0.set_ylim(0, 0.04)
    subfig0.set_xlabel("seed")
    subfig0.set_ylabel("model weights")
    [subfig0.tick_params(axis=ax, labelsize="x-small") for ax in ["x", "y"]]
    subfig0.spines[["right", "top"]].set_visible(False)

    for k, t in zip(range(2), [r"$\beta_0$", r"$\beta_1$"]):
        for i in range(prior_samples.shape[0]):
            sns.kdeplot(
                prior_samples[i, :, k],
                color="lightgrey",
                alpha=0.5,
                ax=subfig1[k],
                lw=3,
            )
            subfig1[k].spines[["right", "top"]].set_visible(False)
        sns.kdeplot(
            prior_samples_expert[0, :, k],
            color="black",
            linestyle=(0, (1, 1)),
            lw=2,
            ax=subfig1[k],
            alpha=0.6,
        )
        sns.kdeplot(
            averaged_priors[:, k],
            color="black",
            linestyle=(0, (5, 1)),
            ax=subfig1[k],
            lw=2,
            alpha=0.6,
        )
        [subfig1[k].tick_params(axis=ax, labelsize="x-small") for
         ax in ["x", "y"]]
        subfig1[k].set_xlabel(t)
    subfigs[0].suptitle(r"$\bf{(a)}$" + " Model averaging weights", x=0.17,
                        y=1.1)
    subfigs[1].suptitle(r"$\bf{(b)}$" + " Marginal priors", x=0.12, y=1.1)
    for x, t, c in zip(
        [0.01, 0.08, 0.21, 0.32],
        [
            "legend:",
            "prior per seed",
            r"average $\mathbf{- -}$",
            r"truth $\mathbf{\cdot\cdot\cdot}$",
        ],
        ["black", "grey", "black", "black"],
    ):
        subfigs[1].text(x, -0.08, t, color=c, fontsize="small")
    subfig1[0].set_ylabel("density", fontsize="x-small")
    subfig1[1].set_ylabel(" ", fontsize="x-small")
    subfigs[1].patches.extend(
        [
            plt.Rectangle(
                (8, 9),
                410,
                30,
                fill=False,
                color="grey",
                alpha=0.2,
                zorder=-1,
                transform=None,
                figure=subfigs[1],
            )
        ]
    )
