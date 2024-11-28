import tensorflow as tf
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow_probability as tfp
from itertools import product


def cor(idx, x):
    return np.corrcoef(x=x[:, idx[0]], y=x[:, idx[1]])[0, 1]


def binomial_convergence(sim_path, expert_path, file, save_fig=True):
    final_res = pd.read_pickle(sim_path + file + "/final_results.pkl")
    total_loss = tf.stack(final_res["loss"], -1)
    component_loss = tf.stack(final_res["loss_component"], -1)
    marginal_means = tf.stack(final_res["hyperparameter"]["means"], 0)
    marginal_sds = tf.stack(final_res["hyperparameter"]["stds"], 0)

    exp_priors = pd.read_pickle(expert_path + "/prior_samples.pkl")
    mod_priors = pd.read_pickle(sim_path + file + "/prior_samples.pkl")
    exp_means = tf.reduce_mean(exp_priors, (0, 1))
    exp_sds = tf.math.reduce_std(exp_priors, (0, 1))

    cmap = mpl.colormaps["gray"]
    col_losses = cmap(np.linspace(0.0, 0.8, component_loss.shape[0]))
    col_betas = ["#b85420", "#205e78"]
    col_exp = "#7d1717"
    col_mod = "#e2a854"

    fig = plt.figure(layout="constrained", figsize=(4, 4))
    subfigs = fig.subfigures(2, 1, hspace=0.03, height_ratios=[1, 0.8])
    axs0 = subfigs[0].subplots(2, 2, sharex=True)
    axs1 = subfigs[1].subplots(2, 2)

    axs0[0, 0].plot(range(total_loss.shape[0]), total_loss, color="black")
    [
        axs0[0, 1].plot(
            range(component_loss.shape[1]), component_loss[i, :],
            color=col_losses[i, :]
        )
        for i in range(component_loss.shape[0])
    ]
    [axs0[0, i].set_yscale("log") for i in range(2)]
    [
        axs0[0, i].set_title(t, fontsize="small")
        for i, t in enumerate(["total loss", "loss components"])
    ]
    [
        axs0[1, 0].axhline(exp_means[i], color="black", linestyle="dashed",
                           lw=1)
        for i in range(2)
    ]
    [
        axs0[1, 1].axhline(exp_sds[i], color="black", linestyle="dashed", lw=1)
        for i in range(2)
    ]
    [
        axs0[1, 0].plot(
            range(marginal_means.shape[0]),
            marginal_means[:, i],
            label=rf"m$(\theta_{i})$",
            color=col_betas[i],
        )
        for i in range(2)
    ]
    [
        axs0[1, 1].plot(
            range(marginal_sds.shape[0]),
            marginal_sds[:, i],
            label=rf"sd$(\theta_{i})$",
            color=col_betas[i],
        )
        for i in range(2)
    ]
    axs0[1, 0].legend(
        handlelength=0.5, ncol=2, fontsize="x-small", frameon=False,
        loc="center right"
    )
    axs0[1, 1].legend(
        handlelength=0.5, ncol=2, fontsize="x-small", frameon=False,
        loc="upper right"
    )
    [
        axs0[j, i].spines[["right", "top"]].set_visible(False)
        for j, i in product([0, 1], [0, 1])
    ]
    [axs0[j, i].yaxis.set_tick_params(labelsize=7) for
     j, i in product([0, 1], [0, 1])]
    [axs0[1, i].xaxis.set_tick_params(labelsize=7) for i in range(2)]
    [
        axs0[1, i].set_title(t, fontsize="small")
        for i, t in enumerate(["marginal means", "marginal std. dev."])
    ]
    [axs0[1, i].set_xlabel("epochs", fontsize="x-small") for i in range(2)]

    sns.kdeplot(
        exp_priors[0, :, 0], color=col_exp, ax=axs1[0, 0], lw=2, label="true",
        zorder=1
    )
    sns.kdeplot(
        exp_priors[0, :, 1], color=col_exp, ax=axs1[1, 1], lw=2, label="true",
        zorder=1
    )
    sns.kdeplot(mod_priors[0, :, 1], lw=3, color=col_mod, ax=axs1[1, 1],
                zorder=0)
    sns.kdeplot(
        mod_priors[0, :, 0],
        lw=3,
        color=col_mod,
        label="learned",
        ax=axs1[0, 0],
        zorder=0,
    )
    axs1[0, 0].spines[["right", "top"]].set_visible(False)
    axs1[1, 1].spines[["right", "top"]].set_visible(False)
    sns.scatterplot(
        x=exp_priors[0, :, 0],
        y=exp_priors[0, :, 1],
        color=col_exp,
        ax=axs1[0, 1],
        marker="X",
        s=7,
        lw=0,
        alpha=0.1,
    )
    sns.kdeplot(
        x=mod_priors[0, :, 0],
        y=mod_priors[0, :, 1],
        color=col_mod,
        ax=axs1[0, 1],
        alpha=0.7,
    )
    axs1[0, 1].spines[["right", "top"]].set_visible(False)
    axs1[0, 0].set_title(r"$\beta_0$", fontsize="small")
    axs1[0, 1].set_title(r"$\beta_1$", fontsize="small")
    [axs1[0, i].set_ylabel(" ") for i in range(2)]
    axs1[1, 1].set_ylabel(" ")
    axs1[1, 0].spines[["right", "top", "left", "bottom"]].set_visible(False)
    axs1[1, 0].get_yaxis().set_visible(False)
    axs1[1, 0].get_xaxis().set_visible(False)
    axs1[1, 0].text(
        0.5,
        0.3,
        f"corr:\n{cor([0,1], mod_priors[0,:,:]):.2f}",
        color="black",
        fontsize="small",
        ha="center",
    )
    [axs1[j, i].xaxis.set_tick_params(labelsize=7) for
     i, j in product([0, 1], [0, 1])]
    [axs1[j, i].yaxis.set_tick_params(labelsize=7) for
     i, j in product([0, 1], [0, 1])]
    axs1[0, 0].legend(
        handlelength=0.3, ncol=1, fontsize="x-small", frameon=False,
        loc=(0.01, 0.4)
    )
    subfigs[1].suptitle(
        r"$\mathbf{(b)}$" + f" Joint prior distribution (seed: {file.split('_')[-1]})", # noqa
        ha="left",
        x=0.01,
        fontsize="medium",
    )
    subfigs[0].suptitle(
        r"$\mathbf{(a)}$" + f" Convergence diagnostics (seed: {file.split('_')[-1]})", # noqa
        ha="left",
        x=0.01,
        fontsize="medium",
    )
    if save_fig:
        plt.savefig(
            "elicit/simulations/LiDO_cluster/sim_results/deep_prior/graphics/convergence_binom.png", # noqa
            dpi=300,
        )
    else:
        plt.show()


def binomial_diagnostics(sim_path, expert_path, file, save_fig=True):
    final_res = pd.read_pickle(sim_path + file + "/final_results.pkl")
    total_loss = tf.stack(final_res["loss"], -1)
    component_loss = tf.stack(final_res["loss_component"], -1)
    marginal_means = tf.stack(final_res["hyperparameter"]["means"], 0)
    marginal_sds = tf.stack(final_res["hyperparameter"]["stds"], 0)
    expert_res = pd.read_pickle(expert_path + "/prior_samples.pkl")
    exp_means = tf.reduce_mean(expert_res, (0, 1))
    exp_sds = tf.math.reduce_std(expert_res, (0, 1))

    cmap = mpl.colormaps["gray"]
    col_losses = cmap(np.linspace(0.0, 0.8, component_loss.shape[0]))
    col_betas = ["#b85420", "#205e78"]

    fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(5, 2),
                            sharex=True)
    axs[0, 0].plot(range(total_loss.shape[0]), total_loss, color="black")
    [
        axs[0, 1].plot(
            range(component_loss.shape[1]), component_loss[i, :],
            color=col_losses[i, :]
        )
        for i in range(component_loss.shape[0])
    ]
    # [axs[0,i].set_yscale('log') for i in range(2)]
    [
        axs[0, i].set_title(t, fontsize="small")
        for i, t in enumerate(["total loss", "loss components"])
    ]
    [
        axs[1, 0].axhline(exp_means[i], color="black", linestyle="dashed",
                          lw=1)
        for i in range(2)
    ]
    [
        axs[1, 1].axhline(exp_sds[i], color="black", linestyle="dashed", lw=1)
        for i in range(2)
    ]
    [
        axs[1, 0].plot(
            range(marginal_means.shape[0]),
            marginal_means[:, i],
            label=rf"m$(\theta_{i})$",
            color=col_betas[i],
        )
        for i in range(2)
    ]
    [
        axs[1, 1].plot(
            range(marginal_sds.shape[0]),
            marginal_sds[:, i],
            label=rf"sd$(\theta_{i})$",
            color=col_betas[i],
        )
        for i in range(2)
    ]
    [
        axs[1, i].legend(handlelength=0.5, ncol=2, fontsize="small",
                         frameon=False)
        for i in range(2)
    ]
    [
        axs[j, i].spines[["right", "top"]].set_visible(False)
        for j, i in product([0, 1], [0, 1])
    ]
    [axs[j, i].yaxis.set_tick_params(labelsize=7) for
     j, i in product([0, 1], [0, 1])]
    [axs[1, i].xaxis.set_tick_params(labelsize=7) for i in range(2)]
    [
        axs[1, i].set_title(t, fontsize="small")
        for i, t in enumerate(["marginal means", "marginal std. dev."])
    ]
    [axs[1, i].set_xlabel("epochs", fontsize="x-small") for i in range(2)]
    if save_fig:
        plt.savefig(
            "elicit/simulations/LiDO_cluster/sim_results/deep_prior/graphics/diagnostics_binom.png", # noqa
            dpi=300,
        )
    else:
        plt.show()


def binomial_sensitivity(
    prior_expert,
    path_expert,
    path_sim,
    elicit_res_agg,
    prior_res_agg,
    cor_res_agg,
    save_fig=True,
):
    col_mod = "#ba6b34"

    exp_elicits = pd.read_pickle(path_expert + "/elicited_statistics.pkl")[
        "custom_ypred"
    ]
    mod_elicits = elicit_res_agg
    cor_expert = tfp.stats.correlation(prior_expert, sample_axis=1,
                                       event_axis=-1)[
        :, 1, 0
    ]

    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(5, 1.5))

    for b in range(mod_elicits.shape[-1]):
        [
            sns.scatterplot(
                x=exp_elicits[0, :, i],
                y=mod_elicits[:, i, b],
                ax=axs[i],
                color=col_mod,
                lw=0,
                alpha=0.1,
                zorder=1,
            )
            for i in range(2)
        ]
    for i in range(2):
        axs[i].axline(
            (0, 0), slope=1, color="black", lw=1, linestyle="dashed", zorder=0
        )
        axs[i].xaxis.set_tick_params(labelsize=7)
        axs[i].yaxis.set_tick_params(labelsize=7)
        axs[i].set_xlabel("true", fontsize="small")
        axs[i].spines[["right", "top"]].set_visible(False)
    axs[0].set_ylabel("learned", fontsize="small")
    axs[2].set_xlabel(r"$\rho$", fontsize="small")
    axs[1].set_title(r"$y \mid x_1$", fontsize="small")
    axs[2].set_title(r"$r(\beta_0,\beta_1)$", fontsize="small")
    axs[0].set_title(r"$y \mid x_0$", fontsize="small")
    sns.scatterplot(
        x=cor_expert,
        y=tf.ones(cor_expert.shape),
        color="#7d1717",
        ax=axs[2],
        zorder=2,
        marker="o",
        label="true",
    )
    sns.scatterplot(
        x=cor_res_agg[0, :],
        y=tf.ones(cor_res_agg.shape[-1]),
        color="#e2a854",
        ax=axs[2],
        zorder=1,
        marker="o",
        alpha=0.6,
        label="learned",
    )
    axs[2].spines[["right", "top"]].set_visible(False)
    axs[2].xaxis.set_tick_params(labelsize=7)
    axs[2].yaxis.set_tick_params(labelsize=7)
    axs[2].set_yticklabels("")
    axs[2].set_xlim(-1, 1)
    axs[2].legend(
        handlelength=0.2, ncol=2, fontsize="x-small", frameon=False,
        loc=(0.03, 0.75)
    )
    if save_fig:
        plt.savefig(
            "elicit/simulations/LiDO_cluster/sim_results/deep_prior/graphics/sensitivity_binom.png", # noqa
            dpi=300,
        )
    else:
        plt.show()


def normals_convergence(sim_path, expert_path, file, model, save_fig=True):
    final_res = pd.read_pickle(sim_path + file + "/final_results.pkl")
    total_loss = tf.stack(final_res["loss"], -1)
    component_loss = tf.stack(final_res["loss_component"], -1)
    marginal_means = tf.stack(final_res["hyperparameter"]["means"], 0)
    marginal_sds = tf.stack(final_res["hyperparameter"]["stds"], 0)

    exp_priors = pd.read_pickle(
        expert_path + "/model_simulations.pkl")["prior_samples"]
    mod_priors = pd.read_pickle(sim_path + file + "/model_simulations.pkl")[
        "prior_samples"
    ]
    exp_means = tf.reduce_mean(exp_priors, (0, 1))
    exp_sds = tf.reduce_mean(tf.math.reduce_std(exp_priors, 1), 0)

    cmap = mpl.colormaps["gray"]
    col_losses = cmap(np.linspace(0.0, 0.8, component_loss.shape[0]))
    col_betas = ["#353238", "#92140c", "#be5a38", "#be7c4d"]
    col_exp = "#7d1717"
    col_mod = "#e2a854"

    fig = plt.figure(layout="constrained", figsize=(4, 5))
    subfigs = fig.subfigures(2, 1, hspace=0.03, height_ratios=[1, 1.0])
    axs0 = subfigs[0].subplots(2, 2, sharex=True)
    axs1 = subfigs[1].subplots(4, 4)

    axs0[0, 0].plot(range(total_loss.shape[0]), total_loss, color="black")
    [
        axs0[0, 1].plot(
            range(component_loss.shape[1]), component_loss[i, :],
            color=col_losses[i, :]
        )
        for i in range(component_loss.shape[0])
    ]
    # [axs0[0,i].set_yscale('log', base=2) for i in range(2)]
    [
        axs0[0, i].set_title(t, fontsize="small")
        for i, t in enumerate(["total loss", "loss components"])
    ]
    [
        axs0[1, 0].axhline(exp_means[i], color="black", linestyle="dashed",
                           lw=1)
        for i in range(4)
    ]
    [
        axs0[1, 1].axhline(exp_sds[i], color="black", linestyle="dashed", lw=1)
        for i in range(4)
    ]
    [
        axs0[1, 0].plot(
            range(marginal_means.shape[0]),
            marginal_means[:, i],
            label=rf"m$(\theta_{i})$",
            color=col_betas[i],
        )
        for i in range(4)
    ]
    [
        axs0[1, 1].plot(
            range(marginal_sds.shape[0]),
            marginal_sds[:, i],
            label=rf"sd$(\theta_{i})$",
            color=col_betas[i],
        )
        for i in range(4)
    ]
    axs0[1, 1].legend(
        handlelength=0.2,
        ncol=2,
        fontsize="x-small",
        frameon=False,
        loc=(0.5, 0.5),
        columnspacing=0.1,
        handletextpad=0.2,
    )
    axs0[1, 0].legend(
        handlelength=0.2,
        ncol=4,
        fontsize="x-small",
        frameon=False,
        loc=(0.0, 0.8),
        columnspacing=0.1,
        handletextpad=0.2,
    )
    [
        axs0[j, i].spines[["right", "top"]].set_visible(False)
        for j, i in product([0, 1], [0, 1])
    ]
    [axs0[j, i].yaxis.set_tick_params(labelsize=7) for
     j, i in product([0, 1], [0, 1])]
    [axs0[1, i].xaxis.set_tick_params(labelsize=7) for i in range(2)]
    [
        axs0[1, i].set_title(t, fontsize="small")
        for i, t in enumerate(["marginal means", "marginal std. dev."])
    ]
    [axs0[1, i].set_xlabel("epochs", fontsize="x-small") for i in range(2)]
    axs0[1, 0].set_ylim(-3, 15)
    axs0[1, 1].set_ylim(-1, 15)

    sns.kdeplot(exp_priors[0, :, 3], color=col_exp, ax=axs1[3, 3], lw=2,
                label="true")
    sns.kdeplot(
        mod_priors[0, :, 3], lw=1, color=col_mod, ax=axs1[3, 3],
        label="learned"
    )
    for i in range(4):
        sns.kdeplot(exp_priors[0, :, i], color=col_exp, ax=axs1[i, i], lw=2)
        sns.kdeplot(mod_priors[0, :, i], lw=2, color=col_mod, ax=axs1[i, i])
        axs1[i, i].spines[["right", "top"]].set_visible(False)
        axs1[i, i].get_xaxis().set_visible(False)
        axs1[i, i].get_yaxis().set_visible(False)
    for i, j in zip([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]):
        sns.scatterplot(
            x=exp_priors[0, :, i],
            y=exp_priors[0, :, j],
            color=col_exp,
            ax=axs1[i, j],
            marker="X",
            s=7,
            lw=0,
            alpha=0.1,
        )
        sns.kdeplot(
            x=mod_priors[0, :, i],
            y=mod_priors[0, :, j],
            color=col_mod,
            ax=axs1[i, j],
            alpha=0.7,
        )
        axs1[i, j].get_xaxis().set_visible(False)
        axs1[i, j].get_yaxis().set_visible(False)
        axs1[j, i].get_yaxis().set_visible(False)
        axs1[j, i].get_xaxis().set_visible(False)
        axs1[j, i].spines[["right", "top", "left", "bottom"]].set_visible(
            False)
        axs1[j, i].text(
            0.5,
            0.3,
            f"corr:\n{cor([j,i], mod_priors[0,:,:]):.2f}",
            color="black",
            fontsize="small",
            ha="center",
        )
    [
        axs1[i, j].spines[["right", "top"]].set_visible(False)
        for i, j in product(range(4), range(4))
    ]
    [axs1[0, i].set_title(rf"$\beta_{i}$", fontsize="small") for i in range(3)]
    axs1[0, 3].set_title(r"$\sigma$", fontsize="small")
    axs1[3, 3].legend(
        handlelength=0.5, ncol=1, fontsize="x-small", frameon=False,
        loc=(0.4, 0.5)
    )

    subfigs[1].suptitle(
        r"$\mathbf{(b)}$" + f" Joint prior distribution (seed: {file.split('_')[-1]})", # noqa
        ha="left",
        x=0.01,
        fontsize="medium",
    )
    subfigs[0].suptitle(
        r"$\mathbf{(a)}$" + f" Convergence diagnostics (seed: {file.split('_')[-1]})", # noqa
        ha="left",
        x=0.01,
        fontsize="medium",
    )
    if save_fig:
        plt.savefig(
            f"elicit/simulations/LiDO_cluster/sim_results/deep_prior/graphics/convergence_{model}.png", # noqa
            dpi=300,
        )
    else:
        plt.show()


def normal_sensitivity(
    prior_expert,
    expert_path,
    prior_res_agg,
    elicits_gr1_agg,
    elicits_gr2_agg,
    elicits_gr3_agg,
    elicits_r2_agg,
    cor_res_agg,
    model,
    save_fig=True,
):
    col_mod = "#e2a854"

    exp_gr1 = pd.read_pickle(expert_path + "/elicited_statistics.pkl")[
        "quantiles_group1"
    ]
    exp_gr2 = pd.read_pickle(expert_path + "/elicited_statistics.pkl")[
        "quantiles_group2"
    ]
    exp_gr3 = pd.read_pickle(expert_path + "/elicited_statistics.pkl")[
        "quantiles_group3"
    ]
    mod_gr1 = elicits_gr1_agg
    mod_gr2 = elicits_gr2_agg
    mod_gr3 = elicits_gr3_agg
    exp_r2 = tf.exp(
        pd.read_pickle(expert_path + "/elicited_statistics.pkl")[
            "quantiles_logR2"]
    )
    mod_r2 = elicits_r2_agg
    if model.startswith("correl"):
        cor_expert = pd.read_pickle(expert_path + "/elicited_statistics.pkl")[
            "identity_correl"
        ][0, :]
    else:
        cor_expert = pd.read_pickle(expert_path + "/elicited_statistics.pkl")[
            "correlation"
        ][0, :]

    fig = plt.figure(layout="constrained", figsize=(3.5, 3.5))
    subfigs = fig.subfigures(2, 1, hspace=0.02, height_ratios=[1.5, 1])
    axs0 = subfigs[0].subplots(2, 2)
    axs1 = subfigs[1].subplots(1, 1)
    for p in range(2):
        axs0[p, 0].spines[["right", "top"]].set_visible(False)
        axs0[p, 0].yaxis.set_tick_params(labelsize=7)
        axs0[p, 0].xaxis.set_tick_params(labelsize=7)
        axs0[0, 0].set_ylabel("density", fontsize="small")
        axs0[0, 0].set_title(r"$y_i \mid gr_1$", fontsize="small")
        axs0[-1, 0].set_xlabel(r"$\theta$", fontsize="small")
        axs0[p, 1].set_ylabel("", fontsize="small")
    sns.scatterplot(
        x=tf.expand_dims(cor_expert[0], -1),
        y=tf.zeros(1),
        color="#7d1717",
        ax=axs1,
        zorder=2,
        marker="o",
        label="true",
    )
    sns.scatterplot(
        x=cor_res_agg[0, :],
        y=tf.zeros(cor_res_agg.shape[-1]),
        color="#e2a854",
        ax=axs1,
        zorder=1,
        marker="o",
        alpha=0.6,
        label="learned",
    )
    for i in range(1, len(cor_expert)):
        sns.scatterplot(
            x=tf.expand_dims(cor_expert[i], -1),
            y=tf.zeros(1) + i,
            color="#7d1717",
            ax=axs1,
            zorder=2,
            marker="o",
        )
        sns.scatterplot(
            x=cor_res_agg[i, :],
            y=tf.zeros(cor_res_agg.shape[-1]) + i,
            color="#e2a854",
            ax=axs1,
            zorder=1,
            marker="o",
            alpha=0.6,
        )
    axs1.spines[["right", "top"]].set_visible(False)
    axs1.xaxis.set_tick_params(labelsize=7)
    axs1.yaxis.set_tick_params(labelsize=7)
    axs1.set_yticks(
        tf.range(len(cor_expert)),
        [
            r"$r(\beta_0,\beta_1)$",
            r"$r(\beta_0,\beta_2)$",
            r"$r(\beta_0,\sigma)$",
            r"$r(\beta_1,\beta_2)$",
            r"$r(\beta_1,\sigma)$",
            r"$r(\beta_2,\sigma)$",
        ],
    )
    axs1.set_title("Correlation", fontsize="small")
    axs1.set_ylim(-0.5, 6.0)
    axs1.legend(
        handlelength=0.5, ncol=2, fontsize="x-small", frameon=False,
        loc=(0.01, 0.9)
    )
    axs1.set_xlim(-1, 1)

    for b in range(mod_gr1.shape[-1]):
        [
            sns.scatterplot(
                x=exp_gr[0, :],
                y=mod_gr[:, b],
                ax=axs0[i, k],
                color=col_mod,
                lw=0,
                zorder=1,
                alpha=0.3,
            )
            for i, k, exp_gr, mod_gr in zip(
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [exp_gr1, exp_gr2, exp_gr3, exp_r2],
                [mod_gr1, mod_gr2, mod_gr3, mod_r2],
            )
        ]
    [
        axs0[i, k].axline(
            (0, 0), slope=1, color="black", lw=1, linestyle="dashed", zorder=0
        )
        for i, k, exp_gr, mod_gr in zip(
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [exp_gr1, exp_gr2, exp_gr3],
            [mod_gr1, mod_gr2, mod_gr3],
        )
    ]
    axs0[-1, 1].axline(
        (0, 0), (1, 1), color="black", lw=1, linestyle="dashed", zorder=0
    )
    [axs0[i, 1].xaxis.set_tick_params(labelsize=7) for i in range(2)]
    [axs0[i, 1].yaxis.set_tick_params(labelsize=7) for i in range(2)]
    axs0[-1, 1].set_ylim(0, 1.1)
    [axs0[i, 0].set_ylabel(" \n learned", fontsize="x-small") for
     i in range(2)]
    [axs0[i, 1].spines[["right", "top"]].set_visible(False) for i in range(2)]
    [axs0[1, i].set_xlabel("true", fontsize="x-small") for i in range(2)]
    [
        axs0[i, j].set_title(t, fontsize="small")
        for i, j, t in zip(
            [0, 1, 1], [1, 0, 1], [r"$ y_i \mid gr_2$", r"$ y_i \mid gr_3$",
                                   r"$ R^2$"]
        )
    ]

    if save_fig:
        plt.savefig(
            f"elicit/simulations/LiDO_cluster/sim_results/deep_prior/graphics/sensitivity_{model}.png", # noqa
            dpi=300,
        )
    else:
        plt.show()
