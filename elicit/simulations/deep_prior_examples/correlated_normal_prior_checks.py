# NOTE: If you want to run this file, you need to disable saving of the global
# dictionary
# you can do this by commenting out the respective line
# ('save_as_pkl(global_dict, path)' in the file run.py)

import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

from elicit.core.run import prior_elicitation
from elicit.user.design_matrices import load_design_matrix_normal
from elicit.user.custom_functions import custom_correlation
from elicit.user.generative_models import NormalModel

tfd = tfp.distributions


# prepare simulations
def run_prior_checks(
    seed, path, mu0, sigma0, mu1, sigma1, mu2, sigma2, a, b, cor01, cor02, cor12
):
    S = [sigma0, sigma1, sigma2]
    M = [[1.0, cor01, cor02], [cor01, 1.0, cor12], [cor02, cor12, 1.0]]
    covariance_matrix = (tf.linalg.diag(S) @ M) @ tf.linalg.diag(S)

    truth_correlated = {
        "theta": tfd.JointDistributionSequential(
            [
                tfd.MultivariateNormalTriL(
                    loc=[mu0, mu1, mu2],
                    scale_tril=tf.linalg.cholesky(covariance_matrix),
                ),
                tfd.Gamma([a], [b]),
            ]
        )
    }
    prior_elicitation(
        model_parameters=dict(
            b0=dict(param_scaling=1.0),
            b1=dict(param_scaling=1.0),
            b2=dict(param_scaling=1.0),
            sigma=dict(param_scaling=1.0),
            independence=None,
        ),
        normalizing_flow=True,
        expert_data=dict(
            from_ground_truth=True,
            simulator_specs=truth_correlated,
            samples_from_prior=10_000,
        ),
        generative_model=dict(
            model=NormalModel,
            additional_model_args={"design_matrix": load_design_matrix_normal(30)},
        ),
        target_quantities=dict(
            group1=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components="all",
            ),
            group2=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components="all",
            ),
            group3=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components="all",
            ),
            logR2=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components="all",
            ),
            correl=dict(
                elicitation_method="identity",
                loss_components="by-group",
                custom_target_function=dict(
                    function=custom_correlation, additional_args=None
                ),
            ),
        ),
        optimization_settings=dict(
            optimizer_specs={
                "learning_rate": 0.00025,
                "clipnorm": 1.0,
            }
        ),
        training_settings=dict(
            method="deep_prior",
            sim_id=f"normalcorrelated_{mu0:.2f}_{sigma0:.2f}_{mu1:.2f}_{sigma1:.2f}_{mu2:.2f}_{sigma2:.2f}_{a:.2f}_{b:.2f}_{cor01:.2f}_{cor02:.2f}_{cor12:.2f}",
            seed=seed,
            output_path=path,
            epochs=1,
        ),
    )


mu0_seq = [0, 5, 10, 15, 20]
sigma0_seq = [0.1, 1.5, 2.0, 3.0, 4.0]
mu1_seq = [0, 5, 10, 15, 20]
sigma1_seq = [0.1, 1.5, 2.0, 3.0, 4.0]
mu2_seq = [0, 5, 10, 15, 20]
sigma2_seq = [0.1, 1.5, 2.0, 3.0, 4.0]
a_seq = [1, 5, 10, 20, 40]
b_seq = [1, 5, 10, 20, 40]
cor_seq = [-0.8, -0.5, 0.0, 0.5, 0.8]


# run simulations
for mu0 in mu0_seq:
    run_prior_checks(
        1,
        path="results/normal_correlated_sensitivity/vary_mu0",
        mu0=mu0,
        sigma0=2.5,
        mu1=7.0,
        sigma1=1.3,
        mu2=2.5,
        sigma2=0.8,
        a=2.0,
        b=2.0,
        cor01=0.3,
        cor02=-0.3,
        cor12=-0.2,
    )

for sigma0 in sigma0_seq:
    run_prior_checks(
        1,
        path="results/normal_correlated_sensitivity/vary_sigma0",
        mu0=10.0,
        sigma0=sigma0,
        mu1=7.0,
        sigma1=1.3,
        mu2=2.5,
        sigma2=0.8,
        a=2,
        b=2,
        cor01=0.3,
        cor02=-0.3,
        cor12=-0.2,
    )

for mu1 in mu1_seq:
    run_prior_checks(
        1,
        path="results/normal_correlated_sensitivity/vary_mu1",
        mu0=10.0,
        sigma0=2.5,
        mu1=mu1,
        sigma1=1.3,
        mu2=2.5,
        sigma2=0.8,
        a=2.0,
        b=2.0,
        cor01=0.3,
        cor02=-0.3,
        cor12=-0.2,
    )

for sigma1 in sigma1_seq:
    run_prior_checks(
        1,
        path="results/normal_correlated_sensitivity/vary_sigma1",
        mu0=10.0,
        sigma0=2.5,
        mu1=7.0,
        sigma1=sigma1,
        mu2=2.5,
        sigma2=0.8,
        a=2.0,
        b=2.0,
        cor01=0.3,
        cor02=-0.3,
        cor12=-0.2,
    )

for mu2 in mu2_seq:
    run_prior_checks(
        1,
        path="results/normal_correlated_sensitivity/vary_mu2",
        mu0=10.0,
        sigma0=2.5,
        mu1=7.0,
        sigma1=1.3,
        mu2=mu2,
        sigma2=0.8,
        a=2.0,
        b=2.0,
        cor01=0.3,
        cor02=-0.3,
        cor12=-0.2,
    )

for sigma2 in sigma2_seq:
    run_prior_checks(
        1,
        path="results/normal_correlated_sensitivity/vary_sigma2",
        mu0=10.0,
        sigma0=2.5,
        mu1=7.0,
        sigma1=1.3,
        mu2=2.5,
        sigma2=sigma2,
        a=2.0,
        b=2.0,
        cor01=0.3,
        cor02=-0.3,
        cor12=-0.2,
    )

for a in a_seq:
    run_prior_checks(
        1,
        path="results/normal_correlated_sensitivity/vary_a",
        mu0=10.0,
        sigma0=2.5,
        mu1=7.0,
        sigma1=1.3,
        mu2=2.5,
        sigma2=0.8,
        a=a,
        b=2.0,
        cor01=0.3,
        cor02=-0.3,
        cor12=-0.2,
    )

for b in b_seq:
    run_prior_checks(
        1,
        path="results/normal_correlated_sensitivity/vary_b",
        mu0=10.0,
        sigma0=2.5,
        mu1=7.0,
        sigma1=1.3,
        mu2=2.5,
        sigma2=0.8,
        a=2.0,
        b=b,
        cor01=0.3,
        cor02=-0.3,
        cor12=-0.2,
    )

for cor01 in cor_seq:
    print(cor01)
    run_prior_checks(
        1,
        path="results/normal_correlated_sensitivity/vary_cor01",
        mu0=10.0,
        sigma0=2.5,
        mu1=7.0,
        sigma1=1.3,
        mu2=2.5,
        sigma2=0.8,
        a=2.0,
        b=2,
        cor01=cor01,
        cor02=-0.3,
        cor12=-0.2,
    )

for cor02 in cor_seq:
    run_prior_checks(
        1,
        path="results/normal_correlated_sensitivity/vary_cor02",
        mu0=10.0,
        sigma0=2.5,
        mu1=7.0,
        sigma1=1.3,
        mu2=2.5,
        sigma2=0.8,
        a=2.0,
        b=2,
        cor01=0.3,
        cor02=cor02,
        cor12=-0.2,
    )

for cor12 in cor_seq:
    run_prior_checks(
        1,
        path="results/normal_correlated_sensitivity/vary_cor12",
        mu0=10.0,
        sigma0=2.5,
        mu1=7.0,
        sigma1=1.3,
        mu2=2.5,
        sigma2=0.8,
        a=2.0,
        b=2,
        cor01=0.3,
        cor02=-0.3,
        cor12=cor12,
    )


# save results in dictionary
res_dict = {
    "id": [],
    "mu0": [],
    "sigma0": [],
    "mu1": [],
    "sigma1": [],
    "mu2": [],
    "sigma2": [],
    "a": [],
    "b": [],
    "cor01": [],
    "cor02": [],
    "cor12": [],
    "group1": [],
    "group2": [],
    "group3": [],
    "R2": [],
}

for vary in [
    "vary_mu0",
    "vary_sigma0",
    "vary_mu1",
    "vary_sigma1",
    "vary_mu2",
    "vary_sigma2",
    "vary_a",
    "vary_b",
    "vary_cor01",
    "vary_cor02",
    "vary_cor12",
]:
    path = "elicit/results/normal_correlated_sensitivity/" + vary + "/deep_prior"
    all_files = os.listdir(path)
    for i in range(len(all_files)):
        labels = all_files[i].split("_")
        res_dict["id"].append(vary)
        res_dict["mu0"].append(labels[1])
        res_dict["sigma0"].append(labels[2])
        res_dict["mu1"].append(labels[3])
        res_dict["sigma1"].append(labels[4])
        res_dict["mu2"].append(labels[5])
        res_dict["sigma2"].append(labels[6])
        res_dict["a"].append(labels[7])
        res_dict["b"].append(labels[8])
        res_dict["cor01"].append(labels[9])
        res_dict["cor02"].append(labels[10])
        res_dict["cor12"].append(labels[11])
        res_dict["group1"].append(
            pd.read_pickle(
                path + f"/{all_files[i]}" + "/expert/elicited_statistics.pkl"
            )["quantiles_group1"][0, :].numpy()
        )
        res_dict["group2"].append(
            pd.read_pickle(
                path + f"/{all_files[i]}" + "/expert/elicited_statistics.pkl"
            )["quantiles_group2"][0, :].numpy()
        )
        res_dict["group3"].append(
            pd.read_pickle(
                path + f"/{all_files[i]}" + "/expert/elicited_statistics.pkl"
            )["quantiles_group3"][0, :].numpy()
        )
        res_dict["R2"].append(
            tf.exp(
                pd.read_pickle(
                    path + f"/{all_files[i]}" + "/expert/elicited_statistics.pkl"
                )["quantiles_logR2"][0, :]
            ).numpy()
        )
    df = pd.DataFrame(res_dict)


# create helpers for plotting
range_list2 = [
    mu0_seq,
    sigma0_seq,
    mu1_seq,
    sigma1_seq,
    mu2_seq,
    sigma2_seq,
    a_seq,
    b_seq,
    cor_seq,
    cor_seq,
    cor_seq,
]
cols_quantiles = ["#21284f", "#00537b", "#007d87", "#00ac79", "#83cf4a"]
true_vals = {
    "mu0": 10,
    "sigma0": 2.5,
    "mu1": 7,
    "sigma1": 1.3,
    "mu2": 2.5,
    "sigma2": 0.8,
    "a": 5,
    "b": 2,
    "cor01": 0.3,
    "cor02": -0.3,
    "cor12": -0.2,
}
range_idx = [
    [0, 4, 1, 2, 3],
    [0, 1, 2, 3, 4],
    [0, 4, 1, 2, 3],
    [0, 1, 2, 3, 4],
    [0, 4, 1, 2, 3],
    [0, 1, 2, 3, 4],
    [0, 4, 1, 2, 3],
    [0, 4, 1, 2, 3],
    [1, 0, 2, 3, 4],
    [1, 0, 2, 3, 4],
    [1, 0, 2, 3, 4],
]


# plot sensitivity
fig, axs = plt.subplots(11, 4, constrained_layout=True, figsize=(7, 11))
for l, (k, xseq, idx) in enumerate(
    zip(
        [
            "vary_mu0",
            "vary_sigma0",
            "vary_mu1",
            "vary_sigma1",
            "vary_mu2",
            "vary_sigma2",
            "vary_a",
            "vary_b",
            "vary_cor01",
            "vary_cor02",
            "vary_cor12",
        ],
        range_list2,
        range_idx,
    )
):
    for j, elicit in enumerate(["group1", "group2", "group3", "R2"]):
        for i, col in list(enumerate(cols_quantiles)):
            axs[l, j].plot(
                xseq,
                tf.gather(np.stack(df[df["id"] == k][elicit], 1)[i], idx),
                "-o",
                color=col,
            )
for j in range(4):
    [
        axs[i, j].set_xlabel(lab, fontsize="small", labelpad=2)
        for i, lab in enumerate(
            [
                r"$\mu_0$",
                r"$\sigma_0$",
                r"$\mu_1$",
                r"$\sigma_1$",
                r"$\mu_2$",
                r"$\sigma_2$",
                r"$a$",
                r"$b$",
                r"$\rho_{01}$",
                r"$\rho_{02}$",
                r"$\rho_{12}$",
            ]
        )
    ]
    [
        axs[i, j].set_xticks(
            range_list2[i], np.array(range_list2[i]).astype(int), fontsize="x-small"
        )
        for i in range(8)
    ]
    [
        axs[i, j].set_xticks(range_list2[i], range_list2[i], fontsize="x-small")
        for i in range(8, 11)
    ]
    [axs[i, j].tick_params(axis="y", labelsize="x-small") for i in range(11)]
[axs[i, 0].set_ylabel(" ", rotation=0, labelpad=10) for i in range(11)]
[axs[0, j].set_title(t) for j, t in enumerate([
           r"quantiles $y_i \mid gr_1$",
           r"quantiles $y_i \mid gr_2$",
           r"quantiles $y_i \mid gr_3$",
           r"quantiles $R^2$",
    ])]
[
    axs[i, j].spines[["right", "top"]].set_visible(False)
    for i, j in itertools.product(range(11), range(4))
]
[axs[i, -1].set_ylim(0, 1) for i in range(11)]
for i, lab, col in zip(
    [0, 0.08, 0.12, 0.16, 0.2, 0.24, 0.3, 0.32],
    [
        "legend: ",
        r"$q_{05}$",
        r"$q_{25}$",
        r"$q_{50}$",
        r"$q_{75}$",
        r"$q_{95}$",
        "|",
        "ground truth",
    ],
    [
        "black",
        "#21284f",
        "#00537b",
        "#007d87",
        "#00ac79",
        "#83cf4a",
        "darkred",
        "black",
    ],
):
    fig.text(i, 1.02, lab, color=col)
fig.suptitle("correlated-normal model", x=0.5, y=1.06)
for k, val in enumerate(true_vals):
    [axs[k, j].axvline(true_vals[val], color="darkred", lw=2) for j in range(4)]
for y in [1280, 998, 713, 432]:
    fig.patches.extend(
        [
            plt.Rectangle(
                (10, y),
                1010,
                3,
                fill=True,
                color="grey",
                alpha=0.2,
                zorder=-1,
                transform=None,
                figure=fig,
            )
        ]
    )
for x, y, lab in zip(
    [0.005] * 5,
    [0.905, 0.73, 0.55, 0.365, 0.145],
    [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$", r"$\sigma$", r"$\rho$"],
):
    fig.text(x, y, lab, fontsize="large", bbox=dict(facecolor="none", edgecolor="grey"))

