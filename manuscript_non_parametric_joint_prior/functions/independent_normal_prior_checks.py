# NOTE: If you want to run this file, you need to disable saving of the global
# dictionary
# you can do this by commenting out the respective line
# ('save_as_pkl(global_dict, path)' in the file run.py)
import os
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

from elicit.core.run import prior_elicitation
from elicit.user.design_matrices import load_design_matrix_normal
from elicit.user.generative_models import NormalModel

tfd = tfp.distributions


# prepare simulations
def run_prior_checks(seed, path, vary, mu0, sigma0, mu1, sigma1, mu2, sigma2,
                     a, b):
    truth_independent = {
        "b0": tfd.Normal(mu0, sigma0),
        "b1": tfd.Normal(mu1, sigma1),
        "b2": tfd.Normal(mu2, sigma2),
        "sigma": tfd.Gamma(a, b),
    }
    prior_elicitation(
        model_parameters=dict(
            b0=dict(param_scaling=1.0),
            b1=dict(param_scaling=1.0),
            b2=dict(param_scaling=1.0),
            sigma=dict(param_scaling=1.0),
            independence=dict(corr_scaling=0.1),
        ),
        normalizing_flow=True,
        expert_data=dict(
            from_ground_truth=True,
            simulator_specs=truth_independent,
            samples_from_prior=10_000,
        ),
        generative_model=dict(
            model=NormalModel,
            additional_model_args={
                "design_matrix": load_design_matrix_normal(30)
                },
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
        ),
        optimization_settings=dict(
            optimizer_specs={
                "learning_rate": 0.00025,
                "clipnorm": 1.0,
            }
        ),
        training_settings=dict(
            method="deep_prior",
            sim_id="normalindependent_" + vary,
            seed=seed,
            output_path=path,
            epochs=1,
        ),
    )


def run_sensitivity(
    seed,
    path_sensitivity_res,
    mu0_seq,
    mu1_seq,
    mu2_seq,
    sigma0_seq,
    sigma1_seq,
    sigma2_seq,
    a_seq,
    b_seq,
    cor_seq,
    skewness1_seq,
    skewness2_seq,
):

    # run simulations
    for mu0 in mu0_seq:
        run_prior_checks(
            seed,
            path=path_sensitivity_res + "/vary_mu0",
            vary=f"mu0_{mu0:.1f}",
            mu0=mu0,
            sigma0=2.5,
            mu1=7.0,
            sigma1=1.3,
            mu2=2.5,
            sigma2=0.8,
            a=5.0,
            b=2.0,
        )

    for sigma0 in sigma0_seq:
        run_prior_checks(
            seed,
            path=path_sensitivity_res + "/vary_sigma0",
            vary=f"sigma0_{sigma0:.1f}",
            mu0=10.0,
            sigma0=sigma0,
            mu1=7.0,
            sigma1=1.3,
            mu2=2.5,
            sigma2=0.8,
            a=5.0,
            b=2.0,
        )

    for mu1 in mu1_seq:
        run_prior_checks(
            seed,
            path=path_sensitivity_res + "/vary_mu1",
            vary=f"mu1_{mu1:.1f}",
            mu0=10.0,
            sigma0=2.5,
            mu1=mu1,
            sigma1=1.3,
            mu2=2.5,
            sigma2=0.8,
            a=5.0,
            b=2.0,
        )

    for sigma1 in sigma1_seq:
        run_prior_checks(
            seed,
            path=path_sensitivity_res + "/vary_sigma1",
            vary=f"sigma1_{sigma1:.1f}",
            mu0=10.0,
            sigma0=2.5,
            mu1=7.0,
            sigma1=sigma1,
            mu2=2.5,
            sigma2=0.8,
            a=5.0,
            b=2.0,
        )

    for mu2 in mu2_seq:
        run_prior_checks(
            seed,
            path=path_sensitivity_res + "/vary_mu2",
            vary=f"mu2_{mu2:.1f}",
            mu0=10.0,
            sigma0=2.5,
            mu1=7.0,
            sigma1=1.3,
            mu2=mu2,
            sigma2=0.8,
            a=5.0,
            b=2.0,
        )

    for sigma2 in sigma2_seq:
        run_prior_checks(
            1,
            path=path_sensitivity_res + "/vary_sigma2",
            vary=f"sigma2_{sigma2:.1f}",
            mu0=10.0,
            sigma0=2.5,
            mu1=7.0,
            sigma1=1.3,
            mu2=2.5,
            sigma2=sigma2,
            a=5.0,
            b=2.0,
        )

    for a in a_seq:
        run_prior_checks(
            seed,
            path=path_sensitivity_res + "/vary_a",
            vary=f"a_{a:.1f}",
            mu0=10.0,
            sigma0=2.5,
            mu1=7.0,
            sigma1=1.3,
            mu2=2.5,
            sigma2=0.8,
            a=a,
            b=2.0,
        )

    for b in b_seq:
        run_prior_checks(
            seed,
            path=path_sensitivity_res + "/vary_b",
            vary=f"b_{b:.1f}",
            mu0=10.0,
            sigma0=2.5,
            mu1=7.0,
            sigma1=1.3,
            mu2=2.5,
            sigma2=0.8,
            a=5.0,
            b=b,
        )


def prep_sensitivity_res(path_sensitivity_res, input_dict):
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
    ]:
        path = "elicit/" + path_sensitivity_res + "/" + vary + "/deep_prior"
        all_files = os.listdir(path)

        for i in range(len(all_files)):
            labels = all_files[i].split("_")
            # update varying value
            input_dict[labels[1]] = labels[2]
            res_dict["id"].append(vary)
            res_dict["mu0"].append(input_dict["mu0"])
            res_dict["sigma0"].append(input_dict["sigma0"])
            res_dict["mu1"].append(input_dict["mu1"])
            res_dict["sigma1"].append(input_dict["sigma1"])
            res_dict["mu2"].append(input_dict["mu2"])
            res_dict["sigma2"].append(input_dict["sigma2"])
            res_dict["a"].append(input_dict["a"])
            res_dict["b"].append(input_dict["b"])

            res_dict["group1"].append(
                pd.read_pickle(
                    path + f"/{all_files[i]}" + "/expert/elicited_statistics.pkl" # noqa
                )["quantiles_group1"][0, :].numpy()
            )
            res_dict["group2"].append(
                pd.read_pickle(
                    path + f"/{all_files[i]}" + "/expert/elicited_statistics.pkl" # noqa
                )["quantiles_group2"][0, :].numpy()
            )
            res_dict["group3"].append(
                pd.read_pickle(
                    path + f"/{all_files[i]}" + "/expert/elicited_statistics.pkl" # noqa
                )["quantiles_group3"][0, :].numpy()
            )
            res_dict["R2"].append(
                np.exp(
                    pd.read_pickle(
                        path + f"/{all_files[i]}" + "/expert/elicited_statistics.pkl" # noqa
                    )["quantiles_logR2"]
                )[0, :]
            )
    df = pd.DataFrame(res_dict)
    return df


def plot_sensitivity(
    df,
    mu0_seq,
    mu1_seq,
    mu2_seq,
    sigma0_seq,
    sigma1_seq,
    sigma2_seq,
    a_seq,
    b_seq,
    cor_seq,
    skewness1_seq,
    skewness2_seq,
):
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
    ]
    range_idx = [
        [0, 4, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [0, 4, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [0, 4, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [0, 4, 1, 2, 3],
        [0, 4, 1, 2, 3],
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
    }

    # plot sensitivity
    fig, axs = plt.subplots(8, 4, constrained_layout=True, figsize=(7, 9))
    for m, (k, xseq, idx) in enumerate(
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
            ],
            range_list2,
            range_idx,
        )
    ):
        for j, elicit in enumerate(["group1", "group2", "group3", "R2"]):
            for i, col in list(enumerate(cols_quantiles)):
                axs[m, j].plot(
                    xseq,
                    tf.gather(np.stack(df[df["id"] == k][elicit], 1)[i], idx),
                    "-o",
                    color=col,
                    ms=5,
                )
                axs[m, j].patch.set_alpha(0.0)
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
                ]
            )
        ]
        [
            axs[i, j].set_xticks(
                range_list2[i], np.array(range_list2[i]).astype(int),
                fontsize="x-small"
            )
            for i in range(8)
        ]
        [axs[i, j].tick_params(axis="y", labelsize="x-small") for
         i in range(8)]
    [
        axs[0, j].set_title(t, pad=10, fontsize="medium")
        for j, t in enumerate(
            [
                r"quantiles $y_i \mid gr_1$",
                r"quantiles $y_i \mid gr_2$",
                r"quantiles $y_i \mid gr_3$",
                r"quantiles $R^2$",
            ]
        )
    ]
    [
        axs[i, j].spines[["right", "top"]].set_visible(False)
        for i, j in itertools.product(range(8), range(4))
    ]
    [axs[i, -1].set_ylim(0, 1) for i in range(8)]
    [axs[i, 0].set_ylabel(" ", rotation=0, labelpad=10) for i in range(8)]
    for k, val in enumerate(true_vals):
        [axs[k, j].axvline(true_vals[val], color="darkred", lw=2) for
         j in range(4)]
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
    fig.suptitle("independent-normal model", x=0.5, y=1.07)
    for y in [952, 639, 323]:
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
        [0.005] * 4,
        [0.85, 0.61, 0.37, 0.13],
        [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$", r"$\sigma$"],
    ):
        fig.text(
            x, y, lab, fontsize="large", bbox=dict(facecolor="none",
                                                   edgecolor="grey")
        )
