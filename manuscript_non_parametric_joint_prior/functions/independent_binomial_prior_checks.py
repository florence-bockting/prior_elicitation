import tensorflow_probability as tfp
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

from elicit.core.run import prior_elicitation
from elicit.user.design_matrices import load_design_matrix_binomial
from elicit.user.generative_models import BinomialModel
from elicit.user.custom_functions import quantiles_per_ypred

tfd = tfp.distributions


# prepare simulations
def run_prior_checks(seed, path, mu0, sigma0, mu1, sigma1):

    prior_elicitation(
        model_parameters=dict(
            b0=dict(param_scaling=1.0),
            b1=dict(param_scaling=1.0),
            independence=dict(corr_scaling=0.1),
        ),
        normalizing_flow=True,
        expert_data=dict(
            from_ground_truth=True,
            simulator_specs={
                "b0": tfd.Normal(mu0, sigma0),
                "b1": tfd.Normal(mu1, sigma1),
            },
            samples_from_prior=10_000,
        ),
        generative_model=dict(
            model=BinomialModel,
            additional_model_args={
                "total_count": 30,
                "design_matrix": load_design_matrix_binomial(50),
            },
            discrete_likelihood=True,
            softmax_gumble_specs={"temperature": 1.0, "upper_threshold": 30},
        ),
        target_quantities=dict(
            ypred=dict(
                elicitation_method=None,
                custom_elicitation_method=dict(
                    function=quantiles_per_ypred,
                    additional_args={"quantiles_specs": (5, 25, 50, 75, 95)},
                ),
                loss_components="by-group",
            )
        ),
        optimization_settings=dict(
            optimizer_specs={
                "learning_rate": 0.0001,
                "clipnorm": 1.0,
            }
        ),
        training_settings=dict(
            method="deep_prior",
            sim_id=f"binomial_{mu0:.2f}_{sigma0:.2f}_{mu1:.2f}_{sigma1:.2f}",
            seed=seed,
            output_path=path,
            epochs=1,
        ),
    )


def run_sensitivity_binom(seed, path_sim_res, mu0_seq, mu1_seq, sigma0_seq,
                          sigma1_seq):
    # run simulations
    for mu0 in mu0_seq:
        run_prior_checks(seed, path_sim_res + "/vary_mu0", mu0, 0.1, -0.1, 0.3)

    for sigma0 in sigma0_seq:
        run_prior_checks(
            seed,
            path_sim_res + "/vary_sigma0",
            0.1,
            sigma0,
            -0.1,
            0.3,
        )

    for mu1 in mu1_seq:
        run_prior_checks(seed, path_sim_res + "/vary_mu1", 0.1, 0.1, mu1, 0.3)

    for sigma1 in sigma1_seq:
        run_prior_checks(
            seed,
            path_sim_res + "/vary_sigma1",
            0.1,
            0.1,
            -0.1,
            sigma1,
        )


def prep_sensitivity_res(path_sensitivity_res):
    # create result table
    res_dict = {
        "id": [],
        "mu0": [],
        "sigma0": [],
        "mu1": [],
        "sigma1": [],
        "X0": [],
        "X1": [],
    }

    for vary in ["vary_mu0", "vary_sigma0", "vary_mu1", "vary_sigma1"]:
        path = path_sensitivity_res + "/" + vary + "/deep_prior"
        all_files = os.listdir(path)
        for i in range(len(all_files)):
            labels = all_files[i].split("_")
            res_dict["id"].append(vary)
            res_dict["mu0"].append(labels[1])
            res_dict["sigma0"].append(labels[2])
            res_dict["mu1"].append(labels[3])
            res_dict["sigma1"].append(labels[4])
            res_dict["X0"].append(
                pd.read_pickle(
                    path + f"/{all_files[i]}" + "/expert/elicited_statistics.pkl" # noqa
                )["custom_ypred"][0, :, 0].numpy()
            )
            res_dict["X1"].append(
                pd.read_pickle(
                    path + f"/{all_files[i]}" + "/expert/elicited_statistics.pkl" # noqa
                )["custom_ypred"][0, :, 1].numpy()
            )
        df = pd.DataFrame(res_dict)
    return df


def plot_sensitivity_binom(df):
    # create sensitivity plot
    def conv_seq(var, no="1.00"):
        return np.array(df[df[var] != no][var], dtype=np.float32)

    range_list = [
        conv_seq("mu0", "0.10"),
        np.array(["0.01", "0.10", "0.30", "0.60", "1.00"], dtype=np.float32),
        conv_seq("mu1", "-0.10"),
        np.array(["0.01", "0.10", "0.30", "0.60", "1.00"], dtype=np.float32),
    ]
    cols_quantiles = ["#21284f", "#00537b", "#007d87", "#00ac79", "#83cf4a"]
    true_vals = {"mu0": 0.1, "sigma0": 0.1, "mu1": -0.1, "sigma1": 0.3}

    def re_dig(x):
        return [x[i].astype(str).replace("0.", ".") for i in range(len(x))]

    fig, axs = plt.subplots(4, 2, constrained_layout=True, figsize=(4, 4))
    for m, (k, xseq) in enumerate(
        zip(["vary_mu0", "vary_sigma0", "vary_mu1", "vary_sigma1"], range_list)
    ):
        for j, elicit in enumerate(["X0", "X1"]):
            for i, col in list(enumerate(cols_quantiles)):
                axs[m, j].plot(
                    xseq,
                    np.stack(df[df["id"] == k][elicit], 1)[i],
                    "-o",
                    color=col,
                    ms=5,
                )
                axs[m, j].patch.set_alpha(0.0)
    for j in range(2):
        [
            axs[i, j].set_xlabel(lab, fontsize="small", labelpad=2)
            for i, lab in enumerate(
                [r"$\mu_0$", r"$\sigma_0$", r"$\mu_1$", r"$\sigma_1$"]
            )
        ]
        [
            axs[i, j].set_xticks(
                range_list[i], re_dig(range_list[i]), fontsize="x-small"
            )
            for i in range(4)
        ]
        [axs[i, j].tick_params(axis="y", labelsize="x-small") for
         i in range(4)]
    [
        axs[0, j].set_title(t, pad=10, fontsize="medium")
        for j, t in enumerate(
            [r"quantiles $y_i \mid x_0$", r"quantiles $y_i \mid x_1$"]
        )
    ]
    [
        axs[i, j].spines[["right", "top"]].set_visible(False)
        for i, j in itertools.product(range(4), range(2))
    ]
    [axs[i, 0].set_ylabel(" ", rotation=0, labelpad=10) for i in range(4)]
    for k, val in enumerate(true_vals):
        [axs[k, j].axvline(true_vals[val], color="darkred", lw=2) for
         j in range(2)]
    for i, lab, col in zip(
        [0, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.44],
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
        fig.text(i, 1.02, lab, color=col, fontsize="small")
    # fig.patches.extend(
    #     [
    #         plt.Rectangle(
    #             (10, 279),
    #             600,
    #             3,
    #             fill=True,
    #             color="grey",
    #             alpha=0.2,
    #             zorder=-1,
    #             transform=None,
    #             figure=fig,
    #         )
    #     ]
    # )
    for x, y, lab in zip([0.01] * 2, [0.73, 0.26],
                         [r"$\beta_0$", r"$\beta_1$"]):
        fig.text(
            x, y, lab, fontsize="medium", bbox=dict(facecolor="none",
                                                    edgecolor="grey")
        )
