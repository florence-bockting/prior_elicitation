# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import elicit as el

from elicit.user.generative_models import NormalModelSimple
from elicit.user.design_matrices import design_matrix

tfd = tfp.distributions

def sim_func(seed, init_method, init_loss, init_iters):
    eliobj = el.Elicit(
        model=el.model(
            obj=NormalModelSimple,
            design_matrix=design_matrix(N=50, quantiles=[25,75]),
            sigma=1.
            ),
        parameters=[
            el.parameter(
                name="beta0",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper("mu0"),
                    scale=el.hyper("sigma0", lower=0)
                    )
            ),
            el.parameter(
                name="beta1",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper("mu1"),
                    scale=el.hyper("sigma1", lower=0)
                    )
            )
        ],
        target_quantities=[
            el.target(
                name="X0",
                elicitation_method=el.eli_method.quantiles((5, 25, 50, 75, 95)),
                loss=el.MMD_energy,
                loss_weight=1.0
            ),
            el.target(
                name="X1",
                elicitation_method=el.eli_method.quantiles((5, 25, 50, 75, 95)),
                loss=el.MMD_energy,
                loss_weight=1.0
            )
        ],
        expert=el.expert.simulate(
            ground_truth = {
                "b0": tfd.Normal(0.5, 0.7),
                "b1": tfd.Normal(-0.5, 1.3),
            },
            num_samples = 10_000
        ),
        optimization_settings=el.optimizer(
            optimizer=tf.keras.optimizers.Adam,
            learning_rate=0.05,
            clipnorm=1.0
            ),
        training_settings=el.train(
            method="parametric_prior",
            name="simple"+"_"+init_method+"_"+str(init_loss)+"_"+str(init_iters),
            seed=seed,
            epochs=400,
            progress_info=0
        ),
        initialization_settings=el.initializer(
            method=init_method,
            loss_quantile=init_loss,
            iterations=init_iters,
            specs=el.init_specs(
                radius=2,
                mean=0
                )
            )
    )

    return eliobj.train(save_file="results")


res_ep, res = sim_func(1, "sobol", 30, 32)

#%% Check uniformness
model1 = lambda q_min: f"simple_sobol_{q_min}_32_1"
model2 = lambda q_min:f"simple_lhs_{q_min}_32_1"
model3 = lambda q_min:f"simple_random_{q_min}_32_1"

import seaborn as sns
_, axs = plt.subplots(2,4, constrained_layout=True, figsize=(8,3),
                      sharey=True)
for j, q in enumerate([0, 10, 30]):
    for mod in [model1(q), model2(q), model3(q)]:
        init=pd.read_pickle(f"elicit/results/parametric_prior/{mod}/initialization_matrix.pkl")

        for i, n in enumerate(["mu0", "sigma0", "mu1", "sigma1"]):
            sns.ecdfplot(init[n], label=mod.split("_")[1], ax=axs[j,i])
            axs[j,i].set_title(n, size="medium")
        axs[0,0].legend(handlelength=0.2, frameon=False, fontsize="small")
        plt.suptitle("ecdf of initializations; min loss, seed=1")


#%% Loss
_, axs=plt.subplots(3,1, constrained_layout=True, figsize=(4,4),
                    sharey=True, sharex=True)
for j, q in enumerate([0, 10, 30]):
    if q==10:
        ls="dashed"
    elif q==30:
        ls="dotted"
    else:
        ls="solid"
    for i, mod in enumerate([model1(q), model2(q), model3(q)]):
        mod_n = mod.split("_")[1]
        if mod_n == "sobol":
            c="red"
        elif mod_n == "lhs":
            c="green"
        else:
            c="blue"
        loss=pd.read_pickle(f"elicit/results/parametric_prior/{mod}/final_results.pkl")["loss"]
        axs[i].plot(loss, label=mod_n+":"+str(q), linestyle=ls, color=c)
        axs[i].legend(frameon=False)
        axs[i].set_xlim(0,100)

#%% Convergence

_, axs=plt.subplots(3,3, constrained_layout=True, figsize=(7,4),
                    sharey=True, sharex=True)
for j, q in enumerate([0, 10, 30]):
    for i, mod in enumerate([model1(q), model2(q), model3(q)]):
        res_ep = pd.read_pickle(f"elicit/results/parametric_prior/{mod}/final_results.pkl")["hyperparameter"]
        axs[j,i].plot(res_ep["mu0"], label=r"$\mu_0$")
        axs[j,i].plot(res_ep["mu1"], label=r"$\mu_1$")
        axs[j,i].plot(res_ep["sigma0"], label=r"$\sigma_0$")
        axs[j,i].plot(res_ep["sigma1"], label=r"$\sigma_1$")
        axs[0,i].set_title(mod.split("_")[1])
axs[0,0].legend(handlelength=0.5, fontsize="small", ncol=4,
              frameon=False, columnspacing=0.8)

