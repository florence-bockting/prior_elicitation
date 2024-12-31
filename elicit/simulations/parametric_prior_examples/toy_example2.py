# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import elicit as el

from elicit.user.generative_models import ToyModel2
from elicit.user.design_matrices import load_design_matrix_toy2

tfd = tfp.distributions


ground_truth = {
    "beta0": tfd.Normal(loc=5, scale=1),
    "beta1": tfd.Normal(loc=2, scale=1),
    "sigma": tfd.HalfNormal(scale=10.0),
}

expert_dat = {
    "quantiles_y_X0": [-12.549973, -0.5716343, 3.294293, 7.1358547, 19.147377],
    "quantiles_y_X1": [-11.18329, 1.4518523, 5.061118, 8.83173, 20.423292],
    "quantiles_y_X2": [-9.279653, 3.0914488, 6.8263884, 10.551274, 23.285913]
}

eliobj = el.Elicit(
    model=el.model(
        obj=ToyModel2,
        design_matrix=load_design_matrix_toy2(N=200, quants=[25,50,75])
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
        ),
        el.parameter(
            name="sigma",
            family=tfd.HalfNormal,
            hyperparams=dict(
                scale=el.hyper("sigma2", lower=0)
                )
        ),
    ],
    target_quantities=[
        el.target(
            name="y_X0",
            elicitation_method=el.eli_method.quantiles((5, 25, 50, 75, 95)),
            loss=el.MMD_energy,
            loss_weight=1.0
        ),
        el.target(
            name="y_X1",
            elicitation_method=el.eli_method.quantiles((5, 25, 50, 75, 95)),
            loss=el.MMD_energy,
            loss_weight=1.0
        ),
        el.target(
            name="y_X2",
            elicitation_method=el.eli_method.quantiles((5, 25, 50, 75, 95)),
            loss=el.MMD_energy,
            loss_weight=1.0
        )
    ],
    expert=el.expert.data(dat = expert_dat),
    # expert=el.expert.simulate(
    #     ground_truth = ground_truth,
    #     num_samples = 10_000
    # ),
    optimization_settings=el.optimizer(
        optimizer=tf.keras.optimizers.Adam,
        learning_rate=0.05,
        clipnorm=1.0
        ),
    training_settings=el.train(
        method="parametric_prior",
        name="toy2",
        seed=1,
        epochs=400,
        progress_info=0
    ),
    initialization_settings=el.initializer(
        method="random",
        loss_quantile=0,
        iterations=10,
        specs=el.init_specs(
            radius=1.,
            mean=0.
            )
        )
)

global_dict = eliobj.inputs

res_ep, res = eliobj.train(save_file=None)

plt.plot(res_ep["hyperparameter"]["mu0"])
plt.plot(res_ep["hyperparameter"]["mu1"])
plt.plot(res_ep["hyperparameter"]["sigma0"])
plt.plot(res_ep["hyperparameter"]["sigma1"])
plt.plot(res_ep["hyperparameter"]["sigma2"])

res["expert_elicited_statistics"]


# %% RESULTS
path = "elicit/results/parametric_prior/toy_example_0"

# loss function
func.plot_loss(path)

# convergence
func.plot_convergence(path)


# elicited statistics
pd.DataFrame(
    {
        "expert": tf.round(expert_data["quantiles_ypred"][0]).numpy(),
        "predictions": tf.round(
            tf.reduce_mean(
                pd.read_pickle(
                    path + "/elicited_statistics.pkl")["quantiles_ypred"], 0
            )
        ).numpy(),
    },
    index=[f"Q_{i}" for i in [5, 25, 50, 75, 95]],
)


# prior distributions
# extract learned hyperparameter values (for each epoch)
hyp = pd.read_pickle(path + "/final_results.pkl")["hyperparameter"]
# compute final learned hyperparameter value as average of the 30 last epochs
learned_hyp = tf.reduce_mean(tf.stack([hyp[k][-30:] for k in hyp], -1), 0)
# extarct global dictionary
gd = pd.read_pickle(path + "/global_dict.pkl")

# x-axis for sigma parameter
xrge = tf.range(0.0, 50.0, 0.01)
# x-axis for mu parameter
xrge2 = tf.range(140.0, 200.0, 0.01)
# probability density for each model parameter
sigma = gd["model_parameters"]["sigma"]["family"](learned_hyp[2]).prob(xrge)
mu = gd["model_parameters"]["mu"]["family"](
    learned_hyp[1], tf.exp(learned_hyp[0])
).prob(xrge2)

# plot prior distributions
_, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 3))
axs[0].plot(xrge, sigma)
axs[0].set_title(rf"$\sigma \sim Normal_+({learned_hyp[2]:.2f})$")
axs[1].plot(xrge2, mu)
axs[1].set_title(
    rf"$\mu \sim Normal({learned_hyp[1]:.2f}, {tf.exp(learned_hyp[0]):.2f})$"
)
plt.show()
