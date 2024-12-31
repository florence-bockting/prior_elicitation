# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import elicit as el

from elicit.user.generative_models import ToyModel
from elicit.plotting import func

tfd = tfp.distributions


ground_truth = {
    "mu": tfd.Normal(loc=5, scale=2),
    "sigma": tfd.HalfNormal(scale=10.0),
}

expert_data = pd.read_pickle(
    "elicit/simulations/parametric_prior_examples/expert_data/toy-example/elicited_statistics.pkl"
)

eliobj = el.Elicit(
    model=el.model(
        obj=ToyModel,
        N=5,
        ),
    parameters=[
        el.parameter(
            name="mu",
            family=tfd.Normal,
            hyperparams=dict(
                loc=el.hyper("mu0"),
                scale=el.hyper("sigma0", lower=0)
                )
        ),
        el.parameter(
            name="sigma",
            family=tfd.HalfNormal,
            hyperparams=dict(
                scale=el.hyper("sigma1", lower=0)
                )
        ),
    ],
    target_quantities=[
        el.target(
            name="ypred",
            elicitation_method=el.eli_method.quantiles((5, 25, 50, 75, 95)),
            loss=el.MMD_energy,
            loss_weight=1.0
        )
    ],
    # expert_data=el.expert.data(
    #     data=None,
    # ),
    expert=el.expert.simulate(
        ground_truth = ground_truth,
        num_samples = 10_000
    ),
    optimization_settings=el.optimizer(
        specs=dict(
            learning_rate=0.05,
            clipnorm=1.0
            )
        ),
    training_settings=el.train(
        method="parametric_prior",
        name="toy_example",
        seed=0,
        epochs=400,
    ),
    initialization_settings=el.initializer(
        method="random",
        loss_quantile=0,
        iterations=3,
        specs=el.init_specs(
            radius=1,
            mean=0
            )
        )
)

global_dict=eliobj.inputs

res_ep, res = eliobj.train(save_file=None)

plt.plot(res_ep["hyperparameter"]["mu0"])
plt.plot(res_ep["hyperparameter"]["sigma0"])
plt.plot(res_ep["hyperparameter"]["sigma1"])

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
