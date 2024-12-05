# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from elicit.main import prior_elicitation
from elicit.user.generative_models import ToyModel
from elicit.user.custom_functions import Normal_log
from elicit.plotting import func

tfd = tfp.distributions

normal_log = Normal_log()

ground_truth = {
    "mu": tfd.Normal(loc=170, scale=2),
    "sigma": tfd.HalfNormal(scale=10.0),
}

expert_data = pd.read_pickle(
    "elicit/simulations/parametric_prior_examples/expert_data/toy-example/elicited_statistics.pkl"
)

prior_elicitation(
    model_parameters=dict(
        mu=dict(
            family=normal_log,
            hyperparams_dict={
                "mu_loc": tfd.Uniform(100.0, 300.0),
                "log_mu_scale": tfd.Uniform(0.0, 5.0),
            },
            param_scaling=1.0,
        ),
        sigma=dict(
            family=tfd.HalfNormal,
            hyperparams_dict={"sigma_scale": tfd.Uniform(1.0, 50.0)},
            param_scaling=1.0,
        ),
        independence=None,
    ),
    expert_data=dict(
        data=expert_data,
        from_ground_truth=True,
        simulator_specs = ground_truth,
        samples_from_prior = 10000
    ),
    generative_model=dict(model=ToyModel, additional_model_args={"N": 200}),
    target_quantities=dict(
        ypred=dict(
            elicitation_method="quantiles",
            quantiles_specs=(5, 25, 50, 75, 95),
            loss_components="all",
        )
    ),
    optimization_settings=dict(optimizer_specs={"learning_rate": 0.1,
                                                "clipnorm": 1.0}),
    training_settings=dict(
        method="parametric_prior",
        sim_id="toy_example",
        warmup_initializations=50,
        seed=0,
        view_ep=50,
        epochs=10,
    ),
)

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
