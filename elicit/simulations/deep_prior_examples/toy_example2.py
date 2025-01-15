# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import elicit as el
import numpy as np

from bayesflow.inference_networks import InvertibleNetwork
from elicit.extras import utils

tfd = tfp.distributions


# numeric, standardized predictor
def std_predictor(N, quantiles):
    X = tf.cast(np.arange(N), tf.float32)
    X_std = (X-tf.reduce_mean(X))/tf.math.reduce_std(X)
    X_sel = tfp.stats.percentile(X_std, quantiles)
    return X_sel


# implemented, generative model
class ToyModel:
    def __call__(self, prior_samples, design_matrix, **kwargs):
        B = prior_samples.shape[0]
        S = prior_samples.shape[1]

        # preprocess shape of design matrix
        X = tf.broadcast_to(design_matrix[None, None,:],
                           (B,S,len(design_matrix)))
        # linear predictor (= mu)
        epred = tf.add(prior_samples[:, :, 0][:,:,None],
                       tf.multiply(prior_samples[:, :, 1][:,:,None], X)
                       )
        # data-generating model
        likelihood = tfd.Normal(
            loc=epred,
            scale=tf.abs(tf.expand_dims(prior_samples[:, :, -1], -1))
        )
        # prior predictive distribution (=height)
        ypred = likelihood.sample()
        
        # selected observations
        y_X0, y_X1, y_X2, y_X3, y_X4 = (ypred[:,:,0], ypred[:,:,1], ypred[:,:,2],
                                        ypred[:,:,3], ypred[:,:,4])

        # log R2 (log for numerical stability)
        log_R2 = utils.log_R2(ypred, epred)

        # correlation
        cor = utils.pearson_correlation(prior_samples)

        return dict(
            likelihood=likelihood,
            ypred=ypred, epred=epred,
            prior_samples=prior_samples,
            y_X0=y_X0, y_X1=y_X1, y_X2=y_X2,y_X3=y_X3, y_X4=y_X4,
            log_R2=log_R2,
            cor=cor
        )

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

elicit = el.Elicit(
    model=el.model(
        obj=ToyModel,
        design_matrix=std_predictor(N=200, quantiles=[5,25,50,75,95])
        ),
    parameters=[
        el.parameter(name="beta0"),
        el.parameter(name="beta1"),
        el.parameter(name="sigma")
    ],
    targets=[
        el.target(
            name=f"y_X{i}",
            query=el.queries.quantiles((5, 25, 50, 75, 95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0
        ) for i in range(5)
        ]+[
        el.target(
            name="log_R2",
            query=el.queries.quantiles((5, 25, 50, 75, 95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0
        ),
        el.target(
            name="correlation",
            query=el.queries.correlation(),
            loss=el.losses.L2,
            weight=1.0
        )
    ],
    #expert=el.expert.data(dat = expert_dat),
    expert=el.expert.simulator(
        ground_truth = ground_truth,
        num_samples = 10_000
    ),
    optimizer=el.optimizer(
        optimizer=tf.keras.optimizers.Adam,
        learning_rate=0.001,
        clipnorm=1.0
        ),
    trainer=el.trainer(
        method="deep_prior",
        name="toy2",
        seed=3,
        epochs=2,#00
    ),
    network=el.networks.NF(
        inference_network=InvertibleNetwork,
        network_specs=dict(
            num_params=3,
            num_coupling_layers=3,
            coupling_design="affine",
            coupling_settings={
                "dropout": False,
                "dense_args": {
                    "units": 128,
                    "activation": "relu",
                    "kernel_regularizer": None,
                },
                "num_dense": 2,
            },
            permutation="fixed"
        ),
        base_distribution=el.networks.base_normal
    )
)

hist = elicit.fit(save_dir=None)


elicit.results["expert_elicited_statistics"]["quantiles_y_X2"]


_, axs = plt.subplots(3,3, constrained_layout=True)
plt.show()

import seaborn as sns

_, axs=plt.subplots(1,6, constrained_layout=True, figsize=(7,3))
for i in range(5):
    axs[i].plot(
        tf.reduce_mean(elicit.results["elicited_statistics"][f"quantiles_y_X{i}"], 0),
        elicit.results["expert_elicited_statistics"][f"quantiles_y_X{i}"][0,:],
        "o", ms=10
        )
    axs[i].axline((0,0), slope=1, color="black", linestyle="dashed")
axs[5].plot(
    tf.reduce_mean(elicit.results["elicited_statistics"][f"quantiles_log_R2"], 0),
    elicit.results["expert_elicited_statistics"][f"quantiles_log_R2"][0,:],
    "o", ms=10
    )
axs[5].axline((0,0), slope=1, color="black", linestyle="dashed")

plt.plot(hist["loss_component"][-100:])

sns.kdeplot(tf.reshape(elicit.results["prior_samples"], (128*200,3))[:,0])
sns.kdeplot(elicit.results["expert_prior_samples"][0,:,0])

plt.plot(hist["hyperparameter"]["means"])
plt.plot(hist["hyperparameter"]["stds"])


tf.reduce_mean(elicit.results["expert_prior_samples"], (0,1))
tf.reduce_mean(elicit.results["prior_samples"], (0,1))

tf.math.reduce_std(elicit.results["expert_prior_samples"], (0,1))
tf.math.reduce_std(elicit.results["prior_samples"], (0,1))

res_ep["hyperparameter"].keys()

tf.stack(res_ep["hyperparameter"]["means"],0)
tf.stack(res_ep["hyperparameter"]["stds"],0)

plt.plot(res_ep["hyperparameter"]["means"])
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
