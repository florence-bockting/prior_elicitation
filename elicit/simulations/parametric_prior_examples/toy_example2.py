# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import elicit as el

from elicit.extras import utils

tfd = tfp.distributions

# numeric, standardized predictor
def std_predictor(N, quantiles):
    X = tf.cast(np.arange(N), tf.float32)
    X_std = (X-tf.reduce_mean(X))/tf.math.reduce_std(X)
    X_sel = tfp.stats.percentile(X_std, quantiles)
    return X_sel

# implemented, generative model
class ToyModel2:
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
            loc=epred, scale=tf.expand_dims(prior_samples[:, :, -1], -1)
        )
        # prior predictive distribution (=height)
        ypred = likelihood.sample()
        
        # selected observations
        y_X0, y_X1, y_X2 = (ypred[:,:,0], ypred[:,:,1], ypred[:,:,2])

        # log R2 (log for numerical stability)
        log_R2 = utils.log_R2(ypred, epred)

        return dict(
            likelihood=likelihood,
            ypred=ypred, epred=epred,
            prior_samples=prior_samples,
            y_X0=y_X0, y_X1=y_X1, y_X2=y_X2,
            log_R2=log_R2
        )


# ground_truth = {
#     "beta0": tfd.Normal(loc=5, scale=1),
#     "beta1": tfd.Normal(loc=2, scale=1),
#     "sigma": tfd.HalfNormal(scale=10.0),
# }

# ground_truth = {
#     "theta": tfd.MultivariateNormalDiag(tf.zeros(3), tf.ones(3))
#     }

ground_truth = {
    "betas": tfd.MultivariateNormalDiag([5.,2.], [1.,1.]),
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
        design_matrix=std_predictor(N=200, quantiles=[25,50,75])
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
    targets=[
        el.target(
            name="y_X0",
            query=el.queries.quantiles((.05, .25, .50, .75, .95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0
        ),
        el.target(
            name="y_X1",
            query=el.queries.quantiles((.05, .25, .50, .75, .95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0
        ),
        el.target(
            name="y_X2",
            query=el.queries.quantiles((.05, .25, .50, .75, .95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0
        ),
        # el.target(
        #     name="log_R2",
        #     query=el.queries.quantiles((.05, .25, .50, .75, .95)),
        #     loss=el.losses.MMD2(kernel="energy"),
        #     weight=1.0
        # )
    ],
    # expert=el.expert.data(dat = expert_dat),
    expert=el.expert.simulator(
        ground_truth = ground_truth,
        num_samples = 10_000
    ),
    optimizer=el.optimizer(
        optimizer=tf.keras.optimizers.Adam,
        learning_rate=0.1,
        clipnorm=1.0
        ),
    trainer=el.trainer(
        method="parametric_prior",
        seed=0,
        epochs=600
    ),
    initializer=el.initializer(
        hyperparams = dict(
            mu0=0., sigma0=el.utils.LowerBound(lower=0.).forward(0.3),
            mu1=1., sigma1=el.utils.LowerBound(lower=0.).forward(0.5),
            sigma2=el.utils.LowerBound(lower=0.).forward(0.4)
        )
    )
    #network = el.networks.NF(...) # TODO vs. el.normalizing_flow(...)
)

eliobj.fit()
#el.utils.get_expert_datformat(targets)

el.plots.hyperparameter(eliobj)

eliobj.save(file="test2")

elicit.update(overwrite=True, expert=el.expert.data(dat = expert_dat),
              name="update_eliobj")

dir(eliobj)

elicit.update(parameter={"some":12})
elicit.save(name="toytest") # saved in ./model1.pkl
elicit.save(file=None, name="model1") # saved in ./res/parametric_prior/model1_1.pkl

tf.reduce_mean(eliobj.results["expert_prior_samples"], (0,1))

elicit_copy.save(file="model1.pkl")

elicit.trainer

elicit.fit()
elicit.save("res")

el.plots.initialization(elicit, cols=5, figsize=(7,3))
el.plots.priors(elicit, constraints=None, figsize=(4,4))

el.utils.save(elicit, "res")

final_res = pd.read_pickle("./results/parametric_prior/toy2_lhs_0.pkl")

final_res["history"].keys()
final_res["results"].keys()

elicit.history

elicit.results.keys() # additional saved results
elicit.history.keys() # equiv. "across_epochs" (loss, loss_component, time, hyperparameter, hyperparameter_gradient)

# save elicit obj
el.utils.save_elicit(elicit, "./results/elicit_empty.pkl")

elicit_loaded2 = el.utils.load_elicit("./results/elicit_empty.pkl")

hist_loaded = elicit_loaded.fit(save_dir=None)

plt.plot(hist["hyperparameter"]["mu0"], label="mu0")
plt.plot(hist["hyperparameter"]["sigma0"], label="sigma0")
plt.plot(hist["hyperparameter"]["mu1"], label="mu1")
plt.plot(hist["hyperparameter"]["sigma1"], label="sigma1")
plt.plot(hist["hyperparameter"]["sigma2"], label="sigma2")
plt.legend()


import seaborn as sns

lhs=elicit.results["init_matrix"]["mu0"][:,0]

sns.ecdfplot(random, label="random")
sns.ecdfplot(sobol, label="sobol")
sns.ecdfplot(lhs, label="lhs")
#plt.axline((0,0), (30,30))
plt.legend()


init_sobol=pd.read_pickle("elicit/results/parametric_prior/toy2_sobol_1/initialization_matrix.pkl")
init_lhs=pd.read_pickle("elicit/results/parametric_prior/toy2_lhs_1/initialization_matrix.pkl")
init_random=pd.read_pickle("elicit/results/parametric_prior/toy2_random_1/initialization_matrix.pkl")


import seaborn as sns
_, axs = plt.subplots(1,4, constrained_layout=True, figsize=(8,2),
                      sharey=True)
for i, n in enumerate(["mu0", "sigma0", "mu1", "sigma1"]):
    sns.ecdfplot(init_sobol[n], label="sobol", ax=axs[i])
    sns.ecdfplot(init_lhs[n], label="lhs", ax=axs[i])
    sns.ecdfplot(init_random[n], label="random", ax=axs[i])
    axs[i].set_title(n, size="medium")
axs[0].legend(handlelength=0.2, frameon=False, fontsize="small")
plt.suptitle("ecdf of initializations using different sampling techniques")




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
