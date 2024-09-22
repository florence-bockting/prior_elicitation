import tensorflow_probability as tfp
import tensorflow as tf
import sys

tfd = tfp.distributions 

from elicit.core.run import prior_elicitation
from elicit.user.generative_models import ToyModel

prior_elicitation(
    model_parameters=dict(
        mu=dict(param_scaling=1.),
        sigma=dict(param_scaling=1.),
        independence = dict(corr_scaling=0.1)
        ),
    normalizing_flow=True,
    expert_data=dict(
        from_ground_truth = True,
        simulator_specs = {
            "mu": tfd.Normal(loc=170, scale=2),
            "sigma": tfd.Gamma(2,5),
            },
        samples_from_prior = 10000
        ),
    generative_model=dict(
        model=ToyModel,
        additional_model_args={
            "N": 200
            }
        ),
    target_quantities=dict(
        ypred=dict(
            elicitation_method="quantiles",
            quantiles_specs=(5,25,50,75,95),
            loss_components = "all"
            )
        ),
    optimization_settings=dict(
        optimizer_specs={
            "learning_rate": 0.001,#tf.keras.optimizers.schedules.CosineDecay(
              #  0.001, 700),
            "clipnorm": 1.0
            }
        ),
    training_settings=dict(
        method="deep_prior",
        sim_id="toy_example",
        seed=2,
        epochs=500
    )
    )

import pandas as pd
import seaborn as sns

prior=pd.read_pickle("elicit/results/deep_prior/toy_example_2/model_simulations.pkl")["prior_samples"]
prior_exp=pd.read_pickle("elicit/results/deep_prior/toy_example_2/expert/model_simulations.pkl")["prior_samples"]

pd.read_pickle("elicit/results/deep_prior/toy_example_2/expert/elicited_statistics.pkl")["quantiles_ypred"]
tf.reduce_mean(
    pd.read_pickle("elicit/results/deep_prior/toy_example_2/elicited_statistics.pkl")["quantiles_ypred"],
    (0))

tfp.stats.correlation(prior,sample_axis=1, event_axis=-1)[:,0,1]
sns.kdeplot(tf.reshape(prior[:,:,0], (128*200)))
sns.kdeplot(tf.reshape(prior_exp[:,:,0], (1*10000)))

sns.kdeplot(tf.reshape(prior_exp[:,:,1],(1*10000)))
sns.kdeplot(tf.reshape(tf.abs(prior[:,:,1]),(128*200)))
pd.read_pickle("elicit/results/deep_prior/toy_example_1/model_simulations.pkl")["prior_samples"]


res = pd.read_pickle("elicit/results/parametric_prior/toy_example_1/final_results.pkl")["hyperparameter"]

res["sigma_scale"]
tf.exp(res["sigma_scale"])




global_dict = pd.read_pickle("elicit/results/parametric_prior/toy_example_0/global_dict.pkl")
create_output_summary("elicit/results/parametric_prior/toy_example_0", global_dict)
