import tensorflow_probability as tfp
import tensorflow as tf
import sys

tfd = tfp.distributions 

from core.run import prior_elicitation
from user.generative_models import ToyModel
from user.custom_functions import Normal_log
from core.write_results import create_output_summary

normal_log = Normal_log()

prior_elicitation(
    model_parameters=dict(
        mu=dict(family=normal_log, 
                hyperparams_dict={
                      "mu_loc": tfd.Uniform(100.,200.),
                      "log_mu_scale": tfd.Uniform(0., 5.)
                      },
                param_scaling=1.
                ),
        sigma=dict(family=tfd.HalfNormal,
                   hyperparams_dict={
                      "sigma_scale": tfd.Uniform(1.,50.)
                      },
                   param_scaling=1.
                   ),
        independence = False
        ),
    expert_data=dict(
        from_ground_truth = True,
        simulator_specs = {
            "mu": tfd.Normal(loc=170, scale=2),
            "sigma": tfd.HalfNormal(scale=10.),
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
            quantiles_specs=(5, 25, 50, 75, 95),
            loss_components = "all"
            )
        ),
    optimization_settings=dict(
        optimizer_specs={
            "learning_rate": tf.keras.optimizers.schedules.CosineDecay(
                0.1, 700),
            "clipnorm": 1.0
            }
        ),
    training_settings=dict(
        method="parametric_prior",
        sim_id="toy_example",
        warmup_initializations=10,
        seed=1,
        epochs=400
    )
    )

import pandas as pd

pd.read_pickle("elicit/results/parametric_prior/toy_example_1/model_simulations.pkl")["ypred"]


pd.read_pickle("elicit/results/parametric_prior/toy_example_1/expert/elicited_statistics.pkl")["quantiles_ypred"]
tf.reduce_mean(
    pd.read_pickle("elicit/results/parametric_prior/toy_example_1/elicited_statistics.pkl")["quantiles_ypred"],
    (0))

res = pd.read_pickle("elicit/results/parametric_prior/toy_example_1/final_results.pkl")["hyperparameter"]

res["sigma_scale"]
tf.exp(res["sigma_scale"])




global_dict = pd.read_pickle("elicit/results/parametric_prior/toy_example_0/global_dict.pkl")
create_output_summary("elicit/results/parametric_prior/toy_example_0", global_dict)
