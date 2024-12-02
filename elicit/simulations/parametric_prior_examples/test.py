import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from elicit.core.run import prior_elicitation
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
        #data=expert_data,
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
        warmup_initializations=1,
        seed=0,
        view_ep=1,
        epochs=3,
        save_log=True
    ),
)



