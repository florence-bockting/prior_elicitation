import tensorflow_probability as tfp

from elicit.main import prior_elicitation
from elicit.user.generative_models import ToyModel

tfd = tfp.distributions

prior_elicitation(
    model_parameters=dict(
        mu=dict(param_scaling=1.0),
        sigma=dict(param_scaling=1.0),
        independence=dict(corr_scaling=0.1),
    ),
    normalizing_flow=True,
    expert_data=dict(
        from_ground_truth=True,
        simulator_specs={
            "mu": tfd.Normal(loc=170, scale=2),
            "sigma": tfd.Gamma(2, 5),
        },
        samples_from_prior=10000,
    ),
    generative_model=dict(model=ToyModel,
                          additional_model_args={"N": 200}),
    target_quantities=dict(
        ypred=dict(
            elicitation_method="quantiles",
            quantiles_specs=(5, 25, 50, 75, 95),
            loss_components="all",
        )
    ),
    optimization_settings=dict(
        optimizer_specs={
            "learning_rate": 0.001,
            "clipnorm": 1.0,
        }
    ),
    initialization_settings=dict(
        method=None,
        loss_quantile=0,
        number_of_iterations=10,
        ),
    training_settings=dict(
        method="deep_prior",
        sim_id="toy_example",
        seed=2,
        epochs=2,  # 500
    ),
)
