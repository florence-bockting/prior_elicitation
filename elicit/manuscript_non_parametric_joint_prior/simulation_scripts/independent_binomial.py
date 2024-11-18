import tensorflow_probability as tfp
import pandas as pd
import sys

from elicit.core.run import prior_elicitation
from elicit.user.design_matrices import load_design_matrix_binomial
from elicit.user.generative_models import BinomialModel
from elicit.user.custom_functions import quantiles_per_ypred

tfd = tfp.distributions

def run_prior_checks(seed, mu0, sigma0, mu1, sigma1):

    prior_elicitation(
        model_parameters=dict(
            b0=dict(param_scaling=1.0),
            b1=dict(param_scaling=1.0),
            independence=dict(corr_scaling=0.1)
        ),
        normalizing_flow=True,
        expert_data=dict(
            data=pd.read_pickle(
                "elicit/simulations/LiDO_cluster/experts/deep_binomial/elicited_statistics.pkl" # noqa
            ),
            from_ground_truth=True,
            simulator_specs={
                "b0": tfd.Normal(0.1, 0.1),
                "b1": tfd.Normal(-0.1, 0.3),
            },
            samples_from_prior=10_000,
        ),
        generative_model=dict(
            model=BinomialModel,
            additional_model_args={
                "total_count": 30,
                "design_matrix": load_design_matrix_binomial(50)
                },
            discrete_likelihood = True,
            softmax_gumble_specs = {
                "temperature": 1.,
                "upper_threshold": 30
            }
        ),
        target_quantities=dict(
            ypred=dict(
                elicitation_method=None,
                custom_elicitation_method=dict(
                    function=quantiles_per_ypred,
                    additional_args={
                        "quantiles_specs": (5, 25, 50, 75, 95)
                        }
                    ),
                loss_components="by-group"
            )
        ),
        optimization_settings=dict(
            optimizer_specs={
                "learning_rate": 0.0001,
                "clipnorm": 1.0,
            }
        ),
        training_settings=dict(
            method="deep_prior",
            sim_id="binomial",
            seed=seed,
            epochs=600
        ),
    )

if __name__ == "__main__":
    seed = int(sys.argv[1])
    
    run_sim(seed)
