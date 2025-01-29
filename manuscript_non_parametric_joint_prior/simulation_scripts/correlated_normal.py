import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import sys

from elicit.core.run import prior_elicitation
from elicit.user.design_matrices import load_design_matrix_normal
from elicit.user.custom_functions import custom_correlation
from elicit.user.generative_models import NormalModel

tfd = tfp.distributions


S = [2.5, 1.3, 0.8]
M = [[1.0, 0.3, -0.3], [0.3, 1.0, -0.2], [-0.3, -0.2, 1.0]]

covariance_matrix = (tf.linalg.diag(S) @ M) @ tf.linalg.diag(S)

truth_correlated = {
    "theta": tfd.JointDistributionSequential(
        [
            tfd.MultivariateNormalTriL(
                loc=[10, 7.0, 2.5],
                scale_tril=tf.linalg.cholesky(covariance_matrix)
            ),
            tfd.Gamma([5.0], [2.0]),
        ]
    )
}


def run_sim(seed):

    prior_elicitation(
        model_parameters=dict(
            b0=dict(param_scaling=1.0),
            b1=dict(param_scaling=1.0),
            b2=dict(param_scaling=1.0),
            sigma=dict(param_scaling=1.0),
            independence=None,
        ),
        normalizing_flow=True,
        expert_data=dict(
            data=pd.read_pickle(
                f"elicit/simulations/LiDO_cluster/experts/deep_correlated_normal/elicited_statistics.pkl"  # noqa
            ),
            from_ground_truth=False,
            # simulator_specs=truth_correlated,
            # samples_from_prior=10_000,
        ),
        generative_model=dict(
            model=NormalModel,
            additional_model_args={
                "design_matrix": load_design_matrix_normal(30)
                },
        ),
        target_quantities=dict(
            group1=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components="all",
            ),
            group2=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components="all",
            ),
            group3=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components="all",
            ),
            logR2=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components="all",
            ),
            correl=dict(
                elicitation_method="identity",
                loss_components="by-group",
                custom_target_function=dict(
                    function=custom_correlation, additional_args=None
                ),
            ),
        ),
        optimization_settings=dict(
            optimizer_specs={
                "learning_rate": 0.0001,
                "clipnorm": 1.0,
            }
        ),
        training_settings=dict(
            method="deep_prior", sim_id="normal_correlated", seed=seed,
            epochs=1500
        ),
    )


if __name__ == "__main__":
    seed = int(sys.argv[1])

    run_sim(seed)
