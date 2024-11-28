import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import sys

from elicit.core.run import prior_elicitation
from elicit.user.design_matrices import load_design_matrix_normal
from elicit.user.generative_models import NormalModel

tfd = tfp.distributions

truth_independent = {
    "b0": tfd.Normal(10.0, 2.5),
    "b1": tfd.Normal(7.0, 1.3),
    "b2": tfd.Normal(2.5, 0.8),
    "sigma": tfd.Gamma(5.0, 2.0),
}

truth_skewed = {
    "b0": tfd.Normal(10.0, 2.5),
    "b1": tfd.TwoPieceNormal(7.0, 1.3, 4),
    "b2": tfd.TwoPieceNormal(2.5, 0.8, 4),
    "sigma": tfd.Gamma(5.0, 2.0),
}

S = [2.5, 1.3, 0.8]
M = [[1.0, 0.95, -0.99], [0.95, 1.0, -0.95], [-0.99, -0.95, 1.0]]
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


def run_sim(seed, scenario):

    prior_elicitation(
        model_parameters=dict(
            b0=dict(param_scaling=1.0),
            b1=dict(param_scaling=1.0),
            b2=dict(param_scaling=1.0),
            sigma=dict(param_scaling=1.0),
            independence=dict(corr_scaling=0.1),
        ),
        normalizing_flow=True,
        expert_data=dict(
            data=pd.read_pickle(
                f"elicit/simulations/LiDO_cluster/experts/deep_{scenario}_normal/elicited_statistics.pkl"  # noqa
            ),
            from_ground_truth=False,
            # simulator_specs=truth_skewed,
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
        ),
        optimization_settings=dict(
            optimizer_specs={
                "learning_rate": 0.00025,
                "clipnorm": 1.0,
            }
        ),
        training_settings=dict(
            method="deep_prior",
            sim_id=f"normal_{scenario}",
            seed=seed,
            epochs=1,  # 500
        ),
    )


if __name__ == "__main__":
    seed = int(sys.argv[1])
    scenario = str(sys.argv[2])

    run_sim(seed, scenario)
