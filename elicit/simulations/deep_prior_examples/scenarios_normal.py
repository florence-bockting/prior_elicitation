import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import sys

from elicit.core.run import prior_elicitation
from elicit.user.design_matrices import load_design_matrix_normal
from elicit.user.generative_models import NormalModel

tfd = tfp.distributions

truth_independent = {
    "b0": tfd.Normal(10., 2.5),
    "b1": tfd.Normal(7., 1.3),
    "b2": tfd.Normal(2.5, 0.8),
    "sigma": tfd.Gamma(5., 2.)
}

truth_skewed = {
    "b0": tfd.Normal(10., 2.5),
    "b1": tfd.TwoPieceNormal(7., 1.3,4),
    "b2": tfd.TwoPieceNormal(2.5, 0.8,4),
    "sigma": tfd.Gamma(5., 2.)
}

S = [2.5, 1.3, .8]
M = [[1., 0.95, -0.99],
     [0.95, 1., -0.95],
     [-0.99, -0.95, 1.]]
covariance_matrix = (tf.linalg.diag(S) @ M) @ tf.linalg.diag(S)

truth_correlated = {
    "theta": tfd.JointDistributionSequential([
        tfd.MultivariateNormalTriL(
            loc=[10, 7., 2.5],
            scale_tril=tf.linalg.cholesky(covariance_matrix)
            ),
        tfd.Gamma([5.],[ 2.])
        ])
}

scenario="independent"
seed=1

def run_sim(seed, scenario):

    prior_elicitation(
        model_parameters=dict(
            b0=dict(param_scaling=1.0),
            b1=dict(param_scaling=1.0),
            b2=dict(param_scaling=1.0),
            sigma=dict(param_scaling=1.0),
            independence=dict(corr_scaling=0.1)
        ),
        normalizing_flow=True,
        expert_data=dict(
            #data=pd.read_pickle(
            #    f"elicit/simulations/LiDO_cluster/experts/deep_{scenario}_normal/elicited_statistics.pkl" # noqa
            #),
            from_ground_truth=True,
            simulator_specs=truth_independent,
            samples_from_prior=10_000,
        ),
        generative_model=dict(
            model=NormalModel,
            additional_model_args={
                "design_matrix": load_design_matrix_normal(30)
                }
        ),
        target_quantities=dict(
            group1=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components="all"
            ),
            group2=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components="all"
            ),
            group3=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components="all"
            ),
            logR2=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),#(5,10,20,30,40,50,60,70,80,90,95),
                loss_components="all"
            )
        ),
        optimization_settings=dict(
            optimizer_specs={
                "learning_rate": 0.00025,
                "clipnorm": 1.0,
            }
        ),
        # loss_function=dict(
        #     use_regularization=True
        #     ),
        training_settings=dict(
            method="deep_prior",
            sim_id=f"normal_{scenario}",
            seed=seed,
            epochs=1500
        ),
    )

if __name__ == "__main__":
    seed = int(sys.argv[1])
    scenario = str(sys.argv[2])
    
    run_sim(seed, scenario)


normals_convergence("elicit/results/deep_prior/normal_independent_1",
                    "elicit/results/deep_prior/normal_independent_1/expert", "", 
                    model=scenario,
                    save_fig=False)

tf.reduce_mean(pd.read_pickle("elicit/results/deep_prior/normal_independent_1/elicited_statistics.pkl")["quantiles_logR2"],0)

tf.reduce_mean(pd.read_pickle("elicit/results/deep_prior/normal_independent_1/expert/elicited_statistics.pkl")["quantiles_logR2"],0)
