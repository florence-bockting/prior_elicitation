import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import sys

tfd = tfp.distributions 

from core.run import prior_elicitation
from user.design_matrices import load_design_matrix_normal
from user.generative_models import NormalModel

###### ground truth: case study 2
independent_normal = {
    "b0": tfd.Normal(10., 2.5),
    "b1": tfd.Normal(7., 1.3), 
    "b2": tfd.Normal(2.5, 0.8),
    "sigma": tfd.Gamma(5., 2.)
    }

###### ground truth: case study 3
skewed_normal = {
    "b0": tfd.Normal(10., 2.5),
    "b1": tfd.TwoPieceNormal(7., 1.3,4), 
    "b2": tfd.TwoPieceNormal(2.5, 0.8,4),
    "sigma": tfd.Gamma(5., 2.)
    }

###### ground truth: case study 4
# Note: For running simulated data from ground truth, you need to comment out
# line 574 in file core/run.py (i.e., the line "save_as_pkl(global_dict, path)")
# As otherwise pickle will throw an error: can't pickle local object

S = [2.5, 1.3, .8]
M = [[1., 0.95, -0.99],
     [0.95, 1., -0.95],
     [-0.99, -0.95, 1.]]

covariance_matrix = (tf.linalg.diag(S) @ M) @ tf.linalg.diag(S)

correlated_normal = {
    "theta": tfd.JointDistributionSequential([
              tfd.MultivariateNormalTriL(
                        loc=[10, 7., 2.5], 
                        scale_tril=tf.linalg.cholesky(covariance_matrix)
                        ),
            tfd.Gamma([5.],[ 2.])
            ])
    }

def case_study(scenario, ground_truth, seed):

    prior_elicitation(
        model_parameters=dict(
            b0=dict(param_scaling=1.),
            b1=dict(param_scaling=1.),
            b2=dict(param_scaling=1.),
            sigma=dict(param_scaling=1.),
            independence = True
            ),
        normalizing_flow=True, 
        expert_data=dict(
            data = pd.read_pickle(f"elicit/simulations/LiDO_cluster/experts/deep_{scenario}/elicited_statistics.pkl"),
            from_ground_truth = False
            ),
        generative_model=dict(
            model=NormalModel,
            additional_model_args = {
                "design_matrix": load_design_matrix_normal(N_group=30)
                }
            ),
        target_quantities=dict(
            group1=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components = "all"
                ),
            group2=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components = "all"
                ),
            group3=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components = "all"
                ),
            logR2=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5,10,20,30,40,50,60,70,80,90,95),
                loss_components = "all"
                )
            ),
        optimization_settings=dict(
            optimizer_specs={
                "learning_rate": tf.keras.optimizers.schedules.CosineDecay(
                    0.0005, 1500),
                "clipnorm": 1.0
                }
            ),
        loss_function=dict(
            use_regularization=True
            ),
        training_settings=dict(
            method="deep_prior",
            sim_id=scenario,
            seed=seed,
            epochs=1000
        )
        )

if __name__ == "__main__":
    scenario = str(sys.argv[1])
    ground_truth = str(sys.argv[2])
    seed = int(sys.argv[3])
    
    case_study(scenario, ground_truth, seed)
    