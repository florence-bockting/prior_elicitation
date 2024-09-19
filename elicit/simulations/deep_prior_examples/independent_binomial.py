import tensorflow_probability as tfp
import tensorflow as tf
import sys
import pandas as pd

tfd = tfp.distributions 

from core.run import prior_elicitation
from user.design_matrices import load_design_matrix_binomial
from user.generative_models import BinomialModel

### ground truth:
binomial_truth = {
    "b0": tfd.Normal(0.1, 0.1),
    "b1": tfd.Normal(-0.1, 0.3),
    }


def simulation_wrapper(seed):
    prior_elicitation(
        model_parameters=dict(
            b0=dict(param_scaling=1.),
            b1=dict(param_scaling=1.),
            independence = True
            ),
        normalizing_flow=True, 
        expert_data=dict(
            data = pd.read_pickle("elicit/simulations/LiDO_cluster/experts/deep_binomial/elicited_statistics.pkl"),
            from_ground_truth = False
            ),
        generative_model=dict(
            model=BinomialModel,
            discrete_likelihood=True,
            softmax_gumble_specs={"temperature": 1.,
                                  "upper_threshold": 30},
            additional_model_args = {
                "design_matrix": load_design_matrix_binomial(20),
                "total_count": 30
                }
            ),
        target_quantities=dict(
            ypred=dict(
                elicitation_method="quantiles",
                quantiles_specs=(5, 25, 50, 75, 95),
                loss_components = "by-group"
                )
            ),
        optimization_settings=dict(
            optimizer_specs={
                "learning_rate": tf.keras.optimizers.schedules.CosineDecay(
                    0.001, 1500),
                "clipnorm": 1.0
                }
            ),
        training_settings=dict(
            method="deep_prior",
            sim_id="binomial",
            seed=seed,
            epochs=1000
        )
        )

if __name__ == "__main__":
    seed = int(sys.argv[1])
    
    simulation_wrapper(seed)