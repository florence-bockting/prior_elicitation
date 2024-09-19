import tensorflow_probability as tfp
import tensorflow as tf
import sys
import pandas as pd

tfd = tfp.distributions 

from core.run import prior_elicitation
from user.design_matrices import load_design_matrix_binomial2
from user.generative_models import BinomialModel2
from user.custom_functions import Normal_log

### ground truth:
binomial_truth = {
    "b0": tfd.Normal(-0.51, 0.06),
    "b1": tfd.Normal(0.26, 0.04)
    }

### initialize custom distribution
normal_log = Normal_log()

prior_elicitation(
    model_parameters=dict(
        b0=dict(
            family=normal_log,
            hyperparams_dict={
                "mu0": tfd.Normal(0.,1.), 
                "log_sigma0": tfd.Uniform(-2.,-3.)
                },
            param_scaling=1.
            ),
        b1=dict(
            family=normal_log,
            hyperparams_dict = {
                "mu1": tfd.Normal(0.,1.), 
                "log_sigma1": tfd.Uniform(-2.,-3.)
                },
            param_scaling=1.
            ),
        independence = False
        ),
    expert_data=dict(
        #data = pd.read_pickle("elicit/simulations/LiDO_cluster/experts/parametric_binomial/elicited_statistics.pkl"),
        from_ground_truth = True,
        simulator_specs = binomial_truth,
        samples_from_prior = 10000
        ),
    generative_model=dict(
        model=BinomialModel2,
        discrete_likelihood=True,
        softmax_gumble_specs={"temperature": 1.,
                              "upper_threshold": 31},
        additional_model_args = {
            "design_matrix": load_design_matrix_binomial2(),
            "total_count": 31
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
                0.01, 1500),
            "clipnorm": 1.0
            }
        ),
    training_settings=dict(
        method="parametric_prior",
        sim_id="binomial",
        warmup_initializations=1,#00,
        seed=1,
        epochs=1,#200
    )
    )

# if __name__ == "__main__":
#     seed = int(sys.argv[1])
    
#     simulation_wrapper(seed)


pd.read_pickle("elicit/results/parametric_prior/binomial_0/prior_samples.pkl")

res["hyperparameter"].keys()

tf.exp(res["hyperparameter"]["log_sigma0"][-3:])
tf.exp(res["hyperparameter"]["log_sigma1"][-3:])
