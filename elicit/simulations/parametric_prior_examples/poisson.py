import tensorflow_probability as tfp
import tensorflow as tf
import sys
import pandas as pd

tfd = tfp.distributions 

from core.run import prior_elicitation
from user.design_matrices import load_design_matrix_poisson
from user.generative_models import PoissonModel
from user.custom_functions import Normal_log, custom_groups

### ground truth:
binomial_truth = {
    "b0": tfd.Normal(2.91, 0.07),
    "b1": tfd.Normal(0.23, 0.05),
    "b2": tfd.Normal(-1.51, 0.135),
    "b3": tfd.Normal(-0.61, 0.105)
    }

### initialize custom distribution
normal_log = Normal_log()

prior_elicitation(
    model_parameters=dict(
        b0=dict(
            family=normal_log,
            hyperparams_dict={
                "mu0": tfd.Uniform(1.,2.5), 
                "log_sigma0": tfd.Uniform(-2.,-5.)
                },
            param_scaling=1.
            ),
        b1=dict(
            family=normal_log,
            hyperparams_dict = {
                "mu1": tfd.Uniform(0.,0.5), 
                "log_sigma1": tfd.Uniform(-2.,-5.)
                },
            param_scaling=1.
            ),
        b2=dict(
            family=normal_log,
            hyperparams_dict = {
                "mu2": tfd.Uniform(-1.,-1.5), 
                "log_sigma2": tfd.Uniform(-2.,-5.)
                },
            param_scaling=1.
            ),
        b3=dict(
            family=normal_log,
            hyperparams_dict = {
                "mu3": tfd.Uniform(-0.5,-1.), 
                "log_sigma3": tfd.Uniform(-2.,-5.)
                },
            param_scaling=1.
            ),
        independence = False
        ),
    expert_data=dict(
        #data = pd.read_pickle("elicit/simulations/LiDO_cluster/experts/deep_binomial/elicited_statistics.pkl"),
        from_ground_truth = True,
        simulator_specs = binomial_truth,
        samples_from_prior = 10000
        ),
    generative_model=dict(
        model=PoissonModel,
        discrete_likelihood=True,
        softmax_gumble_specs={"temperature": 1.,
                              "upper_threshold": 50},
        additional_model_args = {
            "design_matrix": load_design_matrix_poisson()
            }
        ),
    target_quantities=dict(
        group1=dict(
            elicitation_method="quantiles",
            quantiles_specs=(5, 25, 50, 75, 95),
            loss_components = "all",
            custom_target_function={
                "function":custom_groups,
                "additional_args": {
                    "gr": 1
                    }
                }
            ),
        group2=dict(
            elicitation_method="quantiles",
            quantiles_specs=(5, 25, 50, 75, 95),
            loss_components = "all",
            custom_target_function={
                "function":custom_groups,
                "additional_args": {
                    "gr": 2
                    }
                }
            ),
        group3=dict(
            elicitation_method="quantiles",
            quantiles_specs=(5, 25, 50, 75, 95),
            loss_components = "all",
            custom_target_function={
                "function":custom_groups,
                "additional_args": {
                    "gr": 3
                    }
                }
            )
        ),
    optimization_settings=dict(
        optimizer_specs={
            "learning_rate": tf.keras.optimizers.schedules.CosineDecay(
                0.01, 500),
            "clipnorm": 1.0
            }
        ),
    loss_function=dict(
        use_regularization=True
        ),
    training_settings=dict(
        method="parametric_prior",
        sim_id="poisson",
        warmup_initializations=100,
        seed=1,
        epochs=200
    )
    )


# if __name__ == "__main__":
#     seed = int(sys.argv[1])
    
#     simulation_wrapper(seed)


pd.read_pickle("elicit/results/parametric_prior/poisson_1/expert/elicited_statistics.pkl")["quantiles_group3"]
tf.reduce_mean(pd.read_pickle("elicit/results/parametric_prior/poisson_1/elicited_statistics.pkl")["quantiles_group3"],0)

res = pd.read_pickle("elicit/results/parametric_prior/poisson_1/final_results.pkl")

res["hyperparameter"].keys()

res["hyperparameter"]["mu0"][-3:]
res["hyperparameter"]["mu1"][-3:]
res["hyperparameter"]["mu2"][-3:]
res["hyperparameter"]["mu3"][-3:]

tf.exp(res["hyperparameter"]["log_sigma0"][-3:])
tf.exp(res["hyperparameter"]["log_sigma1"][-3:])
tf.exp(res["hyperparameter"]["log_sigma2"][-3:])
tf.exp(res["hyperparameter"]["log_sigma3"][-3:])
