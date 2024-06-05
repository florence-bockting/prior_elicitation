import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import sys

tfd = tfp.distributions 

from functions.loss_functions import MMD_energy
from setup.input_functions import param, model, target, loss, expert, optimization, prior_elicitation, normalizing_flow_specs

def run_simulation(seed):
    #%% Model parameters
    def model_params():  
        return (
            param(name = "b0"),
            param(name = "b1"),
            normalizing_flow_specs(
                num_coupling_layers = 3,
                coupling_design = "affine", 
                coupling_settings = {
                    "dropout": False,
                    "dense_args": {
                        "units": 128,
                        "activation": "softmax",
                        "kernel_regularizer": tf.keras.regularizers.l2(1e-4)
                        },
                    "num_dense": 2
                    },
                permutation="learnable"
                )
            )
    
    #%% Expert input
    def expert_input():
        return expert(data = None,
                      simulate_data = True,
                      simulator_specs = {
                          "b0": tfd.Normal(-0.51, 0.06),
                          "b1": tfd.Normal(0.26, 0.04)
                          })
    
    #%% Generative model
    from user_input.generative_models import GenerativeBinomialModel
    
    design_matrix = pd.read_pickle("elicit/simulations/data/design_matrix_binom.pkl")
    
    def generative_model():
        return model(GenerativeBinomialModel,
                     additional_model_args = {
                         "total_count": 31, 
                         "design_matrix": design_matrix},
                     discrete_likelihood = True,
                     softmax_gumble_specs = {"temperature": 1.,
                                             "upper_threshold": 31}
                    )
    
    #%% Target quantities and elicited statistics
    def target_quantities():
        return (
            target(name = "ypred",
                   elicitation_method = "quantiles",
                   quantiles_specs = (5, 25, 50, 75, 95),
                   loss_components = "by-group"
                   ),
            )   
    #%% Loss function
    def loss_function():
        return loss(loss_function = MMD_energy,
                    loss_weighting = None
                    )
    
    #%% Training settings
    def optimization_settings():
        return optimization(
                        optimizer = tf.keras.optimizers.Adam,
                        optimizer_specs = {
                            "learning_rate": tf.keras.optimizers.schedules.CosineDecayRestarts(
                                0.001, 25),
                            "clipnorm": 1.0
                            }
                        )
    
    ##% global method function
    prior_elicitation(
        method = "deep_prior",
        sim_id = f"binom_{seed}",
        B = 128,
        rep = 200,
        seed = seed,
        burnin = 1,
        epochs = 500,
        output_path = "results",
        model_params = model_params,
        expert_input = expert_input,
        generative_model = generative_model,
        target_quantities = target_quantities,
        loss_function = loss_function,
        optimization_settings = optimization_settings,
        view_ep = 1,
        print_info = True
        )


if __name__ == "__main__":
    seed = int(sys.argv[1])
    
    run_simulation(seed)

#run_simulation(111)
# from validation.plot_binom_res import plot_results_overview

# path = "elicit/simulations/results/data/deep_prior/"
# file = "binom_34738840_7"
# title = "Binomial model 7 coupling layers"

# plot_results_overview(path, file, title)
