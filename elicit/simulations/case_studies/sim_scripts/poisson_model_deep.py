import tensorflow_probability as tfp
import tensorflow as tf
import keras
import pandas as pd
import sys

tfd = tfp.distributions 

from functions.loss_functions import MMD_energy
from setup.input_functions import param, model, target, loss, expert, normalizing_flow_specs, optimization, prior_elicitation

def run_simulation(seed):
    #%% Model parameters
    def model_params():  
        return (
            param(name = "b0"),
            param(name = "b1"),
            param(name = "b2"),
            param(name = "b3"),
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
                    }
                )
            )

    #%% Expert input
    def expert_input():
        return expert(data = None,
                      simulate_data = True,
                      simulator_specs = {
                          "b0": tfd.Normal(2.91, 0.07),
                          "b1": tfd.Normal(0.23, 0.05),
                          "b2": tfd.Normal(-1.51, 0.135),
                          "b3": tfd.Normal(-0.61, 0.105)
                          })

    #%% Generative model
    from user_input.generative_models import GenerativePoissonModel

    # you can either import the design matrix by calling a function 
    # which will downlaod the dataset from internet (which takes a while)
    # from user_input.design_matrices import load_design_matrix_equality
    # design_matrix = load_design_matrix_equality("standardize", selected_obs = [0, 13, 14, 35, 37, 48]e)

    # or you use the saved version from path
    design_matrix = pd.read_pickle("elicit/simulations/data/design_matrix_pois.pkl")

    def generative_model():
        return model(GenerativePoissonModel,
                     additional_model_args = {"total_count": 90, 
                                              "design_matrix": design_matrix},
                     discrete_likelihood = True,
                     softmax_gumble_specs = {"temperature": 1.,
                                             "upper_threshold": 90}
                    )

    #%% Target quantities and elicited statistics
    from user_input.custom_functions import custom_group_means

    def target_quantities():
        return (
            target(name = "ypred",
                    elicitation_method = "histogram",
                    loss_components = "by-group"
                    ),
            target(name = "group_means",
                    elicitation_method = "quantiles",
                    quantiles_specs = (5, 25, 50, 75, 95),
                    custom_target_function = {
                        "function": custom_group_means,
                        "additional_args": {"design_matrix": design_matrix,
                                            "factor_indices": [0,2,3]
                                            }
                        },
                    loss_components = "by-group"
                    )
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

    #%% global method function
    prior_elicitation(
        method = "deep_prior",
        sim_id = f"pois_deep_{seed}",
        B = 128,
        rep = 200,
        seed = seed,
        burnin = 1,
        epochs = 1000,
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

# from validation.misc_plotting.plot_pois_res import plot_results_overview

# path = "elicit/simulations/case_studies/sim_results/deep_prior/"
# file = "pois_deep_34800707"
# title = "Binomial model 7 coupling layers"

# plot_results_overview(path, file, title)