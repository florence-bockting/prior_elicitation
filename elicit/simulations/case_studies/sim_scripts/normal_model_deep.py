import tensorflow_probability as tfp
import tensorflow as tf
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
            param(name = "b2"),
            param(name = "b3"),
            param(name = "b4"),
            param(name = "b5"),
            param(name = "sigma"),
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
    from user_input.custom_functions import Gamma_inv_softplus
    gamma_inv_softplus = Gamma_inv_softplus()
    
    def expert_input():
        return expert(data = None,
                      simulate_data = True,
                      simulator_specs = {
                          "b0": tfd.Normal(0.12, 0.02),
                          "b1": tfd.Normal(0.15, 0.02),
                          "b2": tfd.Normal(-0.02, 0.06),
                          "b3": tfd.Normal(-0.03, 0.06),
                          "b4": tfd.Normal(-0.02, 0.03),
                          "b5": tfd.Normal(-0.04, 0.03),
                          "sigma": gamma_inv_softplus(20., 200.)
                          })
    
    #%% Generative model
    from user_input.generative_models import GenerativeNormalModel
    from user_input.design_matrices import load_design_matrix_truth
    
    design_matrix = load_design_matrix_truth(n_group=60)
    
    def generative_model():
        return model(GenerativeNormalModel,
                     additional_model_args = {
                         "design_matrix": design_matrix},
                     discrete_likelihood = False
                    )
    
    #%% Target quantities and elicited statistics
    def target_quantities():
        return (
            target(name = "marginal_ReP",
                    elicitation_method = "quantiles",
                    quantiles_specs = (5, 25, 50, 75, 95),
                    loss_components = "by-group"
                    ),
            target(name = "marginal_EnC",
                    elicitation_method = "quantiles",
                    quantiles_specs = (5, 25, 50, 75, 95),
                    loss_components = "by-group"
                    ),
            target(name = "mean_effects",
                    elicitation_method = "quantiles",
                    quantiles_specs = (5, 25, 50, 75, 95),
                    loss_components = "by-group"
                    ),
            target(name = "R2",
                    elicitation_method = "histogram",
                    loss_components = "all"
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
    
    ##% global method function
    prior_elicitation(
        method = "deep_prior",
        sim_id = f"normal_deep_{seed}",
        B = 128,
        rep = 200,
        seed = seed,
        epochs = 700,
        output_path = "results",
        burnin = 1,
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

from validation.misc_plotting.plot_norm_res import plot_results_overview
path = "elicit/simulations/case_studies/sim_results/deep_prior/"
file = "norm_34800396"
title = "Normal model - deep prior"
plot_results_overview(path, file, title)
