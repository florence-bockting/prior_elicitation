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
            param(name = "b0", scaling_value=100.),
            param(name = "b1"),
            param(name = "tau0"),
            param(name = "tau1"),
            param(name = "sigma"),
            normalizing_flow_specs(
                num_coupling_layers = 3,
                coupling_design = "affine", 
                coupling_settings = {
                    "dropout": False,
                    "dense_args": {
                        "units": 128,
                        "activation": "softplus",
                        "kernel_regularizer": tf.keras.regularizers.l2(1e-4)
                        },
                    "num_dense": 2
                    },
                permutation="learnable"
                )
            )
    
    #%% Expert input
    # from user_input.custom_functions import Gamma_inv_softplus
    # gamma_inv_softplus = Gamma_inv_softplus()
    
    def expert_input():
        return expert(data = None,
                      simulate_data = True,
                      simulator_specs = {
                          "b0": tfd.Normal(250.4, 7.27),
                          "b1": tfd.Normal(30.26, 4.82),
                          "tau0": tfd.TruncatedNormal(0., 33., low=0., high=500),
                          "tau1": tfd.TruncatedNormal(0., 23., low=0., high=500),
                          "sigma": tfd.Gamma(200., 8.)
                          })

    #%% Generative model
    from user_input.generative_models import GenerativeMultilevelModel
    from user_input.design_matrices import load_design_matrix_sleep
    
    design_matrix = load_design_matrix_sleep("divide_by_std", N_days = 10, 
                                             N_subj = 200, 
                                             selected_days = [0,2,5,6,9])
    
    def generative_model():
        return model(GenerativeMultilevelModel,
                     additional_model_args = {
                         "design_matrix": design_matrix,
                         "selected_days": [0,2,5,6,9],
                         "alpha_lkj": 1.,
                         "N_subj": 200,
                         "N_days": 5
                         },
                     discrete_likelihood = False
                    )
    
    #%% Target quantities and elicited statistics
    from user_input.custom_functions import custom_mu0_sd, custom_mu9_sd
    from setup.create_dictionaries import create_dict
    
    @create_dict
    def target_quantities1(method = "ground_truth"):
        return (
            target(name = "sigma",
                    elicitation_method = "moments",
                    moments_specs=("mean","sd"),
                    loss_components = "all"
                    ),
            target(name = "mu0sdcomp",
                    elicitation_method = "histogram",
                    loss_components = "all"
                    ),
            target(name = "mu9sdcomp",
                    elicitation_method = "histogram",
                    loss_components = "all"
                    ),
            target(name = "meanperday",
                    elicitation_method = "quantiles",
                    quantiles_specs = (25, 50, 75),
                    loss_components = "by-group"
                    )
            )
    
    @create_dict
    def target_quantities2(method = "learning"):
        return (
            target(name = "sigma",
                    elicitation_method = "moments",
                    moments_specs=("mean","sd"),
                    loss_components = "all"
                    ),
            target(name = "mu0sdcomp",
                    elicitation_method = "histogram",
                    loss_components = "all",
                    custom_target_function={
                                        "function": custom_mu0_sd,
                                        "additional_args": {"selected_days": [0,2,5,6,9]}
                                        }
                    ),
            target(name = "mu9sdcomp",
                    elicitation_method = "histogram",
                    loss_components = "all",
                    custom_target_function={
                                        "function": custom_mu9_sd,
                                        "additional_args": {"selected_days": [0,2,5,6,9]}
                                        }
                    ),
            target(name = "meanperday",
                    elicitation_method = "quantiles",
                    quantiles_specs = (25, 50, 75),
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
                                0.001, 100),
                            "clipnorm": 1.0
                            }
                        )
    
    ##% global method function
    prior_elicitation(
        method = "deep_prior",
        sim_id = f"multilevel_deep_{seed}",
        B = 128,
        rep = 200,
        seed = seed,
        epochs = 1000,
        output_path = "results",
        burnin = 1,
        model_params = model_params,
        expert_input = expert_input,
        generative_model = generative_model,
        target_quantities = (target_quantities1, target_quantities2),
        loss_function = loss_function,
        optimization_settings = optimization_settings,
        view_ep = 1,
        print_info = True
        )

if __name__ == "__main__":
    seed = int(sys.argv[1])
    
    run_simulation(seed)

# # run_simulation(123)

# from validation.misc_plotting.plot_mlm_res import plot_results_overview
# path = "elicit/simulations/case_studies/sim_results/deep_prior/"
# file = "mlm_norm_34803858"

# title = "Multilevel model - deep prior"
# plot_results_overview(path, file, title)
