import tensorflow_probability as tfp
import tensorflow as tf
import keras
tfd = tfp.distributions 

from functions.user_interface.input_functions import param, model, target, loss, expert, optimization, prior_elicitation, normalizing_flow_specs

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
        normalizing_flow_specs()
        )

#%% Expert input
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
                      "sigma": tfd.Gamma(8., 8.)
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
from user_input.custom_functions import custom_cor

def target_quantities():
    return (
        target(name = "marginal_ReP",
                elicitation_method = "quantiles",
                quantiles_specs = (10, 20, 30, 40, 50, 60, 70, 80, 90),
                loss_components = "by-group"
                ),
        target(name = "marginal_EnC",
                elicitation_method = "quantiles",
                quantiles_specs = (10, 20, 30, 40, 50, 60, 70, 80, 90),
                loss_components = "by-group"
                ),
        target(name = "mean_effects",
                elicitation_method = "quantiles",
                quantiles_specs = (10, 20, 30, 40, 50, 60, 70, 80, 90),
                loss_components = "by-group"
                ),
        target(name = "R2",
                elicitation_method = "histogram",
                loss_components = "all"
                ),
        target(name = "param_cor",
               elicitation_method = "histogram",
               custom_target_function = {
                   "function": custom_cor,
                   "additional_args": None
                   },
               loss_components = "all"
               )
        )

#%% Loss function
def loss_function():
    return loss(loss_function = "mmd-energy",
                loss_weighting = {"method": "dwa",
                                  "method_specs": {"temperature": 1.6}}
                )

#%% Training settings
def optimization_settings():
    return optimization(
                    optimizer = keras.optimizers.legacy.Adam,
                    optimizer_specs = {
                        "learning_rate": tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate = 0.0001, decay_steps = 10,
                            decay_rate = 0.7, staircase = True),
                        "clipnorm": 1.0
                        }
                    )

##% global method function
prior_elicitation(
    method = "deep_prior",
    sim_id = "norm_01",
    B = 128,
    rep = 200,
    seed = 124,
    epochs = 200,
    output_path = "results",
    burnin = 10,
    model_params = model_params,
    expert_input = expert_input,
    generative_model = generative_model,
    target_quantities = target_quantities,
    loss_function = loss_function,
    optimization_settings = optimization_settings,
    log_info = 0,
    view_ep = 1
    )

import pandas as pd
import matplotlib.pyplot as plt
from elicit.validation.diagnostic_plots import plot_loss, plot_convergence_deep, plot_marginal_priors, plot_joint_prior, plot_elicited_statistics
global_dict = pd.read_pickle("elicit/simulations/results/data/deep_prior/norm_01/global_dict.pkl")

plot_loss(global_dict, save_fig = True)
plot_marginal_priors(global_dict, sims = 100, save_fig = True)
plot_joint_prior(global_dict, save_fig = True)
plot_elicited_statistics(global_dict, sims = 100, save_fig = True)


### convergence plots
truth = pd.read_pickle(global_dict["output_path"]["data"]+"/expert/prior_samples.pkl")
true_selection = truth[:,:,:7]
# locations
true_hyperparams = tf.reduce_mean(true_selection, (0,1)) 
plot_convergence_deep(true_hyperparams, "means", global_dict, 
                 file_name = "convergence_loc", save_fig = True)
# scales
true_hyperparams = tf.reduce_mean(tf.math.reduce_std(true_selection, 1), 0) 
plot_convergence_deep(true_hyperparams, "stds", global_dict, 
                 file_name = "convergence_scale", save_fig = True)
# sigma parameter
gamma1 = tf.stack(pd.read_pickle(global_dict["output_path"]["data"]+"/final_results.pkl")["hyperparameter"]["means"], 0)[:,-1]
gamma2 = tf.stack(pd.read_pickle(global_dict["output_path"]["data"]+"/final_results.pkl")["hyperparameter"]["stds"], 0)[:,-1]
plt.axhline(tf.reduce_mean(truth[:,:,-1], (0,1)), linestyle = "dashed", color = "black")
plt.axhline(tf.math.reduce_std(truth[:,:,-1], 1), linestyle = "dashed", color = "black")
plt.plot(tf.range(len(gamma1)), gamma1, label = "gamma1")
plt.plot(tf.range(len(gamma2)), gamma2, label = "gamma2")
plt.legend()
plt.savefig(global_dict["output_path"]["plots"]+"/convergence_sigma.png")
