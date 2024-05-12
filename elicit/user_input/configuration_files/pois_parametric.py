import tensorflow_probability as tfp
import tensorflow as tf
import keras
tfd = tfp.distributions 

from functions.user_interface.input_functions import param, model, target, loss, expert, optimization, prior_elicitation

#%% Model parameters
def model_params():  
    return (
        param(name = "b0", 
              family = tfd.Normal, 
              hyperparams_dict = {"mu0": tfd.Uniform(0.,1.), 
                                  "sigma0": tfd.Uniform(0.,0.2)}
              ),
        param(name = "b1", 
              family = tfd.Normal, 
              hyperparams_dict = {"mu1": tfd.Uniform(0.,1.), 
                                  "sigma1": tfd.Uniform(0.,0.2)}
              ),
        param(name = "b2", 
              family = tfd.Normal, 
              hyperparams_dict = {"mu2": tfd.Uniform(0.,1.), 
                                  "sigma2": tfd.Uniform(0.,0.2)}
              ),
        param(name = "b3", 
              family = tfd.Normal, 
              hyperparams_dict = {"mu3": tfd.Uniform(0.,1.), 
                                  "sigma3": tfd.Uniform(0.,0.2)}
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
from user_input.design_matrices import load_design_matrix_equality
design_matrix = load_design_matrix_equality("standardize", [1, 11, 27, 33, 17, 15])

def generative_model():
    return model(GenerativePoissonModel,
                 additional_model_args = {
                     "total_count": 80, 
                     "design_matrix": design_matrix},
                 discrete_likelihood = True,
                 softmax_gumble_specs = {"temperature": 1.,
                                         "upper_threshold": 80}
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
               quantiles_specs = (10, 20, 30, 40, 50, 60, 70, 80, 90),
               custom_target_function = {
                   "function": custom_group_means,
                   "additional_args": {"name": "design_matrix", 
                                       "value": design_matrix}
                   },
               loss_components = "by-stats"
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
                            initial_learning_rate = 0.001, decay_steps = 7,
                            decay_rate = 0.9, staircase = True),
                        "clipnorm": 1.0
                        }
                    )

##% global method function
prior_elicitation(
    method = "parametric_prior",
    sim_id = "pois_01",
    B = 128,
    rep = 200,
    seed = 123,
    epochs = 200,
    output_path = "results",
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
from elicit.validation.diagnostic_plots import plot_loss, plot_gradients, plot_convergence, plot_marginal_priors, plot_joint_prior, plot_elicited_statistics

global_dict = pd.read_pickle("results/data/parametric_prior/pois_01/global_dict.pkl")

plot_loss(global_dict, save_fig = True)
plot_gradients(global_dict, save_fig = True)
plot_convergence(global_dict, save_fig = True)
plot_marginal_priors(global_dict, sims = 100, save_fig = True)
plot_joint_prior(global_dict, save_fig = True)
plot_elicited_statistics(global_dict, sims = 100,
                         selected_obs=[1, 11, 27, 33, 17, 15], save_fig = True)





