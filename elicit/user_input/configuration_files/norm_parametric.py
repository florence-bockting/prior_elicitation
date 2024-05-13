import tensorflow_probability as tfp
import tensorflow as tf
import keras
tfd = tfp.distributions 

from functions.user_interface.input_functions import param, model, target, loss, expert, optimization, prior_elicitation

#%% Model parameters
from user_input.custom_functions import TruncNormal
# initialize truncated normal with boundaries
truncnorm = TruncNormal(low=0., high=1.)   

def model_params():  
    return (
        param(name = "b0", 
              family = tfd.Normal, 
              hyperparams_dict = {"mu0": tfd.Normal(0.,0.1), 
                                  "sigma0": tfd.Uniform(0.,0.2)}
              ),
        param(name = "b1", 
              family = tfd.Normal, 
              hyperparams_dict = {"mu1": tfd.Normal(0.,0.1), 
                                  "sigma1": tfd.Uniform(0.,0.2)}
              ),
        param(name = "b2", 
              family = tfd.Normal, 
              hyperparams_dict = {"mu2": tfd.Normal(0.,0.1), 
                                  "sigma2": tfd.Uniform(0.,0.2)}
              ),
        param(name = "b3", 
              family = tfd.Normal, 
              hyperparams_dict = {"mu3": tfd.Normal(0.,0.1), 
                                  "sigma3": tfd.Uniform(0.,0.2)}
              ),
        param(name = "b4", 
              family = tfd.Normal, 
              hyperparams_dict = {"mu4": tfd.Normal(0.,0.1), 
                                  "sigma4": tfd.Uniform(0.,0.2)}
              ),
        param(name = "b5", 
              family = tfd.Normal, 
              hyperparams_dict = {"mu5": tfd.Normal(0.,0.1), 
                                  "sigma5": tfd.Uniform(0.,0.2)}
              ),
        param(name = "sigma", 
              family = tfd.Gamma, 
              hyperparams_dict = {"loc": tfd.Uniform(290.,310.), 
                                  "scale": tfd.Uniform(2600,2750)}
              )
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
                      "sigma": tfd.Gamma(300., 300.*9.)
                    #  "sigma": tfd.TruncatedNormal(0.111,0.006, 0., 1.)
                      })
# sns.kdeplot(tfd.Gamma(300., 300.*9.).sample(1000))
# sns.kdeplot(tfd.TruncatedNormal(0.111,0.006, 0., 1.).sample(1000))
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
                )
        )

#%% Loss function
def loss_function():
    return loss(loss_function = "mmd-energy",
                loss_weighting = {"method": "dwa",
                                  "method_specs": {"temperature": 1.6}}
                )

#%% Training settings

# tf.keras.optimizers.schedules.CosineDecay(
# 0.01, 200),

def optimization_settings():
    return optimization(
                    optimizer = keras.optimizers.legacy.Adam,
                    optimizer_specs = {
                        "learning_rate": tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate = 0.001, decay_steps = 10,
                            decay_rate = 0.85, staircase = True),
                        "clipnorm": 1.0
                        }
                    )

##% global method function
prior_elicitation(
    method = "parametric_prior",
    sim_id = "norm_01",
    B = 128,
    rep = 300,
    seed = 125,
    epochs = 300,
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

global_dict = pd.read_pickle("results/data/parametric_prior/norm_01/global_dict.pkl")

plot_loss(global_dict, save_fig = True)
plot_gradients(global_dict, save_fig = True)
plot_convergence(global_dict, save_fig = True)
plot_marginal_priors(global_dict, sims = 100, save_fig = True)
plot_joint_prior(global_dict, save_fig = True)
plot_elicited_statistics(global_dict, sims = 100, save_fig = True)

tf.reduce_mean(tf.math.reduce_std(pd.read_pickle("results/data/parametric_prior/norm_01/prior_samples.pkl"),1),0)
tf.reduce_mean(pd.read_pickle("results/data/parametric_prior/norm_01/prior_samples.pkl"), (0,1))

tf.reduce_mean(tf.math.reduce_std(pd.read_pickle("results/data/parametric_prior/norm_01/expert/prior_samples.pkl"),1),0)
tf.reduce_mean(pd.read_pickle("results/data/parametric_prior/norm_01/expert/prior_samples.pkl"), (0,1))
