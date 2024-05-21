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
                                  "sigma0": tfd.Uniform(0.,0.5)}
              ),
        param(name = "b1", 
              family = tfd.Normal, 
              hyperparams_dict = {"mu1": tfd.Uniform(0.,1.), 
                                  "sigma1": tfd.Uniform(0.,0.5)}
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
from user_input.design_matrices import load_design_matrix_haberman

design_matrix = load_design_matrix_haberman("standardize", 
                                            [0, 5, 10, 15, 20, 25, 30])
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
               quantiles_specs = (10, 20, 30, 40, 50, 60, 70, 80, 90),
               loss_components = "by-group"
               ),
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
                            initial_learning_rate = 0.01, decay_steps = 5,
                            decay_rate = 0.7, staircase = True),
                        "clipnorm": 1.0
                        }
                    )

##% global method function
prior_elicitation(
    method = "parametric_prior",
    sim_id = "binom_01",
    B = 256,
    rep = 300,
    seed = 123,
    burnin = 10,
    epochs = 100,
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

global_dict = pd.read_pickle("results/data/parametric_prior/binom_01/global_dict.pkl")

plot_loss(global_dict, save_fig = True)
plot_gradients(global_dict, save_fig = True)
plot_marginal_priors(global_dict, sims = 100, save_fig = True)
plot_joint_prior(global_dict, save_fig = True)
plot_elicited_statistics(global_dict, sims = 100, 
                         selected_obs = [0, 5, 10, 15, 20, 25, 30], 
                         save_fig = True)

true_samples = pd.read_pickle("results/data/parametric_prior/binom_01/expert/prior_samples.pkl")
truth = tf.reduce_mean(true_samples, (0,1))
plot_convergence(true_hyperparams = truth, names_hyperparams=["mu0","mu1"], 
                 global_dict=global_dict, file_name="convergence_loc", save_fig = True)

true_samples = pd.read_pickle("results/data/parametric_prior/binom_01/expert/prior_samples.pkl")
truth = tf.reduce_mean(tf.math.reduce_std(true_samples, 1),0)
plot_convergence(true_hyperparams = truth, names_hyperparams=["sigma0","sigma1"], 
                 global_dict=global_dict, file_name="convergence_scale", 
                 pos_only = True, save_fig = True)
