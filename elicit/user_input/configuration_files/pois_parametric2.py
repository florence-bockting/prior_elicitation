import tensorflow_probability as tfp
import tensorflow as tf
import keras
tfd = tfp.distributions 

from functions.user_interface.input_functions import param, model, target, loss, expert, optimization, prior_elicitation
from user_input.custom_functions import Normal_log
#%% Model parameters
normal_log = Normal_log()
def model_params():  
    return (
        param(name = "b0", 
              family = normal_log, 
              hyperparams_dict = {"mu0": tfd.Uniform(1.,2.5), 
                                  "log_sigma0": tfd.Uniform(-2.,-3.)}
              ),
        param(name = "b1", 
              family = normal_log, 
              hyperparams_dict = {"mu1": tfd.Uniform(0.,0.5), 
                                  "log_sigma1": tfd.Uniform(-2.,-3.)}
              ),
        param(name = "b2", 
              family = normal_log, 
              hyperparams_dict = {"mu2": tfd.Uniform(-1.,-1.5), 
                                  "log_sigma2": tfd.Uniform(-2.,-3.)}
              ),
        param(name = "b3", 
              family = normal_log, 
              hyperparams_dict = {"mu3": tfd.Uniform(-0.5,-1.), 
                                  "log_sigma3": tfd.Uniform(-2.,-3.)}
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

design_matrix = pd.read_pickle("elicit/simulations/data/design_matrix_pois.pkl")
#design_matrix = load_design_matrix_equality("standardize", [0, 13, 14, 35, 37, 48])

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
                    "additional_args": {"design_matrix": design_matrix,
                                        "factor_indices": [0,2,3]
                                        }
                    },
                loss_components = "by-group"
                )
        )

#%% Loss function
def loss_function():
    return loss(loss_function = "mmd-energy",
                loss_weighting = {"method": "dwa",
                                  "method_specs": {"temperature": 1.6}}
                )

#%% Training settings
initial_learning_rate = 0.01
final_learning_rate = 0.001
epochs = 200
train_size = 200
batch_size = 128
learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
steps_per_epoch = int(train_size/batch_size)

def optimization_settings():
    return optimization(
                    optimizer = keras.optimizers.legacy.Adam,
                    optimizer_specs = {
                        "learning_rate": tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate = initial_learning_rate, 
                            decay_steps = steps_per_epoch,
                            decay_rate = learning_rate_decay_factor, 
                            staircase = True),
                        "clipnorm": 1.0
                        }
                    )

##% global method function
prior_elicitation(
    method = "parametric_prior",
    sim_id = "pois_01",
    B = 128,
    rep = 200,
    seed = 127,
    burnin = 10,
    epochs = 300,
    output_path = "results",
    model_params = model_params,
    expert_input = expert_input,
    generative_model = generative_model,
    target_quantities = target_quantities,
    loss_function = loss_function,
    optimization_settings = optimization_settings,
    log_info = 0,
    print_info=True,
    view_ep = 1
    )

expert_priors = pd.read_pickle("elicit/simulations/results/parametric_prior/pois_01/expert/prior_samples.pkl")
hyperparams=pd.read_pickle("elicit/simulations/results/parametric_prior/pois_01/final_results.pkl")["hyperparameter"]

_, axs = plt.subplots(1,1)
[plt.plot(range(300),tf.exp(hyperparams[key]), lw = 3) for key in ["sigma0","sigma1","sigma2","sigma3"]]
[plt.plot(range(300),hyperparams[key], lw = 3) for key in ["mu0","mu1","mu2","mu3"]]
[plt.axhline(tf.reduce_mean(expert_priors, (0,1))[i], color = "black", linestyle = "dashed", lw = 1) for i in range(4)]
[plt.axhline(tf.reduce_mean(tf.math.reduce_std(expert_priors,1),0)[i], color = "black", linestyle = "dashed", lw = 1) for i in range(4)]
plt.ylim((-0.1,0.5))

import pandas as pd
from elicit.validation.diagnostic_plots import plot_loss, plot_gradients, plot_convergence, plot_marginal_priors, plot_joint_prior, plot_elicited_statistics

global_dict = pd.read_pickle("results/data/parametric_prior/pois_01/global_dict.pkl")

plot_loss(global_dict, save_fig = True)
plot_gradients(global_dict, save_fig = True)
plot_marginal_priors(global_dict, sims = 100, save_fig = True)
plot_joint_prior(global_dict, save_fig = True)
plot_elicited_statistics(global_dict, sims = 100,
                         selected_obs=[1, 11, 27, 33, 17, 15], save_fig = True)


truth = pd.read_pickle(global_dict["output_path"]["data"]+"/expert/prior_samples.pkl")
names_hyperparams = [f"mu{i}" for i in range(4)]
true_hyperparams = tf.reduce_mean(truth, (0,1))
plot_convergence(true_hyperparams, names_hyperparams, global_dict, 
                 file_name = "convergence_loc", save_fig = True)

names_hyperparams = [f"sigma{i}" for i in range(4)]
true_hyperparams = tf.reduce_mean(tf.math.reduce_std(truth,1),0)
plot_convergence(true_hyperparams, names_hyperparams, global_dict, 
                 file_name = "convergence_scale", save_fig = True)
