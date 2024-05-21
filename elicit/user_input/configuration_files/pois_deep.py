import tensorflow_probability as tfp
import tensorflow as tf
import keras
import pandas as pd

tfd = tfp.distributions 

from functions.user_interface.input_functions import param, model, target, loss, expert, normalizing_flow_specs, optimization, prior_elicitation

#%% Model parameters
def model_params():  
    return (
        param(name = "b0", scaling_value = 1.),
        param(name = "b1", scaling_value = 1.),
        param(name = "b2", scaling_value = 1.),
        param(name = "b3", scaling_value = 1.),
        normalizing_flow_specs()
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
#from user_input.design_matrices import load_design_matrix_equality
#design_matrix = load_design_matrix_equality("standardize", [0, 13, 14, 35, 37, 48])

# or you use the saved version from path
design_matrix = pd.read_pickle("elicit/simulations/data/design_matrix_pois.pkl")

def generative_model():
    return model(GenerativePoissonModel,
                 additional_model_args = {"total_count": 80, 
                                          "design_matrix": design_matrix},
                 discrete_likelihood = True,
                 softmax_gumble_specs = {"temperature": 1.,
                                         "upper_threshold": 80}
                )

#%% Target quantities and elicited statistics
from user_input.custom_functions import custom_group_means, custom_cor

def target_quantities():
    return (
        target(name = "ypred",
               elicitation_method = "histogram",
               loss_components = "by-group"
               ),
        target(name = "group_means",
               elicitation_method = "quantiles",
               quantiles_specs = (5, 25, 50, 75,95),
               custom_target_function = {
                   "function": custom_group_means,
                   "additional_args": {"design_matrix": design_matrix,
                                       "factor_indices": [0,2,3]
                                       }
                   },
               loss_components = "by-group"
               ),
        target(name = "param_cor",
               elicitation_method = "histogram",
               loss_components = "all",
               custom_target_function = {
                   "function": custom_cor,
                   "additional_args": None
                   }
               )
        )

#%% Loss function
def loss_function():
    return loss(loss_function = "mmd-energy",
                loss_weighting = {"method": "dwa",
                                  "method_specs": {"temperature": 1.6}}
                )

#%% Training settings
initial_learning_rate = 0.0001
final_learning_rate = 0.000001
epochs = 100
train_size = 300
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

#%% global method function
prior_elicitation(
    method = "deep_prior",
    sim_id = "pois_01",
    B = 128,
    rep = 300,
    seed = 345,
    burnin = 10,
    epochs = 150,
    output_path = "results",
    model_params = model_params,
    expert_input = expert_input,
    generative_model = generative_model,
    target_quantities = target_quantities,
    loss_function = loss_function,
    optimization_settings = optimization_settings,
    log_info = 0,
    view_ep = 1,
    print_info = True
    )

########### postprocessing
import pandas as pd
from elicit.validation.diagnostic_plots import plot_loss, plot_convergence_deep, plot_marginal_priors, plot_joint_prior, plot_elicited_statistics

global_dict = pd.read_pickle("elicit/simulations/results/data/deep_prior/pois_01/global_dict.pkl")

plot_loss(global_dict, save_fig = True)
plot_marginal_priors(global_dict, sims = 100, save_fig = True)
plot_joint_prior(global_dict, save_fig = True)
plot_elicited_statistics(global_dict, sims = 100, 
                         selected_obs=[0, 13, 14, 35, 37, 48], save_fig = True)

# convergence
truth = pd.read_pickle(global_dict["output_path"]["data"]+"/expert/prior_samples.pkl")
true_hyperparams = tf.reduce_mean(tf.math.reduce_std(truth,1),0)
plot_convergence_deep(true_hyperparams, "stds", global_dict, 
                      file_name = "convergence_scale",
                      save_fig = True)

true_hyperparams = tf.reduce_mean(truth,(0,1))
plot_convergence_deep(true_hyperparams, "means", global_dict, 
                      file_name = "convergence_loc",
                      save_fig = True)

