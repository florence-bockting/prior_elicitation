import configparser
import tensorflow as tf
import pandas as pd
import os

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__name__),'elicit', 'configs', 'config_global.cfg'))

from elicit.functions.prior_simulation import intialize_priors, sample_from_priors
from elicit.functions.helper_functions import LogsInfo
from elicit.functions.model_simulation import simulate_from_generator
from elicit.functions.targets_elicits_computation import computation_target_quantities, computation_elicited_statistics
from elicit.functions.loss_computation import compute_loss_components, compute_discrepancy, dynamic_weight_averaging
from elicit.functions.training import training_loop
from elicit.functions.utils import create_global_config

# create dictionary with global variables 
# file_name is name of custom config file
global_dict = create_global_config('pois_parametric.cfg')

# set seed
tf.random.set_seed(global_dict["seed"])

# initialize feedback behavior
logs = LogsInfo(global_dict["log_info"]) 

def prior_elicitation(global_dict=global_dict):
    def load_prerequisits(global_dict):
        logs("## load design matrix", 2)
        import configs.config_data as ccd
        dmatrix_func = getattr(ccd, global_dict["dmatrix_function_call"])
        design_matrix_path = dmatrix_func(global_dict) 
        return design_matrix_path

    def priors(global_dict, ground_truth=False):
            # initalize generator model
            class Priors(tf.Module):
                def __init__(self, ground_truth, global_dict):
                    self.global_dict = global_dict
                    self.ground_truth = ground_truth
                    if not self.ground_truth:
                        self.init_priors = intialize_priors(self.global_dict)
                    else:
                        self.init_priors = None

                def __call__(self):
                    prior_samples = sample_from_priors(self.init_priors, self.ground_truth, self.global_dict)
                    return prior_samples

            prior_model = Priors(ground_truth, global_dict)
            return prior_model
    
    def one_forward_simulation(prior_model, design_matrix_path, global_dict, ground_truth=False):
        prior_samples = prior_model()
        model_simulations = simulate_from_generator(prior_samples, design_matrix_path, ground_truth, global_dict)
        target_quantities = computation_target_quantities(model_simulations, ground_truth, global_dict)
        elicited_statistics = computation_elicited_statistics(target_quantities, ground_truth, global_dict)
        return elicited_statistics

    def load_expert_data(global_dict, path_to_expert_data=None):
        if global_dict["validation"]:
            design_matrix_path = load_prerequisits(global_dict)
            prior_model = priors(global_dict, ground_truth=True)
            expert_data = one_forward_simulation(prior_model, design_matrix_path, 
                                                 global_dict, ground_truth=True)
        else:
            # TODO: Must have the same shape/form as elicited statistics from model
            assert path_to_expert_data is not None, "path to expert data has to provided"
            # load expert data from file
            expert_data = pd.read_pickle(rf"{path_to_expert_data}")
        return expert_data

    def compute_loss(training_elicited_statistics, expert_elicited_statistics, global_dict, epoch):
        
        def compute_total_loss(epoch, loss_per_component, global_dict):
            if epoch == 0:
                global_dict["loss_per_component_initial"] = loss_per_component
            logs("## compute total loss using dynamic weight averaging", 2)
            loss_per_component_current = loss_per_component
            weighted_total_loss = dynamic_weight_averaging(epoch, loss_per_component_current, 
                                                           global_dict["loss_per_component_initial"],
                                                           global_dict["task_balance_factor"],
                                                           global_dict["saving_path"])
            return weighted_total_loss

        loss_components_expert = compute_loss_components(expert_elicited_statistics, global_dict, expert=True)
        loss_components_training = compute_loss_components(training_elicited_statistics, global_dict, expert=False)
        loss_per_component = compute_discrepancy(loss_components_expert, loss_components_training, global_dict)
        weighted_total_loss = compute_total_loss(epoch, loss_per_component, global_dict)

        return weighted_total_loss

    training_loop(load_prerequisits, load_expert_data, priors,
                  one_forward_simulation, compute_loss, global_dict)
    
    
########### run pipeline
prior_elicitation()


########### postprocessing

from validation.diagnostic_plots import plot_loss, plot_gradients, plot_convergence, plot_marginal_priors, plot_joint_prior, plot_elicited_statistics, plot_elicited_statistics_pois

plot_loss(global_dict, save_fig = True)
# not for deep-prior method:
plot_gradients(global_dict, save_fig = True)
plot_convergence(global_dict, save_fig = True)
plot_marginal_priors(global_dict, save_fig = True)
plot_joint_prior(global_dict, save_fig = True)
plot_elicited_statistics(global_dict, save_fig = True)
plot_elicited_statistics_pois(global_dict, sims = 100, save_fig = True)
