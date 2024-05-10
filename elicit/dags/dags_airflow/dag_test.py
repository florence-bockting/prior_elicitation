import pendulum
import os, sys
import logging

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from airflow.decorators import dag, task, task_group
from functions.helper_functions import save_as_pkl

@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["workflow"]
)
def prior_elicitation_test():
        from elicit.functions.utils import create_global_config
        global_dict = create_global_config('binom_deep.cfg')

        @task_group()
        def load_prerequisits():
            @task()
            def load_design_matrix_call():
                from elicit.configs.config_data import load_design_matrix
                design_matrix_path = load_design_matrix(global_dict) 
                return design_matrix_path
            
            design_matrix_path = load_design_matrix_call()
            return design_matrix_path
            
        @task_group()
        def load_expert_data(design_matrix_path):   
                        
            @task_group(group_id= "one_forward_simulation_expert_id")
            def one_forward_simulation(design_matrix_path, ground_truth):
                @task_group()
                def prior_sampling(ground_truth):       
                    @task()
                    def sample_from_priors(prior_model, ground_truth):
                        # sample from prior
                        prior_samples = prior_model()
                        # save obj to path
                        saving_path = global_dict["saving_path"]
                        if ground_truth:
                            saving_path = saving_path+"/expert"
                        path_prior_samples = saving_path+'/prior_samples.pkl'
                        save_as_pkl(prior_samples, path_prior_samples)
                        return path_prior_samples
                    
                    from elicit.functions.prior_simulation import priors
                    prior_model = priors(global_dict, ground_truth)
                    path_prior_samples = sample_from_priors(prior_model, ground_truth)

                    return path_prior_samples

                @task()
                def simulate_from_generator_call(path_prior_samples, design_matrix_path, ground_truth):
                    import pandas as pd
                    from elicit.functions.model_simulation import simulate_from_generator
                    prior_samples = pd.read_pickle(path_prior_samples)
                    model_simulations = simulate_from_generator(prior_samples, design_matrix_path, ground_truth, global_dict)
                    return model_simulations
                @task()
                def computation_target_quantities(model_simulations, ground_truth):
                    target_quantities = None
                    return target_quantities
                @task()
                def computation_elicited_statistics(target_quantities, ground_truth):
                    elicited_statistics = None
                    return elicited_statistics
                
                path_prior_samples = prior_sampling(ground_truth=True)
                path_model_simulations = simulate_from_generator_call(path_prior_samples, design_matrix_path, ground_truth)
                target_quantities = computation_target_quantities(path_model_simulations, ground_truth)
                elicited_statistics = computation_elicited_statistics(target_quantities, ground_truth)

                return elicited_statistics
            
            expert_elicited_statistics = one_forward_simulation(design_matrix_path, ground_truth=True)
            return expert_elicited_statistics

        @task()
        def initialize_learning_rate_schedule():
            lr_schedule = None
            return lr_schedule
        
        @task_group()
        def training_loop(lr_schedule, expert_elicited_statistics, global_dict):
            @task()
            def intialize_priors(global_dict):
                init_priors = None
                return init_priors
                
            for epoch in range(1):  
            
                @task_group(group_id=f"one_forward_simulation_epoch_{epoch}")
                def one_forward_simulation(prior_model, design_matrix_path, global_dict, ground_truth=False):
                    @task()
                    def sample_from_priors(prior_model):
                        prior_samples = None
                        return prior_samples
                    @task()
                    def simulate_from_generator(prior_samples, design_matrix_path, ground_truth, global_dict):
                        model_simulations = None
                        return model_simulations
                    @task()
                    def computation_target_quantities(model_simulations, ground_truth, global_dict):
                        target_quantities = None
                        return target_quantities
                    @task()
                    def computation_elicited_statistics(target_quantities, ground_truth, global_dict):
                        elicited_statistics = None
                        return elicited_statistics
                    
                    prior_samples = sample_from_priors(prior_model)
                    model_simulations = simulate_from_generator(prior_samples, design_matrix_path, ground_truth, global_dict)
                    target_quantities = computation_target_quantities(model_simulations, ground_truth, global_dict)
                    elicited_statistics = computation_elicited_statistics(target_quantities, ground_truth, global_dict)

                    return elicited_statistics
        
                @task_group(group_id=f"compute_loss_epoch_{epoch}")
                def compute_loss(training_elicited_statistics, expert_elicited_statistics, global_dict, epoch):
                    @task()
                    def compute_loss_components(elicited_statistics, global_dict, expert):
                        loss_components = None
                        return loss_components
                    @task()
                    def compute_discrepancy(loss_components_expert, loss_components_training):
                        loss_per_component = None
                        return loss_per_component
                    @task_group()
                    def compute_total_loss(epoch, loss_per_component, global_dict):
                        @task()
                        def dynamic_weight_averaging(epoch, loss_per_component, global_dict):
                            weighted_total_loss = None
                            return weighted_total_loss
                        
                        weighted_total_loss = dynamic_weight_averaging(
                            epoch, loss_per_component, global_dict)
                        return weighted_total_loss

                    loss_components_expert = compute_loss_components(expert_elicited_statistics, global_dict, expert=True)
                    loss_components_training = compute_loss_components(training_elicited_statistics, global_dict, expert=False)
                    loss_per_component = compute_discrepancy(loss_components_expert, loss_components_training)
                    weighted_total_loss = compute_total_loss(epoch, loss_per_component, global_dict)

                    return weighted_total_loss
                
                @task(task_id = f"initialize_optimizer_epoch_{epoch}")
                def initialize_optimizer(lr_schedule):
                    optimizer = None
                    return optimizer
                @task(task_id = f"compute_gradients_epoch_{epoch}")
                def compute_gradients(total_loss, prior_model):
                    gradients = None
                    return gradients 
                @task(task_id = f"update_hyperparameters_epoch_{epoch}")
                def update_hyperparameters(optimizer, gradients, prior_model):
                    updated_hyperparameters = None
                    return updated_hyperparameters
                @task(task_id = f"save_results_epoch_{epoch}")
                def save_results(training_elicited_statistics, gradients, updated_hyperparameters):
                    result_dictionary = None
                    return result_dictionary
    
                optimizer = initialize_optimizer(lr_schedule)
                prior_model = intialize_priors(global_dict)
                training_elicited_statistics = one_forward_simulation(prior_model, design_matrix_path, global_dict)
                weighted_total_loss = compute_loss(training_elicited_statistics, expert_elicited_statistics, global_dict, epoch)
                gradients = compute_gradients(weighted_total_loss, prior_model)
                updated_hyperparameters = update_hyperparameters(optimizer, gradients, prior_model)
                result_dictionary = save_results(training_elicited_statistics, gradients, updated_hyperparameters)

            return result_dictionary

        lr_schedule = initialize_learning_rate_schedule()
        design_matrix_path = load_prerequisits()
        expert_elicited_statistics = load_expert_data(design_matrix_path)
        result_dictionary = training_loop(lr_schedule, expert_elicited_statistics, global_dict)
        
        return result_dictionary


dag = prior_elicitation_test()
