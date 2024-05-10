import pendulum
import os, sys
import logging

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from airflow.decorators import dag, task, task_group

@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["workflow"]
)
def prior_elicitation(global_dict = None):
        @task_group()
        def load_prerequisits(global_dict):
            @task()
            def load_design_matrix(global_dict):
                design_matrix_path = None
                return design_matrix_path
            
            design_matrix_path = load_design_matrix(global_dict)
            return design_matrix_path

        @task_group()
        def priors(global_dict, ground_truth):
            @task()
            def intialize_priors(global_dict):
                init_priors = None
                return init_priors
            @task()
            def specify_ground_truth(global_dict):
                init_priors = None
                return init_priors
            
            if ground_truth:
                prior_model = specify_ground_truth(global_dict) 
            else:
                prior_model = intialize_priors(global_dict)
            return prior_model
            
        @task_group()
        def load_expert_data(design_matrix_path, global_dict):    
            @task_group(group_id= "one_forward_simulation_expert_id")
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
                
            prior_model = priors(global_dict, ground_truth=True)
            expert_elicited_statistics = one_forward_simulation(prior_model, design_matrix_path, global_dict, ground_truth=True)
            return expert_elicited_statistics

        @task()
        def initialize_learning_rate_schedule(global_dict):
            lr_schedule = None
            return lr_schedule
        
        @task_group()
        def training_loop(lr_schedule, expert_elicited_statistics, prior_model, global_dict):
        
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
                training_elicited_statistics = one_forward_simulation(prior_model, design_matrix_path, global_dict)
                weighted_total_loss = compute_loss(training_elicited_statistics, expert_elicited_statistics, global_dict, epoch)
                gradients = compute_gradients(weighted_total_loss, prior_model)
                updated_hyperparameters = update_hyperparameters(optimizer, gradients, prior_model)
                result_dictionary = save_results(training_elicited_statistics, gradients, updated_hyperparameters)

            return result_dictionary

        lr_schedule = initialize_learning_rate_schedule(global_dict)
        design_matrix_path = load_prerequisits(global_dict)
        expert_elicited_statistics = load_expert_data(design_matrix_path, global_dict)
        prior_model = priors(global_dict, ground_truth=False)
        result_dictionary = training_loop(lr_schedule, expert_elicited_statistics, prior_model, global_dict)
        
        return result_dictionary


dag = prior_elicitation()