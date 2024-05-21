import tensorflow as tf

from functions.prior_simulation import intialize_priors, sample_from_priors
from functions.helper_functions import LogsInfo
from functions.model_simulation import simulate_from_generator
from functions.targets_elicits_computation import computation_target_quantities, computation_elicited_statistics
from functions.loss_computation import compute_loss_components, compute_discrepancy, dynamic_weight_averaging
from functions.training import training_loop


def prior_elicitation_dag(global_dict: dict):
    
    # initialize feedback behavior
    logs = LogsInfo(global_dict["log_info"]) 
    
    # set seed
    tf.random.set_seed(global_dict["seed"])

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
    
    def one_forward_simulation(prior_model, global_dict, ground_truth=False):
        prior_samples = prior_model()
        model_simulations = simulate_from_generator(prior_samples, ground_truth, global_dict)
        target_quantities = computation_target_quantities(model_simulations, ground_truth, global_dict)
        elicited_statistics = computation_elicited_statistics(target_quantities, ground_truth, global_dict)
        return elicited_statistics

    def load_expert_data(global_dict, path_to_expert_data=None):
        if global_dict["expert_input"]["simulate_data"]:
            prior_model = priors(global_dict, ground_truth=True)
            expert_data = one_forward_simulation(
                prior_model, global_dict, ground_truth=True)
        else:
            # TODO: Checking: Must have the same shape/form as elicited statistics from model
            assert global_dict["expert_input"]["data"] is not None, "path to expert data has to provided"
            # load expert data from file
            expert_data = global_dict["expert_input"]["data"]
        return expert_data

    def compute_loss(training_elicited_statistics, expert_elicited_statistics, global_dict, epoch):
        
        def compute_total_loss(epoch, loss_per_component, global_dict):
            logs("## compute total loss using dynamic weight averaging", 2)
            loss_per_component_current = loss_per_component
            # check whether selected method is implemented
            assert global_dict["loss_function"]["loss_weighting"]["method"] == "dwa", "Currently implemented loss weighting methods are: 'dwa'"
            # apply selected loss weighting method
            if global_dict["loss_function"]["loss_weighting"]["method"] == "dwa":
                # dwa needs information about the initial loss per component
                if epoch == 0:
                    global_dict["loss_function"]["loss_weighting"]["method_specs"]["loss_per_component_initial"] = loss_per_component
                # apply method
                weighted_total_loss = dynamic_weight_averaging(
                    epoch, loss_per_component_current, 
                    global_dict["loss_function"]["loss_weighting"]["method_specs"]["loss_per_component_initial"],
                    global_dict["loss_function"]["loss_weighting"]["method_specs"]["temperature"],
                    global_dict["output_path"]["data"])
                
            return weighted_total_loss

        loss_components_expert = compute_loss_components(expert_elicited_statistics, global_dict, expert=True)
        loss_components_training = compute_loss_components(training_elicited_statistics, global_dict, expert=False)
        loss_per_component = compute_discrepancy(loss_components_expert, loss_components_training, global_dict)
        weighted_total_loss = compute_total_loss(epoch, loss_per_component, global_dict)

        return weighted_total_loss
        
    def burnin_phase(expert_elicited_statistics, priors,
                      one_forward_simulation, compute_loss, global_dict):
        
        loss_list = []
        init_var_list = []
        if global_dict["print_info"]:
            print("burnin phase")
        for i in range(global_dict["burnin"]):
            if global_dict["print_info"]:
                print("|", end='')
            # prepare generative model
            prior_model = priors(global_dict)
            # generate simulations from model
            training_elicited_statistics = one_forward_simulation(prior_model, global_dict)
            # comput loss
            weighted_total_loss = compute_loss(training_elicited_statistics, 
                                               expert_elicited_statistics, 
                                               global_dict, epoch = 0)
            
            init_var_list.append(prior_model)
            loss_list.append(weighted_total_loss.numpy())
        if global_dict["print_info"]:
            print(" ")
        return loss_list, init_var_list
    
    # get expert data
    expert_elicited_statistics = load_expert_data(global_dict)
    
    loss_list, init_prior =  burnin_phase(
        expert_elicited_statistics, priors, one_forward_simulation, compute_loss, 
        global_dict)
    
    min_index = tf.argmin(loss_list)
    init_prior_model = init_prior[min_index]
    
    training_loop(expert_elicited_statistics, init_prior_model, 
                  one_forward_simulation, 
                  compute_loss, global_dict)
    