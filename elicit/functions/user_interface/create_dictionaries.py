import os
from functions.helper_functions import save_as_pkl

def create_dict(user_input: callable) -> dict:
    param_tuple = user_input()
    # loop over each element in tuple
    for p in range(len(param_tuple)):
        input_dict = param_tuple[p]
        # check whether element specifies a parameter 
        # (if not it specifies the normalizing flow hyperparameters)
        if "name" in list(input_dict.keys()):
            # first parameter inializes dictionary
            if p == 0:
                param_dict = input_dict
            # remaining parameters update init. dict
            else:
                for key in list(param_dict.keys()):
                    param_dict[key] = param_dict[key] + input_dict[key]
        else:
            # add normalizing flow specs if specified
            param_dict["normalizing_flow_specs"] = input_dict
    return param_dict


# Create necessary config files
def create_global_dict(
        method, sim_id, epochs, B, rep, seed, burnin, model_params, expert_input, 
        generative_model, target_quantities, loss_function, 
        optimization_settings, output_path, log_info, print_info, view_ep) -> dict:
    
    global_dict = dict()
    
    global_dict["method"] = method
    global_dict["sim_id"] = sim_id
    global_dict["epochs"] = epochs
    global_dict["B"] = B
    global_dict["rep"] = rep
    global_dict["seed"] = seed
    global_dict["burnin"] = burnin
    global_dict["model_params"] = create_dict(model_params)
    global_dict["expert_input"] = expert_input()
    global_dict["generative_model"] = generative_model()
    global_dict["target_quantities"] = create_dict(target_quantities)
    global_dict["loss_function"] = loss_function()
    global_dict["optimization_settings"] = optimization_settings()
    global_dict["output_path"] = {
        "data": f"elicit/simulations/results/{method}/{sim_id}",
       # "data": os.path.join(os.path.dirname(__name__), output_path, "data", method, sim_id),
        "plots": os.path.join(os.path.dirname(__name__), output_path, "plots", method, sim_id)
        }
    global_dict["log_info"] = log_info
    global_dict["print_info"] = print_info
    global_dict["view_ep"] = view_ep
    
    # save global dict
    path = global_dict["output_path"]["data"] +'/global_dict.pkl'
    save_as_pkl(global_dict, path)
    
    return global_dict