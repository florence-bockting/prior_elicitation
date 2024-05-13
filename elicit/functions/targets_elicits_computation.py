import tensorflow as tf
import tensorflow_probability as tfp
import bayesflow as bf
import inspect

tfd = tfp.distributions
bfn = bf.networks

from functions.helper_functions import save_as_pkl, LogsInfo

def use_custom_functions(custom_function, model_simulations):    
    # get function
    custom_func = custom_function["function"]
    # create a dict with arguments from model simulations and custom args for custom func 
    args_dict = model_simulations
    if custom_function["additional_args"] is not None:
        args_dict.update({f"{key}": custom_function["additional_args"][key] for key in list(custom_function["additional_args"].keys())})
    # select only relevant keys from args_dict
    custom_args_keys = inspect.getfullargspec(custom_func)[0]
    # check that all args needed for custom function were detected
    assert set(custom_args_keys).issubset(set(args_dict.keys())), f"Custom target function takes arguments which were not specified. Required args: {custom_args_keys} but got {list(args_dict.keys())}"
    args_dict = {f"{key}": args_dict[key] for key in custom_args_keys}
    # evaluate custom function
    custom_quantity = custom_func(**args_dict)
    return custom_quantity

def computation_target_quantities(model_simulations, ground_truth, global_dict):
    # initialize feedback behavior
    logs = LogsInfo(global_dict["log_info"])
    logs("...compute target quantities", 3)
    # names of target quantities
    name_targets = global_dict["target_quantities"]["name"]
    # check for duplicate naming 
    assert len(name_targets) == len(set(name_targets)), "duplicate target quantity name has been detected; target quantities must have unique names."
    # initialize dict for storing results
    targets_res = dict()
    # loop over target quantities
    for i, target in enumerate(name_targets):
        # use custom function for target quantity if it has been defined 
        if global_dict["target_quantities"]["custom_target_function"][i] is not None:
            target_quantity = use_custom_functions(
                global_dict["target_quantities"]["custom_target_function"][i], 
                model_simulations)
        else:
            target_quantity = model_simulations[target]
            
        # select indicated observations from design matrix
        # TODO: I suppose that makes only sense for ypred as target quantity; there should be a warning?, error?
        if global_dict["target_quantities"]["select_obs"][i] is not None:
            target_quantity = tf.gather(target_quantity, 
                                        list(global_dict["target_quantities"]["select_obs"][i]),
                                        axis = -1)
        # save target quantities
        targets_res[target] = target_quantity
        
    # save file in object
    saving_path = global_dict["output_path"]["data"]
    if ground_truth:
        saving_path = saving_path+"/expert"
    path = saving_path+'/target_quantities.pkl'
    save_as_pkl(targets_res, path)
    # return results
    return targets_res
        

def computation_elicited_statistics(target_quantities, ground_truth, global_dict):
    # initialize feedback behavior
    logs = LogsInfo(global_dict["log_info"])
    logs("...compute target quantities", 3)
    # names of elicitation techniques
    name_elicits = global_dict["target_quantities"]["elicitation_method"]
    # names of target quantities
    name_targets = global_dict["target_quantities"]["name"]
    # check for support of elicitation technique 
    assert set(name_elicits).issubset(set(["quantiles", "histogram", "moments"])), "Name error of elicitation techniques. Currently supported elicitation techniques are quantiles, histogram, moments."
    # initialize dict for storing results
    elicits_res = dict()
    # loop over elicitation techniques
    for i, (target, elicit) in enumerate(zip(name_targets, name_elicits)):
        # use custom function for target quantity if it has been defined 
        # TODO: this functionality has to be checked!
        if global_dict["target_quantities"]["custom_elicitation_function"][i] is not None:
            elicited_statistic = use_custom_functions(
                global_dict["target_quantities"]["custom_elicitation_function"][i], 
                target_quantities)
            
        if elicit == "histogram":
            elicited_statistic = target_quantities[target]
        
        if elicit == "quantiles":
            quantiles = global_dict["target_quantities"]["quantiles_specs"][i]
            assert quantiles[-1] > 1, "quantiles must be specified as values between [0, 100]" 
            assert quantiles is not None, "no quantiles in the argument quantiles_specs have been defined"
            # compute quantiles
            computed_quantiles = tfp.stats.percentile(target_quantities[target], q = quantiles, axis = 1)
            # bring quantiles to the last dimension
            elicited_statistic = tf.einsum("ij...->ji...", computed_quantiles)
        
        if elicit == "moments":
            moments = global_dict["target_quantities"]["moments_specs"][i]
            assert moments is not None, "no moments in the argument moments_specs have been defined"
            # for each moment
            # TODO: implement feature for custom moment functions
            for moment in moments:
                # check whether moment is supported
                assert moment in ["sd", "mean"], "currently only 'mean' and 'sd' are supported as moments"
                
                if moment == "mean":
                    computed_mean = tf.reduce_mean(target_quantities[target], axis = 1)
                    elicited_statistic = computed_mean
                if moment == "sd":
                    computed_sd = tf.math.reduce_std(target_quantities[target], axis = 1) 
                    elicited_statistic = computed_sd
                # save all moments in one tensor
                elicits_res[f"{elicit}.{moment}_{target}"] = elicited_statistic

        if elicit != "moments":
            elicits_res[f"{elicit}_{target}"] = elicited_statistic
    # save file in object
    saving_path = global_dict["output_path"]["data"]
    if ground_truth:
        saving_path = saving_path+"/expert"
    path = saving_path+'/elicited_statistics.pkl'
    save_as_pkl(elicits_res, path)
    # return results
    return elicits_res