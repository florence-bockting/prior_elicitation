import tensorflow as tf
import tensorflow_probability as tfp
import bayesflow as bf
import inspect

tfd = tfp.distributions
bfn = bf.networks

from functions.helper_functions import save_as_pkl, LogsInfo

def computation_target_quantities(model_simulations, ground_truth, global_dict):
    # initialize feedback behavior
    logs = LogsInfo(global_dict["log_info"])
    logs("...compute target quantities", 3)
    # names of target quantities
    name_targets = global_dict["target"]
    # custom functions (if specified)
    try:
        global_dict["custom_target_functions"]
    except:
        custom_target_fct = [None]*len(name_targets)
    else:
        custom_target_fct = global_dict["custom_target_functions"]
    # check for duplicate naming 
    assert len(name_targets) == len(set(name_targets)), "duplicate target quantity name has been detected; target quantities must have unique names."
    # extract names from quantities simulated from the generative model
    names_simulated_quantities = list(model_simulations.keys())
    # initialize dict for storing results
    targets_res = dict()
    # loop over target quantities
    for i, target in enumerate(name_targets):
        # use custom function for target quantity if it has been defined 
        if custom_target_fct[i] is not None:
            # import function
            import configs.config_custom_functions as cccf
            custom_func = getattr(cccf, custom_target_fct[i])
            # extract function arguments
            custom_func_args = inspect.getfullargspec(custom_func)[0]
            # evaluate custom function
            target_quantity = custom_func(**{custom_func_args[i]: model_simulations[custom_func_args[i]] for i in range(len(custom_func_args))})
        # use quantities from model simulations
        if target in names_simulated_quantities:
            target_quantity = model_simulations[target]
        assert (target in names_simulated_quantities) or (custom_target_fct[i] is not None), "name of target quantity is not found in model simulations nor is a custom target function specified"
        # save target quantities
        targets_res[target] = target_quantity
    # save file in object
    saving_path = global_dict["saving_path"]
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
    name_elicits = global_dict["elicitation"]
    # names of target quantities
    name_targets = global_dict["target"]
    # check for support of elicitation technique 
    assert set(name_elicits).issubset(set(["quantiles", "histogram", "moments"])), "Name error of elicitation techniques. Currently supported elicitation techniques are quantiles, histogram, moments."
    # initialize dict for storing results
    elicits_res = dict()
    # loop over elicitation techniques
    for i, (target, elicit) in enumerate(zip(name_targets, name_elicits)):
        if elicit == "histogram":
            elicited_statistic = target_quantities[target]
        
        if elicit == "quantiles":
            quantiles = global_dict["quantiles_specs"][i]
            assert quantiles[-1] > 1, "quantiles must be specified as values between [0, 100]" 
            assert quantiles is not None, "no quantiles in the argument quantiles_specs have been defined"
            # compute quantiles
            computed_quantiles = tfp.stats.percentile(target_quantities[target], q = quantiles, axis = 1)
            # bring quantiles to the last dimension
            elicited_statistic = tf.einsum("i...->...i", computed_quantiles)
        
        if elicit == "moments":
            moments = global_dict["moments_specs"][i]
            assert moments is not None, "no moments in the argument moments_specs have been defined"
            # get moments that are not part of built-in functions
            custom_moments = list(set(moments).difference(set(("mean", "sd"))))
            assert all([custom_moments[i].startswith("custom_") for i in range(len(custom_moments))]), "custom moment functions have to start with 'custom_' and match the function name as specified in config_custom_targets.py"
            # for each moment
            for moment in moments:
                if moment == "mean":
                    computed_mean = tf.reduce_mean(target_quantities[target], axis = 1)
                    elicited_statistic = computed_mean
                if moment == "sd":
                    computed_sd = tf.math.reduce_std(target_quantities[target], axis = 1) 
                    elicited_statistic = computed_sd
                if moment.startswith("custom_"):
                    # import function
                    import configs.config_custom_functions as cccf
                    custom_func = getattr(cccf, moment)
                    # compute function
                    computed_custom_mom = custom_func(target_quantities[target])
                    elicited_statistic = computed_custom_mom
                # save all moments in one tensor
                elicits_res[f"{elicit}.{moment}_{target}"] = elicited_statistic

        if elicit != "moments":
            elicits_res[f"{elicit}_{target}"] = elicited_statistic
    # save file in object
    saving_path = global_dict["saving_path"]
    if ground_truth:
        saving_path = saving_path+"/expert"
    path = saving_path+'/elicited_statistics.pkl'
    save_as_pkl(elicits_res, path)
    # return results
    return elicits_res