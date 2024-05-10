import configparser
import os
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def create_global_config(file_name):
    # placeholder for none
    NULL = type(None)()

    CONFIG = configparser.ConfigParser()
    CONFIG.read(os.path.join(os.path.dirname(__name__), 'elicit/configs', 'config_global.cfg'))

    CONFIG_CASE = configparser.ConfigParser()
    CONFIG_CASE.read(os.path.join(os.path.dirname(__name__), 'elicit/configs', 'case_study_specific', file_name))

    # get sections 
    default_sections = CONFIG.sections()
    sections = list(set(CONFIG_CASE.sections()).union(set(default_sections)))

    # loop over all sections and key-value pairs in the default config file
    # if same section and key exists in the case-study config, the value of the global config will be overwritten
    global_dict = {}
    for section in sections:
        try:
            dict(CONFIG.items(section))
        except: 
            global_dict.update(dict(CONFIG_CASE.items(section)))
        else:
            global_dict.update(dict(CONFIG.items(section)))

        try:
            dict(CONFIG_CASE.items(section))
        except:
            pass
        else:
            global_dict.update(dict(CONFIG_CASE.items(section)))
    
    final_dict = {}
    for key, val in zip(global_dict.keys(), global_dict.values()):
        final_dict[key] = eval(val)
    
    #### general global variables
    saving_path = final_dict["save_to_path"]
    if final_dict["save_results"]:
        saving_path = final_dict["save_to_path"]+final_dict["model_id"]
    final_dict["saving_path"] = saving_path

    # check that all values that need to be set have been set:
    for key, val in zip(final_dict.keys(), final_dict.values()):
        if key == "kernel_regularizer":
            pass
        else: 
            assert val is not None, f"No value for {key} has been provided. Check config file."
    
    return final_dict
