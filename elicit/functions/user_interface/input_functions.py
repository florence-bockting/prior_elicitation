import inspect
import keras
import tensorflow as tf
import warnings
from configs.config_loss import MmdEnergy
from configs.config_user_input import create_global_dict
from dags.dag_tf import prior_elicitation_dag

def normalizing_flow_specs(
        num_coupling_layers: int = 7,
        coupling_design: str = "affine", 
        coupling_settings: dict = {
            "dropout": False,
            "dense_args": {
                "units": 128,
                "activation": "softplus",
                "kernel_regularizer": tf.keras.regularizers.l2(1e-4)
                },
            "num_dense": 2
            },
        permutation: str = "fixed",
        **kwargs
        ) -> dict:
    # for more information see BayesFlow documentation
    # https://bayesflow.org/api/bayesflow.inference_networks.html
    
    nf_specs_dict = {
        "num_coupling_layers": num_coupling_layers,
        "coupling_design": coupling_design,      
        "coupling_settings": coupling_settings,                                 
        "permutation": permutation
        }
    
    return nf_specs_dict

def param(name: str, 
          family: callable = None, 
          hyperparams_dict: dict = None,
          scaling_value: float = 1.) -> dict:
    
    # specifies name and value of prior hyperparameters
    if hyperparams_dict is not None:
        hyppar_dict = {key: value for key,value in zip(hyperparams_dict.keys(),
                                                       hyperparams_dict.values())}
    else:
        hyppar_dict = None
    
    # creates final dictionary with prior specifications
    parameter_dict = {
        "name": [name], 
        "family": [family], 
        "hyperparams_dict": [hyppar_dict],
        "scaling_value": [scaling_value]
        }
    
    return parameter_dict


def model(model: callable, 
          additional_model_args: dict, 
          discrete_likelihood: bool,
          softmax_gumble_specs: dict or None = {"temperature": 1.,
                                                "upper_threshold": None}
          ) -> dict:
  
    softmax_gumble_specs_default = {"temperature": 1.,
                                    "upper_threshold": None}
    
    softmax_gumble_specs_default.update(softmax_gumble_specs)
    
    model_dict = {
        "model_function": model,
        "additional_model_args": additional_model_args,
        "softmax_gumble_specs": softmax_gumble_specs_default
        }
    
    # provide a warning if the upper_threshold argument is not specified
    assert discrete_likelihood is True and "upper_threshold" in softmax_gumble_specs, "The upper_threshold argument in the softmax-gumble specifications is None."
    if discrete_likelihood is True and "temperature" not in softmax_gumble_specs:
        warnings.warn(f"The Softmax-Gumble method with default temperature: {softmax_gumble_specs_default['temperature']} is used.")
    # get model arguments
    get_model_args = set(inspect.getfullargspec(model())[0]).difference({"self","prior_samples"})
    # check that all model arguments have been passed as input in the model_args section
    difference_set = get_model_args.difference(set(additional_model_args.keys()))
    assert len(difference_set) == 0, f"model arguments and specified arguments are not identical. Check the following argument(s): {difference_set}"
    
    return model_dict

def target(name: str, 
           elicitation_method: str, 
           loss_components: str, 
           select_obs: list or None = None, 
           quantiles_specs: tuple or None = None, 
           moments_specs: tuple or None = None, 
           custom_target_function: dict or None = None,
           custom_elicitation_function: dict or None = None) -> dict:
    
    def check_custom_func(custom_function, func_type):
        if custom_function is not None:
            # check whether custom_function has correct form if specified
            assert "function" in list(custom_function.keys()), f"custom_{func_type}_function must be a dictionary with required key 'function' and optional key 'additional_args'"
            default_custom_function = {
                "function": None,
                "additional_args": None
                }
            default_custom_function.update(custom_function)
            
            # check whether additional args have been specified 
            # if they are specified check for correct form
            try:
                default_custom_function["additional_args"]
            except: 
                pass
            else:
                if default_custom_function["additional_args"] is not None:
                    assert type(default_custom_function["additional_args"]) is dict, "additional_args must be a dictionary with keys: 'name' (str) and 'value'"
                    assert {"name","value"}.issubset(set(default_custom_function["additional_args"].keys())), "additional_args must be a dictionary with keys: 'name' (str) and 'value'"
            
            custom_function = default_custom_function    
        return custom_function
    
    check_custom_func(custom_target_function, func_type = "target")
    check_custom_func(custom_elicitation_function, func_type = "elicitation")
    # currently implemented loss-component combis
    assert loss_components in ["by-group", "all", "by-stats"], "Currently only available values for loss_components are 'all', 'by-stats' or 'by-group'."
    # currently implemented elicitation methods are histogram, quantiles, or moments
    assert elicitation_method in ["quantiles", "histogram", "moments"], f"The elicitation method {elicitation_method} is not implemented"
    # if elicitation method 'quantiles' is used, quantiles_specs has to be defined
    if elicitation_method == "quantiles":
        assert quantiles_specs is not None, "quantiles_specs has to be defined for elicitation method 'quantiles'"
    if elicitation_method == "moments":
        assert moments_specs is not None, "moments_specs has to be defined for elicitation method 'moments'"
    
    target_dict = {
        "name": [name],
        "elicitation_method": [elicitation_method],
        "select_obs": [select_obs],
        "quantiles_specs": [quantiles_specs],
        "moments_specs": [moments_specs],
        "custom_target_function": [custom_target_function],
        "custom_elicitation_function": [custom_elicitation_function],
        "loss_components": [loss_components]
        }
    
    return target_dict

def expert(data: str or None,
           simulate_data: bool,
           simulator_specs: dict or None) -> dict:
    # either expert data or expert simulator has to be specified
    assert not(data is None and simulate_data is False), "either a path to expert data has to be specified or simulation of expert data has to be true."
    # if expert simulator is true, simulator specifications have to be given
    assert simulate_data is True and simulator_specs is not None, "if expert simulation is true simulator specifications need to be provided"
    
    expert_dict = {
        "data": data,
        "simulate_data": simulate_data,
        "simulator_specs": simulator_specs
        } 
    return expert_dict

def loss(loss_function: str or callable = "mmd-energy", 
         loss_weighting: dict = {
             "method": "dwa",
             "method_specs": { "temperature": 1.6 }
             }
         ) -> dict:
    
    # check whether user-specified loss function is implemented
    if type(loss_function) is str:
        assert loss_function in ["mmd-energy"], "Currently only the following loss functions are implemented 'mmd-energy'."
    if loss_function == "mmd-energy":
        # call class and initialize it
        call_loss_function = MmdEnergy()
    # check whether user-specified loss-weighting is implemented
    assert loss_weighting["method"] in ["dwa"], "Currently only the following loss-weighting methods are implemented 'dwa'."

    loss_dict = {
        "loss_function": call_loss_function,
        "loss_weighting": loss_weighting
        }
    return loss_dict

def optimization(optimizer: callable = keras.optimizers.legacy.Adam,
                 optimizer_specs: dict = {
                     "learning_rate": callable or float,
                     "clipnorm": 1.0
                     }
                 ) -> dict:
    # TODO assert that keywords of optimizer are in optimizer_specs
    optim_dict = {
        "optimizer": optimizer,
        "optimizer_specs": optimizer_specs
        }
    
    return optim_dict

def prior_elicitation(
        method: str,
        sim_id: str,
        epochs: int,
        B: int,
        rep: int,
        seed: int,
        model_params: callable,
        expert_input: callable,
        generative_model: callable,
        target_quantities: callable,
        loss_function: callable,
        optimization_settings: callable,
        output_path: str = "results",
        log_info: int = 0,
        view_ep: int = 1
        ) -> dict:
    
    # create global dict.
    global_dict = create_global_dict(
        method, sim_id, epochs, B, rep, seed, model_params, expert_input, 
        generative_model, target_quantities, loss_function, 
        optimization_settings, output_path, log_info, view_ep)
    
    # run workflow
    prior_elicitation_dag(global_dict)
    
