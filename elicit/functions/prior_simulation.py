import tensorflow as tf
import tensorflow_probability as tfp
import bayesflow as bf

tfd = tfp.distributions
bfn = bf.networks

from functions.helper_functions import save_as_pkl, LogsInfo

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

def intialize_priors(global_dict):
    # initialize feedback behavior
    logs = LogsInfo(global_dict["log_info"])
    logs("...initialize prior distributions", 4)
    # number parameters
    no_param = len(global_dict["name_param"])

    if global_dict["method"] == "parametric_prior":
        # list for saving initialize hyperparameter values
        init_hyperparam_list = []
        # loop over model parameter and initialize each hyperparameter 
        # given the user specified initialization
        for model_param in range(no_param):
            initialized_hyperparam = dict()
            for hyperparam, hyperparam_name in enumerate(global_dict["name_hyperparam"][model_param]):

                init_dist = global_dict["initialization"][model_param][hyperparam]

                initialized_hyperparam[f"{hyperparam_name}"] = tf.Variable(
                    initial_value=init_dist.sample(),
                    trainable=True,
                    name=f"{hyperparam_name}",
                )
            init_hyperparam_list.append(initialized_hyperparam)    
        # save initialized priors
        init_prior = init_hyperparam_list
        # save file in object
        path = global_dict["saving_path"]+'/init_prior.pkl'
        save_as_pkl(init_prior, path)

    if global_dict["method"] == "deep_prior":
        # for more information see BayesFlow documentation
        # https://bayesflow.org/api/bayesflow.inference_networks.html
        coupling_settings = {
            "dense_args": dict(
                units = global_dict["units"],
                activation = global_dict["activation_function"],
                kernel_regularizer = global_dict["kernel_regularizer"]
            ),
            "hidden_layers": global_dict["hidden_layers"],
            "dropout": global_dict["dropout"]
        }
        
        if global_dict["coupling_design"] == "spline":
            coupling_settings["bins"] = global_dict["bins"]
            
        invertible_neural_network = bfn.InvertibleNetwork(
            num_params = no_param,
            num_coupling_layers = global_dict["num_coupling_layers"],
            coupling_design = global_dict["coupling_design"],
            coupling_settings = coupling_settings,
            permutation = global_dict["permutation"],
            use_act_norm = False
        )
        # save initialized priors
        init_prior = invertible_neural_network
        
    return init_prior

def sample_from_priors(initialized_priors, ground_truth, global_dict):
    # extract variables from dict
    rep = global_dict["rep"]
    B = global_dict["b"]
    method = global_dict["method"]
    scale_prior_samples = global_dict["scale_prior_samples"]
    # initialize feedback behavior
    logs = LogsInfo(global_dict["log_info"])
    # number parameters
    no_param = len(global_dict["name_param"])

    if ground_truth:
        priors = []
        for prior in global_dict["true"]:
            # sample from the prior distribution 
            priors.append(prior.sample((1, rep)))
        prior_samples = tf.stack(priors, axis=-1)

    if method == "parametric_prior" and not ground_truth:
        initialized_hyperparameters = initialized_priors
        priors = []
        for model_param in range(no_param):
            # get the prior distribution family as specified by the user
            prior_family = global_dict["family"][model_param]
            if type(prior_family) is str:
                if prior_family.startswith("custom_"):
                    # if prior distribution family is a custom function; import it from custom_functions
                    import configs.config_custom_functions as cccf
                    prior_family = getattr(cccf, prior_family)
            # save hyperparameter values in list
            hyperparams = [initialized_hyperparameters[model_param][key] for key 
                           in initialized_hyperparameters[model_param].keys()]
            # sample from the prior distribution 
            priors.append(prior_family(*hyperparams).sample((B, rep)))
        # stack all prior distributions into one tf.Tensor of
        # shape (B, rep, num_parameters)
        prior_samples = tf.stack(priors, axis=-1)

    if method == "deep_prior" and not ground_truth:
        # initialize base distribution
        base_distribution = tfd.Normal(loc=tf.zeros(no_param), 
                                       scale=tf.ones(no_param))
        # sample from base distribution
        u = base_distribution.sample((B, rep))
        # apply transformation function to samples from base distr.
        prior_samples, _ = initialized_priors(u, condition = None, inverse = False)

    # create results dictionary
    prior_samples_dict = {
        "prior_samples": prior_samples
    }

    if scale_prior_samples is not None and not ground_truth:
        logs("...scale samples from prior distributions", 4)
        # scale prior samples
        scaled_prior_samples = tf.stack([prior_samples[:,:,i]*scaling_factor for i, scaling_factor in enumerate(scale_prior_samples)], -1)
        # save results in dict
        prior_samples_dict["scaled_prior_samples"] = scaled_prior_samples
    
    # save results in path
    saving_path = global_dict["saving_path"]
    if ground_truth:
        saving_path = saving_path+"/expert"
    path_prior_samples_dict = saving_path+'/prior_samples_dict.pkl'
    path_prior_samples = saving_path+'/prior_samples.pkl'
    save_as_pkl(prior_samples_dict, path_prior_samples_dict)
    save_as_pkl(prior_samples, path_prior_samples)
    return prior_samples