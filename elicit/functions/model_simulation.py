import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import inspect

tfd = tfp.distributions

from functions.helper_functions import save_as_pkl, LogsInfo

def softmax_gumbel_trick(model_simulations, design_matrix, global_dict, ground_truth):
    # set batch size to 1 if simulating expert
    if ground_truth:
        B = 1
    else:
        B = global_dict["b"]
    # initialize counter
    number_obs = 0
    # get number of observations
    if len(design_matrix.shape) == 1:
        number_obs = len(design_matrix)
    else:
        number_obs = design_matrix.shape[-2]

    # constant outcome vector (including zero outcome)
    c = tf.range(global_dict["upper_threshold"]+1, delta=1, dtype=tf.float32)
    # broadcast to shape (B, rep, outcome-length)
    c_brct = tf.broadcast_to(c[None, None, None, :], 
                             shape=(B, global_dict["rep"], number_obs, len(c)))
    # compute pmf value
    pi = model_simulations["likelihood"].prob(c_brct)
    # prevent underflow
    pi = tf.where(pi < 1.8 * 10 ** (-30), 1.8 * 10 ** (-30), pi)
    # sample from uniform
    u = tfd.Uniform(0, 1).sample((B, global_dict["rep"], number_obs, len(c)))
    # generate a gumbel sample from uniform sample
    g = -tf.math.log(-tf.math.log(u))
    # softmax gumbel trick
    w = tf.nn.softmax(
        tf.math.divide(tf.math.add(tf.math.log(pi), g), global_dict["temperature_softmax"])
    )
    # reparameterization/linear transformation
    ypred = tf.reduce_sum(tf.multiply(w, c), axis=-1)
    return ypred    
 
def simulate_from_generator(prior_samples, design_matrix_path, ground_truth, global_dict): 
    # initialize feedback behavior
    logs = LogsInfo(global_dict["log_info"])   
    # load design matrix
    design_matrix = pd.read_pickle(rf"{design_matrix_path}")
    # get model and initialize generative model
    import configs.config_models as ccm
    GenerativeModel = getattr(ccm, global_dict["model_name"])
    generative_model = GenerativeModel() 
    # get model specific variables from config file 
    add_model_args = set(inspect.getfullargspec(generative_model)[0]).difference({"self","prior_samples", "design_matrix"})
    dict_mapping = {}
    if len(add_model_args) != 0:
        for key in add_model_args:
            dict_mapping[key] = global_dict[key]
    # simulate from generator
    model_simulations = generative_model(prior_samples, design_matrix, **dict_mapping)
    # estimate gradients for discrete likelihood if necessary
    if model_simulations["likelihood"].reparameterization_type != tfd.FULLY_REPARAMETERIZED:
        logs("...apply softmax-gumbel trick for discrete likelihood", 3)
        model_simulations["ypred"] = softmax_gumbel_trick(model_simulations, design_matrix, global_dict, ground_truth)
    # save file in object
    saving_path = global_dict["saving_path"]
    if ground_truth:
        saving_path = saving_path+"/expert"
    path = saving_path +'/model_simulations.pkl'
    save_as_pkl(model_simulations, path)
    return model_simulations