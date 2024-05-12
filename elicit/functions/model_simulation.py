import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import inspect

tfd = tfp.distributions

from functions.helper_functions import save_as_pkl, LogsInfo

def softmax_gumbel_trick(model_simulations, global_dict, ground_truth):
    # set batch size to 1 if simulating expert
    if ground_truth:
        B = 1
    else:
        B = global_dict["B"]
    # initialize counter
    number_obs = 0
    # get number of observations
    number_obs = model_simulations["epred"].shape[2]
    # constant outcome vector (including zero outcome)
    c = tf.range(global_dict["generative_model"]["softmax_gumble_specs"]["upper_threshold"]+1, 
                 delta=1, dtype=tf.float32)
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
        tf.math.divide(tf.math.add(tf.math.log(pi), g), 
                       global_dict["generative_model"]["softmax_gumble_specs"]["temperature"])
    )
    # reparameterization/linear transformation
    ypred = tf.reduce_sum(tf.multiply(w, c), axis=-1)
    return ypred    
 
def simulate_from_generator(prior_samples, ground_truth, global_dict): 
    # initialize feedback behavior
    logs = LogsInfo(global_dict["log_info"])   
    # get model and initialize generative model
    # TODO: I silently assume that the given model_function is an "uninitialized class"
    GenerativeModel = global_dict["generative_model"]["model_function"]
    generative_model = GenerativeModel() 
    # get model specific arguments (that are not prior samples) 
    add_model_args = global_dict["generative_model"]["additional_model_args"]
    # simulate from generator
    model_simulations = generative_model(prior_samples, **add_model_args)
    # estimate gradients for discrete likelihood if necessary
    if model_simulations["likelihood"].reparameterization_type != tfd.FULLY_REPARAMETERIZED:
        logs("...apply softmax-gumbel trick for discrete likelihood", 3)
        model_simulations["ypred"] = softmax_gumbel_trick(model_simulations, global_dict, ground_truth)
    # save file in object
    saving_path = global_dict["output_path"]["data"]
    if ground_truth:
        saving_path = saving_path+"/expert"
    path = saving_path +'/model_simulations.pkl'
    save_as_pkl(model_simulations, path)
    return model_simulations