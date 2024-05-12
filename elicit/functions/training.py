import tensorflow as tf
import tensorflow_probability as tfp
import time 
import numpy as np
import pandas as pd

tfd = tfp.distributions

from functions.helper_functions import LogsInfo, save_as_pkl
from configs.config_savings import save_hyperparameters, marginal_prior_moments

def training_loop(load_expert_data, priors,
                  one_forward_simulation, compute_loss, global_dict):
    
    # initialize feedback behavior
    logs = LogsInfo(global_dict["log_info"])
    # get expert data
    expert_elicited_statistics = load_expert_data(global_dict)
    # prepare generative model
    prior_model = priors(global_dict)

    total_loss = []
    gradients_ep = []
    time_per_epoch = []
    for epoch in tf.range(global_dict["epochs"]):
        # runtime of one epoch
        epoch_time_start = time.time()
        # initialize the adam optimizer
        get_optimizer = global_dict["optimization_settings"]["optimizer"]
        args_optimizer = global_dict["optimization_settings"]["optimizer_specs"]
        optimizer = get_optimizer(**args_optimizer)
        
        with tf.GradientTape() as tape: 
            # generate simulations from model
            training_elicited_statistics = one_forward_simulation(prior_model, global_dict)
            # comput loss
            weighted_total_loss = compute_loss(training_elicited_statistics, expert_elicited_statistics, global_dict, epoch)
            logs("## compute gradients and update hyperparameters", 2)
            # compute gradient of loss wrt trainable_variables
            gradients = tape.gradient(weighted_total_loss, prior_model.trainable_variables)
            # update trainable_variables using gradient info with adam optimizer
            optimizer.apply_gradients(zip(gradients, prior_model.trainable_variables))

        # time end of epoch
        epoch_time_end = time.time()
        epoch_time = epoch_time_end-epoch_time_start

        # print information for user during training
        # break for loop if loss is NAN and inform about cause
        if tf.math.is_nan(weighted_total_loss):
            print("Loss is NAN. The training process has been stopped.")
            break
        # inform about epoch time, total loss and learning rate
        if epoch % global_dict["view_ep"] == 0:
            if type(global_dict["optimization_settings"]["optimizer_specs"]["learning_rate"]) is float:
                lr = global_dict["optimization_settings"]["optimizer_specs"]["learning_rate"]
            else:
                lr = global_dict["optimization_settings"]["optimizer_specs"]["learning_rate"](epoch)
            print(f"epoch_time: {epoch_time:.3f} sec")
            print(f"Epoch: {epoch}, loss: {weighted_total_loss:.5f}, lr: {lr:.6f}")
        # inform about estimated time until completion
        if epoch > 0 and epoch % global_dict["view_ep"] == 0:
            avg_ep_time = np.mean(time_per_epoch)
            remaining_eps = np.subtract(global_dict["epochs"], epoch)
            estimated_time = np.multiply(remaining_eps,avg_ep_time)
            print(f"Estimated time until completion: {time.strftime('%H:%M:%S', time.gmtime(estimated_time))}")
        if epoch == np.subtract(global_dict["epochs"], 1):
            print("Done :)")

        # save gradients in file
        saving_path = global_dict["output_path"]["data"]
        if global_dict["method"] == "parametric_prior":
            path = saving_path+'/gradients.pkl'
            save_as_pkl(gradients, path)
            # save for each epoch
            gradients_ep.append(gradients)

        # savings per epoch
        time_per_epoch.append(epoch_time)
        total_loss.append(weighted_total_loss)
  
        if global_dict["method"] == "parametric_prior":
            res_dict = save_hyperparameters(prior_model, epoch, global_dict)
        else:
            res_dict = marginal_prior_moments(pd.read_pickle(saving_path+"/prior_samples.pkl"), epoch, global_dict)
        
        
    # save final results in file
    res = {"loss": total_loss, "hyperparameter": res_dict}
    if global_dict["method"] == "parametric_prior":
        res["gradients"] = gradients_ep
    path = saving_path+'/final_results.pkl'
    save_as_pkl(res, path)
    