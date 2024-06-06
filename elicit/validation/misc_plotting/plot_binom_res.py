import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from setup.write_results import write_results 

def plot_results_overview(path, file, title):
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    expert_res = pd.read_pickle(path+file+"/expert/prior_samples.pkl")
    model_samples = pd.read_pickle(path+file+"/prior_samples.pkl")
    
    model_elicit = pd.read_pickle(path+file+"/elicited_statistics.pkl")
    expert_elicit = pd.read_pickle(path+file+"/expert/elicited_statistics.pkl")
    
    expert_loc = tf.reduce_mean(expert_res, (0,1))
    expert_scale = tf.reduce_mean(tf.math.reduce_std(expert_res, 1), 0)
    
    total_loss = tf.stack(final_res["loss"], -1)
    component_loss = tf.stack(final_res["loss_component"], -1)
    hyperparam_loc =  tf.stack(final_res["hyperparameter"]["means"], -1)
    hyperparam_scale =  tf.stack(final_res["hyperparameter"]["stds"], -1)
    
    c_exp = "#03396c"
    c_mod = "#45b5fd"
    
    fig = plt.figure(layout = "constrained", figsize = (8,8))
    subfigs = fig.subfigures(3,1, height_ratios = [1,0.8,0.8])
    
    fig0 = subfigs[0].subplots(2,2, sharex = True)
    fig1 = subfigs[1].subplots(2,2)
    fig2 = subfigs[2].subplots(1,6)
    
    fig0[0,0].plot(range(len(total_loss)), total_loss)
    [fig0[0,1].plot(range(len(total_loss)), component_loss[i,:]) for i in range(component_loss.shape[0])]
    [fig0[1,0].axhline(expert_loc[i], linestyle = "dashed", color = "black") for i in range(expert_loc.shape[0])]
    [fig0[1,1].axhline(expert_scale[i], linestyle = "dashed", color = "black") for i in range(expert_loc.shape[0])]
    [fig0[1,0].plot(range(len(total_loss)), hyperparam_loc[i,:], lw =3) for i in range(hyperparam_loc.shape[0])]
    [fig0[1,1].plot(range(len(total_loss)), hyperparam_scale[i,:], lw =3, label = fr"$\lambda_{i}$") for i in range(hyperparam_scale.shape[0])]
    fig0[1,1].legend(loc="upper right", handlelength=0.4, fontsize="small", ncol=2)
    
    [fig0[0,i].set_title(t) for i,t in enumerate(["total loss", "loss per component"])]
    [fig0[1,i].set_title(t) for i,t in enumerate(["marginal priors: mean", "marginal priors: sd"])] 
    
    [fig0[1,i].set_xlabel("epochs") for i in range(2)]
    [fig0[0,i].set_yscale("log") for i in range(2)]
    fig0[1,1].set_yscale("log") 
    
    for i in range(2):
        [sns.kdeplot(model_samples[b,:,i], color = c_mod, alpha = 0.2, ax = fig1[i,i]) for b in range(100)]
        sns.kdeplot(expert_res[0,:,i], color = c_exp, linestyle = "dashed", ax = fig1[i,i])
    for i,j in zip([0,0,0,1,1,2], [1,2,3,2,3,3]):
        sns.scatterplot(x=model_samples[0,:,0], y = model_samples[0,:,1], color = c_mod,
                        marker="x", ax = fig1[0,1])
        sns.scatterplot(x=expert_res[0,:,0], y = expert_res[0,:,1], marker="+", 
                        color = c_exp, ax = fig1[0,1])
        sns.kdeplot(x=model_samples[0,:,1], y = model_samples[0,:,0], color = c_mod,
                        ax = fig1[1,0])
        sns.kdeplot(x=expert_res[0,:,1], y = expert_res[0,:,0],
                        color = c_exp, ax = fig1[1,0])
    [fig1[i,j].get_xaxis().set_visible(False) for i,j in product(range(2),range(2))]
    [fig1[i,j].get_yaxis().set_visible(False) for i,j in product(range(2),range(2))]
    [fig1[i,j].spines[['right', 'top']].set_visible(False) for i,j in product(range(2),range(2))]
    subfigs[1].suptitle("Joint prior")
    
    subfigs[2].suptitle("elicited statistics")
    
    [fig2[x].axline((model_elicit["quantiles_ypred"][0,0,x],
                    model_elicit["quantiles_ypred"][0,0,x]), 
                    slope = 1, color = "black", linestyle = "dashed") for x in range(6)]
    for x in range(6):
        [sns.scatterplot(
            x = model_elicit["quantiles_ypred"][b,:,x],  
            y = expert_elicit["quantiles_ypred"][0,:,x],
            ax = fig2[x],
            color = c_mod, alpha = 0.2,
            s=50
            ) for b in range(100)]
        fig2[x].set_yticklabels([])
        fig2[x].set_title(f"y_pred {x}")
    fig2[0].set_xlabel("training data")
    fig2[0].set_ylabel("expert data")

    fig.suptitle(title)
    plt.savefig(path+file+".png")
    plt.show()

    # write results table if not done already:
    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    write_results(path+file,global_dict)


def plot_results_overview_param(path, file, title):
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    expert_res = pd.read_pickle(path+file+"/expert/prior_samples.pkl")
    model_samples = pd.read_pickle(path+file+"/prior_samples.pkl")
    
    model_elicit = pd.read_pickle(path+file+"/elicited_statistics.pkl")
    expert_elicit = pd.read_pickle(path+file+"/expert/elicited_statistics.pkl")
    
    expert_loc = tf.reduce_mean(expert_res, (0,1))
    expert_scale = tf.reduce_mean(tf.math.reduce_std(expert_res, 1), 0)
    
    hyperparams = tf.stack([final_res["hyperparameter"][key] for key in final_res["hyperparameter"].keys()], 0)
    hyperparam_loc = tf.gather(hyperparams, [0,2], axis = 0)   
    hyperparam_scale = tf.exp(tf.gather(hyperparams, [1,3], axis = 0))   
    
    total_loss = tf.stack(final_res["loss"], -1)
    component_loss = tf.stack(final_res["loss_component"], -1)
    
    c_exp = "#03396c"
    c_mod = "#45b5fd"
    
    fig = plt.figure(layout = "constrained", figsize = (8,8))
    subfigs = fig.subfigures(3,1, height_ratios = [1,0.8,0.8])
    
    fig0 = subfigs[0].subplots(2,2, sharex = True)
    fig1 = subfigs[1].subplots(2,2)
    fig2 = subfigs[2].subplots(1,6)
    
    fig0[0,0].plot(range(len(total_loss)), total_loss)
    [fig0[0,1].plot(range(len(total_loss)), component_loss[i,:]) for i in range(component_loss.shape[0])]
    [fig0[1,0].axhline(expert_loc[i], linestyle = "dashed", color = "black") for i in range(expert_loc.shape[0])]
    [fig0[1,1].axhline(expert_scale[i], linestyle = "dashed", color = "black") for i in range(expert_loc.shape[0])]
    [fig0[1,0].plot(range(len(total_loss)), hyperparam_loc[i,:], lw =3) for i in range(hyperparam_loc.shape[0])]
    [fig0[1,1].plot(range(len(total_loss)), hyperparam_scale[i,:], lw =3, label = fr"$\lambda_{i}$") for i in range(hyperparam_scale.shape[0])]
    fig0[1,1].legend(loc="upper right", handlelength=0.4, fontsize="small", ncol=2)
    
    [fig0[0,i].set_title(t) for i,t in enumerate(["total loss", "loss per component"])]
    [fig0[1,i].set_title(t) for i,t in enumerate([r"hyperparameter: $\mu_k$", r"hyperparameter: $\sigma_k$"])] 
    
    [fig0[1,i].set_xlabel("epochs") for i in range(2)]
    [fig0[0,i].set_yscale("log") for i in range(2)]
    fig0[1,1].set_yscale("log") 
    
    for i in range(2):
        [sns.kdeplot(model_samples[b,:,i], color = c_mod, alpha = 0.2, ax = fig1[i,i]) for b in range(100)]
        sns.kdeplot(expert_res[0,:,i], color = c_exp, linestyle = "dashed", ax = fig1[i,i])
    for i,j in zip([0,0,0,1,1,2], [1,2,3,2,3,3]):
        sns.scatterplot(x=model_samples[0,:,0], y = model_samples[0,:,1], color = c_mod,
                        marker="x", ax = fig1[0,1])
        sns.scatterplot(x=expert_res[0,:,0], y = expert_res[0,:,1], marker="+", 
                        color = c_exp, ax = fig1[0,1])
        sns.kdeplot(x=model_samples[0,:,1], y = model_samples[0,:,0], color = c_mod,
                        ax = fig1[1,0])
        sns.kdeplot(x=expert_res[0,:,1], y = expert_res[0,:,0],
                        color = c_exp, ax = fig1[1,0])
    [fig1[i,j].get_xaxis().set_visible(False) for i,j in product(range(2),range(2))]
    [fig1[i,j].get_yaxis().set_visible(False) for i,j in product(range(2),range(2))]
    [fig1[i,j].spines[['right', 'top']].set_visible(False) for i,j in product(range(2),range(2))]
    subfigs[1].suptitle("Joint prior")
    
    subfigs[2].suptitle("elicited statistics")
    
    [fig2[x].axline((model_elicit["quantiles_ypred"][0,0,x],
                    model_elicit["quantiles_ypred"][0,0,x]), 
                    slope = 1, color = "black", linestyle = "dashed") for x in range(6)]
    for x in range(6):
        [sns.scatterplot(
            x = model_elicit["quantiles_ypred"][b,:,x],  
            y = expert_elicit["quantiles_ypred"][0,:,x],
            ax = fig2[x],
            color = c_mod, alpha = 0.2,
            s=50
            ) for b in range(100)]
        fig2[x].set_yticklabels([])
        fig2[x].set_title(f"y_pred {x}")
    fig2[0].set_xlabel("training data")
    fig2[0].set_ylabel("expert data")

    fig.suptitle(title)
    plt.show()
    plt.savefig(path+file+".png")

    # write results table if not done already:
    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    write_results(global_dict)


