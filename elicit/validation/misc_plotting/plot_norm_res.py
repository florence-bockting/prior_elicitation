import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

############### CAVEAT: I use tf.abs() for the Gamma parameter... is this reasonable?
from setup.write_results import write_results 

def plot_results_overview(path, file, title):
    
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    expert_res = pd.read_pickle(path+file+"/expert/model_simulations.pkl")["prior_samples"]
    model_samples = pd.read_pickle(path+file+"/model_simulations.pkl")["prior_samples"]
    
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
    
    fig = plt.figure(layout = "constrained", figsize = (8,12))
    subfigs = fig.subfigures(6,1, height_ratios = [0.7,0.7,1,0.5,0.5,0.5])
    
    fig0 = subfigs[0].subplots(1,2, sharex = True)
    fig01 = subfigs[1].subplots(1,3, sharex = True)
    fig1 = subfigs[2].subplots(7,7)
    fig2 = subfigs[3].subplots(1,3)
    fig3 = subfigs[4].subplots(1,3)
    fig4 = subfigs[5].subplots(1,3)
    
    fig0[0].plot(range(len(total_loss)), total_loss)
    [fig0[1].plot(range(len(total_loss)), component_loss[i,:]) for i in range(component_loss.shape[0]-1)]
    [fig01[0].axhline(expert_loc[i], linestyle = "dashed", color = "black") for i in range(expert_loc.shape[0]-1)]
    [fig01[1].axhline(expert_scale[i], linestyle = "dashed", color = "black") for i in range(expert_loc.shape[0]-1)]
    fig01[2].axhline(expert_loc[-1], linestyle = "dashed", color = "black")
    fig01[2].axhline(expert_scale[-1], linestyle = "dashed", color = "black")
    [fig01[0].plot(range(len(total_loss)), hyperparam_loc[i,:], lw =3) for i in range(hyperparam_loc.shape[0]-1)]
    [fig01[1].plot(range(len(total_loss)), hyperparam_scale[i,:], lw =3, label = fr"$\lambda_{i}$") for i in range(hyperparam_scale.shape[0]-1)]
    fig01[2].plot(range(len(total_loss)), hyperparam_loc[-1,:], lw =3)
    fig01[2].plot(range(len(total_loss)), hyperparam_scale[-1,:], lw =3)
    fig01[1].legend(loc="upper right", handlelength=0.4, fontsize="small", ncol=4,
                    columnspacing=0.5)
    
    [fig0[i].set_title(t) for i,t in enumerate(["total loss", "loss per component"])]
    [fig01[i].set_title(t) for i,t in enumerate([r"location $\mu_k$", r"scale $\sigma_k$", "softplus_gamma"])] 
    
    [fig0[i].set_xlabel("epochs") for i in range(2)]
    [fig01[i].set_xlabel("epochs") for i in range(3)]
    [fig0[i].set_yscale("log") for i in range(2)]
    [fig01[i].set_yscale("log") for i in [1,2]] 
    
    for i in range(7):
        [sns.kdeplot(model_samples[b,:,i], color = c_mod, alpha = 0.2, ax = fig1[i,i]) for b in range(100)]
        sns.kdeplot(expert_res[0,:,i], color = c_exp, linestyle = "dashed", ax = fig1[i,i])
    for i,j in zip([0,0,0,0,0,0,1,1,1,1,1,2,2,2,2,3,3,3,4,4,5], [1,2,3,4,5,6,2,3,4,5,6,3,4,5,6,4,5,6,5,6,6]):
        sns.scatterplot(x=model_samples[0,:,i], y = model_samples[0,:,j], color = c_mod,
                        marker="x", ax = fig1[i,j])
        sns.scatterplot(x=expert_res[0,:,i], y = expert_res[0,:,j], marker="+", 
                        color = c_exp, ax = fig1[i,j])
        sns.kdeplot(x=model_samples[0,:,j], y = model_samples[0,:,i], color = c_mod,
                        ax = fig1[j,i])
        sns.kdeplot(x=expert_res[0,:,j], y = expert_res[0,:,i],
                        color = c_exp, ax = fig1[j,i])
    [fig1[i,j].get_xaxis().set_visible(False) for i,j in product(range(7),range(7))]
    [fig1[i,j].get_yaxis().set_visible(False) for i,j in product(range(7),range(7))]
    [fig1[i,j].spines[['right', 'top']].set_visible(False) for i,j in product(range(7),range(7))]
    subfigs[1].suptitle("Marginal priors")
    subfigs[2].suptitle("Joint prior")
    
    [fig2[x].axline((model_elicit["quantiles_marginal_EnC"][0,0,x],
                    model_elicit["quantiles_marginal_EnC"][0,0,x]), 
                    slope = 1, color = "black", linestyle = "dashed") for x in range(3)]
    for x in range(3):
        [sns.scatterplot(
            x = model_elicit["quantiles_marginal_EnC"][b,:,x],  
            y = expert_elicit["quantiles_marginal_EnC"][0,:,x],
            ax = fig2[x],
            color = c_mod, alpha = 0.2,
            s=50
            ) for b in range(100)]
        fig2[x].set_yticklabels([])
        fig2[x].set_title(f"group EnC {x}")
    fig2[0].set_xlabel("training data")
    fig2[0].set_ylabel("expert data")
    
    [fig3[x].axline((model_elicit["quantiles_mean_effects"][0,0,x],
                    model_elicit["quantiles_mean_effects"][0,0,x]), 
                    slope = 1, color = "black", linestyle = "dashed") for x in range(3)]
    for x in range(3):
        [sns.scatterplot(
            x = model_elicit["quantiles_mean_effects"][b,:,x],  
            y = expert_elicit["quantiles_mean_effects"][0,:,x],
            ax = fig3[x],
            color = c_mod, alpha = 0.2,
            s=50
            ) for b in range(100)]
        fig3[x].set_yticklabels([])
        fig3[x].set_title(f"mean effects {x}")
    fig3[0].set_xlabel("training data")
    fig3[0].set_ylabel("expert data")
    
    
    [fig4[x].axline((model_elicit["quantiles_marginal_ReP"][0,0,x],
                    model_elicit["quantiles_marginal_ReP"][0,0,x]), 
                    slope = 1, color = "black", linestyle = "dashed") for x in range(2)]
    for x in range(2):
        [sns.scatterplot(
            x = model_elicit["quantiles_marginal_ReP"][b,:,x],  
            y = expert_elicit["quantiles_marginal_ReP"][0,:,x],
            ax = fig4[x],
            color = c_mod, alpha = 0.2,
            s=50
            ) for b in range(100)]
        fig4[x].set_yticklabels([])
        fig4[x].set_title(f"group ReP {x}")
    fig4[0].set_xlabel("training data")
    fig4[0].set_ylabel("expert data")
    
    
    [sns.histplot(model_elicit["histogram_R2"][b,...], bins = 20, 
                  color = c_mod, stat="density",
                  alpha = 0.2, ax = fig4[2], edgecolor = None) for b in range(100)]
    sns.kdeplot(expert_elicit["histogram_R2"][0,...], ax = fig4[2], 
                color = c_exp, lw = 3)
    fig4[2].set_title("R2") 
    fig4[2].get_yaxis().set_visible(False) 
    fig4[2].set_xlim(0,1)
    subfigs[3].suptitle("elicited statistics")
    
    fig.suptitle(title)
    plt.savefig(path+file+".png")
    # write results table if not done already:
    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    write_results(path+file, global_dict)
    
    
    
def plot_results_overview_param(path, file, title):
    
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    expert_res = pd.read_pickle(path+file+"/expert/model_simulations.pkl")["prior_samples"]
    model_samples = pd.read_pickle(path+file+"/model_simulations.pkl")["prior_samples"]
    
    model_elicit = pd.read_pickle(path+file+"/elicited_statistics.pkl")
    expert_elicit = pd.read_pickle(path+file+"/expert/elicited_statistics.pkl")
    
    expert_loc = tf.reduce_mean(expert_res, (0,1))
    expert_scale = tf.reduce_mean(tf.math.reduce_std(expert_res, 1), 0)
    
    total_loss = tf.stack(final_res["loss"], -1)
    component_loss = tf.stack(final_res["loss_component"], -1)
    hyperparams =  tf.stack([final_res["hyperparameter"][key] for key in final_res["hyperparameter"].keys()],0)
    hyperparam_loc = tf.gather(hyperparams, [0,2,4,6,8,10], axis=0)
    hyperparam_scale =  tf.gather(tf.exp(hyperparams), [1,3,5,7,9,11], axis=0)
    
    sigma =  tf.gather(tf.exp(hyperparams), [12,13], axis=0)
    
    c_exp = "#03396c"
    c_mod = "#45b5fd"
    
    fig = plt.figure(layout = "constrained", figsize = (8,12))
    subfigs = fig.subfigures(6,1, height_ratios = [0.7,0.7,1,0.5,0.5,0.5])
    
    fig0 = subfigs[0].subplots(1,2, sharex = True)
    fig01 = subfigs[1].subplots(1,3, sharex = True)
    fig1 = subfigs[2].subplots(7,7)
    fig2 = subfigs[3].subplots(1,3)
    fig3 = subfigs[4].subplots(1,3)
    fig4 = subfigs[5].subplots(1,3)
    
    fig0[0].plot(range(len(total_loss)), total_loss)
    [fig0[1].plot(range(len(total_loss)), component_loss[i,:]) for i in range(component_loss.shape[0]-1)]
    [fig01[0].axhline(expert_loc[i], linestyle = "dashed", color = "black") for i in range(expert_loc.shape[0]-1)]
    [fig01[1].axhline(expert_scale[i], linestyle = "dashed", color = "black") for i in range(expert_loc.shape[0]-1)]
    fig01[2].axhline(20, linestyle = "dashed", color = "black")
    fig01[2].axhline(200, linestyle = "dashed", color = "black")
    [fig01[0].plot(range(len(total_loss)), hyperparam_loc[i,:], lw =3) for i in range(hyperparam_loc.shape[0]-1)]
    [fig01[1].plot(range(len(total_loss)), hyperparam_scale[i,:], lw =3, label = fr"$\lambda_{i}$") for i in range(hyperparam_scale.shape[0]-1)]
    fig01[2].plot(range(len(total_loss)), sigma[0,:], lw =3)
    fig01[2].plot(range(len(total_loss)), sigma[1,:], lw =3)
    fig01[1].legend(loc="upper right", handlelength=0.4, fontsize="small", ncol=4,
                    columnspacing=0.5)
    
    [fig0[i].set_title(t) for i,t in enumerate(["total loss", "loss per component"])]
    [fig01[i].set_title(t) for i,t in enumerate([r"Normal $\mu_k$", r"Normal $\sigma_k$", r"Gamma $\alpha$,$\beta$"])] 
    
    [fig0[i].set_xlabel("epochs") for i in range(2)]
    [fig01[i].set_xlabel("epochs") for i in range(3)]
    [fig0[i].set_yscale("log") for i in range(2)]
    [fig01[i].set_yscale("log") for i in [1,2]] 
    
    for i in range(7):
        [sns.kdeplot(model_samples[b,:,i], color = c_mod, alpha = 0.2, ax = fig1[i,i]) for b in range(100)]
        sns.kdeplot(expert_res[0,:,i], color = c_exp, linestyle = "dashed", ax = fig1[i,i])
    for i,j in zip([0,0,0,0,0,0,1,1,1,1,1,2,2,2,2,3,3,3,4,4,5], [1,2,3,4,5,6,2,3,4,5,6,3,4,5,6,4,5,6,5,6,6]):
        sns.scatterplot(x=model_samples[0,:,i], y = model_samples[0,:,j], color = c_mod,
                        marker="x", ax = fig1[i,j])
        sns.scatterplot(x=expert_res[0,:,i], y = expert_res[0,:,j], marker="+", 
                        color = c_exp, ax = fig1[i,j])
        sns.kdeplot(x=model_samples[0,:,j], y = model_samples[0,:,i], color = c_mod,
                        ax = fig1[j,i])
        sns.kdeplot(x=expert_res[0,:,j], y = expert_res[0,:,i],
                        color = c_exp, ax = fig1[j,i])
    [fig1[i,j].get_xaxis().set_visible(False) for i,j in product(range(7),range(7))]
    [fig1[i,j].get_yaxis().set_visible(False) for i,j in product(range(7),range(7))]
    [fig1[i,j].spines[['right', 'top']].set_visible(False) for i,j in product(range(7),range(7))]
    subfigs[1].suptitle("Marginal priors")
    subfigs[2].suptitle("Joint prior")
    
    [fig2[x].axline((model_elicit["quantiles_marginal_EnC"][0,0,x],
                    model_elicit["quantiles_marginal_EnC"][0,0,x]), 
                    slope = 1, color = "black", linestyle = "dashed") for x in range(3)]
    for x in range(3):
        [sns.scatterplot(
            x = model_elicit["quantiles_marginal_EnC"][b,:,x],  
            y = expert_elicit["quantiles_marginal_EnC"][0,:,x],
            ax = fig2[x],
            color = c_mod, alpha = 0.2,
            s=50
            ) for b in range(100)]
        fig2[x].set_yticklabels([])
        fig2[x].set_title(f"group EnC {x}")
    fig2[0].set_xlabel("training data")
    fig2[0].set_ylabel("expert data")
    
    [fig3[x].axline((model_elicit["quantiles_mean_effects"][0,0,x],
                    model_elicit["quantiles_mean_effects"][0,0,x]), 
                    slope = 1, color = "black", linestyle = "dashed") for x in range(3)]
    for x in range(3):
        [sns.scatterplot(
            x = model_elicit["quantiles_mean_effects"][b,:,x],  
            y = expert_elicit["quantiles_mean_effects"][0,:,x],
            ax = fig3[x],
            color = c_mod, alpha = 0.2,
            s=50
            ) for b in range(100)]
        fig3[x].set_yticklabels([])
        fig3[x].set_title(f"mean effects {x}")
    fig3[0].set_xlabel("training data")
    fig3[0].set_ylabel("expert data")
    
    
    [fig4[x].axline((model_elicit["quantiles_marginal_ReP"][0,0,x],
                    model_elicit["quantiles_marginal_ReP"][0,0,x]), 
                    slope = 1, color = "black", linestyle = "dashed") for x in range(2)]
    for x in range(2):
        [sns.scatterplot(
            x = model_elicit["quantiles_marginal_ReP"][b,:,x],  
            y = expert_elicit["quantiles_marginal_ReP"][0,:,x],
            ax = fig4[x],
            color = c_mod, alpha = 0.2,
            s=50
            ) for b in range(100)]
        fig4[x].set_yticklabels([])
        fig4[x].set_title(f"group ReP {x}")
    fig4[0].set_xlabel("training data")
    fig4[0].set_ylabel("expert data")
    
    [sns.histplot(model_elicit["histogram_R2"][b,...], bins = 20, 
                  color = c_mod, stat="density",
                  alpha = 0.2, ax = fig4[2], edgecolor = None) for b in range(100)]
    sns.kdeplot(expert_elicit["histogram_R2"][0,...], ax = fig4[2], 
                color = c_exp, lw = 3)
    fig4[2].set_title("R2") 
    fig4[2].get_yaxis().set_visible(False) 
    fig4[2].set_xlim(0,1)
    subfigs[3].suptitle("elicited statistics")
    
    fig.suptitle(title)
    plt.savefig(path+file+".png")
    # write results table if not done already:
    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    write_results(global_dict)
