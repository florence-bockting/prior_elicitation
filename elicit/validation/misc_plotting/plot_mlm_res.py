import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from setup.write_results import write_results 

def plot_results_overview(path, file, title):
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    expert_res = pd.read_pickle(path+file+"/expert/model_simulations.pkl")["prior_samples"]
    model_samples = pd.read_pickle(path+file+"/model_simulations.pkl")["prior_samples"]
    
    expert_preds = pd.read_pickle(path+file+"/expert/model_simulations.pkl")
    model_preds = pd.read_pickle(path+file+"/model_simulations.pkl")
    
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
    subfigs = fig.subfigures(5,1, height_ratios = [0.7,0.7,1,0.5,0.5])
    
    fig0 = subfigs[0].subplots(1,2, sharex = True)
    fig01 = subfigs[1].subplots(1,3, sharex = True)
    fig1 = subfigs[2].subplots(5,5)
    fig2 = subfigs[3].subplots(1,5)
    fig3 = subfigs[4].subplots(1,4)
    
    fig0[0].plot(range(len(total_loss)), total_loss)
    [fig0[1].plot(range(len(total_loss)), component_loss[i,:], label = f"loss {i}") for i in range(component_loss.shape[0])]
    [fig01[0].axhline(expert_loc[i], linestyle = "dashed", color = "black") for i in range(1,5)]
    [fig01[1].axhline(expert_scale[i], linestyle = "dashed", color = "black") for i in range(5)]
    fig01[2].axhline(expert_loc[0], linestyle = "dashed", color = "black") 
    labels = [r"$\mu_1$", r"$\omega_0$", r"$\omega_1$", r"$\sigma$"]
    [fig01[0].plot(range(len(total_loss)), hyperparam_loc[i,:], lw =3, label = labels[j]) for j,i in enumerate(range(1,5))]
    labels2 = [r"$\beta_0$"]+labels
    [fig01[1].plot(range(len(total_loss)), hyperparam_scale[i,:], lw =3, label = labels2[j]) for j,i in enumerate(range(5))]
    fig01[2].plot(range(len(total_loss)), hyperparam_loc[0,:], lw =3, label = r"$\beta_0$")
    [fig01[i].legend(loc="upper left", handlelength=0.4, fontsize="small", ncol=2,
                    columnspacing=0.5) for i in range(3)]
    fig0[1].legend(loc="upper left", handlelength=0.4, fontsize="small", ncol=2,
                    columnspacing=0.5) 
    
    [fig0[i].set_title(t) for i,t in enumerate(["total loss", "loss per component"])]
    #[fig01[i].set_title(t) for i,t in enumerate([r"Normal $\mu_k$", r"Normal $\sigma_k$", r"Gamma $\alpha$,$\beta$"])] 
    
    [fig0[i].set_xlabel("epochs") for i in range(2)]
    [fig01[i].set_xlabel("epochs") for i in range(3)]
    [fig0[i].set_yscale("log") for i in range(2)]
    [fig01[i].set_yscale("log") for i in [1,2]] 
    
    for i in range(5):
        [sns.kdeplot(model_samples[b,:,i], color = c_mod, alpha = 0.2, ax = fig1[i,i]) for b in range(100)]
        sns.kdeplot(expert_res[0,:,i], color = c_exp, linestyle = "dashed", ax = fig1[i,i])
    for i,j in zip([0,0,0,0,1,1,1,2,2,3], [1,2,3,4,2,3,4,3,4,4]):
        sns.scatterplot(x=model_samples[0,:,i], y = model_samples[0,:,j], color = c_mod,
                        marker="x", ax = fig1[i,j])
        sns.scatterplot(x=expert_res[0,:,i], y = expert_res[0,:,j], marker="+", 
                        color = c_exp, ax = fig1[i,j])
        sns.kdeplot(x=model_samples[0,:,j], y = model_samples[0,:,i], color = c_mod,
                        ax = fig1[j,i])
        sns.kdeplot(x=expert_res[0,:,j], y = expert_res[0,:,i],
                        color = c_exp, ax = fig1[j,i])
    [fig1[i,j].get_xaxis().set_visible(False) for i,j in product(range(5),range(5))]
    [fig1[i,j].get_yaxis().set_visible(False) for i,j in product(range(5),range(5))]
    [fig1[i,j].spines[['right', 'top']].set_visible(False) for i,j in product(range(5),range(5))]
    subfigs[1].suptitle("Marginal priors")
    subfigs[2].suptitle("Joint prior")
    
    subfigs[3].suptitle("elicited statistics")
    [fig2[x].axline((model_elicit["quantiles_meanperday"][0,0,x],
                    model_elicit["quantiles_meanperday"][0,0,x]), 
                    slope = 1, color = "black", linestyle = "dashed") for x in range(5)]
    for x in range(5):
        [sns.scatterplot(
            x = model_elicit["quantiles_meanperday"][b,:,x],  
            y = expert_elicit["quantiles_meanperday"][0,:,x],
            ax = fig2[x],
            color = c_mod, alpha = 0.2,
            s=50
            ) for b in range(100)]
        fig2[x].set_yticklabels([])
        fig2[x].set_title(f"day {x}")
    fig2[0].set_xlabel("training data")
    fig2[0].set_ylabel("expert data")
    
    for i, elicit in enumerate(["R2day0","R2day9"]):
        [sns.histplot(model_preds[elicit][b,...], bins = 20, 
                      color = c_mod, stat="density",
                      alpha = 0.2, ax = fig3[i], edgecolor = None) for b in range(100)]
        # if i == 2:
        sns.kdeplot(expert_preds[elicit][0,...], ax = fig3[i], 
                    color = c_exp, lw = 3)
        # else:
        # fig3[i].axvline(expert_elicit[elicit][0,...], color = c_exp, 
        #                 lw = 3, linestyle="dashed")
        fig3[i].get_yaxis().set_visible(False) 
    sns.boxplot(model_elicit["moments.sd_sigma"], orient = "h", ax = fig3[2],
                color = c_mod)
    sns.boxplot(model_elicit["moments.mean_sigma"], orient = "h", ax = fig3[3],
                color = c_mod)
    fig3[2].axvline(expert_elicit["moments.sd_sigma"][0], linestyle = "dashed", 
                    color = c_exp)
    fig3[3].axvline(expert_elicit["moments.mean_sigma"][0], linestyle = "dashed", 
                    color = c_exp)
   
    [fig3[i].set_title(title) for i, title in enumerate(["R2 day0", "R2 day9", 
                                                         "sigma (scale)", "sigma (loc)"])]
    
    fig.suptitle(title)
    plt.savefig(path+file+".png")
    # write results table if not done already:
    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    write_results(path+file, global_dict)



def plot_results_overview_param(path, file, title):
    true_vals = (250.4, 7.27, 30.26, 4.82, 33., 23., 200., 8.)
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    expert_res = pd.read_pickle(path+file+"/expert/model_simulations.pkl")["prior_samples"]
    model_samples = pd.read_pickle(path+file+"/model_simulations.pkl")["prior_samples"]
    
    model_elicit = pd.read_pickle(path+file+"/elicited_statistics.pkl")
    expert_elicit = pd.read_pickle(path+file+"/expert/elicited_statistics.pkl")
    
    expert_loc = tf.reduce_mean(expert_res, (0,1))
    expert_scale = tf.reduce_mean(tf.math.reduce_std(expert_res, 1), 0)
    
    total_loss = tf.stack(final_res["loss"], -1)
    component_loss = tf.stack(final_res["loss_component"], -1)
    hyperparams =  tf.stack([tf.exp(final_res["hyperparameter"][key]) for key in final_res["hyperparameter"].keys()],0)
 
    c_exp = "#03396c"
    c_mod = "#45b5fd"
    
    fig = plt.figure(layout = "constrained", figsize = (8,12))
    subfigs = fig.subfigures(5,1, height_ratios = [0.7,0.7,1,0.5,0.5])
    
    fig0 = subfigs[0].subplots(1,2, sharex = True)
    fig01 = subfigs[1].subplots(1,3, sharex = True)
    fig1 = subfigs[2].subplots(5,5)
    fig2 = subfigs[3].subplots(1,5)
    fig3 = subfigs[4].subplots(1,3)
    
    fig0[0].plot(range(len(total_loss)), total_loss)
    [fig0[1].plot(range(len(total_loss)), component_loss[i,:], label = f"loss {i}") for i in range(component_loss.shape[0])]
    [fig01[0].axhline(true_vals[i], linestyle = "dashed", color = "black") for i in [0,7]]
    [fig01[1].axhline(true_vals[i], linestyle = "dashed", color = "black") for i in [2,4,5]]
    [fig01[2].axhline(true_vals[i], linestyle = "dashed", color = "black") for i in [1,3,6]]
    [fig01[0].plot(range(len(total_loss)), hyperparams[i,:], lw =3, label = list(final_res["hyperparameter"].keys())[i]) for i in [0,7]]
    [fig01[1].plot(range(len(total_loss)), hyperparams[i,:], lw =3, label = list(final_res["hyperparameter"].keys())[i]) for i in [2,4,5]]
    [fig01[2].plot(range(len(total_loss)), hyperparams[i,:], lw =3, label = list(final_res["hyperparameter"].keys())[i]) for i in [1,3,6]]
    [fig01[i].legend(loc="upper left", handlelength=0.4, fontsize="small", ncol=1,
                    columnspacing=0.5) for i in range(3)]
    fig0[1].legend(loc="upper left", handlelength=0.4, fontsize="small", ncol=1,
                    columnspacing=0.5) 
    
    [fig0[i].set_title(t) for i,t in enumerate(["total loss", "loss per component"])]
    #[fig01[i].set_title(t) for i,t in enumerate([r"Normal $\mu_k$", r"Normal $\sigma_k$", r"Gamma $\alpha$,$\beta$"])] 
    
    [fig0[i].set_xlabel("epochs") for i in range(2)]
    [fig01[i].set_xlabel("epochs") for i in range(3)]
    [fig0[i].set_yscale("log") for i in range(2)]
    [fig01[i].set_yscale("log") for i in [1,2]] 
    
    for i in range(5):
        [sns.kdeplot(model_samples[b,:,i], color = c_mod, alpha = 0.2, ax = fig1[i,i]) for b in range(100)]
        sns.kdeplot(expert_res[0,:,i], color = c_exp, linestyle = "dashed", ax = fig1[i,i])
    for i,j in zip([0,0,0,0,1,1,1,2,2,3], [1,2,3,4,2,3,4,3,4,4]):
        sns.scatterplot(x=model_samples[0,:,i], y = model_samples[0,:,j], color = c_mod,
                        marker="x", ax = fig1[i,j])
        sns.scatterplot(x=expert_res[0,:,i], y = expert_res[0,:,j], marker="+", 
                        color = c_exp, ax = fig1[i,j])
        sns.kdeplot(x=model_samples[0,:,j], y = model_samples[0,:,i], color = c_mod,
                        ax = fig1[j,i])
        sns.kdeplot(x=expert_res[0,:,j], y = expert_res[0,:,i],
                        color = c_exp, ax = fig1[j,i])
    [fig1[i,j].get_xaxis().set_visible(False) for i,j in product(range(5),range(5))]
    [fig1[i,j].get_yaxis().set_visible(False) for i,j in product(range(5),range(5))]
    [fig1[i,j].spines[['right', 'top']].set_visible(False) for i,j in product(range(5),range(5))]
    subfigs[1].suptitle("Marginal priors")
    subfigs[2].suptitle("Joint prior")
    
    subfigs[3].suptitle("elicited statistics")
    [fig2[x].axline((model_elicit["quantiles_meanperday"][0,0,x],
                    model_elicit["quantiles_meanperday"][0,0,x]), 
                    slope = 1, color = "black", linestyle = "dashed") for x in range(5)]
    for x in range(5):
        [sns.scatterplot(
            x = model_elicit["quantiles_meanperday"][b,:,x],  
            y = expert_elicit["quantiles_meanperday"][0,:,x],
            ax = fig2[x],
            color = c_mod, alpha = 0.2,
            s=50
            ) for b in range(100)]
        fig2[x].set_yticklabels([])
        fig2[x].set_title(f"day {x}")
    fig2[0].set_xlabel("training data")
    fig2[0].set_ylabel("expert data")
    
    for i, elicit in enumerate(["histogram_R2day0","histogram_R2day9", "histogram_sigma"]):
        [sns.histplot(model_elicit[elicit][b,...], bins = 20, 
                      color = c_mod, stat="density",
                      alpha = 0.2, ax = fig3[i], edgecolor = None) for b in range(100)]
        # if i == 2:
        sns.kdeplot(expert_elicit[elicit][0,...], ax = fig3[i], 
                    color = c_exp, lw = 3)
        #else:
        # fig3[i].axvline(expert_elicit[elicit][0,...], color = c_exp, 
        #                 lw = 3, linestyle="dashed")
        fig3[i].get_yaxis().set_visible(False) 
    [fig3[i].set_title(title) for i, title in enumerate(["mu0 sd", "mu9 sd", "sigma"])]
    
    fig.suptitle(title)
    plt.savefig(path+file+".png")
    # write results table if not done already:
    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    write_results(global_dict)

