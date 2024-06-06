import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import os 

from functions.user_interface.write_results import write_results

path = "elicit/simulations/results/data/deep_prior/"

files_list = []
for name in os.listdir(path):
 if name.startswith("pois_347"):
     files_list.append(name) 

for file in files_list[12:18]:
    file = "pois_test_34737332"
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
    
    fig = plt.figure(layout = "constrained", figsize = (8,12))
    subfigs = fig.subfigures(4,1, height_ratios = [1,1,0.5,0.5])
    
    fig0 = subfigs[0].subplots(2,2, sharex = True)
    fig1 = subfigs[1].subplots(4,4)
    fig2 = subfigs[2].subplots(1,6)
    fig3 = subfigs[3].subplots(1,3)
    
    fig0[0,0].plot(range(len(total_loss)), total_loss)
    [fig0[0,1].plot(range(len(total_loss)), component_loss[i,:]) for i in range(component_loss.shape[0])]
    [fig0[1,0].axhline(expert_loc[i], linestyle = "dashed", color = "black") for i in range(expert_loc.shape[0])]
    [fig0[1,1].axhline(expert_scale[i], linestyle = "dashed", color = "black") for i in range(expert_loc.shape[0])]
    [fig0[1,0].plot(range(len(total_loss)), hyperparam_loc[i,32:], lw =3) for i in range(hyperparam_loc.shape[0])]
    [fig0[1,1].plot(range(len(total_loss)), hyperparam_scale[i,32:], lw =3, label = fr"$\lambda_{i}$") for i in range(hyperparam_scale.shape[0])]
    fig0[1,1].legend(loc="upper right", handlelength=0.4, fontsize="small", ncol=4)
    
    [fig0[0,i].set_title(t) for i,t in enumerate(["total loss", "loss per component"])]
    [fig0[1,i].set_title(t) for i,t in enumerate(["marginal priors: mean", "marginal priors: sd"])] 
    
    [fig0[1,i].set_xlabel("epochs") for i in range(2)]
    [fig0[0,i].set_yscale("log") for i in range(2)]
    fig0[1,1].set_yscale("log") 
    
    for i in range(4):
        [sns.kdeplot(model_samples[b,:,i], color = c_mod, alpha = 0.2, ax = fig1[i,i]) for b in range(100)]
        sns.kdeplot(expert_res[0,:,i], color = c_exp, linestyle = "dashed", ax = fig1[i,i])
    for i,j in zip([0,0,0,1,1,2], [1,2,3,2,3,3]):
        sns.scatterplot(x=model_samples[0,:,i], y = model_samples[0,:,j], color = c_mod,
                        marker="x", ax = fig1[i,j])
        sns.scatterplot(x=expert_res[0,:,i], y = expert_res[0,:,j], marker="+", 
                        color = c_exp, ax = fig1[i,j])
        sns.kdeplot(x=model_samples[0,:,j], y = model_samples[0,:,i], color = c_mod,
                        ax = fig1[j,i])
        sns.kdeplot(x=expert_res[0,:,j], y = expert_res[0,:,i],
                        color = c_exp, ax = fig1[j,i])
    [fig1[i,j].get_xaxis().set_visible(False) for i,j in product(range(4),range(4))]
    [fig1[i,j].get_yaxis().set_visible(False) for i,j in product(range(4),range(4))]
    [fig1[i,j].spines[['right', 'top']].set_visible(False) for i,j in product(range(4),range(4))]
    subfigs[1].suptitle("Joint prior")
    
    for i in range(model_elicit["histogram_ypred"].shape[-1]):
        [sns.histplot(model_elicit["histogram_ypred"][b,...,i], bins = 20, 
                      color = c_mod, stat="density",
                    alpha = 0.2, ax = fig2[i], edgecolor = None) for b in range(100)]
        sns.kdeplot(expert_elicit["histogram_ypred"][0,...,i], ax = fig2[i], 
                    color = c_exp, lw = 3)
        fig2[i].set_title(f"y_pred {i}") 
        fig2[i].get_xaxis().set_visible(False) 
        fig2[i].get_yaxis().set_visible(False) 
    subfigs[2].suptitle("elicited statistics")
    
    [fig3[x].axline((model_elicit["quantiles_group_means"][0,0,x],
                    model_elicit["quantiles_group_means"][0,0,x]), 
                    slope = 1, color = "black", linestyle = "dashed") for x in range(3)]
    for x in range(3):
        [sns.scatterplot(
            x = model_elicit["quantiles_group_means"][b,:,x],  
            y = expert_elicit["quantiles_group_means"][0,:,x],
            ax = fig3[x],
            color = c_mod, alpha = 0.2,
            s=50
            ) for b in range(100)]
        fig3[x].set_yticklabels([])
        fig3[x].set_title(f"group {x}")
    fig3[0].set_xlabel("training data")
    fig3[0].set_ylabel("expert data")
    
    #title_names = file.split(sep="_")
    # if title_names[2] == "1":
    #     lr = 0.0001
    # if title_names[2] == "2":
    #     lr = "cos-decay-res(0.0001, 25)"
    # if title_names[2] == "3":
    #     lr = "cos-decay-res(0.0001, 50)"
    fig.suptitle("Poisson model 7 coupling layers")
    #fig.suptitle(f"seed: {title_names[1]}, learning rate: {title_names[2]}, activation: {title_names[3]}")
    plt.savefig(path+file+".png")


####
global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
write_results(global_dict)


for i in range(10):
     plt.plot(range(200), component_loss[i,:], label = f"loss {i}")
plt.legend()

write_results(global_dict)

pd.read_pickle(path+file+"/loss_components.pkl").keys()
pd.read_pickle(path+file+"/loss_per_component.pkl")
