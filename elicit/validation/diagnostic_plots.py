import os
#os.chdir('/home/flob/prior_elicitation')
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
import seaborn as sns

########### Convergence diagnostics ###########
## % loss
def plot_loss(global_dict, save_fig = False):
    res_dict = pd.read_pickle(global_dict["output_path"]["data"]+"/final_results.pkl")

    loss = tf.stack(res_dict["loss"], 0)
    _, axs = plt.subplots(1,1, constrained_layout = True, figsize = (4,3))
    plt.plot(range(global_dict["epochs"]), loss, lw = 2)
    plt.title("loss function", ha = "left", x = 0)
    axs.set_ylabel("loss")
    axs.set_xlabel("epochs")
    if save_fig:
        path_to_file = global_dict["output_path"]["plots"]+'/loss.png'
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        plt.savefig(path_to_file)
    else:
        plt.show()

## % gradients
def plot_gradients(global_dict, save_fig = False):
    res_dict = pd.read_pickle(global_dict["output_path"]["data"]+"/final_results.pkl")

    gradients = tf.stack(res_dict["gradients"], 0)
    _, axs = plt.subplots(1,1, constrained_layout = True, figsize = (4,3))
    [sns.scatterplot(x=range(global_dict["epochs"]), y=gradients[:,i], 
                    marker="+", label = fr"$\lambda_{{{i}}}$") for i in range(gradients.shape[-1])]
    plt.legend(labelspacing = 0.2, columnspacing = 1, ncols = 2, handletextpad = 0.3, fontsize = "small")
    axs.set_xlabel("epochs")
    axs.set_ylabel("gradient")
    plt.title("gradients", ha = "left", x = 0)
    if save_fig:
        path_to_file = global_dict["output_path"]["plots"]+'/gradients.png'
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        plt.savefig(path_to_file)
    else:
        plt.show()

def plot_convergence(global_dict, save_fig = False):
    res_dict = pd.read_pickle(global_dict["output_path"]["data"]+"/final_results.pkl")
    num_params = len(global_dict["model_params"]["name"])*2
    ## % hyperparameter values
    truth = pd.read_pickle(global_dict["output_path"]["data"]+"/expert/prior_samples.pkl")
    true_means = tf.reduce_mean(truth, (0,1))
    true_sds = tf.reduce_mean(tf.math.reduce_std(truth, 1), 0)

    hyperparams = tf.stack(list(res_dict["hyperparameter"].values()), -1)
    if global_dict["method"] == "deep_prior":
        hyperparams_prep = tf.concat([hyperparams[:,:,0], hyperparams[:,:,1]],-1)
    else:
        hyperparams_prep = tf.concat([hyperparams[:,0::2], abs(hyperparams[:,1::2])], -1)
    
    _, axs = plt.subplots(1,2, constrained_layout = True, figsize = (6,3), sharex=True)
    # learned values
    [axs[0].plot(range(hyperparams_prep.shape[0]), hyperparams_prep[:,i], lw = 2, label = fr"$\lambda_{{{i}}}$") for i in range(int(num_params/2-1))]
    [axs[1].plot(range(hyperparams_prep.shape[0]), hyperparams_prep[:,i], lw = 2, label = fr"$\lambda_{{{i}}}$") for i in range(int(num_params/2),int(num_params-1))]
    # expert
    [axs[0].axhline(true_means[i], linestyle = "dashed", color = "black", lw = 1) for i in range(int(num_params/2))]
    [axs[1].axhline(true_sds[i], linestyle = "dashed", color = "black", lw = 1) for i in range(int(num_params/2))]
    # legend and axes
    [axs[i].legend(labelspacing = 0.2, columnspacing = 1, ncols = 2, handletextpad = 0.3, 
                fontsize = "small", handlelength = 0.5) for i in range(2)]
    [axs[i].set_xlabel("epochs") for i in range(2)]
    axs[0].set_ylabel(r"$\lambda$")
    plt.suptitle("convergence of hyperparameters", ha = "left", x = 0.1)
    if save_fig:
        path_to_file = global_dict["output_path"]["plots"]+'/convergence.png'
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        plt.savefig(path_to_file)
    else:
        plt.show()



########### Learned prior distributions: Marginals ###########
def plot_marginal_priors(global_dict, sims = 100, save_fig = False):
    truth = pd.read_pickle(global_dict["output_path"]["data"]+"/expert/prior_samples.pkl")
    learned = pd.read_pickle(global_dict["output_path"]["data"]+"/prior_samples.pkl")

    _, axs = plt.subplots(1,1, constrained_layout = True, figsize = (4,3))
    for b in range(sims):
        [sns.kdeplot(learned[b,:,i], lw = 2, alpha = 0.2, color = "orange", 
                     ax = axs) for i in range(truth.shape[-1]-1)]
    [sns.kdeplot(truth[0,:,i], lw = 2, color = "black", linestyle = "dashed", 
                 ax = axs) for i in range(truth.shape[-1]-1)]
    axs.set_xlabel(r"model parameters $\beta$")
    axs.set_ylabel("density")
    plt.suptitle("learned prior distributions", ha = "left", x = 0.15)
    if save_fig:
        path_to_file = global_dict["output_path"]["plots"]+'/marginal_prior.png'
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        plt.savefig(path_to_file)
    else:
        plt.show()

########### Learned prior distributions: Joint ###########

def plot_joint_prior(global_dict, save_fig = False):
    truth = pd.read_pickle(global_dict["output_path"]["data"]+"/expert/prior_samples.pkl")
    learned = pd.read_pickle(global_dict["output_path"]["data"]+"/prior_samples.pkl")

    num_params = len(global_dict["model_params"]["name"])
    # prepare data as pandas data frame
    def prep_data(dat, model):
        prepare_data = pd.DataFrame(dat, columns = [fr"$\theta_{{{i}}}$" for i in range(num_params)])
        prepare_data.insert(2, "model", [model]*global_dict["rep"])
        return prepare_data
    frames = [prep_data(truth[0,:,:], "expert"),
              prep_data(learned[0,:,:], "training")]
    df = pd.concat(frames)
    # plot data 
    g = sns.pairplot(df, hue="model", plot_kws=dict(marker="+", linewidth=1), 
                    height=2, aspect = 1.)
    g.map_lower(sns.kdeplot)
    labels = g._legend_data.keys()
    sns.move_legend(g, "upper center",bbox_to_anchor=(0.45, 1.1), 
                    labels=labels, ncol=num_params, title=None, frameon=True)
    if save_fig:
        path_to_file = global_dict["output_path"]["plots"]+'/joint_prior.png'
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        plt.savefig(path_to_file)
    else:
        plt.show()

########### Elicited statistics ###########
def plot_elicited_statistics(global_dict, sims = 100, selected_obs = None,
                             save_fig = False):
    learned_elicits = pd.read_pickle(global_dict["output_path"]["data"]+"/elicited_statistics.pkl")
    true_elicits = pd.read_pickle(global_dict["output_path"]["data"]+"/expert/elicited_statistics.pkl")

    keys_elicit = list(learned_elicits.keys())
    methods = [keys_elicit[i].split(sep="_")[0] for i in range(len(keys_elicit))]
    for key, meth in zip(keys_elicit, methods):
        training_data = learned_elicits[key]
        expert_data = true_elicits[key]

        if meth == "histogram":
            if tf.rank(training_data) == 1:
                _, ax = plt.subplots(1,1, constrained_layout = True, figsize = (3,3))
                sns.histplot(training_data, bins = 20, color = "orange", stat="density",
                            ax = ax, edgecolor = None) 
                ax.axvline(x = expert_data, color = "black", linestyle="dashed")
                plt.suptitle("elicited statistics - histogram", size = "medium")
             
            else:
                if training_data.shape[-1] == global_dict["rep"]:
                    _, ax = plt.subplots(1,1, constrained_layout = True, figsize = (3,3))
                    [sns.histplot(training_data[b,...], bins = 20, color = "orange", stat="density",
                                alpha = 0.2, ax = ax, edgecolor = None) for b in range(sims)]
                    sns.kdeplot(expert_data[0,...], ax = ax, color = "black")
                    plt.suptitle("elicited statistics - histogram")
                
                else:
                    groups = training_data.shape[-1]
                    _, ax = plt.subplots(1,groups, constrained_layout = True, figsize = (int(groups*2),3))
                    
                    for gr in range(groups):
                        [sns.histplot(training_data[b,...,gr], bins = 20, color = "orange", stat="density",
                                    alpha = 0.2, ax = ax[gr], edgecolor = None) for b in range(sims)]
                        sns.kdeplot(expert_data[0,...,gr], ax = ax[gr], color = "black")
                    [ax[i].set_title(f"y_pred {i}") for i in range(groups)]
                    plt.suptitle("elicited statistics - histogram")
                
            if save_fig:
                path_to_file = global_dict["output_path"]["plots"]+'/elicited_statistics_hist.png'
                os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
                plt.savefig(path_to_file)
            else:
                plt.show()
        
        if meth == "quantiles":
            groups = training_data.shape[-1]
            _, axs = plt.subplots(1,groups, constrained_layout=True, figsize = (int(groups*2),2))
            [axs[x].axline((training_data[0,0,x],
                            training_data[0,0,x]), 
                            slope = 1, color = "black", linestyle = "dashed") for x in range(groups)]
            for x in range(groups):
                [sns.scatterplot(
                    x = training_data[b,:,x],  
                    y = expert_data[0,:,x],
                    ax = axs[x],
                    color = "orange", alpha = 0.2,
                    s=50
                    ) for b in range(sims)]
            [axs[x].set_yticklabels([]) for x in range(groups)]
            axs[0].set_xlabel("training data")
            axs[0].set_ylabel("expert data")
            #[axs[i].set_title(fr"$y_{{n,{obs}}}$") for i, obs in enumerate(selected_obs)]
            plt.suptitle("elicited statistics - quantile-based")
            if save_fig:
                path_to_file = global_dict["output_path"]["plots"]+'/elicited_statistics_quant.png'
                os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
                plt.savefig(path_to_file)
            else:
                plt.show()