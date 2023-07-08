# -*- coding: utf-8 -*-
"""
Plotting: Prior predictive, hyperparameter recovery, error analysis
"""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter


def plot_identifiability(var, vals):
    df = pd.DataFrame({
        "mu0": tf.constant(var[0][1], tf.float64)[-vals:,0],
        "mu1": tf.constant(var[0][1], tf.float64)[-vals:,1],
        "sigma0": tf.math.exp(var[1][1])[-vals:,0],
        "sigma1": tf.math.exp(var[1][1])[-vals:,1]
        })

    g = sns.PairGrid(df, diag_sharey=False)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot)

           
def plot_results_binomial(input_settings_model, input_settings_learning, 
                          input_settings_global, var, expert_samples, 
                          model_samples, xrge1, ylim, loss_format):
    
    mus = input_settings_model["hyperparameter"]["mus"]
    sigmas = input_settings_model["hyperparameter"]["sigmas"]
    X = input_settings_model["X"]
    idx = input_settings_model["X_idx"]
    epochs = input_settings_learning["epochs"]
    l_values = input_settings_global["l_values"]
    
    # define labels for plotting
    n_mus = ["mu0","mu1"]
    n_sigmas = ["sigma0","sigma1"]
    
    # define color codes for plotting
    # betas 
    col_betas = ["#2acaea", "#0a75ad"]
    
    if loss_format == "quantiles":
        df = pd.DataFrame({
            "X_idx": idx,
            "y_mean": tf.reduce_mean(model_samples["y_idx_loss"], (0,1)),
            "y_sd": tf.math.reduce_std(tf.reduce_mean(model_samples["y_idx_loss"], 0),0),
            "y_exp": tf.reduce_mean(expert_samples["y_idx_loss"], (0,1))
            })
    if loss_format == "moments":
        df = pd.DataFrame({
            "X_idx": idx,
            "y_mean": tf.reduce_mean(model_samples["y_idx_loss"]["mean_0"], 0),
            "y_sd": tf.math.reduce_mean(model_samples["y_idx_loss"]["sd_0"], 0),
            "y_exp": tf.reduce_mean(expert_samples["y_idx_loss"]["mean_0"], 0)
            })
    
    def avg_val(var, l_values, epochs, sd=True):
        if sd:
            val0, val1 = tf.reduce_mean(tf.exp(var[1][epochs-l_values:epochs]),0).numpy()
        else:
            val0, val1 = tf.reduce_mean(var[1][epochs-l_values:epochs],0).numpy()
        return val0, val1
    
    avg_mu0, avg_mu1 = avg_val(var[0], l_values, epochs, sd=False)
    avg_sigma0, avg_sigma1 = avg_val(var[1], l_values, epochs, sd=True)
    xrge = tf.cast(tf.range(xrge1[0],xrge1[1],0.001), tf.float32)
    pdf0 = tfd.Normal(avg_mu0, avg_sigma0).prob(xrge)
    pdf1 = tfd.Normal(avg_mu1, avg_sigma1).prob(xrge)
    pdf_true0 = tfd.Normal(mus[0], tf.exp(sigmas[0])).prob(xrge)
    pdf_true1 = tfd.Normal(mus[1], tf.exp(sigmas[1])).prob(xrge)
    
    muse = tf.stack([tf.stack((mus-var[0][1][i])/mus,0) for i in range(epochs)],-1)
    sigmase = tf.stack([tf.stack((sigmas-var[1][1][i])/sigmas,0) for i in range(epochs)],-1)
    
    fig, axs = plt.subplots(2,2, figsize=(14,7), 
                            gridspec_kw=dict(top=0.8, hspace=0.40))
    matplotlib.rcParams.update({'font.size': 12})
    if loss_format == "quantiles" or loss_format == "moments":
        [axs[0,0].axhline(y = df["y_exp"][i], color = 'red', linestyle = '--', 
                          linewidth=2) for i in range(len(idx))]
        sns.barplot(x=df["X_idx"], y=df["y_mean"], yerr=df["y_sd"],
                    ax=axs[0,0], palette="mako")
    if loss_format == "hist":
        sns.histplot(model_samples["y_idx_loss"][0,:,:], 
                     bins = model_samples["y_idx_loss"].shape[1], ax=axs[0,0])
        # sns.histplot(expert_samples["y_idx_loss"][0,:,:], 
        #              bins = model_samples["y_idx_loss"].shape[1], ax=axs[0,0])
    axs[0,1].plot(xrge, pdf0, label = f"b0 ~ N({avg_mu0:.2f}, {avg_sigma0:.2f})",
                  color = col_betas[0], linewidth=3)
    axs[0,1].plot(xrge, pdf1, label = f"b1 ~ N({avg_mu1:.2f}, {avg_sigma1:.2f})",
                  color = col_betas[1], linewidth=3)
    axs[0,1].plot(xrge, pdf_true0, linestyle = "dotted", color ="black", linewidth=2)
    axs[0,1].plot(xrge, pdf_true1, linestyle = "dotted", color ="black", linewidth=2) 
    axs[1,0].axhline(y = 0., color = 'black', linestyle = '--', linewidth=2)
    axs[1,1].axhline(y = 0., color = 'black', linestyle = '--', linewidth=2)  
    for i in range(2):
        # mus
        axs[1,0].plot(tf.transpose(tf.squeeze(muse))[:,i],
                      color = col_betas[i],
                      label = n_mus[i],
                      linewidth = 3)
        # sigmas
        axs[1,1].plot(tf.transpose(tf.squeeze(sigmase))[:,i],
                      color = col_betas[i],
                      label = n_sigmas[i],
                      linewidth = 3)
        
    axs[0,1].legend(loc="upper left",handlelength=1)
    axs[1,0].legend(handlelength=1)
    axs[1,1].legend(handlelength=1)
    axs[1,0].set_xlabel("epochs")
    axs[1,1].set_xlabel("epochs")
    axs[0,0].set_xlabel("# detected axillary nodes", labelpad=1)
    axs[0,0].set_ylabel("# pats. died within 5 years", labelpad=1)
    axs[0,0].set_ylim(ylim[0],ylim[1])
    axs[0,1].set_title("Learned prior distributions",
                        fontweight = "bold", fontsize = "large", x=0.32)
    axs[0,0].set_title("Prior predictive data",
                        fontweight = "bold", fontsize = "large", x=0.25)
    axs[1,0].set_title("Absolute error between true and learned hyperparameter",
                        fontweight = "bold", fontsize = "large", x = 0.68)

def plot_results_lm(input_settings_model, input_settings_learning, var, 
                    input_settings_global, samples_d, xrange0, xrange1, 
                    fct_b_lvl,fct_a_lvl):
    
    mus = input_settings_model["hyperparameter"]["mus"]
    sigmas = input_settings_model["hyperparameter"]["sigmas"]
    lambda0 = input_settings_model["hyperparameter"]["lambda0"]
    
    epochs = input_settings_learning["epochs"]
    l_values = input_settings_global["l_values"]
    
    # define labels for plotting
    n_mus = ["mu0","mu1","mu2","mu3","mu4","mu5"]
    n_sigmas = ["sigma0","sigma1","sigma2","sigma3","sigma4","sigma5"]
    
    # define color codes for plotting
    # betas 
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    col_EnC = ["#0a75ad", "#3293f0", "#7ad0ed"]
    col_ReP = ["#ffaa00", "#e86e4d"]
    col_lambda = "#a6b7c6"
    
    
    #### prepare prior predictions
    df = pd.DataFrame({
         "repetition": ["new"]*fct_b_lvl + ["repeated"]*fct_b_lvl,
         "encoding depth": ["C1 deep", "C2 standard", "C3 shallow"]*fct_a_lvl,
         "means": tf.reduce_mean(samples_d["obs_joints"],(0,1)).numpy(),
         "sd_means": tf.reduce_mean(tf.math.reduce_std(samples_d["obs_joints"],1),0).numpy()
         })
     
    vals = df.pivot(index="repetition", columns="encoding depth",values="means")
    yerr = df.pivot(index="repetition", columns="encoding depth",values="sd_means")
     
    # prepare convergence plot
    def avg_val_sig(var, l_values, epochs):
        return tf.reduce_mean(var[1][epochs-l_values:epochs]).numpy()
    def avg_val_contr(var,idx, l_values, epochs, sd=False):
        if sd:
            res = tf.exp(tf.reduce_mean([var[2][1][epochs-l_values:epochs][i][idx] for i in range(l_values)])).numpy()
        else:
            res = tf.reduce_mean([var[1][1][epochs-l_values:epochs][i][idx] for i in range(l_values)]).numpy()
        return res
    def avg_val_inter(var,idx, l_values, epochs, sd=False):
        if sd:
            res = tf.exp(tf.reduce_mean([var[2][1][epochs-l_values:epochs][i][idx] for i in range(l_values)])).numpy()
        else:
            res = tf.reduce_mean([var[1][1][epochs-l_values:epochs][i][idx] for i in range(l_values)]).numpy()
        return res  
    
    avg_val_lambda0 = tf.exp(avg_val_sig(var[0], l_values, epochs))
    xrge0 = tf.cast(tf.range(xrange0[0],xrange0[1],0.001), tf.float32)
    pdf_sigma = tfd.Exponential(avg_val_lambda0).prob(xrge0)
    pdf_sigma_true = tfd.Exponential(tf.exp(lambda0)).prob(xrge0)
    pdf_sigma_m = tf.reduce_mean(tfd.Exponential(avg_val_lambda0).sample(1000))
    
    avg_betas = [avg_val_contr(var, i, l_values, epochs, sd = False) for i in range(6)]
    avg_betas_sd = [avg_val_contr(var, i, l_values, epochs, sd = True) for i in range(6)]
    
    xrge1 = tf.cast(tf.range(xrange1[0],xrange1[1],0.001), tf.float32)
    pdf_betas = [tfd.Normal(loc=avg_betas[i], scale=avg_betas_sd[i]).prob(xrge1) for i in range(6)]
    pdf_betas_true = [tfd.Normal(loc=mus[i], scale=tf.exp(sigmas[i])).prob(xrge1) for i in range(6)]
    
    # prepare error plot
    lambda0e = tf.stack([tf.stack(var[0][1][i]-tf.constant(lambda0,tf.float32),0) for i in range(epochs)],-1)
    muse = tf.stack([tf.stack(tf.constant(var[1][1][i], shape=(6,1))-tf.constant(mus,shape=(6,1),dtype=tf.float32),0) for i in range(epochs)],-1)
    sigmase = tf.stack([tf.stack(tf.constant(var[2][1][i], shape=(6,1))-tf.constant(sigmas,shape=(6,1),dtype=tf.float32),0) for i in range(epochs)],-1)
    
    
    # plot everything
    fig = plt.figure(figsize=(18,10))
    matplotlib.rcParams.update({'font.size': 12})
    subfigs = fig.subfigures(3, 1, wspace=0.07, hspace=0.0)
    axs0 = subfigs[0].subplots(1, 4, gridspec_kw=dict(top=0.8)) 
    axs1 = subfigs[1].subplots(1, 2, gridspec_kw=dict(top=0.8, width_ratios= [2.1,0.95], wspace=0.15))
    axs2 = subfigs[2].subplots(1, 3, gridspec_kw=dict(top=0.8), sharex=True)
    
    # joints
    vals.plot(kind="bar", yerr = yerr, rot=0, ax=axs0[0], color=col_EnC)
    
    # interaction plot
    sns.lineplot(data = df,  x = "repetition",  y = "means",  hue = "encoding depth", 
                 palette = col_EnC, ax = axs0[1])
    # errorbars
    for j,k,z in zip([0,0,0,1,1,1],range(6),[0,1,2]*2):
        axs0[1].vlines(x = j, ymin = df["means"][k]-df["sd_means"][k], 
                       ymax = df["means"][k]+df["sd_means"][k], color = col_EnC[z])

    # interaction plot: endpoints
    sns.scatterplot(data = df, x = "repetition",  y = "means",  hue = "encoding depth",
                    palette = col_EnC, ax = axs0[1])
    
    # marginal: repetition
    sns.barplot(x = ["new", "repeated"],  y = tf.reduce_mean(samples_d["ma"], (0,1)).numpy(), 
                yerr = tf.math.reduce_std(samples_d["ma"], (0,1)).numpy(), 
                ax = axs0[2], palette = col_ReP)
    
    # marginal: encoding depth
    sns.barplot(x = ["C1 deep", "C2 standard", "C3 shallow"], 
                y = tf.reduce_mean(samples_d["mb"], (0,1)).numpy(),
                yerr = tf.math.reduce_std(samples_d["mb"], (0,1)).numpy(), 
                ax = axs0[3], palette = col_EnC)
    
    # convergence: sigma ~ Exp
    ## learned
    axs1[1].plot(xrge0, pdf_sigma,  label = f"s ~ Exp({avg_val_lambda0:.2f})",
                 color = col_lambda, linewidth = 3)
    ## true
    axs1[1].plot(xrge0, pdf_sigma_true,  linestyle = "dotted", color ="black", 
                 linewidth = 2)
    ## mean value of sigma
    axs1[1].axvline(x = pdf_sigma_m,  linestyle="--",  color = col_lambda)
    
    # convergence: betas ~ N
    for i in range(6):
        # learned
        axs1[0].plot(xrge1, pdf_betas[i], color = col_betas[i],linewidth = 3, 
                     label = f"b{i} ~ N({avg_betas[i]:.2f},{avg_betas_sd[i]:.2f})")
        # true
        axs1[0].plot(xrge1, pdf_betas_true[i], linestyle="dotted",  color ="black",
                     linewidth = 2)
        
    # error: lambda0
    axs2[0].plot(lambda0e[0], color = col_lambda, linewidth = 3)
    # error: betas
    for i in range(6):
        # mus
        axs2[1].plot(tf.transpose(tf.squeeze(muse))[:,i],
                     color = col_betas[i], label = n_mus[i],linewidth = 3)  
        # sigmas
        axs2[2].plot(tf.transpose(tf.squeeze(sigmase))[:,i],
                     color = col_betas[i],label = n_sigmas[i],linewidth = 3)
        
    # add title to subfigures
    
    for (i,n),x in list(zip(enumerate([f"Prior predictive data (learned R2: {tf.reduce_mean(samples_d['R2']):.2f})",
                            "Learned prior distributions", "Absolute error between true and learned hyperparameter"]),
                     [0.25,0.21,0.33])):
        subfigs[i].suptitle(n, fontweight = "bold", fontsize = "x-large", x=x)
    
    axs0[1].get_legend().remove()
    axs0[0].legend(loc = "upper left")
    
    for i,n in enumerate(["joint","interactions","marginal 'repetition'",
                          "marginal 'encoding depth'"]):
        axs0[i].set_title(n)
    
    for i,n in enumerate(["beta coefficients",
                          f"random noise s (mean:{pdf_sigma_m:.2f})"]):
        axs1[i].legend()
        axs1[i].set_title(n)
    
    
    [axs2[k].legend(ncol=2,loc="upper right", labelspacing=0.2, handlelength=1) for k in range(1,3)]
    
    for i,n in enumerate(["lambda0", "mus", "sigmas"]):
        axs2[i].set_title(n)
    
    for i in range(3):
        axs2[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs2[i].set_xlabel("epochs")
        axs2[i].axhline(y = 0., color = 'black', linestyle = '--', linewidth=2)
    
    for i in range(4):
        axs0[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs0[i].set_xlabel(None)
        axs0[i].set_ylabel(None)
    
def plot_results_poisson(input_settings_model, input_settings_learning, 
                         input_settings_global, var, expert_samples, ylim):
    mus = input_settings_model["hyperparameter"]["mus"]
    sigmas = input_settings_model["hyperparameter"]["sigmas"]
    X_design = input_settings_model["X"]
    idx = input_settings_model["X_idx"]
    epochs = input_settings_learning["epochs"]
    l_values = input_settings_global["l_values"]
    
    # define labels for plotting
    n_mus = ["mu0","mu1", "mu2", "mu3"]
    n_sigmas = ["sigma0","sigma1", "sigma2", "sigma3"]
    
    # define color codes for plotting
    # betas 
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d"]
    
    X = pd.DataFrame(tf.squeeze(X_design)).sort_values(by=[2,3,1])
    
    # df0 = pd.DataFrame({
    #     "level": ["Democ", "Repub","Swing"],
    #     "mean counts": tf.stack([tf.reduce_mean(expert_samples["y_groups_loss"][:,:,0]),
    #               tf.reduce_mean(expert_samples["y_groups_loss"][:,:,1]),
    #               tf.reduce_mean(expert_samples["y_groups_loss"][:,:,2])],0),
    #     "err": tf.stack([tf.math.reduce_std(expert_samples["y_groups_loss"][:,:,0]),
    #               tf.math.reduce_std(expert_samples["y_groups_loss"][:,:,1]),
    #               tf.math.reduce_std(expert_samples["y_groups_loss"][:,:,2])],0)
    #     })
    
    # df = pd.DataFrame({
    #     "counts": tf.reduce_mean(expert_samples["y_obs_loss"], (0,1)),
    #     "continuous predictor": tf.constant(X[1]).numpy(),
    #     "err": tf.math.reduce_std(tf.reduce_mean(expert_samples["y_obs_loss"],1),0),
    #     "level": tf.repeat((1,2,3),(14,13,22))
    #     })
    
    def avg_val(var, l_values, epochs, sd=True):
        if sd:
            val0, val1, val2, val3 = tf.reduce_mean(tf.exp(var[1][epochs-l_values:epochs]),0).numpy()
        else:
            val0, val1, val2, val3 = tf.reduce_mean(var[1][epochs-l_values:epochs],0).numpy()
        return val0, val1, val2, val3
    
    avg_mu0, avg_mu1,avg_mu2, avg_mu3 = avg_val(var[0], l_values, epochs, sd=False)
    avg_sigma0, avg_sigma1,avg_sigma2, avg_sigma3 = avg_val(var[1], l_values, epochs, sd=True)
    xrge = tf.cast(tf.range(-3.,3.2,0.01), tf.float32)
    pdf0 = tfd.Normal(avg_mu0, avg_sigma0).prob(xrge)
    pdf1 = tfd.Normal(avg_mu1, avg_sigma1).prob(xrge)
    pdf2 = tfd.Normal(avg_mu2, avg_sigma2).prob(xrge)
    pdf3 = tfd.Normal(avg_mu3, avg_sigma3).prob(xrge)
    pdf_true0 = tfd.Normal(mus[0], tf.exp(sigmas[0])).prob(xrge)
    pdf_true1 = tfd.Normal(mus[1], tf.exp(sigmas[1])).prob(xrge)
    pdf_true2 = tfd.Normal(mus[2], tf.exp(sigmas[2])).prob(xrge)
    pdf_true3 = tfd.Normal(mus[3], tf.exp(sigmas[3])).prob(xrge)
    muse = tf.stack([tf.stack(var[0][1][i]-mus,0) for i in range(epochs)],-1)
    sigmase = tf.stack([tf.stack(var[1][1][i]-sigmas,0) for i in range(epochs)],-1)
    
    
    fig = plt.figure(figsize=(14,7))
    matplotlib.rcParams.update({'font.size': 12})
    subfigs = fig.subfigures(2, 1)
    axs0 = subfigs[0].subplots(1, 3, gridspec_kw=dict(top=0.85, hspace=0.4, width_ratios= [1.,1.,2.]))
    axs1 = subfigs[1].subplots(1, 2, gridspec_kw=dict(top=0.85, hspace=0.4))
    
    gr1_mean_e = tf.reduce_mean(expert_samples["y_groups_loss"][:,:,0])
    gr3_mean_e = tf.reduce_mean(expert_samples["y_groups_loss"][:,:,1])
    gr2_mean_e = tf.reduce_mean(expert_samples["y_groups_loss"][:,:,2])
    
    axs0[0].axhline(y=gr1_mean_e, linestyle="--", color="red")
    axs0[0].axhline(y=gr2_mean_e, linestyle="--", color="red")
    axs0[0].axhline(y=gr3_mean_e, linestyle="--", color="red")
    # sns.barplot(x=df0["level"],y=df0["mean counts"], yerr=df0["err"], ax=axs0[0],
    #             palette=["#0a75ad", "#3293f0", "#7ad0ed"])
    [axs0[1].plot(tf.constant(X[1])[idx[i]],tf.reduce_mean(expert_samples["y_obs_loss"],(0,1))[i], 
                  "o", color = "red", markersize = 10, alpha = 0.5, label='_nolegend_') for i in range(len(idx))]
    # sns.lineplot(x=df["continuous predictor"], y=df["counts"], 
    #             hue=df["level"],ax=axs0[1], palette=["#0a75ad", "#3293f0", "#7ad0ed"],
    #             linewidth=3)
    axs0[2].plot(xrge, pdf0, label = f"b0 ~ N({avg_mu0:.2f}, {avg_sigma0:.2f})",
                  color = col_betas[0], linewidth=3)
    axs0[2].plot(xrge, pdf1, label = f"b1 ~ N({avg_mu1:.2f}, {avg_sigma1:.2f})",
                  color = col_betas[1], linewidth=3)
    axs0[2].plot(xrge, pdf2, label = f"b2 ~ N({avg_mu2:.2f}, {avg_sigma2:.2f})",
                  color = col_betas[2], linewidth=3)
    axs0[2].plot(xrge, pdf3, label = f"b3 ~ N({avg_mu3:.2f}, {avg_sigma3:.2f})",
                  color = col_betas[3], linewidth=3)
    axs0[2].plot(xrge, pdf_true0, linestyle = "dotted", color ="black", linewidth=2)
    axs0[2].plot(xrge, pdf_true1, linestyle = "dotted", color ="black", linewidth=2) 
    axs0[2].plot(xrge, pdf_true2, linestyle = "dotted", color ="black", linewidth=2) 
    axs0[2].plot(xrge, pdf_true3, linestyle = "dotted", color ="black", linewidth=2) 
    axs1[0].axhline(y = 0., color = 'black', linestyle = '--', linewidth=2)
    axs1[1].axhline(y = 0., color = 'black', linestyle = '--', linewidth=2)  
    for i in range(4):
        # mus
        axs1[0].plot(tf.transpose(tf.squeeze(muse))[:,i],
                      color = col_betas[i],
                      label = n_mus[i],
                      linewidth = 3)
        # sigmas
        axs1[1].plot(tf.transpose(tf.squeeze(sigmase))[:,i],
                      color = col_betas[i],
                      label = n_sigmas[i],
                      linewidth = 3)  
    axs0[2].legend(loc="upper left",handlelength=1,labelspacing=0.2)
    axs0[1].legend(["Democrats","Republican","Swing"], loc="upper left",handlelength=1,
                    labelspacing=0.2, columnspacing=1.)
    axs1[0].legend(handlelength=1, loc = "upper right", ncol=2, columnspacing=1.)
    axs1[1].legend(handlelength=1, loc = "upper right", ncol=2, columnspacing=1.)
    axs1[0].set_xlabel("epochs")
    axs1[1].set_xlabel("epochs")
    axs0[0].set_xlabel("voting trend", labelpad=1)
    axs0[1].set_xlabel("demographic trend", labelpad=1)
    axs0[1].set_ylim((ylim[0], ylim[1]))
    axs0[2].set_title("Learned prior distributions",
                        fontweight = "bold", fontsize = "large", x=0.3)
    axs0[0].set_title("Prior predictive data",
                        fontweight = "bold", fontsize = "large", x=0.45)
    axs1[0].set_title("Absolute error between true and learned hyperparameter",
                        fontweight = "bold", fontsize = "large", x = 0.62)
    axs0[1].set_ylabel(None)
    axs0[2].set_ylabel("density")

def plot_results_negbinom(input_settings_model, input_settings_learning,
                          input_settings_global, var, expert_samples,
                          xrg,xrg1):
    
    mus = input_settings_model["hyperparameter"]["mus"]
    sigmas = input_settings_model["hyperparameter"]["sigmas"]
    lambda0 = input_settings_model["hyperparameter"]["lambda0"]
    
    epochs = input_settings_learning["epochs"]
    l_values = input_settings_global["l_values"]
    X_design = input_settings_model["X"]
    idx = input_settings_model["X_idx"]
    
    # define labels for plotting
    n_mus = ["mu0","mu1", "mu2", "mu3"]
    n_sigmas = ["sigma0","sigma1", "sigma2", "sigma3"]

    # define color codes for plotting
    # betas 
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d"]
    col_lambda = "#a6b7c6"

    X = pd.DataFrame(tf.squeeze(X_design)).sort_values(by=[2,3,1])

    # df0 = pd.DataFrame({
    #     "level": ["Dem", "Rep","Swing"],
    #     "mean counts": tf.stack([tf.reduce_mean(expert_samples["marginal_gr1"]),
    #               tf.reduce_mean(expert_samples["marginal_gr3"]),
    #               tf.reduce_mean(expert_samples["marginal_gr2"])],0),
    #     "err": tf.stack([tf.reduce_mean(expert_samples["marginal_gr1_sd"]),
    #               tf.reduce_mean(expert_samples["marginal_gr3_sd"]),
    #               tf.reduce_mean(expert_samples["marginal_gr2_sd"])],0)
    #     })

    df = pd.DataFrame({
        "counts": tf.reduce_mean(expert_samples["y_obs"], (0,1)),
        "continuous predictor": tf.constant(X[1]).numpy(),
        "err": tf.math.reduce_std(tf.reduce_mean(expert_samples["y_obs"],1),0),
        "level": tf.repeat((1,2,3),(14,13,22))
        })

    def avg_val(var, l_values, epochs, sd, lambda0, mu):
        if lambda0:
            val0 = tf.reduce_mean(tf.exp(var[1][epochs-l_values:epochs]),0).numpy()
            return val0
        if sd:
            val0, val1, val2, val3 = tf.reduce_mean(tf.exp(var[1][epochs-l_values:epochs]),0).numpy()
            return val0, val1, val2, val3
        if mu:
            val0, val1, val2, val3 = tf.reduce_mean(var[1][epochs-l_values:epochs],0).numpy()
            return val0, val1, val2, val3
        

    avg_mu0, avg_mu1,avg_mu2, avg_mu3 = avg_val(var[1], l_values, epochs, sd=False, lambda0=False, mu=True)
    avg_sigma0, avg_sigma1,avg_sigma2, avg_sigma3 = avg_val(var[2], l_values, epochs, sd=True, lambda0=False, mu=False)
    avg_lambda0 = avg_val(var[0], l_values, epochs, sd=False, lambda0=True, mu=False)
    xrge = tf.cast(tf.range(xrg[0],xrg[1],0.01), tf.float32)
    xrge1 = tf.cast(tf.range(xrg1[0],xrg1[1],0.01), tf.float32)
    pdf0 = tfd.Normal(avg_mu0, avg_sigma0).prob(xrge)
    pdf1 = tfd.Normal(avg_mu1, avg_sigma1).prob(xrge)
    pdf2 = tfd.Normal(avg_mu2, avg_sigma2).prob(xrge)
    pdf3 = tfd.Normal(avg_mu3, avg_sigma3).prob(xrge)
    pdf4 = tfd.Exponential(avg_lambda0).prob(xrge1)
    pdf_true0 = tfd.Normal(mus[0], tf.exp(sigmas[0])).prob(xrge)
    pdf_true1 = tfd.Normal(mus[1], tf.exp(sigmas[1])).prob(xrge)
    pdf_true2 = tfd.Normal(mus[2], tf.exp(sigmas[2])).prob(xrge)
    pdf_true3 = tfd.Normal(mus[3], tf.exp(sigmas[3])).prob(xrge)
    pdf_true4 = tfd.Exponential(tf.exp(lambda0)).prob(xrge1)
    muse = tf.stack([tf.stack(var[1][1][i]-mus,0) for i in range(epochs)],-1)
    sigmase = tf.stack([tf.stack(var[2][1][i]-sigmas,0) for i in range(epochs)],-1)
    lambda0e = tf.stack([tf.stack(var[0][1][i]-lambda0,0) for i in range(epochs)],-1)

    fig = plt.figure(figsize=(17,7))
    matplotlib.rcParams.update({'font.size': 12})
    subfigs = fig.subfigures(2, 1)
    axs0 = subfigs[0].subplots(1, 3, gridspec_kw=dict(top=0.85, hspace=0.4, width_ratios=[1.,1.2,0.8]))
    axs1 = subfigs[1].subplots(1, 4, gridspec_kw=dict(top=0.85, hspace=0.4, width_ratios=[1.5,1.,1.,1.]))

    # gr1_mean_e = tf.reduce_mean(expert_samples["marginal_gr1"])
    # gr3_mean_e = tf.reduce_mean(expert_samples["marginal_gr2"])
    # gr2_mean_e = tf.reduce_mean(expert_samples["marginal_gr3"])

    # axs0[0].axhline(y=gr1_mean_e, linestyle="--", color="red")
    # axs0[0].axhline(y=gr3_mean_e, linestyle="--", color="red")
    # axs0[0].axhline(y=gr2_mean_e, linestyle="--", color="red")
    # sns.barplot(x=df0["level"],y=df0["mean counts"], yerr=df0["err"], ax=axs0[0],
    #             palette=["#0a75ad", "#3293f0", "#7ad0ed"])
    # axs1[0].plot(tf.constant(X[1])[idx[0]],tf.reduce_mean(expert_samples["X_obs_m"],0)[0], 
    #               "o", color = "red", markersize = 10, alpha = 0.5, label='_nolegend_')
    # axs1[0].plot(tf.constant(X[1])[idx[1]],tf.reduce_mean(expert_samples["X_obs_m"],0)[1], 
    #               "o", color = "red", markersize = 10, alpha = 0.5, label='_nolegend_')
    # axs1[0].plot(tf.constant(X[1])[idx[2]],tf.reduce_mean(expert_samples["X_obs_m"],0)[2], 
    #               "o", color = "red", markersize = 10, alpha = 0.5, label='_nolegend_')
    # sns.lineplot(x=df["continuous predictor"], y=df["counts"], 
    #             hue=df["level"],ax=axs1[0], palette=["#0a75ad", "#3293f0", "#7ad0ed"],
    #             linewidth=3)
    axs0[1].plot(xrge, pdf0, label = f"b0 ~ N({avg_mu0:.2f}, {avg_sigma0:.2f})",
                  color = col_betas[0], linewidth=3)
    axs0[1].plot(xrge, pdf1, label = f"b1 ~ N({avg_mu1:.2f}, {avg_sigma1:.2f})",
                  color = col_betas[1], linewidth=3)
    axs0[1].plot(xrge, pdf2, label = f"b2 ~ N({avg_mu2:.2f}, {avg_sigma2:.2f})",
                  color = col_betas[2], linewidth=3)
    axs0[1].plot(xrge, pdf3, label = f"b3 ~ N({avg_mu3:.2f}, {avg_sigma3:.2f})",
                  color = col_betas[3], linewidth=3)
    axs0[2].plot(xrge1, pdf4, label = f"lambda ~ Exp({avg_lambda0:.2f})",
                  color = col_lambda, linewidth=3)
    axs0[1].plot(xrge, pdf_true0, linestyle = "dotted", color ="black", linewidth=2)
    axs0[1].plot(xrge, pdf_true1, linestyle = "dotted", color ="black", linewidth=2) 
    axs0[1].plot(xrge, pdf_true2, linestyle = "dotted", color ="black", linewidth=2) 
    axs0[1].plot(xrge, pdf_true3, linestyle = "dotted", color ="black", linewidth=2) 
    axs0[2].plot(xrge1, pdf_true4, linestyle = "dotted", color ="black", linewidth=2) 
    axs1[1].axhline(y = 0., color = 'black', linestyle = '--', linewidth=2)
    axs1[2].axhline(y = 0., color = 'black', linestyle = '--', linewidth=2)  
    axs1[3].axhline(y = 0., color = 'black', linestyle = '--', linewidth=2)  
    
    for i in range(4):
        # mus
        axs1[1].plot(tf.transpose(tf.squeeze(muse))[:,i],
                      color = col_betas[i],
                      label = n_mus[i],
                      linewidth = 3)
        # sigmas
        axs1[2].plot(tf.transpose(tf.squeeze(sigmase))[:,i],
                      color = col_betas[i],
                      label = n_sigmas[i],
                      linewidth = 3) 
    # lambda
    axs1[3].plot(tf.transpose(tf.squeeze(lambda0e)),
                  color = col_lambda,
                  label = "lambda0",
                  linewidth = 3) 
    axs0[2].legend(loc="upper right",handlelength=1,labelspacing=0.2)
    axs1[0].legend(["Democrats","Republicans","Swing"], loc="upper left",handlelength=1,
                    labelspacing=0.2, columnspacing=1.)
    axs0[1].legend(handlelength=1, loc = "upper right", columnspacing=1.)
    axs1[3].legend(handlelength=1, loc = "upper right")
    axs1[1].legend(handlelength=0.5, loc = "upper right", ncol=2, columnspacing=0.5, labelspacing=0.2)
    axs1[2].legend(handlelength=0.5, loc = "lower right", ncol=2, columnspacing=0.5, labelspacing=0.2)
    axs1[1].set_xlabel("epochs")
    axs1[2].set_xlabel("epochs")
    axs1[3].set_xlabel("epochs")
    axs0[0].set_xlabel("voting trend", labelpad=1)
    axs1[0].set_xlabel("demographic trend", labelpad=1)
    axs0[1].set_title("Learned prior distributions",
                        fontweight = "bold", fontsize = "large", x=0.4)
    axs0[0].set_title("Prior predictive data",
                        fontweight = "bold", fontsize = "large", x=0.3)
    axs1[1].set_title("Absolute error between true and learned hyperparameter",
                        fontweight = "bold", fontsize = "large", x = 1.3)
    axs0[1].set_ylabel(None)
    axs0[1].set_ylabel("density")

def plot_results_mlm(var, samples_d, samples_e, epochs, N_days,
                     lambda0, mu_0, mu_1, sigma_0, sigma_1, Z_days,
                     sigma_tau0, sigma_tau1,idx, model,
                     xrange0=[0.0, 100.0],xrange1=[100.0, 310.0],
                     xrange2=[-2.0, 20.0], xrange3=[0.0, 50.0],l_values=30):
    
    # define color codes for plotting
    # betas 
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    col_EnC = ["#0a75ad", "#3293f0", "#7ad0ed"]
    col_ReP = ["#ffaa00", "#e86e4d"]
    col_lambda = "#a6b7c6"
    
    
    #### prepare prior prediction
    if model == "log":
        df = pd.DataFrame({
             "days": Z_days,
             "means": tf.reduce_mean(samples_d["y_m"],(0,1,2)).numpy(),
             "sd_means": tf.reduce_mean(tf.math.reduce_std(tf.reduce_mean(samples_d["mu_m"],2),1),0).numpy()
             })
    if model == "normal":
        df = pd.DataFrame({
             "days": Z_days,
             "means": tf.reduce_mean(samples_d["y_m"],(0,1,2)).numpy(),
             "sd_means": tf.reduce_mean(tf.math.reduce_std(tf.reduce_mean(samples_d["mu_m"],2),1),0).numpy()
             })
     
    # prepare convergence plot
    def avg_val(var, l_values, epochs):
        return tf.reduce_mean(var[1][epochs-l_values:epochs]).numpy() 
    
    avg_val_lambda0 = tf.exp(avg_val(var[0], l_values, epochs))
    xrge0 = tf.cast(tf.range(xrange0[0],xrange0[1],0.001), tf.float32)
    pdf_sigma = tfd.Exponential(avg_val_lambda0).prob(xrge0)
    pdf_sigma_true = tfd.Exponential(tf.exp(lambda0)).prob(xrge0)
    pdf_sigma_m = tf.reduce_mean(tfd.Exponential(avg_val_lambda0).sample(1000))
    
    avg_beta0 = avg_val(var[1], l_values, epochs) 
    avg_beta1 = avg_val(var[2], l_values, epochs)
    avg_beta0_sd = tf.exp(avg_val(var[3], l_values, epochs))
    avg_beta1_sd = tf.exp(avg_val(var[4], l_values, epochs))
    avg_tau0_sd =  tf.exp(avg_val(var[5], l_values, epochs))
    avg_tau1_sd =  tf.exp(avg_val(var[6], l_values, epochs))
    
    xrge1 = tf.cast(tf.range(xrange1[0],xrange1[1],0.001), tf.float32)
    xrge2 = tf.cast(tf.range(xrange2[0],xrange2[1],0.001), tf.float32)
    pdf_beta0 = tfd.Normal(loc=avg_beta0, scale=avg_beta0_sd).prob(xrge1) 
    pdf_beta0_true = tfd.Normal(loc=mu_0, scale=tf.exp(sigma_0)).prob(xrge1) 
    pdf_beta1 = tfd.Normal(loc=avg_beta1, scale=avg_beta1_sd).prob(xrge2) 
    pdf_beta1_true = tfd.Normal(loc=mu_1, scale=tf.exp(sigma_1)).prob(xrge2) 
    
    xrge3 = tf.cast(tf.range(xrange3[0],xrange3[1],0.001), tf.float32)
    pdf_tau0 = tfd.TruncatedNormal(loc=0., scale=avg_tau0_sd, low=0., high=500).prob(xrge3)
    pdf_tau1 = tfd.TruncatedNormal(loc=0., scale=avg_tau1_sd, low=0., high=500).prob(xrge3)
    pdf_tau0_true = tfd.TruncatedNormal(loc=0., scale=tf.exp(sigma_tau0), low=0., high=500).prob(xrge3)
    pdf_tau1_true = tfd.TruncatedNormal(loc=0., scale=tf.exp(sigma_tau1), low=0., high=500).prob(xrge3)
    
    # prepare error plot
    lambda0e = tf.stack([tf.stack(var[0][1][i]-tf.constant(lambda0,tf.float32),0) for i in range(epochs)],-1)
    mu0e = tf.stack([tf.stack(tf.constant(var[1][1][i])-tf.constant(mu_0,dtype=tf.float32),0) for i in range(epochs)],-1)
    mu1e = tf.stack([tf.stack(tf.constant(var[2][1][i])-tf.constant(mu_1,dtype=tf.float32),0) for i in range(epochs)],-1)
    sigma0e = tf.stack([tf.stack(tf.constant(var[3][1][i])-tf.constant(sigma_0,dtype=tf.float32),0) for i in range(epochs)],-1)
    sigma1e = tf.stack([tf.stack(tf.constant(var[4][1][i])-tf.constant(sigma_1,dtype=tf.float32),0) for i in range(epochs)],-1)
    sigmatau0e = tf.stack([tf.stack(tf.constant(var[5][1][i])-tf.constant(sigma_tau0,dtype=tf.float32),0) for i in range(epochs)],-1)
    sigmatau1e = tf.stack([tf.stack(tf.constant(var[6][1][i])-tf.constant(sigma_tau1,dtype=tf.float32),0) for i in range(epochs)],-1)
    
    # plot everything
    fig = plt.figure(figsize=(20,9))
    matplotlib.rcParams.update({'font.size': 12})
    subfigs = fig.subfigures(2, 1, wspace=0.07, hspace=0.0)
    axs0 = subfigs[0].subplots(1, 4, gridspec_kw=dict(top=0.9)) 
    axs1 = subfigs[1].subplots(1, 3, gridspec_kw=dict(top=0.85, width_ratios=[2.,1.,1.]))
    
    # prior predictions
    axs1[0].fill_between(df["days"], df["means"]+(df["sd_means"]),
                         df["means"]-(df["sd_means"]),  facecolor = "grey",
                         alpha = 0.3)
    sns.lineplot(data = df, x = "days",  y = "means",  color="grey", 
                 alpha = 0.3, ax = axs1[0])
    sns.scatterplot(data = df, x = "days",  y = "means",  color="black", 
                    ax = axs1[0])
    if model == "log":
        axs1[0].plot([idx[0],idx[1]],tf.reduce_mean(tf.exp(samples_e["days_m"]),0),  "o", color = "red", 
                     markersize = 10, alpha = 0.5, label='_nolegend_')
        
        means = tf.reduce_mean(samples_d["days_m"],0)
        sds = tf.reduce_mean(samples_d["days_E_sd"],0)
        
        axs1[0].vlines(x = idx[0], ymin = means[0]-sds[0],  ymax = means[0]+sds[0], 
                       color = "red", linewidth = 2, alpha = 0.5)#, 
                       #color = col_EnC[0])
        axs1[0].vlines(x = idx[1], ymin = means[1]-sds[1],  ymax = means[1]+sds[1],
                       color = "red", linewidth = 2, alpha = 0.5)
                        #color = col_EnC[1])
    if model == "normal":
        axs1[0].plot([Z_days[idx[0]],Z_days[idx[1]]],tf.reduce_mean(samples_e["days_m"],0),  "o", color = "red", 
                     markersize = 10, alpha = 0.5, label='_nolegend_')
        
        means = tf.reduce_mean(samples_e["days_m"],0)
        sds = tf.reduce_mean(samples_e["days_E_sd"],0)
        
        axs1[0].vlines(x = Z_days[idx[0]], ymin = means[0]-sds[0],  ymax = means[0]+sds[0], 
                       color = "red", linewidth = 2, alpha = 0.5)#, 
                       #color = col_EnC[0])
        axs1[0].vlines(x = Z_days[idx[1]], ymin = means[1]-sds[1],  ymax = means[1]+sds[1],
                       color = "red", linewidth = 2, alpha = 0.5)
                        #color = col_EnC[1])
    
    # convergence: betas ~ N
    axs0[0].plot(xrge1, pdf_beta0, color = col_betas[0], linewidth = 3,
            label = f"b0 ~ N({avg_beta0:.2f},{avg_beta0_sd:.2f})")
    axs0[1].plot(xrge2, pdf_beta1, color = col_betas[1], linewidth = 3,
            label = f"b1 ~ N({avg_beta1:.2f},{avg_beta1_sd:.2f})")
    ## true
    axs0[0].plot(xrge1, pdf_beta0_true,  linestyle="dotted",  color ="black", linewidth = 2)
    axs0[1].plot(xrge2, pdf_beta1_true,  linestyle="dotted",  color ="black", linewidth = 2)
    
    # convergence: taus ~ Truncated N
    axs0[2].plot(xrge3, pdf_tau0, color = col_betas[2], linewidth = 3,
            label = f"tau0 ~ TruncN(0,{avg_tau0_sd:.2f})")
    axs0[2].plot(xrge3, pdf_tau1, color = col_betas[3], linewidth = 3,
            label = f"tau1 ~ TruncN(0,{avg_tau1_sd:.2f})")
    ## true
    axs0[2].plot(xrge3, pdf_tau0_true,  linestyle="dotted",  color ="black", linewidth = 2)
    axs0[2].plot(xrge3, pdf_tau1_true,  linestyle="dotted",  color ="black", linewidth = 2)
    
    # convergence: sigma ~ Exp
    axs0[3].plot(xrge0, pdf_sigma, color = col_lambda, linewidth = 3,
                 label = f"s ~ Exp({avg_val_lambda0:.2f})")
    ## true
    axs0[3].plot(xrge0, pdf_sigma_true, color ="black", linewidth = 2,
                 linestyle = "dotted" )
    ## mean value of sigma
    axs0[3].axvline(x = pdf_sigma_m,  linestyle="--", color = col_lambda)
    
    ## legend position upper row
    axs0[0].legend(loc="upper left", labelspacing=0.2, handlelength=0.5)
    axs0[1].legend(loc="upper left", labelspacing=0.2, handlelength=0.5)
    axs0[2].legend(loc="upper right", labelspacing=0.2, handlelength=0.5)
    axs0[3].legend(loc="upper right", labelspacing=0.2, handlelength=0.5)
    
    # error: lambda0
    axs1[1].plot(lambda0e, color = col_lambda, label = "lambda0")
    axs1[1].plot(mu0e, color = col_betas[0], label = "beta0")
    axs1[1].plot(mu1e, color = col_betas[1], label = "beta1")
    axs1[2].plot(sigma0e, color = col_betas[0], label = "sigma0")
    axs1[2].plot(sigma1e, color = col_betas[1], label = "sigma1")
    axs1[2].plot(sigmatau0e, color = col_betas[2], label = "sigma_tau0")
    axs1[2].plot(sigmatau1e, color = col_betas[3], label = "sigma_tau1")
        
    # add title to subfigures
    subfigs[1].suptitle(f"Prior predictive data \n (learned R2: day 0= {tf.reduce_mean(samples_d['R2_0']):.2f}, day 9= {tf.reduce_mean(samples_d['R2_1']):.2f})",
                        fontweight = "bold", fontsize = "x-large", x=0.30)
    subfigs[0].suptitle("Learned prior distributions",
                        fontweight = "bold", fontsize = "x-large", x=0.21)
    axs1[1].set_title("Absolute error between true and learned hyperparameter",
                        fontweight = "bold", fontsize = "x-large", loc = "left",
                        pad = 30.)
    
    axs1[0].set_ylabel("mean RT")
    for i in range(1,3):
        axs1[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs1[i].set_xlabel("epochs")
        axs1[i].axhline(y = 0., color = 'black', linestyle = '--', linewidth=2)
        if model == "normal":
            axs1[i].legend(loc="upper right", labelspacing=0.2, handlelength=0.5)
        else:
            axs1[i].legend(loc="lower right", labelspacing=0.2, handlelength=0.5)
    
    for i in range(4):
        axs0[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs0[i].set_xlabel(None)
    axs0[0].set_ylabel("density")
    
# samples: ideal, incon1, incon2, incon3
def plot_pp_mlm(samples_d, samples_e, X_days, idx):
    
    # titles for plotting
    titles= [f"ideal expert \n (learned R2: day 0= {tf.reduce_mean(samples_d[0]['R2_0']):.2f}, day 9= {tf.reduce_mean(samples_d[0]['R2_1']):.2f})", 
             f"inconsistent I \n (learned R2: day 0= {tf.reduce_mean(samples_d[1]['R2_0']):.2f}, day 9= {tf.reduce_mean(samples_d[1]['R2_1']):.2f})", 
             f"inconsistent II \n (learned R2: day 0= {tf.reduce_mean(samples_d[2]['R2_0']):.2f}, day 9= {tf.reduce_mean(samples_d[2]['R2_1']):.2f})", 
             f"inconsistent III \n (learned R2: day 0= {tf.reduce_mean(samples_d[3]['R2_0']):.2f}, day 9= {tf.reduce_mean(samples_d[3]['R2_1']):.2f})"]
    
    # prior predictions
    fig, axs = plt.subplots(2,2, figsize=(11,7), sharex=True, sharey=True)
    
    for i,j,k,n in list(zip([0,1,2,3], [0,0,1,1], [0,1,0,1], titles)):
        
        df = pd.DataFrame({
             "days": X_days,
             "means": tf.reduce_mean(samples_d[i]["y_m"],(0,1,2)).numpy(),
             "sd_means": tf.reduce_mean(tf.math.reduce_std(tf.reduce_mean(samples_d[i]["mu_m"],2),1),0).numpy(),
             })
        # axs[j,k].fill_between(df["days"], 
        #                       df["means"]+(df["sd_means"]),
        #                       df["means"]-(df["sd_means"]),  
        #                       facecolor = "grey", alpha = 0.3)
        
        # sns.lineplot(data = df, x = "days",  y = "means",  color="grey", 
        #              alpha = 0.3, ax = axs[j,k])
        sns.scatterplot(data = df, x = "days",  y = "means",  color="black", 
                        ax = axs[j,k], zorder=1)
        for t in range(18):
            sns.lineplot(x=X_days[0:10], y=tf.reduce_mean(samples_d[i]["y_m"], (0,1))[t,:], 
                         alpha = 0.7, ax = axs[j,k], zorder=0)
            
        axs[j,k].plot([X_days[idx[0]],X_days[idx[1]]],
                      tf.reduce_mean(samples_e[i]["days_m"],0),  "o", color = "red", 
                      markersize = 10, alpha = 0.5, label='_nolegend_')
        
        means = tf.reduce_mean(samples_e[i]["days_m"],0)
        sds = tf.reduce_mean(samples_e[i]["days_E_sd"],0)
        
        axs[j,k].vlines(x = X_days[idx[0]], ymin = means[0]-sds[0],  ymax = means[0]+sds[0], 
                       color = "red", linewidth = 2, alpha = 0.5)
        
        axs[j,k].vlines(x = X_days[idx[1]], ymin = means[1]-sds[1],  ymax = means[1]+sds[1],
                       color = "red", linewidth = 2, alpha = 0.5)
        
        axs[j,k].set_title(n)
        axs[j,k].set_ylabel("mean RT")
        plt.tight_layout()
        
        

# s = col 1,2,3
# s2 = col 4,5
def plot_input_formats_lm(expert_samples, samples,fct_a_lvl,fct_b_lvl, 
                          s=20, s2=40): 

    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    col_EnC = ["#0a75ad", "#3293f0", "#7ad0ed"]
    col_ReP = ["#ffaa00", "#e86e4d"]
    col_lambda = "#a6b7c6"
    
    fig = plt.figure(figsize = (11,3))
    subfigs = fig.subfigures(1, 5, wspace=0.07, hspace=0.0)
    ax0 = subfigs[0].subplots(2, 1, gridspec_kw=dict(top=0.8), sharey = True)
    ax1 = subfigs[1].subplots(2, 1, gridspec_kw=dict(top=0.8), sharey = True)
    ax2 = subfigs[2].subplots(2, 1, gridspec_kw=dict(top=0.8), sharey = True)
    ax3 = subfigs[3].subplots(2, 1, gridspec_kw=dict(top=0.8), sharey = True)
    ax4 = subfigs[4].subplots(2, 1, gridspec_kw=dict(top=0.8), sharey = True)
    
    [sns.histplot(tf.reduce_mean(expert_samples["mb"][:,:,i],0), stat="probability",
                 ax=ax0[0], bins = s, color=col_EnC[i]) for i in range(fct_b_lvl)]
    [sns.histplot(tf.reduce_mean(expert_samples["ma"][:,:,i],0), stat="probability",
                 ax=ax1[0], bins = s, color=col_ReP[i]) for i in range(fct_a_lvl)]
    [ax2[0].axvline(expert_samples["effects"][0,0,i], 
                    color=col_betas[i]) for i in range(fct_b_lvl)]
    [ax3[0].axvline(expert_samples["effects_sd"][0,i], 
                    color=col_betas[i+3]) for i in range(fct_b_lvl)]
    sns.histplot(tf.reduce_mean(expert_samples["R2"][:,:],0), bins = s2,
                 stat="probability",color = col_lambda, ax=ax4[0]) 
    
    [sns.histplot(tf.reduce_mean(samples["mb"][:,:,i],1), stat="probability",
                 ax=ax0[1], bins = s, color=col_EnC[i]) for i in range(fct_b_lvl)]
    [sns.histplot(tf.reduce_mean(samples["ma"][:,:,i],1), stat="probability",
                 ax=ax1[1], bins = s, color=col_ReP[i]) for i in range(fct_a_lvl)]
    [sns.histplot(tf.reduce_mean(samples["effects"][:,:,i],1), stat="probability",
                 ax=ax2[1], bins = s, color=col_betas[i]) for i in range(fct_b_lvl)]
    [sns.histplot(samples["effects_sd"][:,i], stat="probability",
                 ax=ax3[1], bins = s2, color=col_betas[i+3]) for i in range(fct_b_lvl)]
    sns.histplot(tf.reduce_mean(samples["R2"][:,:],1), bins = s2,
                 stat="probability", color=col_lambda, ax=ax4[1]) 
    
    for a, yn in zip(range(2), ["expert", "model"]):
        for axes,n in zip([ax0, ax1, ax2, ax3, ax4], ["mb","ma","effects","effects_sd","R2"]):
            axes[a].set_xlim((tf.math.minimum(tf.reduce_min(tf.reduce_mean(samples[n],0)),
                                tf.reduce_min(tf.reduce_mean(expert_samples[n],0)))-0.02,
                                tf.math.maximum(tf.reduce_max(tf.reduce_mean(samples[n],0)),
                                tf.reduce_max(tf.reduce_mean(expert_samples[n],0)))+0.02))
            axes[a].set_ylabel(None)
            ax0[a].set_ylabel(yn)
            axes[0].set_xticks([])
    
    for a,t in zip(range(5), ["Marginals\n Encoding","Marginals\n Repetition",
                              "TE per Enc.\n Cond. (m)",
                              "TE per Enc.\n Cond. (sd)","Statistic\n R2"]):
        subfigs[a].suptitle(t)

