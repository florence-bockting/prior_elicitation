# -*- coding: utf-8 -*-
"""
Plotting: Convergence Diagnostics
"""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import itertools


def plot_diagnostics_binomial(var, res, final_epoch):
    # define labels for plotting
    n_mus = ["mu0","mu1"]
    n_sigmas = ["sigma0","sigma1"]
    
    # define color codes for plotting
    # betas 
    col_betas = ["#2acaea", "#0a75ad"]
    
    # norm of gradients
    grad_mu = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][0])) for i in range(final_epoch)],0)
    grad_sigma = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][1])) for i in range(final_epoch)],0) 

    fig = plt.figure(figsize=(14,6))
    matplotlib.rcParams.update({'font.size': 12})
    subfigs = fig.subfigures(1, 2, width_ratios= [1.5,3])
    axs0 = subfigs[0].subplots(2, 1, gridspec_kw=dict(top=0.9, hspace=0.5))
    axs1 = subfigs[1].subplots(2, 2, gridspec_kw=dict(top=0.9,left=0.05, hspace=0.5))
    
    
    # plot convergence
    for i in range(2):
        axs1[1,0].plot(tf.squeeze(tf.stack(var[0][1],-1))[i,:], 
                     color = col_betas[i], linewidth=2)
        axs1[1,1].plot(tf.exp(tf.squeeze(tf.stack(var[1][1],-1)))[i,:], 
                     color = col_betas[i], linewidth=2)
    # plot gradients 
    sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_mu, 
                   linewidth=0, alpha = 0.5, color="black", ax=axs1[0,0])
    sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_sigma, 
                   linewidth=0, alpha = 0.5, color="black", ax=axs1[0,1])
    # plot loss
    axs0[0].plot([res[i]["loss"] for i in range(final_epoch)], 
                 color = "black",linewidth=2)
    axs0[1].plot([res[i]["loss_tasks"] for i in range(final_epoch)],
                 linewidth=2)
    
    subfigs[0].suptitle("Loss", size="large", weight="demibold",x=0.17)
    subfigs[1].suptitle("Gradients", size="large", weight="demibold",x=0.1)
    axs1[1,0].set_title("Convergence of hyperparameter values", 
                        size="large", weight="demibold", x=0.6, y=1.1)
    
    axs1[1,0].legend(n_mus, ncol=2, labelspacing=0.2, 
                     columnspacing=0.3, handlelength=1, loc="upper right")
    axs1[1,1].legend(n_sigmas, ncol=2, labelspacing=0.2,
                     columnspacing=0.3, handlelength=1, loc="upper right")
    names = ["mus", "sigmas"]
    
    for i,n in zip(range(2), names):
        axs1[0,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs1[1,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs1[0,i].set_title(n)
        axs1[1,i].set_xlabel("epochs")
    for k in range(2):
        axs0[k].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    axs0[1].set_xlabel("epochs")
    axs0[0].set_title("total loss")
    axs0[1].set_title("individual losses")
    plt.show()

def plot_diagnostics_lm(var, res, final_epoch):
     # define labels for plotting
     n_mus = ["mu0","mu1","mu2","mu3","mu4","mu5"]
     n_sigmas = ["sigma0","sigma1","sigma2","sigma3","sigma4","sigma5"]
     names_el = [n_mus, n_sigmas]
     # define color codes for plotting
     # betas 
     col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
     col_lambda = "#a6b7c6"
     
     # norm of gradients
     grads = [tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][j])) for i in range(final_epoch)],0) for j in range(3)]
 
     fig = plt.figure(figsize=(15,7))
     matplotlib.rcParams.update({'font.size': 12})
     subfigs = fig.subfigures(1, 2, width_ratios= [1.5,3])
     axs0 = subfigs[0].subplots(2, 1, gridspec_kw=dict(top=0.9, hspace=0.5))
     axs1 = subfigs[1].subplots(2, 3, gridspec_kw=dict(top=0.9, left=0.05, 
                                                       hspace=0.5))
     
     # plot convergence
     axs1[1,0].plot(tf.exp(var[0][1]),color=col_lambda, linewidth=2)
     for k,i in itertools.product(range(1,3),range(6)):
         axs1[1,k].plot(tf.where(k==1,tf.squeeze(tf.stack(var[k][1],-1))[i,:],
                                 tf.exp(tf.squeeze(tf.stack(var[k][1],-1))[i,:])),
                      color = col_betas[i], linewidth=2)
       
     # plot gradients   
     [sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grads[i], 
                    linewidth=0, alpha = 0.5, color="black", ax=axs1[0,i]) for i in range(3)]
   
     # plot loss
     for j,n in enumerate(["loss","loss_tasks"]):
         axs0[j].plot([res[i][n] for i in range(final_epoch)], linewidth=2)  #color = "black",
     
     # add titles
     subfigs[0].suptitle("Loss", size="x-large", weight="demibold",x=0.13)
     subfigs[1].suptitle("Gradients", size="x-large", weight="demibold",x=0.09)
     axs1[1,0].set_title("Convergence of hyperparameter values", 
                         size="x-large", weight="demibold", x=1.1, y=1.1)
     
     # add legends
     axs1[1,0].legend(["lambda0"], loc="lower right", handlelength=1)
     for i,n in enumerate(names_el):
         axs1[1,i+1].legend(n, ncol=2, labelspacing=0.2, columnspacing=0.3, handlelength=1)
     names_cat = ["lambda0", "mus", "sigmas"]
     
     # format axes and set axis-title
     for i,n in zip(range(3), names_cat):
         axs1[0,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
         axs1[1,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
         axs1[0,i].set_title(n)
         axs1[1,i].set_xlabel("epochs")
     for k in range(2):
         axs0[k].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
     
     axs0[1].set_xlabel("epochs")
     axs0[0].set_title("total loss")
     axs0[1].set_title("individual losses")
     plt.show()
     
def plot_diagnostics_poisson(var, res, final_epoch):
    # define labels for plotting
    n_mus = ["mu0","mu1", "mu2", "mu3"]
    n_sigmas = ["sigma0","sigma1", "sigma2", "sigma3"]
    
    # define color codes for plotting
    # betas 
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d"]
    
    # norm of gradients
    grad_mu = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][0])) for i in range(final_epoch)],0)
    grad_sigma = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][1])) for i in range(final_epoch)],0) 

    fig = plt.figure(figsize=(14,6))
    matplotlib.rcParams.update({'font.size': 12})
    subfigs = fig.subfigures(1, 2, width_ratios= [1.5,3])
    axs0 = subfigs[0].subplots(2, 1, gridspec_kw=dict(top=0.9, hspace=0.5))
    axs1 = subfigs[1].subplots(2, 2, gridspec_kw=dict(top=0.9,left=0.05, hspace=0.5))
    
    # plot convergence
    for i in range(4):
        axs1[1,0].plot(tf.squeeze(tf.stack(var[0][1],-1))[i,:], 
                     color = col_betas[i], linewidth=2)
        axs1[1,1].plot(tf.exp(tf.squeeze(tf.stack(var[1][1],-1)))[i,:], 
                     color = col_betas[i], linewidth=2)
    # plot gradients 
    sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_mu, 
                   linewidth=0, alpha = 0.5, color="black", ax=axs1[0,0])
    sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_sigma, 
                   linewidth=0, alpha = 0.5, color="black", ax=axs1[0,1])
    # plot loss
    axs0[0].plot([res[i]["loss"] for i in range(final_epoch)], 
                 color = "black",linewidth=2)
    axs0[1].plot([res[i]["loss_tasks"] for i in range(final_epoch)],
                 linewidth=2)
    
    subfigs[0].suptitle("Loss", size="large", weight="demibold",x=0.17)
    subfigs[1].suptitle("Gradients", size="large", weight="demibold",x=0.1)
    axs1[1,0].set_title("Convergence of hyperparameter values", 
                        size="large", weight="demibold", x=0.6, y=1.1)
    
    axs1[1,0].legend(n_mus, ncol=2, labelspacing=0.2, 
                     columnspacing=0.3, handlelength=1, loc="center right")
    axs1[1,1].legend(n_sigmas, ncol=2, labelspacing=0.2,
                     columnspacing=0.3, handlelength=1, loc="center right")
    names = ["mus", "sigmas"]
    
    for i,n in zip(range(2), names):
        axs1[0,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs1[1,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs1[0,i].set_title(n)
        axs1[1,i].set_xlabel("epochs")
    for k in range(2):
        axs0[k].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    axs0[1].set_xlabel("epochs")
    axs0[0].set_title("total loss")
    axs0[1].set_title("individual losses")
    plt.show()

def plot_diagnostics_negbinom(var, res, final_epoch):
    
    # define labels for plotting
    n_mus = ["mu0","mu1", "mu2", "mu3"]
    n_sigmas = ["sigma0","sigma1", "sigma2", "sigma3"]
    
    # define color codes for plotting
    # betas 
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d"]
    col_lambda = "#a6b7c6"
    # norm of gradients
    grad_lambda = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][0])) for i in range(final_epoch)],0)
    grad_mu = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][1])) for i in range(final_epoch)],0)
    grad_sigma = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][2])) for i in range(final_epoch)],0) 

    fig = plt.figure(figsize=(14,6))
    matplotlib.rcParams.update({'font.size': 12})
    subfigs = fig.subfigures(1, 2, width_ratios= [1,3.5])
    axs0 = subfigs[0].subplots(2, 1, gridspec_kw=dict(top=0.9, hspace=0.5))
    axs1 = subfigs[1].subplots(2, 3, gridspec_kw=dict(top=0.9,left=0.05, hspace=0.5))
    
    # plot convergence
    for i in range(4):
        axs1[1,0].plot(tf.squeeze(tf.stack(var[1][1],-1))[i,:], 
                     color = col_betas[i], linewidth=2)
        axs1[1,1].plot(tf.exp(tf.squeeze(tf.stack(var[2][1],-1)))[i,:], 
                     color = col_betas[i], linewidth=2)
    axs1[1,2].plot(tf.exp(tf.squeeze(tf.stack(var[0][1],-1))), 
                 color = col_lambda, linewidth=2)
    
    # plot gradients 
    sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_mu, 
                   linewidth=0, alpha = 0.5, color="black", ax=axs1[0,0])
    sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_sigma, 
                   linewidth=0, alpha = 0.5, color="black", ax=axs1[0,1])
    sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_lambda, 
                   linewidth=0, alpha = 0.5, color="black", ax=axs1[0,2])
    
    # plot loss
    axs0[0].plot([res[i]["loss"] for i in range(final_epoch)], 
                 color = "black",linewidth=2)
    axs0[1].plot([res[i]["loss_tasks"] for i in range(final_epoch)],
                 linewidth=2)
    
    subfigs[0].suptitle("Loss", size="large", weight="demibold",x=0.17)
    subfigs[1].suptitle("Gradients", size="large", weight="demibold",x=0.1)
    axs1[1,0].set_title("Convergence of hyperparameter values", 
                        size="large", weight="demibold", x=0.8, y=1.1)
    
    axs1[1,0].legend(n_mus, ncol=2, labelspacing=0.2, 
                     columnspacing=0.3, handlelength=1, loc="center right")
    axs1[1,1].legend(n_sigmas, ncol=2, labelspacing=0.2,
                     columnspacing=0.3, handlelength=1, loc="center right")
    
    names = ["mus", "sigmas","lambda"]
    for i,n in zip(range(3), names):
        axs1[0,i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs1[1,i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs1[0,i].set_title(n)
        axs1[1,i].set_xlabel("epochs")
    for k in range(2):
        axs0[k].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    axs0[1].set_xlabel("epochs")
    axs0[0].set_title("total loss")
    axs0[1].set_title("individual losses")
    plt.show()
    
def plot_diagnostics_mlm(var, res, final_epoch):
     
     # define color codes for plotting
     # betas 
     col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
     col_lambda = "#a6b7c6"
     
     # norm of gradients
     grad_lambda = tf.stack([tf.squeeze(res[i]["gradients"][0]) for i in range(final_epoch)],0)
     grad_mu0 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][1])) for i in range(final_epoch)],0)
     grad_mu1 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][2])) for i in range(final_epoch)],0)
     grad_sigma0 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][3])) for i in range(final_epoch)],0) 
     grad_sigma1 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][4])) for i in range(final_epoch)],0) 
     grad_sigma_tau0 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][5])) for i in range(final_epoch)],0) 
     grad_sigma_tau1 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][6])) for i in range(final_epoch)],0) 
 
     fig = plt.figure(figsize=(19,7))
     matplotlib.rcParams.update({'font.size': 12})
     subfigs = fig.subfigures(1, 2, width_ratios= [1.,3])
     axs0 = subfigs[0].subplots(2, 1, gridspec_kw=dict(top=0.9, hspace=0.5))
     axs1 = subfigs[1].subplots(2, 4, gridspec_kw=dict(top=0.9,left=0.05, hspace=0.5))
     
     
     # plot convergence
     axs1[1,0].plot(tf.exp(var[0][1]),color=col_lambda, linewidth=2)
     axs1[1,1].plot(var[1][1], color = col_betas[0], linewidth=2)
     axs1[1,1].plot(var[2][1], color = col_betas[1], linewidth=2)
     axs1[1,2].plot(tf.exp(var[3][1]), color = col_betas[0], linewidth=2)
     axs1[1,2].plot(tf.exp(var[4][1]), color = col_betas[1], linewidth=2)
     axs1[1,3].plot(tf.exp(var[5][1]), color = col_betas[2], linewidth=2)
     axs1[1,3].plot(tf.exp(var[6][1]), color = col_betas[3], linewidth=2)
     # plot gradients   
     sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_lambda, 
                    linewidth=0, alpha = 0.5, color="black", ax=axs1[0,0])
     sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_mu0, 
                    linewidth=0, alpha = 0.5, color="grey", ax=axs1[0,1])
     sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_mu1, 
                    linewidth=0, alpha = 0.5, color="black", ax=axs1[0,1])
     sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_sigma0, 
                    linewidth=0, alpha = 0.5, color="grey", ax=axs1[0,2])
     sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_sigma1, 
                    linewidth=0, alpha = 0.5, color="black", ax=axs1[0,2])
     sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_sigma_tau0, 
                    linewidth=0, alpha = 0.5, color="grey", ax=axs1[0,3])
     sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grad_sigma_tau1, 
                    linewidth=0, alpha = 0.5, color="black", ax=axs1[0,3])
     # plot loss
     axs0[0].plot([res[i]["loss"] for i in range(final_epoch)], 
                  color = "black",linewidth=2)
     axs0[1].plot([res[i]["loss_tasks"] for i in range(final_epoch)],
                  linewidth=2)
     
     subfigs[0].suptitle("Loss", size="x-large", weight="demibold",x=0.13)
     subfigs[1].suptitle("Gradients", size="x-large", weight="demibold",x=0.09)
     axs1[1,0].set_title("Convergence of hyperparameter values", 
                         size="x-large", weight="demibold", x=1., y=1.1)
     
     axs1[1,0].legend(["lambda0"], loc="center right", handlelength=0.5)
     axs1[1,1].legend(["mu0", "mu1"], loc="center right", handlelength=0.5,
                      labelspacing=0.2)
     axs1[1,2].legend(["sigma0", "sigma1"], loc="center right", handlelength=0.5, 
                      labelspacing=0.2)
     axs1[1,3].legend(["sigma_tau0", "sigma_tau1"], loc="center right", 
                      handlelength=0.5, labelspacing=0.2)
     
     axs1[0,0].set_title("lambda0")
     axs1[0,1].set_title("mus")
     axs1[0,2].set_title("sigmas")
     axs1[0,3].set_title("sigma_taus")
     
     for i in range(4):
         axs1[0,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
     axs0[1].set_xlabel("epochs")
     axs0[0].set_title("total loss")
     axs0[1].set_title("individual losses")
     plt.show()