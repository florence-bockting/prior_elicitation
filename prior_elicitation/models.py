# -*- coding: utf-8 -*-
"""
Binomial Model
"""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import patsy as pa
import pandas as pd

class Binomial_Model(tf.Module):
    def __init__(self, input_settings_model, input_settings_learning, input_settings_global):
        
        super(Binomial_Model, self).__init__()
        self.input_settings_global = input_settings_global
        self.input_settings_learning = input_settings_learning
        self.input_settings_model = input_settings_model
        
        self.mus = tf.Variable(initial_value=tf.random.uniform((2,),0., 1.),
                               trainable = True, name = "mus")
        self.sigmas = tf.Variable(initial_value=tf.math.log([0.1, 0.1]),
                                  trainable = True, name = "sigmas")
    
    def __call__(self):
        
        # generate samples from data generating process
        d_samples = self.data_generating_model(self.mus, self.sigmas, sigma_taus=None,
                                               alpha_LKJ=None, lambda0=None,
                                        input_settings_global = self.input_settings_global,
                                        input_settings_learning = self.input_settings_learning, 
                                        input_settings_model = self.input_settings_model,
                                        model_type = "model")
        return d_samples
        
    
    def data_generating_model(self, mus, sigmas, sigma_taus, alpha_LKJ, lambda0, 
                              input_settings_global,
                              input_settings_learning, 
                              input_settings_model, 
                              model_type): 
         # set seed
         if input_settings_global["seed"] is not None:
             tf.random.set_seed(input_settings_global["seed"])
         
         # initialize variables
         if model_type == "expert":
             rep = input_settings_learning["rep_exp"]
             B = 1
         else:
             rep = input_settings_learning["rep_mod"]
             B = input_settings_learning["B"]
             
         X = input_settings_model["X"] 
         X_idx = input_settings_model["X_idx"]
         temp = input_settings_learning["temp"] 
         size = input_settings_model["model_specific"]["size"]
    
         X["no_axillary_nodes"] = tf.constant(X["no_axillary_nodes"], 
                                              dtype=tf.float32)
         x_sd = tf.math.reduce_std(X["no_axillary_nodes"])
         # scale predictor
         X_scaled = tf.constant(X["no_axillary_nodes"],
                                dtype=tf.float32)/x_sd
         
         #select only data points that were selected from expert
         X_scaled = tf.gather(X_scaled,X_idx)
    
         # reshape predictor 
         X = tf.broadcast_to(tf.constant(X_scaled, tf.float32)[None,None,:], (B,rep,len(X_scaled)))
    
         # sample from priors
         beta0 = tfd.Normal(mus[0], tf.exp(sigmas[0])).sample((B,rep,1))
         beta1 = tfd.Normal(mus[1], tf.exp(sigmas[1])).sample((B,rep,1))
         
         betas = tf.stack([beta0, beta1],-1)
         
         # linear predictor
         mu = beta0 + beta1*X
         # map linear predictor to theta
         theta = tf.sigmoid(mu)
         
         # constant outcome vector (including zero outcome)
         c = tf.ones((B,rep,1,size+1))*tf.cast(tf.range(0,size+1), tf.float32)
         # compute pmf value
         pi = tfd.Binomial(total_count=size, probs=theta[:,:,:,None]).prob(c) 
         # prevent underflow
         pi = tf.where(pi < 1.8*10**(-30), 1.8*10**(-30), pi)
         ## sample n-dimensional one-hot-like
         # Gumbel-Max Softmax trick
         # sample from uniform
         u = tfd.Uniform(0,1).sample((B,rep,X.shape[-1],size+1))
         # generate a gumbel sample
         g = -tf.math.log(-tf.math.log(u))
         # softmax trick
         w = tf.nn.softmax(tf.math.divide(tf.math.add(tf.math.log(pi),g),temp))
        
         # reparameterization/ apply linear transformation
         # shape: (B, rep, len(X_idx))
         y_idx = tf.reduce_sum(tf.multiply(w,c), -1)  
         
         # R2
         R2 = tf.math.reduce_variance(mu, -1)/tf.math.reduce_variance(y_idx, -1)
         
         return {"y_idx":y_idx,
                 "R2": R2,
                 "betas": tf.squeeze(betas)}

class LM_Model(tf.Module):
    def __init__(self, input_settings_model, input_settings_learning, 
                 input_settings_global): 
        super(LM_Model, self).__init__()
        tf.random.set_seed(input_settings_global["seed"])
        
        self.input_settings_global = input_settings_global
        self.learning_settings = input_settings_learning
        self.model_settings = input_settings_model
        
        self.mus = tf.Variable(initial_value= tf.random.uniform((6,), 0., 1.), 
                               trainable=True, name = "mus")
        self.sigmas = tf.Variable(initial_value=tf.math.log(tf.random.uniform((6,), 0., 1.)),
                                  trainable=True, name = "sigmas") 
        self.lambda0 = tf.Variable(initial_value=tf.math.log(tf.random.uniform((1,), 0., 1.)), 
                                   trainable=True, name="lambda0") 
    
    def __call__(self):
        
        d_samples =  self.data_generating_model(self.mus, self.sigmas,  sigma_taus=None,
                                               alpha_LKJ=None, lambda0=self.lambda0, 
                                   input_settings_global = self.input_settings_global,
                                   input_settings_learning = self.learning_settings, 
                                   input_settings_model = self.model_settings,
                                   model_type = "model")
        
        return d_samples

    def data_generating_model(self, mus, sigmas,  sigma_taus, alpha_LKJ, lambda0, 
                              input_settings_global,
                              input_settings_learning, 
                              input_settings_model, 
                              model_type):  
        # set seed
        #if input_settings_global["seed"] is not None:
        tf.random.set_seed(input_settings_global["seed"])
        
        # initialize variables
        if model_type == "expert":
            rep = input_settings_learning["rep_exp"]
            B = 1
        else:
            rep = input_settings_learning["rep_mod"]
            B = input_settings_learning["B"]
        
        # initialize hyperparameter for learning
        Nobs_cell = input_settings_model["model_specific"]["Nobs_cell"] 
        fct_a_lvl = input_settings_model["model_specific"]["fct_a_lvl"] 
        fct_b_lvl= input_settings_model["model_specific"]["fct_b_lvl"]
        
        # number of design cells
        no_cells = fct_a_lvl*fct_b_lvl
        
        # design matrix
        X_design = tf.constant(pa.dmatrix("a*b", pa.balanced(
            a=fct_a_lvl, b=fct_b_lvl, repeat=Nobs_cell)), dtype = tf.float32)
        
        model = tfd.JointDistributionNamed(dict(
            ## sample from priors
            beta_int = tfd.Normal(loc=mus[0], scale=tf.exp(sigmas[0])), 
            beta_a2 = tfd.Normal(loc=mus[1], scale=tf.exp(sigmas[1])), 
            beta_b2 = tfd.Normal(loc=mus[2], scale=tf.exp(sigmas[2])), 
            beta_b3 = tfd.Normal(loc=mus[3], scale=tf.exp(sigmas[3])), 
            beta_a2b2 = tfd.Normal(loc=mus[4], scale=tf.exp(sigmas[4])), 
            beta_a2b3 = tfd.Normal(loc=mus[5], scale=tf.exp(sigmas[5]))
            )).sample((B,rep))
        
        sigma = tfd.Gamma(concentration=rep, 
                          rate=(rep*tf.exp(lambda0))+0.0).sample((B,rep))
           
       
        # organize betas into vector
        betas = tf.stack([model["beta_int"],model["beta_a2"],model["beta_b2"],
                          model["beta_b3"],model["beta_a2b2"],model["beta_a2b3"]],
                          axis = -1)  
       
        # linear predictor
        mu = tf.squeeze(tf.matmul(X_design[None,None,:,:], 
                        tf.expand_dims(betas,2), 
                        transpose_b=True), axis = -1)
        
        # observations
        y_obs = tfd.Normal(mu, sigma).sample()
        
        ## expected data: joints
        # a1_b1, a1_b2, a1_b3, a2_b1, a2_b2, a2_b3
        obs_joints_ind = tf.stack([y_obs[:,:,i::no_cells] for i in range(no_cells)],-1)
        # avg. across individuals 
        obs_joints = tf.reduce_mean(obs_joints_ind, axis = 2)
        
        ## marginals
        # marginal factor with 3 levels
        mb = obs_joints[:,:,0:3] + obs_joints[:,:,3:7]
        # marginal factor with 2 levels
        ma = tf.stack([tf.reduce_sum(obs_joints[:,:,0:3],-1),
                       tf.reduce_sum(obs_joints[:,:,3:7],-1)],-1)
        
        ## effects
        # effect of factor mb for each level of ma
        effects1 = tf.stack([obs_joints[:,:,i]-obs_joints[:,:,j] for i,j in zip(range(3,6),range(0,3))] ,-1)
        effects2 = tf.stack([obs_joints[:,:,i]-obs_joints[:,:,j] for i,j in zip([1,1,4,4],[0,2,3,5])] ,-1)
        ## R2
        R2 = tf.divide(tf.math.reduce_variance(mu,2), tf.math.reduce_variance(y_obs,2))
             
        return {"mb":mb, 
                "ma":ma,
                "effects1":effects1, 
                "effects2":effects2,
                "model": model,
                "R2": R2,  
                "y_obs": y_obs, 
                "mu":mu, 
                "obs_joints":obs_joints,
                "betas": betas}

class Poisson_Model(tf.Module):
    def __init__(self, input_settings_model, input_settings_learning,
                 input_settings_global): 
        super(Poisson_Model, self).__init__()
        tf.random.set_seed(input_settings_global["seed"])
        
        self.input_settings_global = input_settings_global
        self.input_settings_learning = input_settings_learning
        self.input_settings_model = input_settings_model
        
        self.mus = tf.Variable(initial_value=[0., 0., 0., 0.], 
                               trainable = True, name = "mus")
        self.sigmas = tf.Variable(initial_value=tf.math.log([1., 1., 1., 1.]) , 
                                  trainable = True, name = "sigmas")
    
    def __call__(self):
        
        d_samples = self.data_generating_model(self.mus, self.sigmas, 
                                        sigma_taus=None,alpha_LKJ=None,lambda0 = None,
                                        input_settings_global = self.input_settings_global,
                                        input_settings_learning= self.input_settings_learning, 
                                        input_settings_model= self.input_settings_model,
                                        model_type = "model")
        
        return d_samples
        
    
    def data_generating_model(self, mus, sigmas, sigma_taus, alpha_LKJ, lambda0, 
                              input_settings_global,
                              input_settings_learning, 
                              input_settings_model,
                              model_type):
        
         # set seed
         tf.random.set_seed(input_settings_global["seed"])
        
         # initialize variables
         if model_type == "expert":
            rep = input_settings_learning["rep_exp"]
            B = 1
         else:
            rep = input_settings_learning["rep_mod"]
            B = input_settings_learning["B"]
        
         X_design = input_settings_model["X"]
         idx = input_settings_model["X_idx"]
         max_number = input_settings_model["threshold_max"]
         temp = input_settings_learning["temp"]
         
         # sort by group and perc_urban in decreasing order
         df = pd.DataFrame(tf.squeeze(X_design)).sort_values(by=[2,3,1])
         # standardize metric predictor
         df[1] = (df[1] - df[1].mean())/df[1].std() 
         # reshape model matrix and create tensor
         X_model = tf.cast(tf.gather(df, idx), tf.float32)
         
         # sample from priors
         # intercept
         beta0 = tfd.Normal(mus[0], tf.exp(sigmas[0])).sample((B,rep,1))
         # percent_urban (metric predictor)
         beta1 = tfd.Normal(mus[1], tf.exp(sigmas[1])).sample((B,rep,1))
         # historical votings GOP vs. Dem
         beta2 = tfd.Normal(mus[2], tf.exp(sigmas[2])).sample((B,rep,1))
         # historical votings Swing vs. Dem
         beta3 = tfd.Normal(mus[3], tf.exp(sigmas[3])).sample((B,rep,1))
         
         # linear predictor
         betas = tf.stack([beta0, beta1, beta2, beta3], -1)
         mu = tf.exp(tf.matmul(X_model, betas, transpose_b=True))
         
         # compute N_obs
         N = len(idx)
         # constant outcome vector
         c = tf.ones((B,rep,1,max_number))*tf.cast(tf.range(0,max_number), tf.float32)
         # compute pmf value
         pi = tfd.Poisson(rate=mu).prob(c)
         # prevent zero value (causes inf for log)
         pi = tf.where(pi < 1.8*10**(-30), 1.8*10**(-30), pi)
         ## sample n-dimensional one-hot-like
         # Gumbel-Max Softmax trick
         # sample from uniform
         u = tfd.Uniform(0,1).sample((B,rep,N,max_number))
         # generate a gumbel sample
         g = -tf.math.log(-tf.math.log(u))
         # softmax trick
         w  = tf.nn.softmax(tf.math.divide(tf.math.add(tf.math.log(pi),g),temp))
         
         # apply transformation
         y_obs = tf.reduce_sum(tf.multiply(w,c), -1)    
         
         # select groups
         # shape = (B,N,n_gr,N_obs)
         y_obs_gr = tf.stack([y_obs[:,:,i:j] for i,j in zip([0,2,4], [2,4,6])], axis=2)
         
         # combine groups (avg. over N_obs)
         # shape: (B,rep)
         y_groups = tf.reduce_mean(y_obs_gr,-1)
        
         # R2
         R2 = tf.divide(tf.math.reduce_variance(tf.squeeze(mu, axis=-1),2),
                         tf.math.reduce_variance(y_obs,2))
         
         return {"y_groups": y_groups,
                 "y_obs": y_obs,
                 "R2": R2
                 }

class Neg_Binom_Model(tf.Module):
    def __init__(self, input_settings_model, input_settings_learning,
                 input_settings_global): 
        super(Neg_Binom_Model, self).__init__()
        tf.random.set_seed(input_settings_global["seed"])
        
        self.input_settings_global = input_settings_global
        self.input_settings_learning = input_settings_learning
        self.input_settings_model = input_settings_model
        
        self.mus = tf.Variable(initial_value=[1., 1., 1., 1.], 
                               trainable = True, name = "mus")
        self.sigmas = tf.Variable(initial_value=tf.math.log([0.01, 0.01, 0.01, 0.01]) , 
                                  trainable = True, name = "sigmas")
        self.lambda0 = tf.Variable(initial_value=tf.math.log(1.01) ,  
                                  trainable = True, name = "lambda0")
    
    def __call__(self):
        
        d_samples = self.data_generating_model(self.mus, self.sigmas, 
                             sigma_taus=None,alpha_LKJ=None,lambda0=self.lambda0,
                             input_settings_global = self.input_settings_global,
                             input_settings_learning = self.input_settings_learning, 
                             input_settings_model= self.input_settings_model,
                             model_type = "model")
        
        return d_samples

    def data_generating_model(self, mus, sigmas,sigma_taus,alpha_LKJ, lambda0, 
                              input_settings_global,
                              input_settings_learning, 
                              input_settings_model,
                              model_type):  
         
         # set seed
         tf.random.set_seed(input_settings_global["seed"])
       
         # initialize variables
         if model_type == "expert":
           rep = input_settings_learning["rep_exp"]
           B = 1
         else:
           rep = input_settings_learning["rep_mod"]
           B = input_settings_learning["B"]
        
         X_design = input_settings_model["X"]
         idx = input_settings_model["X_idx"]
         max_number = input_settings_model["threshold_max"]
         temp = input_settings_learning["temp"]
        
         # sort by group and perc_urban in decreasing order
         df = pd.DataFrame(tf.squeeze(X_design)).sort_values(by=[2,3,1])
         # standardize metric predictor
         df[1] = (df[1] - df[1].mean())/df[1].std() 
         # reshape model matrix and create tensor
         X_model = tf.cast(tf.gather(df, idx), tf.float32)
         
         # compute N_obs
         N = len(idx)
         
         # sample from priors
         # intercept
         beta0 = tfd.Normal(mus[0], tf.exp(sigmas[0])).sample((B,rep,1))
         # percent_urban (metric predictor)
         beta1 = tfd.Normal(mus[1], tf.exp(sigmas[1])).sample((B,rep,1))
         # historical votings GOP vs. Dem
         beta2 = tfd.Normal(mus[2], tf.exp(sigmas[2])).sample((B,rep,1))
         # historical votings Swing vs. Dem
         beta3 = tfd.Normal(mus[3], tf.exp(sigmas[3])).sample((B,rep,1))
         
         if input_settings_model["sigma_prior"] == "gamma":
             sigma = tfd.Gamma(concentration=1., rate=tf.exp(lambda0)).sample((B,rep,1,1))
             
         if input_settings_model["sigma_prior"] == "exponential-avg":
             sigma = tf.reduce_mean(tfd.Exponential(tf.exp(lambda0)).sample((B,rep,rep,1,1)),1)
          
         if input_settings_model["sigma_prior"] == "gamma-avg":
             sigma = tfd.Gamma(concentration=rep, rate=rep*tf.exp(lambda0)).sample((B,rep,1,1))
         
         # for stability
         sigma = tf.broadcast_to(sigma, (B,rep,N,1))
         
         # linear predictor
         betas = tf.stack([beta0, beta1, beta2, beta3], -1)
         mu = tf.matmul(X_model, betas, transpose_b=True)
         theta = tf.exp(mu)
         # constant outcome vector
         c = tf.ones((B,rep,1,max_number))*tf.cast(tf.range(0,max_number), tf.float32)
         # compute pmf value
         negbinom = tfd.NegativeBinomial.experimental_from_mean_dispersion(mean=theta, dispersion=sigma)
         pi = negbinom.prob(c)
         # prevent zero value (causes inf for log)
         pi = tf.where(pi < 1.8*10**(-30), 1.8*10**(-30), pi)
         ## sample n-dimensional one-hot-like
         # Gumbel-Max Softmax trick
         # sample from uniform
         u = tfd.Uniform(0,1).sample((B,rep,N,max_number))
         # generate a gumbel sample
         g = -tf.math.log(-tf.math.log(u))
         # softmax trick
         w  = tf.nn.softmax(tf.math.divide(tf.math.add(tf.math.log(pi),g),temp))
         
         # reparameterization
         y_obs = tf.reduce_sum(tf.multiply(w,c), -1)    
         
         # select groups
         # shape = (B,N,n_gr,N_obs)
         y_obs_gr = tf.stack([y_obs[:,:,i:j] for i,j in zip([0,2,4], [2,4,6])], axis=2)
         
         # combine groups (avg. over N_obs)
         # shape: (B,rep)
         y_groups = tf.reduce_mean(y_obs_gr,-1)
        
         # R2
         R2 = tf.divide(tf.math.reduce_variance(tf.squeeze(theta, axis=-1),2),
                         tf.math.reduce_variance(y_obs,2))
         
         # for dispersion parameter
         ratio_m_var = tf.reduce_mean(y_obs,-1)/tf.math.reduce_variance(y_obs,-1)
        
         
         return {"y_groups":y_groups, 
                "y_obs": y_obs,
                "R2":R2,
                "ratio_m_var":ratio_m_var}
     
class MLM_Model(tf.Module):
    def __init__(self, input_settings_model, input_settings_learning, 
                 input_settings_global): 
        super(MLM_Model, self).__init__()
        tf.random.set_seed(input_settings_global["seed"])
        
        self.input_settings_global = input_settings_global
        self.input_settings_learning = input_settings_learning
        self.input_settings_model = input_settings_model
        
        # model coefficients
        self.mus = tf.Variable(initial_value=[230., 30.], trainable=True, name = "mus")
        self.sigmas = tf.Variable(initial_value=tf.math.log([10., 1.]), trainable=True, name = "sigmas")
        # random effects
        self.sigma_taus = tf.Variable(initial_value=tf.math.log([10., 1.]), trainable=True, name = "sigma_taus") 
        # param for correlation prior (between random effects)
        self.alpha_LKJ = tf.Variable(initial_value=1., trainable=False, name="alpha_LKJ")
        # param for random noise prior
        self.lambda0 = tf.Variable(initial_value=tf.math.log(0.5), trainable=True, name="lambda0") #0.028
        
    def __call__(self):
        
        d_samples =  self.data_generating_model(self.mus, self.sigmas, self.sigma_taus, 
                                                self.alpha_LKJ, self.lambda0, 
                                                input_settings_learning = self.input_settings_learning, 
                                                input_settings_model = self.input_settings_model, 
                                                input_settings_global = self.input_settings_global, 
                                                model_type = "model", 
                                                expert_R2_0= self.input_settings_model["model_specific"]["R2_0"], 
                                                expert_R2_1= self.input_settings_model["model_specific"]["R2_1"])
                
        return d_samples


    def data_generating_model(self, mus, sigmas, sigma_taus, alpha_LKJ, lambda0, 
                              input_settings_learning, input_settings_model,
                              input_settings_global, model_type, 
                              expert_R2_0=None, expert_R2_1=None):
        
        # set seed
        tf.random.set_seed(input_settings_global["seed"])
      
        # initialize variables
        if model_type == "expert":
          rep = input_settings_learning["rep_exp"]
          B = 1
          rep_mod = input_settings_learning["rep_mod"]
        else:
          rep = input_settings_learning["rep_mod"]
          B = input_settings_learning["B"]
        
        X_days = input_settings_model["X"]
        id0x = list(input_settings_model["X_idx"])
        idx = list(input_settings_model["X_idx"])
        N_subj = input_settings_model["model_specific"]["N_subj"]
        N_days = input_settings_model["model_specific"]["N_days"]
        
        sd_x = tf.math.reduce_std(X_days)
        Z_days = (X_days)/sd_x
        
        # model
        model = tfd.JointDistributionNamed(dict(
            ## sample from priors
            # fixed effects: beta0, beta1
            beta0 = tfd.Sample(tfd.Normal(loc=mus[0], scale=tf.exp(sigmas[0])), rep),
            beta1 = tfd.Sample(tfd.Normal(loc=mus[1], scale=tf.exp(sigmas[1])), rep),
            # sd random effects: tau0, tau1
            tau0 = tfd.Sample(tfd.TruncatedNormal(loc=0., scale=tf.exp(sigma_taus[0]), low=0., high=500), rep), 
            tau1 = tfd.Sample(tfd.TruncatedNormal(loc=0., scale=tf.exp(sigma_taus[1]), low=0., high=500), rep),
            # LKJ prior; compute corr matrix: rho01
            corr_matrix = tfd.Sample(tfd.LKJ(2, alpha_LKJ),rep)
            )).sample(B)
        
        # sd random noise: sigma
        if input_settings_model["sigma_prior"] == "gamma":
             sigma = tfd.Gamma(concentration=1., rate=tf.exp(lambda0)).sample((B,rep,1))
             
        if input_settings_model["sigma_prior"] == "exponential-avg":
             sigma = tf.reduce_mean(tfd.Exponential(tf.exp(lambda0)).sample((B,rep,rep,1)),1)
          
        if input_settings_model["sigma_prior"] == "gamma-avg":
             sigma = tfd.Gamma(concentration=rep, rate=rep*tf.exp(lambda0)).sample((B,rep,1))
         
        # broadcast sigma to the needed shape
        sigma_m = tf.squeeze(sigma, -1)
        sigma = tf.broadcast_to(sigma, (B,rep,N_subj*N_days))
        
        ## compute covariance matrix
        tau0_m = tf.reduce_mean(model["tau0"],1)
        tau1_m = tf.reduce_mean(model["tau1"],1)
        # SD matrix
        S = tf.linalg.diag(tf.stack([tau0_m, tau1_m], -1))
        # covariance matrix: Cov=S*R*S
        # for stability
        corr_mat = tf.linalg.diag(diagonal=(1.,1.), padding_value=tf.reduce_mean(model["corr_matrix"]))
        # compute cov mat
        model["cov_mx_subj"] = tf.matmul(tf.matmul(S,corr_mat),S)
        
        # generate by-subject random effects: T0s, T1s
        model["subj_rfx"] = tfd.Sample(tfd.MultivariateNormalTriL(loc= [0,0], 
                      scale_tril=tf.linalg.cholesky(model["cov_mx_subj"])), N_subj).sample()
        
        ## broadcast by-subject rfx and betas to needed shape
        model["T0s"] = tf.reshape(tf.broadcast_to(model["subj_rfx"][:,:,0,None], 
                                                  (B, N_subj, N_days)),  (B,N_subj*N_days))
        model["T1s"] = tf.reshape(tf.broadcast_to(model["subj_rfx"][:,:,1,None], 
                                                  (B,N_subj, N_days)),  (B,N_subj*N_days))
        model["beta0_b"] = tf.broadcast_to(model["beta0"][:,:,None], (B,rep,N_subj*N_days))
        model["beta1_b"] = tf.broadcast_to(model["beta1"][:,:,None], (B,rep,N_subj*N_days))
        
        ## compute mu_s
        # beta_days = beta_1 + T_1s 
        model["beta_days"] = tf.add(model["beta1_b"], tf.expand_dims(model["T1s"],1)) 
        # beta_intercept = beta_0 + T_0s 
        model["beta_intercept"] = tf.add(model["beta0_b"], tf.expand_dims(model["T0s"],1)) 
        
        # mu_s = beta_intercept + beta_days*Z_days
        model["mu"] = model["beta_intercept"] + model["beta_days"] * Z_days
        #model["mu_x"] = (model["beta_intercept"]-(model["beta_days"]*sd_x)/med_x) + (model["beta_days"]/sd_x*X_days)
        
        ## sample observed data
        # y_obs ~ Normal(mu, sigma)
        model["y_obs"] = tfd.Normal(model["mu"], sigma).sample()
                
        ## Transform observed data for training and mapping expert information to model 
        # reshape linear predictor from (B,rep,N_obs) to (B,rep,N_subj, N_days)
        mu_m = tf.stack([model["mu"][:,:,i::N_days] for i in range(N_days)], -1)
        # reshape observed data from (B,rep,N_obs) to (B,rep,N_subj, N_days)
        y_m = tf.stack([model["y_obs"][:,:,i::N_days] for i in range(N_days)], -1)
    
        # predictive dist days
        days = tf.reduce_mean(tf.gather(mu_m, indices=idx, axis=3), 2)
        
        # average RT per day / shape: (B,N_days)
        # mapping: average RT per day <-> mu0 and mu1
        days_m = tf.math.reduce_mean(tf.gather(y_m, indices=idx, axis=3), (1,2))
        
        # standard deviation of average RT per day / shape: (B,1)
        # mapping: uncertainty wrt average RT per day <-> sigma0 and sigma1
        days_E_sd = tf.math.reduce_std(tf.reduce_mean(tf.gather(mu_m, indices=idx, axis=3), 2), 1)
        
        # average RT of day 0 / shape: (B,1)
        # mapping: average RT day0 <-> mu0
        day0_m = days_m[:,0]
        day_m = days_m[:,-1]

       
        if model_type == "expert":
            ## provide quartiles for selected days
            #days_perc = tf.stack([tfp.stats.percentile(days[0,:,i], 
            #                                        perc, axis=-1) for i in range(len(idx))],-1)
            
            # sample uniformly from percentiles and broadcast to shape (B,rep,idx)
            #days = tf.stack([get_quantiles(days_perc[:,i], rep, B) for i in range(len(idx))],-1)
            
            #days = tf.broadcast_to(days[0,:,:], (B,rep,len(idx)))
            
            ## provide R2 as histogram
            # R2 for day0: R2 = tau0/(sigma+tau0)
            # mapping = variance explained by between subj var at day0 <-> tau_sigma0
            R2_0 = tf.divide((tf.math.reduce_variance(model["mu"][:,:,id0x[0]::N_days],-1)), 
                             tf.math.reduce_variance(model["y_obs"][:,:,id0x[0]::N_days],-1))
            
            # broadcast to model tensor shape for input as argument
            R2_0_m = tf.broadcast_to(R2_0[:,0:rep_mod], (B,rep_mod))
            
            # resolution according to sd ("expert part" in loss)
            target_sd_day0 = tf.sqrt(tf.multiply(R2_0,tf.math.reduce_variance(model["y_obs"][:,:,id0x[0]::N_days],-1)))     
            # compute sd of linear predictor (tau0) ("model part" in loss)
            sd_mu_day0 = tf.sqrt(tf.math.reduce_variance(model["mu"][:,:,id0x[0]::N_days],-1))
            
            # R2 for day "x": R2 = tau0+tau1/(sigma+tau0+tau1) 
            # mapping = variance explained by between subj var (total) <-> tau_sigma0, tau_sigma1
            R2_1 = tf.divide(tf.math.reduce_variance(model["mu"][:,:,id0x[1]::N_days],-1), 
                             tf.math.reduce_variance(model["y_obs"][:,:,id0x[1]::N_days],-1))
            
            # broadcast to model tensor shape for input as argument
            R2_1_m = tf.broadcast_to(R2_1[:,0:rep_mod], (B,rep_mod))
            # R2_1 = tf.broadcast_to(R2_1_total[0,:], (B,rep))
            
            # resolution according to sd ("expert part" in loss)
            target_sd_day = tf.sqrt(tf.multiply(R2_1, tf.math.reduce_variance(model["y_obs"][:,:,id0x[1]::N_days],-1)))      
            # compute sd of linear predictor (tau0) ("model part" in loss)
            sd_mu_day = tf.sqrt(tf.math.reduce_variance(model["mu"][:,:,id0x[1]::N_days],-1))
            
        if model_type == "model":
            # R2 for day0: R2 = tau0/(sigma+tau0)
            # mapping = variance explained by between subj var at day0 <-> tau_sigma0
            # resolution of R2 according to sd ("expert part" in loss)
            target_sd_day0 = tf.sqrt(tf.multiply(expert_R2_0, tf.math.reduce_variance(model["y_obs"][:,:,id0x[0]::N_days],-1)))     
            # compute sd of linear predictor (tau0) ("model part" in loss)
            sd_mu_day0 = tf.sqrt(tf.math.reduce_variance(model["mu"][:,:,id0x[0]::N_days],-1))
            # compute resulting R2 for day0 (only for tracking)
            R2_0 = tf.divide(tf.math.reduce_variance(model["mu"][:,:,id0x[0]::N_days],-1), 
                             tf.math.reduce_variance(model["y_obs"][:,:,id0x[0]::N_days],-1))
            
            R2_0_m = None
            # R2 for day "x": R2 = tau0+tau1/(sigma+tau0+tau1) 
            # mapping = variance explained by between subj var (total) <-> tau_sigma0, tau_sigma1
            # resolution according to sd ("expert part" in loss)
            target_sd_day = tf.sqrt(tf.multiply(expert_R2_1, tf.math.reduce_variance(model["y_obs"][:,:,id0x[1]::N_days],-1)))      
            # compute sd of linear predictor (tau0) ("model part" in loss)
            sd_mu_day = tf.sqrt(tf.math.reduce_variance(model["mu"][:,:,id0x[1]::N_days],-1))
            # compute resulting R2 for day "x" (only for tracking)
            R2_1 = tf.divide(tf.math.reduce_variance(model["mu"][:,:,id0x[1]::N_days],-1), 
                              tf.math.reduce_variance(model["y_obs"][:,:,id0x[1]::N_days],-1))
            R2_1_m = None
        # standard deviation of observed data 
        # mapping: total variation in the observed data (parameter knowledge) <-> lambda0  
        sd_yobs = tf.math.reduce_std(tf.gather(y_m, indices=idx, axis=3)[:,0,:,:], 1)
        
        # sd of average RT of day0 
        # mapping: uncertainty wrt average of day0 <-> sigma0
        day0_E_sd = days_E_sd[:,0]
        
        return {"days_m":days_m, "days":days, "y_m":y_m,"mu_m":mu_m,"day_m":day_m, 
                "R2_0": R2_0_m, "R2_1": R2_1_m,
                "days_E_sd":days_E_sd, "day0_m": day0_m, "day0_E_sd": day0_E_sd,
                "days_sd": sigma_m, "tau0":model["tau0"], "tau1":model["tau1"],
                "target_sd_day0":target_sd_day0, "sd_mu_day0":sd_mu_day0,
                "target_sd_day":target_sd_day, "sd_mu_day":sd_mu_day,
                "sd_yobs": sd_yobs, "y_obs":model["y_obs"],
                "mu": model["mu"], "T0s": model["T0s"], "T1s": model["T1s"], "S":S,
                "rfx":model["subj_rfx"]
                } 



