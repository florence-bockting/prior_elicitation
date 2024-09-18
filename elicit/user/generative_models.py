import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
tfd = tfp.distributions
tfb = tfp.bijectors

from user.custom_functions import group_quantiles

class GenerativeBinomialModel(tf.Module):
    def __call__(self, 
                prior_samples,        
                design_matrix,           
                total_count,       
                **kwargs        
                ):  
        """
        Binomial model with one continuous predictor.

        Parameters
        ----------
        prior_samples : dict
            samples from prior distributions.
        design_matrix : tf.Tensor
            design matrix.
        total_count : int
            total counts of Binomial model.
        **kwargs : keyword argument, optional
            additional keyword arguments

        Returns
        -------
        dictionary with the following keys:
            
        - likelihood: model likelihood
        - ypred: model predictions; for discrete likelihoods ypred=None as it
          will be approximated via the softmax-gumble trick
        - epred: predictions of linear predictor
        - prior samples: samples from prior distributions

        """

        # linear predictor
        theta = design_matrix @ tf.expand_dims(prior_samples, axis=-1)

        # map linear predictor to theta
        epred = tf.sigmoid(theta)
        
        # define likelihood
        likelihood = tfd.Binomial(
            total_count = total_count, 
            probs = epred
        )
        
        return dict(likelihood = likelihood,     
                    ypred = None,                 
                    epred = epred,
                    prior_samples = prior_samples                 
                    )

class GenerativeBinomialModelVariant(tf.Module):
    def __call__(self, 
                prior_samples,        
                design_matrix,      
                **kwargs        
                ):  
        """
        Binomial model with one continuous predictor.

        Parameters
        ----------
        prior_samples : dict
            samples from prior distributions.
        design_matrix : tf.Tensor
            design matrix.
        total_count : int
            total counts of Binomial model.
        **kwargs : keyword argument, optional
            additional keyword arguments

        Returns
        -------
        dictionary with the following keys:
            
        - likelihood: model likelihood
        - ypred: model predictions; for discrete likelihoods ypred=None as it
          will be approximated via the softmax-gumble trick
        - epred: predictions of linear predictor
        - prior samples: samples from prior distributions

        """

        # linear predictor
        theta = design_matrix @ tf.expand_dims(prior_samples, axis=-1)

        # map linear predictor to theta
        epred = theta[:,:,:,0]
        
        # define likelihood
        likelihood = tfd.Normal(
            loc = epred, 
            scale = tf.ones(epred.shape)
        )
        
        # sample prior predictions
        ypred = likelihood.sample()
        
        return dict(likelihood = likelihood,     
                    ypred = ypred,                 
                    epred = epred,
                    prior_samples = prior_samples                 
                    )

class GenerativePoissonModel(tf.Module):
    def __call__(self, 
                prior_samples,        
                design_matrix,      
                **kwargs        
                ):  
        """
        Poisson model with one continuous predictor and one categorical 
        predictor with three levels.

        Parameters
        ----------
        prior_samples : dict
            samples from prior distributions.
        design_matrix : tf.Tensor
            design matrix.
        **kwargs : keyword argument, optional
            additional keyword arguments

        Returns
        -------
        dictionary with the following keys:
            
        - likelihood: model likelihood
        - ypred: model predictions; for discrete likelihoods ypred=None as it
          will be approximated via the softmax-gumble trick 
        - epred: predictions of linear predictor
        - prior samples: samples from prior distributions

        """
        # linear predictor
        theta = design_matrix @ tf.expand_dims(prior_samples, -1)
        
        # map linear predictor to theta
        epred = tf.exp(theta)
        
        # define likelihood
        likelihood = tfd.Poisson(
            rate = epred
        )
        
        return dict(likelihood = likelihood,     
                    ypred = None,   
                    epred = epred[:,:,:,0],
                    prior_samples = prior_samples               
                    )

class GenerativeNormalModel(tf.Module):
    def __call__(self, 
                ground_truth, 
                prior_samples,        
                design_matrix,       
                **kwargs        
                ):  
        # compute linear predictor term
        epred = prior_samples[:,:,0:6] @ tf.transpose(design_matrix)
        
        if ground_truth:
            sigma = tf.expand_dims(prior_samples[:,:,-1], -1)
        else:
            sigma = tf.abs(tf.expand_dims(prior_samples[:,:,-1], -1))
        
        # define likelihood
        likelihood = tfd.Normal(
            loc = epred, 
            scale = sigma #tf.ones(epred.shape)*0.11
            )
        
        # sample prior predictive data
        ypred = likelihood.sample()
        
        # create contrast matrix
        cmatrix = tf.cast(pd.DataFrame(design_matrix).drop_duplicates(), tf.float32)
        
        # compute custom target quantity (here: group-differences)
        samples_grouped = tf.stack(
            [
                tf.boolean_mask(ypred, 
                                tf.math.reduce_all(cmatrix[i,:] == design_matrix, axis = 1),
                                    axis = 2) for i in range(cmatrix.shape[0])
            ], axis = -1)

        # compute mean difference between groups
        effect_list = []
        diffs = [(0,3), (1,4), (2,5)]
        
        for i in range(len(diffs)):
            # compute group difference
            diff = tf.math.subtract(
                samples_grouped[:, :, :, diffs[i][0]],
                samples_grouped[:, :, :, diffs[i][1]]
            )
            # average over individual obs within each group
            diff_mean = tf.reduce_mean(diff, axis=2)
            # collect all mean group differences
            effect_list.append(diff_mean)

        mean_effects = tf.stack(effect_list, axis=-1)

        # compute marginals
        ## factor repetition: new, repeated
        marg_ReP = tf.reduce_mean(
            tf.stack([tf.math.add_n([samples_grouped[:, :, :, i] for i in j]) for j in [range(3),range(3,6)]], 
                     axis = -1), 
            axis = 2)
     
        ## factor Encoding depth: deep, standard, shallow
        marg_EnC = tf.reduce_mean(
            tf.stack([tf.math.add_n([samples_grouped[:, :, :, i] for i in j]) for j in [[0,3],[1,4],[2,5]]], 
                     axis = -1), 
            axis = 2)
        
        # compute R2
        log_R2 = tf.math.subtract(tf.math.log(tf.math.reduce_variance(epred, -1)),
                                tf.math.log(tf.math.reduce_variance(ypred, -1)))
     
        prior_samples = tf.concat([prior_samples[:,:,:-1],
                                   prior_samples[:,:,-1][:,:,None]],
                                  axis=-1)
        
        return dict(likelihood = likelihood,     
                    ypred = ypred,   
                    epred = epred,
                    prior_samples = prior_samples,
                    mean_effects = mean_effects,
                    marginal_ReP = marg_ReP,
                    marginal_EnC = marg_EnC,
                    log_R2 = log_R2,
                    sigma = sigma
                    )

class GenerativeNormalModel_param(tf.Module):
    def __call__(self, 
                ground_truth,
                prior_samples,        
                design_matrix,       
                **kwargs        
                ):  
        """
        Normal model for a 2 x 3 factorial design.

        Parameters
        ----------
        prior_samples : dict
            samples from prior distributions.
        design_matrix : tf.Tensor
            design matrix.
        **kwargs : keyword argument, optional
            additional keyword arguments

        Returns
        -------
        dictionary with the following keys:
            
        - likelihood: model likelihood
        - ypred: model predictions; for discrete likelihoods ypred=None as it
          will be approximated via the softmax-gumble trick 
        - epred: predictions of linear predictor
        - prior samples: samples from prior distributions
        - mean_effects: Difference between both factors for each level of factor 2
        - marginal_ReP: marginal distribution of factor 1
        - marginal_EnC: marginal distribution of factor 2
        - R2: variance explained
        - sigma: samples from the noise parameter of the normal likelihood
        """
        # compute linear predictor term
        epred = prior_samples[:,:,0:6] @ tf.transpose(design_matrix)
        
        # define likelihood
        likelihood = tfd.Normal(
            loc = epred, 
            scale = tf.expand_dims(prior_samples[:,:,-1], -1)
            )
        
        # sample prior predictive data
        ypred = likelihood.sample()
        
        # create contrast matrix
        cmatrix = tf.cast(pd.DataFrame(design_matrix).drop_duplicates(), tf.float32)
        
        # compute custom target quantity (here: group-differences)
        samples_grouped = tf.stack(
            [
                tf.boolean_mask(ypred, 
                                tf.math.reduce_all(cmatrix[i,:] == design_matrix, axis = 1),
                                    axis = 2) for i in range(cmatrix.shape[0])
            ], axis = -1)

        # compute mean difference between groups
        effect_list = []
        diffs = [(0,3), (1,4), (2,5)]
        
        for i in range(len(diffs)):
            # compute group difference
            diff = tf.math.subtract(
                samples_grouped[:, :, :, diffs[i][0]],
                samples_grouped[:, :, :, diffs[i][1]]
            )
            # average over individual obs within each group
            diff_mean = tf.reduce_mean(diff, axis=2)
            # collect all mean group differences
            effect_list.append(diff_mean)

        mean_effects = tf.stack(effect_list, axis=-1)

        # compute marginals
        ## factor repetition: new, repeated
        marg_ReP = tf.reduce_mean(
            tf.stack([tf.math.add_n([samples_grouped[:, :, :, i] for i in j]) for j in [range(3),range(3,6)]], 
                     axis = -1), 
            axis = 2)
     
        ## factor Encoding depth: deep, standard, shallow
        marg_EnC = tf.reduce_mean(
            tf.stack([tf.math.add_n([samples_grouped[:, :, :, i] for i in j]) for j in [[0,3],[1,4],[2,5]]], 
                     axis = -1), 
            axis = 2)
        
        # compute R2
        log_R2 = tf.math.subtract(tf.math.log(tf.math.reduce_variance(epred, -1)),
                                tf.math.log(tf.math.reduce_variance(ypred, -1)))
        
        # prior samples on correct scale
     
        return dict(likelihood = likelihood,     
                    ypred = ypred,   
                    epred = epred,
                    prior_samples = prior_samples,
                    mean_effects = mean_effects,
                    marginal_ReP = marg_ReP,
                    marginal_EnC = marg_EnC,
                    log_R2 = log_R2,
                    sigma = prior_samples[:,:,-1]
                    )


class GenerativeMultilevelModel(tf.Module):
    def __call__(self, 
                 ground_truth,
                 prior_samples,        
                 design_matrix, 
                 selected_days,
                 alpha_lkj,
                 N_subj,
                 N_days,
                 **kwargs        
                 ):
        """
        Multilevel model with normal likelihood, one continuous predictor and
        one by-subject random-intercept and random-slope

        Parameters
        ----------
        prior_samples : dict
            samples from prior distributions.
        design_matrix : tf.Tensor
            design matrix.
        selected_days : list of integers
            days for which expert should be queried.
        alpha_lkj : float
            parameter of the LKJ prior on the correlation which will be fixed to 1.
        N_subj : int
            number of participants.
        N_days : int
            number of days (selected days for elicitation).
        **kwargs : keyword argument, optional
            additional keyword arguments.

        Returns
        -------
        dictionary with the following keys:
            
        - likelihood: model likelihood
        - ypred: prior predictions (if likelihood is discrete `ypred=None` as it will be approximated using the Softmax-Gumble method)
        - epred: linear predictor
        - prior_samples: we use it here again as output for easier follow-up computations
        - meanperday: distribution of mean reaction time per selected day
        - R2day0: R2 for day 0 (incl. only variation of the random intercept (individual differences in RT without considering the treatment)
        - R2day9: R2 for day 9 (incl. variation of the random intercept and slope (individual differences in RT considering treatment effect)
        - mu0sdcomp:  standard deviation of linear predictor at day 0
        - mu9sdcomp:  standard deviation of linear predictor at day 9
        - sigma: random noise parameter

        """

        B = prior_samples.shape[0]
        rep = prior_samples.shape[1]
        
        # correlation matrix
        corr_matrix = tfd.LKJ(2, alpha_lkj).sample((B, rep))
        
        # SD matrix
        # shape = (B, 2)
        if ground_truth:
            omegas = tf.reduce_mean(
                        tf.gather(prior_samples, indices=[2,3], axis=-1), 
                    axis=1)
        else:
            omegas = tf.reduce_mean(
                tf.math.softplus(
                    tf.gather(prior_samples, indices=[2,3], axis=-1)
                    ), 
                axis=1)
        
        # shape = (B, 2, 2)
        S = tf.linalg.diag(omegas)
        
        # covariance matrix: Cov=S*R*S
        # shape = (B, 2, 2)
        corr_mat = tf.linalg.diag(diagonal=(1.,1.), 
                                  padding_value=tf.reduce_mean(corr_matrix))
        # compute cov matrix
        # shape = (B, 2, 2)
        cov_mx_subj = tf.matmul(tf.matmul(S,corr_mat), S)
        
        # generate by-subject random effects: T0s, T1s
        # shape = (B, N_subj, 2)
        subj_rfx = tfd.Sample(
            tfd.MultivariateNormalTriL(
                loc= [0,0], 
                scale_tril=tf.linalg.cholesky(cov_mx_subj)), 
            N_subj).sample()
        
        # broadcast by-subject random effects
        # shape = (B, N_obs, 2) with N_obs = N_subj*N_days
        taus = tf.reshape(
            tf.broadcast_to(
                tf.expand_dims(subj_rfx, axis=2), 
                shape=(B, N_subj, N_days, 2)), 
            shape=(B, N_subj*N_days, 2))
        
        # reshape coefficients
        # shape = (B, rep, N_obs, 2) with N_obs = N_subj*N_days
        betas_reshaped = tf.broadcast_to(
            tf.expand_dims(
                tf.gather(prior_samples, indices=[0,1], axis=-1),
                axis=2), 
            shape=(B, rep, N_subj*N_days, 2))
        
        ## compute betas_s
        # shape = (B, rep, N_obs, 2) with N_obs = N_subj*N_days
        betas = tf.add(betas_reshaped, tf.expand_dims(taus, axis=1)) 
        
        # compute linear predictor term
        # shape = (B, rep, N_obs) with N_obs = N_subj*N_days
        epred = tf.add(betas[:,:,:,0]*design_matrix[:,0], 
                       betas[:,:,:,1]*design_matrix[:,1])
        
        # define likelihood
        if ground_truth:
            likelihood = tfd.Normal(loc = epred,
                                    scale = tf.expand_dims(prior_samples[:,:,-1], -1)
                                    )
        else:
            likelihood = tfd.Normal(loc = epred,
                                    scale = tf.expand_dims(tf.math.softplus(prior_samples[:,:,-1]), -1)
                                    )
        
        # sample prior predictive data
        ypred = likelihood.sample()
        
        # R2 for initial day
        def f_R2(epred, ypred):
            # variance of linear predictor 
            var_epred = tf.math.reduce_variance(epred, -1) 
            # variance of difference between ypred and epred
            var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
            var_total = var_epred + var_diff
            # variance of linear predictor divided by total variance
            log_R2 = tf.subtract(tf.math.log(var_epred), tf.math.log(var_total))
            return log_R2
        
        log_R2_day0 = f_R2(epred[:,:,selected_days[0]::N_days], ypred[:,:,selected_days[0]::N_days]) 
        
        # R2 for last day
        log_R2_day9 = f_R2(epred[:,:,selected_days[-1]::N_days], ypred[:,:,selected_days[-1]::N_days]) 
  
        
        return dict(likelihood = likelihood,     
                    ypred = ypred,   
                    epred = epred,
                    prior_samples = prior_samples,
                    log_R2day0 = log_R2_day0,
                    log_R2day9 = log_R2_day9,
                    sigma = prior_samples[:,:,-1]
                    )

# generative model
class GenerativeToyModel(tf.Module):
    def __call__(self, 
                 ground_truth,
                 prior_samples, 
                 design_matrix
                 ):  
        
        # compute linear predictor term
        theta = prior_samples[:,:,0:3] @ tf.transpose(design_matrix)

        # link function
        epred = theta
        
        # define likelihood
        likelihood = tfd.Normal(
            loc = epred, 
            scale = tf.expand_dims(prior_samples[:,:,-1], -1)
        )
        
        # sample prior predictive data
        ypred = likelihood.sample()
        
        log_R2 = tf.math.subtract(tf.math.log(tf.math.reduce_variance(epred, -1)),
                                  tf.math.log(tf.math.reduce_variance(ypred, -1)))
        
        return dict(likelihood = likelihood,          # obligatory: likelihood; callable
                    ypred = ypred,                    # obligatory: prior predictive data
                    epred = epred,                    # obligatory: samples from linear predictor
                    prior_samples = prior_samples,
                    sigma = prior_samples[:,:,-1],
                    log_R2 = log_R2
                    )

class SkewnessModel(tf.Module):
    def __call__(self, 
                 ground_truth,
                 prior_samples,
                 N):  
        X = [0.]*int(N/2)+[1.]*int(N/2)
        
        mu = prior_samples[:,:,0][:,:,None]+prior_samples[:,:,1][:,:,None]*tf.expand_dims(X,0)
        #p = tf.sigmoid(mu)
        epred = mu #tf.expand_dims(p, -1)
        # if ground_truth:
        #     sigma = prior_samples[:,:,2][:,:,None]
        # else:
        #     sigma = tf.exp(prior_samples[:,:,2][:,:,None])
        
        likelihood = tfd.Normal(epred, .1)
        #likelihood = tfd.Binomial(total_count=size, 
        #                          probs=epred)
        ypred = likelihood.sample()
        
        R2 = tf.divide(tf.math.reduce_variance(epred, -1),
                       tf.math.reduce_variance(ypred, -1))
       
        return dict(likelihood = likelihood,     
                    ypred = ypred,                 
                    epred = epred,
                    prior_samples = prior_samples,
                  #  sigma = sigma,
                    R2 = R2
                    )
    
class Horseshoe(tf.Module):
    def __call__(self, ground_truth, prior_samples, design_matrix):
        if ground_truth:
            B = 1
        else:
            B = prior_samples.shape[0]
        
        rep = prior_samples.shape[1]
        N,D = design_matrix.shape
        D = D-1
        
        beta_0 = prior_samples[:,:,0][:,:,None]
        tau = prior_samples[:,:,1][:,:,None]
        sigma = prior_samples[:,:,2][:,:,None]
        
        lambda_j = tfd.HalfCauchy(loc=0., scale=1.).sample((B,rep,D))
        
        beta_1j = tfd.Normal(loc=0.,scale=lambda_j*tau).sample()
        
        # linear predictor term
        epred = beta_0 + beta_1j @ tf.transpose(design_matrix[:,1:])
        
        # Normal likelihood
        likelihood = tfd.Normal(epred, sigma)
        
        ypred = likelihood.sample()
        
        # shrinkage factor
        k_j = tf.divide(1., 1+(N/tf.square(sigma))*tf.square(tau)*tf.square(lambda_j))
        
        # effective number of nonzero coefficients
        m_eff = tf.math.reduce_sum(1-k_j,-1)
        
        log_R2 = tf.math.subtract(tf.math.log(tf.math.reduce_variance(epred, -1)),
                                  tf.math.log(tf.math.reduce_variance(ypred, -1)))
        
        return dict(likelihood = likelihood,     
                    ypred = ypred,                 
                    epred = epred,
                    prior_samples = prior_samples,
                    k_j = k_j,
                    m_eff = m_eff,
                    beta_1j = beta_1j,
                    log_R2 = log_R2
                    )
    
class RegularizedHorseshoe(tf.Module):
    def __call__(self, ground_truth, prior_samples, design_matrix, sigma, nu_local, 
                 sigma_slab, nu_slab):
        if ground_truth:
            B = 1
        else:
            B = prior_samples.shape[0]
        
        rep = prior_samples.shape[1]
        N,D = design_matrix.shape
        
       
        tau = prior_samples[:,:,1][:,:,None]
        
        c2_aux = tfd.InverseGamma(nu_slab/2, nu_slab/2).sample((B,rep,1))
        
        lambda_j = tfd.HalfStudentT(df=nu_local, loc=0., scale=1.).sample((B,rep,D))
        
        c = sigma_slab*tf.sqrt(c2_aux)
        
        lambda_tilde_j = tf.sqrt(tf.divide(
            tf.square(c)*tf.square(lambda_j),
            tf.square(c)+tf.square(lambda_j)*tf.square(tau))
            )
        
        z_j = tfd.Normal(0.,1.).sample((B,rep,D))
        
        # non-intercept coefficients get shrinkage prior
        beta_1j = z_j*lambda_tilde_j*tau
        
        # intercept gets a wider prior (no shrinkage desired)
        beta_0 = prior_samples[:,:,0][:,:,None]
        
        # linear predictor term
        epred = beta_0 + beta_1j @ tf.transpose(design_matrix)
        
        # Normal likelihood
        likelihood = tfd.Normal(epred, sigma)
        
        ypred = likelihood.sample()
        
        k_j = tf.divide(1., 1+(N/tf.square(sigma))*tf.square(tau)*tf.square(lambda_j))
        
        m_eff = tf.math.reduce_sum(1-k_j,-1)
        
        return dict(likelihood = likelihood,     
                    ypred = ypred,                 
                    epred = epred,
                    prior_samples = prior_samples,
                    k_j = k_j,
                    m_eff = m_eff,
                    beta_1j = beta_1j
                    )
    
class SkewnessModel2(tf.Module):
    def __call__(self, 
                 ground_truth,
                prior_samples,
                N
                ):  
       mu = prior_samples
       likelihood = tfd.Normal(mu, 1.)
       ypred = likelihood.sample(N)
       
       return dict(likelihood = likelihood,     
                    ypred = ypred,                 
                    epred = mu,
                    prior_samples = prior_samples                 
                    )
   
class Model0(tf.Module):
    def __call__(self, 
                 ground_truth,
                 prior_samples,
                 design_matrix,
                 total_count):  
        
        epred = prior_samples @ tf.transpose(design_matrix)
        
        probs = tf.sigmoid(epred)
        
        likelihood = tfd.Binomial(total_count = total_count, 
                                  probs = probs[:,:,:,None])
        
        return dict(likelihood = likelihood,     
                    ypred = None,                 
                    epred = epred,
                    prior_samples = prior_samples
                    )

class Model1(tf.Module):
    def __call__(self, 
                 ground_truth,
                 prior_samples,
                 design_matrix):  
            
        epred = prior_samples[:,:,:-1] @ tf.transpose(design_matrix)
        sigma = tf.abs(prior_samples[:,:,-1][:,:,None])
        
        likelihood = tfd.Normal(loc = epred, 
                                scale = sigma)
        
        ypred = likelihood.sample()
        
        group1 = ypred[:,:,0::3]
        group2 = ypred[:,:,1::3]
        group3 = ypred[:,:,2::3]
        var_ypred = tf.math.reduce_variance(ypred, -1)
        # R2
        var_epred = tf.math.reduce_variance(epred, -1) 
        # variance of difference between ypred and epred
        var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
        var_total = var_epred + var_diff
        # variance of linear predictor divided by total variance
        log_R2 = tf.subtract(tf.math.log(var_epred), tf.math.log(var_total))
        
        if ground_truth:
            log_R2 = tf.math.log(tfd.Uniform(0,1).sample(log_R2.shape))
        
        # computation of log R2
        # log_R2 = tf.math.subtract(tf.math.log(tf.math.reduce_variance(epred, -1)),
        #                           tf.math.log(tf.math.reduce_variance(ypred, -1)))
        
        prior_samples = tf.concat([prior_samples[:,:,:-1], 
                                   tf.abs(prior_samples[:,:,-1][:,:,None])],axis=-1)
        
        return dict(likelihood = likelihood,     
                    ypred = ypred,                 
                    epred = epred,
                    prior_samples = prior_samples,
                    log_R2 = log_R2,
                    group1 = group1,
                    group2 = group2,
                    group3 = group3,
                    var_epred = var_epred,
                    var_ypred = var_ypred
                    )
    
class Model2(tf.Module):
    def __call__(self, 
                 ground_truth,
                 prior_samples,
                 design_matrix):  
        
        if ground_truth:
            prior_samples = prior_samples[:,:,0,:]
        
        theta = prior_samples @ tf.transpose(design_matrix)
        
        epred = tf.exp(theta[:,:,:,None])
        
        likelihood = tfd.Poisson(epred)
 
        return dict(likelihood = likelihood,     
                    ypred = None,                 
                    epred = epred[:,:,:,0],
                    prior_samples = prior_samples
                    )

class Model3(tf.Module):
    def __call__(self, 
                 ground_truth,
                 prior_samples,
                 design_matrix):  
        
        epred = prior_samples[:,:,:-1] @ tf.transpose(design_matrix)
        sigma = tf.abs(prior_samples[:,:,-1][:,:,None])
        
        likelihood = tfd.Normal(loc = epred, 
                                scale = sigma)
        
        ypred = likelihood.sample()
        
        if ground_truth:
            # variance of linear predictor 
            var_epred = tf.math.reduce_variance(epred, -1) 
            # variance of difference between ypred and epred
            var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
            var_total = var_epred + var_diff
            # variance of linear predictor divided by total variance
            log_R2_true = tf.subtract(tf.math.log(var_epred), tf.math.log(var_total))
            
            log_R2 = tf.math.log(tfd.Uniform(0,1).sample(log_R2_true.shape))
        else:
            log_R2_true = None
            # variance of linear predictor 
            var_epred = tf.math.reduce_variance(epred, -1) 
            # variance of difference between ypred and epred
            var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
            var_total = var_epred + var_diff
            # variance of linear predictor divided by total variance
            log_R2 = tf.subtract(tf.math.log(var_epred), tf.math.log(var_total))
            
        prior_samples = tf.concat([prior_samples[:,:,:-1], 
                                   tf.abs(prior_samples[:,:,-1][:,:,None])],axis=-1)
        
        return dict(likelihood = likelihood,     
                    ypred = ypred,                 
                    epred = epred,
                    prior_samples = prior_samples,
                    log_R2 = log_R2,
                    log_R2_true = log_R2_true
                    )
    
    
class Model:
    def __call__(self, ground_truth, prior_samples):

        # data-generating model
        likelihood = tfd.Normal(loc=prior_samples[:,:,0],
                                scale=prior_samples[:,:,1])
        # prior predictive distribution
        ypred = likelihood.sample()
        
        return dict(likelihood = likelihood,     
                    ypred = ypred,                 
                    epred = None,
                    prior_samples = prior_samples                 
                    )