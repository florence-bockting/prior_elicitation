import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
tfd = tfp.distributions


# define generative model
class GenerativeBinomialModel(tf.Module):
    def __call__(self, 
                prior_samples,        
                design_matrix,           
                total_count,       
                **kwargs        
                ):  

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
        
   # define generative model
class GenerativePoissonModel(tf.Module):
    def __call__(self, 
                prior_samples,        
                design_matrix,           
                total_count,       
                **kwargs        
                ):  

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
                prior_samples,        
                design_matrix,       
                **kwargs        
                ):  
        
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
        R2 = tf.divide(tf.math.reduce_variance(epred, -1), 
                       tf.math.reduce_variance(ypred, -1))
        
        return dict(likelihood = likelihood,     
                    ypred = ypred,   
                    epred = epred,
                    prior_samples = prior_samples,
                    mean_effects = mean_effects,
                    marginal_ReP = marg_ReP,
                    marginal_EnC = marg_EnC,
                    R2 = R2
                    )
