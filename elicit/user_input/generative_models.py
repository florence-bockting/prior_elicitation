import tensorflow as tf
import tensorflow_probability as tfp

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
                    epred = epred,
                    prior_samples = prior_samples               
                    )