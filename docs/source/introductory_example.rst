.. Make-My-Prior documentation master file, created by
   sphinx-quickstart on Mon Oct 30 10:23:38 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

An introductory example
#######################

Background and problem formulation
==================================
.. image:: _static/toy_example_measure.png
  :width: 200
  :alt: intro toy example
  
Let’s consider the following example: We plan to conduct a study on the height of people (in cm) in our neighborhood. To model the height data, we use a normal likelihood as the data-generating model, with average height :math:`\mu` and random variation :math:`\sigma`. Furthermore, to keep the example as simple as possible, we ignore any individual differences and other potential factors influencing the height of people. Thus, we use an intercept-only model as data-generating model:

.. math::

	\begin{align*}
		\theta &:= (\mu, \sigma)\\
		\mu &= \mathbf{1}_N \beta_0 \\
		height_i &\sim \text{Normal}(\mu, \sigma)
	\end{align*}

As we have prior knowledge about the expected :math:`height_i`​ and would like to incorporate this information into our model, we choose a Bayesian modeling approach. This approach allows us to include our prior knowledge by specifying prior distributions for the model parameters: 

.. math::
	\theta \sim p(\theta).

The challenge is that our prior knowledge pertains to :math:`height_i` (i.e., the outcome variable), but we need to specify prior distributions for :math:`\theta` (i.e., the model parameters). 

+ :math:`\mathcal Q`: Which prior distributions for the model parameters accurately reflects our prior knowledge on the outcome variable? :math:`[p(\theta) = ??]`


Prior elicitation methods
=========================

*Prior elicitation methods* are designed to translate expert information into corresponding prior distributions for the model parameters. We will focus on a specific group of prior elicitation methods that address specifically the following translation problem: The expert provides information about the outcome variable (here: :math:`height_i`) and we need a translation of this information to prior distributions :math:`p(\theta)` that correctly reflect this knowledge. To understand this point, it helps to think generatively. Suppose we specify particular prior distributions :math:`p(\mu)` and :math:`p(\sigma)`. To assess whether these priors reflect our knowledge, we can sample values from them: :math:`\sigma^{S} \sim p(\sigma)` and :math:`\mu^{S} \sim p(\mu)`. Next, we plug these sampled values into our likelihood and simulate data: :math:`height_i^{S} \sim \text{Normal}(\mu^{S}, \sigma^{S})`. Finally, we can ask ourselves: Do these simulated data :math:`height_i^{S}` align with our expectations about :math:`height_i`, then the set of select prior distributions correctly reflects our prior knowledge.

.. image:: _static/generative_workflow.png
  :width: 400
  :alt: generative approach
  
**Parameteric prior elicitation**

In this specific variant, the expert provides information about the prior distribution family for each model parameter. The goal becomes then to learn the hyperparameter :math:`\lambda` (of this parametric model priors) based on prior knowledge about *height*.

|:arrow_forward:| :doc:`Go to the implementation of the parametric-prior variant for our toy example <notebooks/books/toy_parametric_prior>`

**Deep prior elicitation**

In this second variant, we train normalizing flows (NFs) on model simulations to induce a flexible family of prior distributions. NFs are generative models that allow exact density evaluation and entail specialized deep neural networks with trainable parameters :math:`\lambda` that can learn highly complex transformations. Thus, in this variant no specification of prior distribution families for the model parameters is necessary, as we learn a joint prior density on the model parameters. 

This variant is still under construction. The toy example will come soon.