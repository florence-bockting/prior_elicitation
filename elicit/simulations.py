# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


# initalize generator model
class Priors(tf.Module):
    """
    Initializes the hyperparameters of the prior distributions.

    Attributes
    ----------
    ground_truth : bool
        whether samples are drawn from a true prior ('oracle')
    global_dict : dict
        dictionary containing all user and default input settings
    logger : logging method
        retrieves module name for passing it to the logger
    init_priors : dict
        initialized hyperparameters (i.e., trainable variables);
        None if ground_truth = True
    """

    def __init__(self, ground_truth: bool,
                 init_matrix_slice: tf.Tensor or None,
                 trainer: dict, parameters: dict, network: dict, expert: dict,
                 seed: int):
        """
        Initializes the hyperparameters (i.e., trainable variables)

        Parameters
        ----------
        ground_truth : bool
            True if expert data are simulated from a given ground truth (oracle)
        init_matrix_slice : tf.Tensor or None
            samples drawn from the initialization distribution to initialize
            the hyperparameter of the parametric prior distributions
            Only required for method="parametric_prior" otherwise None.
        trainer : callable
            specification of training settings and meta-information for
            workflow using :func:`trainer`
        parameters : list
            list of model parameters specified with :func:`parameter`.
        network : callable or None
            specification of neural network using a method implemented in
            :mod:`elicit.networks`.
            Only required for ``deep_prior`` method. For ``parametric_prior``
            use ``None``. Default value is ``None``.
        expert : callable
            provide input data from expert or simulate data from oracle with
            either the ``data`` or ``simulator`` method of the
            :mod:`elicit.elicit.Expert` module.
        seed : int
            seed used for learning.
        """
        self.ground_truth = ground_truth
        self.init_matrix_slice = init_matrix_slice
        self.trainer=trainer
        self.parameters=parameters
        self.network=network
        self.expert=expert

        # set seed
        tf.random.set_seed(seed)
        # initialize hyperparameter for learning (if true hyperparameter
        # are given, no initialization is needed)
        if not self.ground_truth:
            self.init_priors = intialize_priors(
                self.init_matrix_slice, self.trainer["method"], seed,
                self.parameters, self.network)
        else:
            self.init_priors = None

    def __call__(self) -> tf.Tensor:  # shape=[B,num_samples,num_params]
        """
        Samples from the initialized prior distribution(s).

        Returns
        -------
        prior_samples : tf.Tensor, shape: [B,num_samples,num_params]
            Samples from prior distribution(s).

        """
        prior_samples = sample_from_priors(
            self.init_priors, self.ground_truth, self.trainer["num_samples"],
            self.trainer["B"], self.trainer["seed"], self.trainer["method"],
            self.parameters, self.network, self.expert
            )

        return prior_samples


def intialize_priors(init_matrix_slice: dict[str, tf.Variable] or None,
                     method: str, seed: int, parameters: list[dict],
                     network: dict
                     ) -> dict[str, tf.Tensor]:
    """
    Initialize prior distributions.

    Parameters
    ----------
    init_matrix_slice : tf.Tensor or None
        samples drawn from the initialization distribution to initialize
        the hyperparameter of the parametric prior distributions
        Only for method="parametric_prior", otherwise None.
    method : str
        parametric_prior or deep_prior method as specified in
        :func:`elicit.elicit.trainer`
    seed : int
        seed of current workflow run as specified in
        :func:`elicit.elicit.trainer`
    parameters : list[dict]
        list of model parameter specifications using :func:`parameter`.
    network : dict
        specification of neural network using a method implemented in
        :mod:`elicit.networks`.
        Only required for ``deep_prior`` method. For ``parametric_prior``
        use ``None``. Default value is ``None``.

    Returns
    -------
    init_prior : dict[str, tf.Tensor]
        returns initialized prior distributions ready for prior sampling.

    """
    # set seed
    tf.random.set_seed(seed)

    if method == "parametric_prior":
        # create dict with all hyperparameters
        hyp_dict = dict()
        hp_keys=list()
        param_names=list()
        hp_names=list()
        initialized_hyperparam = dict()

        for i in range(len(parameters)):
            hyperparameter = parameters[i]["hyperparams"]
            num_hyperpar = len(hyperparameter)

            hyp_dict[f"param{i}"] = hyperparameter
            param_names += [parameters[i]["name"]]*num_hyperpar
            hp_keys += list(hyperparameter.keys())
            for j in range(num_hyperpar):
                current_key = list(hyperparameter.keys())[j]
                hp_names.append(hyperparameter[current_key]["name"])

        checked_params=list()
        for j, (i, hp_n, hp_k) in enumerate(zip(tf.unique(param_names).idx,
                                 hp_names, hp_keys)):

            hp_dict = parameters[i]["hyperparams"][hp_k]

            if hp_dict["shared"] and hp_dict["name"] in checked_params:
                pass
            else:
                # get initial value
                initial_value = init_matrix_slice[hp_n]
                # initialize hyperparameter
                initialized_hyperparam[f"{hp_k}_{hp_n}"] = tf.Variable(
                    initial_value=hp_dict["constraint"](initial_value),
                    trainable=True,
                    name=f"{hp_n}",
                )

                # save initialized priors
                init_prior = initialized_hyperparam

            if hp_dict["shared"]:
                checked_params.append(hp_n)

    if method == "deep_prior":
        # for more information see BayesFlow documentation
        # https://bayesflow.org/api/bayesflow.inference_networks.html
        INN = network["inference_network"]

        invertible_neural_network = INN(**network["network_specs"])

        # save initialized priors
        init_prior = invertible_neural_network

    return init_prior


def sample_from_priors(initialized_priors: dict[str, tf.Variable],
                       ground_truth: bool, num_samples: int, B: int, seed: int,
                       method: str, parameters: dict, network: dict,
                       expert: dict
                       ) -> tf.Tensor:  # shape=[B,num_samples,num_params]
    """
    Samples from initialized prior distributions.

    Parameters
    ----------
    initialized_priors : dict[str, tf.Variable]
        initialized prior distributions ready for prior sampling.
    ground_truth : bool
        True if expert data is simulated from ground truth.
    num_samples : int, optional
        number of samples from the prior(s). The default is 200.
    B : int, optional
        batch size. The default is 128.
    seed : int
        seed used for learning.
    method : str
        parametric_prior or deep_prior method as specified in
        :func:`elicit.elicit.trainer`
    parameters : list
        list of model parameters specified with :func:`parameter`.
    network : dict
        specification of neural network using a method implemented in
        :mod:`elicit.networks`.
        Only required for ``deep_prior`` method. For ``parametric_prior``
        use ``None``. Default value is ``None``.
    expert : callable
        provide input data from expert or simulate data from oracle with
        either the ``data`` or ``simulator`` method of the
        :mod:`elicit.elicit.Expert` module.

    Returns
    -------
    prior_samples : tf.Tensor, shape: [B, num_samples, num_params]
        Samples from prior distributions.

    """
    # set seed
    tf.random.set_seed(seed)
    if ground_truth:
        # number of samples for ground truth
        rep_true = expert["num_samples"]
        priors = []

        for prior in list(expert["ground_truth"].values()):
            # sample from the prior distribution
            prior_sample = prior.sample((1, rep_true))
            # ensure that all samples have the same shape
            if len(prior_sample.shape) < 3:
                prior = tf.expand_dims(prior_sample, -1)
            else:
                prior = prior_sample
            priors.append(prior)
        # concatenate all prior samples into one tensor
        prior_samples = tf.concat(priors, axis=-1)

    if (method == "parametric_prior") and (not ground_truth):

        priors = []
        for i in range(len(parameters)):
            # get the prior distribution family as specified by the user
            prior_family = parameters[i]["family"]

            hp_k=list(parameters[i]["hyperparams"].keys())
            init_dict={}
            for k in hp_k:
                hp_n=parameters[i]["hyperparams"][k]["name"]
                init_key = f"{k}_{hp_n}"
                init_dict[f"{k}"]=initialized_priors[init_key]
                
            # sample from the prior distribution
            priors.append(
                prior_family(**init_dict).sample((B, num_samples))
                )
        # stack all prior distributions into one tf.Tensor of
        # shape (B, S, num_parameters)
        prior_samples = tf.concat(priors, axis=-1)

    if (method == "deep_prior") and (not ground_truth):

        # initialize base distribution
        base_dist = network["base_distribution"](num_params=len(parameters))
        # sample from base distribution
        u = base_dist.sample((B, num_samples))
        # apply transformation function to samples from base distr.
        prior_samples, _ = initialized_priors(u, condition=None, inverse=False)

    return prior_samples


def softmax_gumbel_trick(epred: float, likelihood: callable,
                         upper_thres: float, temp: float, seed: int):
    """
    The softmax-gumbel trick computes a continuous approximation of ypred from
    a discrete likelihood and thus allows for the computation of gradients for
    discrete random variables.

    Currently this approach is only implemented for models without upper
    boundary (e.g., Poisson model).

    Corresponding literature:

    - Maddison, C. J., Mnih, A. & Teh, Y. W. The concrete distribution:
        A continuous relaxation of
      discrete random variables in International Conference on Learning
      Representations (2017). https://doi.org/10.48550/arXiv.1611.00712
    - Jang, E., Gu, S. & Poole, B. Categorical reparameterization with
    gumbel-softmax in International Conference on Learning Representations
    (2017). https://openreview.net/forum?id=rkE3y85ee.
    - Joo, W., Kim, D., Shin, S. & Moon, I.-C. Generalized gumbel-softmax
    gradient estimator for generic discrete random variables.
      Preprint at https://doi.org/10.48550/arXiv.2003.01847 (2020).

    Parameters
    ----------
    model_simulations : dict
        dictionary containing all simulated output variables from the
        generative model.
    global_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    ypred : tf.Tensor
        continuously approximated ypred from the discrete likelihood.

    """
    # set seed
    tf.random.set_seed(seed)
    # get batch size
    B = epred.shape[0]
    # get number of simulations from priors
    S = epred.shape[1]
    # get number of observations
    number_obs = epred.shape[2]
    # constant outcome vector (including zero outcome)
    thres = upper_thres
    c = tf.range(thres + 1, delta=1, dtype=tf.float32)
    # broadcast to shape (B, rep, outcome-length)
    c_brct = tf.broadcast_to(c[None, None, None, :], shape=(B, S, number_obs,
                                                            len(c)))
    # compute pmf value
    pi = likelihood.prob(c_brct)
    # prevent underflow
    pi = tf.where(pi < 1.8 * 10 ** (-30), 1.8 * 10 ** (-30), pi)
    # sample from uniform
    u = tfd.Uniform(0, 1).sample((B, S, number_obs, len(c)))
    # generate a gumbel sample from uniform sample
    g = -tf.math.log(-tf.math.log(u))
    # softmax gumbel trick
    w = tf.nn.softmax(
        tf.math.divide(
            tf.math.add(tf.math.log(pi), g), temp,
        )
    )
    # reparameterization/linear transformation
    ypred = tf.reduce_sum(tf.multiply(w, c), axis=-1)
    return ypred


def simulate_from_generator(
        prior_samples: tf.Tensor,  # shape=[B,num_samples,num_params]
        seed: int, model: dict) -> dict[str, tf.Tensor]:
    """
    Simulates data from the specified generative model.

    Parameters
    ----------
    prior_samples : tf.Tensor, shape: [B, num_samples, num_params]
        samples from prior distributions.
    seed : int
        seed used for learning.
    model : callable
        specification of generative model using :func:`model`.

    Returns
    -------
    model_simulations : dict[str, tf.Tensor]
        simulated data from generative model.

    """
    # set seed
    tf.random.set_seed(seed)
    # get model and initialize generative model
    GenerativeModel = model["obj"]
    generative_model = GenerativeModel()
    # get model specific arguments (that are not prior samples)
    add_model_args = model.copy()
    add_model_args.pop("obj")
    add_model_args["seed"]=seed
    # simulate from generator
    if add_model_args is not None:
        model_simulations = generative_model(prior_samples, **add_model_args)
    else:
        model_simulations = generative_model(prior_samples)

    return model_simulations
