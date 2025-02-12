# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import elicit as el
import pandas as pd
import pickle
import os
import warnings

from typing import Tuple

tfd = tfp.distributions


def save_as_pkl(obj: any, save_dir: str) -> None:
    """
    Helper functions to save a file as pickle.

    Parameters
    ----------
    obj : any
        variable that needs to be saved.
    save_dir : str
        path indicating the file location.

    Returns
    -------
    None.

    Examples
    --------
    >>> save_as_pkl(obj, "results/file.pkl")

    """
    # if directory does not exists, create it
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    # save obj to location as pickle
    with open(save_dir, "wb") as file:
        pickle.dump(obj, file=file)


def identity(x: any) -> any:
    """
    identity function. Returns the input

    Parameters
    ----------
    x : any
        arbitrary input object.

    Returns
    -------
    x : any
        returns the input object without any transformation.

    """
    return x


class DoubleBound:
    def __init__(self, lower: float, upper: float):
        """
        A variable constrained to be in the open interval
        (``lower``, ``upper``) is transformed to an unconstrained variable Y
        via a scaled and translated log-odds transform.

        Basis for the here used constraints, is the
        `constraint transforms implementation in Stan <https://mc-stan.org/docs/reference-manual/transforms.html>`_.

        Parameters
        ----------
        lower : float
            lower bound of variable x.
        upper : float
            upper bound of variable x.

        """  # noqa: E501
        self.lower = lower
        self.upper = upper

    def logit(self, u: float):
        r"""
        Helper function that implements the logit transformation for
        :math:`u \in (0,1)`:

        .. math::

            logit(u) = \log\left(\frac{u}{1-u}\right)

        Parameters
        ----------
        u : float
            variable in open unit interval.

        Returns
        -------
        v : float
            log-odds of u.

        """
        # log-odds definition
        v = tf.math.log(u) - tf.math.log(1 - u)
        # cast v into correct dtype
        v = tf.cast(v, dtype=tf.float32)
        return v

    def inv_logit(self, v: float):
        r"""
        Helper function that implements the inverse-logit transformation (i.e.,
        the logistic sigmoid for :math:`v \in (-\infty,+\infty)`:

        .. math::

            logit^{-1}(v) = \frac{1}{1+\exp(-v)}

        Parameters
        ----------
        v : float
            unconstrained variable

        Returns
        -------
        u : float
            logistic sigmoid of the unconstrained variable

        """
        # logistic sigmoid transform
        u = tf.divide(1.0, (1.0 + tf.exp(v)))
        # cast v to correct dtype
        u = tf.cast(u, dtype=tf.float32)
        return u

    def forward(self, x: float):
        r"""
        scaled and translated logit transform of variable x with ``lower`` and
        ``upper`` bound into an unconstrained variable y.

        .. math::

            Y = logit\left(\frac{X - lower}{upper - lower}\right)

        Parameters
        ----------
        x : float
            variable with lower and upper bound.

        Returns
        -------
        y : float
            unconstrained variable.

        """
        # scaled and translated logit transform
        y = self.logit(tf.divide((x - self.lower), (self.upper - self.lower)))
        # cast y to correct dtype
        y = tf.cast(y, dtype=tf.float32)
        return y

    def inverse(self, y: float):
        r"""
        inverse of the log-odds transform applied to the unconstrained
        variable y in order to transform it into a constrained variable x
        with ``lower`` and ``upper`` bound.

        .. math::

            X = lower + (upper - lower) \cdot logit^{-1}(Y)

        Parameters
        ----------
        y : float
            unconstrained variable

        Returns
        -------
        x : float
        constrained variable with lower and upper bound

        """
        # inverse of log-odds transform
        x = self.lower + (self.upper - self.lower) * self.inv_logit(y)
        # cast x to correct dtype
        x = tf.cast(x, dtype=tf.float32)
        return x


class LowerBound:
    def __init__(self, lower: float):
        """
        A variable with a ``lower`` bound is transformed to an
        unconstrained variable Y via an inverse-softplus transform.

        Basis for the here used constraints, is the
        `constraint transforms implementation in Stan <https://mc-stan.org/docs/reference-manual/transforms.html>`_.

        Parameters
        ----------
        lower : float
            lower bound of variable X.

        """  # noqa: E501
        self.lower = lower

    def forward(self, x: float):
        r"""
        inverse-softplus transform of variable x with ``lower`` bound into an
        unconstrained variable y.

        .. math::

            Y = softplus^{-1}(X - lower)

        Parameters
        ----------
        x : float
            variable with a lower bound.

        Returns
        -------
        y : float
            unconstrained variable.

        """
        # inverse softplus transform
        y = tfp.math.softplus_inverse(x - self.lower)
        # cast y into correct type
        y = tf.cast(y, dtype=tf.float32)
        return y

    def inverse(self, y: float):
        r"""
        softplus transform of unconstrained variable y into a constrained
        variable x with ``lower`` bound.

        .. math::

            X = softplus(Y) + lower

        Parameters
        ----------
        y : float
            unconstrained variable.

        Returns
        -------
        x : float
            variable with a lower bound.

        """
        # softplus transform
        x = tf.math.softplus(y) + self.lower
        # cast x into correct dtype
        x = tf.cast(x, dtype=tf.float32)
        return x


class UpperBound:
    def __init__(self, upper: float):
        """
        A variable with an ``upper`` bound is transformed into an
        unconstrained variable Y via an inverse-softplus transform.

        Basis for the here used constraints, is the
        `constraint transforms implementation in Stan <https://mc-stan.org/docs/reference-manual/transforms.html>`_.

        Parameters
        ----------
        upper : float
            upper bound of variable X.

        """  # noqa: E501
        self.upper = upper

    def forward(self, x: float):
        r"""
        inverse-softplus transform of variable x with ``upper`` bound into an
        unconstrained variable y.

        .. math::

            Y = softplus^{-1}(upper - X)

        Parameters
        ----------
        x : float
            variable with an upper bound.

        Returns
        -------
        y : float
            unconstrained variable.

        """
        # logarithmic transform
        y = tfp.math.softplus_inverse(self.upper - x)
        # cast y into correct dtype
        y = tf.cast(y, dtype=tf.float32)
        return y

    def inverse(self, y: float):
        r"""
        softplus transform of unconstrained variable y into a constrained
        variable x with ``upper`` bound.

        .. math::

            X = upper - softplus(Y)

        Parameters
        ----------
        y : float
            unconstrained variable.

        Returns
        -------
        x : float
            variable with an upper bound.

        """
        # exponential transform
        x = self.upper - tf.math.softplus(y)
        # cast x into correct dtype
        x = tf.cast(x, dtype=tf.float32)
        return x


def one_forward_simulation(
    prior_model: el.simulations.Priors, trainer: dict, model: dict,
    targets: list[dict], seed: int
    ) -> Tuple[dict, tf.Tensor, dict, dict]:  # noqa: E125
    """
    One forward simulation from prior samples to elicited statistics.

    Parameters
    ----------
    prior_model : instance of Priors class objects
        initialized prior distributions which can be used for sampling.
    trainer : callable
        specification of training settings and meta-information for
        workflow using :func:`elicit.elicit.trainer`
    model : callable
        specification of generative model using :func:`elicit.elicit.model`.
    targets : list
        list of target quantities specified with :func:`elicit.elicit.target`.

    Returns
    -------
    elicited_statistics : dict
        dictionary containing the elicited statistics that can be used to
        compute the loss components
    prior_samples : tf.Tensor
        samples from prior distributions
    model_simulations : dict
        samples from the generative model (likelihood) given the prior samples
        for the model parameters
    target_quantities : dict
        target quantities as a function of the model simulations.
    seed : int
        internal seed for reproducible results

    """
    # set seed
    tf.random.set_seed(seed)
    # generate samples from initialized prior
    prior_samples = prior_model()
    # simulate prior predictive distribution based on prior samples
    # and generative model
    model_simulations = el.simulations.simulate_from_generator(
        prior_samples, seed, model
    )
    # compute the target quantities
    target_quantities = el.targets.computation_target_quantities(
        model_simulations, prior_samples, targets
    )
    # compute the elicited statistics by applying a specific elicitation
    # method on the target quantities
    elicited_statistics = el.targets.computation_elicited_statistics(
        target_quantities, targets
    )
    return (elicited_statistics, prior_samples, model_simulations,
            target_quantities)


def get_expert_data(
    trainer: dict,
    model: dict,
    targets: list[dict],
    expert: dict,
    parameters: list[dict],
    network: dict,
    seed: int
) -> Tuple[dict[str, tf.Tensor], tf.Tensor]:
    """
    Wrapper for loading the training data which can be expert data or
    data simulations using a pre-defined ground truth.

    Parameters
    ----------
    trainer : callable
        specification of training settings and meta-information for
        workflow using :func:`trainer`
    model : callable
        specification of generative model using :func:`elicit.elicit.model`.
    targets : list
        list of target quantities specified with :func:`elicit.elicit.target`.
    expert : callable
        provide input data from expert or simulate data from oracle with
        either the ``data`` or ``simulator`` method of the
        :mod:`elicit.elicit.Expert` module.
    parameters : list
        list of model parameters specified with :func:`elicit.elicit.parameter`.
    network : callable or None
        specification of neural network using a method implemented in
        :mod:`elicit.networks`.
        Only required for ``deep_prior`` method. For ``parametric_prior``
        use ``None``. Default value is ``None``.
    seed : int
        internal seed for reproducible results

    Returns
    -------
    expert_data : dict
        dictionary containing the training data. Must have same form as the
        model-simulated elicited statistics. Correct specification of
        keys can be checked using :func:`elicit.utils.get_expert_datformat`
    expert_prior : tf.Tensor, shape: [B,num_samples,num_params]
        samples from ground truth. Exists only if expert data are simulated
        from an oracle. Otherwise this output is ``None``

    """  # noqa: E501
    try:
        expert["data"]
    except KeyError:
        oracle = True
    else:
        oracle = False

    if oracle:
        # set seed
        tf.random.set_seed(seed)
        # sample from true priors
        prior_model = el.simulations.Priors(
            ground_truth=True,
            init_matrix_slice=None,
            trainer=trainer,
            parameters=parameters,
            network=network,
            expert=expert,
            seed=seed,
        )
        # compute elicited statistics and target quantities
        expert_data, expert_prior, *_ = one_forward_simulation(
            prior_model, trainer, model, targets, seed
        )
        return expert_data, expert_prior
    else:
        # load expert data from file
        expert_data = expert["data"]
        return expert_data, None


def save(
    eliobj: callable,
    name: str or None = None,
    file: str or None = None,
    overwrite: bool = False,
) -> None:
    # either name or file must be specified
    assert (name is None) ^ (file is None), (
        "Name and file cannot be both "
        + "None or both specified. Either one has to be None."
    )
    if (name is not None) and (file is None):
        if name.endswith(".pkl"):
            name = name.removesuffix(".pkl")
        # create saving path
        path = f"./results/{eliobj.trainer['method']}/{name}_{eliobj.trainer['seed']}"  # noqa
    if (name is None) and (file is not None):
        # postprocess file (or name) to avoid file.pkl.pkl
        if file.endswith(".pkl"):
            file = file.removesuffix(".pkl")
        path = "./" + file
    # check whether saving path is already used
    if os.path.isfile(path + ".pkl") and not overwrite:
        user_ans = input(
            "In provided directory exists already a file with"
            + " identical name. Do you want to overwrite it?"
            + " Press 'y' for overwriting and 'n' for abording."
        )
        while user_ans not in ["n", "y"]:
            user_ans = input(
                "Please press either 'y' for overwriting or 'n'"
                + "for abording the process."
            )

        if user_ans == "n":
            return "Process aborded. File is not overwritten."

    storage = dict()
    # user inputs
    storage["model"] = eliobj.model
    storage["parameters"] = eliobj.parameters
    storage["targets"] = eliobj.targets
    storage["expert"] = eliobj.expert
    storage["optimizer"] = eliobj.optimizer
    storage["trainer"] = eliobj.trainer
    storage["initializer"] = eliobj.initializer
    storage["network"] = eliobj.network
    # results
    storage["results"] = eliobj.results
    storage["history"] = eliobj.history

    save_as_pkl(storage, path + ".pkl")
    print(f"saved in: {path}.pkl")


def load(file: str) -> callable:
    """
    loads a saved ``eliobj`` from specified path.

    Parameters
    ----------
    file : str
        path where ``eliobj`` object is saved.

    Returns
    -------
    eliobj : el.elicit.Elicit obj
        loaded ``eliobj`` object.

    """
    obj = pd.read_pickle(file)

    eliobj = el.Elicit(
        model=obj["model"],
        parameters=obj["parameters"],
        targets=obj["targets"],
        expert=obj["expert"],
        optimizer=obj["optimizer"],
        trainer=obj["trainer"],
        initializer=obj["initializer"],
        network=obj["network"],
    )

    # add results if already fitted
    eliobj.history = obj["history"]
    eliobj.results = obj["results"]

    return eliobj


def parallel(
        runs: int = 4,
        cores: None or int = None,
        seeds : list = None
        ) -> dict:
    """
    Specification for parallelizing training by running multiple training
    instances with different seeds simultaneously.

    Parameters
    ----------
    runs: int, optional
        Number of replication. The default is ``4``.
    cores : int, optional
        Number of cores that should be used. The default is ``None`` which implies ``cores=runs``.
    seeds : list, optional
        A list of seeds. If ``None`` seeds are drawn from a Uniform(0,999999)
        distribution. The seed information corresponding to each chain is
        stored in ``eliobj.results``. The default is ``None``.

    Returns
    -------
    parallel_dict : dict
        dictionary containing the parallelization settings.

    """
    parallel_dict = dict(
        runs=runs,
        cores=cores,
        seeds=seeds
        )

    return parallel_dict


def save_history(
    loss: bool = True,
    loss_component: bool = True,
    time: bool = True,
    hyperparameter: bool = True,
    hyperparameter_gradient: bool = True,
) -> dict:
    """
    Controls whether sub-results of the history object should be included
    or excluded. Results are saved across epochs. By default all
    sub-results are included.

    Parameters
    ----------
    loss : bool, optional
        total loss per epoch. The default is ``True``.
    loss_component : bool, optional
        loss per loss-component per epoch. The default is ``True``.
    time : bool, optional
        time in sec per epoch. The default is ``True``.
    hyperparameter : bool, optional
        'parametric_prior' method: Trainable hyperparameters of parametric
        prior distributions.
        'deep_prior' method: Mean and standard deviation of each marginal
        from the joint prior.
        The default is ``True``.
    hyperparameter_gradient : bool, optional
        Gradients of the hyperparameter. Only for 'parametric_prior' method.
        The default is ``True``.

    Returns
    -------
    save_hist_dict : dict
        dictionary with inclusion/exclusion settings for each sub-result in
        history object.

    Warnings
    --------
    if ``loss_component`` or ``loss`` are excluded, :func:`elicit.plots.loss`
    can't be used as it requires this information.

    if ``hyperparameter`` is excluded, :func:`elicit.plots.hyperparameter`
    can't be used as it requires this information.
    Only parametric_prior method.

    if ``hyperparameter`` is excluded, :func:`elicit.plots.marginals`
    can't be used as it requires this information.
    Only deep_prior method.

    """
    if not loss or not loss_component:
        warnings.warn(
            "el.plots.loss() requires information about "
            + "'loss' and 'loss_component'. If you don't save this information "  # noqa
            + "el.plot.loss() can't be used."
        )
    if not hyperparameter:
        warnings.warn(
            "el.plots.hyperparameter() and el.plots.marginals() require"
            + " information about 'hyperparameter'. If you don't save this"
            + " information these plots can't be used."
        )

    save_hist_dict = dict(
        loss=loss,
        loss_component=loss_component,
        hyperparameter=hyperparameter,
        hyperparameter_gradient=hyperparameter_gradient,
    )
    return save_hist_dict


def save_results(
    target_quantities: bool = True,
    elicited_statistics: bool = True,
    prior_samples: bool = True,
    model_samples: bool = True,
    expert_elicited_statistics: bool = True,
    expert_prior_samples: bool = True,
    init_loss_list: bool = True,
    init_prior: bool = True,
    init_matrix: bool = True,
    loss_tensor_expert: bool = True,
    loss_tensor_model: bool = True,
) -> dict:
    """
    Controls whether sub-results of the result object should be included
    or excluded in the final result file. Results are based on the
    computation of the last epoch.
    By default all sub-results are included.

    Parameters
    ----------
    target_quantities : bool, optional
        simulation-based target quantities. The default is ``True``.
    elicited_statistics : bool, optional
        simulation-based elicited statistics. The default is ``True``.
    prior_samples : bool, optional
        samples from simulation-based prior distributions.
        The default is ``True``.
    model_samples : bool, optional
        output variables from the simulation-based generative model.
        The default is ``True``.
    expert_elicited_statistics : bool, optional
        expert-elicited statistics. The default is ``True``.
    expert_prior_samples : bool, optional
        if oracle is used: samples from the true prior distribution,
        otherwise it is None. The default is ``True``.
    init_loss_list : bool, optional
        initialization phase: Losses related to the samples drawn from the
        initialization distribution.
        Only included for method 'parametric_prior'.
        The default is ``True``.
    init_prior : bool, optional
        initialized elicit model object including the trainable variables.
        Only included for method 'parametric_prior'.
        The default is ``True``.
    init_matrix : bool, optional
        initialization phase: samples drawn from the initialization
        distribution for each hyperparameter.
        Only included for method 'parametric_prior'.
        The default is ``True``.
    loss_tensor_expert : bool, optional
        expert term in loss component for computing the discrepancy.
        The default is ``True``.
    loss_tensor_model : bool, optional
        simulation-based term in loss component for computing the
        discrepancy. The default is ``True``.

    Returns
    -------
    save_res_dict : dict
        dictionary with inclusion/exclusion settings for each sub-result
        in results object.

    Warnings
    --------
    if ``elicited_statistics`` is excluded :func:`elicit.plots.loss` can't be
    used as it requires this information.

    if ``init_matrix`` is excluded :func:`elicit.plots.initialization` can't be
    used as it requires this information.

    if ``prior_samples`` is excluded :func:`elicit.plots.prior_joint` can't be
    used as it requires this information.

    if ``target_quantities`` is excluded :func:`elicit.plots.priorpredictive`
    can't be used as it requires this information.

    if ``expert_elicited_statistics`` or ``elicited_statistics`` is
    excluded :func:`elicit.plots.elicits` can't be used as it requires this
    information.

    """
    if not elicited_statistics:
        warnings.warn(
            "el.plots.loss() requires information about "
            + "'elicited_statistics'. If you don't save this information "
            + "el.plot.loss() can't be used."
        )
    if not init_matrix:
        warnings.warn(
            "el.plots.initialization() requires information about "
            + "'init_matrix'. If you don't save this information "
            + "this plotting function can't be used."
        )
    if not prior_samples:
        warnings.warn(
            "el.plots.priors() requires information about "
            + "'prior_samples'. If you don't save this information "
            + "this plotting function can't be used."
        )
    if not target_quantities:
        warnings.warn(
            "el.plots.priorpredictive() requires information about "
            + "'target_quantities'. If you don't save this information "
            + "this plotting function can't be used."
        )
    if (not expert_elicited_statistics) or (not elicited_statistics):
        warnings.warn(
            "el.plots.elicits() requires information about "
            + "'expert_elicited_statistics' and 'elicited_statistics'. "
            + "If you don't save this information this plotting function"
            + " can't be used."
        )

    save_res_dict = dict(
        target_quantities=target_quantities,
        elicited_statistics=elicited_statistics,
        prior_samples=prior_samples,
        model_samples=model_samples,
        expert_elicited_statistics=expert_elicited_statistics,
        expert_prior_samples=expert_prior_samples,
        init_loss_list=init_loss_list,
        init_prior=init_prior,
        init_matrix=init_matrix,
        loss_tensor_expert=loss_tensor_expert,
        loss_tensor_model=loss_tensor_model,
    )
    return save_res_dict


def clean_savings(history: dict, 
                  results: dict,
                  save_history: dict,
                  save_results: dict
                  ) -> Tuple[dict, dict]:
    """
    Parameters
    ----------
    history : dict
        results that are saved across epochs including among others loss,
        loss_component, time, and hyperparameter. See :func:`save_history` for
        complete list.
    results : dict
        results that are saved for the last epoch only including prior_samples,
        elicited_statistics, target_quantities, etc. See :func:`save_results`
        for complete list.
    save_history : callable, :func:`elicit.utils.save_history`
        Exclude or include sub-results in the final result file.
        In the ``history`` object are all results that are saved across epochs.
        For usage information see
        `How-To: Save and load the eliobj <https://florence-bockting.github.io/prior_elicitation/howto/saving_loading.html>`_
    save_results : callable, :func:`elicit.utils.save_results`
        Exclude or include sub-results in the final result file.
        In the ``results`` object are all results that are saved for the last
        epoch only. For usage information see
        `How-To: Save and load the eliobj <https://florence-bockting.github.io/prior_elicitation/howto/saving_loading.html>`_

    Returns
    -------
    results, history : Tuple[dict, dict]
        final results taking in consideration exclusion criteria as specified
        in :func:`save_history` and :func:`save_results`.

    """  # noqa: E501
    for key_hist in save_history:
        if not save_history[key_hist]:
            history.pop(key_hist)

    for key_res in save_results:
        if not save_results[key_res]:
            results.pop(key_res)
    return results, history


def get_expert_datformat(targets: list[dict]) -> dict[str, list]:
    """
    helper function for the user to inspect which data format for the expert
    data is expected by the method.

    Parameters
    ----------
    targets : list
        list of target quantities specified with :func:`elicit.elicit.target`.

    Returns
    -------
    elicit_dict : dict[str, list]
        expected format of expert data.

    """
    elicit_dict = dict()
    for tar in targets:
        query = tar["query"]["name"]
        target = tar["name"]
        key = query + "_" + target
        elicit_dict[key] = list()

    return elicit_dict


def softmax_gumbel_trick(epred: tf.Tensor, likelihood: callable,
                         upper_thres: float, temp: float=1.6, **kwargs):
    """
    The softmax-gumbel trick computes a continuous approximation of ypred from
    a discrete likelihood and thus allows for the computation of gradients for
    discrete random variables.

    Currently this approach is only implemented for models without upper
    boundary (e.g., Poisson model).

    Corresponding literature:

        - Maddison, C. J., Mnih, A. & Teh, Y. W. The concrete distribution:
          A continuous relaxation of discrete random variables in International
          Conference on Learning Representations (2017).
          https://doi.org/10.48550/arXiv.1611.00712
        - Jang, E., Gu, S. & Poole, B. Categorical reparameterization with
          gumbel-softmax in International Conference on Learning Representations
          (2017). https://openreview.net/forum?id=rkE3y85ee.
        - Joo, W., Kim, D., Shin, S. & Moon, I.-C. Generalized gumbel-softmax
          gradient estimator for generic discrete random variables. Preprint
          at https://doi.org/10.48550/arXiv.2003.01847 (2020).

    Parameters
    ----------
    epred : tf.Tensor, shape = [B, num_samples, num_obs]
        simulated linear predictor from the model simulations 
    likelihood : tfp.distributions object, shape = [B, num_samples, num_obs, 1]
        likelihood function used in the generative model.
        Must be a tfp.distributions object.
    upper_thres : float
        upper threshold at which the distribution of the outcome variable is
        truncated. For double-bounded distribution (e.g. Binomial) this is
        simply the "total count" information. Lower-bounded distribution
        (e.g. Poisson) must be truncated to create an artificial
        double-boundedness.
    temp : float, temp > 0
        temperature hyperparameter of softmax function. A temperature going
        towards zero yields approximates a categorical distribution, while
        a temperature >> 0 approximates a continuous distribution.
        The default value is ``1.6``.
    kwargs : any
        additional keyword arguments including the seed information. **Note**:
        the ``**kwargs`` argument is required in this function (!) as it
        extracts internally the seed information.

    Returns
    -------
    ypred : tf.Tensor
        continuously approximated ypred from the discrete likelihood.

    Raise
    -----
    ValueError
        if rank of ``likelihood`` is not 4. The shape of the likelihood obj
        must have an extra final dimension, i.e., (B, num_samples, num_obs, 1),
        for the softmax-gumbel computation. Use for example
        ``tf.expand_dims(mu,-1)`` for expanding the batch-shape of the
        likelihood.

        if likelihood is not in tfp.distributions module. The likelihood
        must be a tfp.distributions object.

    KeyError
        if ``**kwargs`` is not in function arguments. It is required to pass
        the **kwargs argument as it is used for extracting internally 
        information about the seed.

    """
    # check rank of likelihood object
    if len(likelihood.batch_shape) != 4:
        raise ValueError(
            "The 'likelihood' in the generative model must have"
            +" batch_shape = (B, num_samples, num_obs, 1)."
            +" The additional final axis is required by the softmax-gumbel"
            +" computation. Use for example `tf.expand_dims(mu,-1)` for"
            +" expanding the batch-shape of the likelihood."
            )
    # check value/type of likelihood object
    if likelihood.name.lower() not in dir(tfd):
        raise ValueError(
            "Likelihood in generative model must be a tfp.distribution object."
            )
    if "seed" not in list(kwargs.keys()):
        raise KeyError(
            "Please provide the **kwargs argument in the el.utils.softmax-"
              +"gumble function. This is required for extracting internally"
              +" information about the seed.")

    # set seed
    tf.random.set_seed(kwargs["seed"])
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
