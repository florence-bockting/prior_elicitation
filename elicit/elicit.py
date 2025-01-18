# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import elicit as el

tfd = tfp.distributions


class Dtype:
    def __init__(self, vtype, dim):
        self.vtype = vtype
        self.dim = dim

    def __call__(self, x):

        if self.vtype == "real":
            dtype_dim = tf.cast(x, dtype=tf.float32)
        elif self.vtype == "array":
            dtype_dim = tf.constant(x, dtype=tf.float32, shape=(self.dim,))
        return dtype_dim


def hyper(
    name: str,
    lower: float = float("-inf"),
    upper: float = float("inf"),
    vtype: str = "real",
    dim: int = 1,
    shared: bool = False,
):
    """
    Specification of prior hyperparameters.

    Parameters
    ----------
    name : string
        Custom name of hyperparameter.
    lower : float, optional
        Lower bound of hyperparameter.
        The default is unbounded: ``float("-inf")``.
    upper : float, optional
        Upper bound of hyperparameter.
        The default is unbounded: ``float("inf")``.
    vtype : string, optional
        Type of hyperparameter; either "real" or "array".
        The default is ``"real"``.
    dim : integer, optional
        Dimensionality of variable. Only required if vtype is "array".
        The default is ``1``.
    shared : bool, optional
        Shared hyperparameter between model parameters.
        The default is ``False``.

    Returns
    -------
    hyppar_dict : dict
        Dictionary including all hyperparameter settings.

    Examples
    --------
    >>> # sigma hyperparameter of a parameteric distribution
    >>> el.hyper(name="sigma0", lower=0)

    >>> # shared hyperparameter
    >>> el.hyper(name="sigma", lower=0, shared=True)

    """
    # constraints
    # only lower bound
    if (lower != float("-inf")) and (upper == float("inf")):
        lower_bound = el.utils.LowerBound(lower)
        transform = lower_bound.inverse
    # only upper bound
    elif (upper != float("inf")) and (lower == float("-inf")):
        upper_bound = el.utils.UpperBound(upper)
        transform = upper_bound.inverse
    # upper and lower bound
    elif (upper != float("inf")) and (lower != float("-inf")):
        double_bound = el.utils.DoubleBound(lower, upper)
        transform = double_bound.inverse
    # unbounded
    else:
        transform = el.utils.identity

    # value type
    dtype_dim = Dtype(vtype, dim)

    hyppar_dict = dict(
        name=name, constraint=transform, vtype=dtype_dim, dim=dim,
        shared=shared
    )

    return hyppar_dict


def parameter(
    name: str, family: callable or None = None,
    hyperparams: callable or None = None
):
    """
    Specification of model parameters.

    Parameters
    ----------
    name : string
        Custom name of parameter.
    family : callable or None
        Prior distribution family for model parameter.
        Only required for ``parametric_prior`` method.
        Must be an `tfp.distributions <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions>`_ object.
    hyperparams : dict or None
        Hyperparameters of distribution as specified in **family**.
        Only required for ``parametric_prior`` method.
        Structure of dictionary: *keys* must match arguments of
        `tfp.distributions <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions>`_
        object and *values* have to be specified using the :func:`hyper`
        method.
        Further details are provided in
        `How-To specify prior hyperparameters (TODO) <url>`_.
        Default value is ``None``.

    Returns
    -------
    param_dict : dict
        Dictionary including all model (hyper)parameter settings.

    Examples
    --------
    >>> el.parameter(name="beta0",
    >>>              family=tfd.Normal,
    >>>              hyperparams=dict(loc=el.hyper("mu0"),
    >>>                               scale=el.hyper("sigma0", lower=0)
    >>>                               )
    >>>              )

    """  # noqa: E501

    param_dict = dict(name=name, family=family, hyperparams=hyperparams)

    return param_dict


def model(obj: callable, **kwargs):
    """
    Specification of the generative model.

    Parameters
    ----------
    obj : class
        class that implements the generative model.
        See `How-To specify the generative_model for details (TODO) <url>`_.
    **kwargs : keyword arguments, optional
        additional keyword arguments expected by **obj**.

    Returns
    -------
    generator_dict : dict
        Dictionary including all generative model settings.

    Examples
    --------
    >>> # specify the generative model class
    >>> class ToyModel:
    >>>     def __call__(self, prior_samples, design_matrix, **kwargs):
    >>>         B = prior_samples.shape[0]
    >>>         S = prior_samples.shape[1]
    >>>         # preprocess shape of design matrix
    >>>         X = tf.broadcast_to(design_matrix[None, None,:],
    >>>                            (B,S,len(design_matrix)))
    >>>         # linear predictor (= mu)
    >>>         epred = tf.add(prior_samples[:, :, 0][:,:,None],
    >>>                        tf.multiply(prior_samples[:, :, 1][:,:,None], X)
    >>>                        )
    >>>         # data-generating model
    >>>         likelihood = tfd.Normal(
    >>>             loc=epred, scale=tf.expand_dims(prior_samples[:, :, -1], -1)
    >>>             )
    >>>         # prior predictive distribution (=height)
    >>>         ypred = likelihood.sample()
    >>>
    >>>         return dict(
    >>>             likelihood=likelihood,
    >>>             ypred=ypred, epred=epred,
    >>>             prior_samples=prior_samples
    >>>             )

    >>> # specify the model category in the elicit object
    >>> el.model(obj=ToyModel,
    >>>          design_matrix=std_predictor(N=200, quantiles=[25,50,75])
    >>>          )
    """  # noqa: E501
    generator_dict = dict(obj=obj)

    for key in kwargs:
        generator_dict[key] = kwargs[key]

    return generator_dict


class Queries:
    def quantiles(self, quantiles: tuple):
        """
        Implements a quantile-based elicitation technique.

        Parameters
        ----------
        quants : tuple
            Tuple with respective quantiles ranging from 0 to 100.

        Returns
        -------
        elicit_dict : dict
            Dictionary including the quantile settings.

        """
        return dict(name="quantiles", value=quantiles)

    def identity(self):
        """
        Implements an identity function. Should be used if no further
        transformation of target quantity is required.

        Returns
        -------
        elicit_dict : dict
            Dictionary including the identity settings.

        """
        return dict(name="identity", value=None)

    def correlation(self):
        """
        Implements a method to calculate the pearson correlation between
        model parameters.

        Returns
        -------
        elicit_dict : dict
            Dictionary including the correlation settings.

        """
        return dict(name="pearson_correlation", value=None)

    def custom(self, func: callable, **kwargs):
        """
        Implements a placeholder for custom target methods. The custom method
        can be passed as argument.
        Note: this function hasn't been implemented yet and will raise
        an ``NotImplementedError``.  See for further information the
        corresponding `GitHub issue #33 <https://github.com/florence-bockting/prior_elicitation/issues/33>`_.

        Parameters
        ----------
        func : callable
            Custom target method.

        Returns
        -------
        elicit_dict : dict
            Dictionary including the custom settings.

        """  # noqa: E501
        raise NotImplementedError("The use of custom elicitation methods " +
                                  "hasn't been implemented yet.")
        # args_dict = dict()
        # for key in kwargs:
        #     args_dict[key] = kwargs[key]

        # return dict(name="custom", value=func, add_args=args_dict)


# create an instance of the Queries class
queries = Queries()


def target(
    name: str,
    query: callable = queries,
    loss: callable = el.losses.MMD2(kernel="energy"),
    target_method: callable = None,
    weight: float = 1.0,
):
    """
    Specification of target quantity and corresponding elicitation technique.

    Parameters
    ----------
    name : string
        Name of the target quantity. Two approaches are possible:
        (1) Target quantity is identical to an output from the generative
        model: The name must match the output variable name. (2) Custom target
        quantity is computed using the **target_method** argument.
    query : callable
        Specify the elicitation technique by using one of the methods
        implemented in :func:`Queries`.
        See `How-To specify custom elicitation techniques (TODO) <url>`_.
    loss : callable
        Loss function for computing the discrepancy between expert data and
        model simulations. Implemented classes can be found
        in :mod:`elicit.losses`.
        The default is the maximum mean discrepancy with
        an energy kernel: :func:`elicit.losses.MMD2`
    target_method : callable, optional
        Custom method for computing a target quantity.
        Note: This method hasn't been implemented yet and will raise an
        ``NotImplementedError``. See for further information the corresponding
        `GitHub issue #34 <https://github.com/florence-bockting/prior_elicitation/issues/34>`_.
        The default is ``None``.
    weight : float, optional
        Weight of the corresponding elicited quantity in the total loss.
        The default is ``1.0``.

    Returns
    -------
    target_dict : dict
        Dictionary including all settings regarding the target quantity and
        corresponding elicitation technique.

    Examples
    --------
    >>> el.target(name="y_X0",
    >>>           query=el.queries.quantiles((5, 25, 50, 75, 95)),
    >>>           loss=el.losses.MMD2(kernel="energy"),
    >>>           weight=1.0
    >>>           )

    >>> el.target(name="correlation",
    >>>           query=el.queries.correlation(),
    >>>           loss=el.losses.L2,
    >>>           weight=1.0
    >>>           )
    """  # noqa: E501
    try:
        target_method is not None
    except NotImplementedError:
        print(
            "The use of a custom target quantity hasn't been implemented yet."
            )

    # create instance of loss class
    loss_instance = loss

    target_dict = dict(
        name=name,
        query=query,
        target_method=target_method,
        loss=loss_instance,
        weight=weight,
    )

    return target_dict


class Expert:
    def data(self, dat: dict):
        """
        Provide elicited-expert data for learning prior distributions.

        Parameters
        ----------
        dat : dict
            Elicited data from expert provided as dictionary. Data must be
            provided in a standardized format which is explained in
            `How-To specify the expert data (TODO) <url>`_

        Returns
        -------
        expert_data : dict
            Expert-elicited information used for learning prior distributions.

        Examples
        --------
        >>> expert_dat = {
        >>>     "quantiles_y_X0": [-12.55, -0.57, 3.29, 7.14, 19.15],
        >>>     "quantiles_y_X1": [-11.18, 1.45, 5.06, 8.83, 20.42],
        >>>     "quantiles_y_X2": [-9.28, 3.09, 6.83, 10.55, 23.29]
        >>> }
        """
        dat_prep = {
            f"{key}": tf.expand_dims(
                tf.cast(tf.convert_to_tensor(dat[key]), dtype=tf.float32), 0
            )
            for key in dat
        }

        return dict(data=dat_prep)

    def simulator(self, ground_truth: dict, num_samples: int = 10_000):
        """
        Simulate data from an oracle by defining a ground truth (true prior
        distribution(s)).
        See `Explanation: Simulating from an oracle (TODO) <url>`_ for
        further details.

        Parameters
        ----------
        ground_truth : dict
            True prior distributions with *keys* matching the parameter names
            as specified in :func:`parameter` and *values* being prior
            distributions implemented as
            `tfp.distributions <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions>`_
            object with predetermined hyperparameter values.
        num_samples : int
            Number of draws from the prior distribution.
            It is recommended to use a high value to min. sampling variation.
            The default is ``10_000``.

        Returns
        -------
        expert_data : dict
            Settings of oracle for simulating from ground truth. True elicited
            statistics are used as 'expert-data' in loss function.

        Examples
        --------
        >>> el.expert.simulator(
        >>>     ground_truth = {
        >>>         "beta0": tfd.Normal(loc=5, scale=1),
        >>>         "beta1": tfd.Normal(loc=2, scale=1),
        >>>         "sigma": tfd.HalfNormal(scale=10.0),
        >>>     },
        >>>     num_samples = 10_000
        >>> )
        """  # noqa: E501
        return dict(ground_truth=ground_truth, num_samples=num_samples)


expert = Expert()


def optimizer(optimizer: callable = tf.keras.optimizers.Adam(), **kwargs):
    """
    Specification of optimizer and its settings for SGD.

    Parameters
    ----------
    optimizer : callable, `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ object.
        Optimizer used for SGD implemented as `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ object.
        The default is ``tf.keras.optimizers.Adam``.
    **kwargs : keyword arguments, optional
        Additional keyword arguments expected by **optimizer**.

    Returns
    -------
    optimizer_dict : dict
        Dictionary specifying the SGD optimizer and its additional settings.

    Examples
    --------
    >>> optimizer=el.optimizer(
    >>>     optimizer=tf.keras.optimizers.Adam,
    >>>     learning_rate=0.1,
    >>>     clipnorm=1.0
    >>> )
    """  # noqa: E501
    optimizer_dict = dict(optimizer=optimizer)
    for key in kwargs:
        optimizer_dict[key] = kwargs[key]

    return optimizer_dict


def initializer(
    method: str,
    distribution: callable = el.initialization.uniform(),
    loss_quantile: int = 0,
    iterations: int = 100,
):
    """
    Initialization method for finding initial values of model hyperparameters
    required for instantiating training with SGD. Initial values for each
    hyperparameter are drawn from a uniform distribution ranging from
    ``mean-radius`` to ``mean+radius``.
    Further details on the implemented initialization method can be found in
    `Explanation: Initialization method <url>`_.
    Only necessary for method ``parametric_prior``.

    Parameters
    ----------
    method : string
        Name of initialization method. Currently supported are "random", "lhs",
        and "sobol".
    distribution : callable, :func:`elicit.initialization.uniform`
        Specification of initialization distribution.
        Currently implemented methods: :func:`elicit.initialization.uniform`
    loss_quantile : int,
        Quantile indicating which loss value should be used for selecting the
        initial hyperparameters.
        The default is ``0`` i.e., the minimum loss.
    iterations : int
        Number of samples drawn from the initialization distribution.
        The default is ``100``.

    Returns
    -------
    init_dict : dict
        Dictionary specifying the initialization method.

    Examples
    --------
    >>> el.initializer(
    >>>     method="lhs",
    >>>     loss_quantile=0,
    >>>     iterations=32,
    >>>     distribution=el.initialization.uniform(
    >>>         radius=1,
    >>>         mean=0
    >>>         )
    >>>     )
    """
    init_dict = dict(
        method=method,
        distribution=distribution,
        loss_quantile=loss_quantile,
        iterations=iterations,
    )

    return init_dict


def trainer(
    method: str,
    name: str,
    seed: int,
    epochs: int,
    B: int = 128,
    num_samples: int = 200,
    save_history: callable = el.utils.save_history(),
    save_results: callable = el.utils.save_results(),
):
    """
    Specification of training settings for learning the prior distribution(s).

    Parameters
    ----------
    method : str
        Method for learning the prior distribution. Available is either
        ``parametric_prior`` for learning independent parametric priors
        or ``deep_prior`` for learning a joint non-parameteric prior.
    name : str
        provides model a unique identifier used for saving results.
    seed : int
        seed used for learning.
    epochs : int
        number of iterations until training is stopped.
    B : int, optional
        batch size. The default is 128.
    num_samples : int, optional
        number of samples from the prior(s). The default is 200.
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
    train_dict : dict
        dictionary specifying the training settings for learning the prior
        distribution(s).

    Examples
    --------
    >>> el.trainer(
    >>>     method="parametric_prior",
    >>>     name="toymodel",
    >>>     seed=0,
    >>>     epochs=400,
    >>>     B=128,
    >>>     num_samples=200,
    >>>     save_history=el.utils.save_history(loss_component=False),
    >>>     save_results=el.utils.save_results(model=False)
    >>> )
    """  # noqa: E501
    train_dict = dict(
        method=method,
        name=name,
        seed=seed,
        B=B,
        num_samples=num_samples,
        epochs=epochs,
        save_history=save_history,
        save_results=save_results,
    )

    return train_dict


class Elicit:
    def __init__(
        self,
        model: callable,
        parameters: list,
        targets: list,
        expert: callable,
        trainer: callable,
        optimizer: callable,
        network: callable or None = None,
        initializer: callable or None = None,
    ):
        """
        Parameters
        ----------
        model : callable
            specification of generative model using :func:`model`.
        parameters : list
            list of model parameters specified with :func:`parameter`.
        targets : list
            list of target quantities specified with :func:`target`.
        expert : callable
            provide input data from expert or simulate data from oracle with
            either the ``data`` or ``simulator`` method of the
            :mod:`elicit.elicit.Expert` module.
        trainer : callable
            specification of training settings and meta-information for
            workflow using :func:`trainer`
        optimizer : callable
            specification of SGD optimizer and its settings using
            :func:`optimizer`.
        normalizing_flow : callable or None
            specification of normalizing flow using a method implemented in
            :mod:`elicit.networks`.
            Only required for ``deep_prior`` method. For ``parametric_prior``
            use ``None``. Default value is ``None``.
        initializer : callable
            specification of initialization settings using
            :func:`initializer`. Only required for ``parametric_prior`` method.
            Otherwise the argument should be ``None``. Default value is
            ``None.``

        Returns
        -------
        eliobj : class instance
            specification of all settings to run the elicitation workflow and
            fit the eliobj.

        """
        self.model = model
        self.parameters = parameters
        self.targets = targets
        self.expert = expert
        self.trainer = trainer
        self.optimizer = optimizer
        self.network = network
        self.initializer = initializer

        self.history = dict()
        self.results = dict()

        # set seed
        tf.random.set_seed(self.trainer["seed"])

    def fit(self, overwrite=False):
        """
        method for fitting the eliobj and learn prior distributions.

        Parameters
        ----------
        overwrite : bool, optional
            If the eliobj was already fitted and the user wants to refit it,
            the user is asked whether they want to overwrite the previous
            fitting results. Setting ``overwrite=True`` allows the user to
            force overfitting without being prompted. The default is ``False``.

        Examples
        --------
        >>> eliobj.fit()

        >>> eliobj.fit(overwrite=True)

        """
        # check whether elicit object is already fitted
        if len(self.history.keys()) != 0 and not overwrite:
            user_answ = input(
                "eliobj is already fitted."
                + " Do you want to fit it again and overwrite the results?"
                + " Press 'n' to stop process and 'y' to continue fitting."
            )

            while user_answ not in ["n", "y"]:
                user_answ = input(
                    "Please press either 'y' for fitting or 'n'"
                    + " for abording the process."
                )

            if user_answ == "n":
                return "Process aborded; eliobj is not re-fitted."

        # set seed
        tf.random.set_seed(self.trainer["seed"])

        # get expert data
        expert_elicits, expert_prior = el.utils.get_expert_data(
            self.trainer,
            self.model,
            self.targets,
            self.expert,
            self.parameters,
            self.network,
        )

        # initialization of hyperparameter
        (init_prior_model, loss_list, init_prior, init_matrix) = (
            el.initialization.init_prior(
                expert_elicits,
                self.initializer,
                self.parameters,
                self.trainer,
                self.model,
                self.targets,
                self.network,
                self.expert,
            )
        )

        # run dag with optimal set of initial values
        # save results in corresp. attributes
        self.history, self.results = el.optimization.sgd_training(
            expert_elicits,
            init_prior_model,
            self.trainer,
            self.optimizer,
            self.model,
            self.targets,
        )
        # add saving path
        self.trainer["output_path"] = f"/{self.trainer['method']}/{self.trainer['name']}_{self.trainer['seed']}"  # noqa
        # add some additional results
        self.results["expert_elicited_statistics"] = expert_elicits
        try:
            self.expert["ground_truth"]
        except KeyError:
            pass
        else:
            self.results["expert_prior_samples"] = expert_prior

        if self.trainer["method"] == "parametric_prior":
            self.results["init_loss_list"] = loss_list
            self.results["init_prior"] = init_prior
            self.results["init_matrix"] = init_matrix


        for key_hist in self.trainer["save_history"]:
            if not self.trainer["save_history"][key_hist]:
                self.history.pop(key_hist)

        for key_res in self.trainer["save_results"]:
            if not self.trainer["save_results"][key_res]:
                self.results.pop(key_res)

    def save(self, save_dir: str, overwrite: bool=False):
        """
        method for saving the eliobj on disk

        Parameters
        ----------
        save_dir : str
            directory name where to store the eliobj. For example, if
            ``save_dir="res"`` the eliobj is saved under following path:
            ``res/{method}/{name}_{seed}.pkl`` whereby 'method', 'name', and
            'seed' are arguments of :func:`elicit.elicit.trainer`.
        overwrite : bool, optional
            If already a fitted object exists in the same path, the user is
            asked whether the eliobj should be refitted and the results
            overwritten. With the ``overwrite`` argument you can silent this
            behavior. In this case the results are automatically overwritten
            without prompting the user. The default is ``False``.

        Examples
        --------
        >>> eliobj.save(save_dir="res")

        >>> eliobj.save(save_dir="res", overwrite=True)

        """
        # add a saving path
        return el.utils.save(self, save_dir=save_dir, overwrite=overwrite)

    def update(self, **kwargs):
        """
        method for updating the attributes of the Elicit class.

        Parameters
        ----------
        **kwargs : any
            keyword argument used for updating an attribute of Elicit class.
            Key must correspond to one attribute of the class and value refers
            to the updated value.

        Examples
        --------
        >>> eliobj.update(parameter = updated_parameter_dict)

        """
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"updated attribute {key}")