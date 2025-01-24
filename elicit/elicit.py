# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import elicit as el
import inspect

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
) -> dict:
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

    Raises
    ------
    ValueError
        ``lower``, ``upper`` take only values that are float or "-inf"/"inf".

        ``lower`` value should not be higher than ``upper`` value.

        ``vtype`` value can only be either 'real' or 'array'.

        ``dim`` value can't be '1' if 'vtype="array"'

    Examples
    --------
    >>> # sigma hyperparameter of a parameteric distribution
    >>> el.hyper(name="sigma0", lower=0)

    >>> # shared hyperparameter
    >>> el.hyper(name="sigma", lower=0, shared=True)

    """
    # check correct value for lower
    if lower == "-inf":
        lower = float("-inf")

    if (type(lower) is str) and (lower != "-inf"):
        raise ValueError(
            "lower must be either '-inf' or a float."
            + " Other strings are not allowed."
        )

    # check correct value for upper
    if upper == "inf":
        upper = float("inf")
    if (type(upper) is str) and (upper != "inf"):
        raise ValueError(
            "upper must be either 'inf' or a float." +
            " Other strings are not allowed."
        )

    if lower > upper:
        raise ValueError(
            "The value for 'lower' must be smaller than the value for 'upper'."
        )

    # check values for vtype are implemented
    if vtype not in ["real", "array"]:
        raise ValueError(
            f"vtype must be either 'real' or 'array'. You provided '{vtype}'."
        )

    # check that dimensionality is adapted when "array" is chosen
    if (vtype == "array") and dim == 1:
        raise ValueError(
            "For vtype='array', the 'dim' argument must have a value "
            + "greater 1."
        )

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
) -> dict:
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

    Raises
    ------
    ValueError
        ``family`` has to be a tfp.distributions object.

        ``hyperparams`` value is a dict with keys corresponding to arguments of
        tfp.distributions object in 'family'. Raises error if key does not
        correspond to any argument of distribution.

    Examples
    --------
    >>> el.parameter(name="beta0",
    >>>              family=tfd.Normal,
    >>>              hyperparams=dict(loc=el.hyper("mu0"),
    >>>                               scale=el.hyper("sigma0", lower=0)
    >>>                               )
    >>>              )

    """  # noqa: E501

    # check that family is a tfp.distributions object
    if family is not None:
        if family.__module__.split(".")[-1] not in dir(tfd):
            raise ValueError(
                "[section: parameters] The argument 'family'"
                + "has to be a tfp.distributions object."
            )

    # check whether keys of hyperparams dict correspond to arguments of family
    for key in hyperparams:
        if key not in inspect.getfullargspec(family)[0]:
            raise ValueError(
                f"[section: parameters] '{family.__module__.split('.')[-1]}'"
                + f" family has no argument '{key}'. Check keys of "
                + "'hyperparams' dict."
            )

    param_dict = dict(name=name, family=family, hyperparams=hyperparams)

    return param_dict


def model(obj: callable, **kwargs) -> dict:
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

    Raises
    ------
    ValueError
        generative model in ``obj`` requires the input argument
        'prior_samples', but argument has not been found.

        optional argument(s) of the generative model specified in ``obj`` are
        not specified

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
    >>>         # linear predictor
    >>>         epred = tf.add(prior_samples[:, :, 0][:,:,None],
    >>>                        tf.multiply(prior_samples[:, :, 1][:,:,None], X)
    >>>                        )
    >>>         # data-generating model
    >>>         likelihood = tfd.Normal(
    >>>             loc=epred, scale=tf.expand_dims(prior_samples[:, :, -1], -1)
    >>>             )
    >>>         # prior predictive distribution
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
    # get input arguments of generative model class
    input_args = inspect.getfullargspec(obj.__call__)[0]
    # check correct input form of generative model class
    if "prior_samples" not in input_args:
        raise ValueError(
            "[section: model] The generative model class 'obj' requires the"
            + " input variable 'prior_samples' but argument has not been found"
            + " in 'obj'."
        )

    # check that all optional arguments have been provided by the user
    optional_args = set(input_args).difference({"prior_samples", "self"})
    for arg in optional_args:
        if arg not in list(kwargs.keys()):
            raise ValueError(
                f"[section: model] The argument '{arg}' required by the"
                + " generative model class 'obj' is missing."
            )

    generator_dict = dict(obj=obj)

    for key in kwargs:
        generator_dict[key] = kwargs[key]

    return generator_dict


class Queries:
    def quantiles(self, quantiles: tuple[float]) -> dict:
        """
        Implements a quantile-based elicitation technique.

        Parameters
        ----------
        quants : tuple
            Tuple with respective quantiles ranging between 0 and 1.

        Returns
        -------
        elicit_dict : dict
            Dictionary including the quantile settings.

        Raises
        ------
        ValueError
            ``quantiles`` have to be specified as probability ranging between
            0 and 1.

        """
        # compute percentage from probability
        quantiles_perc = tuple([q * 100 for q in quantiles])

        # check that quantiles are provided as percentage
        for quantile in quantiles:
            if (quantile < 0) or (quantile > 1):
                raise ValueError(
                    "[section: targets] Quantiles have to be expressed as"
                    + " probability (between 0 and 1)."
                    + f" Found quantile={quantile}"
                )

        elicit_dict = dict(name="quantiles", value=quantiles_perc)
        return elicit_dict

    def identity(self) -> dict:
        """
        Implements an identity function. Should be used if no further
        transformation of target quantity is required.

        Returns
        -------
        elicit_dict : dict
            Dictionary including the identity settings.

        """
        elicit_dict = dict(name="identity", value=None)
        return elicit_dict

    def correlation(self) -> dict:
        """
        Implements a method to calculate the pearson correlation between
        model parameters.

        Returns
        -------
        elicit_dict : dict
            Dictionary including the correlation settings.

        """
        elicit_dict = dict(name="pearson_correlation", value=None)
        return elicit_dict

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

        Raises
        ------
        NotImplementedError
            This option for implementing a custom elicitation method is not
            implemented yet.

        """  # noqa: E501
        raise NotImplementedError(
            "[section targets]: The use of custom elicitation methods "
            + "hasn't been implemented yet."
        )
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
) -> dict:
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

    Raises
    ------
    NotImplementedError
        ``target_method`` for implementing a custom target quantity is not
        implemented yet.

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
            "[section: targets] The use of a custom target quantity hasn't"
            + " been implemented yet."
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
    def data(self, dat: dict[str, list]) -> dict[str, dict]:
        """
        Provide elicited-expert data for learning prior distributions.

        Parameters
        ----------
        dat : dict
            Elicited data from expert provided as dictionary. Data must be
            provided in a standardized format.
            Use :func:`elicit.utils.get_expert_datformat` to get correct data
            format for your method specification.

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
        # Note: check for correct expert data format is done in Elicit class
        dat_prep = {
            f"{key}": tf.expand_dims(
                tf.cast(tf.convert_to_tensor(dat[key]), dtype=tf.float32), 0
            )
            for key in dat
        }

        return dict(data=dat_prep)

    def simulator(self, ground_truth: dict, num_samples: int = 10_000) -> dict:
        """
        Simulate data from an oracle by defining a ground truth (true prior
        distribution(s)).
        See `Explanation: Simulating from an oracle (TODO) <url>`_ for
        further details.

        Parameters
        ----------
        ground_truth : dict
            True prior distribution(s). *Keys* refer to parameter names and
            *values* to prior distributions implemented as
            `tfp.distributions <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions>`_
            object with predetermined hyperparameter values.
            You can specify a prior distribution for each model parameter or
            a joint prior for all model parameters at once or any approach in
            between. Only requirement is that the dimensionality of all priors
            in ground truth match with the number of model parameters.
            Order of priors in ground truth must match order of
            :func:`elicit.elicit.Elicit` argument ``parameters``.
        num_samples : int
            Number of draws from the prior distribution.
            It is recommended to use a high value to min. sampling variation.
            The default is ``10_000``.

        Returns
        -------
        expert_data : dict
            Settings of oracle for simulating from ground truth. True elicited
            statistics are used as `expert-data` in loss function.

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

        >>> el.expert.simulator(
        >>>     ground_truth = {
        >>>         "betas": tfd.MultivariateNormalDiag([5.,2.], [1.,1.]),
        >>>         "sigma": tfd.HalfNormal(scale=10.0),
        >>>     },
        >>>     num_samples = 10_000
        >>> )

        >>> el.expert.simulator(
        >>>     ground_truth = {
        >>>         "thetas": tfd.MultivariateNormalDiag([5.,2.,1.],
        >>>                                              [1.,1.,1.]),
        >>>     },
        >>>     num_samples = 10_000
        >>> )
        """  # noqa: E501
        # Note: check whether dimensionality of ground truth and number of
        # model parameters is identical is done in Elicit class

        expert_data = dict(ground_truth=ground_truth,
                           num_samples=int(num_samples))
        return expert_data


# create an instantiation of Expert class
expert = Expert()


def optimizer(optimizer: callable = tf.keras.optimizers.Adam(),
              **kwargs) -> dict:
    """
    Specification of optimizer and its settings for SGD.

    Parameters
    ----------
    optimizer : callable, `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ object.
        Optimizer used for SGD implemented.
        Must be an object implemented in `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ object.
        The default is ``tf.keras.optimizers.Adam``.
    **kwargs : keyword arguments, optional
        Additional keyword arguments expected by **optimizer**.

    Returns
    -------
    optimizer_dict : dict
        Dictionary specifying the SGD optimizer and its additional settings.

    Raises
    ------
    TypeError
        ``optimizer`` is not a tf.keras.optimizers object
    ValueError
        ``optimizer`` could not be found in tf.keras.optimizers

    Examples
    --------
    >>> optimizer=el.optimizer(
    >>>     optimizer=tf.keras.optimizers.Adam,
    >>>     learning_rate=0.1,
    >>>     clipnorm=1.0
    >>> )
    """  # noqa: E501

    # check whether optimizer is a tf.keras.optimizers object
    opt_module = ".".join(optimizer.__module__.split(".")[:-1])
    if opt_module != "keras.src.optimizers":
        raise TypeError(
            "[section: optimizer] The 'optimizer' must be a"
            + " tf.keras.optimizers object."
        )

    # check whether the optimizer object can be found in tf.keras.optimizers
    opt_name = str(optimizer).split(".")[-1][:-2]
    if opt_name not in dir(tf.keras.optimizers):
        raise ValueError(
            "[section: optimizer] The argument 'optimizer' has to be a"
            + " tf.keras.optimizers object."
            + f" Couldn't find {opt_name} in list of tf.keras.optimizers."
        )

    optimizer_dict = dict(optimizer=optimizer)
    for key in kwargs:
        optimizer_dict[key] = kwargs[key]

    return optimizer_dict


def initializer(
    method: str,
    distribution: callable = el.initialization.uniform(),
    loss_quantile: float = .0,
    iterations: int = 100,
) -> dict:
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
    loss_quantile : float,
        Quantile indicating which loss value should be used for selecting the
        initial hyperparameters.Specified as probability value between 0-1.
        The default is ``0`` i.e., the minimum loss.
    iterations : int
        Number of samples drawn from the initialization distribution.
        The default is ``100``.

    Returns
    -------
    init_dict : dict
        Dictionary specifying the initialization method.

    Raises
    ------
    ValueError
        ``method`` can only take the values "random", "sobol", or "lhs"

        ``loss_quantile`` must be a probability ranging between 0 and 1.

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
    # compute percentage from probability
    quantile_perc = loss_quantile * 100

    # check that method is implemented
    if method not in ["random", "lhs", "sobol"]:
        raise ValueError(
            "[section: initializer] Currently implemented initialization"
            + f" methods are 'random', 'sobol', and 'lhs', but got '{method}'"
            + " as input."
        )

    # check that quantile is provided as probability
    if (quantile_perc < 0) or (quantile_perc > 1):
        raise ValueError(
            "[section: initializer] 'loss_quantile' must be a value between 0"
            + f" and 1. Found 'loss_quantile={loss_quantile}'."
        )

    init_dict = dict(
        method=method,
        distribution=distribution,
        loss_quantile=loss_quantile,
        iterations=int(iterations),
    )

    return init_dict


def trainer(
    method: str,
    seed: int,
    epochs: int,
    B: int = 128,
    num_samples: int = 200
):
    """
    Specification of training settings for learning the prior distribution(s).

    Parameters
    ----------
    method : str
        Method for learning the prior distribution. Available is either
        ``parametric_prior`` for learning independent parametric priors
        or ``deep_prior`` for learning a joint non-parameteric prior.
    seed : int
        seed used for learning.
    epochs : int
        number of iterations until training is stopped.
    B : int, optional
        batch size. The default is 128.
    num_samples : int, optional
        number of samples from the prior(s). The default is 200.

    Returns
    -------
    train_dict : dict
        dictionary specifying the training settings for learning the prior
        distribution(s).

    Raises
    ------
    ValueError
        ``method`` can only take the value "parametric_prior" or "deep_prior"

    Examples
    --------
    >>> el.trainer(
    >>>     method="parametric_prior",
    >>>     seed=0,
    >>>     epochs=400,
    >>>     B=128,
    >>>     num_samples=200
    >>> )
    """  # noqa: E501
    # check that method is implemented
    if method not in ["parametric_prior", "deep_prior"]:
        raise ValueError(
            "[section: trainer] Currently only the methods 'deep_prior' and"
            + f" 'parametric prior' are implemented but got '{method}'."
        )

    train_dict = dict(
        method=method,
        seed=int(seed),
        B=int(B),
        num_samples=int(num_samples),
        epochs=int(epochs)
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
        network : callable or None
            specification of neural network using a method implemented in
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

        Raises
        ------
        AssertionError
            ``expert`` data are not in the required format. Correct specification of
            keys can be checked using el.utils.get_expert_datformat

            Dimensionality of ``ground_truth`` for simulating expert data, must be
            the same as the number of model parameters.

        ValueError
            if ``method="deep_prior"``, ``network`` can't be None and ``initialization``
            should be None.

            if ``method="parametric_prior"``, ``network`` should be None and
            ``initialization`` can't be None.

            if ``method ="parametric_prior" and multiple hyperparameter have
            the same name but are not shared by setting ``shared=True``."

        """  # noqa: E501
        # check expert data
        expected_dict = el.utils.get_expert_datformat(targets)
        try:
            expert["ground_truth"]
        except KeyError:
            # input expert data: ensure data has expected format
            if list(expert["data"].keys()) != list(expected_dict.keys()):
                raise AssertionError(
                    "[section: expert] Provided expert data is not in the "
                    + "correct format. Please use "
                    + "el.utils.get_expert_datformat to check expected format."
                )
        else:
            # oracle: ensure ground truth has same dim as number of model param
            expected_params = [param["name"] for param in parameters]
            num_params = 0
            for k in expert["ground_truth"]:
                num_params += expert["ground_truth"][k].sample(1).shape[-1]

            if len(expected_params) != num_params:
                raise AssertionError(
                    "[section: expert] Dimensionality of ground truth in"
                    + " 'expert' is not the same  as number of model"
                    + f" parameters.Got {num_params}, expected"
                    + f" {len(expected_params)}."
                )
        # check that network architecture is provided when method is deep prior
        # and initializer is none
        if trainer["method"] == "deep_prior":
            if network is None:
                raise ValueError(
                    "[section network] If method is 'deep prior',"
                    + " the section 'network' can't be None."
                )
            if initializer is not None:
                raise ValueError(
                    "[section initializer] For method 'deep_prior' the "
                    + "'initializer' is not used and should be set to None."
                )
        # check that initializer is provided when method=parametric prior
        # and network is none
        if trainer["method"] == "parametric_prior":
            if initializer is None:
                raise ValueError(
                    "[section initializer] If method is 'parametric_prior',"
                    + " the section 'initializer' can't be None."
                )
            if network is not None:
                raise ValueError(
                    "[section network] If method is 'parametric prior'"
                    + " the 'network' is not used and should be set to None."
                )
        # check that hyperparameter names are not redundant
        if trainer["method"] == "parametric_prior":
            hyp_names = []
            hyp_shared = []
            for i in range(len(parameters)):
                hyp_names.append(
                    [parameters[i]["hyperparams"][key]["name"] for
                     key in parameters[i]["hyperparams"].keys()])
                hyp_shared.append(
                    [parameters[i]["hyperparams"][key]["shared"] for
                     key in parameters[i]["hyperparams"].keys()])
            # flatten nested list
            hyp_names_flat = sum(hyp_names, [])
            hyp_shared_flat = sum(hyp_shared, [])

            seen = []
            duplicate = []
            share = []
            for n, s in zip(hyp_names_flat, hyp_shared_flat):
                if n not in seen:
                    seen.append(n)
                else:
                    if s:
                        share.append(n)
                    else:
                        duplicate.append(n)

            if len(duplicate) != 0:
                raise ValueError(
                    "[parameters] The following hyperparameter have the same"
                    +f" name but are not shared: {duplicate}."
                    +" Have you forgot to set shared=True?")

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

    def fit(self,
            overwrite=False,
            save_history: callable = el.utils.save_history(),
            save_results: callable = el.utils.save_results()
            ) -> None:
        """
        method for fitting the eliobj and learn prior distributions.

        Parameters
        ----------
        overwrite : bool, optional
            If the eliobj was already fitted and the user wants to refit it,
            the user is asked whether they want to overwrite the previous
            fitting results. Setting ``overwrite=True`` allows the user to
            force overfitting without being prompted. The default is ``False``.
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

        Examples
        --------
        >>> eliobj.fit()

        >>> eliobj.fit(overwrite=True,
        >>>            save_history=el.utils.save_history(
        >>>                loss_component=False
        >>>                )
        >>>            )

        """  # noqa: E501
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

        for key_hist in save_history:
            if not save_history[key_hist]:
                self.history.pop(key_hist)

        for key_res in save_results:
            if not save_results[key_res]:
                self.results.pop(key_res)

    def save(
        self,
        name: str or None = None,
        file: str or None = None,
        overwrite: bool = False,
    ):
        """
        method for saving the eliobj on disk

        Parameters
        ----------
        name: str or None
            file name used to store the eliobj. Saving is done
            according to the following rule: ``./{method}/{name}_{seed}.pkl``
            with 'method' and 'seed' being arguments of
            :func:`elicit.elicit.trainer`.
        file : str or None
            user-specific path for saving the eliobj. If file is specified
            **name** must be ``None``. The default value is ``None``.
        overwrite : bool, optional
            If already a fitted object exists in the same path, the user is
            asked whether the eliobj should be refitted and the results
            overwritten. With the ``overwrite`` argument you can silent this
            behavior. In this case the results are automatically overwritten
            without prompting the user. The default is ``False``.

    Raises
    ------
    AssertionError
        ``name`` and ``file`` can't be specified simultaneously.

        Examples
        --------
        >>> eliobj.save(name="toymodel")

        >>> eliobj.save(file="res/toymodel", overwrite=True)

        """
        # check that either name or file is specified
        if not (name is None) ^ (file is None):
            raise  AssertionError(
            "Name and file cannot be both None or both specified."
            + " Either one has to be None."
        )

        # add a saving path
        return el.utils.save(self, name=name, file=file, overwrite=overwrite)

    def update(self, **kwargs):
        """
        method for updating the attributes of the Elicit class. Updating
        an eliobj leads to an automatic reset of results.

        Parameters
        ----------
        **kwargs : any
            keyword argument used for updating an attribute of Elicit class.
            Key must correspond to one attribute of the class and value refers
            to the updated value.

        Raises
        ------
        ValueError
            key of provided keyword argument is not an eliobj attribute. Please
            check dir(eliobj).

        Examples
        --------
        >>> eliobj.update(parameter = updated_parameter_dict")

        """
        # check that arguments exist as eliobj attributes
        for key in kwargs:
            if str(key) not in [
                "model",
                "parameters",
                "targets",
                "expert",
                "trainer",
                "optimizer",
                "network",
                "initializer",
            ]:
                raise ValueError(
                    f"{key} is not an eliobj attribute."
                    + " Use dir() to check for attributes."
                )

        for key in kwargs:
            setattr(self, key, kwargs[key])
            # reset results
            self.results = dict()
            self.history = dict()
            # inform user about reset of results
            print("INFO: Results have been reset.")
