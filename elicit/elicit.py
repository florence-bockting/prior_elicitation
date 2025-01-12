# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import logging
import elicit as el
import sys


tfd = tfp.distributions


class Dtype: 
    def __init__(self, vtype, dim):
        self.vtype = vtype
        self.dim = dim

    def __call__(self, x):

        if self.vtype=="real":
            dtype_dim = tf.cast(x, dtype=tf.float32)
        elif self.vtype=="array":
            dtype_dim = tf.constant(x, dtype=tf.float32, shape=(self.dim,))
        return dtype_dim


def hyper(name: str, lower: float=float("-inf"), upper: float=float("inf"),
          vtype: str="real", dim: int=1, shared: bool=False):
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

    """
    # constraints
    # only lower bound
    if (lower != float("-inf")) and (upper == float("inf")):
        lower_bound = el.helpers.LowerBound(lower)
        transform=lower_bound.inverse
    # only upper bound
    elif ((upper != float("inf")) and (lower == float("-inf"))):
        upper_bound = el.helpers.UpperBound(upper)
        transform=upper_bound.inverse
    # upper and lower bound
    elif ((upper != float("inf")) and (lower != float("-inf"))):
        double_bound = el.helpers.DoubleBound(lower, upper)
        transform=double_bound.inverse
    # unbounded
    else:
        transform=el.helpers.identity

    # value type
    dtype_dim = Dtype(vtype, dim)

    hyppar_dict = dict(
        name = name,
        constraint = transform,
        vtype = dtype_dim,
        dim = dim,
        shared = shared
    )

    return hyppar_dict


def parameter(name: str, family: callable or None=None,
              hyperparams: dict or None=None):
    """
    Specification of model parameters.

    Parameters
    ----------
    name : string
        Custom name of parameter.
    family : callable or None
        Prior distribution family for model parameter.
        Only required for method ``parametric_prior``.
        Must be an `tfp.distributions <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions>`_ object.
    hyperparams : dict or None
        Hyperparameters of distribution as specified in **family**.
        Only required for method ``parametric_prior``.
        Structure of dictionary: Keys must match arguments of
        `tfp.distributions <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions>`_
        object and values have to be specified using the :func:`hyper` method.
        Further details are provided in `How-To specify prior hyperparameters (TODO) <url>`_.

    Returns
    -------
    param_dict : dict
        Dictionary including all model parameter settings.

    """

    param_dict = dict(
        name=name,
        family=family,
        hyperparams=hyperparams
    )

    return param_dict


def model(obj: callable, **kwargs):
    """
    Specification of generative model.

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

    """
    generator_dict = dict(
        obj=obj
        )

    for key in kwargs:
        generator_dict[key] = kwargs[key]

    return generator_dict


class Queries:
    def quantiles(self, quants: tuple):
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
        return dict(name="quantiles", value=quants)


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

        Parameters
        ----------
        func : callable
            Custom target method.

        Returns
        -------
        elicit_dict : dict
            Dictionary including the custom settings.

        """
        args_dict=dict()
        for key in kwargs:
            args_dict[key] = kwargs[key]
        
        return dict(name="custom",
                    value=func,
                    add_args=args_dict)

# create an instance of the ElicitationMethod class
queries = Queries()


def target(name : str, loss : callable,
           query : callable=queries,
           target_method : callable=None, weight : float=1.0):
    """
    Specification of target quantity and corresponding elicitation technique.

    Parameters
    ----------
    name : string
        Name of the target quantity. Two approaches are possible:
        (1) Target quantity is identical to an output from the generative
        model: The name must match the output variable name. (2) Target
        quantity is computed via custom method: Provide custom function via 
        **target_method**.
    elicitation_method : callable
        Specify built-in elicitation technique or use the ``custom`` method to
        pass a custom elicitation method.
        The default is an instance of :class:`ElicitationMethod`.
        See `How-To specify custom elicitation techniques (TODO) <url>`_.
    loss : callable
        Loss function for computing the discrepancy between expert data and
        model simulations.
        Implemented classes can be found in :doc:`elicit.loss_functions`.
    target_method : callable, optional
        Custom method for computing a target quantity.
        See `How-To specify a custom target quantity (TODO) <url>`_.
        The default is ``None``.
    loss_weight : float, optional
        Weight used for the elicited quantity in the total loss.
        The default is ``1.0``.

    Returns
    -------
    target_dict : dict
        Dictionary including all settings regarding the target quantity and
        corresponding elicitation technique.

    """
    # create instance of loss class
    loss_instance = loss

    target_dict = dict(
        name=name,
        query=query,
        target_method=target_method,
        loss=loss_instance,
        weight=weight
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

        """
        dat_prep = {f"{key}": tf.expand_dims(tf.cast(
                        tf.convert_to_tensor(dat[key]),
                        dtype=tf.float32), 0) for key in dat}

        return dict(data=dat_prep)


    def simulator(self, ground_truth: dict, num_samples: int=10_000):
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

        """
        return dict(ground_truth=ground_truth, num_samples=num_samples)

expert = Expert()


def optimizer(optimizer: callable=tf.keras.optimizers.Adam, **kwargs):
    """
    Specification of optimizer and its settings for SGD.

    Parameters
    ----------
    optimizer : callable, `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ object
        Optimizer used for SGD implemented as `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ object.
        The default is ``tf.keras.optimizers.Adam``.
    **kwargs : keyword arguments, optional
        Additional keyword arguments expected by **optimizer**.

    Returns
    -------
    optimizer_dict : dict
        Dictionary specifying the SGD optimizer and its additional settings.

    """
    optimizer_dict = dict(optimizer=optimizer)
    for key in kwargs:
        optimizer_dict[key] = kwargs[key]

    return optimizer_dict


def initializer(method: str, distribution: el.initialization.uniform, 
                loss_quantile: int=0, iterations: int=100):
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
        Name of initialization method. Currently supported are "random" and
        "sobol".
    specs : callable, :func:`init_specs`
        Specification of uniform distribution from which initial values are
        drawn per hyperparameter 
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

    """
    init_dict = dict(
        method=method,
        distribution=distribution,
        loss_quantile=loss_quantile,
        iterations=iterations
        )

    return init_dict


def trainer(method: str, name: str, seed: int, epochs: int, B: int=128,
          num_samples: int=200, output_path: str="results",
          save_log: bool=False):
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
        number of draws sampled from the prior(s). The default is 200.
    output_path : str, optional
        path (incl. foldername) in which results are stored.
        The default is "results".
    progress_info : int, optional
        whether information about loss, remaining time, etc should be printed
        during training. ``1`` prints information and ``0`` is silence.
        The default is 1.
    view_ep : int, optional
        if information regarding the training progress should be printed, this
        arguments allow to specify after how many epochs information should be
        printed. The default is 1, thus information is printed after each
        iteration.
    save_log : bool, optional
        Saves the computational pipeline (which computation is carried out)
        in an extra file. Might help to understand the computational steps
        carried out by the method. The default is False.

    Returns
    -------
    train_dict : dict
        dictionary specifying the training settings for learning the prior
        distribution(s).

    """
    train_dict = dict(
        method=method,
        name=name,
        seed=seed,
        B=B,
        num_samples=num_samples,
        epochs=epochs,
        output_path=output_path,
        save_log=save_log
    )
    return train_dict


class Elicit:

   def __init__(self,
        model: callable,
        parameters: list,
        targets: list,
        expert: callable,
        trainer: callable,
        optimizer: callable,
        network: callable or None=None,
        initializer: callable or None=None
    ):
        """
        Parameters
        ----------
        model : callable
            specification of generative model using
            :func:`elicit.prior_elicitation.generator`.
        parameter : list
            list of model parameters specified with
            :func:`elicit.prior_elicitation.par`.
        targets : list
            list of target quantities specified with
            :func:`el.target`.
        expert : callable
            specification of input data from expert or oracle using
            :func:`el.expert.data` or func:`el.expert.simulate`
        trainer : callable
            specification of training settings for learning prior distribution(s)
            using :func:`elicit.prior_elicitation.train`
        optimizer : callable
            specification of optimizer using
            :func:`elicit.prior_elicitation.optimizer`.
        normalizing_flow : callable or None
            specification of normalizing flow using :func:`elicit.prior_elicitation.nf`.
            Only required for ``deep_prior`` method is used. If 
            ``parametric_prior`` is used this argument should be ``None``. Default
            value is None.
        initializer : callable
            specification of initialization settings using
            :func:`elicit.prior_elicitation.initializer`. For method
            'parametric_prior' the argument should be None. Default value is None.

        Returns
        -------
        global_dict : dict
            specification of all settings to run the optimization procedure.

        """
        self.model=model
        self.parameters=parameters
        self.targets=targets
        self.expert=expert
        self.trainer=trainer
        self.optimizer=optimizer
        self.network=network
        self.initializer=initializer

        self.history=dict()
        self.results=dict()

        # set seed
        tf.random.set_seed(self.trainer["seed"])

   def fit(self, save_dir: str or None=None, silent=False):
        # check whether elicit object is already fitted
        if len(self.history.keys()) != 0:
            user_answ = input("elicit object is already fitted."+
                 " Do you want to fit it again and overwrite the results?"+
                 " Press 'n' to stop process and 'y' to continue fitting.")
           
            while user_answ not in ["n", "y"]:
                user_answ = input("Please press either 'y' for fitting or 'n'"+
                                 " for abording the process.")

            if user_answ == "n":
                return("Process aborded; elicit object is not re-fitted.")

        logger = logging.getLogger(__name__)

        # set seed
        tf.random.set_seed(self.trainer["seed"])

        if save_dir is not None:
            # create saving path
            self.trainer["output_path"
            ] = f"./elicit/{save_dir}/{self.trainer['method']}/{self.trainer['name']}_{self.trainer['seed']}"  # noqa
        else:
            self.trainer["output_path"] = None

        # get expert data
        expert_elicits, expert_prior = el.utils.get_expert_data(
            self.trainer, self.model, self.targets, self.expert,
            self.parameters, self.network)

        # initialization of hyperparameter
        (init_prior_model, loss_list,
         init_prior, init_matrix) = el.initialization.pre_training(
            expert_elicits, self.initializer, self.parameters, self.trainer,
            self.model, self.targets, self.network, self.expert)

        # run dag with optimal set of initial values
        logger.info("Training Phase (only first epoch)")
        # save results in corresp. attributes
        self.history, self.results = el.optimization.sgd_training(
            expert_elicits,
            init_prior_model,
            self.trainer,
            self.optimizer,
            self.model,
            self.targets
        )

        self.results["expert_elicited_statistics"] = expert_elicits
        try:
            self.expert["ground_truth"]
        except KeyError:
            pass
        else:
            self.results["expert_prior_samples"] = expert_prior

        if self.trainer["method"]=="parametric_prior":
            self.results["init_loss_list"] = loss_list
            self.results["init_prior"] = init_prior
            self.results["init_matrix"] = init_matrix

        # remove saved files that are not of interest for follow-up analysis
        if save_dir is not None:
            el.remove_unneeded_files(self.trainer["output_path"], save_dir)  # noqa

        # return history by default
        if not silent:
            return self.history