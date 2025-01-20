# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import elicit as el
import pandas as pd
import pickle
import os


def save_as_pkl(obj: any, save_dir: str):
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


def identity(x):
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
        # logarithmic transform
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
        # exponential transform
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


def one_forward_simulation(prior_model, trainer, model, targets):
    """
    One forward simulation from prior samples to elicited statistics.

    Parameters
    ----------
    prior_model : instance of Priors class objects
        initialized prior distributions which can be used for sampling.
    global_dict : dict
        global dictionary with all user input specifications.
    ground_truth : bool, optional
        Is true if model should be learned with simulated data that
        represent a pre-defined ground truth. The default is False.

    Returns
    -------
    elicited_statistics : dict
        dictionary containing the elicited statistics that can be used to
        compute the loss components

    """
    # set seed
    tf.random.set_seed(trainer["seed"])
    # generate samples from initialized prior
    prior_samples = prior_model()
    # simulate prior predictive distribution based on prior samples
    # and generative model
    model_simulations = el.simulations.simulate_from_generator(
        prior_samples, trainer["seed"], model
    )
    # compute the target quantities
    target_quantities = el.targets.computation_target_quantities(
        model_simulations, targets
    )
    # compute the elicited statistics by applying a specific elicitation
    # method on the target quantities
    elicited_statistics = el.targets.computation_elicited_statistics(
        target_quantities, targets
    )
    return (elicited_statistics, prior_samples, model_simulations,
            target_quantities)


#%% simulate expert data or get input data
def get_expert_data(trainer, model, targets, expert, parameters, network):
    """
    Wrapper for loading the training data which can be expert data or
    data simulations using a pre-defined ground truth.

    Parameters
    ----------
    global_dict : dict
        global dictionary with all user input specifications.
    path_to_expert_data : str, optional
        path to file location where expert data has been saved

    Returns
    -------
    expert_data : dict
        dictionary containing the training data. Must have same form as the
        model-simulated elicited statistics.

    """

    try:
        expert["data"]
    except KeyError:
        oracle=True
    else:
        oracle=False

    if oracle:
        # set seed
        tf.random.set_seed(trainer["seed"])
        # sample from true priors
        prior_model = el.simulations.Priors(
            ground_truth=True, 
            init_matrix_slice=None,
            trainer=trainer, parameters=parameters, network=network,
            expert=expert,
            seed=trainer["seed"])
        # compute elicited statistics and target quantities
        expert_data, expert_prior, *_ = one_forward_simulation(
            prior_model, trainer, model, targets
        )
        return expert_data, expert_prior

    else:
        # load expert data from file
        # TODO Expert data must have same name and structure as sim-based
        # elicited statistics
        expert_data = expert["data"]
        return expert_data, None


def save(eliobj: callable, path: str, overwrite: bool = False):
    # create saving path
    path = f"./{path}"+eliobj.trainer["output_path"]
    # check whether saving path is already used
    if os.path.isfile(path) and not overwrite:
        user_ans = input("In provided directory exists already a file with"+
                         " identical name. Do you want to overwrite it?"+
                         " Press 'y' for overwriting and 'n' for abording.")
        while user_ans not in ["n", "y"]:
            user_ans = input("Please press either 'y' for overwriting or 'n'"+
                             "for abording the process.")

        if user_ans == "n":
            return("Process aborded. File is not overwritten.")

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

    save_as_pkl(storage, path)
    print(f"saved in: {path}")


def load(path: str):
    obj = pd.read_pickle(path)

    eliobj = el.Elicit(
        model = obj["model"],
        parameters = obj["parameters"],
        targets = obj["targets"],
        expert = obj["expert"],
        optimizer = obj["optimizer"],
        trainer = obj["trainer"],
        initializer = obj["initializer"],
        network = obj["network"]
        )

    # add results if already fitted
    eliobj.history = obj["history"]
    eliobj.results = obj["results"]

    return eliobj


def save_history(
    loss: bool = True,
    loss_component: bool = True,
    time: bool = True,
    hyperparameter: bool = True,
    hyperparameter_gradient: bool = True,
):
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

    """
    if not loss or not loss_component:
        print(
            "INFO: el.plots.loss() requires information about "+
            "'loss' and 'loss_component'. If you don't save this information "+
            "el.plot.loss() can't be used.")

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
    model: bool = True,
    expert_elicited_statistics: bool = True,
    expert_prior_samples: bool = True,
    init_loss_list: bool = True,
    init_prior: bool = True,
    init_matrix: bool = True,
    loss_tensor_expert: bool = True,
    loss_tensor_model: bool = True,
):
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
    model : bool, optional
        fitted elicit model object including the trainable variables.
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

        """
    if not elicited_statistics:
        print(
            "INFO: el.plots.loss() requires information about "+
            "'elicited_statistics'. If you don't save this information "+
            "el.plot.loss() can't be used.")

    save_res_dict = dict(
        target_quantities=target_quantities,
        elicited_statistics=elicited_statistics,
        prior_samples=prior_samples,
        model_samples=model_samples,
        model=model,
        expert_elicited_statistics=expert_elicited_statistics,
        expert_prior_samples=expert_prior_samples,
        init_loss_list=init_loss_list,
        init_prior=init_prior,
        init_matrix=init_matrix,
        loss_tensor_expert=loss_tensor_expert,
        loss_tensor_model=loss_tensor_model,
    )
    return save_res_dict