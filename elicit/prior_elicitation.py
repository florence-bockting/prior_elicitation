import tensorflow as tf
import tensorflow_probability as tfp

from elicit.helper_functions import LowerBound, UpperBound, DoubleBound, identity

tfd = tfp.distributions


def hyppar(name: str, lower: float = float("-inf"),
           upper: float = float("inf"),
           vtype: str ="real", dim: int = 1):
    """
    Specification of prior hyperparameter with respect to name and value
    constraints.

    Parameters
    ----------
    name : str
        Name of the hyperparameter.
    lower : float, optional
        Whether the hyperparameter has a lower bound.
        The default is float("-inf"), thus unbounded.
    upper : float, optional
        Whether the hyperparameter has an upper bound.
        The default is float("inf"), thus unbounded.
    vtype : string, optional
        Variable type. Can be either "real" or "array". The default is "real".
    dim : integer, optional
        Dimensionality of variable. The default for type "array" is 1.

    Returns
    -------
    hyppar_dict : dict
        dictionary including hyperparameter details such as name, constraints
        and variable type/dimensionality.

    """
    # constraints
    # only lower bound
    if (lower != float("-inf")) and (upper == float("inf")):
        lower_bound = LowerBound(lower)
        transform=lower_bound.inverse
    # only upper bound
    elif ((upper != float("inf")) and (lower == float("-inf"))):
        upper_bound = UpperBound(upper)
        transform=upper_bound.inverse
    # upper and lower bound
    elif ((upper != float("inf")) and (lower != float("-inf"))):
        double_bound = DoubleBound(lower, upper)
        transform=double_bound.inverse
    # unbounded
    else:
        transform=identity

    # value type
    if "real":
        dtype_dim=lambda x: tf.cast(x, dtype=tf.float32)
    elif "array":
        dtype_dim=lambda x: tf.constant(x, dtype=tf.float32, shape=(dim,))

    hyppar_dict = dict(
        name = name,
        constraint = transform,
        vtype = dtype_dim
    )

    return hyppar_dict

def par(name: str, family: callable or None=None,
        hyperparams: dict or None=None):
    """
    Specification of model parameters with respect to name and if needed
    parametric prior distribution family with corresponding hyperparameter.

    Parameters
    ----------
    name : str
        Name of model parameter.
    family : callable or None
        prior distribution family for model parameter. Only required for 
        parametric prior method. Must be a tfp.distribution object.
    hyperparams : dict or None
        Specification of hyperparameters of parametric prior distribution
        family. Only required for parametric prior method. See function 
        'hyppar' for details on how to specify the hyperparameter.

    Returns
    -------
    param_dict : dict
        dictionary with specifications of model parameter with respect to its
        name, prior family (optional), and hyperparameter specification
        (optional).

    """

    param_dict = dict(
        name=name,
        family=family,
        hyperparams=hyperparams
    )

    return param_dict

def softmax_gumble_specs(upper_threshold: int, temperature: float=1.6):
    """
    Specification of hyperparameters of softmax gumble trick (used to estimate
    gradients for discrete random variables).

    Parameters
    ----------
    upper_threshold : int
        refers to the upper bound for double bounded likelihoods. If likelihood
        has no upper bound, then this value introduces an artificial truncation
        of the distribution on the right.
    temperature : float
        temperature hyperparameter of the softmax gumble trick.
        The default is 1.6.

    Returns
    -------
    softmax_dict : dict
        dictionary with specifications of hyperparameters required by the 
        softmax-gumble trick.

    """
    softmax_dict = dict(
        temperature=temperature,
        upper_threshold=upper_threshold
        )
    
    return softmax_dict

def generator(model: callable, additional_model_args: dict or None,
              discrete_likelihood: bool=False,
              approx_technique: str="softmax-gumble",
              approx_technique_specs: dict or None=None):
    """
    Specification of the generative model.

    Parameters
    ----------
    model : callable
        class that implements the generative model. See generative_model class
        for details.
    additional_model_args : dict or None
        if the generative model requires additional arguments they can be
        specified here.
    discrete_likelihood : bool
        indicates whether the likelihood is continuous or discrete.
        The default is False.
    approx_technique : str, optional
        if the .likelihood is discrete this argument specifies the technique
        used to get an continuous approximation.
        The default is "softmax-gumble".
    approx_technique_specs : dict, optional
        if the approximation technique requires additional arguments they can
        be specified here.
        The default is None.

    Returns
    -------
    generator_dict : dict
        dictionary specifying the generative model.

    """
    generator_dict = dict(
        model=model,
        additional_model_args=additional_model_args,
        discrete_likelihood=discrete_likelihood,
        approx_technique=approx_technique,
        approx_technique_specs=approx_technique_specs)
    
    return generator_dict

def tar(name: str, elicitation_method: str, loss: callable,
        custom_target_function: callable=None,
        custom_elicitation_function: callable=None,
        loss_weight: float=1.0, quantiles_specs: tuple or None=None):
    """
    Specification of target quantity and corresponding elicitation technique.

    Parameters
    ----------
    name : str
        name of the target quantity. If name matches an output variable from
        the generative model, this output is used.
    elicitation_method : str
        specify one of the built-in elicitation techniques.
    loss : callable
        loss function used for computing the discrepancy between the expert
        data and the model simulations.
    custom_target_function : callable, optional
        specification of a custom target quantity. The default is None.
    custom_elicitation_function : callable, optional
        specification of a custom elicitation technique. The default is None.
    loss_weight : float, optional
        Weighting of this elicited quantity in the total loss.
        The default is 1.0.
    quantiles_specs : tuple or None, optional
        only required if elicitation_method is quantiles. Specifies the 
        quantiles that should be computed from the target quantities. 
        Quantiles should be provided in a tuple and range between 0 and 100%.
        The default is None.

    Returns
    -------
    target_dict : dict
        dictionary specifying the target quantity and corresponding elicitation
        technique.

    """
    target_dict = dict(
        name=name,
        elicitation_method=elicitation_method,
        quantiles_specs=quantiles_specs,
        loss=loss,
        loss_weight=loss_weight,
        custom_elicitation_function=custom_elicitation_function,
        custom_target_function=custom_target_function
    )

    return target_dict


def expert_input(data: dict or None, from_ground_truth: bool=False,
                 simulator_specs: dict or None=None,
                 samples_from_prior: int or None=10_000):
    """
    Specification of expert input. Two approaches are possible, either providing
    the expert data in a dictionary or simulating expert data from an oracle
    (i.e., specifying a ground truth). The latter approach can for example be
    used for model checking or sensitivity analysis.

    Parameters
    ----------
    data : dict or None
        expert input provided as dictionary of specific form. See <TODO> for
        details. If expert data should be simulated from a ground truth, use
        'None' here.
    from_ground_truth : bool
        whether expert data should be simulated from a prespecified ground
        truth. The default is False. If True the data argument should be None.
    simulator_specs : dict or None, optional
        if data are simulated from ground truth, the ground truth can be
        specified here in terms of a dictionary of specific from. For details
        see <ToDo>. The default is None.
    samples_from_prior : int or None, optional
        if data are simulated from ground truth, this argument specifies how
        many samples should be drawn from the true prior distribution(s). It is
        recommended to use a large number of samples to reduce sampling
        variation. The default is 10_000.

    Returns
    -------
    expert_dict : dict
        dictionary specifying the expert input and, if an oracle is used,
        the true prior distribution(s).

    """
    expert_dict = dict(
        data=data,
        from_ground_truth=from_ground_truth,
        simulator_specs = simulator_specs,
        samples_from_prior = samples_from_prior
    )
    
    return expert_dict

def optimizer(optimizer_specs: dict, 
              optimizer: callable=tf.keras.optimizers.Adam):
    """
    Specification of optimizer and its settings for SGD.

    Parameters
    ----------
    optimizer_specs : dict
        dictionary specifying additional parameter required for the optimizer
        (e.g., the learning rate)
    optimizer : callable, optional
        optimizer used for SGD. Must be selected from tf.keras.optimizers.
        The default is tf.keras.optimizers.Adam.

    Returns
    -------
    optimizer_dict : dict
        dictionary specifying the SGD optimizer and its additional settings.

    """
    optimizer_dict = dict(
        optimizer=optimizer,
        optimizer_specs=optimizer_specs
        )
    return optimizer_dict


def initializer(method: str, hyppar: list, radius: list, mean: list,
                loss_quantile: int=0, number_of_iterations: int=100):
    """
    Specification of the initialization method used for finding an initial value
    for the hyperparameter values when the parameteric_prior method is used.

    Parameters
    ----------
    method : str
        name of initialization method.
    loss_quantile : int,
        quantile indicating which loss value should be used for selecting the 
        initial hyperparameters. The default is 0, thus the minimum loss.
    number_of_iterations : int
        How many samples should be drawn from the initialization distribution.
        The default is 100.

    Returns
    -------
    init_dict : dict
        dictionary specifying the initialization method.

    """
    init_dict = dict(
        method=method,
        hyppar=hyppar,
        radius=radius,
        mean=mean,
        loss_quantile=loss_quantile,
        number_of_iterations=number_of_iterations
        )

    return init_dict


def train(method: str, sim_id: str, seed: int, epochs: int, B: int=128,
          samples_from_prior: int=200, output_path: str="results",
          progress_info: int=1, view_ep: int=1, save_log: bool=False):
    """
    Specification of training settings for learning the prior distribution(s).

    Parameters
    ----------
    method : str
        Method for learning the prior distribution. Available is either 
        ``parametric_prior`` for learning independent parametric priors
        or ``deep_prior`` for learning a joint non-parameteric prior.
    sim_id : str
        provides model a unique identifier used for saving results.
    seed : int
        seed used for learning.
    epochs : int
        number of iterations until training is stopped.
    B : int, optional
        batch size. The default is 128.
    samples_from_prior : int, optional
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
        sim_id=sim_id,
        seed=seed,
        B=B,
        samples_from_prior=samples_from_prior,
        epochs=epochs,
        output_path=output_path,
        progress_info=progress_info,
        view_ep=view_ep,
        save_log=save_log
    )
    return train_dict


def nf(inference_network: callable, network_specs: dict,
       base_distribution: callable):
    """
    specification of the normalizing flow used from BayesFlow library

    Parameters
    ----------
    inference_network : callable
        type of inference network as specified by bayesflow.inference_networks.
    network_specs : dict
        specification of normalizing flow architecture. Arguments are inherited
        from chosen bayesflow.inference_networks.
    base_distribution : callable
        Base distribution from which should be sampled during learning.
        Normally the base distribution is a multivariate normal.
    input_dim : int
        number of model parameters.

    Returns
    -------
    nf_dict : dict
        dictionary specifying the normalizing flow settings.

    """
    nf_dict = dict(
        inference_network=inference_network,
        network_specs=network_specs,
        base_distribution=base_distribution
    )

    return nf_dict


def prior_elicitation(
    generative_model: callable,
    model_parameters: list,
    target_quantities: list,
    expert_data: callable,
    training_settings: callable,
    optimization_settings: callable,
    normalizing_flow: callable or None=None,
    initialization_settings: callable or None=None,
):
    """
    

    Parameters
    ----------
    generative_model : callable
        specification of generative model using
        :func:`elicit.prior_elicitation.generator`.
    model_parameters : list
        list of model parameters specified with
        :func:`elicit.prior_elicitation.par`.
    target_quantities : list
        list of target quantities specified with
        :func:`elicit.prior_elicitation.tar`.
    expert_data : callable
        specification of input data from expert or oracle using
        :func:`elicit.prior_elicitation.expert_input`
    training_settings : callable
        specification of training settings for learning prior distribution(s)
        using :func:`elicit.prior_elicitation.train`
    optimization_settings : callable
        specification of optimizer using
        :func:`elicit.prior_elicitation.optimizer`.
    normalizing_flow : callable or None
        specification of normalizing flow using :func:`elicit.prior_elicitation.nf`.
        Only required for ``deep_prior`` method is used. If 
        ``parametric_prior`` is used this argument should be ``None``. Default
        value is None.
    initialization_settings : callable
        specification of initialization settings using
        :func:`elicit.prior_elicitation.initializer`. For method
        'parametric_prior' the argument should be None. Default value is None.

    Returns
    -------
    global_dict : dict
        specification of all settings to run the optimization procedure.

    """
    global_dict = dict(
        generative_model=generative_model,
        model_parameters=model_parameters,
        target_quantities=target_quantities,
        expert_data=expert_data,
        training_settings=training_settings,
        optimization_settings=optimization_settings,
        normalizing_flow=normalizing_flow,
        initialization_settings=initialization_settings,
        )

    return global_dict