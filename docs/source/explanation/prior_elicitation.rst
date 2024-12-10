.. _prior_elicitation_explanation:

Input arguments of ``prior_elicitation``
########################################

:octicon:`move-to-bottom;1em;sd-text-info` Jump to the :ref:`prior_elicitation_func`

.. _model_parameters:

Model parameters
****************

**Notes:**

+ ``{(hyper)parameter_name}`` indicates a placeholder; any (hyper)parameter name can be used as key
+ ``<callable_function>`` indicates a placeholder; any distribution function from which samples can be drawn is allowed. It is expected that the callable has a ``sample()`` method corresponding to ``tfp.distribution`` objects. For the argument ``family`` it is expected that the distribution considers any transformation of hyperparameters into the unconstrained space <link to a tutorial>.

.. tab-set::
    :sync-group: category

    .. tab-item:: parametric_prior
        :sync: key1

        .. code-block:: python 

            model_parameters=dict(
                {parameter_name}=dict(
                    param_scaling=1.0, 
                    family=<callable_function>,
                    hyperparams_dict=dict(
                        {hyperparameter_name}=<callable_function>
                        )
                    ),
                independence=None,
                )

    .. tab-item:: deep_prior
        :sync: key2

        .. code-block:: python 

            model_parameters=dict(
                {parameter_name}=dict(
                    param_scaling=1.0
                    ),
                independence=dict(
                    corr_scaling=0.1
                    ),
                )

Normalizing flow
****************

**Notes:**

+ only for ``deep_prior`` method
+ if ``normalizing_flow=True`` the default architecture (as shown below) is used
+ architecture is heritated from <bayesflow.module> see for details and explanations <here>
+ the variable ``num_params`` is created automatically and does not need to be specified

.. tab-set::
    :sync-group: category

    .. tab-item:: parametric_prior
        :sync: key1

        .. code-block:: python 

            normalizing_flow=False

    .. tab-item:: deep_prior
        :sync: key2

        .. code-block:: python 

            normalizing_flow=dict(
                coupling_flow=dict(
                    num_coupling_layers=3,
                    coupling_design="affine",
                    coupling_settings={
                        "dropout": False,
                        "dense_args": {
                            "units": 128,
                            "activation": "relu",
                            "kernel_regularizer": None,
                        },
                        "num_dense": 2,
                    },
                    permutation="fixed",
                ),
                base_distribution=tfd.MultivariateNormalTriL(
                    loc=tf.zeros(num_params),
                    scale_tril=tf.linalg.cholesky(tf.eye(num_params))
                ),
            )

Expert data
************

**Notes:**

+ identical for ``parametric_prior`` and ``deep_prior`` method
+ two approaches: (1) simulating from oracle and (2) including elicited expert data (see for details :ref:`expert_data_howto`)
+ ``{parameter_name}`` indicates a placeholder and should match specififcation made in :ref:`model_parameters` (check this for joint priors...)
+ ``<callable_function>`` refers to the true prior distribution; any distribution function from which samples can be drawn is allowed. It is expected that the callable has a ``sample()`` method corresponding to ``tfp.distribution`` objects.
+ ``{expert_data}`` indicates a placeholder for the elicited expert data (see for details :ref:`expert_data_howto`)

.. code-block:: python

        # simulating from oracle
        expert_data=dict(
            data=None,
            from_ground_truth=True,
            simulator_specs = dict(
                {parameter_name}=<callable_function>
                ),
            samples_from_prior = 10_000
        ),

        # include elicited expert data
        expert_data=dict(
            data={expert_data},
            from_ground_truth=False
        )

Generative model
****************

**Notes:**

+ identical for ``parametric_prior`` and ``deep_prior`` method
+ ``callable_class`` specification of the generative model as a class object (see for details :ref:`generative_model_howto`)
+ if the ``generative_model`` class has additional arguments besides the internally required arguments ``ground_truth``, ``prior_samples``, they have to be specified in the argument ``additional_model_args``, otherwise ``additional_model_args=None``,
    + *keys* refer the argument names (here: {argument_name})
    + *values* refer to the argument values (here: {argument_value})

.. code-block:: python

    generative_model=dict(
        model=<callable_class>, 
        additional_model_args=dict(
            {argument_name}={argument_value}
        )
    )

Target quantities & elicitation techniques
******************************************

**Notes:**

+ ``{target_quantity}`` indicates a placeholder for the name of the target quantity (see for details :ref:`target_quantities_howto`)
+ ToDo
+ restructure input: Allow only for "quantiles" and "identity"
+ remove the loss_components argument

.. code-block:: python

    target_quantities=dict(
        {target_quantity}=dict(
            elicitation_method="quantiles",  # or "identity"
            quantiles_specs=(5, 25, 50, 75, 95),  # if elicitation_method="quantiles"
            custom_elicitation_function=None,
            custom_target_function=None,
        )
    )

Loss settings
*************

**Notes:**

+ ToDo

.. code-block:: python

    loss_settings = dict(
        loss=<callable_class>,  # default is MMD_energy imported from elicit.loss_functions
        loss_weighting=None, 
        use_regularization=False
    )

Optimization settings
*********************

**Notes:**

+ default ``optimizer`` is Adam (a list of other optimizers can be found in `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_
+ additional arguments for the optimizer need to be specified in ``optimizer_specs``
    + ``learning_rate`` can be fixed or a callable from `tf.keras.optimizers.schedules <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules>`_

.. code-block:: python

    optimization_settings=dict(
        optimizer=tf.keras.optimizers.Adam,
        optimizer_specs=dict(
            learning_rate=0.1,  # or callable from tf.keras.optimizers.schedules
            clipnorm=1.0
        )
    )

Initialization settings
***********************

**Notes:**

+ ToDo

.. code-block:: python

    initialization = dict(
        method="random",  # or "sobol" , "lhs"
        loss_quantile=0,
        number_of_iterations=200
    )

Training settings
*****************

**Notes:**

+ ToDo

.. tab-set::
    :sync-group: category

    .. tab-item:: parametric_prior
        :sync: key1

        .. code-block:: python 

            training_settings=dict(
                method="parametric_prior",
                sim_id="toy_example",
                seed=0,
                B=128,
                samples_from_prior=200,
                epochs=500,
                output_path="results",
                progress_info=1,
                view_ep=1,
                save_log=False
            )

    .. tab-item:: deep_prior
        :sync: key2

        .. code-block:: python 

            training_settings=dict(
                method="deep_prior",
                sim_id="toy_example",
                seed=0,
                B=128,
                samples_from_prior=200,
                epochs=500,
                output_path="results",
                progress_info=1,
                view_ep=1,
                save_log=False
            )

.. _prior_elicitation_func:

Full ``prior_elicitation`` function
***********************************

.. tab-set::
    :sync-group: category

    .. tab-item:: parametric_prior
        :sync: key1

        .. code-block:: python 

            prior_elicitation(
                model_parameters=dict(
                    {parameter_name}=dict(
                        param_scaling=1.0, 
                        family=<callable_function>,
                        hyperparams_dict=dict(
                            {hyperparameter_name}=<callable_function>
                        )
                    ),
                    independence=None,
                ),
                normalizing_flow=False,
                expert_data=dict(
                    data=None,
                    from_ground_truth=True,  # or False
                    simulator_specs = dict(
                        {parameter_name}=<callable_function>
                        ),
                    samples_from_prior = 10_000
                ),
                generative_model=dict(
                    model=<callable_class>, 
                    additional_model_args=dict(
                        {argument_name}={argument_value}
                    )
                ),
                target_quantities=dict(
                    {target_quantity}=dict(
                        elicitation_method="quantiles",  # or "identity"
                        quantiles_specs=(5, 25, 50, 75, 95),  # if elicitation_method="quantiles"
                        custom_elicitation_function=None,
                        custom_target_function=None,
                    )
                ),
                loss_settings = dict(
                    loss=<callable_class>,  # default is MMD_energy imported from elicit.loss_functions
                    loss_weighting=None, 
                    use_regularization=False
                ),
                optimization_settings=dict(
                    optimizer=tf.keras.optimizers.Adam,
                    optimizer_specs=dict(
                        learning_rate=0.1,  # or callable from tf.keras.optimizers.schedules
                        clipnorm=1.0
                    )
                ),
                initialization = dict(
                    method="random",  # or "sobol" , "lhs"
                    loss_quantile=0,
                    number_of_iterations=200
                ),
                training_settings=dict(
                    method="parametric_prior",
                    sim_id="toy_example",
                    seed=0,
                    B=128,
                    samples_from_prior=200,
                    epochs=500,
                    output_path="results",
                    progress_info=1,
                    view_ep=1,
                    save_log=False
                )
            )
            

    .. tab-item:: deep_prior
        :sync: key2

        .. code-block:: python 

            prior_elicitation(
                model_parameters=dict(
                    {parameter_name}=dict(
                        param_scaling=1.0
                    ),
                    independence=dict(
                        corr_scaling=0.1
                    ),
                ),
                normalizing_flow=dict(
                    coupling_flow=dict(
                        num_coupling_layers=3,
                        coupling_design="affine",
                        coupling_settings={
                            "dropout": False,
                            "dense_args": {
                                "units": 128,
                                "activation": "relu",
                                "kernel_regularizer": None,
                            },
                            "num_dense": 2,
                        },
                        permutation="fixed",
                    ),
                    base_distribution=tfd.MultivariateNormalTriL(
                        loc=tf.zeros(num_params),
                        scale_tril=tf.linalg.cholesky(tf.eye(num_params))
                    ),
                ),
                expert_data=dict(
                    data=None,
                    from_ground_truth=True,
                    simulator_specs = dict(
                        {parameter_name}=<callable_function>
                        ),
                    samples_from_prior = 10_000
                ),
                generative_model=dict(
                    model=<callable_class>, 
                    additional_model_args=dict(
                        {argument_name}={argument_value}
                    )
                ),
                target_quantities=dict(
                    {target_quantity}=dict(
                        elicitation_method="quantiles",  # or "identity"
                        quantiles_specs=(5, 25, 50, 75, 95),  # if elicitation_method="quantiles"
                        custom_elicitation_function=None,
                        custom_target_function=None,
                    )
                ),
                loss_settings = dict(
                    loss=<callable_class>,  # default is MMD_energy imported from elicit.loss_functions
                    loss_weighting=None, 
                    use_regularization=False
                ),
                optimization_settings=dict(
                    optimizer=tf.keras.optimizers.Adam,
                    optimizer_specs=dict(
                        learning_rate=0.1,  # or callable from tf.keras.optimizers.schedules
                        clipnorm=1.0
                    )
                ),
                initialization = dict(
                    method="random",  # or "sobol" , "lhs"
                    loss_quantile=0,
                    number_of_iterations=200
                ),
                training_settings=dict(
                    method="deep_prior",
                    sim_id="toy_example",
                    seed=0,
                    B=128,
                    samples_from_prior=200,
                    epochs=500,
                    output_path="results",
                    progress_info=1,
                    view_ep=1,
                    save_log=False
                )
            )