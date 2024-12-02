# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
from elicit.functions.loss_functions import MMD_energy


def global_dict(
    model_parameters=dict(
        param1=dict(
            family=None,
            hyperparams_dict=dict(
                hyppar1=None
                ),
            param_scaling=1.
            ),
        independence=True
        ),
    expert_data=dict(
         data=None,
         from_ground_truth=True,
         simulator_specs=dict(
             param1=None
             ),
         samples_from_prior=10000
         ),
    generative_model=dict(
         model=None,
         additional_model_args=dict(
             model_arg1=None
             ),
         discrete_likelihood=False,
         softmax_gumble_specs=dict(
             temperature=1.,
             upper_threshold=None
             )
         ),
    target_quantities=dict(
         target1=dict(
             elicitation_method=None,
             custom_elicitation_method=dict(
                 function=None,
                 additional_args=dict(
                     func_arg1=None
                     )
                 ),
             custom_target_function=dict(
                 function=None,
                 additional_args=dict(
                     func_arg1=None
                     )
                 ),
             quantiles_specs=None,
             moments_specs=None,
             loss_components="all"
             )
         ),
    loss_function=dict(
        loss=None,
        loss_weighting=None,
        use_regularization=None,
        loss_scaling=dict(
            corr=0.1,
            targets=None
            )
        ),
    optimization_settings=dict(
        optimizer=tf.keras.optimizers.Adam,
        optimizer_specs=dict(
            learning_rate=None,
            clipnorm=1.0
            )
        ),
    initialization_settings=dict(
        method=None,
        loss_quantile=None,
        number_of_iterations=10
        ),
    training_settings=dict(
         method=None,
         sim_id=None,
         seed=None,
         view_ep=1,
         epochs=500,
         B=128,
         samples_from_prior=200,
         output_path="results",
         progress_info=1
         )):
    """
    User settings for prior elicitation tool

    Parameters
    ----------
    model_parameters : dict
        Description of model parameters in generative model with keys
        referring to the name of the model parameter and the value is a
        dictionary with additional model parameter specifications, which
        depend on the selected method (see below).

        With the additional argument 'independence' can be specified whether
        the model parameters are assumed to be independent or not. For the
        method "parametric_prior" we assume always independence, thus the
        value has to be true (and is by default true).
        For the method "deep_prior" the user can specify whether the model
        parameters are assumed to be independent (independence=true) or
        dependent (independence=false).

        Method="deep_prior":
            For the deep prior approach we assume a joint prior on all model
            parameters. Through the additional argument 'param_scaling' it is
            possible to scale the sample from the marginal prior by the
            specified value. By default the scaling value is 1., thus no
            scaling is applied.
            TODO: Do we really need the param_scaling argument?

            Code Example::

                mu=dict(
                    param_scaling=1.
                    )

        Method="parametric_prior":
            For the parametric approach an independent parametric prior
            distribution family is specified for each model parameter.

            Additionally, it is required to specify an initialization for each
            hyperparameter of the parametric distributions, according to which
            samples can be drawn for the first epoch.Initialization can be a
            distribution or a specific value.

            Through the additional argument 'param_scaling' it is
            possible to scale the sample from the marginal prior by the
            specified value. By default the scaling value is 1., thus no
            scaling is applied.
            TODO: Do we really need the param_scaling argument?

            Code Example::

                mu=dict(
                    family=tfd.Normal,
                    hyperparams_dict=dict(
                        loc_mu=tfd.Normal(0.,1.),
                        scale_mu=0.5
                        ),
                    param_scaling=1.
                    )

"""
    global_dict = dict(
        model_parameters=dict(
            param1=dict(
                family=None,
                hyperparams_dict=dict(
                    hyppar1=None
                    ),
                param_scaling=1.
                ),
            independence=None
            ),
        expert_data=dict(
            data=None,
            from_ground_truth=True,
            simulator_specs=dict(
                param1=None
                 ),
            samples_from_prior=10000
            ),
        generative_model=dict(
            model=None,
            additional_model_args=dict(
                model_arg1=None
                ),
            discrete_likelihood=False,
            softmax_gumble_specs=dict(
                temperature=1.,
                upper_threshold=None
                )
            ),
        target_quantities=dict(
            target1=dict(
                elicitation_method=None,
                custom_elicitation_method=dict(
                    function=None,
                    additional_args=dict(
                         func_arg1=None
                         )
                     ),
                custom_target_function=dict(
                    function=None,
                    additional_args=dict(
                        func_arg1=None
                        )
                    ),
                quantiles_specs=None,
                moments_specs=None,
                loss_components="all"
                )
            ),
        loss_function=dict(
            loss=MMD_energy,
            loss_weighting=dict(
                method=None,         # dwa, custom
                method_specs=dict(
                    temperature=1.6  # dwa-specific
                    ),
                weights=None
                ),
            use_regularization=None
            ),
        optimization_settings=dict(
            optimizer=tf.keras.optimizers.Adam,
            optimizer_specs=dict(
                learning_rate=None,
                clipnorm=1.0
                )
            ),
        initialization_settings=dict(
            method=None,
            loss_quantile=None,
            number_of_iterations=10
            ),
        training_settings=dict(
             method=None,
             sim_id=None,
             seed=None,
             view_ep=1,
             epochs=500,
             B=128,
             samples_from_prior=200,
             output_path="results",
             progress_info=1
             )
        )
    return global_dict


global_dict()
