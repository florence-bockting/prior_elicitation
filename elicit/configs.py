# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0


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
        total loss per epoch. The default is True.
    loss_component : bool, optional
        loss per loss-component per epoch. The default is True.
    time : bool, optional
        time in sec per epoch. The default is True.
    hyperparameter : bool, optional
        'parametric_prior' method: Trainable hyperparameters of parametric
        prior distributions.
        'deep_prior' method: Mean and standard deviation of each marginal
        from the joint prior.
        The default is True.
    hyperparameter_gradient : bool, optional
        Gradients of the hyperparameter. Only for 'parametric_prior' method.
        The default is True.

    Returns
    -------
    save_hist_dict : dict
        dictionary with inclusion/exclusion settings for each sub-result in
        history object.

    """
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
            simulation-based target quantities. The default is True.
        elicited_statistics : bool, optional
            simulation-based elicited statistics. The default is True.
        prior_samples : bool, optional
            samples from simulation-based prior distributions.
            The default is True.
        model_samples : bool, optional
            output variables from the simulation-based generative model.
            The default is True.
        model : bool, optional
            fitted elicit model object including the trainable variables.
            The default is True.
        expert_elicited_statistics : bool, optional
            expert-elicited statistics. The default is True.
        expert_prior_samples : bool, optional
            if oracle is used: samples from the true prior distribution,
            otherwise it is None. The default is True.
        init_loss_list : bool, optional
            initialization phase: Losses related to the samples drawn from the
            initialization distribution.
            Only included for method 'parametric_prior'.
            The default is True.
        init_prior : bool, optional
            initialized elicit model object including the trainable variables.
            Only included for method 'parametric_prior'.
            The default is True.
        init_matrix : bool, optional
            initialization phase: samples drawn from the initialization
            distribution for each hyperparameter.
            Only included for method 'parametric_prior'.
            The default is True.
        loss_tensor_expert : bool, optional
            expert term in loss component for computing the discrepancy.
            The default is True.
        loss_tensor_model : bool, optional
            simulation-based term in loss component for computing the
            discrepancy. The default is True.

        Returns
        -------
        save_res_dict : dict
            dictionary with inclusion/exclusion settings for each sub-result
            in results object.

        """
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
