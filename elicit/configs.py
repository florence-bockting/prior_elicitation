# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0


def save_history(loss: bool=True, loss_component: bool=True,
                 hyperparameter: bool=True, hyperparameter_gradient: bool=True
                 ):
    save_hist_dict = dict(
        loss=loss,
        loss_component=loss_component,
        hyperparameter=hyperparameter,
        hyperparameter_gradient=hyperparameter_gradient,
    )
    return save_hist_dict


def save_results(target_quantities: bool=True, elicited_statistics: bool=True,
                 prior_samples: bool=True, model_samples: bool=True,
                 model: bool=True, expert_elicited_statistics: bool=True,
                 expert_prior_samples: bool=True, init_loss_list: bool=True,
                 init_prior: bool=True, init_matrix: bool=True,
                 loss_tensor_expert: bool=True, loss_tensor_model: bool=True):

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
        loss_tensor_model=loss_tensor_model
    )
    return save_res_dict