# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow_probability as tfp
import pytest
import pandas as pd

from elicit.prior_simulation import intialize_priors
from os import listdir

tfd = tfp.distributions


@pytest.fixture
def global_dict():
    return dict(
        training_settings=dict(
            method="parametric_prior",
            sim_id="toy_example",
            seed=2,
            epochs=2,
            output_path="results",
        ),
        model_parameters=dict(
            mu=dict(
                family=tfd.Normal,
                hyperparams_dict={
                    "mu_loc": tfd.Uniform(0.0, 1.0),
                    "mu_scale": tfd.Uniform(0.0, 1.0),
                },
                param_scaling=1.0,
            ),
            sigma=dict(
                family=tfd.HalfNormal,
                hyperparams_dict={"sigma_scale": tfd.Uniform(1.0, 50.0)},
                param_scaling=1.0,
            ),
            independence=None,
        ),
    )


# initialize_priors
def test_intialize_priors_counts(global_dict):
    res = intialize_priors(global_dict)
    parameters = set(global_dict["model_parameters"].keys()).difference(
        {"independence"}
    )

    # correct number of parameters
    num_param = len(parameters)
    assert num_param == len(res)


def test_intialize_priors_labels(global_dict):
    res = intialize_priors(global_dict)
    parameters = sorted(
        list(
            set(
                global_dict["model_parameters"].keys()
                ).difference({"independence"})
            )
    )

    # correct labelling of hyperparameters
    expected_hyperparam = []
    observed_hyperparam = []
    for i, param in enumerate(parameters):
        expected_hyperparam.append(
            global_dict["model_parameters"][param]["hyperparams_dict"].keys()
        )
        observed_hyperparam.append(res[i].keys())

    assert expected_hyperparam == observed_hyperparam


def test_initialize_priors_saving(global_dict):
    expected_path = global_dict["training_settings"]["output_path"]

    # file is saved correctly
    assert "init_prior.pkl" in listdir(expected_path)


def test_initialize_priors_file(global_dict):

    # saved file is equivalent to output results
    expected_data = intialize_priors(global_dict)
    observed_file = pd.read_pickle(
        global_dict["training_settings"]["output_path"] + "/init_prior.pkl"
    )

    assert expected_data == observed_file


# sample_from_priors
@pytest.fixture
def global_dict2():
    return dict(
        training_settings=dict(
            method="parametric_prior",
            sim_id="toy_example",
            seed=2,
            epochs=2,
            output_path="results",
            samples_from_prior=200,
            B=100
        ),
        model_parameters=dict(
            mu=dict(
                family=tfd.Normal,
                hyperparams_dict={
                    "mu_loc": tfd.Uniform(0.0, 1.0),
                    "mu_scale": tfd.Uniform(0.0, 1.0),
                },
                param_scaling=1.0,
            ),
            sigma=dict(
                family=tfd.HalfNormal,
                hyperparams_dict={"sigma_scale": tfd.Uniform(1.0, 50.0)},
                param_scaling=1.0,
            ),
            independence=None,
        ),
        expert_data=dict(
            from_ground_truth=True,
            simulator_specs={
                "mu": tfd.Normal(loc=17, scale=2),
                "sigma": tfd.Gamma(2, 5),
            },
            samples_from_prior=200,
        )
    )


@pytest.fixture
def intialized_prior():
    return [
        dict(
            mu_loc=1.,
            mu_scale=0.2
        ),
        dict(
            sigma_scale=0.5
        )
    ]
