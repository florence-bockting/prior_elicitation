# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow_probability as tfp
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf

from elicit.prior_simulation import intialize_priors, sample_from_priors

tfd = tfp.distributions


#@pytest.fixture
global_dict = dict(
        training_settings=dict(
            method="parametric_prior",
            sim_id="test_initialize_priors",
            seed=2,
            epochs=2,
            B=256,
            samples_from_prior=200,
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
            no_params=2
        ),
    )

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
        loc=tf.zeros(2),
        scale_tril=tf.linalg.cholesky(tf.eye(2))
    ),
)

#%% Test number of hyperparameters (i.e., trainable variables); both methods

# counter number of hyperparameters from initialize_prior output
# (if method is parametric_prior)
def count_hyperparams(res):
    counter=0
    for i in range(len(res)):
        counter += len(res[i])
    return counter

# number of trainable variables
# parametric_prior: number of prior hyperparameters
# deep_prior: number of weights in NN=num_coupling_layers*2 (bias & scale)
test_data1 = [
    (global_dict, "parametric_prior", None, None, 3),
    (global_dict, "deep_prior", dict(corr_scaling=0.1), normalizing_flow, 3*2)
    ]

# initialize_priors
@pytest.mark.parametrize(
    "global_dict, method, independence, normalizing_flow, expected_hyperparam",
    test_data1)
def test_intialize_priors_counts(global_dict, method, independence,
                                 normalizing_flow, expected_hyperparam):
    global_dict["normalizing_flow"] = normalizing_flow
    global_dict["training_settings"]["independence"] = independence
    global_dict["training_settings"]["method"] = method
    # get initialized hyperparameters (i.e., trainable variables)
    res = intialize_priors(global_dict)

    if method == "parametric_prior":
        assert expected_hyperparam == count_hyperparams(res)
    else:
        assert expected_hyperparam == len(res.trainable_variables)

#%% Test labels of hyperparameters; method=parametric_prior

test_data2 = [(global_dict, ["mu_loc", "mu_scale", "sigma_scale"])]

def get_labels(res):
    labels=[]
    for i in range(len(res)):
        labels.append(list(res[i].keys()))
    return list(np.concatenate(labels, axis=0))

# INFO: Only for method=parametric_prior
@pytest.mark.parametrize("global_dict, expected_labels", test_data2)
def test_intialize_priors_labels(global_dict, expected_labels):
    global_dict["training_settings"]["method"] = "parametric_prior"

    res = intialize_priors(global_dict)

    assert expected_labels == get_labels(res)

#%% Test saving path;, both methods

test_data3=[
    (global_dict, "deep_prior", dict(corr_scaling=0.1), normalizing_flow),
    (global_dict, "parametric_prior", None, None)
    ]

@pytest.mark.parametrize(
    "global_dict, method, independence, normalizing_flow", test_data3)
def test_initialize_priors_saving(
        global_dict, method, independence, normalizing_flow):
    global_dict["normalizing_flow"] = normalizing_flow
    global_dict["training_settings"]["independence"] = independence
    global_dict["training_settings"]["method"] = method
    # initialize and save hyperparameters
    expected_data=intialize_priors(global_dict)
    # read data from file
    observed_data=pd.read_pickle(
            global_dict["training_settings"][
                "output_path"] + "/init_hyperparameters.pkl"
        )
    # initialize and save hyperparameters
    if method == "parametric_prior":
        # check if file is saved correctly
        assert expected_data == observed_data
    else:
        expected=expected_data.trainable_variables
        # check if file is saved correctly
        assert tf.math.reduce_all(
            tf.equal(expected, observed_data)).numpy()

#%% Test shape of samples from prior distribution

global_dict.update(
    expert_data=dict(
        from_ground_truth=True,
        simulator_specs={
            "mu": tfd.Normal(loc=5., scale=0.8),
            "sigma": tfd.HalfNormal(1.5),
        },
        samples_from_prior=10_000,
    )
    )

test_data4 = [
    (global_dict, "deep_prior", dict(corr_scaling=0.1), normalizing_flow,
     True, (1, 10_000, 2)),
    (global_dict, "deep_prior", dict(corr_scaling=0.1), normalizing_flow,
     False, (256, 200, 2)),
    (global_dict, "parametric_prior", None, None, True, (1, 10_000, 2)),
    (global_dict, "parametric_prior", None, None, False, (256, 200, 2))
    ]

@pytest.mark.parametrize(
    "global_dict, method, independence, normalizing_flow, ground_truth, expected_shape",
    test_data4)
def test_sample_from_priors_shape(
        global_dict, method, independence, normalizing_flow, ground_truth,
        expected_shape):
    global_dict["normalizing_flow"] = normalizing_flow
    global_dict["training_settings"]["independence"] = independence
    global_dict["training_settings"]["method"] = method
    # initialize and save hyperparameters
    init_priors=intialize_priors(global_dict)
    # sample from init_priors
    prior_samples=sample_from_priors(init_priors, ground_truth, global_dict)
    # check whether samples have expected shape
    assert prior_samples.shape == expected_shape

#%% Test whether prior samples are identical for same seed

test_data5 = [
    (global_dict, "deep_prior", dict(corr_scaling=0.1), normalizing_flow),
    (global_dict, "parametric_prior", None, None)
    ]

@pytest.mark.parametrize(
    "global_dict, method, independence, normalizing_flow", test_data5)
def test_sample_from_priors_seed(
        global_dict, method, independence, normalizing_flow):
    global_dict["normalizing_flow"] = normalizing_flow
    global_dict["training_settings"]["independence"]=independence
    global_dict["training_settings"]["method"]=method

    # first run
    global_dict["training_settings"]["sim_id"]="test_initialize_priors_rep1"
    # initialize and save hyperparameters
    init_priors1=intialize_priors(global_dict)
    # sample from init_priors
    prior_samples1=sample_from_priors(init_priors1, False, global_dict)

    # second run (with same seed)
    global_dict["training_settings"]["sim_id"]="test_initialize_priors_rep2"
    # initialize and save hyperparameters
    init_priors2=intialize_priors(global_dict)
    # sample from init_priors
    prior_samples2=sample_from_priors(init_priors2, False, global_dict)

    # check whether samples are equal due to common seed
    assert tf.reduce_all(tf.equal(prior_samples1, prior_samples2)).numpy()

#%% Test whether prior samples correspond to given mean and std of oracle

prior1=tfd.Normal(loc=5., scale=0.8)
prior2=tfd.HalfNormal(1.5)

global_dict["expert_data"]=dict(
        from_ground_truth=True,
        simulator_specs={
            "mu": prior1,
            "sigma": prior2,
        },
        samples_from_prior=10_000,
    )

mu1=tf.reduce_mean(prior1.sample(10_000))
mu2=tf.reduce_mean(prior2.sample(10_000))
sd1=tf.math.reduce_std(prior1.sample(10_000))
sd2=tf.math.reduce_std(prior2.sample(10_000))

test_data6 = [
    (global_dict, "deep_prior", dict(corr_scaling=0.1), normalizing_flow,
     np.stack([mu1,mu2]), np.stack([sd1,sd2])),
    (global_dict, "parametric_prior", None, None, np.stack([mu1,mu2]),
     np.stack([sd1,sd2]))
    ]

@pytest.mark.parametrize(
    "global_dict, method, independence, normalizing_flow, expected_mus, expected_stds", test_data6)
def test_sample_from_priors_values(
        global_dict, method, independence, normalizing_flow, expected_mus,
        expected_stds):
    global_dict["normalizing_flow"] = normalizing_flow
    global_dict["training_settings"]["independence"]=independence
    global_dict["training_settings"]["method"]=method

    # initialize and save hyperparameters
    init_priors=intialize_priors(global_dict)
    # sample from init_priors
    prior_samples=sample_from_priors(init_priors, True, global_dict)
    # compute mean and std-dev of prior samples
    mus=tf.reduce_mean(prior_samples, (0,1))
    stds=tf.math.reduce_std(prior_samples, (0,1))

    assert expected_mus[0] == pytest.approx(mus[0], rel=0.01)
    assert expected_mus[1] == pytest.approx(mus[1], rel=0.01)
    assert expected_stds[0] == pytest.approx(stds[0], rel=0.01)
    assert expected_stds[1] == pytest.approx(stds[1], rel=0.01)