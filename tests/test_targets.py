# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import elicit as el
import pytest

tfd = tfp.distributions

#%% elicitation technique: quantiles
# constants
B, num_samples, num_obs = (1_000, 10_000, 3)

@pytest.fixture
def targets1():
    return [
        el.target(
            name="tar",
            query=el.queries.quantiles((.05, .25, .50, .75, .95))
        )
    ]

@pytest.fixture
def target_quantities1():
    return dict(
        tar=tfd.Normal(0.,1.).sample((B,num_samples))
        )

def test_computation_elicited_statistics_quantiles(
        targets1, target_quantities1):
    # compute elicited statistics
    res = el.targets.computation_elicited_statistics(
        target_quantities1, targets1)

    # check correct naming of target quantities
    assert "quantiles_tar" == list(res.keys())[0]

    # check correct values of elicited statistics
    expected_quants_tar1 = tfd.Normal(0.,1.).quantile(
        (.05, .25, .50, .75, .95))

    for i, q in enumerate(tf.reduce_mean(res["quantiles_tar"],0)):
        assert q.numpy() == pytest.approx(expected_quants_tar1[i], abs=0.01)

#%% elicitation technique: correlation

# constants
cor=0.3

@pytest.fixture
def targets2():
    return [
        el.target(
            name="tar",
            query=el.queries.correlation()
        )
    ]

@pytest.fixture
def target_quantities2():
    return dict(
        tar=tfd.MultivariateNormalFullCovariance (
            loc=[0., 0.],
            covariance_matrix=[[1., cor],[cor, 1.]]
            ).sample((B,num_samples))
        )

def test_computation_elicited_statistics_cor(targets2, target_quantities2):
    # compute elicited statistics
    res = el.targets.computation_elicited_statistics(
        target_quantities2, targets2)

    # check correct naming of target quantities
    assert "cor_tar" == list(res.keys())[0]

    # check correct values of elicited statistics
    mean_cor = tf.reduce_mean(res["cor_tar"]).numpy()
    assert cor == pytest.approx(mean_cor, abs=0.01)


#%% elicitation technique: identity
@pytest.fixture
def targets3():
    return [
        el.target(
            name="tar",
            query=el.queries.identity()
        )
    ]

@pytest.fixture
def target_quantities3():
    return dict(
        tar=tfd.Normal(0.,1.).sample((B,num_samples))
        )

def test_computation_elicited_statistics_identity(
        targets3, target_quantities3):
    # compute elicited statistics
    res = el.targets.computation_elicited_statistics(
        target_quantities3, targets3)

    # check correct naming of target quantities
    assert "identity_tar" == list(res.keys())[0]

    # check correct values of elicited statistics
    assert tf.reduce_all(target_quantities3["tar"] == res["identity_tar"])


#%% target quantity: correlation
@pytest.fixture
def prior_samples():
    return tfd.MultivariateNormalFullCovariance(
        loc=[0., 0.],
        covariance_matrix=[[1., 0.3],[0.3, 1.]]
        ).sample((B,num_samples))

def test_computation_target_quantities_cor(prior_samples, targets2):
    res = el.targets.computation_target_quantities(
        model_simulations=None, prior_samples=prior_samples,
        targets=targets2)

    assert tf.reduce_all(res["tar"] == prior_samples)


#%% target quantity: custom target quantity
@pytest.fixture
def custom():
    pass

@pytest.fixture
def targets4():
    return [
        el.target(
            name="arbitrary",
            query=el.queries.identity(),
            target_method=custom
            )
        ]

def test_computation_target_quantities_custom(targets4):
    with pytest.raises(NotImplementedError):
        el.targets.computation_target_quantities(
            model_simulations=None, prior_samples=None,
            targets=targets4)


#%% target quantities from model simulations
@pytest.fixture
def targets5():
    return [
        el.target(
            name="ypred",
            query=el.queries.identity()
        )
    ]

@pytest.fixture
def model_simulations():
    return dict(
    ypred = tfd.Normal(0.,1.).sample((B,num_samples,num_obs)),
    epred = tfd.Normal(1.,3.).sample((B,num_samples,num_obs))
    )

def test_computation_target_quantities(targets5, model_simulations):
        model_sim = el.targets.computation_target_quantities(
            model_simulations, prior_samples=None,
            targets=targets5)
        # check that correct model simulation has been selected
        assert tf.reduce_all(model_simulations["ypred"] == model_sim["ypred"])

