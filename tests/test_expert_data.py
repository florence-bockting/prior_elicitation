import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import pytest

from elicit.main import one_forward_simulation
from elicit.expert_data import get_expert_data

tfd = tfp.distributions


class TestModel:
    def __call__(self, ground_truth, prior_samples):
        mu = prior_samples[:, :, 0]
        sigma = prior_samples[:, :, 1]
        # data-generating model
        likelihood = tfd.Normal(loc=mu, scale=sigma)
        # prior predictive distribution (=height)
        ypred = likelihood.sample()

        return dict(
            likelihood=likelihood,
            ypred=ypred,
            epred=None,
            mu=mu,
            sigma=sigma,
            prior_samples=prior_samples,
        )


global_dict = dict(
    initialization_settings=dict(
        method="random", loss_quantile=0, number_of_iterations=2
    ),
    training_settings=dict(
        method="parametric_prior",
        sim_id="test_expert_data",
        seed=1,
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
        no_params=2,
    ),
    generative_model=dict(
        model=TestModel, additional_model_args=None, discrete_likelihood=False
    ),
    target_quantities=dict(
        mu=dict(
            elicitation_method="identity",
            loss_components="all",
            custom_target_function=None,
            custom_elicitation_method=None,
        ),
        sigma=dict(
            elicitation_method="identity",
            loss_components="all",
            custom_target_function=None,
            custom_elicitation_method=None,
        ),
    ),
    expert_data=dict(
        from_ground_truth=True,
        simulator_specs={
            "mu": tfd.Normal(loc=5.0, scale=0.8),
            "sigma": tfd.HalfNormal(1.5),
        },
        samples_from_prior=10_000,
    ),
)

normalizing_flow = dict(
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
        loc=tf.zeros(2), scale_tril=tf.linalg.cholesky(tf.eye(2))
    ),
)


def comp_mean_sd(x):
    mean = tf.math.reduce_mean(x)
    sd = tf.math.reduce_std(x)
    return (mean, sd)


# %% Test correct output shape, when oracle is used

test_data1 = [
    (
        global_dict,
        "deep_prior",
        dict(corr_scaling=0.1),
        normalizing_flow,
        (1, 10_000, 1),
    ),
    (global_dict, "parametric_prior", None, None, (1, 10_000, 1)),
]


@pytest.mark.parametrize(
    "global_dict, method, independence, normalizing_flow, expected_shape",  # noqa
    test_data1,
)
def test_expert_data_shape(
    global_dict, method, independence, normalizing_flow, expected_shape
):
    global_dict["normalizing_flow"] = normalizing_flow
    global_dict["training_settings"]["independence"] = independence
    global_dict["training_settings"]["method"] = method

    dat = get_expert_data(global_dict, one_forward_simulation,
                          path_to_expert_data=None)

    # expected shape
    dat["identity_mu"].shape == expected_shape
    dat["identity_sigma"].shape == expected_shape


# %% Test whether true priors correspond to oracle specification

test_data2 = [
    (
        global_dict,
        "deep_prior",
        dict(corr_scaling=0.1),
        normalizing_flow,
        tfd.Normal(loc=5.0, scale=0.8),
        tfd.HalfNormal(1.5),
    ),
    (
        global_dict,
        "parametric_prior",
        None,
        None,
        tfd.Normal(loc=5.0, scale=0.8),
        tfd.HalfNormal(1.5),
    ),
]


@pytest.mark.parametrize(
    "global_dict, method, independence, normalizing_flow, true_mu_func, true_sigma_func",  # noqa
    test_data2,
)
def test_expert_data_numeric(
    global_dict, method, independence, normalizing_flow, true_mu_func,
    true_sigma_func
):
    global_dict["normalizing_flow"] = normalizing_flow
    global_dict["training_settings"]["independence"] = independence
    global_dict["training_settings"]["method"] = method

    dat = get_expert_data(global_dict, one_forward_simulation,
                          path_to_expert_data=None)

    # sample from true distribution
    true_mu = true_mu_func.sample(10_000)
    true_sigma = true_sigma_func.sample(10_000)

    # mu: expected and observed mean and sd
    mu_exp_m, mu_exp_sd = comp_mean_sd(true_mu)
    mu_obs_m, mu_obs_sd = comp_mean_sd(dat["identity_mu"])
    mu_exp_m.numpy() == pytest.approx(mu_obs_m.numpy(), rel=0.05)
    mu_exp_sd.numpy() == pytest.approx(mu_obs_sd.numpy(), rel=0.05)

    # sigma: expected and observed mean and sd
    sigma_exp_m, sigma_exp_sd = comp_mean_sd(true_sigma)
    sigma_obs_m, sigma_obs_sd = comp_mean_sd(dat["identity_sigma"])
    sigma_exp_m.numpy() == pytest.approx(sigma_obs_m.numpy(), rel=0.05)
    sigma_exp_sd.numpy() == pytest.approx(sigma_obs_sd.numpy(), rel=0.05)


# %% Test correct saving location

test_data3 = [
    (global_dict, "deep_prior", dict(corr_scaling=0.1), normalizing_flow),
    (global_dict, "parametric_prior", None, None),
]


@pytest.mark.parametrize(
    "global_dict, method, independence, normalizing_flow", test_data3
)
def test_expert_data_saving(global_dict, method, independence,
                            normalizing_flow):
    global_dict["normalizing_flow"] = normalizing_flow
    global_dict["training_settings"]["independence"] = independence
    global_dict["training_settings"]["method"] = method

    dat = get_expert_data(global_dict, one_forward_simulation,
                          path_to_expert_data=None)

    # read data from file
    elicited_stats = pd.read_pickle(
        global_dict["training_settings"]["output_path"]
        + "/expert/elicited_statistics.pkl"
    )

    tf.reduce_all(tf.equal(dat["identity_mu"],
                           elicited_stats["identity_mu"]))
    tf.reduce_all(tf.equal(dat["identity_sigma"],
                           elicited_stats["identity_sigma"]))
