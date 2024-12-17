import tensorflow_probability as tfp
import pandas as pd
import seaborn as sns
import tensorflow as tf
import elicit.prior_elicitation as pe

from bayesflow.inference_networks import InvertibleNetwork
from elicit.user.generative_models import ToyModel
from elicit.loss_functions import MMD_energy, L2
from elicit.main import run

tfd = tfp.distributions


global_dict = pe.prior_elicitation(
    generative_model=pe.generator(
        model=ToyModel,
        additional_model_args=dict(
            N=200
            )
    ),
    model_parameters=[
        pe.par(name="mu"),
        pe.par(name="sigma")
    ],
    target_quantities=[
        pe.tar(
            name="ypred",
            elicitation_method="quantiles",
            quantiles_specs=(5, 25, 50, 75, 95),
            loss=MMD_energy,
            loss_weight=1.0
        ),
        pe.tar(
            name="correlation",
            elicitation_method="pearson_correlation",
            loss=L2,
            loss_weight=1.0
        )
    ],
    expert_data=pe.expert_input(
        data=None,
        from_ground_truth=True,
        simulator_specs = {
            "mu": tfd.Normal(loc=5, scale=2),
            "sigma": tfd.HalfNormal(10.),
        },
        samples_from_prior = 10_000
    ),
    optimization_settings=pe.optimizer(
        optimizer_specs=dict(
            learning_rate=0.001,
            clipnorm=1.0
            )
    ),
    normalizing_flow=pe.nf(
        inference_network=InvertibleNetwork,
        network_specs=dict(
            num_params=2,
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
            permutation="fixed"
        ),
        base_distribution=tfd.MultivariateNormalDiag(
            loc=tf.zeros(2),
            scale_diag=tf.ones(2)
        )
    ),
    training_settings=pe.train(
        method="deep_prior",
        sim_id="toy_example",
        seed=2,
        epochs=400
    ),
    initialization_settings=None
)


run(global_dict)

path = "elicit/results/deep_prior/toy_example_2"

prior = pd.read_pickle(path + "/model_simulations.pkl")["prior_samples"]
prior2 = pd.read_pickle(path + "/expert/model_simulations.pkl")["prior_samples"]


sns.kdeplot(prior2[0,:,0])
sns.kdeplot(tf.reshape(prior, (128*200,2))[:,0])

sns.kdeplot(prior2[0,:,1])
sns.kdeplot(tf.abs(tf.reshape(prior, (128*200,2))[:,1]))
