# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow_probability as tfp
import tensorflow as tf
import elicit as el
import sys

from elicit.user.generative_models import NormalModelSimple
from elicit.user.design_matrices import design_matrix

tfd = tfp.distributions


def sim_func(seed, init_method, init_loss, init_iters):
    eliobj = el.Elicit(
        model=el.model(
            obj=NormalModelSimple,
            design_matrix=design_matrix(N=50, quantiles=[25,75]),
            sigma=1.
            ),
        parameters=[
            el.parameter(
                name="beta0",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper("mu0"),
                    scale=el.hyper("sigma0", lower=0)
                    )
            ),
            el.parameter(
                name="beta1",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper("mu1"),
                    scale=el.hyper("sigma1", lower=0)
                    )
            )
        ],
        target_quantities=[
            el.target(
                name="X0",
                elicitation_method=el.eli_method.quantiles((5, 25, 50, 75, 95)),
                loss=el.MMD_energy,
                loss_weight=1.0
            ),
            el.target(
                name="X1",
                elicitation_method=el.eli_method.quantiles((5, 25, 50, 75, 95)),
                loss=el.MMD_energy,
                loss_weight=1.0
            )
        ],
        expert=el.expert.simulate(
            ground_truth = {
                "b0": tfd.Normal(0.5, 0.7),
                "b1": tfd.Normal(-0.5, 1.3),
            },
            num_samples = 10_000
        ),
        optimization_settings=el.optimizer(
            optimizer=tf.keras.optimizers.Adam,
            learning_rate=0.05,
            clipnorm=1.0
            ),
        training_settings=el.train(
            method="parametric_prior",
            name="simple"+"_"+init_method+"_"+str(init_loss)+"_"+str(init_iters),
            seed=seed,
            epochs=400,
            progress_info=0
        ),
        initialization_settings=el.initializer(
            method=init_method,
            loss_quantile=init_loss,
            iterations=init_iters,
            specs=el.init_specs(
                radius=2,
                mean=0
                )
            )
    )

    res_ep, res = eliobj.train(save_file="results")


if __name__ == "__main__":
    seed = int(sys.argv[1])
    init_method = str(sys.argv[2])
    init_loss = int(sys.argv[3])
    init_iters = int(sys.argv[4])

    sim_func(seed, init_method, init_loss, init_iters)

