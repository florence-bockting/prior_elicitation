# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0


save_results = dict(
    # all generated initializations during pre-training
    initialization_matrix=True,
    # tuple: loss values corresp. to each set of generated initial values
    pre_training_results=True,
    # initialized hyperparameter values
    init_hyperparameters=False,
    # prior samples of last epoch
    prior_samples=False,
)
