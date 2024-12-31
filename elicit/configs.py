# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

#%% configuration for logging information
import logging
import logging.config

LOGGING = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "json": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
        }
    },
    "handlers": {
        "json_file": {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'logs.json',
            'formatter': 'json',
            }
    },
    "loggers": {"": {"handlers": ["json_file"], "level": "INFO"}},
}


logging.config.dictConfig(LOGGING)

#%% configuration for saving results
save_results = dict(
    # all generated initializations during pre-training
    initialization_matrix=True,
    # tuple: loss values corresp. to each set of generated initial values
    pre_training_results=True,
    # initialized hyperparameter values
    init_hyperparameters=False,
    # prior samples of last epoch
    prior_samples=False,
    # elicited statistics of last epoch
    elicited_statistics=True,
)
