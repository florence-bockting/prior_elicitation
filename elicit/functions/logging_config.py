import logging.config
from pythonjsonlogger import jsonlogger

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
        }
    },
    "handlers": {
        # "stdout": {
        #     "class": "logging.StreamHandler",
        #     "stream": "ext://sys.stdout",
        #     "formatter": "json",
        # },
        "json_file": {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'logs.json',
         #   'filemode': 'w',
            'formatter': 'json',
            }
    },
    "loggers": {"": {"handlers": ["json_file"], "level": "INFO"}},
}


logging.config.dictConfig(LOGGING)

