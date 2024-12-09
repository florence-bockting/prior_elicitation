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
