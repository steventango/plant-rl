import logging
import os


def log_config(cfg):
    def get_print_attrs(cfg):
        attrs = dict(cfg.__dict__)
        for k in ["logger"]:
            del attrs[k]
        return attrs

    attrs = get_print_attrs(cfg)
    for param, value in attrs.items():
        cfg.logger.info("{}: {}".format(param, value))


class Logger:
    def __init__(self, config, log_dir):
        log_file = os.path.join(log_dir, "log")
        self._logger = logging.getLogger()

        file_handler = logging.FileHandler(log_file, mode="w")
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        self._logger.setLevel(level=logging.INFO)

        self.config = config

    def info(self, log_msg):
        self._logger.info(log_msg)
