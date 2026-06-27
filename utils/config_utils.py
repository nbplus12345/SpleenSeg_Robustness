import argparse
import os

import yaml


class Map(dict):
    """Dictionary wrapper that supports attribute-style access.

    Nested dictionaries and lists are recursively wrapped so existing training
    code can use config.train.lr while the source file remains plain YAML.
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for key, value in arg.items():
                    self[key] = self._wrap(value)
        if kwargs:
            for key, value in kwargs.items():
                self[key] = self._wrap(value)

    def _wrap(self, value):
        """Recursively wrap nested configuration values."""
        if isinstance(value, dict):
            return Map(value)
        if isinstance(value, list):
            return [self._wrap(item) for item in value]
        return value

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self[key] = value


def load_config(config_path="./config.yaml"):
    """Load YAML configuration into a Map object."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return Map(config_dict)


def get_args():
    """Parse the common --config argument used by training utilities."""
    parser = argparse.ArgumentParser(description="SpleenSeg_UNet command launcher")
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()
