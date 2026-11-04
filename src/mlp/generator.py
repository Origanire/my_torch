"""my_torch_generator : crée des réseaux vierges à partir de fichiers de configuration."""

import os
import sys

import numpy as np

from .config import load_config
from .network import Network

USAGE = """USAGE
    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
    config_file_i   Configuration file containing description of a neural network we want to generate.
    nb_i            Number of neural networks to generate based on the configuration file."""


def generate(config_path, count):
    """Crée `count` réseaux nommés <nom_du_conf>_<i>.nn ; graine seed + i pour des poids différents."""
    config = load_config(config_path)
    name = os.path.splitext(os.path.basename(config_path))[0]
    for i in range(1, count + 1):
        rng = np.random.default_rng(config["training"]["seed"] + i)
        Network.from_config(config, rng).save(f"{name}_{i}.nn")


def main(argv):
    if len(argv) == 1 and argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if not argv or len(argv) % 2 != 0:
        print(USAGE, file=sys.stderr)
        return 84
    try:
        jobs = [(argv[i], argv[i + 1]) for i in range(0, len(argv), 2)]
        if not all(count.isdigit() and int(count) > 0 for _, count in jobs):
            raise ValueError("nb must be a positive integer")
        jobs = [(path, int(count)) for path, count in jobs]
        for path, count in jobs:
            generate(path, count)
    except (ValueError, OSError) as err:
        print(f"my_torch_generator: {err}", file=sys.stderr)
        return 84
    return 0
