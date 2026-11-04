"""Lecture et validation des fichiers de configuration (.conf, format INI).

[network]  : hyperparamètres du modèle (architecture)
[training] : hyperparamètres de l'algorithme d'apprentissage
"""

import configparser

from .activations import ACTIVATIONS
from .optimizers import OPTIMIZERS

DEFAULTS = {
    "network": {
        "input_size": 769,
        "output_size": 5,
        "hidden_layers": [256, 128],
        "activation": "relu",
        "init": "he",
        "dropout": 0.0,
    },
    "training": {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 20,
        "weight_decay": 0.0,
        "lr_decay": 1.0,
        "validation_split": 0.1,
        "patience": 5,
        "seed": 42,
    },
}

TYPES = {
    "input_size": int, "output_size": int, "dropout": float,
    "learning_rate": float, "batch_size": int, "epochs": int, "weight_decay": float,
    "lr_decay": float, "validation_split": float, "patience": int, "seed": int,
}


def parse_value(key, raw):
    """Convertit une valeur texte dans le type attendu pour cette clé."""
    if key == "hidden_layers":
        return [int(x) for x in raw.replace(",", " ").split()]
    if key in TYPES:
        return TYPES[key](raw)
    return raw.strip().lower()


def validate(config):
    """Vérifie la cohérence de la configuration, lève ValueError sinon."""
    net, train = config["network"], config["training"]
    if net["input_size"] <= 0 or net["output_size"] <= 0 or any(n <= 0 for n in net["hidden_layers"]):
        raise ValueError("layer sizes must be positive")
    if net["activation"] not in ACTIVATIONS:
        raise ValueError(f"unknown activation: '{net['activation']}'")
    if net["init"] not in ("he", "xavier", "lecun"):
        raise ValueError(f"unknown init: '{net['init']}'")
    if not 0.0 <= net["dropout"] < 1.0:
        raise ValueError("dropout must be in [0, 1[")
    if train["optimizer"] not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer: '{train['optimizer']}'")
    if train["learning_rate"] <= 0 or train["batch_size"] <= 0 or train["epochs"] <= 0:
        raise ValueError("learning_rate, batch_size and epochs must be positive")
    if not 0.0 < train["validation_split"] < 1.0:
        raise ValueError("validation_split must be in ]0, 1[")
    if not 0.0 < train["lr_decay"] <= 1.0:
        raise ValueError("lr_decay must be in ]0, 1]")
    return config


def load_config(path):
    """Lit un fichier .conf et renvoie un dict complété par les valeurs par défaut."""
    # Autorise les commentaires en fin de ligne : "dropout = 0.2  ; commentaire"
    parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    if not parser.read(path):
        raise ValueError(f"cannot read config file: '{path}'")
    config = {section: dict(values) for section, values in DEFAULTS.items()}
    for section in parser.sections():
        if section not in DEFAULTS:
            raise ValueError(f"unknown section: '[{section}]'")
        for key, raw in parser[section].items():
            if key not in DEFAULTS[section]:
                raise ValueError(f"unknown key: '{key}' in [{section}]")
            try:
                config[section][key] = parse_value(key, raw)
            except ValueError:
                raise ValueError(f"invalid value for '{key}': '{raw}'") from None
    return validate(config)
