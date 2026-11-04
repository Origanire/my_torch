"""Benchmark d'une alternative d'encodage : one-hot (769 valeurs) contre valeurs de pièces (65 valeurs).

Encodage "valeurs" : une valeur par case (pion 1, cavalier 2, fou 3, tour 4, dame 5,
roi 6 ; négatif pour les noirs, 0 si vide) + le trait. Il est 12 fois plus compact
mais impose un ordre artificiel entre les pièces.

Usage : python3 -m src.Graphique.bench_encoding CONF
"""

import json
import os
import sys
import time

import numpy as np

from src.Graphique.benchmark import RESULTS_DIR, TEST_FILE, TRAIN_FILE
from src.Graphique.plots import plot_curves, plot_scores
from src.clean.dataset import load_cached
from src.clean.labels import CLASSES
from src.mlp.config import load_config
from src.mlp.metrics import classification_report
from src.mlp.network import Network
from src.mlp.trainer import train

# Code de chaque plan one-hot (ordre PNBRQKpnbrqk de src/clean/fen.py)
PIECE_CODES = np.array([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6], dtype=np.float32)


def to_piece_values(one_hot):
    """(N, 769) one-hot -> (N, 65) : valeur signée de la pièce sur chaque case + trait."""
    planes = np.asarray(one_hot[:, :768], dtype=np.float32).reshape(-1, 12, 64)
    board = np.einsum("npc,p->nc", planes, PIECE_CODES)
    side = np.asarray(one_hot[:, 768:], dtype=np.float32)
    return np.hstack([board, side])


def run(name, config, inputs, targets, test_inputs, test_targets):
    network = Network.from_config(config, np.random.default_rng(config["training"]["seed"]))
    start = time.perf_counter()
    history = train(network, inputs, targets, config["training"], log=lambda m: print(f"[{name}] {m}", flush=True))
    report = classification_report(test_targets, network.predict(test_inputs), len(CLASSES))
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    return {"name": name, "config": config, "history": history, "best_epoch": best_epoch,
            "train_time": time.perf_counter() - start, "test": report}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        sys.exit(84)
    config = load_config(sys.argv[1])
    x_train, y_train = load_cached(TRAIN_FILE)
    x_test, y_test = load_cached(TEST_FILE)

    results = [run("one_hot_769", config, x_train, y_train, x_test, y_test)]
    values_config = json.loads(json.dumps(config))
    values_config["network"]["input_size"] = 65
    results.append(run("piece_values_65", values_config, to_piece_values(x_train), y_train,
                       to_piece_values(x_test), y_test))

    out_dir = os.path.join(RESULTS_DIR, "encoding")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as file:
        json.dump(results, file, indent=1)
    plot_curves(results, os.path.join(out_dir, "curves.png"))
    plot_scores(results, os.path.join(out_dir, "scores.png"))
    for r in results:
        print(f"| {r['name']} | {r['test']['accuracy']:.2%} | {r['test']['macro_f1']:.2%} "
              f"| {r['best_epoch']}/{len(r['history'])} | {r['train_time']:.0f}s |")
