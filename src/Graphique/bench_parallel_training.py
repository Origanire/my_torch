"""Benchmark du parallélisme de la recherche d'hyperparamètres.

Mesure le temps pour entraîner k réseaux (une epoch chacun), soit l'un après
l'autre, soit en même temps dans k processus, comme le fait benchmark.py.
Accélération = temps séquentiel / temps parallèle.

Usage : python3 -m src.Graphique.bench_parallel_training [nb_exemples] [k...]
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.Graphique.benchmark import RESULTS_DIR, TRAIN_FILE


def train_one_epoch(n):
    """Entraîne un réseau de la config finale pendant une epoch sur n exemples."""
    import numpy as np
    from src.clean.dataset import load_cached
    from src.mlp.config import load_config
    from src.mlp.network import Network
    from src.mlp.trainer import make_optimizer
    config = load_config(os.path.join("configs", "my_torch_network.conf"))
    inputs, targets = load_cached(TRAIN_FILE)
    inputs, targets = np.asarray(inputs[:n]), targets[:n]
    network = Network.from_config(config, np.random.default_rng(0))
    network.fit_epoch(inputs, targets, make_optimizer(config["training"]),
                      config["training"]["batch_size"], np.random.default_rng(0))


def timed(k, n, parallel):
    """Temps pour k entraînements, en parallèle (k processus) ou séquentiels (1 processus)."""
    with Pool(k if parallel else 1) as pool:
        start = time.perf_counter()
        pool.map(train_one_epoch, [n] * k)
        return time.perf_counter() - start


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    counts = [int(k) for k in sys.argv[2:]] or [1, 2, 4, 8]
    results = {}
    for k in counts:
        sequential, parallel = timed(k, n, False), timed(k, n, True)
        results[k] = {"sequential": sequential, "parallel": parallel, "speedup": sequential / parallel}
        print(f"{k} entraînements : séquentiel {sequential:.1f} s, parallèle {parallel:.1f} s, "
              f"accélération x{sequential / parallel:.2f}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "parallel_training.json"), "w") as file:
        json.dump({"examples": n, "results": results}, file, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.35
    x = range(len(counts))
    ax1.bar([i - width / 2 for i in x], [results[k]["sequential"] for k in counts], width, label="séquentiel")
    ax1.bar([i + width / 2 for i in x], [results[k]["parallel"] for k in counts], width, label="parallèle")
    ax1.set_xticks(list(x), [str(k) for k in counts])
    ax1.set(xlabel="Nombre de réseaux entraînés", ylabel="Temps total (s)", title="Temps pour k entraînements")
    ax1.legend()
    ax2.plot(counts, [results[k]["speedup"] for k in counts], "o-", label="mesuré")
    ax2.plot(counts, counts, "--", color="gray", label="idéal (linéaire)")
    ax2.set(xlabel="Processus en parallèle", ylabel="Accélération", title="Accélération (speedup)")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "parallel_training.png"), dpi=120)
