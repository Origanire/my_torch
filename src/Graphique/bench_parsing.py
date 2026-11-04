"""Benchmark du parallélisme : temps de conversion des FEN selon le nombre de processus.

Usage : python3 -m src.Graphique.bench_parsing [fichier] [processus...]
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.clean.dataset import load_chess_file

RESULTS_DIR = os.path.join("doc", "benchmarks")


def run(path, worker_counts, repeats=3):
    """Mesure le meilleur temps sur `repeats` essais pour chaque nombre de processus."""
    times = {}
    for workers in worker_counts:
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            inputs, _ = load_chess_file(path, require_labels=True, workers=workers)
            best = min(best, time.perf_counter() - start)
        times[workers] = best
        print(f"{workers:2d} processus : {best:.2f} s ({len(inputs)} positions)")
    return times


def plot(times, out_path):
    """Courbe temps + accélération (speedup = temps à 1 processus / temps à n processus)."""
    workers = sorted(times)
    speedup = [times[1] / times[w] for w in workers]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(workers, [times[w] for w in workers], "o-")
    ax1.set(xlabel="Nombre de processus", ylabel="Temps (s)", title="Temps de conversion FEN -> vecteur")
    ax2.plot(workers, speedup, "o-", label="mesuré")
    ax2.plot(workers, workers, "--", color="gray", label="idéal (linéaire)")
    ax2.set(xlabel="Nombre de processus", ylabel="Accélération", title="Accélération (speedup)")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("dataset", "train.txt")
    counts = [int(x) for x in sys.argv[2:]] or [1, 2, 4, 8, os.cpu_count() or 1]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run(path, sorted(set([1] + counts)))
    with open(os.path.join(RESULTS_DIR, "parsing.json"), "w") as file:
        json.dump({"file": path, "times": results}, file, indent=2)
    plot(results, os.path.join(RESULTS_DIR, "parsing.png"))
