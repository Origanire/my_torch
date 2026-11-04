"""Benchmark : entraîne plusieurs configurations en parallèle et compare leurs résultats.

Chaque configuration est entraînée sur dataset/train.txt dans son propre processus,
puis évaluée sur dataset/test.txt (jamais vu pendant l'entraînement).
Résultats : doc/benchmarks/<nom>/ (results.json, curves.png, scores.png, confusion.png).

Usage : python3 -m src.Graphique.benchmark NOM conf1.conf [conf2.conf...] [--workers N] [--limit N] [--train FICHIER]
        conf.conf@FICHIER : entraîne cette configuration sur un autre fichier.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

TRAIN_FILE = os.path.join("dataset", "train.txt")
TEST_FILE = os.path.join("dataset", "test.txt")
RESULTS_DIR = os.path.join("doc", "benchmarks")


def run_one(args):
    """Entraîne une configuration et renvoie son historique et ses scores de test."""
    conf_path, limit, train_file = args
    import numpy as np
    from src.clean.dataset import load_cached
    from src.clean.labels import CLASSES
    from src.mlp.config import load_config
    from src.mlp.metrics import classification_report
    from src.mlp.network import Network
    from src.mlp.trainer import train

    name = os.path.splitext(os.path.basename(conf_path))[0]
    config = load_config(conf_path)
    inputs, targets = load_cached(train_file)
    if limit:
        inputs, targets = inputs[:limit], targets[:limit]
    test_inputs, test_targets = load_cached(TEST_FILE)

    network = Network.from_config(config, np.random.default_rng(config["training"]["seed"]))
    start = time.perf_counter()
    history = train(network, inputs, targets, config["training"],
                    log=lambda msg: print(f"[{name}] {msg}", flush=True))
    train_time = time.perf_counter() - start
    report = classification_report(test_targets, network.predict(test_inputs), len(CLASSES))
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    return {"name": name, "config": config, "history": history, "best_epoch": best_epoch,
            "train_time": train_time, "test": report}


def main():
    parser = argparse.ArgumentParser(description="Compare several network configurations.")
    parser.add_argument("name", help="benchmark name (output folder)")
    parser.add_argument("configs", nargs="+", help="configuration files to compare (conf@trainfile to override)")
    parser.add_argument("--workers", type=int, default=4, help="configurations trained in parallel")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N training examples")
    parser.add_argument("--train", default=TRAIN_FILE, help="training file (default: dataset/train.txt)")
    args = parser.parse_args()

    # Partage des cœurs : chaque processus limite les threads de son calcul matriciel
    threads = str(max(1, (os.cpu_count() or 1) // args.workers))
    for var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = threads

    from src.clean.dataset import load_cached
    from src.Graphique.plots import plot_confusion, plot_curves, plot_scores
    jobs = []
    for spec in args.configs:
        path, _, train_file = spec.partition("@")
        jobs.append((path, args.limit, train_file or args.train))
    # Construit le cache une seule fois avant de lancer les processus
    for train_file in {job[2] for job in jobs} | {TEST_FILE}:
        load_cached(train_file)
    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(run_one, jobs)

    out_dir = os.path.join(RESULTS_DIR, args.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as file:
        json.dump(results, file, indent=1)
    plot_curves(results, os.path.join(out_dir, "curves.png"))
    plot_scores(results, os.path.join(out_dir, "scores.png"))
    best = max(results, key=lambda r: r["test"]["macro_f1"])
    plot_confusion(best, os.path.join(out_dir, "confusion.png"))

    print(f"\n| Config | Test acc | F1 macro | Best epoch | Time |\n|---|---|---|---|---|")
    for r in results:
        print(f"| {r['name']} | {r['test']['accuracy']:.2%} | {r['test']['macro_f1']:.2%} "
              f"| {r['best_epoch']}/{len(r['history'])} | {r['train_time']:.0f}s |")


if __name__ == "__main__":
    main()
