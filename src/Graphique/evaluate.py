"""Évaluation d'un réseau entraîné sur un jeu de test : métriques, courbes, matrice de confusion.

Compare aussi la prédiction classique et le MC dropout (moyenne de plusieurs passes avec dropout actif).

Usage : python3 -m src.Graphique.evaluate RESEAU.nn FICHIER_TEST DOSSIER_SORTIE [mc_samples]
"""

import json
import os
import sys
import time

from src.Graphique.plots import plot_confusion, plot_curves
from src.clean.dataset import load_chess_file
from src.clean.labels import CLASSES
from src.mlp.metrics import classification_report
from src.mlp.network import Network


def score(network, inputs, targets, mc_samples=0):
    """Rapport de classification et temps de prédiction."""
    start = time.perf_counter()
    predictions = network.predict(inputs, mc_samples=mc_samples)
    elapsed = time.perf_counter() - start
    return classification_report(targets, predictions, len(CLASSES)), elapsed


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        sys.exit(84)
    net_path, test_path, out_dir = sys.argv[1:4]
    mc = int(sys.argv[4]) if len(sys.argv) == 5 else 10
    network = Network.load(net_path)
    inputs, targets = load_chess_file(test_path, require_labels=True)

    report, elapsed = score(network, inputs, targets)
    mc_report, mc_elapsed = score(network, inputs, targets, mc)
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(net_path))[0]
    result = {"name": name, "history": network.metadata.get("history", []), "test": report,
              "predict_time": elapsed, "mc_dropout": {"samples": mc, "test": mc_report, "predict_time": mc_elapsed},
              "config": network.config, "metadata": {k: v for k, v in network.metadata.items() if k != "history"}}
    with open(os.path.join(out_dir, "results.json"), "w") as file:
        json.dump(result, file, indent=1)
    plot_confusion(result, os.path.join(out_dir, "confusion.png"))
    if result["history"]:
        plot_curves([result], os.path.join(out_dir, "curves.png"))

    print(f"{len(targets)} positions de test")
    print(f"prédiction classique : acc {report['accuracy']:.2%}  F1 macro {report['macro_f1']:.2%}  ({elapsed:.1f}s)")
    print(f"MC dropout ({mc} passes) : acc {mc_report['accuracy']:.2%}  F1 macro {mc_report['macro_f1']:.2%}  ({mc_elapsed:.1f}s)")
    print("| Classe | Precision | Recall | F1 |\n|---|---|---|---|")
    for i, c in enumerate(CLASSES):
        print(f"| {c} | {report['precision'][i]:.2%} | {report['recall'][i]:.2%} | {report['f1'][i]:.2%} |")
