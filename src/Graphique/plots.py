"""Graphiques des benchmarks (courbes d'apprentissage, scores, matrice de confusion)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.clean.labels import CLASSES


def plot_curves(results, path):
    """Loss et précision par epoch ; plein = validation, pointillés = entraînement."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, result in enumerate(results):
        color = f"C{i}"
        epochs = [h["epoch"] for h in result["history"]]
        for key, ax in (("loss", ax_loss), ("acc", ax_acc)):
            ax.plot(epochs, [h[f"val_{key}"] for h in result["history"]], color=color, label=result["name"])
            ax.plot(epochs, [h[f"train_{key}"] for h in result["history"]], color=color, ls="--", alpha=0.5)
    ax_loss.set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss (plein = validation, pointillés = entraînement)")
    ax_acc.set(xlabel="Epoch", ylabel="Précision", title="Précision (plein = validation, pointillés = entraînement)")
    ax_acc.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_acc.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_scores(results, path):
    """Barres : précision et F1 macro sur le jeu de test, plus le temps d'entraînement."""
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    fig, (ax_score, ax_time) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    for offset, key, label in ((-0.2, "accuracy", "Précision"), (0.2, "macro_f1", "F1 macro")):
        values = [r["test"][key] for r in results]
        bars = ax_score.bar(x + offset, values, 0.4, label=label)
        ax_score.bar_label(bars, labels=[f"{v:.1%}" for v in values], fontsize=7)
    ax_score.set(ylabel="Score sur le jeu de test", title="Scores sur le jeu de test", ylim=(0, 1.05))
    ax_score.set_xticks(x, names, rotation=20, ha="right")
    ax_score.legend()
    bars = ax_time.bar(x, [r["train_time"] for r in results], color="C2")
    ax_time.bar_label(bars, labels=[f"{r['train_time']:.0f}s" for r in results], fontsize=7)
    ax_time.set(ylabel="Secondes", title="Temps d'entraînement")
    ax_time.set_xticks(x, names, rotation=20, ha="right")
    for ax in (ax_score, ax_time):
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(result, path):
    """Matrice de confusion normalisée par ligne (part de chaque vraie classe)."""
    matrix = np.array(result["test"]["confusion_matrix"], dtype=float)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{normalized[i, j]:.1%}\n({int(matrix[i, j])})", ha="center", va="center",
                    fontsize=7, color="white" if normalized[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="Classe prédite", ylabel="Vraie classe", title=f"Matrice de confusion : {result['name']}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
