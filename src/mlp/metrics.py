"""Métriques d'évaluation d'un classifieur.

precision(c) : parmi les exemples prédits c, part qui sont vraiment c.
recall(c)    : parmi les vrais c, part que le réseau a retrouvés.
f1(c)        : moyenne harmonique de precision et recall.
"""

import numpy as np


def confusion_matrix(targets, predictions, n_classes):
    """matrix[vraie classe, classe prédite] = nombre d'exemples."""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    return matrix


def classification_report(targets, predictions, n_classes):
    """Précision globale, precision/recall/f1 par classe et F1 macro (moyenne des classes)."""
    matrix = confusion_matrix(targets, predictions, n_classes)
    true_pos = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    precision = np.divide(true_pos, predicted, out=np.zeros(n_classes), where=predicted > 0)
    recall = np.divide(true_pos, actual, out=np.zeros(n_classes), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=denom > 0)
    return {
        "accuracy": float(true_pos.sum() / max(matrix.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "confusion_matrix": matrix.tolist(),
    }
