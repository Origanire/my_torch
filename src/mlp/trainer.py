"""Boucle d'entraînement avec validation, early stopping et décroissance du learning rate.

Validation : une partie des données, jamais utilisée pour apprendre, mesure la
généralisation. Si la loss de validation remonte alors que celle d'entraînement
baisse, le réseau surapprend (overfitting).
Early stopping : on s'arrête quand la validation ne progresse plus depuis
`patience` epochs et on garde les meilleurs poids.
"""

import time

import numpy as np

from .layers import Dense
from .optimizers import OPTIMIZERS

# La précision d'entraînement est estimée sur un échantillon pour gagner du temps
TRAIN_EVAL_SAMPLE = 20000


def split_validation(inputs, targets, ratio, rng):
    """Sépare aléatoirement (entraînement, validation)."""
    order = rng.permutation(len(inputs))
    n_val = max(1, int(len(inputs) * ratio))
    val, train = order[:n_val], order[n_val:]
    return inputs[train], targets[train], inputs[val], targets[val]


def snapshot(network):
    """Copie des poids de toutes les couches denses."""
    return [(l.weights.copy(), l.bias.copy()) for l in network.layers if isinstance(l, Dense)]


def restore(network, weights):
    for layer, (w, b) in zip([l for l in network.layers if isinstance(l, Dense)], weights):
        layer.weights[...] = w
        layer.bias[...] = b


def make_optimizer(training):
    """Instancie l'optimiseur décrit dans la section [training] de la config."""
    return OPTIMIZERS[training["optimizer"]](
        learning_rate=training["learning_rate"], weight_decay=training["weight_decay"])


def log_line(message):
    """Affiche immédiatement (flush), même si la sortie est redirigée vers un fichier."""
    print(message, flush=True)


def train(network, inputs, targets, training, log=log_line):
    """Entraîne le réseau selon `training` et renvoie l'historique epoch par epoch."""
    rng = np.random.default_rng(training["seed"])
    x_train, y_train, x_val, y_val = split_validation(inputs, targets, training["validation_split"], rng)
    sample = rng.choice(len(x_train), min(TRAIN_EVAL_SAMPLE, len(x_train)), replace=False)
    optimizer = make_optimizer(training)

    history = []
    best_loss, best_weights, stale = float("inf"), snapshot(network), 0
    log(f"training on {len(x_train)} examples, validating on {len(x_val)}")
    for epoch in range(1, training["epochs"] + 1):
        start = time.perf_counter()
        train_loss = network.fit_epoch(x_train, y_train, optimizer, training["batch_size"], rng)
        _, train_acc = network.evaluate(x_train[sample], y_train[sample])
        val_loss, val_acc = network.evaluate(x_val, y_val)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "learning_rate": optimizer.learning_rate, "time": time.perf_counter() - start,
        })
        log(f"epoch {epoch:3d}  loss {train_loss:.4f}  acc {train_acc:.2%}  "
            f"val_loss {val_loss:.4f}  val_acc {val_acc:.2%}  ({history[-1]['time']:.1f}s)")

        if val_loss < best_loss:
            best_loss, best_weights, stale = val_loss, snapshot(network), 0
        else:
            stale += 1
            if stale >= training["patience"]:
                log(f"early stopping: no improvement for {stale} epochs")
                break
        optimizer.learning_rate *= training["lr_decay"]

    restore(network, best_weights)
    best = min(history, key=lambda h: h["val_loss"])
    network.metadata["epochs_trained"] = network.metadata.get("epochs_trained", 0) + len(history)
    network.metadata["best_val_acc"] = best["val_acc"]
    network.metadata["best_val_loss"] = best["val_loss"]
    network.metadata["history"] = network.metadata.get("history", []) + history
    return history
