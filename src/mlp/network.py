"""Réseau de neurones : une suite de couches entraînée par descente de gradient."""

import json
import zipfile

import numpy as np

from .activations import ACTIVATIONS
from .config import validate
from .layers import Dense, Dropout
from .losses import SoftmaxCrossEntropy, softmax

FORMAT_VERSION = 1


class Network:
    """Empile des couches ; chacune expose forward, backward et params."""

    def __init__(self, layers, config=None):
        self.layers = layers
        self.config = config
        self.loss = SoftmaxCrossEntropy()
        self.metadata = {"epochs_trained": 0}

    @classmethod
    def from_config(cls, config, rng=None):
        """Crée un réseau vierge (poids aléatoires) à partir d'une configuration."""
        rng = rng or np.random.default_rng()
        net = config["network"]
        sizes = [net["input_size"]] + net["hidden_layers"] + [net["output_size"]]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(Dense(sizes[i], sizes[i + 1], init=net["init"], rng=rng))
            # Pas d'activation après la dernière couche : la softmax est dans la loss
            if i < len(sizes) - 2:
                layers.append(ACTIVATIONS[net["activation"]]())
                if net["dropout"] > 0:
                    layers.append(Dropout(net["dropout"], rng=rng))
        return cls(layers, config)

    def set_training(self, training):
        """Active ou coupe le dropout (actif seulement à l'entraînement)."""
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = training

    def forward(self, inputs):
        """Fait traverser le batch de la première à la dernière couche."""
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        """Fait remonter l'erreur de la dernière à la première couche (rétropropagation)."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params(self):
        """Tous les couples (paramètre, gradient) du réseau."""
        return [p for layer in self.layers for p in layer.params()]

    def train_step(self, inputs, targets, optimizer):
        """Un pas d'apprentissage sur un batch : forward, loss, backward, mise à jour."""
        logits = self.forward(inputs)
        loss = self.loss.forward(logits, targets)
        self.backward(self.loss.backward())
        optimizer.step(self.params())
        return loss

    def fit_epoch(self, inputs, targets, optimizer, batch_size=64, rng=None):
        """Une epoch : parcourt tout le dataset mélangé par mini-batchs. Renvoie la loss moyenne."""
        rng = rng or np.random.default_rng()
        self.set_training(True)
        # Mélange à chaque epoch : le réseau ne voit pas les exemples toujours dans le même ordre
        order = rng.permutation(len(inputs))
        total = 0.0
        for start in range(0, len(inputs), batch_size):
            batch = order[start:start + batch_size]
            x = inputs[batch].astype(np.float32)
            total += self.train_step(x, targets[batch], optimizer) * len(batch)
        return total / len(inputs)

    def predict_proba(self, inputs, batch_size=4096, mc_samples=0):
        """Probabilités de chaque classe, forme (N, nb_classes).

        mc_samples > 0 : MC dropout, moyenne de plusieurs prédictions avec dropout actif.
        """
        self.set_training(mc_samples > 0)
        passes = max(mc_samples, 1)
        outputs = []
        for i in range(0, len(inputs), batch_size):
            x = inputs[i:i + batch_size].astype(np.float32)
            outputs.append(sum(softmax(self.forward(x)) for _ in range(passes)) / passes)
        self.set_training(False)
        return np.concatenate(outputs) if outputs else np.zeros((0, self.output_size()))

    def predict(self, inputs, mc_samples=0):
        """Indice de la classe la plus probable pour chaque exemple."""
        return self.predict_proba(inputs, mc_samples=mc_samples).argmax(axis=1)

    def evaluate(self, inputs, targets):
        """Loss et précision (proportion de bonnes réponses) sur un jeu de données."""
        probs = self.predict_proba(inputs)
        n = len(targets)
        loss = float(-np.log(probs[np.arange(n), targets] + 1e-12).mean())
        return loss, float((probs.argmax(axis=1) == targets).mean())

    def output_size(self):
        return [l for l in self.layers if isinstance(l, Dense)][-1].bias.shape[0]

    def save(self, path):
        """Sauvegarde config + poids + métadonnées dans un fichier NPZ (archive de tableaux)."""
        dense = [l for l in self.layers if isinstance(l, Dense)]
        arrays = {
            "format_version": np.array(FORMAT_VERSION),
            "config": np.array(json.dumps(self.config)),
            "metadata": np.array(json.dumps(self.metadata)),
        }
        for i, layer in enumerate(dense):
            arrays[f"weights_{i}"] = layer.weights
            arrays[f"bias_{i}"] = layer.bias
        # Fichier ouvert à la main : sinon numpy ajoute l'extension .npz
        with open(path, "wb") as file:
            np.savez_compressed(file, **arrays)

    @classmethod
    def load(cls, path):
        """Recharge un réseau sauvegardé avec save(). Lève ValueError si le fichier est invalide."""
        try:
            with np.load(path, allow_pickle=False) as data:
                if int(data["format_version"]) != FORMAT_VERSION:
                    raise ValueError("unsupported network file version")
                config = validate(json.loads(str(data["config"])))
                network = cls.from_config(config)
                network.metadata = json.loads(str(data["metadata"]))
                dense = [l for l in network.layers if isinstance(l, Dense)]
                for i, layer in enumerate(dense):
                    weights, bias = data[f"weights_{i}"], data[f"bias_{i}"]
                    if weights.shape != layer.weights.shape or bias.shape != layer.bias.shape:
                        raise ValueError("weights do not match the architecture")
                    layer.weights[...] = weights
                    layer.bias[...] = bias
        except (OSError, KeyError, EOFError, ValueError, zipfile.BadZipFile) as err:
            raise ValueError(f"invalid network file '{path}'") from None
        return network
