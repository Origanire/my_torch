"""Réseau de neurones : une suite de couches entraînée par descente de gradient."""

import numpy as np

from .losses import SoftmaxCrossEntropy, softmax


class Network:
    """Empile des couches ; chacune expose forward, backward et params."""

    def __init__(self, layers):
        self.layers = layers
        self.loss = SoftmaxCrossEntropy()

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
        # Mélange à chaque epoch : le réseau ne voit pas les exemples toujours dans le même ordre
        order = rng.permutation(len(inputs))
        total = 0.0
        for start in range(0, len(inputs), batch_size):
            batch = order[start:start + batch_size]
            total += self.train_step(inputs[batch], targets[batch], optimizer) * len(batch)
        return total / len(inputs)

    def predict_proba(self, inputs, batch_size=1024):
        """Probabilités de chaque classe, forme (N, nb_classes)."""
        outputs = [softmax(self.forward(inputs[i:i + batch_size]))
                   for i in range(0, len(inputs), batch_size)]
        return np.concatenate(outputs)

    def predict(self, inputs):
        """Indice de la classe la plus probable pour chaque exemple."""
        return self.predict_proba(inputs).argmax(axis=1)
