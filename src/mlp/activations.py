"""Fonctions d'activation : rendent le réseau non linéaire.

Backward : dL/dX = dL/dY * f'(X) (dérivation en chaîne, élément par élément).
"""

import numpy as np


class ReLU:
    """f(x) = max(0, x)."""

    def forward(self, inputs):
        self.mask = inputs > 0
        return inputs * self.mask

    def backward(self, grad_output):
        # Dérivée : 1 si x > 0, sinon 0
        return grad_output * self.mask

    def params(self):
        return []


class Tanh:
    """f(x) = tanh(x), sortie entre -1 et 1."""

    def forward(self, inputs):
        self.outputs = np.tanh(inputs)
        return self.outputs

    def backward(self, grad_output):
        # Dérivée : 1 - tanh(x)²
        return grad_output * (1.0 - self.outputs ** 2)

    def params(self):
        return []


class Sigmoid:
    """f(x) = 1 / (1 + e^-x), sortie entre 0 et 1."""

    def forward(self, inputs):
        # Clip : évite un dépassement de capacité de exp() pour les grandes valeurs
        self.outputs = 1.0 / (1.0 + np.exp(-np.clip(inputs, -50, 50)))
        return self.outputs

    def backward(self, grad_output):
        # Dérivée : s(x) * (1 - s(x))
        return grad_output * self.outputs * (1.0 - self.outputs)

    def params(self):
        return []


ACTIVATIONS = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid}
