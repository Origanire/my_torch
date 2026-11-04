"""Couches du réseau."""

import numpy as np


class Dense:
    """Couche entièrement connectée : Y = X · W + b."""

    def __init__(self, input_size, output_size, init="he", rng=None):
        rng = rng or np.random.default_rng()
        # Écart-type des poids initiaux : garde la variance stable d'une couche à l'autre
        if init == "he":
            scale = np.sqrt(2.0 / input_size)
        elif init == "xavier":
            scale = np.sqrt(2.0 / (input_size + output_size))
        elif init == "lecun":
            scale = np.sqrt(1.0 / input_size)
        else:
            raise ValueError(f"unknown init: '{init}'")
        self.weights = (rng.standard_normal((input_size, output_size)) * scale).astype(np.float32)
        self.bias = np.zeros(output_size, dtype=np.float32)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self.inputs = None

    def forward(self, inputs):
        """Calcule la sortie pour un batch (N, input_size) -> (N, output_size)."""
        # On garde l'entrée : elle sert au calcul de dW pendant le backward
        self.inputs = inputs
        return inputs @ self.weights + self.bias

    def backward(self, grad_output):
        """Reçoit dL/dY, stocke dL/dW et dL/db, renvoie dL/dX à la couche précédente."""
        # Écriture en place : l'optimiseur garde une référence vers ces tableaux
        self.grad_weights[...] = self.inputs.T @ grad_output
        self.grad_bias[...] = grad_output.sum(axis=0)
        return grad_output @ self.weights.T

    def params(self):
        """Paires (paramètre, gradient) que l'optimiseur mettra à jour."""
        return [(self.weights, self.grad_weights), (self.bias, self.grad_bias)]


class Dropout:
    """Éteint au hasard une proportion `rate` des neurones pendant l'entraînement.

    Dropout inversé : on divise par (1 - rate) à l'entraînement pour que la sortie
    garde le même ordre de grandeur qu'en prédiction, où rien n'est éteint.
    """

    def __init__(self, rate, rng=None):
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"invalid dropout rate: {rate}")
        self.rate = rate
        self.rng = rng or np.random.default_rng()
        self.training = True

    def forward(self, inputs):
        if not self.training or self.rate == 0.0:
            self.mask = None
            return inputs
        keep = self.rng.random(inputs.shape, dtype=np.float32) >= self.rate
        self.mask = keep.astype(inputs.dtype) / (1.0 - self.rate)
        return inputs * self.mask

    def backward(self, grad_output):
        # Les neurones éteints ne reçoivent pas d'erreur
        return grad_output if self.mask is None else grad_output * self.mask

    def params(self):
        return []
