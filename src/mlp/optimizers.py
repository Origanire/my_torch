"""Optimiseurs : mettent à jour les poids à partir des gradients.

weight_decay (régularisation L2) : ajoute weight_decay * W au gradient,
ce qui rapproche les poids de 0 et limite le surapprentissage.
"""

import numpy as np


class SGD:
    """Descente de gradient : W = W - lr * dW."""

    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def regularized(self, param, grad):
        """Gradient + terme L2 (le biais n'est pas régularisé : 1 dimension)."""
        if self.weight_decay and param.ndim > 1:
            return grad + self.weight_decay * param
        return grad

    def step(self, params):
        """params : liste de paires (paramètre, gradient), modifiées en place."""
        for param, grad in params:
            param -= self.learning_rate * self.regularized(param, grad)


class Momentum(SGD):
    """Garde une vitesse v = beta * v + dW : accélère dans les directions constantes."""

    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.velocity = {}

    def step(self, params):
        for param, grad in params:
            v = self.velocity.setdefault(id(param), np.zeros_like(param))
            v *= self.beta
            v += self.regularized(param, grad)
            param -= self.learning_rate * v


class Adam(SGD):
    """Momentum (m) + pas adapté à chaque poids selon la taille moyenne de ses gradients (v)."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moments = {}
        self.t = 0

    def step(self, params):
        self.t += 1
        # Correction de biais : m et v partent de 0, on compense au début
        correction1 = 1 - self.beta1 ** self.t
        correction2 = 1 - self.beta2 ** self.t
        for param, grad in params:
            grad = self.regularized(param, grad)
            m, v = self.moments.setdefault(id(param), (np.zeros_like(param), np.zeros_like(param)))
            m *= self.beta1
            m += (1 - self.beta1) * grad
            v *= self.beta2
            v += (1 - self.beta2) * grad ** 2
            param -= self.learning_rate * (m / correction1) / (np.sqrt(v / correction2) + self.epsilon)


OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "adam": Adam}
