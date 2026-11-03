"""Optimiseurs : mettent à jour les poids à partir des gradients."""


class SGD:
    """Descente de gradient : W = W - learning_rate * dW."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, params):
        """params : liste de paires (paramètre, gradient), modifiées en place."""
        for param, grad in params:
            param -= self.learning_rate * grad
