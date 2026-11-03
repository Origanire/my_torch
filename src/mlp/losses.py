"""Fonctions de coût : mesurent l'écart entre la prédiction et la bonne réponse."""

import numpy as np


def softmax(logits):
    """Transforme les sorties brutes (N, C) en probabilités dont la somme vaut 1."""
    # Soustraire le max ne change pas le résultat mais évite que exp() déborde
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """Softmax + cross-entropy : L = -log(probabilité de la bonne classe), moyennée sur le batch."""

    def forward(self, logits, targets):
        """targets : indices des bonnes classes, forme (N,). Renvoie la loss moyenne."""
        self.probs = softmax(logits)
        self.targets = targets
        n = logits.shape[0]
        # 1e-12 : évite log(0)
        correct = self.probs[np.arange(n), targets]
        return float(-np.log(correct + 1e-12).mean())

    def backward(self):
        """Renvoie dL/dlogits = (p - y) / N, y étant la bonne réponse en one-hot."""
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.targets] -= 1.0
        return grad / n
