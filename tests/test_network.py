"""Vérifie les calculs du réseau : gradients, softmax, optimiseurs, sauvegarde."""

import copy
import os
import tempfile
import unittest

import numpy as np

from src.mlp.activations import ReLU
from src.mlp.config import validate, DEFAULTS
from src.mlp.layers import Dense, Dropout
from src.mlp.losses import SoftmaxCrossEntropy, softmax
from src.mlp.network import Network
from src.mlp.optimizers import Adam


def small_network(rng):
    return Network([Dense(6, 5, rng=rng), ReLU(), Dense(5, 3, rng=rng)])


def loss_of(network, inputs, targets):
    return network.loss.forward(network.forward(inputs), targets)


class GradientTests(unittest.TestCase):
    def test_backward_matches_numerical_gradient(self):
        """Le gradient calculé par backward() égale la dérivée mesurée par différences finies."""
        rng = np.random.default_rng(0)
        network = small_network(rng)
        for layer in network.layers:
            if isinstance(layer, Dense):
                layer.weights = layer.weights.astype(np.float64)
                layer.bias = layer.bias.astype(np.float64)
                layer.grad_weights = np.zeros_like(layer.weights)
                layer.grad_bias = np.zeros_like(layer.bias)
        inputs = rng.standard_normal((4, 6))
        targets = np.array([0, 2, 1, 2])

        loss_of(network, inputs, targets)
        network.backward(network.loss.backward())

        eps = 1e-6
        for layer in (l for l in network.layers if isinstance(l, Dense)):
            for param, grad in layer.params():
                numerical = np.zeros_like(param)
                for index in np.ndindex(param.shape):
                    saved = param[index]
                    param[index] = saved + eps
                    plus = loss_of(network, inputs, targets)
                    param[index] = saved - eps
                    minus = loss_of(network, inputs, targets)
                    param[index] = saved
                    numerical[index] = (plus - minus) / (2 * eps)
                np.testing.assert_allclose(grad, numerical, rtol=1e-4, atol=1e-7)


class LossTests(unittest.TestCase):
    def test_softmax_is_stable_and_normalised(self):
        probs = softmax(np.array([[1000.0, 1000.0, -1000.0], [0.0, 1.0, 2.0]]))
        self.assertTrue(np.all(np.isfinite(probs)))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        np.testing.assert_allclose(probs[0, :2], 0.5)

    def test_cross_entropy_is_low_for_confident_correct_prediction(self):
        loss = SoftmaxCrossEntropy()
        good = loss.forward(np.array([[10.0, 0.0, 0.0]]), np.array([0]))
        bad = loss.forward(np.array([[10.0, 0.0, 0.0]]), np.array([1]))
        self.assertLess(good, 1e-3)
        self.assertGreater(bad, 5.0)


class LayerTests(unittest.TestCase):
    def test_dropout_keeps_expected_value_and_is_off_in_prediction(self):
        layer = Dropout(0.5, rng=np.random.default_rng(1))
        inputs = np.ones((2000, 50), dtype=np.float32)
        self.assertAlmostEqual(float(layer.forward(inputs).mean()), 1.0, delta=0.02)
        layer.training = False
        np.testing.assert_array_equal(layer.forward(inputs), inputs)


class TrainingTests(unittest.TestCase):
    def test_adam_learns_a_separable_problem(self):
        rng = np.random.default_rng(2)
        inputs = rng.standard_normal((300, 6)).astype(np.float32)
        targets = (inputs[:, 0] > 0).astype(int) + (inputs[:, 1] > 0).astype(int)
        network = small_network(rng)
        optimizer = Adam(learning_rate=0.05)
        for _ in range(200):
            network.train_step(inputs, targets, optimizer)
        _, accuracy = network.evaluate(inputs, targets)
        self.assertGreater(accuracy, 0.9)


class SaveLoadTests(unittest.TestCase):
    def test_save_then_load_gives_identical_predictions(self):
        config = validate(copy.deepcopy(DEFAULTS))
        network = Network.from_config(config, rng=np.random.default_rng(3))
        network.set_training(False)
        inputs = np.random.default_rng(4).random((8, 769)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "net.nn")
            network.save(path)
            loaded = Network.load(path)
        np.testing.assert_allclose(network.predict_proba(inputs), loaded.predict_proba(inputs), rtol=1e-6)

    def test_load_rejects_invalid_file(self):
        with tempfile.NamedTemporaryFile(suffix=".nn") as file:
            file.write(b"not a network")
            file.flush()
            with self.assertRaises(ValueError):
                Network.load(file.name)


if __name__ == "__main__":
    unittest.main()
