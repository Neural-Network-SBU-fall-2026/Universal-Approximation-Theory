"""Run with: python -m unittest discover -s 'Cybenko proof' -p 'test_*.py'."""

import unittest
import numpy as np
from approximate_sine import fit, predict, sigmoid


class SineApproximationTests(unittest.TestCase):
    def test_stable_sigmoid_and_symmetry(self):
        with np.errstate(over="raise", invalid="raise"):
            values = sigmoid(np.array([-1000., -1., 0., 1., 1000.]))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(values + values[::-1], 1.0)
        self.assertEqual(values[2], 0.5)

    def test_requested_network_and_unseen_accuracy(self):
        x = np.linspace(-np.pi, np.pi, 256)
        test = -np.pi + (np.arange(10000) + 0.5) * (2*np.pi/10000)
        self.assertEqual(np.intersect1d(x, test).size, 0)
        w, b, alpha, _, _ = fit(x, np.sin(x), 32)
        self.assertEqual(w.shape, (32,))
        self.assertEqual(b.shape, (32,))
        self.assertEqual(alpha.shape, (32,))
        self.assertEqual(w[0], 0)
        self.assertEqual(b[0], 0)
        direct_sum = np.array([sum(alpha[j]*sigmoid(w[j]*t+b[j]) for j in range(32)) for t in test[::100]])
        np.testing.assert_allclose(predict(test[::100], w, b, alpha), direct_sum, atol=1e-12)
        error = predict(test, w, b, alpha) - np.sin(test)
        self.assertLess(np.sqrt(np.mean(error**2)), 1e-6)
        self.assertLess(np.max(np.abs(error)), 2e-6)
        np.testing.assert_allclose(predict([-np.pi, np.pi], w, b, alpha), 0, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
