"""Offline correctness tests. Synthetic fixtures are not MNIST benchmarks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import adversarial_stress_tester as stress


def write_fixture(path, labels, missing=False):
    pixels = np.zeros((len(labels), 784))
    for row, label in enumerate(labels):
        pixels[row, label * 60:(label + 1) * 60] = 255
    if missing:
        pixels[:, -1] = np.nan
    frame = pd.DataFrame(pixels, columns=[f"pixel{i}" for i in range(784)])
    frame.insert(0, "label", labels)
    frame.to_csv(path, index=False)


class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0.1, 0.5, 0.9, 1.0]])

    def test_zero_is_identity_and_does_not_mutate(self):
        before = self.X.copy()
        result = stress.inject_noise(self.X, 0, np.random.default_rng(4))
        np.testing.assert_array_equal(result, before)
        np.testing.assert_array_equal(self.X, before)
        self.assertFalse(np.shares_memory(result, self.X))

    def test_bounds_and_budget(self):
        for eps in (0.05, 0.2, 0.5):
            result = stress.inject_noise(self.X, eps, np.random.default_rng(4))
            self.assertEqual(result.shape, self.X.shape)
            self.assertTrue(((result >= 0) & (result <= 1)).all())
            self.assertLessEqual(np.max(np.abs(result - self.X)), eps + 1e-12)

    def test_seed_repeats_noise(self):
        a = stress.inject_noise(self.X, 0.2, np.random.default_rng(4))
        b = stress.inject_noise(self.X, 0.2, np.random.default_rng(4))
        np.testing.assert_array_equal(a, b)

    def test_invalid_budget_and_input(self):
        for eps in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                stress.inject_noise(self.X, eps, np.random.default_rng(4))
        for X in (np.array([[-0.1]]), np.array([[1.1]]), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                stress.inject_noise(X, 0.1, np.random.default_rng(4))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.train = Path(self.temp.name) / "train.csv"
        self.test = Path(self.temp.name) / "test.csv"
        write_fixture(self.train, [0, 1, 2] * 3, missing=True)
        write_fixture(self.test, [0, 1, 2], missing=True)

    def test_normalization_imputation_and_feature_count(self):
        X_train, X_test, y_train, y_test = stress.load_data(self.train, self.test)
        self.assertEqual(X_train.shape, (9, 784))
        self.assertEqual(X_test.shape, (3, 784))
        self.assertTrue(np.isfinite(X_train).all())
        self.assertTrue(np.isfinite(X_test).all())
        self.assertEqual(X_train.max(), 1)
        self.assertTrue((X_train[:, -1] == 0).all())
        np.testing.assert_array_equal(y_test, [0, 1, 2])

    def test_reject_bad_width_and_empty_data(self):
        for frame in (pd.DataFrame({"label": [0], "pixel": [0]}),
                      pd.read_csv(self.train).iloc[:0]):
            frame.to_csv(self.test, index=False)
            with self.assertRaises(ValueError):
                stress.read_mnist_csv(self.test)

    def test_reject_bad_labels_and_pixels(self):
        original = pd.read_csv(self.test)
        for column, value in (("label", 10), ("label", 1.5), ("label", np.nan),
                              ("pixel0", -1), ("pixel0", 256), ("pixel0", np.inf)):
            frame = original.astype(float).copy()
            frame.loc[0, column] = value
            frame.to_csv(self.test, index=False)
            with self.assertRaises(ValueError):
                stress.read_mnist_csv(self.test)

    def test_reject_mismatched_column_order(self):
        frame = pd.read_csv(self.test)
        cols = list(frame.columns)
        cols[1], cols[2] = cols[2], cols[1]
        frame[cols].to_csv(self.test, index=False)
        with self.assertRaises(ValueError):
            stress.load_data(self.train, self.test)

    def test_reject_headerless_file(self):
        pd.read_csv(self.train).to_csv(self.train, index=False, header=False)
        with self.assertRaises(ValueError):
            stress.read_mnist_csv(self.train)

    def test_reject_single_training_class_and_unseen_test_class(self):
        write_fixture(self.train, [0, 0, 0])
        with self.assertRaises(ValueError):
            stress.load_data(self.train, self.test)
        write_fixture(self.train, [0, 1, 0, 1])
        with self.assertRaises(ValueError):
            stress.load_data(self.train, self.test)


class EvaluationTests(unittest.TestCase):
    def test_per_model_crossings_and_no_crossing(self):
        eps = [0, 0.1, 0.2]
        self.assertEqual(stress.first_half_baseline_crossing(eps, [0.9, 0.45, 0.3], 0.9), 0.1)
        self.assertEqual(stress.first_half_baseline_crossing(eps, [0.8, 0.45, 0.3], 0.8), 0.2)
        self.assertIsNone(stress.first_half_baseline_crossing(eps, [0.9, 0.8, 0.7], 0.9))

    def test_invalid_epsilon_grid(self):
        for eps in ([], [0.1, 0.2], [0, 0.1, 0.1], [0, np.nan], [0, -0.1]):
            with self.assertRaises(ValueError):
                stress.validate_epsilons(eps)

    def test_shared_inputs_and_failure_denominator(self):
        class RecordingModel:
            def __init__(self):
                self.inputs = []
            def predict(self, X):
                self.inputs.append(X.copy())
                return np.array([0, 0, 2]) if len(self.inputs) <= 2 else np.array([1, 1, 2])
        a, b = RecordingModel(), RecordingModel()
        metrics, baseline = stress.evaluate(
            {"a": a, "b": b}, np.full((3, 784), 0.5), np.array([0, 1, 2]), [0, 0.2])
        for first, second in zip(a.inputs, b.inputs):
            np.testing.assert_array_equal(first, second)
        row = metrics[(metrics.model == "a") & (metrics.epsilon == 0.2)].iloc[0]
        self.assertEqual(row.initially_correct, 2)
        self.assertEqual(row.correct_to_wrong, 1)
        self.assertEqual(row.failure_rate_on_clean_correct, 0.5)
        self.assertAlmostEqual(baseline["a"], 2 / 3)

    def test_no_initially_correct_examples(self):
        class WrongModel:
            def predict(self, X):
                return np.ones(len(X), dtype=int)
        metrics, _ = stress.evaluate({"wrong": WrongModel()}, np.zeros((2, 784)),
                                     np.zeros(2, dtype=int), [0])
        self.assertIsNone(metrics.iloc[0].failure_rate_on_clean_correct)

    def test_empty_test_data(self):
        with self.assertRaises(ValueError):
            stress.evaluate({}, np.zeros((0, 784)), np.array([]))

    def test_end_to_end_outputs_and_multiclass_compatibility(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            train, test, out = folder / "train.csv", folder / "test.csv", folder / "out"
            write_fixture(train, [0, 1, 2] * 8, missing=True)
            write_fixture(test, [0, 1, 2] * 2, missing=True)
            stress.main(["--train-csv", str(train), "--test-csv", str(test),
                         "--output-dir", str(out), "--trees", "5"])
            metrics = pd.read_csv(out / "metrics.csv")
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(len(metrics), 18)
            self.assertEqual(report["training_rows"], 24)
            self.assertEqual(report["features"], 784)
            self.assertEqual(report["models"]["logistic_regression"]["solver"], "liblinear")
            self.assertEqual(set(report["clean_accuracy"]), {"random_forest", "logistic_regression"})
            self.assertEqual(len(report["datasets"]["train_sha256"]), 64)
            for name in ("fragility_curve.png", "visual_audit.png"):
                self.assertEqual((out / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
