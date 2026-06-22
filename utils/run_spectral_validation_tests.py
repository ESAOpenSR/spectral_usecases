import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import unittest

import numpy as np

from utils.spectral_validation import aggregate_nested_mean, compute_band_metrics


class SpectralValidationTests(unittest.TestCase):
    def test_aggregate_nested_mean_exact_4x_blocks(self):
        lr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        sr = np.repeat(np.repeat(lr, 4, axis=0), 4, axis=1)

        aggregated = aggregate_nested_mean(sr, lr.shape)

        np.testing.assert_allclose(aggregated, lr)
        metrics = compute_band_metrics(lr, aggregated)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["bias"], 0.0)

    def test_known_bias_metrics_match_injected_offset(self):
        lr = np.array([[0.10, 0.20], [0.30, 0.40]], dtype="float32")
        sr = lr + 0.025

        metrics = compute_band_metrics(lr, sr)

        self.assertEqual(metrics["valid_pixels"], 4)
        np.testing.assert_allclose(metrics["mae"], 0.025, rtol=1e-6)
        np.testing.assert_allclose(metrics["bias"], 0.025, rtol=1e-6)
        np.testing.assert_allclose(metrics["median_error"], 0.025, rtol=1e-6)
        np.testing.assert_allclose(metrics["rmse"], 0.025, rtol=1e-6)
        np.testing.assert_allclose(metrics["slope"], 1.0, rtol=1e-6)
        np.testing.assert_allclose(metrics["intercept"], 0.025, rtol=1e-6)

    def test_nodata_and_masked_pixels_are_excluded(self):
        lr = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype="float32")
        sr = np.array([[1.2, 5.0], [3.4, 4.8]], dtype="float32")
        valid_mask = np.array([[True, True], [False, True]])

        metrics = compute_band_metrics(lr, sr, valid_mask=valid_mask, nodata=-9999.0)

        self.assertEqual(metrics["valid_pixels"], 2)
        np.testing.assert_allclose(metrics["mae"], 0.5, rtol=1e-6)
        np.testing.assert_allclose(metrics["bias"], 0.5, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

