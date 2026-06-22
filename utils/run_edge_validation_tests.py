import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import unittest

import numpy as np

from utils.edge_validation import (
    boundary_distance_summary,
    edge_band_mask,
    edge_confusion,
    signed_distance_to_mask,
    spectral_angles_deg,
)
from utils.spectral_validation import aggregate_nested_mean, compute_band_metrics


class EdgeValidationTests(unittest.TestCase):
    def test_signed_distance_orientation_and_edge_band(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True

        signed = signed_distance_to_mask(mask, pixel_size_m=2.0)

        self.assertGreater(signed[2, 2], 0)
        self.assertLess(signed[2, 3], 0)
        np.testing.assert_allclose(signed[2, 2], 2.0)
        np.testing.assert_allclose(signed[2, 3], -2.0)

        edge = edge_band_mask(signed, width_m=2.0)
        self.assertTrue(edge[2, 2])
        self.assertTrue(edge[2, 3])
        self.assertFalse(edge[0, 0])

    def test_sr_to_lr_aggregation_conserves_boundary_cell_mean(self):
        lr = np.array([[0.25, 0.75], [0.10, 0.90]], dtype="float32")
        sr = np.repeat(np.repeat(lr, 4, axis=0), 4, axis=1)
        sr[0:4, 0:4] = np.array(
            [
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.5],
            ],
            dtype="float32",
        )

        aggregated = aggregate_nested_mean(sr, lr.shape)

        np.testing.assert_allclose(aggregated, lr)

    def test_known_edge_bias_and_spectral_angle(self):
        lr_band = np.array([[0.10, 0.20], [0.30, 0.40]], dtype="float32")
        sr_band = lr_band + 0.02
        edge = np.array([[True, False], [True, False]])

        metrics = compute_band_metrics(lr_band, sr_band, valid_mask=edge)

        np.testing.assert_allclose(metrics["bias"], 0.02, rtol=1e-6)
        np.testing.assert_allclose(metrics["mae"], 0.02, rtol=1e-6)

        lr_stack = np.array([[[1.0]], [[0.0]]], dtype="float32")
        sr_stack = np.array([[[0.0]], [[1.0]]], dtype="float32")
        angles = spectral_angles_deg(lr_stack, sr_stack, np.array([[True]]))

        self.assertEqual(angles.size, 1)
        np.testing.assert_allclose(angles[0], 90.0, atol=1e-6)

    def test_boundary_distance_shows_sr_closer_to_reference(self):
        reference = np.zeros((9, 9), dtype=bool)
        reference[:, 4:] = True
        lr_pred = np.zeros((9, 9), dtype=bool)
        lr_pred[:, 6:] = True
        sr_pred = np.zeros((9, 9), dtype=bool)
        sr_pred[:, 5:] = True

        lr_summary = boundary_distance_summary(reference, lr_pred, pixel_size_m=1.0)
        sr_summary = boundary_distance_summary(reference, sr_pred, pixel_size_m=1.0)

        self.assertGreater(
            lr_summary["gt_to_pred_mean_m"], sr_summary["gt_to_pred_mean_m"]
        )
        self.assertGreater(
            lr_summary["symmetric_mean_m"], sr_summary["symmetric_mean_m"]
        )

    def test_edge_restricted_confusion_matches_hand_count(self):
        gt = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
        pred = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
        eval_mask = np.array([[1, 1, 1], [0, 1, 0]], dtype=bool)

        confusion = edge_confusion(pred, gt, eval_mask)

        self.assertEqual(confusion["tp"], 2)
        self.assertEqual(confusion["fp"], 1)
        self.assertEqual(confusion["fn"], 1)
        self.assertEqual(confusion["tn"], 0)
        np.testing.assert_allclose(confusion["recall"], 2 / 3)
        np.testing.assert_allclose(confusion["precision"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
