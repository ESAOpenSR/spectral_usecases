from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import binary_dilation, binary_erosion, sobel


@dataclass
class ConfusionMetrics:
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    specificity: float
    f1: float
    accuracy: float
    iou: float
    mcc: float
    balanced_accuracy: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "iou": self.iou,
            "mcc": self.mcc,
            "balanced_accuracy": self.balanced_accuracy,
        }


@dataclass
class DetectionMetrics:
    N_LR: int
    N_SR: int
    rel_change: float
    median_LR: float
    median_SR: float
    high_LR: float
    high_SR: float
    high_rel_change: float
    edge_gain: float
    grad_LR: float
    grad_SR: float
    boundary_grad_LR: float
    boundary_grad_SR: float
    p2a_LR: float
    p2a_SR: float
    p2a_rel_change: float
    confusion_LR: ConfusionMetrics
    confusion_SR: ConfusionMetrics

    def as_dict(self) -> Dict[str, object]:
        return {
            "N_LR": self.N_LR,
            "N_SR": self.N_SR,
            "rel_change": self.rel_change,
            "median_LR": self.median_LR,
            "median_SR": self.median_SR,
            "high_LR": self.high_LR,
            "high_SR": self.high_SR,
            "high_rel_change": self.high_rel_change,
            "edge_gain": self.edge_gain,
            "grad_LR": self.grad_LR,
            "grad_SR": self.grad_SR,
            "boundary_grad_LR": self.boundary_grad_LR,
            "boundary_grad_SR": self.boundary_grad_SR,
            "p2a_LR": self.p2a_LR,
            "p2a_SR": self.p2a_SR,
            "p2a_rel_change": self.p2a_rel_change,
            "confusion_LR": self.confusion_LR.as_dict(),
            "confusion_SR": self.confusion_SR.as_dict(),
        }


def _pixel_size(transform):
    """Return (pixel_width, pixel_height) from an affine transform."""

    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)
    return pixel_width, pixel_height


def _perimeter_to_area_ratio(mask: np.ndarray, transform) -> float:
    """
    Compute perimeter-to-area ratio for a binary mask.

    Perimeter is derived from horizontal/vertical edge transitions and outer
    borders, scaled by the pixel spacing. Area uses the pixel area from the
    geotransform. Returns NaN when area is zero.
    """

    mask_bool = mask.astype(np.bool_)
    pixel_width, pixel_height = _pixel_size(transform)

    area = mask_bool.sum(dtype=np.int64) * (pixel_width * pixel_height)
    if area == 0:
        return np.nan

    # Internal boundaries (between 0/1) contribute perimeter equal to the
    # dimension orthogonal to the transition.
    horizontal_transitions = (mask_bool[:, 1:] != mask_bool[:, :-1]).sum(dtype=np.int64)
    vertical_transitions = (mask_bool[1:, :] != mask_bool[:-1, :]).sum(dtype=np.int64)

    perimeter = (horizontal_transitions * pixel_height) + (
        vertical_transitions * pixel_width
    )

    # Outer borders where the mask touches the raster edge also contribute.
    perimeter += mask_bool[0, :].sum(dtype=np.int64) * pixel_height
    perimeter += mask_bool[-1, :].sum(dtype=np.int64) * pixel_height
    perimeter += mask_bool[:, 0].sum(dtype=np.int64) * pixel_width
    perimeter += mask_bool[:, -1].sum(dtype=np.int64) * pixel_width

    return perimeter / area


def _mean_boundary_gradient(signal: np.ndarray, mask: np.ndarray) -> float:
    """Mean Sobel gradient magnitude of a signal sampled along mask boundaries."""

    mask_bool = mask.astype(np.bool_)
    if not mask_bool.any():
        return np.nan

    se = np.ones((3, 3))
    dil = binary_dilation(mask_bool, structure=se)
    ero = binary_erosion(mask_bool, structure=se)
    boundary = dil ^ ero

    grad_x = sobel(signal, axis=1, mode="nearest")
    grad_y = sobel(signal, axis=0, mode="nearest")
    grad_mag = np.hypot(grad_x, grad_y)

    valid_boundary = boundary & ~np.isnan(signal)
    if not valid_boundary.any():
        return np.nan

    return float(np.nanmean(grad_mag[valid_boundary]))


def _mean_gradient(signal: np.ndarray, mask: np.ndarray) -> float:
    """Mean Sobel gradient magnitude of a signal sampled inside a mask."""

    mask_bool = mask.astype(np.bool_)
    if not mask_bool.any():
        return np.nan

    grad_x = sobel(signal, axis=1, mode="nearest")
    grad_y = sobel(signal, axis=0, mode="nearest")
    grad_mag = np.hypot(grad_x, grad_y)

    valid_region = mask_bool & ~np.isnan(signal)
    if not valid_region.any():
        return np.nan

    return float(np.nanmean(grad_mag[valid_region]))


def _reproject_mask_to_target(
    mask_arr, src_transform, src_crs, dst_shape, dst_transform, dst_crs
):
    """Nearest-neighbour reproject/resample of a mask to a target grid."""
    dst = np.zeros(dst_shape, dtype=mask_arr.dtype)
    reproject(
        source=mask_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return dst


def _compute_confusion(pred_mask: np.ndarray, gt_mask: np.ndarray) -> ConfusionMetrics:
    pred_bin = pred_mask.astype(np.bool_)
    gt_bin = gt_mask.astype(np.bool_)

    tp = int(np.nansum(pred_bin & gt_bin))
    tn = int(np.nansum(~pred_bin & ~gt_bin))
    fp = int(np.nansum(pred_bin & ~gt_bin))
    fn = int(np.nansum(~pred_bin & gt_bin))

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else np.nan

    if np.isnan(recall) or np.isnan(specificity):
        balanced_accuracy = np.nan
    else:
        balanced_accuracy = 0.5 * (recall + specificity)

    return ConfusionMetrics(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        accuracy=accuracy,
        iou=iou,
        mcc=mcc,
        balanced_accuracy=balanced_accuracy,
    )


def compute_detection_metrics(
    lr_signal_path: Path,
    sr_signal_path: Path,
    lr_det_path: Path,
    sr_det_path: Path,
    gt_path: Path,
    high_thr: float,
    edge_valid_mask_path: Path | None = None,
) -> DetectionMetrics:
    """Compute shared flood/fire metrics on LR and SR datasets."""

    lr_signal_src = rasterio.open(lr_signal_path)
    sr_signal_src = rasterio.open(sr_signal_path)
    lr_det_src = rasterio.open(lr_det_path)
    sr_det_src = rasterio.open(sr_det_path)
    gt_src = rasterio.open(gt_path)

    lr_signal = lr_signal_src.read(1).astype("float32")
    sr_signal = sr_signal_src.read(1).astype("float32")
    det_LR_raw = lr_det_src.read(1).astype("uint8")
    det_SR_raw = sr_det_src.read(1).astype("uint8")
    gt = gt_src.read(1).astype("uint8")

    lr_nod = lr_signal_src.nodata if lr_signal_src.nodata is not None else -9999
    sr_nod = sr_signal_src.nodata if sr_signal_src.nodata is not None else -9999

    det_LR = (det_LR_raw > 0).astype("uint8")
    det_SR = (det_SR_raw > 0).astype("uint8")

    if (
        gt.shape != lr_signal.shape
        or gt_src.transform != lr_signal_src.transform
        or gt_src.crs != lr_signal_src.crs
    ):
        gt_lr_signal = _reproject_mask_to_target(
            gt,
            src_transform=gt_src.transform,
            src_crs=gt_src.crs,
            dst_shape=lr_signal.shape,
            dst_transform=lr_signal_src.transform,
            dst_crs=lr_signal_src.crs,
        )
    else:
        gt_lr_signal = gt.copy()

    if (
        gt.shape != sr_signal.shape
        or gt_src.transform != sr_signal_src.transform
        or gt_src.crs != sr_signal_src.crs
    ):
        gt_sr_signal = _reproject_mask_to_target(
            gt,
            src_transform=gt_src.transform,
            src_crs=gt_src.crs,
            dst_shape=sr_signal.shape,
            dst_transform=sr_signal_src.transform,
            dst_crs=sr_signal_src.crs,
        )
    else:
        gt_sr_signal = gt.copy()

    if (
        gt.shape != det_LR.shape
        or gt_src.transform != lr_det_src.transform
        or gt_src.crs != lr_det_src.crs
    ):
        gt_lr_det = _reproject_mask_to_target(
            gt,
            src_transform=gt_src.transform,
            src_crs=gt_src.crs,
            dst_shape=det_LR.shape,
            dst_transform=lr_det_src.transform,
            dst_crs=lr_det_src.crs,
        )
    else:
        gt_lr_det = gt.copy()

    if (
        gt.shape != det_SR.shape
        or gt_src.transform != sr_det_src.transform
        or gt_src.crs != sr_det_src.crs
    ):
        gt_sr_det = _reproject_mask_to_target(
            gt,
            src_transform=gt_src.transform,
            src_crs=gt_src.crs,
            dst_shape=det_SR.shape,
            dst_transform=sr_det_src.transform,
            dst_crs=sr_det_src.crs,
        )
    else:
        gt_sr_det = gt.copy()

    if (
        det_LR.shape != det_SR.shape
        or lr_det_src.transform != sr_det_src.transform
        or lr_det_src.crs != sr_det_src.crs
    ):
        det_LR_sr = _reproject_mask_to_target(
            det_LR,
            src_transform=lr_det_src.transform,
            src_crs=lr_det_src.crs,
            dst_shape=det_SR.shape,
            dst_transform=sr_det_src.transform,
            dst_crs=sr_det_src.crs,
        )
    else:
        det_LR_sr = det_LR.copy()

    if (
        sr_signal.shape != det_SR.shape
        or sr_signal_src.transform != sr_det_src.transform
        or sr_signal_src.crs != sr_det_src.crs
    ):
        det_SR_signal_grid = _reproject_mask_to_target(
            det_SR,
            src_transform=sr_det_src.transform,
            src_crs=sr_det_src.crs,
            dst_shape=sr_signal.shape,
            dst_transform=sr_signal_src.transform,
            dst_crs=sr_signal_src.crs,
        )
        det_LR_signal_grid = _reproject_mask_to_target(
            det_LR_sr,
            src_transform=sr_det_src.transform,
            src_crs=sr_det_src.crs,
            dst_shape=sr_signal.shape,
            dst_transform=sr_signal_src.transform,
            dst_crs=sr_signal_src.crs,
        )
    else:
        det_SR_signal_grid = det_SR.copy()
        det_LR_signal_grid = det_LR_sr.copy()

    if (
        lr_signal.shape != det_LR.shape
        or lr_signal_src.transform != lr_det_src.transform
        or lr_signal_src.crs != lr_det_src.crs
    ):
        det_LR_signal = _reproject_mask_to_target(
            det_LR,
            src_transform=lr_det_src.transform,
            src_crs=lr_det_src.crs,
            dst_shape=lr_signal.shape,
            dst_transform=lr_signal_src.transform,
            dst_crs=lr_signal_src.crs,
        )
    else:
        det_LR_signal = det_LR.copy()

    lr_signal[lr_signal == lr_nod] = np.nan
    sr_signal[sr_signal == sr_nod] = np.nan

    N_LR = int(np.nansum(det_LR_sr.astype(np.int64)))
    N_SR = int(np.nansum(det_SR.astype(np.int64)))
    rel_change = (float(N_SR) - float(N_LR)) / max(float(N_LR), 1.0)

    p2a_LR = _perimeter_to_area_ratio(det_LR, lr_det_src.transform)
    p2a_SR = _perimeter_to_area_ratio(det_SR, sr_det_src.transform)
    p2a_rel_change = (p2a_SR - p2a_LR) / max(p2a_LR, 1e-9)

    median_LR = np.nanmedian(lr_signal[gt_lr_signal == 1])
    median_SR = np.nanmedian(sr_signal[gt_sr_signal == 1])

    high_LR = np.nanmean((lr_signal[gt_lr_signal == 1] >= high_thr).astype("float32"))
    high_SR = np.nanmean((sr_signal[gt_sr_signal == 1] >= high_thr).astype("float32"))
    high_rel_change = (high_SR - high_LR) / max(high_LR, 1e-9)

    grad_LR = _mean_gradient(lr_signal, det_LR_signal)
    grad_SR = _mean_gradient(sr_signal, det_SR_signal_grid)

    boundary_grad_LR = _mean_boundary_gradient(lr_signal, det_LR_signal)
    boundary_grad_SR = _mean_boundary_gradient(sr_signal, det_SR_signal_grid)

    se = np.ones((5, 5))
    dil_sr = binary_dilation(gt_sr_signal, structure=se)
    ero_sr = binary_erosion(gt_sr_signal, structure=se)
    edge_sr = (dil_sr.astype("uint8") - ero_sr.astype("uint8")) == 1
    if edge_valid_mask_path is not None:
        with rasterio.open(edge_valid_mask_path) as edge_valid_src:
            edge_valid_mask = edge_valid_src.read(1).astype("uint8")
            if (
                edge_valid_mask.shape != edge_sr.shape
                or edge_valid_src.transform != sr_signal_src.transform
                or edge_valid_src.crs != sr_signal_src.crs
            ):
                edge_valid_mask = _reproject_mask_to_target(
                    edge_valid_mask,
                    src_transform=edge_valid_src.transform,
                    src_crs=edge_valid_src.crs,
                    dst_shape=edge_sr.shape,
                    dst_transform=sr_signal_src.transform,
                    dst_crs=sr_signal_src.crs,
                )

        edge_sr &= edge_valid_mask == 1

    LR_edge_detected = np.nansum(det_LR_signal_grid[edge_sr])
    SR_edge_detected = np.nansum(det_SR_signal_grid[edge_sr])
    if LR_edge_detected == 0:
        edge_gain = np.nan
    else:
        edge_gain = (SR_edge_detected - LR_edge_detected) / LR_edge_detected

    confusion_LR = _compute_confusion(det_LR, gt_lr_det)
    confusion_SR = _compute_confusion(det_SR, gt_sr_det)

    lr_signal_src.close()
    sr_signal_src.close()
    lr_det_src.close()
    sr_det_src.close()
    gt_src.close()

    return DetectionMetrics(
        N_LR=N_LR,
        N_SR=N_SR,
        rel_change=rel_change,
        median_LR=median_LR,
        median_SR=median_SR,
        high_LR=high_LR,
        high_SR=high_SR,
        high_rel_change=high_rel_change,
        edge_gain=edge_gain,
        grad_LR=grad_LR,
        grad_SR=grad_SR,
        boundary_grad_LR=boundary_grad_LR,
        boundary_grad_SR=boundary_grad_SR,
        p2a_LR=p2a_LR,
        p2a_SR=p2a_SR,
        p2a_rel_change=p2a_rel_change,
        confusion_LR=confusion_LR,
        confusion_SR=confusion_SR,
    )


def print_pretty_table(
    metrics: DetectionMetrics, title: str, spectral_name: str
) -> None:
    print(f"\n================ {title} ================\n")
    print(f"{'Metric':<25} {'LR':>15} {'SR':>15} {'Δ vs LR':>15}")
    print("-" * 70)

    print(
        f"{'Detected Pixels':<25} "
        f"{metrics.N_LR:>15,.0f} "
        f"{metrics.N_SR:>15,.0f} "
        f"{metrics.rel_change*100:>14.2f}%"
    )
    print(
        f"{f'Median {spectral_name}':<25} "
        f"{metrics.median_LR:>15.4f} "
        f"{metrics.median_SR:>15.4f} "
        f"{'--':>15}"
    )
    print(
        f"{'High-Conf. Fraction':<25} "
        f"{metrics.high_LR:>15.4f} "
        f"{metrics.high_SR:>15.4f} "
        f"{metrics.high_rel_change*100:>14.2f}%"
    )
    print(
        f"{'Edge-Region Gain':<25} "
        f"{'--':>15} "
        f"{'--':>15} "
        f"{metrics.edge_gain*100:>14.2f}%"
    )
    print(
        f"{f'Mean Grad ({spectral_name})':<25} "
        f"{metrics.grad_LR:>15.6f} "
        f"{metrics.grad_SR:>15.6f} "
        f"{'--':>15}"
    )
    print(
        f"{f'Boundary Grad ({spectral_name})':<25} "
        f"{metrics.boundary_grad_LR:>15.6f} "
        f"{metrics.boundary_grad_SR:>15.6f} "
        f"{'--':>15}"
    )
    print(
        f"{'Perimeter/Area (P2A)':<25} "
        f"{metrics.p2a_LR:>15.6f} "
        f"{metrics.p2a_SR:>15.6f} "
        f"{metrics.p2a_rel_change*100:>14.2f}%"
    )

    print("\n================================================\n")


def write_metrics_csv(
    csv_path: Path, metrics: DetectionMetrics, spectral_name: str
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["Metric", "LR", "SR", "Relative Change"])

        writer.writerow(
            ["Detected Pixels", metrics.N_LR, metrics.N_SR, metrics.rel_change]
        )
        writer.writerow(
            [f"Median {spectral_name}", metrics.median_LR, metrics.median_SR, ""]
        )
        writer.writerow(
            [
                "High-Conf Fraction",
                metrics.high_LR,
                metrics.high_SR,
                metrics.high_rel_change,
            ]
        )
        writer.writerow(["Edge-Region Gain", "", "", metrics.edge_gain])
        writer.writerow(
            [f"Mean Grad ({spectral_name})", metrics.grad_LR, metrics.grad_SR, ""]
        )
        writer.writerow(
            [
                f"Boundary Grad ({spectral_name})",
                metrics.boundary_grad_LR,
                metrics.boundary_grad_SR,
                "",
            ]
        )
        writer.writerow(
            [
                "Perimeter/Area (P2A)",
                metrics.p2a_LR,
                metrics.p2a_SR,
                metrics.p2a_rel_change,
            ]
        )

        writer.writerow(
            ["True Positives", metrics.confusion_LR.tp, metrics.confusion_SR.tp, ""]
        )
        writer.writerow(
            ["True Negatives", metrics.confusion_LR.tn, metrics.confusion_SR.tn, ""]
        )
        writer.writerow(
            ["False Positives", metrics.confusion_LR.fp, metrics.confusion_SR.fp, ""]
        )
        writer.writerow(
            ["False Negatives", metrics.confusion_LR.fn, metrics.confusion_SR.fn, ""]
        )
        writer.writerow(
            [
                "Precision",
                metrics.confusion_LR.precision,
                metrics.confusion_SR.precision,
                "",
            ]
        )
        writer.writerow(
            ["Recall", metrics.confusion_LR.recall, metrics.confusion_SR.recall, ""]
        )
        writer.writerow(
            [
                "Specificity",
                metrics.confusion_LR.specificity,
                metrics.confusion_SR.specificity,
                "",
            ]
        )
        writer.writerow(
            ["F1 Score", metrics.confusion_LR.f1, metrics.confusion_SR.f1, ""]
        )
        writer.writerow(
            [
                "Accuracy",
                metrics.confusion_LR.accuracy,
                metrics.confusion_SR.accuracy,
                "",
            ]
        )
        writer.writerow(["IoU", metrics.confusion_LR.iou, metrics.confusion_SR.iou, ""])
        writer.writerow(["MCC", metrics.confusion_LR.mcc, metrics.confusion_SR.mcc, ""])
        writer.writerow(
            [
                "Balanced Accuracy",
                metrics.confusion_LR.balanced_accuracy,
                metrics.confusion_SR.balanced_accuracy,
                "",
            ]
        )
