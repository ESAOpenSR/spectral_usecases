from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, uniform_filter

from utils.metrics import _compute_confusion
from utils.spectral_validation import (
    BAND_LABELS,
    BAND_WAVELENGTHS_NM,
    REFLECTANCE_SCALE,
    _aggregate_band_to_lr_grid,
    _mask_on_lr_grid,
    aggregate_nested_mean,
    compute_band_metrics,
)


EDGE_WIDTH_M = 20.0
PROFILE_MIN_M = -30.0
PROFILE_MAX_M = 30.0
PROFILE_BIN_WIDTH_M = 2.5
BOOTSTRAP_ITERATIONS = 100
BOOTSTRAP_MAX_SAMPLES = 20_000


@dataclass(frozen=True)
class EdgeCase:
    name: str
    label: str
    lr_reflectance_path: Path
    sr_reflectance_path: Path
    lr_index_path: Path
    sr_index_path: Path
    lr_detection_path: Path
    sr_detection_path: Path
    gt_mask_path: Path
    index_name: str
    target_direction: str
    threshold: float | None = None
    valid_mask_path: Path | None = None


def signed_distance_to_mask(mask: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Return signed distance in metres, positive inside mask and negative outside."""

    mask_bool = mask.astype(bool)
    outside = distance_transform_edt(~mask_bool) * pixel_size_m
    signed = -outside.astype("float32")
    del outside

    inside = distance_transform_edt(mask_bool) * pixel_size_m
    signed[mask_bool] = inside[mask_bool].astype("float32")
    return signed


def edge_band_mask(signed_distance_m: np.ndarray, width_m: float = EDGE_WIDTH_M) -> np.ndarray:
    return np.abs(signed_distance_m) <= width_m


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return np.zeros(mask_bool.shape, dtype=bool)
    return binary_dilation(mask_bool) ^ binary_erosion(mask_bool)


def spectral_angles_deg(
    lr_stack: np.ndarray, sr_stack: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Return per-pixel spectral angles between LR and aggregated SR spectra."""

    if lr_stack.shape != sr_stack.shape:
        raise ValueError("lr_stack and sr_stack must have the same shape.")
    if lr_stack.ndim != 3:
        raise ValueError("Stacks must have shape (bands, height, width).")

    lr = np.moveaxis(lr_stack, 0, -1)[valid_mask]
    sr = np.moveaxis(sr_stack, 0, -1)[valid_mask]
    finite = np.isfinite(lr).all(axis=1) & np.isfinite(sr).all(axis=1)
    lr = lr[finite].astype("float64")
    sr = sr[finite].astype("float64")
    if lr.size == 0:
        return np.array([], dtype="float64")

    lr_norm = np.linalg.norm(lr, axis=1)
    sr_norm = np.linalg.norm(sr, axis=1)
    nonzero = (lr_norm > 0) & (sr_norm > 0)
    if not np.any(nonzero):
        return np.array([], dtype="float64")

    lr = lr[nonzero]
    sr = sr[nonzero]
    denom = lr_norm[nonzero] * sr_norm[nonzero]
    cos_theta = np.sum(lr * sr, axis=1) / denom
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def boundary_distance_arrays(
    reference_mask: np.ndarray,
    pred_mask: np.ndarray,
    pixel_size_m: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Distances from GT boundary to prediction and prediction boundary to GT."""

    reference = reference_mask.astype(bool)
    pred = pred_mask.astype(bool)
    valid = np.ones(reference.shape, dtype=bool) if valid_mask is None else valid_mask.astype(bool)

    reference_boundary = boundary_mask(reference) & valid
    pred_boundary = boundary_mask(pred) & valid
    if not reference_boundary.any() or not pred_boundary.any():
        return np.array([], dtype="float64"), np.array([], dtype="float64")

    dist_to_pred = distance_transform_edt(~pred_boundary) * pixel_size_m
    ref_to_pred = dist_to_pred[reference_boundary]
    del dist_to_pred

    dist_to_reference = distance_transform_edt(~reference_boundary) * pixel_size_m
    pred_to_ref = dist_to_reference[pred_boundary]
    return ref_to_pred.astype("float64"), pred_to_ref.astype("float64")


def boundary_distance_summary(
    reference_mask: np.ndarray,
    pred_mask: np.ndarray,
    pixel_size_m: float,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    ref_to_pred, pred_to_ref = boundary_distance_arrays(
        reference_mask, pred_mask, pixel_size_m, valid_mask
    )
    combined = np.concatenate([ref_to_pred, pred_to_ref])
    if combined.size == 0:
        return {
            "boundary_points": 0,
            "gt_to_pred_mean_m": np.nan,
            "pred_to_gt_mean_m": np.nan,
            "symmetric_mean_m": np.nan,
            "symmetric_median_m": np.nan,
            "symmetric_p95_m": np.nan,
        }

    return {
        "boundary_points": int(combined.size),
        "gt_to_pred_mean_m": float(np.mean(ref_to_pred)),
        "pred_to_gt_mean_m": float(np.mean(pred_to_ref)),
        "symmetric_mean_m": float(np.mean(combined)),
        "symmetric_median_m": float(np.median(combined)),
        "symmetric_p95_m": float(np.percentile(combined, 95)),
    }


def edge_confusion(pred_mask: np.ndarray, gt_mask: np.ndarray, eval_mask: np.ndarray) -> dict[str, float]:
    metrics = _compute_confusion(pred_mask[eval_mask], gt_mask[eval_mask])
    return metrics.as_dict()


def _pixel_size(transform) -> float:
    return float((abs(transform.a) + abs(transform.e)) / 2.0)


def _same_grid(src, dst_shape, dst_transform, dst_crs) -> bool:
    return src.shape == dst_shape and src.transform == dst_transform and src.crs == dst_crs


def _read_raster_on_grid(
    path: Path,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    resampling: Resampling,
) -> tuple[np.ndarray, float | int | None]:
    with rasterio.open(path) as src:
        nodata = src.nodata
        if _same_grid(src, dst_shape, dst_transform, dst_crs):
            arr = src.read(1).astype("float32")
        else:
            fill = nodata if nodata is not None else np.nan
            arr = np.full(dst_shape, fill, dtype="float32")
            kwargs = {
                "source": rasterio.band(src, 1),
                "destination": arr,
                "src_transform": src.transform,
                "src_crs": src.crs,
                "dst_transform": dst_transform,
                "dst_crs": dst_crs,
                "resampling": resampling,
            }
            if nodata is not None:
                kwargs["src_nodata"] = nodata
                kwargs["dst_nodata"] = nodata
            reproject(**kwargs)

    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, nodata


def _read_mask_on_grid(
    path: Path,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
) -> np.ndarray:
    arr, _ = _read_raster_on_grid(path, dst_shape, dst_transform, dst_crs, Resampling.nearest)
    return np.nan_to_num(arr, nan=0).astype("uint8") > 0


def _threshold_mask(values: np.ndarray, threshold: float, target_direction: str) -> np.ndarray:
    if target_direction == "high":
        return values >= threshold
    if target_direction == "low":
        return values <= threshold
    raise ValueError("target_direction must be 'high' or 'low'.")


def _otsu_threshold(values_a: np.ndarray, values_b: np.ndarray, bins: int = 256) -> float:
    finite_values = [
        vals[np.isfinite(vals)]
        for vals in (values_a, values_b)
        if vals.size > 0 and np.isfinite(vals).any()
    ]
    if not finite_values:
        raise ValueError("No valid values available for Otsu thresholding.")

    vmin = min(float(np.min(vals)) for vals in finite_values)
    vmax = max(float(np.max(vals)) for vals in finite_values)
    hist = np.zeros(bins, dtype="float64")
    for vals in finite_values:
        hist += np.histogram(vals, bins=bins, range=(vmin, vmax))[0]

    prob = hist / hist.sum()
    bin_edges = np.linspace(vmin, vmax, bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    w0 = np.cumsum(prob)
    w1 = 1.0 - w0
    valid = (w0 > 0) & (w1 > 0)

    mu0_cum = np.cumsum(prob * bin_mids)
    mu_total = mu0_cum[-1]
    mu0 = mu0_cum / np.where(w0 == 0, 1, w0)
    mu1 = (mu_total - mu0_cum) / np.where(w1 == 0, 1, w1)
    sigma_between = w0 * w1 * (mu0 - mu1) ** 2
    sigma_between[~valid] = -np.inf
    return float(bin_mids[np.argmax(sigma_between)])


def _safe_stat(values: np.ndarray, fn) -> float:
    if values.size == 0:
        return np.nan
    return float(fn(values))


def _relative_change(sr: float, lr: float) -> float:
    if not np.isfinite(sr) or not np.isfinite(lr) or abs(lr) < 1e-12:
        return np.nan
    return float((sr - lr) / abs(lr))


def _sample_indices(n: int, max_samples: int, seed: int) -> np.ndarray:
    if n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_samples, replace=False)


def compute_transition_profile(
    signed_distance_m: np.ndarray,
    lr_index_on_sr_grid: np.ndarray,
    sr_index: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
    target_direction: str,
    bin_width_m: float = PROFILE_BIN_WIDTH_M,
    min_m: float = PROFILE_MIN_M,
    max_m: float = PROFILE_MAX_M,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    bins = np.arange(min_m, max_m + bin_width_m, bin_width_m)

    for bin_start, bin_end in zip(bins[:-1], bins[1:]):
        in_bin = (
            valid_mask
            & (signed_distance_m >= bin_start)
            & (signed_distance_m < bin_end)
            & np.isfinite(lr_index_on_sr_grid)
            & np.isfinite(sr_index)
        )
        for product, values in (("LR", lr_index_on_sr_grid[in_bin]), ("SR", sr_index[in_bin])):
            rate = _threshold_mask(values, threshold, target_direction).mean() if values.size else np.nan
            rows.append(
                {
                    "product": product,
                    "distance_bin_start_m": float(bin_start),
                    "distance_bin_end_m": float(bin_end),
                    "distance_bin_mid_m": float((bin_start + bin_end) / 2.0),
                    "valid_pixels": int(values.size),
                    "median": _safe_stat(values, np.median),
                    "p25": _safe_stat(values, lambda v: np.percentile(v, 25)),
                    "p75": _safe_stat(values, lambda v: np.percentile(v, 75)),
                    "threshold_crossing_rate": float(rate),
                }
            )

    return rows


def transition_slope_delta(
    signed_distance_m: np.ndarray,
    lr_index_on_sr_grid: np.ndarray,
    sr_index: np.ndarray,
    valid_mask: np.ndarray,
    target_direction: str,
    window_m: float = 15.0,
    max_samples: int = BOOTSTRAP_MAX_SAMPLES,
    seed: int = 2027,
) -> tuple[float, float, float]:
    valid = (
        valid_mask
        & (np.abs(signed_distance_m) <= window_m)
        & np.isfinite(signed_distance_m)
        & np.isfinite(lr_index_on_sr_grid)
        & np.isfinite(sr_index)
    )
    distance = signed_distance_m[valid].astype("float64")
    lr = lr_index_on_sr_grid[valid].astype("float64")
    sr = sr_index[valid].astype("float64")
    if distance.size < 2 or np.std(distance) == 0:
        return np.nan, np.nan, np.nan

    idx = _sample_indices(distance.size, max_samples, seed)
    distance = distance[idx]
    direction = 1.0 if target_direction == "high" else -1.0
    lr = lr[idx] * direction
    sr = sr[idx] * direction

    lr_slope = float(np.polyfit(distance, lr, 1)[0])
    sr_slope = float(np.polyfit(distance, sr, 1)[0])
    return lr_slope, sr_slope, sr_slope - lr_slope


def _summarize_confusion_rows(
    case: EdgeCase,
    threshold: float,
    edge_width_m: float,
    lr_conf: dict[str, float],
    sr_conf: dict[str, float],
    lr_boundary: dict[str, float | int],
    sr_boundary: dict[str, float | int],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for metric in (
        "tp",
        "tn",
        "fp",
        "fn",
        "precision",
        "recall",
        "specificity",
        "f1",
        "accuracy",
        "iou",
        "mcc",
        "balanced_accuracy",
    ):
        lr = float(lr_conf[metric])
        sr = float(sr_conf[metric])
        rows.append(
            {
                "case": case.name,
                "label": case.label,
                "index_name": case.index_name,
                "threshold": threshold,
                "edge_width_m": edge_width_m,
                "metric": f"edge_{metric}",
                "LR": lr,
                "SR": sr,
                "delta": sr - lr,
                "relative_change": _relative_change(sr, lr),
            }
        )

    for metric in (
        "gt_to_pred_mean_m",
        "pred_to_gt_mean_m",
        "symmetric_mean_m",
        "symmetric_median_m",
        "symmetric_p95_m",
    ):
        lr = float(lr_boundary[metric])
        sr = float(sr_boundary[metric])
        rows.append(
            {
                "case": case.name,
                "label": case.label,
                "index_name": case.index_name,
                "threshold": threshold,
                "edge_width_m": edge_width_m,
                "metric": f"boundary_{metric}",
                "LR": lr,
                "SR": sr,
                "delta": sr - lr,
                "relative_change": _relative_change(sr, lr),
            }
        )

    return rows


def compute_edge_detection_metrics(
    case: EdgeCase,
    output_dir: Path,
    edge_width_m: float = EDGE_WIDTH_M,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    dict[str, np.ndarray],
]:
    with rasterio.open(case.sr_index_path) as sr_src:
        sr_index = sr_src.read(1).astype("float32")
        sr_nodata = sr_src.nodata
        sr_shape = sr_index.shape
        sr_transform = sr_src.transform
        sr_crs = sr_src.crs
    if sr_nodata is not None:
        sr_index[sr_index == sr_nodata] = np.nan

    lr_index, _ = _read_raster_on_grid(
        case.lr_index_path, sr_shape, sr_transform, sr_crs, Resampling.bilinear
    )
    lr_det = _read_mask_on_grid(case.lr_detection_path, sr_shape, sr_transform, sr_crs)
    sr_det = _read_mask_on_grid(case.sr_detection_path, sr_shape, sr_transform, sr_crs)
    gt = _read_mask_on_grid(case.gt_mask_path, sr_shape, sr_transform, sr_crs)
    if case.valid_mask_path is None:
        valid = np.ones(sr_shape, dtype=bool)
    else:
        valid = _read_mask_on_grid(case.valid_mask_path, sr_shape, sr_transform, sr_crs)

    valid &= np.isfinite(lr_index) & np.isfinite(sr_index)
    threshold = (
        case.threshold
        if case.threshold is not None
        else _otsu_threshold(lr_index[valid], sr_index[valid])
    )

    pixel_size_m = _pixel_size(sr_transform)
    signed_distance = signed_distance_to_mask(gt, pixel_size_m)
    edge = edge_band_mask(signed_distance, edge_width_m) & valid

    lr_conf = edge_confusion(lr_det, gt, edge)
    sr_conf = edge_confusion(sr_det, gt, edge)
    lr_boundary = boundary_distance_summary(gt, lr_det, pixel_size_m, valid)
    sr_boundary = boundary_distance_summary(gt, sr_det, pixel_size_m, valid)
    detection_rows = _summarize_confusion_rows(
        case, float(threshold), edge_width_m, lr_conf, sr_conf, lr_boundary, sr_boundary
    )

    edge_target = edge & gt
    for product, pred in (("LR", lr_det), ("SR", sr_det)):
        tp = int(np.sum(pred & edge_target))
        fn = int(np.sum((~pred) & edge_target))
        fp = int(np.sum(pred & edge & (~gt)))
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        commission = fp / (tp + fp) if (tp + fp) else np.nan
        omission = 1.0 - recall if np.isfinite(recall) else np.nan
        detection_rows.extend(
            [
                {
                    "case": case.name,
                    "label": case.label,
                    "index_name": case.index_name,
                    "threshold": float(threshold),
                    "edge_width_m": edge_width_m,
                    "metric": f"{product.lower()}_edge_target_pixels",
                    "LR": tp + fn if product == "LR" else np.nan,
                    "SR": tp + fn if product == "SR" else np.nan,
                    "delta": np.nan,
                    "relative_change": np.nan,
                },
                {
                    "case": case.name,
                    "label": case.label,
                    "index_name": case.index_name,
                    "threshold": float(threshold),
                    "edge_width_m": edge_width_m,
                    "metric": f"{product.lower()}_edge_omission_rate",
                    "LR": omission if product == "LR" else np.nan,
                    "SR": omission if product == "SR" else np.nan,
                    "delta": np.nan,
                    "relative_change": np.nan,
                },
                {
                    "case": case.name,
                    "label": case.label,
                    "index_name": case.index_name,
                    "threshold": float(threshold),
                    "edge_width_m": edge_width_m,
                    "metric": f"{product.lower()}_edge_commission_rate",
                    "LR": commission if product == "LR" else np.nan,
                    "SR": commission if product == "SR" else np.nan,
                    "delta": np.nan,
                    "relative_change": np.nan,
                },
            ]
        )

    profile_rows = compute_transition_profile(
        signed_distance,
        lr_index,
        sr_index,
        valid,
        float(threshold),
        case.target_direction,
    )
    for row in profile_rows:
        row.update(
            {
                "case": case.name,
                "label": case.label,
                "index_name": case.index_name,
                "threshold": float(threshold),
            }
        )

    lr_slope, sr_slope, slope_delta = transition_slope_delta(
        signed_distance, lr_index, sr_index, valid, case.target_direction
    )
    detection_rows.append(
        {
            "case": case.name,
            "label": case.label,
            "index_name": case.index_name,
            "threshold": float(threshold),
            "edge_width_m": edge_width_m,
            "metric": "transition_slope_per_m",
            "LR": lr_slope,
            "SR": sr_slope,
            "delta": slope_delta,
            "relative_change": _relative_change(sr_slope, lr_slope),
        }
    )

    arrays = {
        "signed_distance": signed_distance,
        "lr_index": lr_index,
        "sr_index": sr_index,
        "lr_det": lr_det,
        "sr_det": sr_det,
        "gt": gt,
        "valid": valid,
        "edge": edge,
        "threshold": np.array(float(threshold)),
        "pixel_size_m": np.array(pixel_size_m),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    return detection_rows, profile_rows, arrays


def compute_edge_spectral_fidelity(
    case: EdgeCase,
    edge_width_m: float = EDGE_WIDTH_M,
) -> tuple[list[dict[str, float | int | str]], dict[str, np.ndarray]]:
    rows: list[dict[str, float | int | str]] = []

    with rasterio.open(case.lr_reflectance_path) as lr_src, rasterio.open(
        case.sr_reflectance_path
    ) as sr_src:
        valid_mask = _mask_on_lr_grid(case.valid_mask_path, lr_src)
        if valid_mask is None:
            valid_mask = np.ones((lr_src.height, lr_src.width), dtype=bool)
        gt = _mask_on_lr_grid(case.gt_mask_path, lr_src)
        if gt is None:
            raise ValueError(f"No ground-truth mask available for {case.name}.")

        signed_distance = signed_distance_to_mask(gt, _pixel_size(lr_src.transform))
        strata = {
            "edge": np.abs(signed_distance) <= edge_width_m,
            "target_interior": signed_distance > edge_width_m,
            "background": signed_distance < -edge_width_m,
        }

        lr_bands = []
        sr_bands = []
        for band_idx in range(1, lr_src.count + 1):
            lr_band = lr_src.read(band_idx).astype("float32") / REFLECTANCE_SCALE
            if lr_src.nodata is not None:
                lr_band[lr_band == lr_src.nodata / REFLECTANCE_SCALE] = np.nan
            sr_band = _aggregate_band_to_lr_grid(sr_src, lr_src, band_idx) / REFLECTANCE_SCALE
            if sr_src.nodata is not None:
                sr_band[sr_band == sr_src.nodata / REFLECTANCE_SCALE] = np.nan
            lr_bands.append(lr_band)
            sr_bands.append(sr_band)

        lr_stack = np.stack(lr_bands)
        sr_stack = np.stack(sr_bands)
        finite_stack = valid_mask & np.isfinite(lr_stack).all(axis=0) & np.isfinite(sr_stack).all(axis=0)

        for stratum, stratum_mask in strata.items():
            eval_mask = finite_stack & stratum_mask
            angles = spectral_angles_deg(lr_stack, sr_stack, eval_mask)
            rows.append(
                {
                    "case": case.name,
                    "label": case.label,
                    "stratum": stratum,
                    "metric_type": "spectral_angle",
                    "band_index": "",
                    "band_label": "",
                    "wavelength_nm": "",
                    "valid_pixels": int(angles.size),
                    "mae": "",
                    "bias": "",
                    "median_error": "",
                    "rmse": "",
                    "pearson_r": "",
                    "slope": "",
                    "intercept": "",
                    "sam_mean_deg": _safe_stat(angles, np.mean),
                    "sam_median_deg": _safe_stat(angles, np.median),
                    "sam_p25_deg": _safe_stat(angles, lambda v: np.percentile(v, 25)),
                    "sam_p75_deg": _safe_stat(angles, lambda v: np.percentile(v, 75)),
                }
            )

            for band_zero, (band_label, wavelength) in enumerate(
                zip(BAND_LABELS, BAND_WAVELENGTHS_NM)
            ):
                metrics = compute_band_metrics(
                    lr_stack[band_zero], sr_stack[band_zero], valid_mask=eval_mask
                )
                rows.append(
                    {
                        "case": case.name,
                        "label": case.label,
                        "stratum": stratum,
                        "metric_type": "band",
                        "band_index": band_zero + 1,
                        "band_label": band_label,
                        "wavelength_nm": wavelength,
                        "valid_pixels": int(metrics["valid_pixels"]),
                        "mae": metrics["mae"],
                        "bias": metrics["bias"],
                        "median_error": metrics["median_error"],
                        "rmse": metrics["rmse"],
                        "pearson_r": metrics["pearson_r"],
                        "slope": metrics["slope"],
                        "intercept": metrics["intercept"],
                        "sam_mean_deg": "",
                        "sam_median_deg": "",
                        "sam_p25_deg": "",
                        "sam_p75_deg": "",
                    }
                )

    arrays = {
        "lr_stack": lr_stack,
        "sr_stack": sr_stack,
        "valid": finite_stack,
        "edge": finite_stack & strata["edge"],
    }
    return rows, arrays


def _nested_block_view(arr: np.ndarray, dst_shape: tuple[int, int]) -> np.ndarray:
    dst_height, dst_width = dst_shape
    src_height, src_width = arr.shape
    if src_height % dst_height != 0 or src_width % dst_width != 0:
        raise ValueError(f"{arr.shape} is not nested over {dst_shape}.")
    y_factor = src_height // dst_height
    x_factor = src_width // dst_width
    return arr.reshape(dst_height, y_factor, dst_width, x_factor)


def _subpixel_stats(
    lr: np.ndarray,
    sr: np.ndarray,
    boundary_cells: np.ndarray,
    nodata: float | int | None,
) -> dict[str, float | int]:
    sr_float = sr.astype("float32")
    if nodata is not None:
        sr_float = sr_float.copy()
        sr_float[sr_float == nodata] = np.nan

    blocks = _nested_block_view(sr_float, lr.shape)
    with np.errstate(invalid="ignore"):
        block_mean = np.nanmean(blocks, axis=(1, 3))
        block_std = np.nanstd(blocks, axis=(1, 3))
        block_max = np.nanmax(blocks, axis=(1, 3))
        block_min = np.nanmin(blocks, axis=(1, 3))
    contrast = block_max - block_min

    lr_float = lr.astype("float32")
    if nodata is not None:
        lr_float = lr_float.copy()
        lr_float[lr_float == nodata] = np.nan

    valid = boundary_cells & np.isfinite(lr_float) & np.isfinite(block_mean)
    diff = block_mean[valid] - lr_float[valid]
    return {
        "valid_boundary_cells": int(valid.sum()),
        "sr_subpixel_std_mean": _safe_stat(block_std[valid], np.nanmean),
        "sr_subpixel_std_median": _safe_stat(block_std[valid], np.nanmedian),
        "sr_subpixel_contrast_mean": _safe_stat(contrast[valid], np.nanmean),
        "sr_subpixel_contrast_median": _safe_stat(contrast[valid], np.nanmedian),
        "block_mean_mae": _safe_stat(np.abs(diff), np.nanmean),
        "block_mean_bias": _safe_stat(diff, np.nanmean),
        "block_mean_rmse": float(np.sqrt(np.nanmean(diff**2))) if diff.size else np.nan,
    }


def compute_unmixing_metrics(case: EdgeCase) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []

    with rasterio.open(case.lr_index_path) as lr_index_src, rasterio.open(
        case.sr_index_path
    ) as sr_index_src:
        lr_index = lr_index_src.read(1).astype("float32")
        sr_index = sr_index_src.read(1).astype("float32")
        gt_lr = _read_mask_on_grid(
            case.gt_mask_path, lr_index.shape, lr_index_src.transform, lr_index_src.crs
        )
        boundary_cells = boundary_mask(gt_lr)
        local_target_fraction = uniform_filter(gt_lr.astype("float32"), size=3, mode="nearest")
        target_fraction = local_target_fraction[boundary_cells]
        stats = _subpixel_stats(lr_index, sr_index, boundary_cells, lr_index_src.nodata)
        rows.append(
            {
                "case": case.name,
                "label": case.label,
                "asset": case.index_name,
                "valid_boundary_cells": stats["valid_boundary_cells"],
                "target_fraction_median": _safe_stat(target_fraction, np.median),
                **stats,
            }
        )

    with rasterio.open(case.lr_reflectance_path) as lr_src, rasterio.open(
        case.sr_reflectance_path
    ) as sr_src:
        for band_idx, band_label in enumerate(BAND_LABELS, start=1):
            lr_band = lr_src.read(band_idx).astype("float32") / REFLECTANCE_SCALE
            sr_band = sr_src.read(band_idx).astype("float32") / REFLECTANCE_SCALE
            nodata = None if lr_src.nodata is None else lr_src.nodata / REFLECTANCE_SCALE
            stats = _subpixel_stats(lr_band, sr_band, boundary_cells, nodata)
            rows.append(
                {
                    "case": case.name,
                    "label": case.label,
                    "asset": band_label,
                    "valid_boundary_cells": stats["valid_boundary_cells"],
                    "target_fraction_median": _safe_stat(target_fraction, np.median),
                    **stats,
                }
            )

    return rows


def _bootstrap_ci(
    values: np.ndarray,
    statistic,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 1991,
) -> tuple[float, float, float, int]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan, 0

    idx = _sample_indices(values.size, BOOTSTRAP_MAX_SAMPLES, seed)
    values = values[idx]
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype="float64")
    for i in range(iterations):
        sample = values[rng.integers(0, values.size, size=values.size)]
        estimates[i] = statistic(sample)

    return (
        float(statistic(values)),
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
        int(values.size),
    )


def _bootstrap_transition_slope_delta(
    signed_distance_m: np.ndarray,
    lr_index_on_sr_grid: np.ndarray,
    sr_index: np.ndarray,
    valid_mask: np.ndarray,
    target_direction: str,
    window_m: float = 15.0,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 2028,
) -> tuple[float, float, float, float, float, int]:
    valid = (
        valid_mask
        & (np.abs(signed_distance_m) <= window_m)
        & np.isfinite(signed_distance_m)
        & np.isfinite(lr_index_on_sr_grid)
        & np.isfinite(sr_index)
    )
    distance = signed_distance_m[valid].astype("float64")
    lr = lr_index_on_sr_grid[valid].astype("float64")
    sr = sr_index[valid].astype("float64")
    if distance.size < 2 or np.std(distance) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0

    idx = _sample_indices(distance.size, BOOTSTRAP_MAX_SAMPLES, seed)
    distance = distance[idx]
    direction = 1.0 if target_direction == "high" else -1.0
    lr = lr[idx] * direction
    sr = sr[idx] * direction

    def slopes(sample_idx: np.ndarray) -> tuple[float, float, float]:
        x = distance[sample_idx]
        if np.std(x) == 0:
            return np.nan, np.nan, np.nan
        lr_slope = float(np.polyfit(x, lr[sample_idx], 1)[0])
        sr_slope = float(np.polyfit(x, sr[sample_idx], 1)[0])
        return lr_slope, sr_slope, sr_slope - lr_slope

    base_idx = np.arange(distance.size)
    lr_slope, sr_slope, estimate = slopes(base_idx)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype="float64")
    for i in range(iterations):
        _, _, estimates[i] = slopes(rng.integers(0, distance.size, size=distance.size))

    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return estimate, np.nan, np.nan, lr_slope, sr_slope, int(distance.size)
    return (
        estimate,
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
        lr_slope,
        sr_slope,
        int(distance.size),
    )


def compute_bootstrap_summary(
    case: EdgeCase,
    detection_arrays: dict[str, np.ndarray],
    spectral_arrays: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    edge = detection_arrays["edge"].astype(bool)
    gt = detection_arrays["gt"].astype(bool)
    lr_det = detection_arrays["lr_det"].astype(bool)
    sr_det = detection_arrays["sr_det"].astype(bool)

    edge_target = edge & gt
    recall_values = np.stack([lr_det[edge_target], sr_det[edge_target]], axis=1).astype("float64")
    if recall_values.size:
        recall_delta = recall_values[:, 1] - recall_values[:, 0]
        estimate, lo, hi, n = _bootstrap_ci(recall_delta, np.mean, seed=11)
        rows.append(_bootstrap_row(case, "edge_recall_delta", estimate, lo, hi, n))

    pixel_size_m = float(detection_arrays["pixel_size_m"])
    valid = detection_arrays["valid"].astype(bool)
    lr_ref_to_pred, _ = boundary_distance_arrays(gt, lr_det, pixel_size_m, valid)
    sr_ref_to_pred, _ = boundary_distance_arrays(gt, sr_det, pixel_size_m, valid)
    n = min(lr_ref_to_pred.size, sr_ref_to_pred.size)
    if n:
        reduction = lr_ref_to_pred[:n] - sr_ref_to_pred[:n]
        estimate, lo, hi, n_used = _bootstrap_ci(reduction, np.mean, seed=12)
        rows.append(_bootstrap_row(case, "boundary_distance_reduction_m", estimate, lo, hi, n_used))

    slope_delta, lo, hi, lr_slope, sr_slope, n = _bootstrap_transition_slope_delta(
        detection_arrays["signed_distance"],
        detection_arrays["lr_index"],
        detection_arrays["sr_index"],
        detection_arrays["valid"].astype(bool),
        case.target_direction,
    )
    rows.append(
        _bootstrap_row(
            case,
            "transition_slope_delta_per_m",
            slope_delta,
            lo,
            hi,
            n,
            lr_value=lr_slope,
            sr_value=sr_slope,
        )
    )

    angles = spectral_angles_deg(
        spectral_arrays["lr_stack"], spectral_arrays["sr_stack"], spectral_arrays["edge"]
    )
    estimate, lo, hi, n = _bootstrap_ci(angles, np.mean, seed=13)
    rows.append(_bootstrap_row(case, "edge_sam_mean_deg", estimate, lo, hi, n))

    lr_edge = np.moveaxis(spectral_arrays["lr_stack"], 0, -1)[spectral_arrays["edge"]]
    sr_edge = np.moveaxis(spectral_arrays["sr_stack"], 0, -1)[spectral_arrays["edge"]]
    diff = sr_edge - lr_edge
    if diff.size:
        mean_abs_bias = np.nanmean(np.abs(diff), axis=1)
        estimate, lo, hi, n = _bootstrap_ci(mean_abs_bias, np.mean, seed=14)
        rows.append(_bootstrap_row(case, "edge_mean_abs_band_bias", estimate, lo, hi, n))

    return rows


def _bootstrap_row(
    case: EdgeCase,
    metric: str,
    estimate: float,
    ci_low: float,
    ci_high: float,
    sample_size: int,
    lr_value: float = np.nan,
    sr_value: float = np.nan,
) -> dict[str, float | int | str]:
    return {
        "case": case.name,
        "label": case.label,
        "metric": metric,
        "estimate": estimate,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "sample_size": sample_size,
        "LR": lr_value,
        "SR": sr_value,
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_edge_bias_sam(
    spectral_rows: Sequence[dict[str, object]],
    output_dir: Path,
) -> None:
    edge_band_rows = [
        row
        for row in spectral_rows
        if row["metric_type"] == "band" and row["stratum"] == "edge"
    ]
    case_names = list(dict.fromkeys(str(row["case"]) for row in edge_band_rows))
    bias = np.full((len(case_names), len(BAND_LABELS)), np.nan, dtype="float64")
    mae = np.full_like(bias, np.nan)
    for row in edge_band_rows:
        case_idx = case_names.index(str(row["case"]))
        band_idx = int(row["band_index"]) - 1
        bias[case_idx, band_idx] = float(row["bias"])
        mae[case_idx, band_idx] = float(row["mae"])

    sam_rows = [
        row
        for row in spectral_rows
        if row["metric_type"] == "spectral_angle" and row["stratum"] == "edge"
    ]
    sam = [float(row["sam_mean_deg"]) for row in sam_rows]
    sam_labels = [str(row["case"]) for row in sam_rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=True)
    vmax = np.nanmax(np.abs(bias)) if np.isfinite(bias).any() else 1.0
    im0 = axes[0].imshow(bias, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Edge signed bias")
    axes[0].set_xticks(np.arange(len(BAND_LABELS)), BAND_LABELS, rotation=45, ha="right")
    axes[0].set_yticks(np.arange(len(case_names)), case_names)
    fig.colorbar(im0, ax=axes[0], label="SR aggregated - LR reflectance")

    im1 = axes[1].imshow(mae, aspect="auto", cmap="viridis")
    axes[1].set_title("Edge MAE")
    axes[1].set_xticks(np.arange(len(BAND_LABELS)), BAND_LABELS, rotation=45, ha="right")
    axes[1].set_yticks(np.arange(len(case_names)), case_names)
    fig.colorbar(im1, ax=axes[1], label="Reflectance")

    axes[2].bar(sam_labels, sam, color="#4c78a8")
    axes[2].set_title("Edge spectral angle")
    axes[2].set_ylabel("Mean SAM (degrees)")
    axes[2].grid(axis="y", alpha=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "edge_bias_sam_heatmap.png", dpi=260)
    plt.close(fig)


def plot_transition_profiles(
    profile_rows: Sequence[dict[str, object]],
    output_dir: Path,
) -> None:
    cases = list(dict.fromkeys(str(row["case"]) for row in profile_rows))
    fig, axes = plt.subplots(1, len(cases), figsize=(6.0 * len(cases), 4.0), constrained_layout=True)
    if len(cases) == 1:
        axes = [axes]

    colors = {"LR": "#1f77b4", "SR": "#ff7f0e"}
    for ax, case_name in zip(axes, cases):
        case_rows = [row for row in profile_rows if row["case"] == case_name]
        index_name = str(case_rows[0]["index_name"])
        for product in ("LR", "SR"):
            rows = [row for row in case_rows if row["product"] == product]
            rows = sorted(rows, key=lambda row: float(row["distance_bin_mid_m"]))
            x = np.array([float(row["distance_bin_mid_m"]) for row in rows])
            median = np.array([float(row["median"]) for row in rows])
            p25 = np.array([float(row["p25"]) for row in rows])
            p75 = np.array([float(row["p75"]) for row in rows])
            ax.plot(x, median, color=colors[product], label=product, linewidth=2)
            ax.fill_between(x, p25, p75, color=colors[product], alpha=0.18)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{case_name}: edge transition")
        ax.set_xlabel("Signed distance to reference border (m)")
        ax.set_ylabel(index_name)
        ax.grid(True, alpha=0.25)
        ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "signed_distance_profile_panel.png", dpi=260)
    plt.close(fig)


def plot_boundary_distance_summary(
    case_arrays: Sequence[tuple[EdgeCase, dict[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    violin_values = []
    violin_labels = []
    colors = {"LR": "#1f77b4", "SR": "#ff7f0e"}

    for case, arrays in case_arrays:
        gt = arrays["gt"].astype(bool)
        valid = arrays["valid"].astype(bool)
        pixel_size_m = float(arrays["pixel_size_m"])
        for product, pred_key in (("LR", "lr_det"), ("SR", "sr_det")):
            ref_to_pred, _ = boundary_distance_arrays(
                gt, arrays[pred_key].astype(bool), pixel_size_m, valid
            )
            if ref_to_pred.size == 0:
                continue
            values = ref_to_pred
            if values.size > BOOTSTRAP_MAX_SAMPLES:
                values = values[_sample_indices(values.size, BOOTSTRAP_MAX_SAMPLES, 44)]
            sorted_values = np.sort(values)
            cdf = np.linspace(0, 1, sorted_values.size)
            axes[0].plot(
                sorted_values,
                cdf,
                color=colors[product],
                linestyle="-" if case == case_arrays[0][0] else "--",
                label=f"{case.name} {product}",
            )
            violin_values.append(values)
            violin_labels.append(f"{case.name}\n{product}")

    axes[0].set_title("Distance from reference border")
    axes[0].set_xlabel("Distance to predicted border (m)")
    axes[0].set_ylabel("CDF")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    parts = axes[1].violinplot(violin_values, showmedians=True, showextrema=False)
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor("#1f77b4" if "LR" in violin_labels[idx] else "#ff7f0e")
        body.set_alpha(0.45)
    axes[1].set_title("Boundary-distance distributions")
    axes[1].set_ylabel("Distance (m)")
    axes[1].set_xticks(np.arange(1, len(violin_labels) + 1), violin_labels, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "boundary_distance_cdf_violin.png", dpi=260)
    plt.close(fig)


def plot_unmixing_summary(
    unmixing_rows: Sequence[dict[str, object]],
    output_dir: Path,
) -> None:
    rows = [row for row in unmixing_rows if row["asset"] not in BAND_LABELS]
    if not rows:
        rows = list(unmixing_rows)

    labels = [f"{row['case']}\n{row['asset']}" for row in rows]
    std = [float(row["sr_subpixel_std_mean"]) for row in rows]
    contrast = [float(row["sr_subpixel_contrast_mean"]) for row in rows]
    mae = [float(row["block_mean_mae"]) for row in rows]

    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    ax.bar(x - width, std, width=width, label="SR subpixel std")
    ax.bar(x, contrast, width=width, label="SR p95-p05")
    ax.bar(x + width, mae, width=width, label="4x4 mean MAE")
    ax.set_xticks(x, labels)
    ax.set_title("Subpixel edge-cell unmixing")
    ax.set_ylabel("Index units")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "subpixel_edge_cell_distribution.png", dpi=260)
    plt.close(fig)


def write_manuscript_framing(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Edge-Focused Spectral Validation: Manuscript Framing

## Methods addition
We added an edge-focused validation targeted at the mixed pixels that occur along
flood-water and burn-scar borders. Ground-truth masks were reprojected to the
2.5 m SR grid and converted to signed distance fields, with positive distances
inside the mapped target and negative distances outside. The primary edge zone
was defined as pixels within 20 m of the mapped boundary, and transition
profiles were summarized in 2.5 m bins from 30 m outside to 30 m inside.

The LR spectral-index layers were bilinearly resampled to the SR grid so that LR
and SR detections could be compared against the same boundary pixels. In
parallel, SR reflectance was aggregated back to the native 10 m LR grid before
computing all-band spectral fidelity metrics, ensuring that edge sharpness was
not confused with a change in LR-scale spectral support.

## Results framing
The edge metrics should be interpreted jointly. Boundary-distance and edge
recall metrics test whether SR better follows mapped borders. Signed-distance
profiles test whether the index transition is sharper at the border. The
edge-local all-band metrics and spectral angle checks test whether this sharper
transition is accompanied by acceptable spectral preservation and low signed
bias. The subpixel unmixing table further checks that SR creates within-cell
contrast in boundary cells while the 4x4 SR mean remains close to the LR value.

## Limitation statement
This analysis strengthens the evidence for the two available case studies only:
the Valencia flood and Palisades burn-scar scenes included in this repository.
It should not be presented as a broad generalization result until additional
independent events are processed with the same workflow.
"""
    path.write_text(text)


def run_edge_validation(cases: Sequence[EdgeCase], output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    all_spectral_rows: list[dict[str, object]] = []
    all_detection_rows: list[dict[str, object]] = []
    all_profile_rows: list[dict[str, object]] = []
    all_unmixing_rows: list[dict[str, object]] = []
    all_bootstrap_rows: list[dict[str, object]] = []
    case_detection_arrays: list[tuple[EdgeCase, dict[str, np.ndarray]]] = []

    for case in cases:
        print(f"Running edge validation for {case.label}...")
        detection_rows, profile_rows, detection_arrays = compute_edge_detection_metrics(
            case, output_dir
        )
        print(f"  detection and distance profiles complete for {case.name}")
        spectral_rows, spectral_arrays = compute_edge_spectral_fidelity(case)
        print(f"  LR-scale spectral fidelity complete for {case.name}")
        unmixing_rows = compute_unmixing_metrics(case)
        print(f"  subpixel unmixing checks complete for {case.name}")
        bootstrap_rows = compute_bootstrap_summary(case, detection_arrays, spectral_arrays)
        print(f"  bootstrap summary complete for {case.name}")

        all_detection_rows.extend(detection_rows)
        all_profile_rows.extend(profile_rows)
        all_spectral_rows.extend(spectral_rows)
        all_unmixing_rows.extend(unmixing_rows)
        all_bootstrap_rows.extend(bootstrap_rows)
        case_detection_arrays.append((case, detection_arrays))

    write_csv(
        output_dir / "edge_spectral_fidelity.csv",
        all_spectral_rows,
        (
            "case",
            "label",
            "stratum",
            "metric_type",
            "band_index",
            "band_label",
            "wavelength_nm",
            "valid_pixels",
            "mae",
            "bias",
            "median_error",
            "rmse",
            "pearson_r",
            "slope",
            "intercept",
            "sam_mean_deg",
            "sam_median_deg",
            "sam_p25_deg",
            "sam_p75_deg",
        ),
    )
    write_csv(
        output_dir / "edge_detection_metrics.csv",
        all_detection_rows,
        (
            "case",
            "label",
            "index_name",
            "threshold",
            "edge_width_m",
            "metric",
            "LR",
            "SR",
            "delta",
            "relative_change",
        ),
    )
    write_csv(
        output_dir / "edge_transition_profiles.csv",
        all_profile_rows,
        (
            "case",
            "label",
            "index_name",
            "threshold",
            "product",
            "distance_bin_start_m",
            "distance_bin_end_m",
            "distance_bin_mid_m",
            "valid_pixels",
            "median",
            "p25",
            "p75",
            "threshold_crossing_rate",
        ),
    )
    write_csv(
        output_dir / "edge_unmixing_metrics.csv",
        all_unmixing_rows,
        (
            "case",
            "label",
            "asset",
            "valid_boundary_cells",
            "target_fraction_median",
            "sr_subpixel_std_mean",
            "sr_subpixel_std_median",
            "sr_subpixel_contrast_mean",
            "sr_subpixel_contrast_median",
            "block_mean_mae",
            "block_mean_bias",
            "block_mean_rmse",
        ),
    )
    write_csv(
        output_dir / "edge_bootstrap_summary.csv",
        all_bootstrap_rows,
        ("case", "label", "metric", "estimate", "ci95_low", "ci95_high", "sample_size", "LR", "SR"),
    )

    plot_edge_bias_sam(all_spectral_rows, figures_dir)
    plot_transition_profiles(all_profile_rows, figures_dir)
    plot_boundary_distance_summary(case_detection_arrays, figures_dir)
    plot_unmixing_summary(all_unmixing_rows, figures_dir)
    write_manuscript_framing(output_dir / "manuscript_edge_framing.md")
