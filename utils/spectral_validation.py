from __future__ import annotations

import csv
import math
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


BAND_LABELS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
BAND_WAVELENGTHS_NM = (492, 559, 665, 704, 740, 782, 833, 865, 1610, 2180)
REFLECTANCE_SCALE = 10000.0


@dataclass(frozen=True)
class SpectralCase:
    name: str
    label: str
    lr_path: Path
    sr_path: Path
    metrics_path: Path
    target_mask_path: Path | None = None
    valid_mask_path: Path | None = None


@dataclass(frozen=True)
class BandMetric:
    case: str
    band_index: int
    band_label: str
    wavelength_nm: int
    valid_pixels: int
    mae: float
    bias: float
    median_error: float
    rmse: float
    pearson_r: float
    slope: float
    intercept: float


@dataclass(frozen=True)
class SpectralAngleSummary:
    case: str
    valid_pixels: int
    mean_deg: float
    median_deg: float
    p25_deg: float
    p75_deg: float


def aggregate_nested_mean(
    sr_band: np.ndarray, dst_shape: tuple[int, int], nodata: float | int | None = None
) -> np.ndarray:
    """Aggregate an exactly nested SR band to a lower-resolution grid by block mean."""

    if sr_band.ndim != 2:
        raise ValueError("sr_band must be a 2D array.")

    dst_height, dst_width = dst_shape
    src_height, src_width = sr_band.shape
    if src_height % dst_height != 0 or src_width % dst_width != 0:
        raise ValueError(
            f"Source shape {sr_band.shape} is not an integer multiple of {dst_shape}."
        )

    y_factor = src_height // dst_height
    x_factor = src_width // dst_width

    arr = sr_band.astype("float32")
    if nodata is not None:
        arr = arr.copy()
        arr[arr == nodata] = np.nan

    blocks = arr.reshape(dst_height, y_factor, dst_width, x_factor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=(1, 3)).astype("float32")


def compute_band_metrics(
    lr_band: np.ndarray,
    sr_band_on_lr_grid: np.ndarray,
    valid_mask: np.ndarray | None = None,
    nodata: float | int | None = None,
) -> dict[str, float | int]:
    """Compute spectral fidelity metrics for one LR/SR band pair."""

    if lr_band.shape != sr_band_on_lr_grid.shape:
        raise ValueError("LR and SR arrays must have the same shape.")

    valid = np.isfinite(lr_band) & np.isfinite(sr_band_on_lr_grid)
    if nodata is not None:
        valid &= lr_band != nodata
        valid &= sr_band_on_lr_grid != nodata
    if valid_mask is not None:
        if valid_mask.shape != lr_band.shape:
            raise ValueError("valid_mask must match the band shape.")
        valid &= valid_mask.astype(bool)

    x = lr_band[valid].astype("float64")
    y = sr_band_on_lr_grid[valid].astype("float64")
    n = int(x.size)
    if n == 0:
        return {
            "valid_pixels": 0,
            "mae": np.nan,
            "bias": np.nan,
            "median_error": np.nan,
            "rmse": np.nan,
            "pearson_r": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        }

    diff = y - x
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    median_error = float(np.median(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))

    if n < 2 or np.std(x) == 0 or np.std(y) == 0:
        pearson_r = np.nan
        slope = np.nan
        intercept = np.nan
    else:
        pearson_r = float(np.corrcoef(x, y)[0, 1])
        slope, intercept = np.polyfit(x, y, 1)
        slope = float(slope)
        intercept = float(intercept)

    return {
        "valid_pixels": n,
        "mae": mae,
        "bias": bias,
        "median_error": median_error,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "slope": slope,
        "intercept": intercept,
    }


def compute_spectral_angle_summary(
    lr_stack: np.ndarray, sr_stack_on_lr_grid: np.ndarray, valid_mask: np.ndarray
) -> SpectralAngleSummary:
    """Summarize per-pixel spectral angle between LR and aggregated SR spectra."""

    if lr_stack.shape != sr_stack_on_lr_grid.shape:
        raise ValueError("LR and SR stacks must have the same shape.")
    if lr_stack.ndim != 3:
        raise ValueError("Stacks must have shape (bands, height, width).")
    if valid_mask.shape != lr_stack.shape[1:]:
        raise ValueError("valid_mask must match stack spatial dimensions.")

    lr = np.moveaxis(lr_stack, 0, -1)[valid_mask]
    sr = np.moveaxis(sr_stack_on_lr_grid, 0, -1)[valid_mask]
    finite = np.isfinite(lr).all(axis=1) & np.isfinite(sr).all(axis=1)
    lr = lr[finite].astype("float64")
    sr = sr[finite].astype("float64")

    lr_norm = np.linalg.norm(lr, axis=1)
    sr_norm = np.linalg.norm(sr, axis=1)
    nonzero = (lr_norm > 0) & (sr_norm > 0)
    if not np.any(nonzero):
        return SpectralAngleSummary("", 0, np.nan, np.nan, np.nan, np.nan)

    lr = lr[nonzero]
    sr = sr[nonzero]
    denom = lr_norm[nonzero] * sr_norm[nonzero]
    cos_theta = np.sum(lr * sr, axis=1) / denom
    angles = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    return SpectralAngleSummary(
        case="",
        valid_pixels=int(angles.size),
        mean_deg=float(np.mean(angles)),
        median_deg=float(np.median(angles)),
        p25_deg=float(np.percentile(angles, 25)),
        p75_deg=float(np.percentile(angles, 75)),
    )


def _is_exact_nested_grid(sr_src, lr_src) -> bool:
    same_origin = (
        abs(sr_src.transform.c - lr_src.transform.c) < 1e-6
        and abs(sr_src.transform.f - lr_src.transform.f) < 1e-6
    )
    same_crs = sr_src.crs == lr_src.crs
    same_extent = sr_src.bounds == lr_src.bounds
    x_factor = lr_src.transform.a / sr_src.transform.a
    y_factor = lr_src.transform.e / sr_src.transform.e
    integer_factor = abs(x_factor - round(x_factor)) < 1e-6 and abs(
        y_factor - round(y_factor)
    ) < 1e-6
    shape_matches = (
        sr_src.height == lr_src.height * int(round(abs(y_factor)))
        and sr_src.width == lr_src.width * int(round(abs(x_factor)))
    )
    return same_origin and same_crs and same_extent and integer_factor and shape_matches


def _aggregate_band_to_lr_grid(sr_src, lr_src, band_index: int) -> np.ndarray:
    sr_nodata = sr_src.nodata

    if _is_exact_nested_grid(sr_src, lr_src):
        sr_band = sr_src.read(band_index)
        return aggregate_nested_mean(sr_band, (lr_src.height, lr_src.width), sr_nodata)

    dst = np.full((lr_src.height, lr_src.width), np.nan, dtype="float32")
    reproject_kwargs = {
        "source": rasterio.band(sr_src, band_index),
        "destination": dst,
        "src_transform": sr_src.transform,
        "src_crs": sr_src.crs,
        "dst_transform": lr_src.transform,
        "dst_crs": lr_src.crs,
        "resampling": Resampling.average,
    }
    if sr_nodata is not None:
        reproject_kwargs["src_nodata"] = sr_nodata
        reproject_kwargs["dst_nodata"] = np.nan
    reproject(**reproject_kwargs)
    return dst


def _mask_on_lr_grid(mask_path: Path | None, lr_src) -> np.ndarray | None:
    if mask_path is None:
        return None

    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read(1).astype("uint8")
        if (
            mask.shape == (lr_src.height, lr_src.width)
            and mask_src.transform == lr_src.transform
            and mask_src.crs == lr_src.crs
        ):
            return mask == 1

        dst = np.zeros((lr_src.height, lr_src.width), dtype="uint8")
        reproject(
            source=mask,
            destination=dst,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=lr_src.transform,
            dst_crs=lr_src.crs,
            resampling=Resampling.nearest,
        )
        return dst == 1


def _base_valid_mask(lr_stack: np.ndarray, sr_stack: np.ndarray) -> np.ndarray:
    return np.isfinite(lr_stack).all(axis=0) & np.isfinite(sr_stack).all(axis=0)


def _sample_xy(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    xv = x[valid]
    yv = y[valid]
    if xv.size <= max_points:
        return xv, yv
    rng = np.random.default_rng(seed)
    idx = rng.choice(xv.size, size=max_points, replace=False)
    return xv[idx], yv[idx]


def _spectral_stats(stack: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    values = stack[:, mask]
    return {
        "mean": np.nanmean(values, axis=1),
        "p25": np.nanpercentile(values, 25, axis=1),
        "p75": np.nanpercentile(values, 75, axis=1),
        "p05": np.nanpercentile(values, 5, axis=1),
        "p95": np.nanpercentile(values, 95, axis=1),
    }


def _plot_equivalence_panel(
    case: SpectralCase,
    lr_stack: np.ndarray,
    sr_stack: np.ndarray,
    valid_mask: np.ndarray,
    band_metrics: list[BandMetric],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for idx, ax in enumerate(axes.flat):
        lr = lr_stack[idx]
        sr = sr_stack[idx]
        valid = valid_mask & np.isfinite(lr) & np.isfinite(sr)
        x, y = _sample_xy(lr, sr, valid, max_points=250_000, seed=idx + 17)

        hb = ax.hexbin(x, y, gridsize=65, bins="log", mincnt=1, cmap="viridis")
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            lo, hi = np.percentile(np.concatenate([x[finite], y[finite]]), [1, 99])
            pad = (hi - lo) * 0.08 if hi > lo else 0.01
            lo -= pad
            hi += pad
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, label="1:1")
            metric = band_metrics[idx]
            if np.isfinite(metric.slope) and np.isfinite(metric.intercept):
                ax.plot(
                    [lo, hi],
                    [metric.slope * lo + metric.intercept, metric.slope * hi + metric.intercept],
                    color="#d62728",
                    linewidth=1.2,
                    label="fit",
                )
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        ax.set_title(f"{band_metrics[idx].band_label}")
        ax.set_xlabel("LR reflectance")
        ax.set_ylabel("SR aggregated reflectance")
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    fig.colorbar(hb, ax=axes.ravel().tolist(), label="Pixel count (log)")
    fig.suptitle(f"{case.label}: LR vs SR spectral equivalence")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{case.name}_lr_sr_equivalence.png", dpi=220)
    plt.close(fig)


def _plot_residual_panel(
    case: SpectralCase,
    lr_stack: np.ndarray,
    sr_stack: np.ndarray,
    valid_mask: np.ndarray,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.8), constrained_layout=True)
    for idx, ax in enumerate(axes.flat):
        residual = sr_stack[idx] - lr_stack[idx]
        values = residual[valid_mask & np.isfinite(residual)]
        if values.size > 250_000:
            rng = np.random.default_rng(idx + 101)
            values = values[rng.choice(values.size, size=250_000, replace=False)]
        lo, hi = np.percentile(values, [1, 99]) if values.size else (-0.01, 0.01)
        ax.hist(values, bins=100, range=(lo, hi), color="#4c78a8", alpha=0.82)
        ax.axvline(0, color="black", linewidth=1.1, linestyle="--")
        ax.set_title(BAND_LABELS[idx])
        ax.set_xlabel("SR aggregated - LR")
        ax.set_ylabel("Pixels")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{case.label}: per-band residual distributions")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{case.name}_residual_distributions.png", dpi=220)
    plt.close(fig)


def _residual_sample(
    lr_stack: np.ndarray,
    sr_stack: np.ndarray,
    valid_mask: np.ndarray,
    band_index: int,
    max_values: int,
    seed: int,
) -> np.ndarray:
    residual = sr_stack[band_index] - lr_stack[band_index]
    values = residual[valid_mask & np.isfinite(residual)]
    if values.size > max_values:
        rng = np.random.default_rng(seed)
        values = values[rng.choice(values.size, size=max_values, replace=False)]
    return values


def plot_scenario_residual_distributions(
    scenario_cases: Sequence[tuple[str, Sequence[SpectralCase]]],
    output_dir: Path,
    output_names: Sequence[str] = (
        "flood_residual_distributions.png",
        "flood_fire_residual_distributions.png",
    ),
    max_values_per_scenario: int = 250_000,
) -> None:
    """Overlay per-band residual distributions for publication-level scenario comparison."""

    scenario_band_values: list[list[np.ndarray]] = []
    for scenario_idx, (_, cases) in enumerate(scenario_cases):
        per_band_parts: list[list[np.ndarray]] = [[] for _ in BAND_LABELS]
        case_limit = max(1, max_values_per_scenario // max(1, len(cases)))
        for case_idx, case in enumerate(cases):
            lr_stack, sr_stack, _, stack_valid_mask, _ = _load_case_arrays(case)
            for band_idx in range(len(BAND_LABELS)):
                per_band_parts[band_idx].append(
                    _residual_sample(
                        lr_stack,
                        sr_stack,
                        stack_valid_mask,
                        band_idx,
                        max_values=case_limit,
                        seed=(scenario_idx + 1) * 1000 + (case_idx + 1) * 100 + band_idx,
                    )
                )

        scenario_values = []
        for band_idx, parts in enumerate(per_band_parts):
            values = np.concatenate([part for part in parts if part.size])
            if values.size > max_values_per_scenario:
                rng = np.random.default_rng((scenario_idx + 1) * 10_000 + band_idx)
                values = values[
                    rng.choice(values.size, size=max_values_per_scenario, replace=False)
                ]
            scenario_values.append(values)
        scenario_band_values.append(scenario_values)

    colors = ("#4c78a8", "#f58518", "#54a24b", "#b279a2")
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.8), constrained_layout=True)
    for band_idx, ax in enumerate(axes.flat):
        combined = np.concatenate(
            [
                scenario_values[band_idx]
                for scenario_values in scenario_band_values
                if scenario_values[band_idx].size
            ]
        )
        lo, hi = np.percentile(combined, [1, 99]) if combined.size else (-0.01, 0.01)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -0.01, 0.01
        bins = np.linspace(lo, hi, 101)

        for scenario_idx, ((label, _), scenario_values) in enumerate(
            zip(scenario_cases, scenario_band_values)
        ):
            values = scenario_values[band_idx]
            if not values.size:
                continue
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="stepfilled",
                color=colors[scenario_idx % len(colors)],
                edgecolor=colors[scenario_idx % len(colors)],
                linewidth=1.0,
                alpha=0.36,
                label=label,
            )

        ax.axvline(0, color="black", linewidth=1.1, linestyle="--")
        ax.set_title(BAND_LABELS[band_idx])
        ax.set_xlabel("SR aggregated - LR reflectance")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.2)
        if band_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Flood and fire: per-band residual distributions")
    output_dir.mkdir(parents=True, exist_ok=True)
    for output_name in output_names:
        fig.savefig(output_dir / output_name, dpi=220)
    plt.close(fig)


def _plot_spectra_panel(
    case: SpectralCase,
    lr_stack: np.ndarray,
    sr_stack: np.ndarray,
    valid_mask: np.ndarray,
    target_mask: np.ndarray | None,
    output_dir: Path,
) -> None:
    strata: list[tuple[str, np.ndarray]] = [("All valid pixels", valid_mask)]
    if target_mask is not None:
        strata.append(("Target mask", valid_mask & target_mask))
        strata.append(("Background", valid_mask & ~target_mask))

    fig, axes = plt.subplots(
        1, len(strata), figsize=(5.2 * len(strata), 4.3), constrained_layout=True
    )
    if len(strata) == 1:
        axes = [axes]

    x = np.arange(len(BAND_LABELS))
    for ax, (title, mask) in zip(axes, strata):
        if not np.any(mask):
            ax.set_title(f"{title}\n(no pixels)")
            continue

        lr_stats = _spectral_stats(lr_stack, mask)
        sr_stats = _spectral_stats(sr_stack, mask)
        ax.fill_between(x, lr_stats["p25"], lr_stats["p75"], color="#1f77b4", alpha=0.18)
        ax.fill_between(x, sr_stats["p25"], sr_stats["p75"], color="#ff7f0e", alpha=0.18)
        ax.plot(x, lr_stats["mean"], marker="o", color="#1f77b4", label="LR")
        ax.plot(x, sr_stats["mean"], marker="o", color="#ff7f0e", label="SR aggregated")
        ax.set_title(title)
        ax.set_xticks(x, BAND_LABELS, rotation=45)
        ax.set_ylabel("Reflectance")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(f"{case.label}: mean spectra with IQR")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{case.name}_mean_spectra.png", dpi=220)
    plt.close(fig)


def _write_case_metrics(
    path: Path,
    band_metrics: Iterable[BandMetric],
    spectral_angle: SpectralAngleSummary,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
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
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for metric in band_metrics:
            writer.writerow(
                {
                    "case": metric.case,
                    "metric_type": "band",
                    "band_index": metric.band_index,
                    "band_label": metric.band_label,
                    "wavelength_nm": metric.wavelength_nm,
                    "valid_pixels": metric.valid_pixels,
                    "mae": metric.mae,
                    "bias": metric.bias,
                    "median_error": metric.median_error,
                    "rmse": metric.rmse,
                    "pearson_r": metric.pearson_r,
                    "slope": metric.slope,
                    "intercept": metric.intercept,
                }
            )
        writer.writerow(
            {
                "case": spectral_angle.case,
                "metric_type": "spectral_angle",
                "valid_pixels": spectral_angle.valid_pixels,
                "sam_mean_deg": spectral_angle.mean_deg,
                "sam_median_deg": spectral_angle.median_deg,
                "sam_p25_deg": spectral_angle.p25_deg,
                "sam_p75_deg": spectral_angle.p75_deg,
            }
        )


def _load_case_arrays(
    case: SpectralCase,
    band_labels: tuple[str, ...] = BAND_LABELS,
    reflectance_scale: float = REFLECTANCE_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with rasterio.open(case.lr_path) as lr_src, rasterio.open(case.sr_path) as sr_src:
        if lr_src.count != sr_src.count:
            raise ValueError(
                f"Band count mismatch for {case.name}: LR={lr_src.count}, SR={sr_src.count}"
            )
        if lr_src.count != len(band_labels):
            raise ValueError(
                f"{case.name} has {lr_src.count} bands but {len(band_labels)} labels were provided."
            )

        valid_mask = _mask_on_lr_grid(case.valid_mask_path, lr_src)
        if valid_mask is None:
            valid_mask = np.ones((lr_src.height, lr_src.width), dtype=bool)
        target_mask = _mask_on_lr_grid(case.target_mask_path, lr_src)

        lr_bands = []
        sr_bands = []
        lr_nodata = lr_src.nodata

        for band_idx in range(1, lr_src.count + 1):
            lr_band = lr_src.read(band_idx).astype("float32") / reflectance_scale
            if lr_nodata is not None:
                lr_band[lr_band == lr_nodata / reflectance_scale] = np.nan

            sr_band = _aggregate_band_to_lr_grid(sr_src, lr_src, band_idx) / reflectance_scale
            if sr_src.nodata is not None:
                sr_band[sr_band == sr_src.nodata / reflectance_scale] = np.nan

            lr_bands.append(lr_band)
            sr_bands.append(sr_band)

    lr_stack = np.stack(lr_bands)
    sr_stack = np.stack(sr_bands)
    stack_valid_mask = valid_mask & _base_valid_mask(lr_stack, sr_stack)
    return lr_stack, sr_stack, valid_mask, stack_valid_mask, target_mask


def run_spectral_case(
    case: SpectralCase,
    figures_dir: Path,
    band_labels: tuple[str, ...] = BAND_LABELS,
    wavelengths_nm: tuple[int, ...] = BAND_WAVELENGTHS_NM,
    reflectance_scale: float = REFLECTANCE_SCALE,
) -> tuple[list[BandMetric], SpectralAngleSummary]:
    """Run all-band spectral validation for one LR/SR image pair."""

    lr_stack, sr_stack, valid_mask, stack_valid_mask, target_mask = _load_case_arrays(
        case, band_labels=band_labels, reflectance_scale=reflectance_scale
    )
    band_metrics = []
    for band_idx, (band_label, wavelength) in enumerate(
        zip(band_labels, wavelengths_nm), start=1
    ):
        metrics = compute_band_metrics(
            lr_stack[band_idx - 1], sr_stack[band_idx - 1], valid_mask=valid_mask
        )
        band_metrics.append(
            BandMetric(
                case=case.name,
                band_index=band_idx,
                band_label=band_label,
                wavelength_nm=wavelength,
                valid_pixels=int(metrics["valid_pixels"]),
                mae=float(metrics["mae"]),
                bias=float(metrics["bias"]),
                median_error=float(metrics["median_error"]),
                rmse=float(metrics["rmse"]),
                pearson_r=float(metrics["pearson_r"]),
                slope=float(metrics["slope"]),
                intercept=float(metrics["intercept"]),
            )
        )

    spectral_angle = compute_spectral_angle_summary(lr_stack, sr_stack, stack_valid_mask)
    spectral_angle = SpectralAngleSummary(
        case=case.name,
        valid_pixels=spectral_angle.valid_pixels,
        mean_deg=spectral_angle.mean_deg,
        median_deg=spectral_angle.median_deg,
        p25_deg=spectral_angle.p25_deg,
        p75_deg=spectral_angle.p75_deg,
    )

    _write_case_metrics(case.metrics_path, band_metrics, spectral_angle)
    _plot_equivalence_panel(case, lr_stack, sr_stack, stack_valid_mask, band_metrics, figures_dir)
    _plot_residual_panel(case, lr_stack, sr_stack, stack_valid_mask, figures_dir)
    _plot_spectra_panel(case, lr_stack, sr_stack, stack_valid_mask, target_mask, figures_dir)

    return band_metrics, spectral_angle


def write_summary_csv(
    output_path: Path,
    all_band_metrics: Iterable[BandMetric],
    spectral_angles: Iterable[SpectralAngleSummary],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
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
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for metric in all_band_metrics:
            writer.writerow(
                {
                    "case": metric.case,
                    "metric_type": "band",
                    "band_index": metric.band_index,
                    "band_label": metric.band_label,
                    "wavelength_nm": metric.wavelength_nm,
                    "valid_pixels": metric.valid_pixels,
                    "mae": metric.mae,
                    "bias": metric.bias,
                    "median_error": metric.median_error,
                    "rmse": metric.rmse,
                    "pearson_r": metric.pearson_r,
                    "slope": metric.slope,
                    "intercept": metric.intercept,
                }
            )
        for angle in spectral_angles:
            writer.writerow(
                {
                    "case": angle.case,
                    "metric_type": "spectral_angle",
                    "valid_pixels": angle.valid_pixels,
                    "sam_mean_deg": angle.mean_deg,
                    "sam_median_deg": angle.median_deg,
                    "sam_p25_deg": angle.p25_deg,
                    "sam_p75_deg": angle.p75_deg,
                }
            )


def plot_mae_bias_summary(all_band_metrics: list[BandMetric], output_dir: Path) -> None:
    cases = list(dict.fromkeys(metric.case for metric in all_band_metrics))
    mae = np.full((len(cases), len(BAND_LABELS)), np.nan, dtype="float32")
    bias = np.full_like(mae, np.nan)

    case_index = {case: idx for idx, case in enumerate(cases)}
    for metric in all_band_metrics:
        row = case_index[metric.case]
        col = metric.band_index - 1
        mae[row, col] = metric.mae
        bias[row, col] = metric.bias

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    for ax, data, title, cmap in (
        (axes[0], mae, "L1 / MAE", "magma"),
        (axes[1], bias, "Signed bias (SR aggregated - LR)", "coolwarm"),
    ):
        if title.startswith("Signed"):
            vmax = float(np.nanmax(np.abs(data)))
            vmin = -vmax
        else:
            vmin = 0.0
            vmax = float(np.nanmax(data))
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(BAND_LABELS)), BAND_LABELS, rotation=45)
        ax.set_yticks(np.arange(len(cases)), cases)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                ax.text(
                    col,
                    row,
                    f"{data[row, col]:.4f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(data[row, col]) > (vmax * 0.55 if vmax else 0) else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "all_band_mae_bias_summary.png", dpi=240)
    plt.close(fig)


def write_manuscript_framing(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        """# Spectral Consistency Validation: Manuscript Framing

## Methods addition
To evaluate whether super-resolution preserves spectral information, each SR image was aggregated from 2.5 m to the native 10 m grid of the corresponding LR Sentinel-2 stack using area-average resampling. The original LR image was then treated as the spectral reference at equivalent spatial support. For each of the ten available Sentinel-2 bands (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12), we computed L1 error/MAE, signed bias, median error, RMSE, Pearson correlation, and the slope/intercept of the LR-vs-SR equivalence fit. We also summarized per-pixel spectral angle across all bands.

## Results framing
These spectral metrics separate radiometric preservation from downstream thematic performance. The LR-vs-SR equivalence plots and bias spectra show whether the SR model introduces systematic band-wise shifts after returning the SR image to the same spatial support as the LR reference. The existing MNDWI/dNBR and mask-based results should be presented as downstream task validation after this all-band spectral consistency assessment.

## Limitation statement
The present analysis strengthens the spectral assessment for the flood and burn-scar case studies included in this repository. Because no additional independent scenes are added here, the results should not be described as demonstrating broad generalization across a wider population of Sentinel-2 images.
""",
        encoding="utf-8",
    )

