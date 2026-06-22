from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.spectral_validation import (  # noqa: E402
    BAND_LABELS,
    BAND_WAVELENGTHS_NM,
    REFLECTANCE_SCALE,
    BandMetric,
    SpectralCase,
    _aggregate_band_to_lr_grid,
    _base_valid_mask,
    _mask_on_lr_grid,
    compute_band_metrics,
)


PAPER_READY_DIR = REPO_ROOT / "metrics" / "spectral_validation" / "paper_ready"
SHARED_REFLECTANCE_EXTENT = (0.0, 0.6, 0.0, 0.6)
HEXBIN_GRIDSIZE = 65
HEXBIN_CMAP = plt.get_cmap("viridis").copy()


CASES = (
    SpectralCase(
        name="fire_pre",
        label="Fire",
        lr_path=REPO_ROOT / "fire_workflow" / "data_fire" / "raster_data" / "lr_before.tif",
        sr_path=REPO_ROOT / "fire_workflow" / "data_fire" / "raster_data" / "sr_before.tif",
        metrics_path=REPO_ROOT
        / "fire_workflow"
        / "data_fire"
        / "metrics"
        / "spectral_validation"
        / "fire_pre_spectral_metrics.csv",
        valid_mask_path=REPO_ROOT
        / "fire_workflow"
        / "data_fire"
        / "products"
        / "valid_land_mask.tif",
    ),
    SpectralCase(
        name="flood",
        label="Flood",
        lr_path=REPO_ROOT / "flood_workflow" / "data_flood" / "raster_data" / "lr.tif",
        sr_path=REPO_ROOT / "flood_workflow" / "data_flood" / "raster_data" / "sr.tif",
        metrics_path=REPO_ROOT
        / "flood_workflow"
        / "data_flood"
        / "metrics"
        / "spectral_validation"
        / "flood_spectral_metrics.csv",
    ),
)


@dataclass(frozen=True)
class CaseArrays:
    case: SpectralCase
    lr_stack: np.ndarray
    sr_stack: np.ndarray
    valid_mask: np.ndarray
    band_metrics: list[BandMetric]


def load_case_arrays(
    case: SpectralCase,
    band_labels: tuple[str, ...] = BAND_LABELS,
    wavelengths_nm: tuple[int, ...] = BAND_WAVELENGTHS_NM,
    reflectance_scale: float = REFLECTANCE_SCALE,
) -> CaseArrays:
    with rasterio.open(case.lr_path) as lr_src, rasterio.open(case.sr_path) as sr_src:
        valid_mask = _mask_on_lr_grid(case.valid_mask_path, lr_src)
        if valid_mask is None:
            valid_mask = np.ones((lr_src.height, lr_src.width), dtype=bool)

        lr_bands = []
        sr_bands = []
        band_metrics = []
        lr_nodata = lr_src.nodata

        for band_idx, (band_label, wavelength) in enumerate(
            zip(band_labels, wavelengths_nm), start=1
        ):
            lr_band = lr_src.read(band_idx).astype("float32") / reflectance_scale
            if lr_nodata is not None:
                lr_band[lr_band == lr_nodata / reflectance_scale] = np.nan

            sr_band = _aggregate_band_to_lr_grid(sr_src, lr_src, band_idx) / reflectance_scale
            if sr_src.nodata is not None:
                sr_band[sr_band == sr_src.nodata / reflectance_scale] = np.nan

            metrics = compute_band_metrics(lr_band, sr_band, valid_mask=valid_mask)
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
            lr_bands.append(lr_band)
            sr_bands.append(sr_band)

    lr_stack = np.stack(lr_bands)
    sr_stack = np.stack(sr_bands)
    stack_valid_mask = valid_mask & _base_valid_mask(lr_stack, sr_stack)
    return CaseArrays(case, lr_stack, sr_stack, stack_valid_mask, band_metrics)


def _plot_percent_hexbin(
    ax: plt.Axes,
    lr: np.ndarray,
    sr: np.ndarray,
    valid: np.ndarray,
    metric: BandMetric,
    shared_extent: tuple[float, float, float, float],
):
    x = lr[valid]
    y = sr[valid]
    if x.size == 0:
        ax.set_title(metric.band_label)
        ax.text(0.5, 0.5, "No valid pixels", ha="center", va="center", transform=ax.transAxes)
        return None

    weights = np.full(x.shape, 100.0 / x.size, dtype="float32")
    hb = ax.hexbin(
        x,
        y,
        C=weights,
        extent=shared_extent,
        reduce_C_function=np.sum,
        gridsize=HEXBIN_GRIDSIZE,
        mincnt=1,
        cmap=HEXBIN_CMAP,
    )

    lo, hi = shared_extent[0], shared_extent[1]

    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, label="1:1")
    if np.isfinite(metric.slope) and np.isfinite(metric.intercept):
        ax.plot(
            [lo, hi],
            [metric.slope * lo + metric.intercept, metric.slope * hi + metric.intercept],
            color="#d62728",
            linewidth=1.0,
            label="fit",
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(metric.band_label, fontsize=9)
    ax.grid(True, alpha=0.22, linewidth=0.5)
    ax.tick_params(labelsize=7, length=2.5)
    return hb


def plot_stacked_percent_equivalence(cases: tuple[SpectralCase, ...], output_dir: Path) -> None:
    loaded_cases = [load_case_arrays(case) for case in cases]
    shared_extent = SHARED_REFLECTANCE_EXTENT

    fig, axes = plt.subplots(
        len(loaded_cases) * 2,
        5,
        figsize=(15, 12),
        sharex=False,
        sharey=False,
    )
    hexbin_artists = []

    for case_idx, case_arrays in enumerate(loaded_cases):
        row_offset = case_idx * 2
        case_axes = axes[row_offset : row_offset + 2, :].ravel()
        for band_idx, ax in enumerate(case_axes):
            lr = case_arrays.lr_stack[band_idx]
            sr = case_arrays.sr_stack[band_idx]
            valid = case_arrays.valid_mask & np.isfinite(lr) & np.isfinite(sr)
            hb = _plot_percent_hexbin(
                ax,
                lr,
                sr,
                valid,
                case_arrays.band_metrics[band_idx],
                shared_extent,
            )
            if hb is not None:
                hexbin_artists.append(hb)

            if band_idx == 0:
                ax.legend(fontsize=7, loc="upper left", frameon=False)
            if band_idx % 5 == 0:
                ax.set_ylabel("SR aggregated reflectance", fontsize=8)
            if band_idx >= 5:
                ax.set_xlabel("LR reflectance", fontsize=8)

    positive_values = np.concatenate(
        [artist.get_array()[artist.get_array() > 0] for artist in hexbin_artists]
    )
    norm = LogNorm(vmin=float(np.min(positive_values)), vmax=float(np.max(positive_values)))
    for artist in hexbin_artists:
        artist.set_norm(norm)

    fig.subplots_adjust(left=0.075, right=0.89, bottom=0.07, top=0.93, hspace=0.42, wspace=0.32)
    fig.text(0.5, 0.975, "LR vs SR Spectral Equivalence", ha="center", va="top", fontsize=15)

    case_label_y = [0.735, 0.285]
    for y, case_arrays in zip(case_label_y, loaded_cases):
        fig.text(
            0.018,
            y,
            case_arrays.case.label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    cbar_ax = fig.add_axes([0.915, 0.18, 0.018, 0.64])
    mappable = ScalarMappable(norm=norm, cmap=HEXBIN_CMAP)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, cax=cbar_ax)
    colorbar.set_label("Percentage of valid pixels per hexbin", fontsize=9)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    colorbar.ax.tick_params(labelsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fire_flood_lr_sr_equivalence_percent.png"
    pdf_path = output_dir / "fire_flood_lr_sr_equivalence_percent.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    plot_stacked_percent_equivalence(CASES, PAPER_READY_DIR)
    print(f"Wrote {PAPER_READY_DIR / 'fire_flood_lr_sr_equivalence_percent.png'}")
    print(f"Wrote {PAPER_READY_DIR / 'fire_flood_lr_sr_equivalence_percent.pdf'}")


if __name__ == "__main__":
    main()
