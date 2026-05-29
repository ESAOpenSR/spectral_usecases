from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "metrics" / "graphs"
HIST_DIR = ROOT / "metrics" / "histograms"

LR_COLOR = "#1f77b4"
SR_COLOR = "#ff7f0e"
THRESHOLD_COLOR = "black"


CASES = (
    {
        "name": "flood",
        "metric": "MNDWI",
        "overall_title": "Flood MNDWI: overall LR vs SR values",
        "edge_title": "Flood edge MNDWI: LR vs SR edge-region values",
        "panel_overall_title": "Flood overall MNDWI",
        "panel_edge_title": "Flood edge-region MNDWI",
        "lr_path": ROOT / "flood_workflow" / "data_flood" / "products" / "lr_mndwi.tif",
        "sr_path": ROOT / "flood_workflow" / "data_flood" / "products" / "sr_mndwi.tif",
        "gt_path": ROOT / "flood_workflow" / "data_flood" / "raster_data" / "flood_mask.tif",
        "valid_mask_path": None,
        "overall_output": GRAPH_DIR / "mndwi_histograms.png",
        "edge_output": HIST_DIR / "flood_edge_mndwi_histograms.png",
    },
    {
        "name": "fire",
        "metric": "dNBR",
        "overall_title": "Fire dNBR: overall LR vs SR values",
        "edge_title": "Fire edge dNBR: LR vs SR edge-region values",
        "panel_overall_title": "Fire overall dNBR",
        "panel_edge_title": "Fire edge-region dNBR",
        "lr_path": ROOT / "fire_workflow" / "data_fire" / "products" / "lr_dnbr.tif",
        "sr_path": ROOT / "fire_workflow" / "data_fire" / "products" / "sr_dnbr.tif",
        "gt_path": ROOT / "fire_workflow" / "data_fire" / "raster_data" / "fire_mask.tif",
        "valid_mask_path": ROOT / "fire_workflow" / "data_fire" / "products" / "valid_land_mask.tif",
        "overall_output": GRAPH_DIR / "dnbr_histograms.png",
        "edge_output": HIST_DIR / "fire_edge_dnbr_histograms.png",
    },
)


OVERALL_PANEL_OUTPUT_PNG = HIST_DIR / "lr_sr_overall_histogram_panel.png"
EDGE_PANEL_OUTPUT_PNG = HIST_DIR / "lr_sr_edge_histogram_panel.png"


def reproject_mask(mask_path, dst_shape, dst_transform, dst_crs):
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype("uint8")

        if mask.shape == dst_shape and src.transform == dst_transform and src.crs == dst_crs:
            return mask

        dst = np.zeros(dst_shape, dtype="uint8")
        reproject(
            source=mask,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        return dst


def raster_values(path, valid_mask_path=None):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata if src.nodata is not None else -9999.0
        valid = np.isfinite(arr) & (arr != nodata)

        if valid_mask_path is not None:
            valid_mask = reproject_mask(valid_mask_path, arr.shape, src.transform, src.crs)
            valid &= valid_mask == 1

    return arr[valid]


def otsu_threshold(values_a, values_b, bins=200):
    finite_values = [vals[np.isfinite(vals)] for vals in (values_a, values_b) if vals.size > 0]
    if not finite_values:
        raise ValueError("No valid raster values available for thresholding.")

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

    mu0 = np.cumsum(prob * bin_mids)
    mu_total = mu0[-1]
    mu0 = mu0 / np.where(w0 == 0, 1, w0)
    mu1 = (mu_total - mu0 * w0) / np.where(w1 == 0, 1, w1)

    sigma_between = w0 * w1 * (mu0 - mu1) ** 2
    sigma_between[~valid] = -np.inf

    return float(bin_mids[np.argmax(sigma_between)])


def reproject_metric_to_reference(metric_path, dst_shape, dst_transform, dst_crs):
    with rasterio.open(metric_path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata if src.nodata is not None else -9999.0
        dst = np.full(dst_shape, nodata, dtype="float32")

        reproject(
            source=arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=nodata,
            dst_nodata=nodata,
            resampling=Resampling.bilinear,
        )

    return dst, nodata


def edge_values_on_sr_grid(lr_path, sr_path, gt_path, valid_mask_path=None, edge_size=5):
    with rasterio.open(sr_path) as sr_src:
        sr = sr_src.read(1).astype("float32")
        sr_nodata = sr_src.nodata if sr_src.nodata is not None else -9999.0
        dst_shape = sr.shape
        dst_transform = sr_src.transform
        dst_crs = sr_src.crs

    lr, lr_nodata = reproject_metric_to_reference(
        lr_path,
        dst_shape,
        dst_transform,
        dst_crs,
    )

    valid_lr = np.isfinite(lr) & (lr != lr_nodata)
    valid_sr = np.isfinite(sr) & (sr != sr_nodata)

    if valid_mask_path is not None:
        valid_mask = reproject_mask(valid_mask_path, dst_shape, dst_transform, dst_crs) == 1
        valid_lr &= valid_mask
        valid_sr &= valid_mask

    gt = reproject_mask(gt_path, dst_shape, dst_transform, dst_crs) == 1
    structure = np.ones((edge_size, edge_size), dtype="uint8")
    edge = binary_dilation(gt, structure=structure) ^ binary_erosion(gt, structure=structure)

    return lr[valid_lr & edge], sr[valid_sr & edge]


def robust_range(values, threshold):
    finite = np.concatenate([vals[np.isfinite(vals)] for vals in values if vals.size > 0])
    if finite.size == 0:
        raise ValueError("No finite values available for plotting.")

    lo, hi = np.percentile(finite, [0.5, 99.5])
    lo = min(float(lo), threshold)
    hi = max(float(hi), threshold)
    pad = (hi - lo) * 0.08 if hi > lo else 0.1
    return lo - pad, hi + pad


def smooth_histogram(hist, sigma=2.0):
    smoothed = gaussian_filter1d(hist.astype("float64"), sigma=sigma, mode="nearest")
    if smoothed.sum() > 0:
        smoothed *= hist.sum() / smoothed.sum()
    return smoothed


def build_distribution(lr_values, sr_values, threshold, bins=120, x_range=None):
    if x_range is None:
        x_range = robust_range([lr_values, sr_values], threshold)
    lr_over = int(np.sum(lr_values >= threshold))
    sr_over = int(np.sum(sr_values >= threshold))
    lr_rate = lr_over / lr_values.size if lr_values.size else np.nan
    sr_rate = sr_over / sr_values.size if sr_values.size else np.nan

    lr_weights = np.full(lr_values.shape, 100.0 / lr_values.size)
    sr_weights = np.full(sr_values.shape, 100.0 / sr_values.size)
    bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    lr_hist, _ = np.histogram(lr_values, bins=bin_edges, weights=lr_weights)
    sr_hist, _ = np.histogram(sr_values, bins=bin_edges, weights=sr_weights)

    return {
        "x": bin_centers,
        "lr_curve": smooth_histogram(lr_hist),
        "sr_curve": smooth_histogram(sr_hist),
        "x_range": x_range,
        "threshold": threshold,
        "lr_rate": lr_rate,
        "sr_rate": sr_rate,
    }


def draw_distribution(
    ax,
    dist,
    title,
    metric,
    lr_label="LR",
    sr_label="SR",
    panel_label=None,
    show_inset=True,
):
    sr_above_lr = (dist["x"] >= dist["threshold"]) & (dist["sr_curve"] > dist["lr_curve"])
    ax.fill_between(
        dist["x"],
        dist["lr_curve"],
        dist["sr_curve"],
        where=sr_above_lr,
        interpolate=True,
        color="#d62728",
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        dist["x"],
        dist["lr_curve"],
        color=LR_COLOR,
        linewidth=2.0,
        label=lr_label,
        zorder=3,
    )
    ax.plot(
        dist["x"],
        dist["sr_curve"],
        color=SR_COLOR,
        linewidth=2.0,
        label=sr_label,
        zorder=3,
    )
    ax.axvline(
        dist["threshold"],
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Detection threshold",
    )
    ax.set_title(title)
    ax.set_xlabel(f"{metric} value")
    ax.set_ylabel("Pixels per bin (%)")
    ax.set_xlim(dist["x_range"])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    if show_inset:
        inset = ax.inset_axes([0.69, 0.62, 0.25, 0.28])
        rates = [dist["lr_rate"] * 100.0, dist["sr_rate"] * 100.0]
        inset.bar([0, 1], rates, color=[LR_COLOR, SR_COLOR])
        inset.set_xticks([0, 1], ["LR", "SR"])
        inset.tick_params(axis="both", labelsize=8)
        inset.grid(axis="y", alpha=0.25)
        ymax = max(rates)
        inset.set_ylim(0, ymax * 1.25 if ymax > 0 else 1)

        for idx, value in enumerate(rates):
            inset.text(idx, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

    if panel_label is not None:
        ax.text(
            -0.12,
            1.08,
            panel_label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
        )


def save_single_plot(dist, title, metric, output, show_inset=True):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    draw_distribution(ax, dist, title, metric, lr_label="LR", sr_label="SR", show_inset=show_inset)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_split_panels(panel_data):
    overall_specs = [
        (panel_data["flood"]["overall"], "A", "Flood overall MNDWI", "MNDWI"),
        (panel_data["fire"]["overall"], "B", "Fire overall dNBR", "dNBR"),
    ]
    edge_specs = [
        (panel_data["flood"]["edge"], "A", "Flood edge-region MNDWI", "MNDWI"),
        (panel_data["fire"]["edge"], "B", "Fire edge-region dNBR", "dNBR"),
    ]

    HIST_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)
    for ax, (dist, label, title, metric) in zip(axes, overall_specs):
        draw_distribution(
            ax,
            dist,
            title,
            metric,
            lr_label="LR",
            sr_label="SR",
            panel_label=label,
            show_inset=False,
        )
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        ax.tick_params(axis="both", labelsize=8)
        ax.legend(loc="upper left", fontsize=8)
    fig.savefig(OVERALL_PANEL_OUTPUT_PNG, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)
    for ax, (dist, label, title, metric) in zip(axes, edge_specs):
        draw_distribution(
            ax,
            dist,
            title,
            metric,
            lr_label="LR",
            sr_label="SR",
            panel_label=label,
            show_inset=True,
        )
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        ax.tick_params(axis="both", labelsize=8)
        ax.legend(loc="upper left", fontsize=8)
    fig.savefig(EDGE_PANEL_OUTPUT_PNG, dpi=300)
    plt.close(fig)


def build_case_figures(case):
    lr_all = raster_values(case["lr_path"], case["valid_mask_path"])
    sr_all = raster_values(case["sr_path"], case["valid_mask_path"])
    threshold = otsu_threshold(lr_all, sr_all)

    overall_dist = build_distribution(lr_all, sr_all, threshold, x_range=(-1.0, 1.0))
    save_single_plot(
        overall_dist,
        case["overall_title"],
        case["metric"],
        case["overall_output"],
        show_inset=False,
    )

    del lr_all, sr_all

    lr_edge, sr_edge = edge_values_on_sr_grid(
        case["lr_path"],
        case["sr_path"],
        case["gt_path"],
        valid_mask_path=case["valid_mask_path"],
    )
    edge_dist = build_distribution(lr_edge, sr_edge, threshold)
    save_single_plot(edge_dist, case["edge_title"], case["metric"], case["edge_output"], show_inset=True)

    print(
        f"{case['name']}: threshold={threshold:.6f}, "
        f"overall wrote {case['overall_output']}, edge wrote {case['edge_output']}"
    )

    return {"overall": overall_dist, "edge": edge_dist}


def main():
    panel_data = {}
    for case in CASES:
        panel_data[case["name"]] = build_case_figures(case)

    save_split_panels(panel_data)
    print(f"overall panel wrote {OVERALL_PANEL_OUTPUT_PNG}")
    print(f"edge panel wrote {EDGE_PANEL_OUTPUT_PNG}")


if __name__ == "__main__":
    main()
