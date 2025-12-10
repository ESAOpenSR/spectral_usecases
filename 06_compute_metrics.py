import os
import csv
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, binary_erosion
from rasterio.warp import reproject
from rasterio.enums import Resampling


def _reproject_mask_to_target(mask_arr, src_transform, src_crs, dst_shape, dst_transform, dst_crs):
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


def compute_metrics(
    lr_mndwi_path,
    sr_mndwi_path,
    lr_det_path,
    sr_det_path,
    gt_path,
    high_thr=0.2,
):
    # --- Open all rasters with geoinfo ---
    lr_mndwi_src = rasterio.open(lr_mndwi_path)
    sr_mndwi_src = rasterio.open(sr_mndwi_path)
    lr_det_src = rasterio.open(lr_det_path)
    sr_det_src = rasterio.open(sr_det_path)
    gt_src = rasterio.open(gt_path)

    lr = lr_mndwi_src.read(1).astype("float32")
    sr = sr_mndwi_src.read(1).astype("float32")
    det_LR_raw = lr_det_src.read(1).astype("uint8")
    det_SR_raw = sr_det_src.read(1).astype("uint8")
    gt = gt_src.read(1).astype("uint8")

    lr_nod = lr_mndwi_src.nodata if lr_mndwi_src.nodata is not None else -9999
    sr_nod = sr_mndwi_src.nodata if sr_mndwi_src.nodata is not None else -9999

    # Binarise detections (defensive)
    det_LR = (det_LR_raw > 0).astype("uint8")
    det_SR = (det_SR_raw > 0).astype("uint8")

    # --- Reproject GT to LR grid (for LR spectral stats) ---
    if (gt.shape != lr.shape or gt_src.transform != lr_mndwi_src.transform or gt_src.crs != lr_mndwi_src.crs):
        gt_lr = _reproject_mask_to_target(
            gt,
            src_transform=gt_src.transform,
            src_crs=gt_src.crs,
            dst_shape=lr.shape,
            dst_transform=lr_mndwi_src.transform,
            dst_crs=lr_mndwi_src.crs,
        )
    else:
        gt_lr = gt.copy()

    # --- Reproject GT to SR grid (for SR spectral stats + edge metrics) ---
    if (gt.shape != sr.shape or gt_src.transform != sr_mndwi_src.transform or gt_src.crs != sr_mndwi_src.crs):
        gt_sr = _reproject_mask_to_target(
            gt,
            src_transform=gt_src.transform,
            src_crs=gt_src.crs,
            dst_shape=sr.shape,
            dst_transform=sr_mndwi_src.transform,
            dst_crs=sr_mndwi_src.crs,
        )
    else:
        gt_sr = gt.copy()

    # --- Upsample LR detections to SR grid (NN) ---
    if (det_LR.shape != det_SR.shape or lr_det_src.transform != sr_det_src.transform or lr_det_src.crs != sr_det_src.crs):
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

    # Close datasets
    lr_mndwi_src.close()
    sr_mndwi_src.close()
    lr_det_src.close()
    sr_det_src.close()
    gt_src.close()

    # --- Nodata handling for MNDWI ---
    lr[lr == lr_nod] = np.nan
    sr[sr == sr_nod] = np.nan

    # --- 1. Detected pixels on SR grid ---
    N_LR = int(np.nansum(det_LR_sr.astype(np.int64)))   # LR detections upsampled to SR grid
    N_SR = int(np.nansum(det_SR.astype(np.int64)))

    rel_change = (float(N_SR) - float(N_LR)) / max(float(N_LR), 1.0)

    # --- 2. Median MNDWI inside GT flood extent (native grids) ---
    median_LR = np.nanmedian(lr[gt_lr == 1])
    median_SR = np.nanmedian(sr[gt_sr == 1])

    # --- 3. High-confidence fraction (MNDWI ≥ high_thr) within GT ---
    high_LR = np.nanmean((lr[gt_lr == 1] >= high_thr).astype("float32"))
    high_SR = np.nanmean((sr[gt_sr == 1] >= high_thr).astype("float32"))
    high_rel_change = (high_SR - high_LR) / max(high_LR, 1e-9)

    # --- 4. Edge-region gain (single SR-grid edge band) ---
    se = np.ones((5, 5))
    dil_sr = binary_dilation(gt_sr, structure=se)
    ero_sr = binary_erosion(gt_sr, structure=se)
    edge_sr = (dil_sr.astype("uint8") - ero_sr.astype("uint8")) == 1

    LR_edge_detected = np.nansum(det_LR_sr[edge_sr])
    SR_edge_detected = np.nansum(det_SR[edge_sr])

    if LR_edge_detected == 0:
        edge_gain = np.nan
    else:
        edge_gain = (SR_edge_detected - LR_edge_detected) / LR_edge_detected

    return {
        "N_LR": N_LR,
        "N_SR": N_SR,
        "rel_change": rel_change,
        "median_LR": median_LR,
        "median_SR": median_SR,
        "high_LR": high_LR,
        "high_SR": high_SR,
        "high_rel_change": high_rel_change,
        "edge_gain": edge_gain,
    }


def print_pretty_table(metrics):
    """Metrics dict → nicely formatted console table."""
    print("\n================ Flood Metrics ================\n")
    print(f"{'Metric':<25} {'LR':>15} {'SR':>15} {'Δ vs LR':>15}")
    print("-" * 70)

    print(f"{'Detected Pixels':<25} {metrics['N_LR']:>15,.0f} {metrics['N_SR']:>15,.0f} {metrics['rel_change']*100:>14.2f}%")
    print(f"{'Median MNDWI':<25} {metrics['median_LR']:>15.4f} {metrics['median_SR']:>15.4f} {'--':>15}")
    print(
        f"{'High-Conf. Fraction':<25} "
        f"{metrics['high_LR']:>15.4f} "
        f"{metrics['high_SR']:>15.4f} "
        f"{metrics['high_rel_change']*100:>14.2f}%"
    )
    print(f"{'Edge-Region Gain':<25} {'--':>15} {'--':>15} {metrics['edge_gain']*100:>14.2f}%")

    print("\n================================================\n")


if __name__ == "__main__":
    base_prods = "data_flood/products/"
    base_data = "data_flood/raster_data/"

    m = compute_metrics(
        lr_mndwi_path=base_prods + "lr_mndwi.tif",
        sr_mndwi_path=base_prods + "sr_mndwi.tif",
        lr_det_path=base_prods + "lr_detections.tif",
        sr_det_path=base_prods + "sr_detections.tif",
        gt_path=base_data + "flood_mask.tif",
        high_thr=0.2,
    )

    print_pretty_table(m)

    # --- Save metrics to CSV ---
    os.makedirs("metrics", exist_ok=True)
    csv_path = "metrics/flood_metrics.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "LR", "SR", "Relative Change"])

        writer.writerow(["Detected Pixels", m["N_LR"], m["N_SR"], m["rel_change"]])
        writer.writerow(["Median MNDWI", m["median_LR"], m["median_SR"], ""])
        writer.writerow(["High-Conf Fraction", m["high_LR"], m["high_SR"], m["high_rel_change"]])
        writer.writerow(["Edge-Region Gain", "", "", m["edge_gain"]])

    print(f"CSV saved to {csv_path}")

