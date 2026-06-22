import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_fire"
PRODUCTS_DIR = DATA_DIR / "products"
GRAPH_OUTPUTS_DIR = DATA_DIR / "graph_outputs"


def load_valid_pixels(path, mask_path=None):
    """
    Load raster values and return valid, non-nodata pixels.
    If mask_path is provided, mask is NN-resampled to the raster grid if needed,
    and only pixels where mask == 1 are returned.
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nod = src.nodata if src.nodata is not None else -9999.0
        raster_transform = src.transform
        raster_crs = src.crs
        raster_shape = arr.shape  # (height, width)

    mask = (arr != nod) & np.isfinite(arr)

    if mask_path is not None:
        with rasterio.open(mask_path) as msrc:
            m = msrc.read(1).astype("uint8")
            mask_transform = msrc.transform
            mask_crs = msrc.crs
            mask_shape = m.shape

        # If grid does not match, reproject/resample mask to raster grid (NN)
        if (mask_shape != raster_shape) or (mask_transform != raster_transform) or (mask_crs != raster_crs):
            m_resampled = np.zeros(raster_shape, dtype="uint8")
            reproject(
                source=m,
                destination=m_resampled,
                src_transform=mask_transform,
                src_crs=mask_crs,
                dst_transform=raster_transform,
                dst_crs=raster_crs,
                resampling=Resampling.nearest,
            )
            m = m_resampled

        mask &= (m == 1)

    return arr[mask]


def otsu_threshold(values, nbins=256):
    """Compute Otsu threshold from 1D distribution."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No valid values provided for thresholding.")

    hist, bin_edges = np.histogram(values, bins=nbins)
    hist = hist.astype("float64")
    prob = hist / hist.sum()

    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    w0 = np.cumsum(prob)
    w1 = 1.0 - w0
    valid = (w0 > 0) & (w1 > 0)

    mu0 = np.cumsum(prob * bin_mids)
    muT = mu0[-1]
    mu0 = mu0 / np.where(w0 == 0, 1, w0)
    mu1 = (muT - mu0 * w0) / np.where(w1 == 0, 1, w1)

    sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2
    sigma_b2[~valid] = -np.inf

    return float(bin_mids[np.argmax(sigma_b2)])


def plot_dnbr_histograms_with_threshold(lr_path, sr_path, lr_mask, sr_mask, bins=200, title="dNBR distribution"):
    """
    Plot LR & SR dNBR histograms using ONLY pixels where mask==1.
    Mask is automatically NN-resampled to each raster grid if needed.
    """
    lr_vals = load_valid_pixels(lr_path, lr_mask)
    sr_vals = load_valid_pixels(sr_path, sr_mask)

    all_vals = np.concatenate([lr_vals, sr_vals])
    thr = otsu_threshold(all_vals, nbins=bins)

    plt.figure(figsize=(10, 5))
    plt.hist(lr_vals, bins=bins, alpha=0.5, label="LR dNBR", density=True)
    plt.hist(sr_vals, bins=bins, alpha=0.5, label="SR dNBR", density=True)

    plt.axvline(thr, linestyle="--", linewidth=2, color="red", label=f"Threshold = {thr:.4f}")

    plt.xlabel("dNBR value")
    plt.ylabel("Density")
    plt.title(f"{title}\nOtsu threshold = {thr:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    GRAPH_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(GRAPH_OUTPUTS_DIR / "dnbr_histograms.png")
    plt.close()

    print(f"Otsu threshold (LR+SR combined, masked): {thr:.6f}")
    return thr


def write_detections(
    dnbr_path, out_path, threshold, reference_path=None, resampling=Resampling.bilinear
):
    """
    Threshold a dNBR raster and write binary detections to disk.

    If ``reference_path`` is provided, the dNBR is first resampled onto that
    grid (e.g., upsampling LR dNBR to the SR grid) before thresholding. This
    ensures that detections are generated on the target grid rather than being
    interpolated afterwards. Bilinear resampling is used by default for this
    upsampling step so LR/SR performance is more directly comparable.
    """
    with rasterio.open(dnbr_path) as src:
        arr = src.read(1).astype("float32")
        meta = src.meta.copy()
        nod = src.nodata if src.nodata is not None else -9999.0

    # Optionally resample the source dNBR onto a reference grid before
    # thresholding, so that LR detections are generated on the SR grid.
    if reference_path is not None:
        with rasterio.open(reference_path) as ref:
            dst_shape = (ref.height, ref.width)
            dst_transform = ref.transform
            dst_crs = ref.crs

        arr_resampled = np.full(dst_shape, nod, dtype="float32")

        reproject(
            source=arr,
            destination=arr_resampled,
            src_transform=meta["transform"],
            src_crs=meta["crs"],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=nod,
            dst_nodata=nod,
        )

        arr = arr_resampled
        meta.update({
            "transform": dst_transform,
            "crs": dst_crs,
            "height": dst_shape[0],
            "width": dst_shape[1],
        })

    det = np.zeros_like(arr, dtype="uint8")
    valid = (arr != nod) & np.isfinite(arr)
    det[valid & (arr >= threshold)] = 1

    meta.update({"dtype": "uint8", "count": 1, "nodata": 0})

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(det, 1)

    print(f"Detections written to {out_path}")

    

if __name__ == "__main__":
    base = PRODUCTS_DIR

    lr_dnbr = base / "lr_dnbr.tif"
    sr_dnbr = base / "sr_dnbr.tif"

    # these can now be LR- or SR-based; they will be resampled as needed
    lr_mask = base / "valid_land_mask.tif"
    sr_mask = base / "valid_land_mask.tif"  # e.g. reuse LR mask for SR

    threshold = plot_dnbr_histograms_with_threshold(
        lr_path=lr_dnbr,
        sr_path=sr_dnbr,
        lr_mask=lr_mask,
        sr_mask=sr_mask,
    )

    # --- write detections using that threshold ---
    write_detections(
        lr_dnbr,
        base / "lr_detections.tif",
        threshold,
        reference_path=sr_dnbr,  # upsample LR first so detection happens on SR grid
        resampling=Resampling.bilinear,  # explicit bilinear resampling for upsampling
    )
    write_detections(sr_dnbr, base / "sr_detections.tif", threshold)
