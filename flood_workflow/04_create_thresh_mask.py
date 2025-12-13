from pathlib import Path

import numpy as np
import rasterio

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_flood"
RASTER_DIR = DATA_DIR / "raster_data"
PRODUCTS_DIR = DATA_DIR / "products"


def make_water_mask(
    input_tif,
    green_band=2,   # B03 (1-based index)
    swir_band=9,    # B11 (1-based index)
    thr=None,       # If None: Otsu threshold on MNDWI
    out_tif=None,
):
    """
    Build a WATER mask from a multiband Sentinel-2 TIFF.

    Mask semantics:
        1 = water (flood or standing water)
        0 = non-water

    Internally uses MNDWI (green - swir) / (green + swir)
    and splits water vs non-water using a threshold.
    """

    def otsu_threshold(values, nbins=256):
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("No valid values for Otsu thresholding.")
        hist, bin_edges = np.histogram(values, bins=nbins)
        hist = hist.astype("float64")
        prob = hist / hist.sum()
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        w0 = np.cumsum(prob)
        w1 = 1.0 - w0

        mu0 = np.cumsum(prob * bin_mids)
        muT = mu0[-1]
        mu0 = mu0 / np.where(w0 == 0, 1, w0)
        mu1 = (muT - mu0 * w0) / np.where(w1 == 0, 1, w1)

        sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2
        sigma_b2[(w0 == 0) | (w1 == 0)] = -np.inf

        return float(bin_mids[np.argmax(sigma_b2)])

    # --- Load input ---
    with rasterio.open(input_tif) as src:
        green = src.read(green_band).astype("float32")
        swir  = src.read(swir_band).astype("float32")
        meta  = src.meta.copy()
        nod   = src.nodata if src.nodata is not None else -9999.0

    # --- Compute MNDWI ---
    denom = green + swir
    valid_px = (green != nod) & (swir != nod) & (denom != 0)

    mndwi = np.full_like(green, np.nan, dtype="float32")
    mndwi[valid_px] = (green[valid_px] - swir[valid_px]) / denom[valid_px]

    # --- Determine threshold ---
    if thr is None:
        thr = otsu_threshold(mndwi[np.isfinite(mndwi)])
        print(f"[make_water_mask] Otsu MNDWI threshold = {thr:.4f}")

    # --- Build WATER mask ---
    # Here, higher MNDWI = water
    water_mask = np.zeros_like(green, dtype="uint8")
    water_mask[(mndwi >= thr) & np.isfinite(mndwi)] = 1   # <-- 1 = water

    # --- Save output optionally ---
    if out_tif is not None:
        meta.update({"dtype": "uint8", "count": 1, "nodata": 0})
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(water_mask, 1)

    return water_mask, mndwi, thr


if __name__ == "__main__":
    input_tif = RASTER_DIR / "lr.tif"
    output_tif = PRODUCTS_DIR / "water_mask.tif"

    water_mask, _, thr = make_water_mask(
        input_tif,
        green_band=2,
        swir_band=9,
        thr=None,      # optional manual threshold
        out_tif=output_tif
    )

    print("Water mask saved to:", output_tif)
