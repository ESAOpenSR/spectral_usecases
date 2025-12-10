import os
import numpy as np
import rasterio


def compute_dmndwi_for_prefix(
    prefix,
    in_dir=".",
    out_dir=".",
    green_band=2,  # B03 (1-based index in rasterio, 1..n)
    swir_band=9,   # B11 (1-based index in rasterio, 1..n)
):
    """
    Compute MNDWI_pre, MNDWI_post and dMNDWI for Sentinel-2 stacks.

    Expected files in in_dir:
      {prefix}_before.tif -> multiband S2 stack, bands: B02..B12
      {prefix}_after.tif  -> same structure as _before

    Outputs in out_dir:
      {prefix}_mndwi_pre.tif
      {prefix}_mndwi_post.tif
      {prefix}_dmndwi.tif
    """

    # Path helpers
    def p_in(name: str) -> str:
        return os.path.join(in_dir, name)

    def p_out(name: str) -> str:
        return os.path.join(out_dir, name)

    pre_path  = p_in(f"{prefix}_before.tif")
    post_path = p_in(f"{prefix}_after.tif")

    out_mndwi_pre  = p_out(f"{prefix}_mndwi_pre.tif")
    out_mndwi_post = p_out(f"{prefix}_mndwi_post.tif")
    out_dmndwi     = p_out(f"{prefix}_dmndwi.tif")

    # --- 1. Open inputs ---
    with rasterio.open(pre_path) as src_pre, \
         rasterio.open(post_path) as src_post:

        green_pre = src_pre.read(green_band).astype("float32")
        swir_pre  = src_pre.read(swir_band).astype("float32")
        green_post = src_post.read(green_band).astype("float32")
        swir_post  = src_post.read(swir_band).astype("float32")

        meta = src_pre.meta.copy()
        nodata_in = src_pre.nodata
        if nodata_in is None:
            nodata_in = -9999.0  # define a nodata if missing

    # --- 2. Valid masks (avoid nodata & division by zero) ---
    denom_pre  = green_pre + swir_pre
    denom_post = green_post + swir_post

    valid_pre = (
        (green_pre != nodata_in) &
        (swir_pre != nodata_in) &
        (denom_pre != 0)
    )

    valid_post = (
        (green_post != nodata_in) &
        (swir_post != nodata_in) &
        (denom_post != 0)
    )

    # --- 3. Compute MNDWI_pre and MNDWI_post ---
    mndwi_pre  = np.full_like(green_pre, nodata_in, dtype="float32")
    mndwi_post = np.full_like(green_post, nodata_in, dtype="float32")

    mndwi_pre[valid_pre]   = (green_pre[valid_pre] - swir_pre[valid_pre]) / denom_pre[valid_pre]
    mndwi_post[valid_post] = (green_post[valid_post] - swir_post[valid_post]) / denom_post[valid_post]

    # --- 4. Compute dMNDWI (post - pre) where both valid ---
    valid_dmndwi = valid_pre & valid_post

    dmndwi = np.full_like(green_pre, nodata_in, dtype="float32")
    dmndwi[valid_dmndwi] = mndwi_post[valid_dmndwi] - mndwi_pre[valid_dmndwi]

    # --- 5. Write outputs ---
    meta_out = meta.copy()
    meta_out.update({
        "dtype": "float32",
        "count": 1,
        "nodata": nodata_in
    })

    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(out_mndwi_pre, "w", **meta_out) as dst:
        dst.write(mndwi_pre, 1)

    with rasterio.open(out_mndwi_post, "w", **meta_out) as dst:
        dst.write(mndwi_post, 1)

    with rasterio.open(out_dmndwi, "w", **meta_out) as dst:
        dst.write(dmndwi, 1)

    print(f"[{prefix}] wrote:")
    print(" ", out_mndwi_pre)
    print(" ", out_mndwi_post)
    print(" ", out_dmndwi)


if __name__ == "__main__":
    compute_dmndwi_for_prefix("lr", in_dir="data_flood/raster_data", out_dir="data_flood/products")
    compute_dmndwi_for_prefix("sr", in_dir="data_flood/raster_data", out_dir="data_flood/products")

