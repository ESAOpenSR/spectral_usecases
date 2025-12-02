import os
import numpy as np
import rasterio

def compute_dnbr_for_prefix(
    prefix,
    in_dir=".",
    out_dir=".",
    nir_band=7,   # B08 (1-based index in rasterio)
    swir_band=10  # B12 (1-based index in rasterio)
):
    """
    Compute NBR_pre, NBR_post and masked dNBR for multi-band Sentinel-2 stacks.

    Expected files in base_dir:
      {prefix}_pre.tif   -> multiband S2 stack, bands: B02..B12
      {prefix}_post.tif  -> same structure as pre
      {prefix}_mask.tif  -> binary mask (1 = area of interest, 0 = outside)

    Outputs:
      {prefix}_nbr_pre.tif
      {prefix}_nbr_post.tif
      {prefix}_dnbr_masked.tif
    """
    # Resolve paths
    def p_in(name: str) -> str:
        return os.path.join(in_dir, name)
    def p_out(name: str) -> str:
        return os.path.join(out_dir, name)

    pre_path  = p_in(f"{prefix}_pre.tif")
    post_path = p_in(f"{prefix}_post.tif")
    mask_path = p_in(f"{prefix}_mask.tif")

    out_nbr_pre   = p_out(f"{prefix}_nbr_pre.tif")
    out_nbr_post  = p_out(f"{prefix}_nbr_post.tif")
    out_dnbr_mask = p_out(f"{prefix}_dnbr_masked.tif")

    # --- 1. Open inputs ---
    with rasterio.open(pre_path) as src_pre, \
         rasterio.open(post_path) as src_post, \
         rasterio.open(mask_path) as src_mask:

        # Read NIR and SWIR bands (1-based band indices)
        nir_pre  = src_pre.read(nir_band).astype("float32")
        swir_pre = src_pre.read(swir_band).astype("float32")
        nir_post = src_post.read(nir_band).astype("float32")
        swir_post= src_post.read(swir_band).astype("float32")

        mask = src_mask.read(1).astype("uint8")

        meta = src_pre.meta.copy()
        nodata_in = src_pre.nodata
        if nodata_in is None:
            nodata_in = -9999.0  # define a nodata if missing

    # --- 2. Build valid masks (inside AOI + avoid nodata & division by zero) ---
    denom_pre  = nir_pre + swir_pre
    denom_post = nir_post + swir_post

    valid_pre = (mask == 1)
    valid_post = (mask == 1)

    valid_pre &= (nir_pre != nodata_in) & (swir_pre != nodata_in) & (denom_pre != 0)
    valid_post &= (nir_post != nodata_in) & (swir_post != nodata_in) & (denom_post != 0)

    # --- 3. Compute NBR_pre and NBR_post ---
    nbr_pre  = np.full_like(nir_pre, nodata_in, dtype="float32")
    nbr_post = np.full_like(nir_post, nodata_in, dtype="float32")

    nbr_pre[valid_pre]   = (nir_pre[valid_pre]  - swir_pre[valid_pre])  / denom_pre[valid_pre]
    nbr_post[valid_post] = (nir_post[valid_post] - swir_post[valid_post]) / denom_post[valid_post]

    # --- 4. Compute dNBR (pre - post) only where both NBRs valid ---
    valid_dnbr = valid_pre & valid_post

    dnbr = np.full_like(nir_pre, nodata_in, dtype="float32")
    dnbr[valid_dnbr] = nbr_pre[valid_dnbr] - nbr_post[valid_dnbr]

    # --- 5. Write outputs ---
    meta_out = meta.copy()
    meta_out.update({
        "dtype": "float32",
        "count": 1,
        "nodata": nodata_in
    })

    with rasterio.open(out_nbr_pre, "w", **meta_out) as dst:
        dst.write(nbr_pre, 1)

    with rasterio.open(out_nbr_post, "w", **meta_out) as dst:
        dst.write(nbr_post, 1)

    with rasterio.open(out_dnbr_mask, "w", **meta_out) as dst:
        dst.write(dnbr, 1)

    print(f"[{prefix}] wrote:")
    print(" ", out_nbr_pre)
    print(" ", out_nbr_post)
    print(" ", out_dnbr_mask)



# Example usage:
# compute_dnbr_for_prefix("lr", base_dir="path/to/data")
# compute_dnbr_for_prefix("sr", base_dir="path/to/data")
if __name__ == "__main__":
    # Example calls (uncomment and set base_dir as needed)
    compute_dnbr_for_prefix("lr", in_dir="data/raster_data", out_dir="data/products")
    compute_dnbr_for_prefix("sr", in_dir="data/raster_data", out_dir="data/products")