import os
import numpy as np
import rasterio


def compute_mndwi(
    input_path,
    output_path,
    green_band=2,  # B03 (1-based index in rasterio)
    swir_band=9,   # B11 (1-based index in rasterio)
):
    """
    Compute the Modified Normalized Difference Water Index (MNDWI) for a
    single Sentinel-2 stack and write it to disk.

    Inputs
    ------
    input_path: path to a multiband Sentinel-2 TIFF (B02..B12 order assumed)
    output_path: destination GeoTIFF for the MNDWI layer
    green_band: 1-based band index for the green band (default: B03)
    swir_band: 1-based band index for the SWIR band (default: B11)
    """

    with rasterio.open(input_path) as src:
        green = src.read(green_band).astype("float32")
        swir = src.read(swir_band).astype("float32")
        meta = src.meta.copy()
        nodata_in = src.nodata if src.nodata is not None else -9999.0

    denom = green + swir
    valid = (green != nodata_in) & (swir != nodata_in) & (denom != 0)

    mndwi = np.full_like(green, nodata_in, dtype="float32")
    mndwi[valid] = (green[valid] - swir[valid]) / denom[valid]

    meta_out = meta.copy()
    meta_out.update({
        "dtype": "float32",
        "count": 1,
        "nodata": nodata_in,
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **meta_out) as dst:
        dst.write(mndwi, 1)

    print(f"Wrote MNDWI to {output_path}")


if __name__ == "__main__":
    compute_mndwi(
        input_path="data_flood/raster_data/lr.tif",
        output_path="data_flood/products/lr_mndwi.tif",
    )
    compute_mndwi(
        input_path="data_flood/raster_data/sr.tif",
        output_path="data_flood/products/sr_mndwi.tif",
    )
