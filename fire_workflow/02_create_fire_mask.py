import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_fire")
RASTER_DIR = os.path.join(DATA_DIR, "raster_data")
VECTOR_DIR = os.path.join(DATA_DIR, "vector_data")

# --- INPUTS ---
raster_path = os.path.join(RASTER_DIR, "lr_after.tif")
vector_path = os.path.join(VECTOR_DIR, "palisades_fire_extent.gpkg")
out_path = os.path.join(RASTER_DIR, "fire_mask.tif")

# --- 1. Read raster (for shape, transform, CRS) ---
with rasterio.open(raster_path) as src:
    meta = src.meta.copy()
    out_shape = (src.height, src.width)
    transform = src.transform
    raster_crs = src.crs

# --- 2. Read shapefile and reproject to raster CRS ---
gdf = gpd.read_file(vector_path)
if gdf.crs != raster_crs:
    gdf = gdf.to_crs(raster_crs)

# --- 3. Rasterize polygon(s) to binary mask ---
# value 1 = inside polygon, 0 = outside
shapes = ((geom, 1) for geom in gdf.geometry)

mask = features.rasterize(
    shapes=shapes,
    out_shape=out_shape,
    transform=transform,
    fill=0,
    dtype="uint8"
)

# --- 4. Save mask as GeoTIFF aligned with original raster ---
meta.update({
    "dtype": "uint8",
    "count": 1,
    "nodata": 0
})

with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(mask, 1)

print("Done, mask written to:", out_path)