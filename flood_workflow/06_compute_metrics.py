import sys
import os
from pathlib import Path


REPO_ROOT = "/data1/simon/GitHub/spectral_usecases/"
DATA_DIR = os.path.join(REPO_ROOT, "data_flood")
RASTER_DIR = os.path.join(DATA_DIR, "raster_data")
PRODUCTS_DIR = os.path.join(DATA_DIR, "products")

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.metrics import compute_detection_metrics, print_pretty_table, write_metrics_csv


def compute_metrics(
    lr_mndwi_path,
    sr_mndwi_path,
    lr_det_path,
    sr_det_path,
    gt_path,
    high_thr=0.2,
):
    return compute_detection_metrics(
        lr_signal_path=lr_mndwi_path,
        sr_signal_path=sr_mndwi_path,
        lr_det_path=lr_det_path,
        sr_det_path=sr_det_path,
        gt_path=gt_path,
        high_thr=high_thr,
    )


if __name__ == "__main__":
    base_prods = PRODUCTS_DIR
    base_data = RASTER_DIR

    m = compute_metrics(
        lr_mndwi_path=os.path.join(base_prods, "lr_mndwi.tif"),
        sr_mndwi_path=os.path.join(base_prods, "sr_mndwi.tif"),
        lr_det_path=os.path.join(base_prods, "lr_detections.tif"),
        sr_det_path=os.path.join(base_prods, "sr_detections.tif"),
        gt_path=os.path.join(base_data, "flood_mask.tif"),
        high_thr=0.2,
    )

    print_pretty_table(m, title="Flood Metrics", spectral_name="MNDWI")

    csv_path = Path("metrics/flood_metrics.csv")
    write_metrics_csv(csv_path, m, spectral_name="MNDWI")
    print(f"CSV saved to {csv_path}")

