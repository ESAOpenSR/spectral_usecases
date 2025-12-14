import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data_flood"
RASTER_DIR = DATA_DIR / "raster_data"
PRODUCTS_DIR = DATA_DIR / "products"

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.metrics import (
    compute_detection_metrics,
    plot_distribution_separation,
    print_pretty_table,
    write_metrics_csv,
)


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
        return_samples=True,
    )


if __name__ == "__main__":
    base_prods = PRODUCTS_DIR
    base_data = RASTER_DIR

    m, samples = compute_metrics(
        lr_mndwi_path=base_prods / "lr_mndwi.tif",
        sr_mndwi_path=base_prods / "sr_mndwi.tif",
        lr_det_path=base_prods / "lr_detections.tif",
        sr_det_path=base_prods / "sr_detections.tif",
        gt_path=base_data / "flood_mask.tif",
        high_thr=0.2,
    )

    print_pretty_table(m, title="Flood Metrics", spectral_name="MNDWI")

    csv_path = Path("metrics/flood_metrics.csv")
    write_metrics_csv(csv_path, m, spectral_name="MNDWI")
    print(f"CSV saved to {csv_path}")

    plot_path = Path("metrics/graphs/mndwi_separation.png")
    plot_distribution_separation(samples, m, spectral_name="MNDWI", output_path=plot_path)
    print(f"Distribution separation plot saved to {plot_path}")

