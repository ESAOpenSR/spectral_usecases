import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data_fire"
RASTER_DIR = DATA_DIR / "raster_data"
PRODUCTS_DIR = DATA_DIR / "products"

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.metrics import compute_detection_metrics, print_pretty_table, write_metrics_csv

def compute_metrics(
    lr_dnbr_path,
    sr_dnbr_path,
    lr_det_path,
    sr_det_path,
    gt_path,
    high_thr=0.5,
):
    return compute_detection_metrics(
        lr_signal_path=lr_dnbr_path,
        sr_signal_path=sr_dnbr_path,
        lr_det_path=lr_det_path,
        sr_det_path=sr_det_path,
        gt_path=gt_path,
        high_thr=high_thr,
    )
if __name__ == "__main__":
    base_prods = PRODUCTS_DIR
    base_data = RASTER_DIR

    m = compute_metrics(
        lr_dnbr_path=base_prods / "lr_dnbr.tif",
        sr_dnbr_path=base_prods / "sr_dnbr.tif",
        lr_det_path=base_prods / "lr_detections.tif",
        sr_det_path=base_prods / "sr_detections.tif",
        gt_path=base_data / "fire_mask.tif",
        high_thr=0.5,
    )

    print_pretty_table(m, title="Burn-Scar Metrics", spectral_name="dNBR")

    csv_path = Path("metrics/burnscar_metrics.csv")
    write_metrics_csv(csv_path, m, spectral_name="dNBR")
    print(f"CSV saved to {csv_path}")