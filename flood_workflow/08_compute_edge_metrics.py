import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_DIR.parent
DATA_DIR = WORKFLOW_DIR / "data_flood"
RASTER_DIR = DATA_DIR / "raster_data"
PRODUCTS_DIR = DATA_DIR / "products"
OUTPUT_DIR = DATA_DIR / "metrics" / "edge_validation"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.edge_validation import EdgeCase, run_edge_validation  # noqa: E402


CASES = (
    EdgeCase(
        name="flood",
        label="Flood",
        lr_reflectance_path=RASTER_DIR / "lr.tif",
        sr_reflectance_path=RASTER_DIR / "sr.tif",
        lr_index_path=PRODUCTS_DIR / "lr_mndwi.tif",
        sr_index_path=PRODUCTS_DIR / "sr_mndwi.tif",
        lr_detection_path=PRODUCTS_DIR / "lr_detections.tif",
        sr_detection_path=PRODUCTS_DIR / "sr_detections.tif",
        gt_mask_path=RASTER_DIR / "flood_mask.tif",
        index_name="MNDWI",
        target_direction="high",
        threshold=None,
    ),
)


def main() -> None:
    run_edge_validation(CASES, OUTPUT_DIR)
    print(f"Edge validation outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
