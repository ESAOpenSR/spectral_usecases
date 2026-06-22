import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_DIR.parent
DATA_DIR = WORKFLOW_DIR / "data_fire"
RASTER_DIR = DATA_DIR / "raster_data"
PRODUCTS_DIR = DATA_DIR / "products"
OUTPUT_DIR = DATA_DIR / "metrics" / "edge_validation"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.edge_validation import EdgeCase, run_edge_validation  # noqa: E402


CASES = (
    EdgeCase(
        name="fire",
        label="Burn scar",
        lr_reflectance_path=RASTER_DIR / "lr_after.tif",
        sr_reflectance_path=RASTER_DIR / "sr_after.tif",
        lr_index_path=PRODUCTS_DIR / "lr_dnbr.tif",
        sr_index_path=PRODUCTS_DIR / "sr_dnbr.tif",
        lr_detection_path=PRODUCTS_DIR / "lr_detections.tif",
        sr_detection_path=PRODUCTS_DIR / "sr_detections.tif",
        gt_mask_path=RASTER_DIR / "fire_mask.tif",
        index_name="dNBR",
        target_direction="high",
        threshold=None,
        valid_mask_path=PRODUCTS_DIR / "valid_land_mask.tif",
    ),
)


def main() -> None:
    run_edge_validation(CASES, OUTPUT_DIR)
    print(f"Edge validation outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
