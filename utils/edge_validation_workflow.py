from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.edge_validation import EdgeCase, run_edge_validation


FLOOD_DATA_DIR = REPO_ROOT / "flood_workflow" / "data_flood"
FIRE_DATA_DIR = REPO_ROOT / "fire_workflow" / "data_fire"

FLOOD_CASE = EdgeCase(
    name="flood",
    label="Flood",
    lr_reflectance_path=FLOOD_DATA_DIR / "raster_data" / "lr.tif",
    sr_reflectance_path=FLOOD_DATA_DIR / "raster_data" / "sr.tif",
    lr_index_path=FLOOD_DATA_DIR / "products" / "lr_mndwi.tif",
    sr_index_path=FLOOD_DATA_DIR / "products" / "sr_mndwi.tif",
    lr_detection_path=FLOOD_DATA_DIR / "products" / "lr_detections.tif",
    sr_detection_path=FLOOD_DATA_DIR / "products" / "sr_detections.tif",
    gt_mask_path=FLOOD_DATA_DIR / "raster_data" / "flood_mask.tif",
    index_name="MNDWI",
    target_direction="high",
    threshold=None,
)

FIRE_CASE = EdgeCase(
    name="fire",
    label="Burn scar",
    lr_reflectance_path=FIRE_DATA_DIR / "raster_data" / "lr_after.tif",
    sr_reflectance_path=FIRE_DATA_DIR / "raster_data" / "sr_after.tif",
    lr_index_path=FIRE_DATA_DIR / "products" / "lr_dnbr.tif",
    sr_index_path=FIRE_DATA_DIR / "products" / "sr_dnbr.tif",
    lr_detection_path=FIRE_DATA_DIR / "products" / "lr_detections.tif",
    sr_detection_path=FIRE_DATA_DIR / "products" / "sr_detections.tif",
    gt_mask_path=FIRE_DATA_DIR / "raster_data" / "fire_mask.tif",
    index_name="dNBR",
    target_direction="high",
    threshold=None,
    valid_mask_path=FIRE_DATA_DIR / "products" / "valid_land_mask.tif",
)


def main() -> None:
    for case, data_dir in (
        (FLOOD_CASE, FLOOD_DATA_DIR),
        (FIRE_CASE, FIRE_DATA_DIR),
    ):
        output_dir = data_dir / "metrics" / "edge_validation"
        run_edge_validation((case,), output_dir)
        print(f"{case.label} edge validation outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
