import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_DIR.parent
DATA_DIR = WORKFLOW_DIR / "data_flood"
RASTER_DIR = DATA_DIR / "raster_data"
METRICS_DIR = DATA_DIR / "metrics" / "spectral_validation"
FIGURES_DIR = DATA_DIR / "graph_outputs" / "spectral_validation"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.spectral_validation import (  # noqa: E402
    SpectralCase,
    plot_mae_bias_summary,
    run_spectral_case,
    write_summary_csv,
)


CASES = (
    SpectralCase(
        name="flood",
        label="Flood",
        lr_path=RASTER_DIR / "lr.tif",
        sr_path=RASTER_DIR / "sr.tif",
        metrics_path=METRICS_DIR / "flood_spectral_metrics.csv",
        target_mask_path=RASTER_DIR / "flood_mask.tif",
    ),
)


def main() -> None:
    all_band_metrics = []
    spectral_angles = []

    for case in CASES:
        print(f"Running spectral validation for {case.label}...")
        band_metrics, spectral_angle = run_spectral_case(case, FIGURES_DIR)
        all_band_metrics.extend(band_metrics)
        spectral_angles.append(spectral_angle)
        print(f"  wrote {case.metrics_path}")

    summary_path = METRICS_DIR / "flood_spectral_validation_summary.csv"
    write_summary_csv(summary_path, all_band_metrics, spectral_angles)
    plot_mae_bias_summary(all_band_metrics, FIGURES_DIR)

    print(f"Summary CSV saved to {summary_path}")
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
