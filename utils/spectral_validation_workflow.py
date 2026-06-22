import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.spectral_validation import (
    SpectralCase,
    plot_mae_bias_summary,
    plot_scenario_residual_distributions,
    run_spectral_case,
    write_manuscript_framing,
    write_summary_csv,
)

FLOOD_DATA_DIR = REPO_ROOT / "flood_workflow" / "data_flood"
FIRE_DATA_DIR = REPO_ROOT / "fire_workflow" / "data_fire"
FLOOD_METRICS_DIR = FLOOD_DATA_DIR / "metrics" / "spectral_validation"
FIRE_METRICS_DIR = FIRE_DATA_DIR / "metrics" / "spectral_validation"
FLOOD_FIGURES_DIR = FLOOD_DATA_DIR / "graph_outputs" / "spectral_validation"
FIRE_FIGURES_DIR = FIRE_DATA_DIR / "graph_outputs" / "spectral_validation"
PAPER_READY_DIR = REPO_ROOT / "metrics" / "spectral_validation" / "paper_ready"


FLOOD_CASES = (
    SpectralCase(
        name="flood",
        label="Flood",
        lr_path=FLOOD_DATA_DIR / "raster_data" / "lr.tif",
        sr_path=FLOOD_DATA_DIR / "raster_data" / "sr.tif",
        metrics_path=FLOOD_METRICS_DIR / "flood_spectral_metrics.csv",
        target_mask_path=FLOOD_DATA_DIR / "raster_data" / "flood_mask.tif",
    ),
)

FIRE_CASES = (
    SpectralCase(
        name="fire_pre",
        label="Fire pre-event",
        lr_path=FIRE_DATA_DIR / "raster_data" / "lr_before.tif",
        sr_path=FIRE_DATA_DIR / "raster_data" / "sr_before.tif",
        metrics_path=FIRE_METRICS_DIR / "fire_pre_spectral_metrics.csv",
        target_mask_path=FIRE_DATA_DIR / "raster_data" / "fire_mask.tif",
        valid_mask_path=FIRE_DATA_DIR / "products" / "valid_land_mask.tif",
    ),
    SpectralCase(
        name="fire_post",
        label="Fire post-event",
        lr_path=FIRE_DATA_DIR / "raster_data" / "lr_after.tif",
        sr_path=FIRE_DATA_DIR / "raster_data" / "sr_after.tif",
        metrics_path=FIRE_METRICS_DIR / "fire_post_spectral_metrics.csv",
        target_mask_path=FIRE_DATA_DIR / "raster_data" / "fire_mask.tif",
        valid_mask_path=FIRE_DATA_DIR / "products" / "valid_land_mask.tif",
    ),
)


def run_case_group(cases, metrics_dir: Path, figures_dir: Path, summary_name: str):
    all_band_metrics = []
    spectral_angles = []

    for case in cases:
        print(f"Running spectral validation for {case.label}...")
        band_metrics, spectral_angle = run_spectral_case(case, figures_dir)
        all_band_metrics.extend(band_metrics)
        spectral_angles.append(spectral_angle)
        print(f"  wrote {case.metrics_path}")

    summary_path = metrics_dir / summary_name
    write_summary_csv(summary_path, all_band_metrics, spectral_angles)
    plot_mae_bias_summary(all_band_metrics, figures_dir)
    write_manuscript_framing(metrics_dir / "manuscript_framing.md")

    print(f"Summary CSV saved to {summary_path}")
    print(f"Figures saved to {figures_dir}")
    print(f"Manuscript framing saved to {metrics_dir / 'manuscript_framing.md'}")


def main() -> None:
    run_case_group(
        FLOOD_CASES,
        FLOOD_METRICS_DIR,
        FLOOD_FIGURES_DIR,
        "flood_spectral_validation_summary.csv",
    )
    run_case_group(
        FIRE_CASES,
        FIRE_METRICS_DIR,
        FIRE_FIGURES_DIR,
        "fire_spectral_validation_summary.csv",
    )

    plot_scenario_residual_distributions(
        (
            ("Flood", FLOOD_CASES),
            ("Fire (pre + post)", FIRE_CASES),
        ),
        PAPER_READY_DIR,
    )
    print(f"Paper-ready comparison figures saved to {PAPER_READY_DIR}")


if __name__ == "__main__":
    main()
