from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from rasterio.windows import Window, bounds, from_bounds

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "metrics" / "a4_examples"
REFLECTANCE_SCALE = 10000.0

FIGURE_INCHES = (7.2, 11.69)
COL_TITLES = ("LR RGB", "SR RGB", "LR R-SWIR", "SR R-SWIR", "LR/SR mask")

BANDS = {
    "B02": 1,
    "B03": 2,
    "B04": 3,
    "B05": 4,
    "B06": 5,
    "B07": 6,
    "B08": 7,
    "B8A": 8,
    "B11": 9,
    "B12": 10,
}
RGB_BANDS = (BANDS["B04"], BANDS["B03"], BANDS["B02"])
RSWIR_BANDS = (BANDS["B12"], BANDS["B8A"], BANDS["B04"])

MASK_COLORS = {
    "background": "#f7f7f2",
    "agreement": "#8dbcdb",
    "lr_only": "#ffbc77",
    "sr_only": "#ff787c",
}
MASK_CMAP = ListedColormap(
    [
        MASK_COLORS["background"],
        MASK_COLORS["agreement"],
        MASK_COLORS["lr_only"],
        MASK_COLORS["sr_only"],
    ]
)
MASK_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], MASK_CMAP.N)


@dataclass(frozen=True)
class ExampleCase:
    key: str
    label: str
    lr_image_path: Path
    sr_image_path: Path
    lr_detection_path: Path
    sr_detection_path: Path
    aoi_path: Path


@dataclass(frozen=True)
class CropExample:
    case: ExampleCase
    row_off: int
    col_off: int
    size: int
    agreement_pixels: int
    lr_only_pixels: int
    sr_only_pixels: int


CASES = {
    "flood": ExampleCase(
        key="flood",
        label="Flood",
        lr_image_path=REPO_ROOT
        / "flood_workflow"
        / "data_flood"
        / "raster_data"
        / "lr.tif",
        sr_image_path=REPO_ROOT
        / "flood_workflow"
        / "data_flood"
        / "raster_data"
        / "sr.tif",
        lr_detection_path=REPO_ROOT
        / "flood_workflow"
        / "data_flood"
        / "products"
        / "lr_detections.tif",
        sr_detection_path=REPO_ROOT
        / "flood_workflow"
        / "data_flood"
        / "products"
        / "sr_detections.tif",
        aoi_path=REPO_ROOT
        / "flood_workflow"
        / "data_flood"
        / "vector_data"
        / "flood_extent.geojson",
    ),
    "fire": ExampleCase(
        key="fire",
        label="Fire",
        lr_image_path=REPO_ROOT
        / "fire_workflow"
        / "data_fire"
        / "raster_data"
        / "lr_after.tif",
        sr_image_path=REPO_ROOT
        / "fire_workflow"
        / "data_fire"
        / "raster_data"
        / "sr_after.tif",
        lr_detection_path=REPO_ROOT
        / "fire_workflow"
        / "data_fire"
        / "products"
        / "lr_detections.tif",
        sr_detection_path=REPO_ROOT
        / "fire_workflow"
        / "data_fire"
        / "products"
        / "sr_detections.tif",
        aoi_path=REPO_ROOT / "fire_workflow" / "data_fire" / "AOI_fire.geojson",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an A4 LR/SR crop comparison figure from existing products."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASES),
        default=("flood", "fire"),
        help="Cases to include, in row order.",
    )
    parser.add_argument(
        "--examples-per-case",
        type=int,
        default=4,
        help="Number of crop rows selected for each case.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=128,
        help="Square crop size in SR-grid pixels.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Candidate crop stride in SR-grid pixels. Defaults to half the crop size.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the PNG output is written.",
    )
    parser.add_argument(
        "--basename",
        default="lr_sr_a4_examples",
        help="Output filename stem.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output PNG DPI.")
    return parser.parse_args()


def _integral_image(mask: np.ndarray) -> np.ndarray:
    summed = mask.astype("int64").cumsum(axis=0).cumsum(axis=1)
    return np.pad(summed, ((1, 0), (1, 0)), mode="constant")


def _window_sum(integral: np.ndarray, row_off: int, col_off: int, size: int) -> int:
    row_end = row_off + size
    col_end = col_off + size
    value = (
        integral[row_end, col_end]
        - integral[row_off, col_end]
        - integral[row_end, col_off]
        + integral[row_off, col_off]
    )
    return int(value)


def _candidate_offsets(length: int, crop_size: int, stride: int) -> list[int]:
    if crop_size > length:
        raise ValueError(f"Crop size {crop_size} exceeds raster dimension {length}.")

    offsets = list(range(0, length - crop_size + 1, stride))
    final_offset = length - crop_size
    if offsets[-1] != final_offset:
        offsets.append(final_offset)
    return offsets


def _windows_overlap(a: tuple[int, int], b: tuple[int, int], size: int) -> bool:
    row_a, col_a = a
    row_b, col_b = b
    return abs(row_a - row_b) < size and abs(col_a - col_b) < size


def _validate_detection_grids(case: ExampleCase, lr_src, sr_src) -> None:
    if lr_src.shape != sr_src.shape:
        raise ValueError(f"{case.label}: LR/SR detection shape mismatch.")
    if lr_src.crs != sr_src.crs or lr_src.transform != sr_src.transform:
        raise ValueError(f"{case.label}: LR/SR detections are not on the same grid.")


def _validate_image_grids(case: ExampleCase, sr_image_src, sr_detection_src) -> None:
    if sr_image_src.shape != sr_detection_src.shape:
        raise ValueError(
            f"{case.label}: SR image and detections have different shapes."
        )
    if (
        sr_image_src.crs != sr_detection_src.crs
        or sr_image_src.transform != sr_detection_src.transform
    ):
        raise ValueError(f"{case.label}: SR image and detections are not aligned.")


def _geojson_crs(data: dict) -> str:
    crs = data.get("crs")
    if not crs:
        return "EPSG:4326"
    name = crs.get("properties", {}).get("name")
    if not name or "CRS84" in name:
        return "EPSG:4326"
    return name


def _geojson_geometries(data: dict) -> list[dict]:
    if data.get("type") == "FeatureCollection":
        return [
            feature["geometry"]
            for feature in data.get("features", [])
            if feature.get("geometry") is not None
        ]
    if data.get("type") == "Feature":
        geometry = data.get("geometry")
        return [geometry] if geometry is not None else []
    if data.get("type") in {"Polygon", "MultiPolygon"}:
        return [data]
    raise ValueError(
        "AOI GeoJSON must be a FeatureCollection, Feature, Polygon, or MultiPolygon."
    )


def _aoi_mask_on_grid(case: ExampleCase, grid_src) -> np.ndarray:
    with case.aoi_path.open() as f:
        data = json.load(f)

    geometries = _geojson_geometries(data)
    if not geometries:
        raise ValueError(
            f"{case.label}: AOI file contains no geometries: {case.aoi_path}"
        )

    src_crs = _geojson_crs(data)
    transformed = [
        transform_geom(src_crs, grid_src.crs, geometry) for geometry in geometries
    ]
    return geometry_mask(
        transformed,
        out_shape=grid_src.shape,
        transform=grid_src.transform,
        invert=True,
    )


def select_crop_examples(
    case: ExampleCase,
    examples_per_case: int,
    crop_size: int,
    stride: int,
) -> list[CropExample]:
    if examples_per_case < 1:
        return []

    with rasterio.open(case.lr_detection_path) as lr_src, rasterio.open(
        case.sr_detection_path
    ) as sr_src:
        _validate_detection_grids(case, lr_src, sr_src)
        lr_mask = lr_src.read(1).astype(bool)
        sr_mask = sr_src.read(1).astype(bool)
        aoi_mask = _aoi_mask_on_grid(case, sr_src)

    agreement = lr_mask & sr_mask & aoi_mask
    lr_only = lr_mask & ~sr_mask & aoi_mask
    sr_only = sr_mask & ~lr_mask & aoi_mask

    aoi_integral = _integral_image(aoi_mask)
    agreement_integral = _integral_image(agreement)
    lr_only_integral = _integral_image(lr_only)
    sr_only_integral = _integral_image(sr_only)

    height, width = agreement.shape
    row_offsets = _candidate_offsets(height, crop_size, stride)
    col_offsets = _candidate_offsets(width, crop_size, stride)

    candidates: list[tuple[float, int, int, int, int, int, int, int]] = []
    for row_off in row_offsets:
        for col_off in col_offsets:
            if (
                _window_sum(aoi_integral, row_off, col_off, crop_size)
                != crop_size * crop_size
            ):
                continue

            agreement_count = _window_sum(
                agreement_integral, row_off, col_off, crop_size
            )
            lr_only_count = _window_sum(lr_only_integral, row_off, col_off, crop_size)
            sr_only_count = _window_sum(sr_only_integral, row_off, col_off, crop_size)
            target_count = agreement_count + lr_only_count + sr_only_count
            if target_count == 0:
                continue

            disagreement_count = lr_only_count + sr_only_count
            balance_count = min(lr_only_count, sr_only_count)
            score = target_count + 4.0 * disagreement_count + 8.0 * balance_count
            candidates.append(
                (
                    score,
                    balance_count,
                    disagreement_count,
                    agreement_count,
                    lr_only_count,
                    sr_only_count,
                    row_off,
                    col_off,
                )
            )

    candidates.sort(reverse=True)
    selected: list[CropExample] = []
    selected_offsets: list[tuple[int, int]] = []

    for (
        _,
        _,
        _,
        agreement_count,
        lr_only_count,
        sr_only_count,
        row_off,
        col_off,
    ) in candidates:
        offset = (row_off, col_off)
        if any(
            _windows_overlap(offset, chosen, crop_size) for chosen in selected_offsets
        ):
            continue
        selected.append(
            CropExample(
                case=case,
                row_off=row_off,
                col_off=col_off,
                size=crop_size,
                agreement_pixels=agreement_count,
                lr_only_pixels=lr_only_count,
                sr_only_pixels=sr_only_count,
            )
        )
        selected_offsets.append(offset)
        if len(selected) == examples_per_case:
            break

    if len(selected) < examples_per_case:
        already = {(example.row_off, example.col_off) for example in selected}
        for (
            _,
            _,
            _,
            agreement_count,
            lr_only_count,
            sr_only_count,
            row_off,
            col_off,
        ) in candidates:
            if (row_off, col_off) in already:
                continue
            selected.append(
                CropExample(
                    case=case,
                    row_off=row_off,
                    col_off=col_off,
                    size=crop_size,
                    agreement_pixels=agreement_count,
                    lr_only_pixels=lr_only_count,
                    sr_only_pixels=sr_only_count,
                )
            )
            if len(selected) == examples_per_case:
                break

    if len(selected) < examples_per_case:
        raise ValueError(
            f"{case.label}: only found {len(selected)} usable crops, "
            f"requested {examples_per_case}."
        )

    return selected


def _read_reflectance(
    src, bands: tuple[int, int, int], window: Window, out_size: int
) -> np.ndarray:
    data = src.read(
        bands,
        window=window,
        out_shape=(len(bands), out_size, out_size),
        resampling=Resampling.bilinear,
    ).astype("float32")
    if src.nodata is not None:
        data[data == src.nodata] = np.nan
    return data / REFLECTANCE_SCALE


def _read_sr_composite(src, bands: tuple[int, int, int], window: Window) -> np.ndarray:
    return _read_reflectance(src, bands, window, int(window.height))


def _read_lr_composite_on_sr_grid(
    lr_src,
    sr_src,
    bands: tuple[int, int, int],
    sr_window: Window,
) -> np.ndarray:
    left, bottom, right, top = bounds(sr_window, sr_src.transform)
    lr_window = from_bounds(left, bottom, right, top, transform=lr_src.transform)
    return _read_reflectance(lr_src, bands, lr_window, int(sr_window.height))


def _stretch_pair(
    lr_composite: np.ndarray,
    sr_composite: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> tuple[np.ndarray, np.ndarray]:
    lr_out = np.empty_like(lr_composite, dtype="float32")
    sr_out = np.empty_like(sr_composite, dtype="float32")

    for channel in range(lr_composite.shape[0]):
        values = np.concatenate(
            [lr_composite[channel].ravel(), sr_composite[channel].ravel()]
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = np.percentile(values, [lower_percentile, upper_percentile])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
            if hi <= lo:
                lo, hi = 0.0, 1.0

        lr_out[channel] = np.clip((lr_composite[channel] - lo) / (hi - lo), 0.0, 1.0)
        sr_out[channel] = np.clip((sr_composite[channel] - lo) / (hi - lo), 0.0, 1.0)

    return np.moveaxis(lr_out, 0, -1), np.moveaxis(sr_out, 0, -1)


def _read_mask_classes(
    lr_detection_src, sr_detection_src, window: Window
) -> np.ndarray:
    lr_mask = lr_detection_src.read(1, window=window).astype(bool)
    sr_mask = sr_detection_src.read(1, window=window).astype(bool)

    classes = np.zeros(lr_mask.shape, dtype="uint8")
    classes[lr_mask & sr_mask] = 1
    classes[lr_mask & ~sr_mask] = 2
    classes[sr_mask & ~lr_mask] = 3
    return classes


def _read_example_panels(
    example: CropExample,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    window = Window(example.col_off, example.row_off, example.size, example.size)

    with rasterio.open(example.case.lr_image_path) as lr_image_src, rasterio.open(
        example.case.sr_image_path
    ) as sr_image_src, rasterio.open(
        example.case.lr_detection_path
    ) as lr_detection_src, rasterio.open(
        example.case.sr_detection_path
    ) as sr_detection_src:
        _validate_detection_grids(example.case, lr_detection_src, sr_detection_src)
        _validate_image_grids(example.case, sr_image_src, sr_detection_src)

        lr_rgb = _read_lr_composite_on_sr_grid(
            lr_image_src, sr_image_src, RGB_BANDS, window
        )
        sr_rgb = _read_sr_composite(sr_image_src, RGB_BANDS, window)
        lr_rswir = _read_lr_composite_on_sr_grid(
            lr_image_src, sr_image_src, RSWIR_BANDS, window
        )
        sr_rswir = _read_sr_composite(sr_image_src, RSWIR_BANDS, window)
        mask_classes = _read_mask_classes(lr_detection_src, sr_detection_src, window)

    lr_rgb, sr_rgb = _stretch_pair(lr_rgb, sr_rgb)
    lr_rswir, sr_rswir = _stretch_pair(lr_rswir, sr_rswir)
    return lr_rgb, sr_rgb, lr_rswir, sr_rswir, mask_classes


def _scenario_groups(examples: list[CropExample]) -> list[tuple[int, int, str]]:
    groups: list[tuple[int, int, str]] = []
    start_idx = 0
    for row_idx in range(1, len(examples) + 1):
        if (
            row_idx == len(examples)
            or examples[row_idx].case.key != examples[start_idx].case.key
        ):
            groups.append((start_idx, row_idx - 1, examples[start_idx].case.label))
            start_idx = row_idx
    return groups


def _expand_gap_after_row(axes: np.ndarray, row_idx: int, gap_fraction: float) -> None:
    up_shift = gap_fraction / 2.0
    down_shift = gap_fraction / 2.0
    for idx in range(axes.shape[0]):
        shift = up_shift if idx <= row_idx else -down_shift
        for ax in axes[idx]:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 + shift, pos.width, pos.height])


def _add_scenario_labels_and_separators(
    fig: plt.Figure, axes: np.ndarray, examples: list[CropExample]
) -> None:
    groups = _scenario_groups(examples)
    for group_idx, (start_row, end_row, label) in enumerate(groups):
        top_pos = axes[start_row, 0].get_position()
        bottom_pos = axes[end_row, 0].get_position()
        y_mid = (top_pos.y1 + bottom_pos.y0) / 2.0
        fig.text(
            top_pos.x0 - 0.032,
            y_mid,
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=8.5,
            fontweight="bold",
        )

        if group_idx == 0 and len(groups) > 1:
            next_pos = axes[end_row + 1, 0].get_position()
            y_sep = (bottom_pos.y0 + next_pos.y1) / 2.0
            left = axes[0, 0].get_position().x0
            right = axes[0, -1].get_position().x1
            fig.add_artist(
                plt.Line2D(
                    [left, right],
                    [y_sep, y_sep],
                    transform=fig.transFigure,
                    color="black",
                    linewidth=1.0,
                )
            )


def plot_examples(
    examples: list[CropExample], output_dir: Path, basename: str, dpi: int
) -> None:
    if not examples:
        raise ValueError("No crop examples were selected.")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(examples),
        len(COL_TITLES),
        figsize=FIGURE_INCHES,
        dpi=dpi,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for row_idx, example in enumerate(examples):
        panels = _read_example_panels(example)

        for col_idx, (ax, panel) in enumerate(zip(axes[row_idx], panels)):
            if col_idx == len(COL_TITLES) - 1:
                ax.imshow(
                    panel, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest"
                )
            else:
                ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(COL_TITLES[col_idx], fontsize=7.5, pad=3)

    legend_handles = [
        Patch(facecolor=MASK_COLORS["agreement"], edgecolor="none", label="Agreement"),
        Patch(facecolor=MASK_COLORS["lr_only"], edgecolor="none", label="LR only"),
        Patch(facecolor=MASK_COLORS["sr_only"], edgecolor="none", label="SR only"),
        Patch(
            facecolor=MASK_COLORS["background"], edgecolor="#dddddd", label="Background"
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, 0.014),
        handlelength=1.0,
        handleheight=0.8,
    )
    fig.subplots_adjust(
        left=0.075, right=0.995, top=0.965, bottom=0.055, wspace=0.018, hspace=0.035
    )
    groups = _scenario_groups(examples)
    if len(groups) > 1:
        _expand_gap_after_row(axes, groups[0][1], gap_fraction=0.012)
    _add_scenario_labels_and_separators(fig, axes, examples)

    png_path = output_dir / f"{basename}.png"
    fig.savefig(png_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Wrote {png_path}")


def main() -> None:
    args = _parse_args()
    if args.examples_per_case < 1:
        raise ValueError("--examples-per-case must be at least 1.")
    if args.crop_size < 1:
        raise ValueError("--crop-size must be at least 1.")

    stride = args.stride if args.stride is not None else max(1, args.crop_size // 2)
    if stride < 1:
        raise ValueError("--stride must be at least 1.")

    examples: list[CropExample] = []
    for case_key in args.cases:
        case = CASES[case_key]
        examples.extend(
            select_crop_examples(
                case=case,
                examples_per_case=args.examples_per_case,
                crop_size=args.crop_size,
                stride=stride,
            )
        )

    plot_examples(examples, args.output_dir, args.basename, args.dpi)


if __name__ == "__main__":
    main()
