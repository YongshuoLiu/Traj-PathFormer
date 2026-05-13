#!/usr/bin/env python3
"""Create a publication-style overview figure for the MSTD dataset.

The source file uses an Excel workbook container even though it has a .csv
extension. This script reads the workbook directly from XML so it does not
depend on optional Excel readers such as openpyxl.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mstd")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SOURCE_ORDER = ("bd", "AIS", "radar")
SOURCE_LABELS = {"bd": "BDS", "AIS": "AIS", "radar": "Radar"}
SOURCE_COLORS = {"bd": "#0072B2", "AIS": "#D55E00", "radar": "#009E73"}


@dataclass(frozen=True)
class TrajectoryPoint:
    lon: float
    lat: float
    speed: float | None
    course: float | None
    time: datetime | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the MSTD workbook-like file. If omitted, the script searches workdir/data.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("paper/figures/mstd_dataset_overview"),
        help="Output path without extension.",
    )
    parser.add_argument("--dpi", type=int, default=350, help="PNG resolution.")
    return parser.parse_args()


def infer_input_path() -> Path:
    candidates = sorted(Path("workdir/data").glob("6h*4.8*(1).csv"))
    if not candidates:
        candidates = sorted(Path("workdir/data").glob("*4.8*.csv"))
    if not candidates:
        raise FileNotFoundError("Could not find the MSTD source file under workdir/data.")
    return candidates[0]


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root.findall("main:si", NS):
        texts = [node.text or "" for node in si.findall(".//main:t", NS)]
        strings.append("".join(texts))
    return strings


def excel_col_to_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    idx = 0
    for letter in letters:
        idx = idx * 26 + ord(letter) - ord("A") + 1
    return idx - 1


def cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("main:v", NS)
    if cell_type == "s":
        if value_node is None or value_node.text is None:
            return ""
        return shared_strings[int(value_node.text)]
    if cell_type == "inlineStr":
        texts = [node.text or "" for node in cell.findall(".//main:t", NS)]
        return "".join(texts)
    return "" if value_node is None or value_node.text is None else value_node.text


def read_first_sheet_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = read_shared_strings(zf)
        sheet_path = "xl/worksheets/sheet1.xml"
        if sheet_path not in zf.namelist():
            raise FileNotFoundError(f"{sheet_path} not found in workbook.")
        root = ET.fromstring(zf.read(sheet_path))

    rows: list[list[str]] = []
    for row in root.findall(".//main:sheetData/main:row", NS):
        values: dict[int, str] = {}
        max_idx = -1
        for cell in row.findall("main:c", NS):
            idx = excel_col_to_index(cell.attrib.get("r", "A1"))
            values[idx] = cell_text(cell, shared_strings)
            max_idx = max(max_idx, idx)
        rows.append([values.get(i, "") for i in range(max_idx + 1)])
    return rows


def parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def parse_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_trajectory(value: str) -> list[TrajectoryPoint]:
    if not value or not value.strip():
        return []
    try:
        raw_points = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(raw_points, (list, tuple)):
        return []

    points: list[TrajectoryPoint] = []
    for raw in raw_points:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        lon = parse_float(raw[0])
        lat = parse_float(raw[1])
        if lon is None or lat is None:
            continue
        speed = parse_float(raw[2]) if len(raw) > 2 else None
        course = parse_float(raw[3]) if len(raw) > 3 else None
        timestamp = parse_time(raw[4]) if len(raw) > 4 else None
        points.append(TrajectoryPoint(lon=lon, lat=lat, speed=speed, course=course, time=timestamp))

    if any(point.time is not None for point in points):
        points.sort(key=lambda point: point.time or datetime.min)
    return points


def load_dataset(path: Path) -> list[dict[str, object]]:
    rows = read_first_sheet_rows(path)
    if not rows:
        return []
    header = rows[0]
    index = {name: idx for idx, name in enumerate(header)}

    dataset: list[dict[str, object]] = []
    for raw_row in rows[1:]:
        item: dict[str, object] = {
            "round": raw_row[index["轮数"]] if "轮数" in index and index["轮数"] < len(raw_row) else "",
            "mmsi": raw_row[index["MMSI"]] if "MMSI" in index and index["MMSI"] < len(raw_row) else "",
        }
        for source in SOURCE_ORDER:
            value = raw_row[index[source]] if source in index and index[source] < len(raw_row) else ""
            item[source] = parse_trajectory(value)
        dataset.append(item)
    return dataset


def point_arrays(points: Iterable[TrajectoryPoint]) -> tuple[np.ndarray, np.ndarray]:
    lons = np.array([point.lon for point in points], dtype=np.float64)
    lats = np.array([point.lat for point in points], dtype=np.float64)
    return lons, lats


def robust_bounds(lons: np.ndarray, lats: np.ndarray, low: float = 0.4, high: float = 99.6) -> tuple[float, float, float, float]:
    lon_min, lon_max = np.percentile(lons, [low, high])
    lat_min, lat_max = np.percentile(lats, [low, high])
    lon_pad = max((lon_max - lon_min) * 0.04, 0.01)
    lat_pad = max((lat_max - lat_min) * 0.04, 0.01)
    return lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad


def in_bounds(lons: np.ndarray, lats: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    lon_min, lon_max, lat_min, lat_max = bounds
    return (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)


def source_interval_minutes(points: list[TrajectoryPoint]) -> list[float]:
    times = [point.time for point in points if point.time is not None]
    if len(times) < 2:
        return []
    intervals: list[float] = []
    for left, right in zip(times[:-1], times[1:]):
        delta = (right - left).total_seconds() / 60.0
        if 0 < delta <= 90:
            intervals.append(delta)
    return intervals


def collect_stats(dataset: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for source in SOURCE_ORDER:
        counts: list[int] = []
        intervals: list[float] = []
        lons: list[float] = []
        lats: list[float] = []
        for item in dataset:
            points = item[source]
            assert isinstance(points, list)
            counts.append(len(points))
            intervals.extend(source_interval_minutes(points))
            for point in points:
                lons.append(point.lon)
                lats.append(point.lat)
        stats[source] = {
            "counts": counts,
            "intervals": intervals,
            "lons": np.array(lons, dtype=np.float64),
            "lats": np.array(lats, dtype=np.float64),
        }
    return stats


def select_representative_rows(dataset: list[dict[str, object]], n_rows: int = 3) -> list[dict[str, object]]:
    scored: list[tuple[float, dict[str, object]]] = []
    for item in dataset:
        source_points = [item[source] for source in SOURCE_ORDER]
        if any(not isinstance(points, list) or len(points) < 45 for points in source_points):
            continue
        all_points = [point for points in source_points for point in points]
        lons, lats = point_arrays(all_points)
        bounds = robust_bounds(lons, lats, low=2.0, high=98.0)
        lon_span = bounds[1] - bounds[0]
        lat_span = bounds[3] - bounds[2]
        if lon_span < 0.03 or lat_span < 0.03 or lon_span > 1.6 or lat_span > 1.2:
            continue
        counts = [len(points) for points in source_points]
        balance = min(counts) / max(counts)
        score = sum(counts) + 150.0 * (lon_span + lat_span) + 90.0 * balance
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    selected: list[dict[str, object]] = []
    used_mmsi: set[str] = set()
    for _, item in scored:
        mmsi = str(item.get("mmsi", ""))
        if mmsi in used_mmsi and len(selected) < n_rows - 1:
            continue
        selected.append(item)
        used_mmsi.add(mmsi)
        if len(selected) == n_rows:
            break
    return selected


def set_geo_aspect(ax: plt.Axes, lat_center: float) -> None:
    _ = lat_center
    # Fixed map aspect creates large empty areas in multi-panel paper figures.
    # The axes are therefore kept compact while the numeric longitude/latitude
    # ticks preserve the spatial scale.
    ax.set_aspect("auto")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
    )


def plot_coverage(ax: plt.Axes, stats: dict[str, dict[str, object]]) -> tuple[float, float, float, float]:
    all_lons = np.concatenate([stats[source]["lons"] for source in SOURCE_ORDER])
    all_lats = np.concatenate([stats[source]["lats"] for source in SOURCE_ORDER])
    bounds = robust_bounds(all_lons, all_lats, low=0.35, high=99.65)
    rng = np.random.default_rng(2026)
    max_points = 26000

    for source in SOURCE_ORDER:
        lons = stats[source]["lons"]
        lats = stats[source]["lats"]
        assert isinstance(lons, np.ndarray) and isinstance(lats, np.ndarray)
        mask = in_bounds(lons, lats, bounds)
        lons = lons[mask]
        lats = lats[mask]
        if len(lons) > max_points:
            indices = rng.choice(len(lons), size=max_points, replace=False)
            lons = lons[indices]
            lats = lats[indices]
        ax.scatter(
            lons,
            lats,
            s=2.2,
            alpha=0.13,
            linewidths=0,
            color=SOURCE_COLORS[source],
            label=SOURCE_LABELS[source],
            rasterized=True,
        )

    lon_min, lon_max, lat_min, lat_max = bounds
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    set_geo_aspect(ax, (lat_min + lat_max) * 0.5)
    ax.set_title("Spatial Coverage", loc="left", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="#E6E2DA", linewidth=0.6)
    ax.legend(frameon=False, loc="upper right", markerscale=3.2, handletextpad=0.2)
    return bounds


def plot_sample(ax: plt.Axes, item: dict[str, object], index: int) -> None:
    all_points = [point for source in SOURCE_ORDER for point in item[source]]
    lons, lats = point_arrays(all_points)
    bounds = robust_bounds(lons, lats, low=1.5, high=98.5)

    for source in SOURCE_ORDER:
        points = item[source]
        assert isinstance(points, list)
        lons, lats = point_arrays(points)
        mask = in_bounds(lons, lats, bounds)
        lons = lons[mask]
        lats = lats[mask]
        if len(lons) == 0:
            continue
        ax.plot(lons, lats, color=SOURCE_COLORS[source], linewidth=1.15, alpha=0.78)
        ax.scatter(lons, lats, color=SOURCE_COLORS[source], s=7, alpha=0.68, linewidths=0)
        ax.scatter(lons[0], lats[0], color=SOURCE_COLORS[source], s=18, marker="o", edgecolor="white", linewidth=0.35, zorder=4)
        ax.scatter(lons[-1], lats[-1], color=SOURCE_COLORS[source], s=28, marker="^", edgecolor="white", linewidth=0.35, zorder=4)

    lon_min, lon_max, lat_min, lat_max = bounds
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    set_geo_aspect(ax, (lat_min + lat_max) * 0.5)
    title = f"Sample {index}: MMSI {item.get('mmsi', '-')}"
    ax.set_title(title, loc="left", fontsize=9.6, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="#E6E2DA", linewidth=0.55)


def make_boxplot(ax: plt.Axes, data: list[list[float]], labels: list[str], colors: list[str]) -> None:
    box = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False, widths=0.56)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.3)
    for median in box["medians"]:
        median.set_color("#262626")
        median.set_linewidth(1.4)
    for key in ("whiskers", "caps"):
        for artist in box[key]:
            artist.set_color("#6B6257")
            artist.set_linewidth(1.0)


def plot_count_distribution(ax: plt.Axes, stats: dict[str, dict[str, object]]) -> None:
    labels = [SOURCE_LABELS[source] for source in SOURCE_ORDER]
    colors = [SOURCE_COLORS[source] for source in SOURCE_ORDER]
    counts = [stats[source]["counts"] for source in SOURCE_ORDER]
    make_boxplot(ax, counts, labels, colors)
    for idx, values in enumerate(counts, start=1):
        median = float(np.median(values))
        ax.text(idx, median, f"{median:.0f}", ha="center", va="bottom", fontsize=8.5, color="#262626")
    ax.set_title("Observations per Sequence", loc="left", fontsize=9.8, fontweight="bold")
    ax.set_ylabel("Number of points")
    ax.grid(True, axis="y", color="#E6E2DA", linewidth=0.6)


def plot_interval_distribution(ax: plt.Axes, stats: dict[str, dict[str, object]]) -> None:
    labels = [SOURCE_LABELS[source] for source in SOURCE_ORDER]
    colors = [SOURCE_COLORS[source] for source in SOURCE_ORDER]
    intervals = []
    for source in SOURCE_ORDER:
        values = np.array(stats[source]["intervals"], dtype=np.float64)
        values = values[(values > 0) & (values <= 40)]
        intervals.append(values.tolist())
    make_boxplot(ax, intervals, labels, colors)
    upper = np.percentile(np.concatenate([np.array(values) for values in intervals if values]), 98)
    ax.set_ylim(0, max(8, min(28, upper * 1.25)))
    ax.set_title("Sampling Interval", loc="left", fontsize=9.8, fontweight="bold")
    ax.set_ylabel("Minutes")
    ax.grid(True, axis="y", color="#E6E2DA", linewidth=0.6)


def write_summary(path: Path, dataset: list[dict[str, object]], stats: dict[str, dict[str, object]]) -> None:
    source_summary: dict[str, dict[str, float]] = {}
    for source in SOURCE_ORDER:
        counts = np.array(stats[source]["counts"], dtype=np.float64)
        intervals = np.array(stats[source]["intervals"], dtype=np.float64)
        intervals = intervals[(intervals > 0) & (intervals <= 90)]
        source_summary[SOURCE_LABELS[source]] = {
            "total_points": int(counts.sum()),
            "median_points_per_sequence": float(np.median(counts)),
            "p10_points_per_sequence": float(np.percentile(counts, 10)),
            "p90_points_per_sequence": float(np.percentile(counts, 90)),
            "median_sampling_interval_min": float(np.median(intervals)) if len(intervals) else float("nan"),
            "p10_sampling_interval_min": float(np.percentile(intervals, 10)) if len(intervals) else float("nan"),
            "p90_sampling_interval_min": float(np.percentile(intervals, 90)) if len(intervals) else float("nan"),
        }
    complete = sum(all(len(item[source]) > 0 for source in SOURCE_ORDER) for item in dataset)
    payload = {
        "num_trajectory_groups": len(dataset),
        "num_complete_three_source_groups": complete,
        "total_observations": int(sum(summary["total_points"] for summary in source_summary.values())),
        "sources": source_summary,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input or infer_input_path()
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(input_path)
    if not dataset:
        raise RuntimeError(f"No samples parsed from {input_path}")
    stats = collect_stats(dataset)
    selected = select_representative_rows(dataset, n_rows=3)
    if len(selected) < 3:
        raise RuntimeError("Could not find enough representative complete trajectories.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.facecolor": "#FCFAF6",
            "figure.facecolor": "#FCFAF6",
            "axes.edgecolor": "#4C4944",
            "axes.linewidth": 0.8,
            "xtick.color": "#3F3A34",
            "ytick.color": "#3F3A34",
            "axes.labelcolor": "#2D2925",
            "text.color": "#2D2925",
        }
    )

    fig = plt.figure(figsize=(13.8, 7.35), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=4,
        width_ratios=[1.42, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 0.86],
        left=0.055,
        right=0.985,
        top=0.86,
        bottom=0.09,
        wspace=0.34,
        hspace=0.38,
    )

    ax_coverage = fig.add_subplot(grid[:, 0])
    ax_sample_1 = fig.add_subplot(grid[0, 1])
    ax_sample_2 = fig.add_subplot(grid[0, 2])
    ax_sample_3 = fig.add_subplot(grid[0, 3])
    ax_counts = fig.add_subplot(grid[1, 1:3])
    ax_intervals = fig.add_subplot(grid[1, 3])

    plot_coverage(ax_coverage, stats)
    for idx, (axis, item) in enumerate(zip((ax_sample_1, ax_sample_2, ax_sample_3), selected), start=1):
        plot_sample(axis, item, idx)
    plot_count_distribution(ax_counts, stats)
    plot_interval_distribution(ax_intervals, stats)

    for label, axis in zip("ABCDEF", (ax_coverage, ax_sample_1, ax_sample_2, ax_sample_3, ax_counts, ax_intervals)):
        add_panel_label(axis, label)

    total_points = sum(int(np.sum(stats[source]["counts"])) for source in SOURCE_ORDER)
    complete_groups = sum(all(len(item[source]) > 0 for source in SOURCE_ORDER) for item in dataset)
    fig.suptitle("Multi-Source Ship Trajectory Dataset (MSTD)", fontsize=13.4, fontweight="bold", y=0.962)
    fig.text(
        0.5,
        0.925,
        f"{len(dataset):,} trajectory groups, {complete_groups:,} complete three-source groups, "
        f"{total_points:,} observations from BDS, AIS, and Radar",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#4C4944",
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    summary_path = output_prefix.with_suffix(".summary.json")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    write_summary(summary_path, dataset, stats)
    print(json.dumps({"png": str(png_path), "pdf": str(pdf_path), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
