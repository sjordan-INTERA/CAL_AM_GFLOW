# -*- coding: utf-8 -*-
"""
Plot GFLOW head contours and the main analytic-element features.

This is a post-processing script. It reads GFLOW's text exports, not the
binary .gfl Access database. Point INPUT_PATH at either a GFLOW run folder
or a specific .dat file written by GFLOW.
"""

from pathlib import Path
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import contextily as cx


INPUT_PATH = "GFLOW"
OUTPUT_FIGURE = Path("figures") / "gflow_model_contours.png"
CONTOUR_INTERVAL = 15
PLOT_WORLD_COORDS = True
MISSING_VALUE = -9999.0


def fail(message):
    raise SystemExit(f"\nCannot plot GFLOW results:\n{message}\n")


def choose_dat_file(input_path):
    # -- Resolve the user input to one readable GFLOW command/export file.
    if input_path.is_file():
        if input_path.suffix.lower() == ".gfl":
            fail(
                f"{input_path} is a binary GFLOW model database. Export/write a "
                "GFLOW .dat file from the model first, then point this script at "
                "that .dat file or its folder."
            )
        return input_path

    dat_files = sorted(input_path.glob("*.dat"), key=lambda p: p.stat().st_mtime)
    dat_files = [
        p
        for p in dat_files
        if p.name.lower() != "regrid.dat" and not p.name.startswith("_")
    ]
    if not dat_files:
        fail(
            f"No .dat file was found in {input_path}. GFLOW's .gfl file is not "
            "enough for this script; export/write the model commands to .dat."
        )

    scored_files = []
    for dat_file in dat_files:
        text = dat_file.read_text(errors="ignore").lower()
        score = 0
        score += 5 if "inhomogeneity" in text else 0
        score += 3 if "linesink" in text else 0
        score += 2 if "\n well " in f"\n{text}" else 0
        score += 100 if companion_grid_exists(dat_file) else 0
        scored_files.append((score, dat_file.stat().st_mtime, dat_file))

    scored_files.sort()
    return scored_files[-1][2]


def companion_grid_exists(dat_file):
    for grid_file in dat_file.parent.glob("*.GRD"):
        if grid_file.stem.lower() == dat_file.stem.lower():
            return True
    return False


def choose_grid_file(folder, dat_file, require_same_base=False):
    # -- Prefer the grid with the same base name as the .dat run.
    for grid_file in folder.glob("*.GRD"):
        if grid_file.stem.lower() == dat_file.stem.lower():
            return grid_file

    if require_same_base:
        fail(
            f"No .GRD file matching {dat_file.name} was found. Run GFLOW's "
            f"grid/surfer export for this model so {dat_file.stem}.GRD exists."
        )

    grid_files = sorted(folder.glob("*.GRD"), key=lambda p: p.stat().st_mtime)
    if not grid_files:
        fail(
            f"No .GRD file was found in {folder}. Run GFLOW's grid/surfer export "
            "for heads before using this post-processor."
        )
    return grid_files[-1]


def parse_float_tokens(line):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)]


def parse_model_origin(lines):
    # -- GFLOW .dat coordinates are commonly model-local; this line stores the offset.
    for line in lines:
        if line.strip().lower().startswith("modelorigin"):
            values = parse_float_tokens(line)
            if len(values) >= 2:
                return values[0], values[1]
    return 0.0, 0.0


def maybe_to_model_coords(x, y, origin_x, origin_y, grid_bounds):
    # -- Convert absolute State Plane-like coordinates to GFLOW model coordinates.
    xmin, xmax, ymin, ymax = grid_bounds
    if xmin <= x <= xmax and ymin <= y <= ymax:
        return x, y

    shifted_x = x - origin_x
    shifted_y = y - origin_y
    if xmin <= shifted_x <= xmax and ymin <= shifted_y <= ymax:
        return shifted_x, shifted_y

    return x, y


def maybe_to_world_coords(x, y, origin_x, origin_y):
    if PLOT_WORLD_COORDS:
        return x + origin_x, y + origin_y
    return x, y


def read_surfer_ascii_grid(grid_file):
    # -- Read GFLOW's Surfer ASCII grid export.
    tokens = grid_file.read_text(errors="ignore").split()
    if not tokens or tokens[0].upper() != "DSAA":
        fail(f"{grid_file} is not a Surfer ASCII DSAA grid.")

    nx = int(tokens[1])
    ny = int(tokens[2])
    xmin = float(tokens[3])
    xmax = float(tokens[4])
    ymin = float(tokens[5])
    ymax = float(tokens[6])
    zmin = float(tokens[7])
    zmax = float(tokens[8])

    values = np.array([float(x) for x in tokens[9:]], dtype=float)
    if values.size != nx * ny:
        fail(
            f"{grid_file} says it has {nx} by {ny} cells, but the file contains "
            f"{values.size} grid values."
        )

    z = values.reshape((ny, nx))
    z[z <= MISSING_VALUE + 1] = np.nan
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y)
    return xx, yy, z, (xmin, xmax, ymin, ymax), (zmin, zmax)


def parse_dat_features(dat_file, grid_bounds):
    # -- Walk the GFLOW command file and keep only plot-friendly geometry.
    lines = dat_file.read_text(errors="ignore").splitlines()
    origin_x, origin_y = parse_model_origin(lines)

    barriers = []
    inhomogeneities = []
    linesinks = []
    wells = []

    section = None
    subsection = None
    current_points = []
    current_name = None

    def save_current_polyline():
        if section == "inhomogeneity" and current_points:
            target = barriers if current_name == "barrier" else inhomogeneities
            target.append(current_points.copy())

    for raw_line in lines:
        line = raw_line.strip()
        low = line.lower()
        if not line:
            continue

        if low == "quit":
            save_current_polyline()
            section = None
            subsection = None
            current_points = []
            current_name = None
            continue

        if low in ["inhomogeneity", "well", "linesink"]:
            save_current_polyline()
            section = low
            subsection = None
            current_points = []
            current_name = None
            continue

        if section == "inhomogeneity":
            if low.startswith("slurry"):
                save_current_polyline()
                current_name = "barrier"
                current_points = []
                continue
            if low.startswith("transmissivity"):
                save_current_polyline()
                current_name = "inhomogeneity"
                current_points = []
                continue
            values = parse_float_tokens(line)
            if len(values) >= 2 and ("hb_" in low or "in_" in low):
                x, y = maybe_to_world_coords(values[0], values[1], origin_x, origin_y)
                current_points.append((x, y))
            continue

        if section == "well":
            if low in ["discharge", "head"]:
                subsection = low
                continue
            values = parse_float_tokens(line)
            if len(values) >= 4 and "wl_" in low:
                x, y = maybe_to_world_coords(values[0], values[1], origin_x, origin_y)
                wells.append((x, y, values[2], subsection))
            continue

        if section == "linesink":
            if low in ["head", "discharge"]:
                subsection = low
                continue
            if low.startswith(("resistance", "width", "depth")):
                continue
            values = parse_float_tokens(line)
            if len(values) >= 5 and "ls_" in low:
                x1, y1 = maybe_to_world_coords(values[0], values[1], origin_x, origin_y)
                x2, y2 = maybe_to_world_coords(values[2], values[3], origin_x, origin_y)
                linesinks.append((x1, y1, x2, y2, values[4], subsection))
            continue

    return {
        "origin": (origin_x, origin_y),
        "barriers": barriers,
        "inhomogeneities": inhomogeneities,
        "linesinks": linesinks,
        "wells": wells,
    }


def read_observation_points(folder, origin_x, origin_y, grid_bounds):
    # -- Observation/test-point files are optional overlays.
    tp_files = sorted(folder.glob("*.tp"), key=lambda p: p.stat().st_mtime)
    if not tp_files:
        return pd.DataFrame(columns=["x", "y", "head", "label"])

    rows = []
    for tp_file in tp_files:
        for line in tp_file.read_text(errors="ignore").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                head = float(parts[2])
            except ValueError:
                continue
            label = parts[-1] if len(parts) >= 6 else tp_file.stem
            x, y = maybe_to_model_coords(x, y, origin_x, origin_y, grid_bounds)
            x, y = maybe_to_world_coords(x, y, origin_x, origin_y)
            rows.append((x, y, head, label))

    return pd.DataFrame(rows, columns=["x", "y", "head", "label"])


def build_contour_levels(z):
    # -- Use a simple fixed interval so repeated runs are easy to compare.
    if np.all(np.isnan(z)):
        fail("The grid contains only missing values inside the plotting array.")
    zmin = np.floor(np.nanmin(z) / CONTOUR_INTERVAL) * CONTOUR_INTERVAL
    zmax = np.ceil(np.nanmax(z) / CONTOUR_INTERVAL) * CONTOUR_INTERVAL
    return np.arange(zmin, zmax + CONTOUR_INTERVAL, CONTOUR_INTERVAL)


def plot_polyline_collection(ax, polylines, **kwargs):
    # -- Plot each GFLOW string as one continuous line.
    labels = kwargs.pop("labels", None)
    colors = kwargs.pop("colors", None)
    linestyles = kwargs.pop("linestyles", None)
    for i, points in enumerate(polylines):
        if len(points) < 2:
            continue
        xy = np.array(points)
        line_label = labels[i]
        color = colors[i]
        ls = linestyles[i]
        ax.plot(xy[:, 0], xy[:, 1], label=line_label, color=color,ls=ls)


def plot_gflow_results(input_path):
    # -- Load GFLOW outputs and model features.
    input_path = Path(input_path)
    require_same_grid = input_path.is_file()
    dat_file = choose_dat_file(input_path)
    folder = dat_file.parent
    grid_file = choose_grid_file(folder, dat_file, require_same_grid)
    xx, yy, z, grid_bounds, _ = read_surfer_ascii_grid(grid_file)
    features = parse_dat_features(dat_file, grid_bounds)
    origin_x, origin_y = features["origin"]
    obs = read_observation_points(folder, origin_x, origin_y, grid_bounds)

    if PLOT_WORLD_COORDS:
        xx = xx + origin_x
        yy = yy + origin_y

    # -- Build the map.
    fig, ax = plt.subplots(figsize=(11, 8.5))
    levels = build_contour_levels(z)
    filled = ax.contourf(xx, yy, z, levels=levels, cmap="viridis", alpha=0.55)
    contours = ax.contour(xx, yy, z, levels=levels, colors="black", linewidths=0.75)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    cbar = fig.colorbar(filled, ax=ax, shrink=0.5)
    cbar.set_label("Simulated head (ft-amsl)")

    # plot_polyline_collection(
    #     ax,
    #     features["inhomogeneities"],
    #     color="0.20",
    #     linewidth=1.2,
    #     linestyle="-",
    #     label="Inhomogeneity boundary",
    # )
    plot_polyline_collection(
        ax,
        features["barriers"],
        colors=["k","tab:red"],
        linewidth=2.0,
        linestyles=["-","--"],
        labels=["Basin Bounds", "Raymond Fault"],
    )

    for i, (x1, y1, x2, y2, head, kind) in enumerate(features["linesinks"]):
        label = "San Gabriel River (Line Sink)" if i == 0 else None
        ax.plot([x1, x2], [y1, y2], color="dodgerblue", linewidth=1.8, label=label)

    if features["wells"]:
        wells = pd.DataFrame(features["wells"], columns=["x", "y", "q", "kind"])
        ax.scatter(
            wells["x"],
            wells["y"],
            s=20,
            marker="v",
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            label="Specified-discharge wells",
            zorder=5,
        )

    if len(obs) > 0:
        ax.scatter(
            obs["x"],
            obs["y"],
            s=26,
            marker="o",
            facecolor="tab:orange",
            edgecolor="black",
            linewidth=0.5,
            label="Observation wells",
            zorder=6,
        )
    
    # -- Add basemap
    cx.add_basemap(
        ax=ax,
        crs=2229,
        source="https://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer/tile/{z}/{y}/{x}",
    )
    
    # -- Format and save.
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Simulated GFLOW Heads")
    # ax.set_xlabel("x coordinate")
    # ax.set_ylabel("y coordinate")
    ax.legend(loc="best", frameon=True, framealpha=0.95)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=250)
    # plt.close(fig)

    print(f"Read model features from: {dat_file}")
    print(f"Read head grid from:      {grid_file}")
    print(f"Saved figure to:         {OUTPUT_FIGURE}")
    print(f"Horizontal barriers:     {len(features['barriers'])}")
    print(f"Line-sink segments:      {len(features['linesinks'])}")
    print(f"Wells:                   {len(features['wells'])}")
    print(f"Observation points:      {len(obs)}")


if __name__ == "__main__":
    # -- Optional simple command-line override without argparse.
    if len(sys.argv) > 1:
        INPUT_PATH = Path(sys.argv[1])
    plot_gflow_results(INPUT_PATH)
