#%%
from pathlib import Path
import csv
import random
import re
import shutil
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPoint, Point
from shapely.prepared import prep


#%%
# User settings

PROJECT_DIR = Path(__file__).resolve().parents[1]

GFLOW_EXE = PROJECT_DIR / "executables/GFLOW/epa-gflow1/release/gflow2.exe"

GFLOW_DIR = PROJECT_DIR / "GFLOW"
MODEL_DAT = GFLOW_DIR / "9426a08.dat"
TRACE_DAT = GFLOW_DIR / "regrid.dat"

OUT_DIR = PROJECT_DIR / "data/generated/gflow_hk_capture_zone"

N_REALIZATIONS = 20
HK_LOW = 10.0
HK_HIGH = 50.0
RANDOM_SEED = 20260904

PATH_BUFFER_FT = 100.0
GRID_SIZE = 180
PLOT_PADDING_FT = 1500.0
GFLOW_TIMEOUT_SECONDS = 300


#%%
# Small helper functions

def read_text(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def get_base_name(dat_text):
    # -- Read the GFLOW base filename from the command deck.
    match = re.search(r"^\s*bfname\s+(\S+)", dat_text, flags=re.MULTILINE)
    if not match:
        raise ValueError("Could not find 'bfname' in the model .dat file.")
    return match.group(1)


def get_trace_well(trace_text):
    # -- Read the traced well from the existing trace deck.
    match = re.search(
        r"^\s*well\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+(\d+)",
        trace_text,
        flags=re.MULTILINE,
    )
    if not match:
        raise ValueError("Could not find the trace 'well x y z n' command in regrid.dat.")

    return {
        "x": float(match.group(1)),
        "y": float(match.group(2)),
        "z": float(match.group(3)),
        "n_particles": int(match.group(4)),
    }


def set_aquifer_hk(dat_text, hk_ft_per_day):
    # -- Replace the first aquifer permeability value and leave everything else unchanged.
    pattern = re.compile(r"(^\s*permeability\s+)([-+0-9.Ee]+)(\s*$)", re.MULTILINE)
    new_text, n_changed = pattern.subn(
        rf"\g<1>{hk_ft_per_day:.7E}\g<3>",
        dat_text,
        count=1,
    )
    if n_changed != 1:
        raise ValueError("Could not replace exactly one aquifer permeability value.")
    return new_text


def make_solve_only_deck(dat_text):
    # -- Keep the model setup, solve, and first save; skip extract/grid/trace commands.
    lines = dat_text.splitlines()
    trimmed_lines = []
    found_first_save = False
    found_save_answer = False

    for line in lines:
        trimmed_lines.append(line)

        if re.match(r"^\s*save\s+\S+\s*$", line):
            found_first_save = True
            continue

        if found_first_save and line.strip().lower() == "y":
            found_save_answer = True
            break

    if not found_save_answer:
        raise ValueError("Could not trim the model deck after its first save command.")

    trimmed_lines.append("stop")
    return "\n".join(trimmed_lines) + "\n"


def run_gflow(exe_path, run_dir, deck_name):
    if not exe_path.exists():
        raise FileNotFoundError(
            f"Set GFLOW_EXE to the GFLOW executable before running this script: {exe_path}"
        )

    # -- Pass the command deck as a filename; GFLOW1 does not read decks from stdin.
    deck_path = run_dir / deck_name
    if not deck_path.exists():
        raise FileNotFoundError(f"Missing GFLOW command deck: {deck_path}")

    startupinfo = None
    creationflags = 0
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        result = subprocess.run(
            [str(exe_path), deck_path.name],
            cwd=run_dir,
            text=True,
            capture_output=True,
            timeout=GFLOW_TIMEOUT_SECONDS,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"GFLOW did not finish within {GFLOW_TIMEOUT_SECONDS} seconds for {deck_path}"
        ) from error

    # -- Retain the console streams for diagnosing individual scenarios.
    (run_dir / f"{deck_path.stem}_stdout.log").write_text(
        result.stdout,
        encoding="utf-8",
    )
    (run_dir / f"{deck_path.stem}_stderr.log").write_text(
        result.stderr,
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"GFLOW failed for {deck_path} with return code {result.returncode}. "
            f"See {deck_path.stem}_stdout.log and {deck_path.stem}_stderr.log."
        )


def parse_pth(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing expected particle-track file: {path}")

    # -- Split the GFLOW PTH file into one array of xy points per particle.
    tracks = []
    current_track = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("START"):
            current_track = []
            parts = stripped.split()
            current_track.append((float(parts[1]), float(parts[2])))
            continue

        if stripped.startswith("END"):
            parts = stripped.split()
            current_track.append((float(parts[1]), float(parts[2])))
            tracks.append(np.array(current_track))
            current_track = []
            continue

        if current_track:
            parts = stripped.split()
            current_track.append((float(parts[0]), float(parts[1])))

    if not tracks:
        raise ValueError(f"No particle tracks found in {path}")

    return tracks


def tracks_to_polygon(tracks):
    # -- Use a deliberately simple hull of all particle-track points as the capture-zone extent.
    xy = np.vstack(tracks)
    polygon = MultiPoint([tuple(row) for row in xy]).convex_hull.buffer(PATH_BUFFER_FT)
    if polygon.is_empty:
        raise ValueError("Particle tracks produced an empty capture-zone polygon.")
    return polygon


def write_tracks_csv(path, run_id, hk, tracks):
    # -- Save parsed tracks so the GFLOW output can be checked without reparsing PTH files.
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["run_id", "hk_ft_per_day", "particle_id", "point_id", "x_ft", "y_ft"])
        for particle_id, track in enumerate(tracks, start=1):
            for point_id, (x, y) in enumerate(track, start=1):
                writer.writerow([run_id, f"{hk:.6f}", particle_id, point_id, f"{x:.3f}", f"{y:.3f}"])


#%%
# Read the existing model and trace setup

model_text = read_text(MODEL_DAT)
trace_text = read_text(TRACE_DAT)

base_name = get_base_name(model_text)
trace_well = get_trace_well(trace_text)
pth_name = f"{base_name.upper()}.PTH"

print(f"GFLOW base name: {base_name}")
print(f"Trace well x/y: {trace_well['x']:.1f}, {trace_well['y']:.1f}")
print(f"Particles per trace: {trace_well['n_particles']}")


#%%
# Draw random HK values

rng = random.Random(RANDOM_SEED)
hk_values = [rng.uniform(HK_LOW, HK_HIGH) for _ in range(N_REALIZATIONS)]

OUT_DIR.mkdir(parents=True, exist_ok=True)

with (OUT_DIR / "hk_values.csv").open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["run_id", "hk_ft_per_day"])
    for i, hk in enumerate(hk_values, start=1):
        writer.writerow([f"run_{i:02d}", f"{hk:.6f}"])

hk_values


#%%
# Run GFLOW once for each HK value and collect capture-zone polygons

polygons = []
all_tracks = []

for i, hk in enumerate(hk_values, start=1):
    run_id = f"run_{i:02d}"
    run_dir = OUT_DIR / run_id

    # -- Start each scenario from a clean copy of the current GFLOW folder.
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(GFLOW_DIR, run_dir)

    run_model_text = set_aquifer_hk(model_text, hk)

    run_model_dat = run_dir / MODEL_DAT.name
    run_model_dat.write_text(run_model_text, encoding="utf-8", newline="")

    solve_only_dat = run_dir / "solve_only.dat"
    solve_only_dat.write_text(make_solve_only_deck(run_model_text), encoding="utf-8", newline="")

    print(f"{run_id}: running HK = {hk:.2f} ft/day")
    solution_path = run_dir / f"{base_name.upper()}.SOL"
    pathline_path = run_dir / pth_name

    # -- Remove copied results so every scenario must produce its own solution and paths.
    solution_path.unlink(missing_ok=True)
    pathline_path.unlink(missing_ok=True)

    run_gflow(GFLOW_EXE, run_dir, solve_only_dat.name)
    if not solution_path.exists():
        raise FileNotFoundError(f"GFLOW did not create the expected solution: {solution_path}")

    run_gflow(GFLOW_EXE, run_dir, TRACE_DAT.name)
    if not pathline_path.exists():
        raise FileNotFoundError(f"GFLOW did not create the expected pathlines: {pathline_path}")

    tracks = parse_pth(pathline_path)
    polygon = tracks_to_polygon(tracks)

    write_tracks_csv(run_dir / "particle_tracks.csv", run_id, hk, tracks)
    polygons.append(polygon)
    all_tracks.append((run_id, hk, tracks))

print(f"Finished {len(polygons)} GFLOW realizations.")


#%%
# Convert the 10 capture-zone polygons into a simple probability grid

combined_bounds = np.array([polygon.bounds for polygon in polygons])
x_min = combined_bounds[:, 0].min() - PLOT_PADDING_FT
y_min = combined_bounds[:, 1].min() - PLOT_PADDING_FT
x_max = combined_bounds[:, 2].max() + PLOT_PADDING_FT
y_max = combined_bounds[:, 3].max() + PLOT_PADDING_FT

x_grid = np.linspace(x_min, x_max, GRID_SIZE)
y_grid = np.linspace(y_min, y_max, GRID_SIZE)
xx, yy = np.meshgrid(x_grid, y_grid)

probability = np.zeros(xx.shape)
prepared_polygons = [prep(polygon) for polygon in polygons]

for polygon in prepared_polygons:
    inside = [
        polygon.contains(Point(x, y)) or polygon.touches(Point(x, y))
        for x, y in zip(xx.ravel(), yy.ravel())
    ]
    probability += np.array(inside).reshape(xx.shape)

probability = probability / len(polygons)


#%%
# Plot the capture-zone probability around well 1902424

fig, ax = plt.subplots(figsize=(8, 7))

levels = np.linspace(0.0, 1.0, 11)
filled = ax.contourf(
    xx,
    yy,
    probability,
    levels=levels,
    cmap="Blues",
    alpha=0.8,
)

for run_id, hk, tracks in all_tracks:
    for track in tracks:
        ax.plot(track[:, 0], track[:, 1], color="0.25", lw=0.35, alpha=0.18)

ax.scatter(
    trace_well["x"],
    trace_well["y"],
    s=55,
    color="crimson",
    edgecolor="white",
    linewidth=0.8,
    zorder=10,
    label="Well 1902424",
)

cbar = fig.colorbar(filled, ax=ax)
cbar.set_label("Capture-zone probability")

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("GFLOW local x (ft)")
ax.set_ylabel("GFLOW local y (ft)")
ax.set_title("Simple HK Uncertainty Capture Zone: Well 1902424")
ax.legend(loc="upper right")

plot_path = OUT_DIR / "well_1902424_capture_zone_probability.png"
fig.tight_layout()
fig.savefig(plot_path, dpi=250)

print(f"Wrote: {plot_path}")
# plt.close(fig)
