import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer


WELL_CSV = Path("data/generated/gflow_wells_2014_15.csv")
OUT_CSV = Path("data/generated/figure_5_11_digitized_heads_2015.csv")
OUT_GFLOW_CSV = Path("GFLOW/figure_5_11_observed_heads_2015.csv")
PAGE_DIR = Path("tmp/pdfs/fig_5_11_all")

SOURCE_EPSG = 26799  # NAD27 / California zone VII, ftUS
TARGET_EPSG = 2229  # NAD83 / California zone 5, ftUS

PLOTS = [
    ["page-118.png", "top", "1900018", "City of Alhambra Garfield Well", "4069", 1970, 2020, 80, 240],
    ["page-118.png", "bottom", "1901014", "City of Arcadia Longden 2", "4198G", 1970, 2020, 140, 300],
    ["page-119.png", "top", "8000127", "City of Arcadia Well Live Oak 1", "", 1970, 2020, 120, 320],
    ["page-119.png", "bottom", "1902537", "City of Azusa Well Genesis 2", "", 1970, 2020, 140, 340],
    ["page-120.png", "top", "1903088", "Calmat Well Reliance 1", "", 1970, 2020, 140, 340],
    ["page-120.png", "bottom", "1900355", "CAWC Buena Vista", "4227A", 1960, 2020, 140, 400],
    ["page-121.png", "top", "1902713", "SGVWC-El Monte Well 11C", "", 1970, 2020, 100, 300],
    ["page-121.png", "bottom", "1903018", "CAWC-Duarte Crown Haven Well", "", 1970, 2020, 120, 360],
    ["page-122.png", "top", "1900926", "CAWC-San Marino Grand", "2920G", 1970, 2020, 80, 260],
    ["page-122.png", "bottom", "1900920", "CAWC-San Marino Well Mission View 2", "", 1970, 2020, 100, 260],
    ["page-123.png", "top", "1900354", "CAWC Santa Fe Well", "4246", 1970, 2020, 100, 450],
    ["page-123.png", "bottom", "1900885", "Covina Irrigating Company Well Bal 1", "", 1970, 2020, 140, 320],
    ["page-124.png", "top", "1901699", "City of El Monte Well 10", "", 1970, 2020, 120, 300],
    ["page-124.png", "bottom", "1901524", "City of Glendora Well 04E", "", 1970, 2020, 140, 340],
    ["page-125.png", "top", "", "LA County Well 3030F (Key Well)", "3030F", 1970, 2020, 100, 360],
    ["page-125.png", "bottom", "1902027", "GSWC Persimmon 1", "2960K", 1970, 2020, 120, 300],
    ["page-126.png", "top", "1940104", "City of Monrovia Well 05", "", 1970, 2020, 120, 320],
    ["page-126.png", "bottom", "1902372", "City of Monterey Park Well 7", "", 1970, 2020, 100, 260],
    ["page-127.png", "top", "8000063", "Calmat Well Durbin West", "", 1970, 2020, 140, 320],
    ["page-127.png", "bottom", "1901694", "City of El Monte Well 4", "2963B", 1970, 2020, 100, 300],
    ["page-128.png", "top", "1902790", "Rincon Ditch Company Well 4", "", 1970, 2020, 100, 260],
    ["page-128.png", "bottom", "8000123", "SGCWD Well 12", "", 1970, 2020, 80, 240],
    ["page-129.png", "top", "1902525", "SGVWC Well B2", "", 1970, 2020, 120, 260],
    ["page-129.png", "bottom", "1900718", "SGVWC Well B5A", "2994V", 1970, 2020, 120, 300],
    ["page-130.png", "top", "1903093", "SGVWC Well B6C", "", 1970, 2020, 140, 300],
    ["page-130.png", "bottom", "8000187", "SGVWC Well B25A", "", 1970, 2020, 120, 300],
    ["page-131.png", "top", "1903067", "SWS 140W-3", "", 1970, 2020, 140, 300],
    ["page-131.png", "bottom", "8000087", "SWS 125W-2", "", 1970, 2020, 180, 340],
    ["page-132.png", "top", "1900031", "VCWD Well Paddy Ln", "", 1970, 2020, 140, 320],
    ["page-132.png", "bottom", "1900034", "VCWD Well Arrow", "4239F", 1970, 2020, 140, 340],
    ["page-133.png", "top", "8000071", "City of Whittier Well 15", "", 1970, 2020, 100, 280],
    ["page-133.png", "bottom", "8000136", "City of Whittier Well 18", "", 1970, 2020, 100, 260],
    ["page-134.png", "top", "1902148", "GSWC-San Dimas Artesia 3", "", 1970, 2020, 700, 1100],
    ["page-134.png", "bottom", "1902150", "GSWC-San Dimas Well Highway", "", 1970, 2020, 700, 1100],
    ["page-135.png", "top", "1900029", "VCWD Well Morada", "", 1970, 2020, 140, 340],
    ["page-135.png", "bottom", "8000060", "VCWD Lante Well", "", 1970, 2020, 120, 360],
    ["page-136.png", "top", "1902287", "GSWC-San Dimas Well Malone", "", 1970, 2020, 900, 1400],
    ["page-136.png", "bottom", "1900736", "SGVWC Well 8A", "", 1970, 2020, 100, 300],
    ["page-137.png", "top", "1900881", "Covina Irrigating Company Contract Well", "4288A", 1970, 2020, 140, 360],
    ["page-137.png", "bottom", "1902854", "City of Arcadia Well Peck 1", "", 1970, 2020, 140, 300],
    ["page-138.png", "top", "1900035", "VCWD Big Dalton", "3042F", 1970, 2020, 140, 320],
    ["page-138.png", "bottom", "1902077", "Arcadia Camino 1", "4177A", 1970, 2020, 120, 320],
    ["page-139.png", "top", "1902792", "Sunny Slope Water Company Well 09", "", 1970, 2020, 80, 240],
    ["page-139.png", "bottom", "1901441", "CAWC-San Marino Well Blue Ribbon 1", "", 1970, 2020, 100, 260],
    ["page-140.png", "top", "1901599", "SWS 139W-2", "", 1970, 2020, 140, 300],
    ["page-140.png", "bottom", "8000190", "SGVWC Well B26B", "", 1970, 2020, 120, 300],
    ["page-141.png", "top", "1902786", "SGCWD Well 10", "", 1970, 2020, 80, 240],
    ["page-141.png", "bottom", "1902148", "SCWC-San Dimas Well Baseline 3", "", 1970, 2020, 700, 1100],
]

PLOTS = [
    {
        "page": page,
        "panel": panel,
        "recordation_number": recordation_number,
        "site_name": site_name,
        "la_county_id": la_county_id,
        "x_year_min": x_year_min,
        "x_year_max": x_year_max,
        "y_head_min": y_head_min,
        "y_head_max": y_head_max,
    }
    for (
        page,
        panel,
        recordation_number,
        site_name,
        la_county_id,
        x_year_min,
        x_year_max,
        y_head_min,
        y_head_max,
    ) in PLOTS
]


def red_mask(image):
    array = np.asarray(image.convert("RGB"))
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    return (red > 180) & (green < 90) & (blue < 90)


def black_mask(image):
    array = np.asarray(image.convert("RGB"))
    return (array[:, :, 0] < 40) & (array[:, :, 1] < 40) & (array[:, :, 2] < 40)


def line_runs(mask_counts, threshold):
    indices = np.where(mask_counts > threshold)[0]
    runs = []
    if len(indices) == 0:
        return runs

    start = indices[0]
    previous = indices[0]
    for index in indices[1:]:
        if index > previous + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return runs


def true_runs(mask):
    indices = np.where(mask)[0]
    runs = []
    if len(indices) == 0:
        return runs

    start = indices[0]
    previous = indices[0]
    for index in indices[1:]:
        if index > previous + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return runs


def center(run):
    return 0.5 * (run[0] + run[1])


def find_plot_boxes(image):
    dark = black_mask(image)
    height, width = dark.shape

    row_counts = dark.sum(axis=1)
    row_runs = line_runs(row_counts, int(width * 0.35))

    y_edges = [
        center(run)
        for run in row_runs
        if 150 < center(run) < height - 250
        and 1000 < row_counts[int(center(run))] < 1700
    ]

    boxes = []
    for y0, y1 in zip(y_edges[0::2], y_edges[1::2]):
        y_sample = int(round(y0))
        horizontal = dark[max(0, y_sample - 1) : y_sample + 2, :].any(axis=0)
        segments = [run for run in true_runs(horizontal) if run[1] - run[0] > 1000]
        if not segments:
            raise ValueError("Could not find horizontal plot border")
        x0, x1 = max(segments, key=lambda run: run[1] - run[0])
        boxes.append((x0, y0, x1, y1))
    return boxes


def connected_components(mask):
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components = []

    # -- Find connected red-pixel groups using a simple stack walk.
    for y, x in zip(*np.where(mask & ~visited)):
        if visited[y, x]:
            continue
        stack = [(x, y)]
        visited[y, x] = True
        xs = []
        ys = []
        while stack:
            px, py = stack.pop()
            xs.append(px)
            ys.append(py)
            for nx in range(max(0, px - 1), min(width, px + 2)):
                for ny in range(max(0, py - 1), min(height, py + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
        if len(xs) >= 20:
            components.append(
                {
                    "x": float(np.mean(xs)),
                    "y": float(np.mean(ys)),
                    "n": len(xs),
                    "x_min": min(xs),
                    "x_max": max(xs),
                    "y_min": min(ys),
                    "y_max": max(ys),
                }
            )

    return components


def pixel_to_year(x_pixel, plot, x0, x1):
    fraction = (x_pixel - x0) / (x1 - x0)
    return plot["x_year_min"] + fraction * (plot["x_year_max"] - plot["x_year_min"])


def pixel_to_head(y_pixel, plot, y0, y1):
    fraction = (y_pixel - y0) / (y1 - y0)
    return plot["y_head_max"] - fraction * (plot["y_head_max"] - plot["y_head_min"])


def digitize_plot(plot):
    image = Image.open(PAGE_DIR / plot["page"])
    boxes = find_plot_boxes(image)
    box = boxes[0] if plot["panel"] == "top" else boxes[1]
    x0, y0, x1, y1 = box

    red = red_mask(image)
    red[: int(y0), :] = False
    red[int(y1) :, :] = False
    red[:, : int(x0)] = False
    red[:, int(x1) :] = False

    year_min_pixel = x0 + (2014.0 - plot["x_year_min"]) / (
        plot["x_year_max"] - plot["x_year_min"]
    ) * (x1 - x0)
    year_max_pixel = x0 + (2016.0 - plot["x_year_min"]) / (
        plot["x_year_max"] - plot["x_year_min"]
    ) * (x1 - x0)
    red_window = red.copy()
    red_window[:, : int(year_min_pixel)] = False
    red_window[:, int(year_max_pixel) :] = False

    components = connected_components(red_window)
    digitizing_note = "rightmost red observation symbol in 2014-2016 window"
    if not components:
        components = connected_components(red)
        digitizing_note = "no 2014-2016 red symbol found; used latest red observation symbol"
    points = []
    for component in components:
        year = pixel_to_year(component["x"], plot, x0, x1)
        head = pixel_to_head(component["y"], plot, y0, y1)
        points.append((year, head, component))

    if not points:
        raise ValueError(f"No 2015 observations found for {plot['recordation_number']}")

    # -- Use the rightmost observed symbol in the 2014-2016 window.
    year, head, component = max(points, key=lambda item: item[0])
    return {
        **plot,
        "digitized_year": year,
        "observed_head_ft_msl": head,
        "plot_x_pixel": component["x"],
        "plot_y_pixel": component["y"],
        "n_red_components_2014_2016": len(points),
        "digitizing_note": digitizing_note,
    }


def add_locations(rows):
    wells = pd.read_csv(WELL_CSV, dtype={"recordation_number": str})
    wells = wells.set_index("recordation_number")
    transformer = Transformer.from_crs(SOURCE_EPSG, TARGET_EPSG, always_xy=True)

    out_rows = []
    for row in rows:
        match_note = "matched by recordation_number"
        recordation_number = row["recordation_number"]
        if not recordation_number or recordation_number not in wells.index:
            well = pd.Series(dtype=object)
            x_source = np.nan
            y_source = np.nan
            x_gflow = np.nan
            y_gflow = np.nan
            match_note = "no matching recordation_number in extracted well-location table"
        else:
            well = wells.loc[recordation_number]
            if isinstance(well, pd.DataFrame):
                well = well.iloc[0]
                match_note = "matched by recordation_number; duplicate table rows present"
            x_source = float(well["x_stateplane_nad27_ca_zone7_ft"])
            y_source = float(well["y_stateplane_nad27_ca_zone7_ft"])
            x_gflow, y_gflow = transformer.transform(x_source, y_source)

        out_rows.append(
            {
                "recordation_number": row["recordation_number"],
                "site_name": row["site_name"],
                "la_county_id": row["la_county_id"],
                "owner": well.get("owner", ""),
                "well_id": well.get("well_id", ""),
                "x_stateplane_nad27_ca_zone7_ft": "" if np.isnan(x_source) else round(x_source, 3),
                "y_stateplane_nad27_ca_zone7_ft": "" if np.isnan(y_source) else round(y_source, 3),
                "x_gflow_epsg2229_ft": "" if np.isnan(x_gflow) else round(x_gflow, 3),
                "y_gflow_epsg2229_ft": "" if np.isnan(y_gflow) else round(y_gflow, 3),
                "observed_head_2015_ft_msl": round(row["observed_head_ft_msl"], 1),
                "digitized_year": round(row["digitized_year"], 2),
                "source_figure": "Figure 5-11",
                "source_pdf_page": int(row["page"].split("-")[1].split(".")[0]),
                "location_match_note": match_note,
                "digitizing_note": row["digitizing_note"],
            }
        )
    return out_rows


def main():
    # -- Digitize the six observed-head panels from the rendered report figure.
    digitized = [digitize_plot(plot) for plot in PLOTS]
    rows = add_locations(digitized)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gflow_rows = []
    for row in rows:
        if row["x_gflow_epsg2229_ft"] == "":
            continue
        gflow_rows.append(
            {
                "x_gflow_epsg2229_ft": row["x_gflow_epsg2229_ft"],
                "y_gflow_epsg2229_ft": row["y_gflow_epsg2229_ft"],
                "head_ft_msl": row["observed_head_2015_ft_msl"],
                "label": row["recordation_number"],
                "site_name": row["site_name"],
            }
        )

    OUT_GFLOW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_GFLOW_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gflow_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gflow_rows)

    print(f"wrote: {OUT_CSV}")
    print(f"wrote: {OUT_GFLOW_CSV}")
    for row in rows:
        print(
            row["recordation_number"],
            row["site_name"],
            row["observed_head_2015_ft_msl"],
            row["digitized_year"],
        )


if __name__ == "__main__":
    main()
