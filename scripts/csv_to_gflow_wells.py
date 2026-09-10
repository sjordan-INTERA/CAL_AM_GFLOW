from pathlib import Path
import csv
import re

from pyproj import Transformer


INPUT_CSV = Path("data/generated/gflow_wells_2014_15.csv")
OUTPUT_WL = Path("data/generated/gflow_wells_2014_15.wl")
OUTPUT_GROUPED_CSV = Path("data/generated/gflow_wells_2014_15_grouped_export_table.csv")

SOURCE_EPSG = 26799  # NAD27 / California zone VII, Los Angeles, ftUS
TARGET_EPSG = 2229  # NAD83 / California zone 5, ftUS

RADIUS_FT = 1.0
USE_NEGATIVE_DISCHARGE = False
MINIMUM_PRODUCTION_AFY = 0.0


def number(value):
    return float(str(value).replace(",", "").strip())


def clean_label(value):
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return label.strip("_")


def read_well_rows():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    # -- Read the well rows exported from the PDF tables.
    rows = []
    with INPUT_CSV.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            production_afy = number(row["production_2014_15_afy"])
            if production_afy <= MINIMUM_PRODUCTION_AFY:
                continue
            rows.append(row)

    return rows


def transform_and_group_wells(rows):
    transformer = Transformer.from_crs(SOURCE_EPSG, TARGET_EPSG, always_xy=True)
    grouped = {}

    # -- Transform coordinates and sum all wells that land at the same point.
    for row in rows:
        x_source = number(row["x_stateplane_nad27_ca_zone7_ft"])
        y_source = number(row["y_stateplane_nad27_ca_zone7_ft"])
        x_target, y_target = transformer.transform(x_source, y_source)

        x_key = round(x_target, 3)
        y_key = round(y_target, 3)
        key = (x_key, y_key)

        production_afy = number(row["production_2014_15_afy"])
        discharge_cfd = production_afy * 43560.0 / 365.0
        if USE_NEGATIVE_DISCHARGE:
            discharge_cfd *= -1.0

        label = clean_label(row["recordation_number"])
        if key not in grouped:
            grouped[key] = {
                "x": x_key,
                "y": y_key,
                "discharge_cfd": 0.0,
                "production_afy": 0.0,
                "labels": [],
                "well_ids": [],
                "owners": [],
            }

        grouped[key]["discharge_cfd"] += discharge_cfd
        grouped[key]["production_afy"] += production_afy
        grouped[key]["labels"].append(label)
        grouped[key]["well_ids"].append(clean_label(row["well_id"]))
        grouped[key]["owners"].append(row["owner"])

    return grouped


def write_gflow_wl(grouped):
    OUTPUT_WL.parent.mkdir(parents=True, exist_ok=True)

    # -- Write GFLOW's simple well-list import format with Windows line endings.
    with OUTPUT_WL.open("w", newline="", encoding="utf-8") as well_file:
        for well in sorted(grouped.values(), key=lambda item: (item["x"], item["y"])):
            label = "_".join(well["labels"])
            line = (
                f"{well['x']:.3f},"
                f"{well['y']:.3f},"
                f"{well['discharge_cfd']:.6g},"
                f"{RADIUS_FT:.2f},"
                f"{label}"
            )
            well_file.write(line + "\r\n")


def write_duplicate_summary(grouped):
    summary_csv = OUTPUT_WL.with_name(OUTPUT_WL.stem + "_duplicate_summary.csv")

    # -- Write a small audit file showing which imported wells were merged.
    with summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "x_epsg2229_ft",
            "y_epsg2229_ft",
            "n_source_wells",
            "source_labels",
            "production_2014_15_afy",
            "discharge_ft3_day",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for well in sorted(grouped.values(), key=lambda item: (item["x"], item["y"])):
            if len(well["labels"]) == 1:
                continue
            writer.writerow(
                {
                    "x_epsg2229_ft": f"{well['x']:.3f}",
                    "y_epsg2229_ft": f"{well['y']:.3f}",
                    "n_source_wells": len(well["labels"]),
                    "source_labels": "_".join(well["labels"]),
                    "production_2014_15_afy": f"{well['production_afy']:.3f}",
                    "discharge_ft3_day": f"{well['discharge_cfd']:.3f}",
                }
            )

    return summary_csv


def write_grouped_export_table(grouped):
    OUTPUT_GROUPED_CSV.parent.mkdir(parents=True, exist_ok=True)

    # -- Write every final GFLOW well, including singleton and merged locations.
    with OUTPUT_GROUPED_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "gflow_label",
            "x_epsg2229_ft",
            "y_epsg2229_ft",
            "n_source_wells",
            "source_recordation_numbers",
            "source_well_ids",
            "owners",
            "production_2014_15_afy",
            "discharge_ft3_day",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for well in sorted(grouped.values(), key=lambda item: (item["x"], item["y"])):
            writer.writerow(
                {
                    "gflow_label": "_".join(well["labels"]),
                    "x_epsg2229_ft": f"{well['x']:.3f}",
                    "y_epsg2229_ft": f"{well['y']:.3f}",
                    "n_source_wells": len(well["labels"]),
                    "source_recordation_numbers": "_".join(well["labels"]),
                    "source_well_ids": "_".join(well["well_ids"]),
                    "owners": " | ".join(sorted(set(well["owners"]))),
                    "production_2014_15_afy": f"{well['production_afy']:.3f}",
                    "discharge_ft3_day": f"{well['discharge_cfd']:.3f}",
                }
            )


def main():
    # -- Build the GFLOW import file from the extracted report table data.
    rows = read_well_rows()
    grouped = transform_and_group_wells(rows)
    write_gflow_wl(grouped)
    summary_csv = write_duplicate_summary(grouped)
    write_grouped_export_table(grouped)

    total_production = sum(well["production_afy"] for well in grouped.values())
    total_discharge = sum(well["discharge_cfd"] for well in grouped.values())
    duplicate_groups = sum(1 for well in grouped.values() if len(well["labels"]) > 1)

    print(f"source wells: {len(rows)}")
    print(f"gflow wells after grouping: {len(grouped)}")
    print(f"duplicate coordinate groups: {duplicate_groups}")
    print(f"total production: {total_production:.3f} AFY")
    print(f"total discharge: {total_discharge:.3f} ft3/day")
    print(f"wrote: {OUTPUT_WL}")
    print(f"wrote: {summary_csv}")
    print(f"wrote: {OUTPUT_GROUPED_CSV}")


if __name__ == "__main__":
    main()
