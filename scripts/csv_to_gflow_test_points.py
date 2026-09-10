from pathlib import Path
import csv
import math


DATASETS = [
    (
        Path("data/generated/figure_5_11_digitized_heads_2015.csv"),
        Path("GFLOW/figure_5_11_observed_heads_2015.tp"),
    ),
    (
        Path("data/generated/raymond_basin_observed_water_levels.csv"),
        Path("GFLOW/raymond_basin_observed_water_levels.tp"),
    ),
]

DEFAULT_RADIUS = 0.0
POINT_TYPE = "Piezometer"


def number(value):
    if value is None or str(value).strip() == "":
        return math.nan
    return float(str(value).replace(",", "").strip())


def read_head_rows(input_csv):
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    # -- Read observed heads that have GFLOW coordinates.
    rows = []
    with input_csv.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            x = number(row["x_gflow_epsg2229_ft"])
            y = number(row["y_gflow_epsg2229_ft"])
            head = number(row["observed_head_2015_ft_msl"])
            if math.isnan(x) or math.isnan(y) or math.isnan(head):
                continue
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "head": head,
                    "label": row["recordation_number"],
                }
            )

    return rows


def write_test_points(rows, output_tp):
    output_tp.parent.mkdir(parents=True, exist_ok=True)

    # -- Write GFLOW test point format matching the 2023 piezometer file.
    with output_tp.open("w", newline="", encoding="utf-8") as tp_file:
        for row in rows:
            line = (
                f"{row['x']:.2f},"
                f"{row['y']:.2f},"
                f"{row['head']:.1f},"
                f"{DEFAULT_RADIUS:.1f},"
                f"{POINT_TYPE},"
                f"{row['label']}"
            )
            tp_file.write(line + "\r\n")


def main():
    # -- Convert each observed-head CSV to a GFLOW test-point file.
    for input_csv, output_tp in DATASETS:
        rows = read_head_rows(input_csv)
        write_test_points(rows, output_tp)

        print(f"source rows with coordinates: {len(rows)}")
        print(f"wrote: {output_tp}")


if __name__ == "__main__":
    main()
