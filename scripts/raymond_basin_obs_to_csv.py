from pathlib import Path

import geopandas as gpd


INPUT_SHP = Path("gis/shp/generated/raymond_basin_obs.shp")
OUTPUT_CSV = Path("data/generated/raymond_basin_observed_water_levels.csv")
EXPECTED_CRS = "EPSG:2229"


def main():
    if not INPUT_SHP.exists():
        raise FileNotFoundError(f"Missing input shapefile: {INPUT_SHP}")

    # -- Read and validate the Raymond Basin observation points.
    observations = gpd.read_file(INPUT_SHP)
    if observations.crs is None or observations.crs.to_epsg() != 2229:
        raise ValueError(f"Expected {EXPECTED_CRS}, found {observations.crs}")
    if not observations.geometry.geom_type.eq("Point").all():
        raise ValueError("All Raymond Basin observations must be point geometries")

    required_columns = ["name", "waterLevel"]
    missing_columns = [
        column for column in required_columns if column not in observations.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing shapefile fields: {', '.join(missing_columns)}")

    valid = observations[required_columns + ["geometry"]].dropna().copy()
    if len(valid) != len(observations):
        raise ValueError("Raymond Basin observations contain missing names, heads, or geometry")

    # -- Match the column names consumed by csv_to_gflow_test_points.py.
    output = valid.assign(
        recordation_number=valid["name"],
        x_gflow_epsg2229_ft=valid.geometry.x,
        y_gflow_epsg2229_ft=valid.geometry.y,
        observed_head_2015_ft_msl=valid["waterLevel"],
    )[
        [
            "recordation_number",
            "x_gflow_epsg2229_ft",
            "y_gflow_epsg2229_ft",
            "observed_head_2015_ft_msl",
        ]
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    print(f"source points: {len(observations)}")
    print(f"wrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
