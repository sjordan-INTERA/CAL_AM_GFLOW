import csv
import pathlib
import re

import pdfplumber


PDF_PATH = "docs/MSGB Groundwater Model Development Report w Appendices (Stetson 2017).pdf"
OUT_DIR = pathlib.Path("data/generated")

# Well has pumpage, missing from location table, matched via map (Figure 5-10)
MANUAL_LOCATIONS = [
    {
        "record": "1902907",
        "well_id": "WILEY",
        "owner": "California American Water Company",
        "x": 4307635.274634279,
        "y": 4167516.3072442496,
        "ground_surface_ft": None,
        "source_page": "manual EPSG:3857 map coordinate",
    }
]


def clean(value):
    return re.sub(r"\s+", " ", (value or "").replace("\n", " ")).strip()


def record_variants(record):
    record = clean(record)
    variants = {record}
    changed = True
    while changed:
        changed = False
        for item in list(variants):
            candidates = []
            if len(item) > 7 and item[0].isdigit() and (
                item[1:].startswith("19") or item[1:].startswith("80")
            ):
                candidates.append(item[1:])
            if len(item) > 7 and item[-1].isdigit():
                base = item[:-1]
                if base.startswith(("19", "80", "X", "A")):
                    candidates.append(base)
            for candidate in candidates:
                if candidate not in variants:
                    variants.add(candidate)
                    changed = True
    return variants


def to_number(value):
    value = clean(value).replace(",", "")
    if value in ("", "--", "NA", "N/A"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_locations():
    # -- Extract the well-location tables from the report appendices.
    locations = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for page_number in range(216, 226):
            for table in pdf.pages[page_number - 1].extract_tables() or []:
                for row in table[1:]:
                    if not row or len(row) < 8:
                        continue
                    record = clean(row[0])
                    x = to_number(row[3])
                    y = to_number(row[4])
                    ground_surface = to_number(row[5])
                    if (
                        not record
                        or record.upper().startswith("TABLE")
                        or x is None
                        or y is None
                    ):
                        continue
                    locations.append(
                        {
                            "record": record,
                            "well_id": clean(row[1]),
                            "owner": clean(row[2]),
                            "x": x,
                            "y": y,
                            "ground_surface_ft": ground_surface,
                            "source_page": page_number,
                        }
                    )
    locations.extend(MANUAL_LOCATIONS)
    return locations


def read_production():
    # -- Extract the 2014-15 production tables from the report appendices.
    production = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for page_number in range(336, 350):
            for table in pdf.pages[page_number - 1].extract_tables() or []:
                for row in table[1:]:
                    if not row or len(row) < 6:
                        continue
                    record = clean(row[0])
                    if not record or record.upper().startswith(
                        ("SUBTOTAL", "TOTAL", "REPORT", "RECORDATION")
                    ):
                        continue
                    if not re.match(r"^[A-Z]*\d", record):
                        continue
                    afy = to_number(row[5])
                    if afy is None:
                        continue
                    production.append(
                        {
                            "record": record,
                            "appendix_name": clean(row[1]),
                            "afy_2014_15": afy,
                            "source_page": page_number,
                        }
                    )
    return production


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- Read the source tables and match production records to well locations.
    locations = read_locations()
    production = read_production()

    locations_by_key = {}
    for location in locations:
        for variant in record_variants(location["record"]):
            locations_by_key.setdefault(variant, []).append(location)

    rows = []
    unmatched = []
    ambiguous = []

    for item in production:
        matches = []
        seen = set()
        for variant in record_variants(item["record"]):
            for location in locations_by_key.get(variant, []):
                key = (
                    location["record"],
                    location["x"],
                    location["y"],
                    location["owner"],
                    location["well_id"],
                )
                if key not in seen:
                    seen.add(key)
                    matches.append(location)

        if len(matches) == 1:
            location = matches[0]
            afy = item["afy_2014_15"]
            cubic_feet_per_day = afy * 43560.0 / 365.0
            rows.append(
                {
                    "recordation_number": item["record"],
                    "location_recordation_number": location["record"],
                    "well_id": location["well_id"],
                    "appendix_name": item["appendix_name"],
                    "owner": location["owner"],
                    "x_stateplane_nad27_ca_zone7_ft": f"{location['x']:.0f}",
                    "y_stateplane_nad27_ca_zone7_ft": f"{location['y']:.0f}",
                    "ground_surface_ft_msl": ""
                    if location["ground_surface_ft"] is None
                    else f"{location['ground_surface_ft']:.0f}",
                    "production_2014_15_afy": f"{afy:.3f}".rstrip("0").rstrip("."),
                    "gflow_pumping_ft3_day_positive_extraction": f"{cubic_feet_per_day:.3f}",
                    "modflow_well_ft3_day_negative_extraction": f"{-cubic_feet_per_day:.3f}",
                    "production_pdf_page": item["source_page"],
                    "location_pdf_page": location["source_page"],
                }
            )
        elif len(matches) > 1:
            ambiguous.append(item)
        else:
            unmatched.append(item)

    rows.sort(key=lambda row: (row["owner"], row["well_id"], row["recordation_number"]))

    # -- Write the matched records and a smaller table for manual review.
    output_path = OUT_DIR / "gflow_wells_2014_15.csv"
    if rows:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    review_path = OUT_DIR / "gflow_wells_2014_15_unmatched_or_ambiguous.csv"
    review_written = True
    try:
        with review_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "status",
                    "recordation_number",
                    "appendix_name",
                    "production_2014_15_afy",
                    "production_pdf_page",
                ],
            )
            writer.writeheader()
            for item in unmatched:
                writer.writerow(
                    {
                        "status": "unmatched",
                        "recordation_number": item["record"],
                        "appendix_name": item["appendix_name"],
                        "production_2014_15_afy": item["afy_2014_15"],
                        "production_pdf_page": item["source_page"],
                    }
                )
            for item in ambiguous:
                writer.writerow(
                    {
                        "status": "ambiguous",
                        "recordation_number": item["record"],
                        "appendix_name": item["appendix_name"],
                        "production_2014_15_afy": item["afy_2014_15"],
                        "production_pdf_page": item["source_page"],
                    }
                )
    except PermissionError:
        review_written = False

    print(f"locations: {len(locations)}")
    print(f"production rows: {len(production)}")
    print(f"matched: {len(rows)}")
    print(f"unmatched: {len(unmatched)}")
    print(f"ambiguous: {len(ambiguous)}")
    print(f"sum matched afy: {sum(float(row['production_2014_15_afy']) for row in rows):.3f}")
    print(f"wrote: {output_path}")
    if review_written:
        print(f"wrote: {review_path}")
    else:
        print(f"skipped locked review file: {review_path}")


if __name__ == "__main__":
    main()
