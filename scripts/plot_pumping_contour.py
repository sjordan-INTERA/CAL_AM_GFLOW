"""Plot a smooth 2014-15 pumping-intensity map for the San Gabriel Basin."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter


PROJECT_DIR = Path(__file__).resolve().parents[1]
WELL_CSV = PROJECT_DIR / "data" / "generated" / "gflow_wells_2014_15_nonzero.csv"
BASIN_SHP = PROJECT_DIR / "gis" / "shp" / "generated" / "san_gabriel_basin_outline.shp"
OUTPUT_FIGURE = PROJECT_DIR / "figures" / "pumping_contour_2014_15.png"

WELL_CRS = "EPSG:26799"  # NAD27 / California zone VII (ftUS)
GRID_SIZE = 450
SMOOTHING_MILES = 0.5
FEET_PER_MILE = 5280.0


def load_map_data():
    """Load pumping wells and transform them to the basin coordinate system."""
    # -- Read the basin and pumping table.
    basin = gpd.read_file(BASIN_SHP)
    wells = pd.read_csv(WELL_CSV)

    required_columns = {
        "x_stateplane_nad27_ca_zone7_ft",
        "y_stateplane_nad27_ca_zone7_ft",
        "production_2014_15_afy",
    }
    missing_columns = required_columns.difference(wells.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required well columns: {missing}")

    # -- Convert the NAD27 well coordinates to the basin CRS (EPSG:2229).
    wells = wells.dropna(subset=list(required_columns)).copy()
    wells = gpd.GeoDataFrame(
        wells,
        geometry=gpd.points_from_xy(
            wells["x_stateplane_nad27_ca_zone7_ft"],
            wells["y_stateplane_nad27_ca_zone7_ft"],
        ),
        crs=WELL_CRS,
    ).to_crs(basin.crs)

    return wells, basin


def build_pumping_surface(wells, basin):
    """Spread well pumping over a regular grid with a Gaussian kernel."""
    # -- Define a basin-sized raster and place each well's annual pumping in it.
    xmin, ymin, xmax, ymax = basin.total_bounds
    x_edges = np.linspace(xmin, xmax, GRID_SIZE + 1)
    y_edges = np.linspace(ymin, ymax, GRID_SIZE + 1)
    pumping, _, _ = np.histogram2d(
        wells.geometry.y,
        wells.geometry.x,
        bins=(y_edges, x_edges),
        weights=wells["production_2014_15_afy"],
    )

    # Raster rows run from north to south; histogram rows run south to north.
    pumping = np.flipud(pumping)
    cell_width = (xmax - xmin) / GRID_SIZE
    cell_height = (ymax - ymin) / GRID_SIZE

    # -- Smooth the pumping totals and express the result per square mile.
    smoothing_feet = SMOOTHING_MILES * FEET_PER_MILE
    sigma = (smoothing_feet / cell_height, smoothing_feet / cell_width)
    pumping = gaussian_filter(pumping, sigma=sigma, mode="constant")
    cell_area_square_miles = cell_width * cell_height / FEET_PER_MILE**2
    pumping_density = pumping / cell_area_square_miles

    # -- Mask raster cells outside the basin outline.
    transform = from_bounds(xmin, ymin, xmax, ymax, GRID_SIZE, GRID_SIZE)
    inside_basin = geometry_mask(
        basin.geometry,
        out_shape=pumping_density.shape,
        transform=transform,
        invert=True,
    )
    pumping_density = np.ma.masked_where(~inside_basin, pumping_density)

    return pumping_density, (xmin, xmax, ymin, ymax)


def plot_pumping_contour():
    """Create and save the pumping-intensity map."""
    # -- Load inputs and calculate the smooth pumping surface.
    wells, basin = load_map_data()
    pumping_density, extent = build_pumping_surface(wells, basin)

    # -- Draw the pumping colormap and basin boundary.
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    image = ax.imshow(
        pumping_density,
        extent=extent,
        origin="upper",
        cmap="viridis",
        norm=PowerNorm(gamma=0.5, vmin=0, vmax=pumping_density.max()),
        alpha=0.78,
        interpolation="bilinear",
        zorder=1,
    )
    basin.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=2)

    # -- Overlay wells, using square-root scaling to balance marker sizes.
    pumping_afy = wells["production_2014_15_afy"].to_numpy()
    marker_sizes = 8 + 70 * np.sqrt(pumping_afy / pumping_afy.max())
    ax.scatter(
        wells.geometry.x,
        wells.geometry.y,
        s=marker_sizes,
        facecolor="deepskyblue",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.85,
        zorder=3,
    )

    # -- Format and save the figure.
    colorbar = fig.colorbar(image, ax=ax, shrink=0.6, pad=0.02)
    colorbar.set_label("Pumping Intensity")
    ax.set_title("San Gabriel Basin Pumping, 2014-15")
    ax.set_axis_off()
    fig.tight_layout()

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapped {len(wells)} pumping wells")
    print(f"Saved figure to: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    plot_pumping_contour()
