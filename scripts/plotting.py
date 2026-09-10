# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:51:28 2026

Project plotting script

@author: shjordan
"""

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import contextily as cx
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds


def load_well_geometry():
    well_csv = Path('..') / 'data' / 'generated' / 'gflow_wells_2014_15_nonzero.csv'
    wells = pd.read_csv(well_csv)
    well_geo = gpd.GeoDataFrame(
        data=wells,
        geometry=[
            Point(x,y) for x,y in zip(
                wells['x_stateplane_nad27_ca_zone7_ft'],
                wells['y_stateplane_nad27_ca_zone7_ft']
                )
            ],
        crs=26799
    )
    return well_geo


def load_obs_geometry():
    obs_csv = Path('..') / 'data' / 'generated' / 'figure_5_11_digitized_heads_2015.csv'
    obs = pd.read_csv(obs_csv)
    obs = obs.dropna(subset='x_stateplane_nad27_ca_zone7_ft')
    obs_geo = gpd.GeoDataFrame(
        data=obs,
        geometry=[
            Point(x,y) for x,y in zip(
                obs['x_stateplane_nad27_ca_zone7_ft'],
                obs['y_stateplane_nad27_ca_zone7_ft']
                )
            ],
        crs=26799
    )
    return obs_geo


def build_contours(gdf,basin,smooth_cells=8):
    gdf = gdf.dropna(subset=[
        "x_stateplane_nad27_ca_zone7_ft",
        "y_stateplane_nad27_ca_zone7_ft",
        "observed_head_2015_ft_msl",
    ])
    xy = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    z = gdf["observed_head_2015_ft_msl"].values
    xmin, ymin, xmax, ymax = basin.total_bounds
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 500),
        np.linspace(ymax, ymin, 500)
    )
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
    
    # -- Build a smooth regional trend that can extend outside the observation hull.
    x0 = xy[:,0].mean()
    y0 = xy[:,1].mean()
    scale = np.max(np.ptp(xy, axis=0))
    x = (xy[:,0] - x0) / scale
    y = (xy[:,1] - y0) / scale
    gx = (grid_xy[:,0] - x0) / scale
    gy = (grid_xy[:,1] - y0) / scale
    trend_terms = np.column_stack([
        np.ones(len(x)),
        x,
        y,
        x * x,
        y * y,
        x * y,
    ])
    grid_terms = np.column_stack([
        np.ones(len(gx)),
        gx,
        gy,
        gx * gx,
        gy * gy,
        gx * gy,
    ])
    coefficients = np.linalg.lstsq(trend_terms, z, rcond=None)[0]
    trend = grid_terms @ coefficients
    residual = z - trend_terms @ coefficients
    
    # -- Add local residual detail, damped away from observations to avoid noisy extrapolation.
    n_neighbors = min(12, len(xy))
    tree = cKDTree(xy)
    distance, index = tree.query(grid_xy, k=n_neighbors)
    distance = np.atleast_2d(distance)
    index = np.atleast_2d(index)
    if distance.shape[0] == 1:
        distance = distance.T
        index = index.T
    bandwidth = scale / 8
    weights = np.exp(-0.5 * (distance / bandwidth) ** 2)
    local_residual = np.sum(weights * residual[index], axis=1) / np.sum(weights, axis=1)
    nearest_distance = distance[:,0]
    residual_decay = scale / 3
    zz = trend + 0.75 * local_residual * np.exp(-nearest_distance / residual_decay)
    zz = zz.reshape(xx.shape)
    
    # -- Smooth the gridded estimate so contours read as estimated basin-scale lines.
    zz = gaussian_filter(
        zz,
        sigma=smooth_cells,
        mode="nearest"
    )
    
    # -- Mask everything outside basin.
    xmin = xx.min()
    xmax = xx.max()
    ymin = yy.min()
    ymax = yy.max()
    nrows, ncols = zz.shape
    transform = from_bounds(
        xmin, ymin, xmax, ymax,
        ncols, nrows
    )
    mask = geometry_mask(
        basin.geometry,
        out_shape=zz.shape,
        transform=transform,
        invert=True,
        all_touched=False
    )
    zz_masked = np.where(mask, zz, np.nan)
    return xx,yy,zz_masked

    
def plot_wl_contours():
    # -- Load inputs
    well_geo = load_well_geometry()
    obs_geo = load_obs_geometry()
    basin = gpd.read_file(
        Path('..') / 'gis' / 'shp' / 'generated' / 'san_gabriel_basin_outline.shp'
    )
    basin = basin.to_crs(well_geo.crs)
    
    # -- Build figure
    fig,ax = plt.subplots(figsize=(11,8))
    well_geo.plot(
        ax=ax,
        color='tab:blue',
        label='Pumping Wells',
        edgecolor='k',
    )
    obs_geo.plot(
        ax=ax,
        color='tab:orange',
        label='Observation Wells',
        edgecolor='k',
    )
    basin.boundary.plot(
        ax=ax,
        color='k',
        label='San-Gabriel Basin'
    )
    
    # -- Add contours
    X,Y,Z = build_contours(obs_geo,basin)
    contour_interval = 20
    zmin = np.floor(np.nanmin(Z) / contour_interval) * contour_interval
    zmax = np.ceil(np.nanmax(Z) / contour_interval) * contour_interval
    levels = np.arange(zmin, zmax + contour_interval, contour_interval)
    contour = ax.contour(
        X, 
        Y, 
        Z, 
        levels=levels, 
        colors='k', 
        linewidths=1,
        label='Head Contours (ft)'
    )
    plt.clabel(
        contour, 
        inline=True, 
        fontsize=8,
    )
    
    # -- Add a basemap
    cx.add_basemap(
        ax=ax,
        crs=basin.crs,
        source="https://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer/tile/{z}/{y}/{x}",
    )
    cx.add_basemap(
        ax=ax,
        crs=basin.crs,
        source="https://basemap.nationalmap.gov/arcgis/rest/services/USGSHydroCached/MapServer/tile/{z}/{y}/{x}",
        zorder=0,
    )
    
    # -- Format figure
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        frameon=True,
        framealpha=1.0,
        ncols=3,
        bbox_to_anchor=[0.84,0]
    )
    plt.savefig(
        Path("..") / "figures" / "water_level_contours.png",
        dpi=250,
        bbox_inches='tight'
    )


def plot_pumpage_heatmap():
    pass


if __name__ == "__main__":
    plot_wl_contours()
