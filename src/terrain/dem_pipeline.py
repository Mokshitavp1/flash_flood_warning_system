"""Station-Level Terrain Feature Extraction and Caching Pipeline.

Implements Functional Requirements:
- FR3.1: Retrieve DEM once per unique station location (cached across timesteps).
- FR3.2: Compute elevation, slope, aspect, D8 flow direction, and flow accumulation.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.ingestion.ghcnh_parser import StationMetadata
from src.terrain.d8_flow import D8FlowRouter


@dataclass
class StationTerrainFeatures:
    """Static topographic features for a specific weather station."""
    station_id: str
    latitude: float
    longitude: float
    elevation_m: float
    slope_degrees: float
    aspect_degrees: float
    flow_direction_d8: int
    flow_accumulation_cells: int
    flow_accumulation_log: float
    drainage_area_km2: float
    is_elevated_terrain: bool  # True when DEM-derived local relief exceeds the configured threshold
    relative_height_m: float = 0.0

    def to_feature_vector(self) -> List[float]:
        """Returns normalized 5D terrain feature vector for neural network branch (FR4.3)."""
        # [norm_relative_height, norm_slope, norm_aspect, log_flow_accum, is_elevated]
        norm_elev = min(1.0, max(0.0, (self.elevation_m - 1000.0) / 3000.0))
        norm_slope = min(1.0, self.slope_degrees / 60.0)
        norm_aspect = self.aspect_degrees / 360.0
        norm_accum = min(1.0, self.flow_accumulation_log / 12.0)
        return [
            round(min(1.0, max(0.0, self.relative_height_m / 500.0)), 4),
            round(norm_slope, 4),
            round(norm_aspect, 4),
            round(norm_accum, 4),
            1.0 if self.is_elevated_terrain else 0.0,
        ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DEMPipeline:
    """Manages DEM grids and static station terrain feature computation."""

    def __init__(
        self,
        grid_radius_cells: int = 15,
        cell_size_m: float = 30.0,
        elevated_threshold_m: float = 40.0,
        dem_directory: str | Path = "data/dem",
        dem_loader: Optional[
            Callable[[StationMetadata], Tuple[List[List[float]], int, int]]
        ] = None,
    ):
        self.grid_radius_cells = grid_radius_cells
        self.cell_size_m = cell_size_m
        self.elevated_threshold_m = elevated_threshold_m
        self.dem_directory = Path(dem_directory)
        self.dem_loader = dem_loader
        self.router = D8FlowRouter(cell_size_m=cell_size_m)
        self.station_cache: Dict[str, StationTerrainFeatures] = {}

    def is_station_elevated(self, station_meta: StationMetadata) -> bool:
        """Return the DEM-derived local-relief gate for a station."""
        return self.extract_station_terrain(station_meta).is_elevated_terrain

    def load_station_dem(
        self, station_meta: StationMetadata
    ) -> Tuple[List[List[float]], int, int]:
        """Load a real GeoTIFF window centered on the station."""
        if self.dem_loader is not None:
            return self.dem_loader(station_meta)
        try:
            import rasterio
            from rasterio.transform import rowcol
            from rasterio.warp import transform
        except ImportError as exc:
            raise RuntimeError(
                "Real DEM integration requires rasterio. Install dependencies "
                "and place USGS 3DEP GeoTIFFs under data/dem."
            ) from exc

        dem_files = sorted(
            list(self.dem_directory.glob("*.tif"))
            + list(self.dem_directory.glob("*.tiff"))
        )
        if not dem_files:
            raise FileNotFoundError(
                f"No DEM GeoTIFFs found in {self.dem_directory}. "
                "Add USGS 3DEP GeoTIFFs before running terrain extraction."
            )

        for dem_path in dem_files:
            with rasterio.open(dem_path) as dataset:
                if dataset.crs is None:
                    raise ValueError(f"DEM has no coordinate reference system: {dem_path}")
                if not dataset.crs.is_projected:
                    raise ValueError(
                        f"DEM must use a projected metric CRS for D8 routing: {dem_path} "
                        f"uses {dataset.crs}. Run src.terrain.reproject_dem first."
                    )

                x_values, y_values = transform(
                    "EPSG:4326",
                    dataset.crs,
                    [station_meta.longitude],
                    [station_meta.latitude],
                )
                x, y = x_values[0], y_values[0]
                row, col = rowcol(dataset.transform, x, y)
                if not (0 <= row < dataset.height and 0 <= col < dataset.width):
                    continue

                window = rasterio.windows.Window(
                    col_off=col - self.grid_radius_cells,
                    row_off=row - self.grid_radius_cells,
                    width=2 * self.grid_radius_cells + 1,
                    height=2 * self.grid_radius_cells + 1,
                )
                if (
                    window.col_off < 0
                    or window.row_off < 0
                    or window.col_off + window.width > dataset.width
                    or window.row_off + window.height > dataset.height
                ):
                    raise ValueError(
                        f"DEM window around {station_meta.station_id} reaches "
                        f"the raster edge: {dem_path}"
                    )

                values = dataset.read(1, window=window, masked=True)
                if values.mask.any():
                    raise ValueError(
                        f"DEM window around {station_meta.station_id} contains "
                        f"nodata cells: {dem_path}"
                    )

                cell_width = abs(dataset.transform.a)
                cell_height = abs(dataset.transform.e)
                if not math.isclose(cell_width, cell_height, rel_tol=1e-3):
                    raise ValueError(
                        f"DEM cells must be square for D8 routing; got "
                        f"{cell_width} x {cell_height} in {dem_path}"
                    )
                if not 0.1 <= cell_width <= 1000.0:
                    raise ValueError(
                        f"DEM cell size must be expressed in meters for D8 routing; "
                        f"got {cell_width} in {dem_path}"
                    )
                self.cell_size_m = cell_width
                self.router = D8FlowRouter(cell_size_m=self.cell_size_m)
                return values.astype(float).tolist(), self.grid_radius_cells, self.grid_radius_cells

        raise ValueError(
            f"No DEM GeoTIFF in {self.dem_directory} covers station "
            f"{station_meta.station_id} ({station_meta.latitude}, {station_meta.longitude})."
        )

    def extract_station_terrain(
        self, station_meta: StationMetadata
    ) -> StationTerrainFeatures:
        """
        Computes or retrieves cached terrain features for a station (FR3.1).
        Pulls DEM once per unique station location, not per-timestep.
        """
        st_id = station_meta.station_id
        if st_id in self.station_cache:
            return self.station_cache[st_id]

        dem, cr, cc = self.load_station_dem(station_meta)
        slope_grid, aspect_grid = self.router.compute_slope_and_aspect(dem)
        fdir_grid = self.router.compute_flow_direction(dem)
        accum_grid = self.router.compute_flow_accumulation(fdir_grid)

        elevation_m = round(dem[cr][cc], 2)
        slope_deg = slope_grid[cr][cc]
        aspect_deg = aspect_grid[cr][cc]
        fdir = fdir_grid[cr][cc]
        accum_cells = accum_grid[cr][cc]
        log_accum = round(math.log(max(1.0, float(accum_cells))), 3)
        cell_area_km2 = (self.cell_size_m * self.cell_size_m) / 1e6
        drainage_km2 = round(accum_cells * cell_area_km2, 4)
        local_lowest_elevation = min(min(row) for row in dem)
        relative_height_m = round(elevation_m - local_lowest_elevation, 2)
        is_elevated = relative_height_m >= self.elevated_threshold_m

        features = StationTerrainFeatures(
            station_id=st_id,
            latitude=station_meta.latitude,
            longitude=station_meta.longitude,
            elevation_m=elevation_m,
            slope_degrees=slope_deg,
            aspect_degrees=aspect_deg,
            flow_direction_d8=fdir,
            flow_accumulation_cells=accum_cells,
            flow_accumulation_log=log_accum,
            drainage_area_km2=drainage_km2,
            is_elevated_terrain=is_elevated,
            relative_height_m=relative_height_m,
        )

        self.station_cache[st_id] = features
        return features

    def get_downstream_path_from_station(
        self, station_meta: StationMetadata
    ) -> List[Dict[str, Any]]:
        """Calculates downstream flow trajectory from the station location."""
        dem, cr, cc = self.load_station_dem(station_meta)
        fdir_grid = self.router.compute_flow_direction(dem)
        return self.router.trace_downstream_path(fdir_grid, dem, cr, cc)
