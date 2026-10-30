# USGS 3DEP DEM input

Place the downloaded USGS 3DEP GeoTIFF tiles for the study area in this
directory. These are source rasters. Run
`python -m src.terrain.reproject_dem` to create metric rasters in
`data/dem_projected`; the pipeline reads the projected directory.

The pipeline intentionally fails if:

- no GeoTIFF exists here;
- a station is outside every GeoTIFF;
- the station window reaches the raster edge;
- the window contains nodata cells;
- the raster has no CRS; or
- the raster cells are not square; or
- the raster CRS is geographic rather than projected in meters.

Keep the source download
metadata, coverage, resolution, and checksum with the dataset outside the
source tree if the GeoTIFFs are not committed.
