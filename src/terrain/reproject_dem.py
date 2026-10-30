"""Reproject USGS DEM GeoTIFFs into a metric CRS for D8 calculations."""

from __future__ import annotations

import argparse
from pathlib import Path


def reproject_directory(
    source_directory: str | Path,
    destination_directory: str | Path,
    target_crs: str = "EPSG:32614",
    resolution_m: float = 30.0,
) -> list[Path]:
    """Create projected DEMs while preserving source nodata values."""
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform, reproject
    except ImportError as exc:
        raise RuntimeError("DEM reprojection requires rasterio.") from exc

    source_dir = Path(source_directory)
    destination_dir = Path(destination_directory)
    destination_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for source_path in sorted(
        list(source_dir.glob("*.tif")) + list(source_dir.glob("*.tiff"))
    ):
        if destination_dir.resolve() == source_path.parent.resolve():
            continue
        destination_path = destination_dir / f"{source_path.stem}_metric.tif"
        with rasterio.open(source_path) as source:
            if source.crs is None:
                raise ValueError(f"DEM has no CRS: {source_path}")
            transform, width, height = calculate_default_transform(
                source.crs,
                target_crs,
                source.width,
                source.height,
                *source.bounds,
                resolution=resolution_m,
            )
            profile = source.profile.copy()
            profile.update(
                crs=target_crs,
                transform=transform,
                width=width,
                height=height,
                compress="deflate",
                tiled=True,
                BIGTIFF="IF_SAFER",
            )
            with rasterio.open(destination_path, "w", **profile) as destination:
                for band in range(1, source.count + 1):
                    reproject(
                        source=rasterio.band(source, band),
                        destination=rasterio.band(destination, band),
                        src_transform=source.transform,
                        src_crs=source.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        src_nodata=source.nodata,
                        dst_nodata=source.nodata,
                        resampling=Resampling.bilinear,
                    )
        outputs.append(destination_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/dem")
    parser.add_argument("--destination", default="data/dem_projected")
    parser.add_argument("--target-crs", default="EPSG:32614")
    parser.add_argument("--resolution-m", type=float, default=30.0)
    args = parser.parse_args()
    outputs = reproject_directory(
        args.source, args.destination, args.target_crs, args.resolution_m
    )
    print(f"Created {len(outputs)} projected DEMs in {args.destination}")


if __name__ == "__main__":
    main()
