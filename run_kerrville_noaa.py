#!/usr/bin/env python3
"""Build and clean a real GOES-East trajectory for Kerrville at 10Z."""

from __future__ import annotations

import argparse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.ingestion.ghcnh_parser import GHCNhObservation, StationMetadata
from src.trajectory.manifest import MissingFrameManifest
from src.trajectory.noaa_aws_client import NOAAAWSGOESClient
from src.trajectory.trajectory_builder import SatelliteTrajectoryBuilder


KERRVILLE = StationMetadata(
    station_id="USW00012961",
    station_name="KERRVILLE MUNI AP",
    latitude=29.9833,
    longitude=-99.0833,
    elevation_m=492.9,
)


def _observation() -> GHCNhObservation:
    return GHCNhObservation(
        station_id=KERRVILLE.station_id,
        timestamp="2025-07-04T10:00:00+00:00",
        year=2025,
        month=7,
        day=4,
        hour=10,
        minute=0,
        latitude=KERRVILLE.latitude,
        longitude=KERRVILLE.longitude,
        elevation_m=KERRVILLE.elevation_m,
        temperature_c=None,
        dew_point_c=None,
        relative_humidity=None,
        wind_direction_deg=None,
        wind_speed_mps=None,
        station_pressure_hpa=None,
        sea_level_pressure_hpa=None,
        precipitation_mm=24.1,
        precipitation_1h_mm=24.1,
        precipitation_3h_mm=None,
        precipitation_6h_mm=None,
        present_weather=None,
        quality_flag="V",
    )


def _preflight(
    client: NOAAAWSGOESClient,
    target_time: datetime,
    tolerance_seconds: float,
) -> tuple[list[dict[str, object]], int]:
    candidates: dict[str, dict[str, object]] = {}
    for offset in range(-120, 121, 30):
        target = target_time + timedelta(minutes=offset)
        options = client._list_candidates(target)
        if not options:
            continue
        selected = min(options, key=lambda item: abs((item["time"] - target).total_seconds()))
        if abs((selected["time"] - target).total_seconds()) > tolerance_seconds:
            continue
        candidates[str(selected["key"])] = selected

    total_bytes = 0
    for item in candidates.values():
        request = urllib.request.Request(
            f"{client.bucket_url}/{item['key']}", method="HEAD"
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            item["size_bytes"] = int(response.headers.get("Content-Length", "0"))
            total_bytes += int(item["size_bytes"])
    return list(candidates.values()), total_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/manifests/satellite_trajectory_manifest.csv")
    parser.add_argument("--max-gb", type=float, default=5.0)
    args = parser.parse_args()

    manifest_path = Path(args.output)
    manifest_path.unlink(missing_ok=True)
    client = NOAAAWSGOESClient(
        bands=["CMI_C02", "CMI_C13", "CMI_C14"],
        patch_size=32,
        cache_directory="data/goes_cache",
    )
    target_time = datetime(2025, 7, 4, 10, tzinfo=timezone.utc)
    items, total_bytes = _preflight(client, target_time, 450.0)
    total_gb = total_bytes / (1024 ** 3)
    print(f"GOES-East source: {client.bucket_url}")
    print(f"Candidate files to download: {len(items)}")
    print(f"Total download size: {total_gb:.3f} GB")
    if total_gb > args.max_gb:
        raise SystemExit(
            f"Aborting: preflight size {total_gb:.3f} GB exceeds {args.max_gb:.3f} GB."
        )

    manifest = MissingFrameManifest()
    builder = SatelliteTrajectoryBuilder(
        window_hours=2.0,
        step_minutes=30,
        tolerance_minutes=7.5,
        patch_size=32,
        gee_client=client,
    )
    builder.build_trajectory(KERRVILLE, _observation(), manifest)
    manifest.to_csv(manifest_path)
    downloaded = {
        record.source_file_name
        for record in manifest.records
        if record.source_file_name
    }
    for file_name in downloaded:
        path = client.cache_directory / file_name
        if path.exists():
            path.unlink()
    valid = [record for record in manifest.records if record.status == "ok"]
    print(f"Manifest rows: {len(manifest.records)}")
    print(f"Real frames: {len(valid)}")
    print(f"Missing frames: {len(manifest.records) - len(valid)}")
    print("Downloaded source files were deleted after patch extraction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
