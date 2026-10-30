"""Public NOAA AWS GOES-16 ABI image client.

The client downloads ABI MCMIPF NetCDF files into a local cache and extracts
station-centered three-band patches. Missing files remain explicit ``None``
results for the trajectory mask.
"""

from __future__ import annotations

import logging
import re
import os
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NOAAAWSGOESClient:
    """Fetch real GOES-16 ABI MCMIPF data from the public NOAA S3 bucket."""

    bucket_url = "https://noaa-goes16.s3.amazonaws.com"  # GOES-16 (kept for compatibility)
    goes19_bucket_url = "https://noaa-goes19.s3.amazonaws.com"
    # GOES-19 became operational GOES-East on 2025-04-07 15:10 UTC and GOES-16
    # distribution then stopped, so later times must come from the GOES-19 bucket.
    goes19_cutover = datetime(2025, 4, 7, 15, 10, tzinfo=timezone.utc)
    filename_time = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")
    satellite_in_key = re.compile(r"_G(\d{2})_")

    @classmethod
    def _satellite_for_time(cls, when: datetime) -> tuple[str, str]:
        """Return (satellite_id, bucket_url) of the GOES-East satellite at ``when``."""
        if when >= cls.goes19_cutover:
            return "GOES-19", cls.goes19_bucket_url
        return "GOES-16", cls.bucket_url

    @classmethod
    def _bucket_for_key(cls, key: str) -> str:
        match = cls.satellite_in_key.search(Path(key).name)
        if match and match.group(1) == "19":
            return cls.goes19_bucket_url
        return cls.bucket_url

    def __init__(
        self,
        bands: Optional[List[str]] = None,
        patch_size: int = 32,
        resolution_km: float = 2.0,
        cache_directory: str | Path = "data/goes_cache",
    ):
        self.bands = bands or ["CMI_C02", "CMI_C13", "CMI_C14"]
        self.patch_size = patch_size
        self.resolution_km = resolution_km
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.synthetic = False
        self.simulation = False
        try:
            import certifi
            self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ca_file = os.getenv("SSL_CERT_FILE")
            self.ssl_context = (
                ssl.create_default_context(cafile=ca_file)
                if ca_file
                else ssl.create_default_context()
            )

    def query_nearest_image(
        self,
        latitude: float,
        longitude: float,
        target_time: datetime,
        tolerance_seconds: float = 450.0,
        storm_intensity: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        del storm_intensity
        target_time = target_time.astimezone(timezone.utc)
        satellite_id, _ = self._satellite_for_time(target_time)
        # Recoverable per-frame failures (network, corrupt file, patch off the
        # raster edge) become an explicit None so the manifest marks the frame
        # "missing" (NFR1). Missing-dependency RuntimeErrors are NOT caught:
        # they are setup errors and must stay loud.
        try:
            candidates = self._list_candidates(target_time)
            if not candidates:
                return None
            candidate = min(
                candidates,
                key=lambda item: abs((item["time"] - target_time).total_seconds()),
            )
            delta = abs((candidate["time"] - target_time).total_seconds())
            if delta > tolerance_seconds:
                return None
            path = self._download(candidate["key"])
            patch = self._read_patch(path, latitude, longitude)
        except (OSError, ET.ParseError, ValueError) as exc:
            logger.warning(
                "GOES frame unavailable for %s at (%.3f, %.3f): %s: %s",
                target_time.isoformat(), latitude, longitude,
                type(exc).__name__, exc,
            )
            return None
        return {
            "satellite_id": satellite_id,
            "collection": "NOAA AWS ABI-L2-MCMIPF",
            "bands": self.bands,
            "matched_time_utc": candidate["time"].isoformat(),
            "time_delta_seconds": round(delta, 1),
            "data": patch,
            "patch_mean_ir": round(sum(patch[1][r][c] for r in range(self.patch_size) for c in range(self.patch_size)) / (self.patch_size ** 2), 4),
            "cloud_fraction": None,
            "source_file_name": path.name,
        }

    def _list_candidates(self, target_time: datetime) -> List[Dict[str, Any]]:
        year = target_time.year
        day = target_time.timetuple().tm_yday
        prefix = f"ABI-L2-MCMIPF/{year}/{day:03d}/{target_time.hour:02d}/"
        query = urllib.parse.urlencode({"list-type": "2", "prefix": prefix})
        _, bucket_url = self._satellite_for_time(target_time)
        with urllib.request.urlopen(
            f"{bucket_url}/?{query}", timeout=60, context=self.ssl_context
        ) as response:
            root = ET.fromstring(response.read())
        keys = [
            item.text
            for item in root.findall("{*}Contents/{*}Key")
            if item.text and item.text.endswith(".nc")
        ]
        results = []
        for key in keys:
            match = self.filename_time.search(key)
            if not match:
                continue
            y, doy, hour, minute, second = (int(value) for value in match.groups())
            results.append(
                {
                    "key": key,
                    "time": datetime(y, 1, 1, tzinfo=timezone.utc)
                    + timedelta(days=doy - 1, hours=hour, minutes=minute, seconds=second),
                }
            )
        return results

    def _download(self, key: str) -> Path:
        destination = self.cache_directory / Path(key).name
        if destination.exists():
            try:
                from netCDF4 import Dataset
                with Dataset(destination):
                    pass
                return destination
            except Exception:
                destination.unlink()
        url = f"{self._bucket_for_key(key)}/{urllib.parse.quote(key)}"
        partial = destination.with_suffix(destination.suffix + ".part")
        if partial.exists():
            partial.unlink()
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=120, context=self.ssl_context) as response:
                with open(partial, "wb") as output:
                    output.write(response.read())
            from netCDF4 import Dataset
            with Dataset(partial):
                pass
            os.replace(partial, destination)
        except Exception:
            if partial.exists():
                partial.unlink()
            raise
        return destination

    def _read_patch(self, path: Path, latitude: float, longitude: float) -> List[List[List[float]]]:
        try:
            import numpy as np
            from netCDF4 import Dataset
            from pyproj import CRS, Transformer
        except ImportError as exc:
            raise RuntimeError(
                "Real NOAA GOES ingestion requires netCDF4 and pyproj."
            ) from exc

        with Dataset(path) as dataset:
            projection = dataset["goes_imager_projection"]
            crs = CRS.from_proj4(
                f"+proj=geos +lon_0={projection.longitude_of_projection_origin} "
                f"+h={projection.perspective_point_height} "
                f"+x_0=0 +y_0=0 +a={projection.semi_major_axis} "
                f"+b={projection.semi_minor_axis} +sweep={projection.sweep_angle_axis}"
            )
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            x_m, y_m = transformer.transform(longitude, latitude)
            x = np.asarray(dataset["x"][:])
            y = np.asarray(dataset["y"][:])
            x_index = int(np.abs(x - (x_m / projection.perspective_point_height)).argmin())
            y_index = int(np.abs(y - (y_m / projection.perspective_point_height)).argmin())
            radius = self.patch_size // 2
            y0, y1 = y_index - radius, y_index + radius
            x0, x1 = x_index - radius, x_index + radius
            if y0 < 0 or x0 < 0 or y1 > len(y) or x1 > len(x):
                raise ValueError(f"GOES patch reaches raster edge in {path.name}")

            channels: List[List[List[float]]] = []
            for band in self.bands:
                values = np.asarray(dataset[band][y0:y1, x0:x1], dtype="float32")
                if band == "CMI_C02":
                    normalized = np.clip(values, 0.0, 1.0)
                else:
                    normalized = np.clip((330.0 - values) / 130.0, 0.0, 1.0)
                channels.append(normalized.tolist())
            return channels