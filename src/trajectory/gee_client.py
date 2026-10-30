"""Google Earth Engine (GEE) Geostationary Satellite Query Client.

Queries GOES-16 ABI / Himawari collections for spatial image patches centered on station coordinates.
Synthetic image synthesis is available only when explicitly enabled with
``simulation=True``. Initialization failures are never converted silently.
"""

from __future__ import annotations

import math
import os
import random
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GEEClient:
    """Interface to GEE for retrieving geostationary satellite image patches."""

    def __init__(
        self,
        collection_id: str = "NOAA/GOES/16/MCMIPF",
        bands: Optional[List[str]] = None,
        patch_size: int = 32,
        resolution_km: float = 2.0,
        simulated_missing_rate: float = 0.15,
        seed: int = 101,
        simulation: bool = False,
        project_id: Optional[str] = None,
    ):
        self.collection_id = collection_id
        self.bands = bands or ["CMI_C02", "CMI_C13", "CMI_C14"]
        self.patch_size = patch_size
        self.resolution_km = resolution_km
        self.simulated_missing_rate = simulated_missing_rate
        self.rng = random.Random(seed)
        self.is_gee_initialized = False
        self.simulation = simulation
        self.synthetic = simulation
        self.initialization_error: Optional[str] = None
        self.project_id = project_id or os.getenv("EARTHENGINE_PROJECT") or os.getenv("GEE_PROJECT")
        self._ee: Any = None

        if simulation:
            logger.warning(
                "GEEClient simulation mode enabled; satellite frames are synthetic "
                "and must not be reported as real coverage."
            )
            return

        try:
            import ee
            has_persistent_credentials = bool(ee.data.get_persistent_credentials())
            credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            service_account = os.getenv("EARTHENGINE_SERVICE_ACCOUNT")
            has_application_credentials = bool(credentials_file)
            if not has_persistent_credentials and not has_application_credentials:
                raise RuntimeError(
                    "Earth Engine credentials are not configured. Run "
                    "`earthengine authenticate` or configure "
                    "GOOGLE_APPLICATION_CREDENTIALS."
                )
            if not self.project_id:
                raise RuntimeError(
                    "Earth Engine Cloud project is not configured. Set "
                    "EARTHENGINE_PROJECT or pass project_id."
                )
            credentials = None
            if credentials_file:
                if not service_account:
                    raise RuntimeError(
                        "EARTHENGINE_SERVICE_ACCOUNT must be set when using "
                        "GOOGLE_APPLICATION_CREDENTIALS."
                    )
                credentials = ee.ServiceAccountCredentials(service_account, credentials_file)
            ee.Initialize(credentials=credentials, project=self.project_id)
            self._ee = ee
            self.is_gee_initialized = True
        except Exception as exc:
            self.initialization_error = f"{type(exc).__name__}: {exc}"
            logger.exception("GEE initialization failed: %s", self.initialization_error)
            raise RuntimeError(
                "GEEClient could not initialize. Pass simulation=True explicitly "
                "to use synthetic satellite frames."
            ) from exc

    def query_nearest_image(
        self,
        latitude: float,
        longitude: float,
        target_time: datetime,
        tolerance_seconds: float = 450.0,  # ±7.5 minutes
        storm_intensity: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Queries GEE for the nearest satellite image observation to target_time within tolerance.
        Returns patch dictionary with tensor data, matched timestamp, and metadata, or None if missing.
        """
        if not self.simulation:
            return self._query_gee_image(
                latitude=latitude,
                longitude=longitude,
                target_time=target_time,
                tolerance_seconds=tolerance_seconds,
            )
        # If real GEE is initialized, we could call ee.ImageCollection
        # When simulated, we reproduce real geostationary orbit dynamics
        # Check if an observation is available within tolerance
        time_jitter_seconds = self.rng.uniform(-300.0, 300.0)
        matched_time = target_time.timestamp() + time_jitter_seconds

        # Simulate geostationary gap / eclipse / missing scan
        if self.rng.random() < self.simulated_missing_rate:
            return None  # Frame is missing!

        # Check tolerance boundary (FR2.4)
        time_delta = abs(matched_time - target_time.timestamp())
        if time_delta > tolerance_seconds:
            return None  # Beyond tolerance window!

        matched_dt = datetime.fromtimestamp(matched_time, tz=timezone.utc)

        # Generate realistic 3-band patch: [Red_Visible, Clean_IR_10.3um, Longwave_IR_11.2um]
        # Channels: (C, H, W)
        patch = self._synthesize_patch(storm_intensity=storm_intensity)

        # Cloud top brightness temperature (IR ~200-240K indicates convective cloud tops)
        mean_ir = round(self.rng.uniform(205.0, 235.0) if storm_intensity > 10.0 else self.rng.uniform(270.0, 295.0), 1)
        cloud_frac = round(min(1.0, 0.2 + 0.8 * (storm_intensity / 50.0)), 2) if storm_intensity > 0 else 0.1

        return {
            "synthetic": self.synthetic,
            "satellite_id": "GOES-16",
            "collection": self.collection_id,
            "bands": self.bands,
            "matched_time_utc": matched_dt.isoformat(),
            "time_delta_seconds": round(time_delta, 1),
            "data": patch,  # List of shape [C, H, W]
            "patch_mean_ir": mean_ir,
            "cloud_fraction": cloud_frac,
        }

    def _query_gee_image(
        self,
        latitude: float,
        longitude: float,
        target_time: datetime,
        tolerance_seconds: float,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve and normalize one real Earth Engine image patch."""
        if not self.is_gee_initialized or self._ee is None:
            raise RuntimeError("Earth Engine client is not initialized.")

        start = target_time.timestamp() - tolerance_seconds
        end = target_time.timestamp() + tolerance_seconds
        try:
            point = self._ee.Geometry.Point([longitude, latitude])
            collection = (
                self._ee.ImageCollection(self.collection_id)
                .filterBounds(point)
                .filterDate(
                    datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
                    datetime.fromtimestamp(end, tz=timezone.utc).isoformat(),
                )
                .select(self.bands)
                .sort("system:time_start")
            )
            image = self._ee.Image(collection.first())
            image_info = image.getInfo()
            if not image_info:
                return None

            properties = image_info.get("properties", {})
            timestamp_ms = properties.get("system:time_start")
            if timestamp_ms is None:
                logger.warning("GEE image has no system:time_start near %s", target_time.isoformat())
                return None
            matched_time = datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=timezone.utc)
            time_delta = abs((matched_time - target_time).total_seconds())
            if time_delta > tolerance_seconds:
                return None

            radius_m = self.patch_size * self.resolution_km * 500.0
            region = point.buffer(radius_m).bounds()
            sampled = image.sampleRectangle(region=region, defaultValue=0).getInfo()
            patch = self._sampled_patch_to_tensor(sampled)
            source_id = properties.get("system:id") or image_info.get("id") or "GEE"
            return {
                "synthetic": False,
                "satellite_id": self._satellite_id(source_id),
                "collection": self.collection_id,
                "bands": self.bands,
                "matched_time_utc": matched_time.isoformat(),
                "time_delta_seconds": round(time_delta, 1),
                "data": patch,
                "patch_mean_ir": round(
                    sum(patch[min(1, len(patch) - 1)][r][c]
                        for r in range(self.patch_size)
                        for c in range(self.patch_size))
                    / (self.patch_size ** 2),
                    4,
                ),
                "cloud_fraction": None,
                "source_file_name": source_id,
            }
        except (ImportError, ModuleNotFoundError):
            raise
        except Exception as exc:
            logger.warning("GEE image query failed near %s: %s", target_time.isoformat(), exc)
            return None

    def _sampled_patch_to_tensor(self, sampled: Dict[str, Any]) -> List[List[List[float]]]:
        """Convert sampleRectangle band arrays to fixed-size normalized channels."""
        channels: List[List[List[float]]] = []
        for band in self.bands:
            values = sampled.get(band)
            if not isinstance(values, list) or not values or not isinstance(values[0], list):
                raise ValueError(f"GEE sampleRectangle returned no 2D data for band {band}.")
            height = len(values)
            width = len(values[0])
            if width == 0:
                raise ValueError(f"GEE sampleRectangle returned an empty band for {band}.")
            channel: List[List[float]] = []
            for row in range(self.patch_size):
                source_row = values[min(height - 1, int(row * height / self.patch_size))]
                output_row: List[float] = []
                for col in range(self.patch_size):
                    raw = float(source_row[min(width - 1, int(col * width / self.patch_size))])
                    normalized = raw if band == "CMI_C02" else (330.0 - raw) / 130.0
                    output_row.append(round(max(0.0, min(1.0, normalized)), 4))
                channel.append(output_row)
            channels.append(channel)
        return channels

    @staticmethod
    def _satellite_id(source_id: str) -> str:
        for satellite in ("GOES-19", "GOES-18", "GOES-16", "Himawari"):
            if satellite.replace("-", "") in source_id.replace("-", ""):
                return satellite
        return "GEE"

    def _synthesize_patch(self, storm_intensity: float = 0.0) -> List[List[List[float]]]:
        """
        Creates a synthetic 3D patch (3, H, W) normalized to [0, 1].
        Band 0: Visible reflectance (bright cloud tops)
        Band 1: Inverted IR (high cold clouds = bright)
        Band 2: Water Vapor (atmospheric moisture)
        """
        c, h, w = len(self.bands), self.patch_size, self.patch_size
        patch: List[List[List[float]]] = [[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(c)]

        center_y = h / 2.0
        center_x = w / 2.0
        radius = 8.0 + storm_intensity * 0.2

        for i in range(h):
            for j in range(w):
                dist = math.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                # Convective cloud core
                cloud_intensity = math.exp(-0.5 * (dist / max(1.0, radius)) ** 2) if storm_intensity > 0 else 0.05
                noise = self.rng.uniform(-0.05, 0.05)

                val_vis = max(0.0, min(1.0, cloud_intensity * 0.9 + noise + 0.1))
                val_ir = max(0.0, min(1.0, cloud_intensity * 0.95 + noise + 0.05))
                val_wv = max(0.0, min(1.0, cloud_intensity * 0.85 + noise + 0.15))

                patch[0][i][j] = round(val_vis, 4)
                patch[1][i][j] = round(val_ir, 4)
                patch[2][i][j] = round(val_wv, 4)

        return patch
