"""NOAA GHCN-Hourly (GHCNh) Parser and Data Cleaning Pipeline.

Implements Functional Requirements:
- FR1.1: Ingestion of NOAA GHCN-hourly records (station ID, lat/lon, timestamp,
  precip, temp, dew point/humidity, pressure, wind).
- FR1.2: Deduplication and cleaning (missing sentinels, QC flags, malformed rows).
- FR1.3: Conformance to NOAA GHCNh published pipe-separated values (PSV) schema.
"""

from __future__ import annotations

import csv
import gzip
import io
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Known NOAA GHCNh missing value sentinel representations
NOAA_MISSING_SENTINELS = {
    "-9999",
    "-9999.0",
    "-999.9",
    "-99.9",
    "9999",
    "9999.0",
    "999.9",
    "99.9",
    "999",
    "-999",
    "NA",
    "N/A",
    "null",
    "NULL",
    "",
}


@dataclass
class StationMetadata:
    """Metadata for an individual weather station."""
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    elevation_m: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GHCNhObservation:
    """Cleaned single-timestep observation from a GHCNh station."""
    station_id: str
    timestamp: str  # ISO 8601 UTC
    year: int
    month: int
    day: int
    hour: int
    minute: int
    latitude: float
    longitude: float
    elevation_m: float
    temperature_c: Optional[float]
    dew_point_c: Optional[float]
    relative_humidity: Optional[float]
    wind_direction_deg: Optional[int]
    wind_speed_mps: Optional[float]
    station_pressure_hpa: Optional[float]
    sea_level_pressure_hpa: Optional[float]
    precipitation_mm: Optional[float]
    precipitation_1h_mm: Optional[float]
    precipitation_3h_mm: Optional[float]
    precipitation_6h_mm: Optional[float]
    present_weather: str
    quality_flag: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GHCNhParser:
    """Parser for NOAA GHCN-Hourly (GHCNh) PSV files."""

    def __init__(
        self,
        reject_qc_flags: Optional[List[str]] = None,
        precipitation_fill: float = 0.0,
    ):
        """
        Args:
            reject_qc_flags: List of quality flags to discard (e.g. ['Q', 'X', 'M']).
            precipitation_fill: Default value when precip is not reported during clear skies (0.0 mm).
        """
        self.reject_qc_flags = set(reject_qc_flags or ["Q", "X", "M"])
        self.precipitation_fill = precipitation_fill

    @staticmethod
    def _parse_float(val: Optional[str]) -> Optional[float]:
        if val is None:
            return None
        cleaned = val.strip()
        if not cleaned or cleaned in NOAA_MISSING_SENTINELS:
            return None
        try:
            f = float(cleaned)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except ValueError:
            return None

    @staticmethod
    def _parse_int(val: Optional[str]) -> Optional[int]:
        if val is None:
            return None
        cleaned = val.strip()
        if not cleaned or cleaned in NOAA_MISSING_SENTINELS:
            return None
        try:
            return int(float(cleaned))
        except ValueError:
            return None

    @staticmethod
    def _normalize_header(header: List[str]) -> Dict[str, int]:
        """Maps canonical column keys to column indices, case-insensitively."""
        mapping: Dict[str, int] = {}
        for idx, col in enumerate(header):
            col_clean = col.strip().lower()
            mapping[col_clean] = idx
        return mapping

    def parse_file(self, file_path: str | Path) -> Tuple[StationMetadata, List[GHCNhObservation]]:
        """Parses a single GHCNh PSV file (.psv or .psv.gz)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"GHCNh file not found: {path}")

        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                content = f.read()
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

        return self.parse_content(content, filename=path.name)

    def parse_content(
        self, content: str, filename: str = "stream"
    ) -> Tuple[StationMetadata, List[GHCNhObservation]]:
        """Parses GHCNh PSV text content and returns station metadata + observations."""
        lines = content.strip().splitlines()
        if not lines:
            raise ValueError(f"Empty GHCNh content in {filename}")

        # Pipe-delimited reader
        reader = csv.reader(lines, delimiter="|")
        header = next(reader, None)
        if not header:
            raise ValueError(f"No header found in GHCNh content in {filename}")

        col_map = self._normalize_header(header)

        def find_col(*names: str) -> Optional[int]:
            for name in names:
                if name in col_map:
                    return col_map[name]
            return None

        # Mandatory columns
        station_col = find_col("station_id", "station")
        lat_col = find_col("latitude", "lat")
        lon_col = find_col("longitude", "lon")
        elev_col = find_col("elevation", "elev")
        station_name_col = find_col("station_name", "name")

        if station_col is None or lat_col is None or lon_col is None:
            raise ValueError(
                f"Missing critical station/coord columns in header: {header} in {filename}"
            )

        # Weather variable columns
        temp_col = find_col("temperature", "dry_bulb_temperature", "temp")
        dew_col = find_col("dew_point_temperature", "dew_point")
        rh_col = find_col("relative_humidity", "rh")
        wdir_col = find_col("wind_direction", "wind_dir")
        wspd_col = find_col("wind_speed", "wind_spd")
        stn_pres_col = find_col("station_level_pressure", "station_pressure")
        slp_col = find_col("sea_level_pressure", "sea_pressure")
        precip_col = find_col("precipitation", "precip")
        precip_1h_col = find_col("precipitation_1_hour", "precip_1h")
        precip_3h_col = find_col("precipitation_3_hour", "precip_3h")
        precip_6h_col = find_col("precipitation_6_hour", "precip_6h")
        wx_col = find_col("present_weather", "weather")
        qc_col = find_col("quality_flag", "qc_flag", "flag")

        # Time columns
        year_col = find_col("year")
        month_col = find_col("month")
        day_col = find_col("day")
        hour_col = find_col("hour")
        min_col = find_col("minute", "min")
        date_col = find_col("date", "timestamp")

        raw_observations: List[GHCNhObservation] = []
        station_meta: Optional[StationMetadata] = None

        def get_val(row: List[str], idx: Optional[int]) -> Optional[str]:
            if idx is not None and 0 <= idx < len(row):
                return row[idx].strip()
            return None

        for row_idx, row in enumerate(reader, start=2):
            if not row or not any(row):
                continue

            station_id = get_val(row, station_col)
            if not station_id:
                continue

            lat = self._parse_float(get_val(row, lat_col))
            lon = self._parse_float(get_val(row, lon_col))
            elev = self._parse_float(get_val(row, elev_col)) or 0.0
            station_name = get_val(row, station_name_col) or station_id

            if lat is None or lon is None:
                continue  # Skip unlocated record

            if station_meta is None:
                station_meta = StationMetadata(
                    station_id=station_id,
                    station_name=station_name,
                    latitude=lat,
                    longitude=lon,
                    elevation_m=elev,
                )

            # Check QC Flag
            qc = get_val(row, qc_col) or ""
            if qc in self.reject_qc_flags:
                continue  # Discard flagged suspicious or failed record

            # Parse timestamp
            year = self._parse_int(get_val(row, year_col))
            month = self._parse_int(get_val(row, month_col))
            day = self._parse_int(get_val(row, day_col))
            hour = self._parse_int(get_val(row, hour_col))
            minute = self._parse_int(get_val(row, min_col)) or 0

            if year is None or month is None or day is None or hour is None:
                date_str = get_val(row, date_col)
                if date_str:
                    try:
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        year, month, day, hour, minute = (
                            dt.year,
                            dt.month,
                            dt.day,
                            dt.hour,
                            dt.minute,
                        )
                    except ValueError:
                        continue
                else:
                    continue

            try:
                dt_utc = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
                iso_ts = dt_utc.isoformat()
            except (ValueError, OverflowError):
                continue

            # Weather variables
            temp = self._parse_float(get_val(row, temp_col))
            dew = self._parse_float(get_val(row, dew_col))
            rh = self._parse_float(get_val(row, rh_col))
            wdir = self._parse_int(get_val(row, wdir_col))
            wspd = self._parse_float(get_val(row, wspd_col))
            stn_p = self._parse_float(get_val(row, stn_pres_col))
            slp = self._parse_float(get_val(row, slp_col))

            precip = self._parse_float(get_val(row, precip_col))
            p_1h = self._parse_float(get_val(row, precip_1h_col))
            p_3h = self._parse_float(get_val(row, precip_3h_col))
            p_6h = self._parse_float(get_val(row, precip_6h_col))
            wx = get_val(row, wx_col) or ""

            # Standardize precipitation
            if p_1h is None and precip is not None:
                p_1h = precip
            elif p_1h is None and precip is None:
                p_1h = self.precipitation_fill

            obs = GHCNhObservation(
                station_id=station_id,
                timestamp=iso_ts,
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                latitude=lat,
                longitude=lon,
                elevation_m=elev,
                temperature_c=temp,
                dew_point_c=dew,
                relative_humidity=rh,
                wind_direction_deg=wdir,
                wind_speed_mps=wspd,
                station_pressure_hpa=stn_p,
                sea_level_pressure_hpa=slp,
                precipitation_mm=precip or 0.0,
                precipitation_1h_mm=p_1h or 0.0,
                precipitation_3h_mm=p_3h,
                precipitation_6h_mm=p_6h,
                present_weather=wx,
                quality_flag=qc,
            )
            raw_observations.append(obs)

        if station_meta is None:
            raise ValueError(f"Could not extract station metadata from {filename}")

        # Deduplicate and aggregate to hourly steps (FR1.2)
        cleaned_observations = self._clean_and_aggregate_hourly(raw_observations)
        return station_meta, cleaned_observations

    def _clean_and_aggregate_hourly(
        self, obs_list: List[GHCNhObservation]
    ) -> List[GHCNhObservation]:
        """
        Deduplicates multiple observations within the same hour (e.g. synoptic vs METAR).
        Sorts chronologically and computes 3-hour and 6-hour rolling accumulation if missing.
        """
        if not obs_list:
            return []

        # Sort by timestamp
        obs_list.sort(key=lambda o: (o.year, o.month, o.day, o.hour, o.minute))

        # Group by (year, month, day, hour)
        hourly_buckets: Dict[Tuple[int, int, int, int], List[GHCNhObservation]] = {}
        for obs in obs_list:
            key = (obs.year, obs.month, obs.day, obs.hour)
            hourly_buckets.setdefault(key, []).append(obs)

        hourly_records: List[GHCNhObservation] = []
        for (y, m, d, h), bucket in sorted(hourly_buckets.items()):
            # Best representative reading in the hour: prefer closest to :00 or valid precip reading
            # If multiple, sum precipitation within the hour, take last temp/pressure
            best = min(bucket, key=lambda o: abs(o.minute - 0))
            total_hourly_precip = max(
                (o.precipitation_1h_mm or 0.0) for o in bucket
            )

            # Construct clean hourly record
            clean_rec = GHCNhObservation(
                station_id=best.station_id,
                timestamp=datetime(y, m, d, h, 0, tzinfo=timezone.utc).isoformat(),
                year=y,
                month=m,
                day=d,
                hour=h,
                minute=0,
                latitude=best.latitude,
                longitude=best.longitude,
                elevation_m=best.elevation_m,
                temperature_c=best.temperature_c,
                dew_point_c=best.dew_point_c,
                relative_humidity=best.relative_humidity,
                wind_direction_deg=best.wind_direction_deg,
                wind_speed_mps=best.wind_speed_mps,
                station_pressure_hpa=best.station_pressure_hpa,
                sea_level_pressure_hpa=best.sea_level_pressure_hpa,
                precipitation_mm=total_hourly_precip,
                precipitation_1h_mm=total_hourly_precip,
                precipitation_3h_mm=None,
                precipitation_6h_mm=None,
                present_weather=best.present_weather,
                quality_flag=best.quality_flag,
            )
            hourly_records.append(clean_rec)

        # Compute rolling 3-hour and 6-hour accumulations
        for i, rec in enumerate(hourly_records):
            p3 = sum(hourly_records[j].precipitation_1h_mm or 0.0 for j in range(max(0, i - 2), i + 1))
            p6 = sum(hourly_records[j].precipitation_1h_mm or 0.0 for j in range(max(0, i - 5), i + 1))
            rec.precipitation_3h_mm = round(p3, 2)
            rec.precipitation_6h_mm = round(p6, 2)

        return hourly_records

    def ingest_directory(
        self, directory_path: str | Path
    ) -> Dict[str, Tuple[StationMetadata, List[GHCNhObservation]]]:
        """Scans a directory for all GHCNh PSV files and parses them."""
        dir_path = Path(directory_path)
        if not dir_path.exists():
            return {}

        results: Dict[str, Tuple[StationMetadata, List[GHCNhObservation]]] = {}
        patterns = ["GHCNh_*.psv", "GHCNh_*.psv.gz", "*.psv", "*.psv.gz"]
        found_files = []
        for pat in patterns:
            found_files.extend(dir_path.glob(pat))

        for file_path in sorted(set(found_files)):
            try:
                meta, obs = self.parse_file(file_path)
                results[meta.station_id] = (meta, obs)
            except Exception as e:
                # Log or note error, continue scanning
                print(f"[GHCNhParser] Warning: failed to parse {file_path}: {e}")

        return results
