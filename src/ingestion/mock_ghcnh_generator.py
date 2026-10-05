"""Synthetic NOAA GHCN-hourly (GHCNh) Generator for Pipeline Testing.

Generates authentic synthetic GHCNh PSV files conforming to NOAA's published schema.
Supports:
- Elevated mountain stations experiencing high-intensity convective rainfall
- Downhill valley stations (which may experience zero local rainfall while floodwaters travel downhill)
- Realistic meteorological dynamics: sharp pressure drops before storms, humidity spikes,
  and convective precipitation bursts.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SAMPLE_STATIONS: List[Dict[str, Any]] = [
    {
        "station_id": "USW00093037",
        "station_name": "ALBUQUERQUE FOOTHILLS MOUNTAIN",
        "latitude": 35.1522,
        "longitude": -106.4851,
        "elevation_m": 2420.0,
        "is_elevated": True,
        "storm_peak_hour": 16,
        "storm_intensity_mm": 48.5,
    },
    {
        "station_id": "USW00023188",
        "station_name": "SANDIA CREST PEAK HIGH",
        "latitude": 35.2103,
        "longitude": -106.4497,
        "elevation_m": 3255.0,
        "is_elevated": True,
        "storm_peak_hour": 15,
        "storm_intensity_mm": 54.0,
    },
    {
        "station_id": "USW00013019",
        "station_name": "TIJERAS CANYON DOWNHILL VALLEY",
        "latitude": 35.0841,
        "longitude": -106.3812,
        "elevation_m": 1920.0,
        "is_elevated": False,
        "storm_peak_hour": None,  # Zero local rain! Flood arrives from upstream
        "storm_intensity_mm": 0.0,
    },
    {
        "station_id": "USW00004122",
        "station_name": "RIO GRANDE BASIN LOWLAND",
        "latitude": 34.9810,
        "longitude": -106.6800,
        "elevation_m": 1510.0,
        "is_elevated": False,
        "storm_peak_hour": None,  # Downhill valley receptor
        "storm_intensity_mm": 0.0,
    },
    {
        "station_id": "USW00014023",
        "station_name": "SANTA FE RIDGELINE PASS",
        "latitude": 35.6870,
        "longitude": -105.9378,
        "elevation_m": 2680.0,
        "is_elevated": True,
        "storm_peak_hour": 18,
        "storm_intensity_mm": 38.0,
    },
]


class MockGHCNhGenerator:
    """Generates synthetic GHCNh PSV files."""

    HEADER = (
        "Station_ID|Station_name|Latitude|Longitude|Elevation|Year|Month|Day|Hour|Minute|"
        "temperature|dew_point_temperature|relative_humidity|wind_direction|wind_speed|"
        "wind_gust|station_level_pressure|sea_level_pressure|precipitation|"
        "precipitation_1_hour|precipitation_3_hour|precipitation_6_hour|"
        "precipitation_24_hour|present_weather|quality_flag"
    )

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_station_records(
        self,
        station_info: Dict[str, Any],
        start_date: datetime,
        num_hours: int = 48,
    ) -> str:
        """Generates PSV text for a single station over num_hours."""
        lines = [self.HEADER]

        st_id = station_info["station_id"]
        st_name = station_info["station_name"]
        lat = station_info["latitude"]
        lon = station_info["longitude"]
        elev = station_info["elevation_m"]
        peak_hour = station_info.get("storm_peak_hour")
        peak_intensity = station_info.get("storm_intensity_mm", 0.0)

        # Baseline weather based on elevation
        base_temp = 32.0 - (elev / 1000.0) * 6.5
        base_pressure = 1013.25 * math.exp(-elev / 8400.0)

        for step in range(num_hours):
            current_time = start_date + timedelta(hours=step)
            y = current_time.year
            m = current_time.month
            d = current_time.day
            h = current_time.hour
            minute = 0

            # Diurnal temperature cycle
            diurnal_temp = base_temp + 6.0 * math.sin((h - 8) * math.pi / 12)
            dew_point = diurnal_temp - 12.0
            rh = 40.0
            stn_pressure = base_pressure
            slp = 1014.0 + 2.0 * math.sin((h - 4) * math.pi / 12)
            wind_dir = (180 + int(self.rng.gauss(0, 20))) % 360
            wind_spd = max(1.0, round(self.rng.gauss(3.5, 1.2), 1))
            wind_gust = round(wind_spd + abs(self.rng.gauss(3.0, 1.5)), 1)
            precip = 0.0
            weather = ""
            qc_flag = "V"

            # Inject storm event if this station is set for convective rainfall
            if peak_hour is not None:
                # Event centered around step == 16 (or peak_hour)
                hours_from_peak = (step % 24) - peak_hour
                if abs(hours_from_peak) <= 2:
                    storm_fraction = math.exp(-0.5 * (hours_from_peak / 0.8) ** 2)
                    precip = round(peak_intensity * storm_fraction + self.rng.uniform(0, 2), 1)
                    diurnal_temp -= 8.0 * storm_fraction  # Rain cooling
                    dew_point += 4.0 * storm_fraction
                    rh = min(98.0, 50.0 + 45.0 * storm_fraction)
                    stn_pressure -= 7.0 * storm_fraction  # Convective low drop
                    slp -= 7.0 * storm_fraction
                    wind_spd += 9.0 * storm_fraction
                    wind_gust += 16.0 * storm_fraction
                    wind_dir = (wind_dir + 90) % 360
                    weather = "TSRA" if precip > 20.0 else "RA"

            # Introduce rare missing sentinel on dew_point or wind gust
            dew_str = f"{dew_point:.1f}"
            if self.rng.random() < 0.02:
                dew_str = "-999.9"

            gust_str = f"{wind_gust:.1f}"
            if self.rng.random() < 0.05:
                gust_str = "-999.9"

            row = [
                st_id,
                st_name,
                f"{lat:.4f}",
                f"{lon:.4f}",
                f"{elev:.1f}",
                str(y),
                f"{m:02d}",
                f"{d:02d}",
                f"{h:02d}",
                f"{minute:02d}",
                f"{diurnal_temp:.1f}",
                dew_str,
                f"{rh:.1f}",
                str(wind_dir),
                f"{wind_spd:.1f}",
                gust_str,
                f"{stn_pressure:.1f}",
                f"{slp:.1f}",
                f"{precip:.1f}",
                f"{precip:.1f}",
                "",  # 3h
                "",  # 6h
                "",  # 24h
                weather,
                qc_flag,
            ]
            lines.append("|".join(row))

        return "\n".join(lines) + "\n"

    def write_mock_dataset(
        self,
        output_dir: str | Path,
        start_date: Optional[datetime] = None,
        num_hours: int = 48,
    ) -> List[Path]:
        """Writes synthetic GHCNh PSV files for sample stations into output_dir."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if start_date is None:
            start_date = datetime(2023, 8, 15, 0, 0, tzinfo=timezone.utc)

        created_files: List[Path] = []
        for station in SAMPLE_STATIONS:
            filename = f"GHCNh_{station['station_id']}_{start_date.year}.psv"
            target = out_path / filename
            content = self.generate_station_records(
                station, start_date=start_date, num_hours=num_hours
            )
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            created_files.append(target)

        return created_files
