"""Test Suite for NOAA GHCN-Hourly (GHCNh) Ingestion and Data Cleaning.

Tests:
- Parsing of pipe-delimited PSV format (FR1.1, FR1.3)
- Correct handling of NOAA missing sentinels (-999.9, -9999, etc.)
- Rejection of failed/suspect quality flags (FR1.2)
- Hourly deduplication and rolling accumulation calculations
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.ingestion.ghcnh_parser import GHCNhParser
from src.ingestion.mock_ghcnh_generator import MockGHCNhGenerator


class TestGHCNhIngestion(unittest.TestCase):

    def setUp(self):
        self.parser = GHCNhParser()

    def test_parse_real_schema_content(self):
        sample_psv = (
            "Station_ID|Station_name|Latitude|Longitude|Elevation|Year|Month|Day|Hour|Minute|"
            "temperature|dew_point_temperature|relative_humidity|wind_direction|wind_speed|"
            "wind_gust|station_level_pressure|sea_level_pressure|precipitation|"
            "precipitation_1_hour|precipitation_3_hour|precipitation_6_hour|"
            "precipitation_24_hour|present_weather|quality_flag\n"
            "USW00093037|ALBUQUERQUE INTL AP|35.0433|-106.6128|1619.1|2023|08|15|14|00|"
            "26.7|12.2|40.0|180|4.1|7.2|841.2|1012.4|0.0|0.0|||||V\n"
            "USW00093037|ALBUQUERQUE INTL AP|35.0433|-106.6128|1619.1|2023|08|15|15|00|"
            "28.1|13.0|39.0|190|5.5|8.8|840.1|1011.0|0.0|0.0|||||V\n"
            "USW00093037|ALBUQUERQUE INTL AP|35.0433|-106.6128|1619.1|2023|08|15|16|00|"
            "22.4|18.5|78.0|240|12.8|19.5|836.4|1006.2|34.8|34.8||||TSRA|V\n"
        )

        meta, obs = self.parser.parse_content(sample_psv, filename="test.psv")

        self.assertEqual(meta.station_id, "USW00093037")
        self.assertEqual(meta.station_name, "ALBUQUERQUE INTL AP")
        self.assertAlmostEqual(meta.latitude, 35.0433, places=3)
        self.assertAlmostEqual(meta.longitude, -106.6128, places=3)
        self.assertAlmostEqual(meta.elevation_m, 1619.1, places=1)

        self.assertEqual(len(obs), 3)
        self.assertEqual(obs[2].precipitation_1h_mm, 34.8)
        self.assertEqual(obs[2].present_weather, "TSRA")
        self.assertEqual(obs[2].temperature_c, 22.4)
        # Check rolling accumulation
        self.assertAlmostEqual(obs[2].precipitation_3h_mm, 34.8, places=1)

    def test_missing_sentinels_cleaned_to_none(self):
        """Verifies that -999.9, -9999, NA are cleaned and do not corrupt calculations."""
        sample_psv = (
            "Station_ID|Station_name|Latitude|Longitude|Elevation|Year|Month|Day|Hour|Minute|"
            "temperature|dew_point_temperature|relative_humidity|wind_direction|wind_speed|"
            "wind_gust|station_level_pressure|sea_level_pressure|precipitation|"
            "precipitation_1_hour|precipitation_3_hour|precipitation_6_hour|"
            "precipitation_24_hour|present_weather|quality_flag\n"
            "USW00093037|ALBUQUERQUE INTL AP|35.0433|-106.6128|1619.1|2023|08|15|14|00|"
            "-999.9|-9999|NA|999|-99.9|-999.9|9999.0|-9999.9|0.0|0.0|||||V\n"
        )
        _, obs = self.parser.parse_content(sample_psv)
        self.assertEqual(len(obs), 1)
        self.assertIsNone(obs[0].temperature_c)
        self.assertIsNone(obs[0].dew_point_c)
        self.assertIsNone(obs[0].relative_humidity)
        self.assertIsNone(obs[0].wind_direction_deg)
        self.assertIsNone(obs[0].wind_speed_mps)

    def test_qc_flag_filtering(self):
        """Verifies that records flagged with Q (failed quality control) are discarded."""
        sample_psv = (
            "Station_ID|Station_name|Latitude|Longitude|Elevation|Year|Month|Day|Hour|Minute|"
            "temperature|dew_point_temperature|relative_humidity|wind_direction|wind_speed|"
            "wind_gust|station_level_pressure|sea_level_pressure|precipitation|"
            "precipitation_1_hour|precipitation_3_hour|precipitation_6_hour|"
            "precipitation_24_hour|present_weather|quality_flag\n"
            "USW00093037|TEST|35.0|-106.0|1500|2023|08|15|14|00|20|10|50|180|5|8|850|1010|0|0|||||V\n"
            "USW00093037|TEST|35.0|-106.0|1500|2023|08|15|15|00|99|99|99|180|5|8|850|1010|90|90|||||Q\n"
        )
        _, obs = self.parser.parse_content(sample_psv)
        # The row with Q should be discarded!
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].hour, 14)


if __name__ == "__main__":
    unittest.main()
