"""Test Suite for Alert Propagation (Method v0 Circular vs Method v1 DEM Flow Routing).

Tests:
- FR5.1: Circular-buffer baseline behavior (fixed radius, broadcasts uphill).
- FR5.2: DEM Flow-routing alert behavior (downstream path walk, distance decay, accumulation boost).
- FR5.3: Method attribution logging and audit trail.
- Elimination of false alarms in uphill directions.
"""

from __future__ import annotations

import unittest

from src.alerts.alert_logger import AlertLogger
from src.alerts.circular_buffer import CircularBufferAlertEngine
from src.alerts.flow_routing_alert import DEMFlowRoutingAlertEngine
from src.ingestion.ghcnh_parser import StationMetadata
from src.terrain.dem_pipeline import DEMPipeline


class TestAlerts(unittest.TestCase):

    def setUp(self):
        self.dem_pipeline = DEMPipeline(elevated_threshold_m=2000.0)
        self.logger = AlertLogger()

        # Trigger station: Mountain summit (elevated, 3200m)
        self.mountain_station = StationMetadata(
            station_id="STN_MOUNTAIN",
            station_name="SANDIA PEAK",
            latitude=35.2100,
            longitude=-106.4500,
            elevation_m=3200.0,
        )

        # Candidate 1: Downhill valley station along river drainage (1500m, 18km away)
        self.downhill_valley_station = StationMetadata(
            station_id="STN_VALLEY",
            station_name="ALBUQUERQUE VALLEY",
            latitude=35.0800,
            longitude=-106.6500,
            elevation_m=1500.0,
        )

        # Candidate 2: Uphill adjacent mountain ridge (3350m, 12km away)
        self.uphill_ridge_station = StationMetadata(
            station_id="STN_UPHILL",
            station_name="NORTH CREST RIDGE",
            latitude=35.2800,
            longitude=-106.4200,
            elevation_m=3350.0,
        )

    def test_circular_buffer_triggers_uphill_false_alarm(self):
        """FR5.1: Circular buffer baseline sends false alarms uphill because it ignores terrain."""
        circ_engine = CircularBufferAlertEngine(radius_km=25.0, trigger_risk_threshold=0.65)
        candidates = [self.downhill_valley_station, self.uphill_ridge_station]

        alerts = circ_engine.evaluate_and_propagate(
            trigger_station=self.mountain_station,
            trigger_risk_score=0.85,
            timestamp_utc="2023-08-15T16:00:00Z",
            candidate_stations=candidates,
            logger=self.logger,
        )

        # Circular buffer fires to BOTH stations because both are within 25km!
        target_ids = [a.target_station_id for a in alerts]
        self.assertIn("STN_VALLEY", target_ids)
        self.assertIn("STN_UPHILL", target_ids, "Circular buffer naively alerts uphill ridge stations!")

        # Verify method attribution logging (FR5.3)
        for a in alerts:
            self.assertEqual(a.method, "v0_circular_buffer")

    def test_dem_flow_routing_eliminates_uphill_false_alarm(self):
        """FR5.2: DEM Flow-routing alerts downstream valley, but NEVER alerts uphill peak."""
        flow_engine = DEMFlowRoutingAlertEngine(
            dem_pipeline=self.dem_pipeline,
            trigger_risk_threshold=0.65,
        )
        candidates = [self.downhill_valley_station, self.uphill_ridge_station]

        alerts = flow_engine.evaluate_and_propagate(
            trigger_station=self.mountain_station,
            trigger_risk_score=0.85,
            timestamp_utc="2023-08-15T16:00:00Z",
            candidate_stations=candidates,
            logger=self.logger,
        )

        # Flow routing MUST alert the downhill valley station
        # and MUST NOT alert the uphill ridge station!
        target_ids = [a.target_station_id for a in alerts]
        self.assertIn("STN_VALLEY", target_ids)
        self.assertNotIn("STN_UPHILL", target_ids, "DEM Flow routing must NOT alert uphill peaks!")

        valley_alert = next(a for a in alerts if a.target_station_id == "STN_VALLEY")
        self.assertEqual(valley_alert.method, "v1_dem_flow_routing")
        self.assertTrue(valley_alert.is_downhill)
        self.assertGreater(valley_alert.estimated_arrival_lead_time_minutes, 0.0)
        self.assertLessEqual(valley_alert.alert_strength, 0.85)


if __name__ == "__main__":
    unittest.main()
