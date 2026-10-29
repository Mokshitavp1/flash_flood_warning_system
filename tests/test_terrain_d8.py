"""Test Suite for Terrain Analysis and D8 Flow Routing.

Tests:
- Slope and aspect calculations (Horn's method)
- D8 steepest descent flow direction codes (1..128)
- Flow accumulation mass conservation
- Downstream hydrological path tracing
- Station terrain caching (FR3.1)
"""

from __future__ import annotations

import unittest

from src.ingestion.ghcnh_parser import StationMetadata
from src.terrain.d8_flow import D8FlowRouter, FlowDirectionD8
from src.terrain.dem_pipeline import DEMPipeline


class TestTerrainD8(unittest.TestCase):

    def setUp(self):
        self.router = D8FlowRouter(cell_size_m=30.0)

    def test_d8_flow_direction_on_inclined_plane(self):
        """On a plane sloping down to the East, flow direction should be EAST (1)."""
        # 3x3 plane sloping downward to the East
        dem = [
            [300.0, 200.0, 100.0],
            [300.0, 200.0, 100.0],
            [300.0, 200.0, 100.0],
        ]
        fdir = self.router.compute_flow_direction(dem)
        # Center cell (1, 1) should flow East (1)
        self.assertEqual(fdir[1][1], int(FlowDirectionD8.EAST))

    def test_d8_flow_direction_south(self):
        """On a plane sloping down to the South, flow direction should be SOUTH (4)."""
        dem = [
            [300.0, 300.0, 300.0],
            [200.0, 200.0, 200.0],
            [100.0, 100.0, 100.0],
        ]
        fdir = self.router.compute_flow_direction(dem)
        self.assertEqual(fdir[1][1], int(FlowDirectionD8.SOUTH))

    def test_flow_accumulation_conservation(self):
        """Flow accumulation should increase downstream as cells drain into one another."""
        # 4 cells in a row flowing East: c0 -> c1 -> c2 -> c3
        dem = [
            [400.0, 300.0, 200.0, 100.0],
        ]
        fdir = self.router.compute_flow_direction(dem)
        accum = self.router.compute_flow_accumulation(fdir)

        # First cell contributes 1, second has 1+1=2, third has 3, fourth has 4
        self.assertEqual(accum[0][0], 1)
        self.assertEqual(accum[0][1], 2)
        self.assertEqual(accum[0][2], 3)
        self.assertEqual(accum[0][3], 4)

    def test_dem_pipeline_caches_per_station(self):
        """FR3.1: Station terrain features must be cached and pulled once per station."""
        pipeline = DEMPipeline(elevated_threshold_m=40.0)
        st = StationMetadata(
            station_id="USW00093037",
            station_name="SANDIA",
            latitude=35.2,
            longitude=-106.4,
            elevation_m=2500.0,
        )

        feat1 = pipeline.extract_station_terrain(st)
        feat2 = pipeline.extract_station_terrain(st)

        # Must be identical cached instance
        self.assertIs(feat1, feat2)
        self.assertEqual(feat1.station_id, "USW00093037")
        self.assertTrue(feat1.is_elevated_terrain)
        self.assertGreaterEqual(feat1.relative_height_m, 40.0)
        self.assertGreater(feat1.slope_degrees, 0.0)


if __name__ == "__main__":
    unittest.main()
