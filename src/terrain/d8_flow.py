"""D8 Flow Routing, Slope, Aspect, and Flow Accumulation Algorithms.

Implements standard ESRI/USGS D8 flow direction encoding and drainage accumulation.
Supports downstream flow path tracing for flash-flood alert routing (FR3.2, FR5.2).
"""

from __future__ import annotations

import enum
import math
from typing import Dict, List, Optional, Set, Tuple


class FlowDirectionD8(enum.IntEnum):
    """Standard ESRI D8 flow direction power-of-2 encodings."""
    SINK = 0
    EAST = 1
    SOUTHEAST = 2
    SOUTH = 4
    SOUTHWEST = 8
    WEST = 16
    NORTHWEST = 32
    NORTH = 64
    NORTHEAST = 128


# (delta_row, delta_col, distance_multiplier, D8_code)
# row increases downward (South), col increases rightward (East)
D8_NEIGHBORS = [
    (0, 1, 1.0, FlowDirectionD8.EAST),
    (1, 1, math.sqrt(2.0), FlowDirectionD8.SOUTHEAST),
    (1, 0, 1.0, FlowDirectionD8.SOUTH),
    (1, -1, math.sqrt(2.0), FlowDirectionD8.SOUTHWEST),
    (0, -1, 1.0, FlowDirectionD8.WEST),
    (-1, -1, math.sqrt(2.0), FlowDirectionD8.NORTHWEST),
    (-1, 0, 1.0, FlowDirectionD8.NORTH),
    (-1, 1, math.sqrt(2.0), FlowDirectionD8.NORTHEAST),
]


class D8FlowRouter:
    """Hydrological terrain analysis engine using D8 steepest descent."""

    def __init__(self, cell_size_m: float = 30.0):
        self.cell_size_m = cell_size_m

    def compute_slope_and_aspect(
        self, dem: List[List[float]]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Computes topographic slope (degrees) and aspect (degrees from North)
        using Horn's 3x3 finite-difference window.
        """
        rows = len(dem)
        cols = len(dem[0]) if rows > 0 else 0
        slope_grid = [[0.0 for _ in range(cols)] for _ in range(rows)]
        aspect_grid = [[0.0 for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                # 3x3 window with clamp-to-edge boundary conditions
                r_prev = max(0, r - 1)
                r_next = min(rows - 1, r + 1)
                c_prev = max(0, c - 1)
                c_next = min(cols - 1, c + 1)

                z1 = dem[r_prev][c_prev]
                z2 = dem[r_prev][c]
                z3 = dem[r_prev][c_next]
                z4 = dem[r][c_prev]
                z6 = dem[r][c_next]
                z7 = dem[r_next][c_prev]
                z8 = dem[r_next][c]
                z9 = dem[r_next][c_next]

                # Horn's weighting
                dz_dx = ((z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)) / (8.0 * self.cell_size_m)
                dz_dy = ((z7 + 2.0 * z8 + z9) - (z1 + 2.0 * z2 + z3)) / (8.0 * self.cell_size_m)

                rise_run = math.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)
                slope_deg = math.degrees(math.atan(rise_run))
                slope_grid[r][c] = round(slope_deg, 2)

                # Aspect calculation (degrees clockwise from North)
                if abs(dz_dx) < 1e-7 and abs(dz_dy) < 1e-7:
                    aspect_deg = 0.0  # Flat
                else:
                    aspect_rad = math.atan2(dz_dy, -dz_dx)
                    aspect_deg = math.degrees(aspect_rad)
                    if aspect_deg < 0.0:
                        aspect_deg += 360.0
                aspect_grid[r][c] = round(aspect_deg, 2)

        return slope_grid, aspect_grid

    def compute_flow_direction(self, dem: List[List[float]]) -> List[List[int]]:
        """
        Computes D8 flow direction matrix from a DEM.
        For each cell, finds the neighboring cell with the steepest downward slope:
        drop = (z_center - z_neighbor) / (dist_mult * cell_size)
        """
        rows = len(dem)
        cols = len(dem[0]) if rows > 0 else 0
        fdir = [[0 for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                z_center = dem[r][c]
                max_drop = 0.0
                best_dir = FlowDirectionD8.SINK

                for dr, dc, dist_mult, code in D8_NEIGHBORS:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        z_neighbor = dem[nr][nc]
                        drop = (z_center - z_neighbor) / (dist_mult * self.cell_size_m)
                        if drop > max_drop:
                            max_drop = drop
                            best_dir = code

                fdir[r][c] = int(best_dir)

        return fdir

    def compute_flow_accumulation(self, fdir: List[List[int]]) -> List[List[int]]:
        """
        Computes flow accumulation matrix (number of upstream contributing cells).
        Each cell contributes 1 unit to itself, plus all upstream inflows.
        """
        rows = len(fdir)
        cols = len(fdir[0]) if rows > 0 else 0
        in_degree = [[0 for _ in range(cols)] for _ in range(rows)]
        downstream_target: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # Decode flow directions into target neighbors
        d8_to_delta = {
            int(FlowDirectionD8.EAST): (0, 1),
            int(FlowDirectionD8.SOUTHEAST): (1, 1),
            int(FlowDirectionD8.SOUTH): (1, 0),
            int(FlowDirectionD8.SOUTHWEST): (1, -1),
            int(FlowDirectionD8.WEST): (0, -1),
            int(FlowDirectionD8.NORTHWEST): (-1, -1),
            int(FlowDirectionD8.NORTH): (-1, 0),
            int(FlowDirectionD8.NORTHEAST): (-1, 1),
        }

        for r in range(rows):
            for c in range(cols):
                code = fdir[r][c]
                delta = d8_to_delta.get(code)
                if delta:
                    nr = r + delta[0]
                    nc = c + delta[1]
                    if 0 <= nr < rows and 0 <= nc < cols:
                        in_degree[nr][nc] += 1
                        downstream_target[(r, c)] = (nr, nc)

        # Topological traversal using queue of zero in-degree cells (headwaters / ridge cells)
        queue: List[Tuple[int, int]] = [
            (r, c) for r in range(rows) for c in range(cols) if in_degree[r][c] == 0
        ]
        accumulation = [[1 for _ in range(cols)] for _ in range(rows)]

        visited_count = 0
        while queue:
            r, c = queue.pop(0)
            visited_count += 1
            target = downstream_target.get((r, c))
            if target:
                tr, tc = target
                accumulation[tr][tc] += accumulation[r][c]
                in_degree[tr][tc] -= 1
                if in_degree[tr][tc] == 0:
                    queue.append((tr, tc))

        return accumulation

    def trace_downstream_path(
        self,
        fdir: List[List[int]],
        dem: List[List[float]],
        start_row: int,
        start_col: int,
        max_steps: int = 150,
    ) -> List[Dict[str, Any]]:
        """
        Traces the downstream water flow path from a starting cell along D8 flow directions.
        Returns a sequence of path steps with (row, col, elevation, distance_accum_m).
        """
        rows = len(fdir)
        cols = len(fdir[0]) if rows > 0 else 0
        visited: Set[Tuple[int, int]] = set()

        d8_to_delta_and_dist = {
            int(FlowDirectionD8.EAST): (0, 1, self.cell_size_m),
            int(FlowDirectionD8.SOUTHEAST): (1, 1, self.cell_size_m * math.sqrt(2.0)),
            int(FlowDirectionD8.SOUTH): (1, 0, self.cell_size_m),
            int(FlowDirectionD8.SOUTHWEST): (1, -1, self.cell_size_m * math.sqrt(2.0)),
            int(FlowDirectionD8.WEST): (0, -1, self.cell_size_m),
            int(FlowDirectionD8.NORTHWEST): (-1, -1, self.cell_size_m * math.sqrt(2.0)),
            int(FlowDirectionD8.NORTH): (-1, 0, self.cell_size_m),
            int(FlowDirectionD8.NORTHEAST): (-1, 1, self.cell_size_m * math.sqrt(2.0)),
        }

        path: List[Dict[str, Any]] = []
        curr_r, curr_c = start_row, start_col
        total_dist_m = 0.0

        for step in range(max_steps):
            if (curr_r, curr_c) in visited:
                break  # Cycle detected
            visited.add((curr_r, curr_c))

            elev = dem[curr_r][curr_c]
            path.append({
                "step": step,
                "row": curr_r,
                "col": curr_c,
                "elevation_m": round(elev, 1),
                "cumulative_dist_m": round(total_dist_m, 1),
                "cumulative_dist_km": round(total_dist_m / 1000.0, 3),
            })

            code = fdir[curr_r][curr_c]
            if code == FlowDirectionD8.SINK or code not in d8_to_delta_and_dist:
                break  # Reached sink, pit, or boundary

            dr, dc, dist_step = d8_to_delta_and_dist[code]
            next_r = curr_r + dr
            next_c = curr_c + dc

            if not (0 <= next_r < rows and 0 <= next_c < cols):
                break  # Flow leaves domain

            total_dist_m += dist_step
            curr_r, curr_c = next_r, next_c

        return path
