# HydroAlert: Station-Anchored Multimodal Flash-Flood Early Warning System

> **A station-anchored multimodal early warning pipeline that fuses NOAA GHCN-hourly (GHCNh) ground observations, per-station geostationary satellite image sequences (GOES-16 ABI) with explicit missing-frame masking, and DEM D8 hydrological flow routing to warn communities *downhill* of elevated convective storms.**

---

## 1. Problem & Core Novelty

Flash floods often strike communities that experience **zero rainfall themselves**: convective storms stall over elevated mountain terrain, and torrential runoff rushes down canyons and arroyos into downhill valleys with little or no warning. 

Existing operational systems either:
1. **Alert only where rain is currently falling**, completely missing downstream valley populations.
2. **Rely on coarse, slow hydrological models** like NOAA's Flash Flood Guidance (FFG), which exhibit documented weak skill scores (CSI 0.00–0.44).
3. **Use gridded basin averages** that are not anchored to individual weather station coordinates or fine-grained storm trajectories.

### The HydroAlert Architectural Solution:
1. **Station-Anchored Satellite Trajectories:** Constructs a ±2 hour geostationary satellite "storm movie" (30-min cadence) centered on each station's exact coordinates.
2. **Explicit Missing-Frame Masking (NFR1):** Telemetry dropouts, eclipses, or timing offsets are flagged as `status = "missing"`. Missing frames are **strictly masked out** of neural attention—**never forward-filled, duplicated, or imputed with zero-value noise**.
3. **Multimodal Deep Learning:** Fuses satellite storm trajectory embeddings, ground GHCNh weather time series (precip, temp, pressure, wind), and static DEM topography (elevation, slope, aspect, flow accumulation) into a calibrated risk score.
4. **DEM Downhill Flow-Routing Alerts:** Instead of broadcasting alerts uniformly in a circle (which wastes resources sending false alarms up mountain peaks), the system traces steepest descent paths downhill along computed D8 hydrological flow paths, decaying alert strength with downstream distance and computing physical arrival lead times.

---

## 2. PRD Requirement Traceability Matrix

Every functional requirement from the Product Requirements Document (PRD) maps directly to concrete, runnable code:

| PRD Req | Description | Implementation File | Key Functions / Classes |
|---|---|---|---|
| **FR1.1** | Ingest NOAA GHCN-hourly records (station ID, lat/lon, timestamp, weather variables) | `src/ingestion/ghcnh_parser.py` | `GHCNhParser.parse_file`, `GHCNhParser.parse_content` |
| **FR1.2** | Deduplicate and clean station records (missing sentinels, QC flags, malformed rows) | `src/ingestion/ghcnh_parser.py` | `GHCNhParser._clean_and_aggregate_hourly`, `NOAA_MISSING_SENTINELS` |
| **FR1.3** | Native NOAA GHCNh pipe-delimited (`\|`) PSV file schema documentation & drop-in folder | `data/README.md`, `src/ingestion/ghcnh_parser.py` | Complete column dictionary, `MockGHCNhGenerator` fallback |
| **FR2.1** | Query GEE for satellite imagery in ±2h window around station observation | `src/trajectory/gee_client.py` | `GEEClient.query_nearest_image` |
| **FR2.2** | Extract spatial image patch around station coordinates (preserve storm motion) | `src/trajectory/gee_client.py`, `src/trajectory/trajectory_builder.py` | `_synthesize_patch`, `build_trajectory` (32×32 pixel patch) |
| **FR2.3** | Fixed sub-interval sampling (30-min cadence = 9 timesteps: -120m to +120m) | `src/trajectory/trajectory_builder.py` | `SatelliteTrajectoryBuilder.offset_minutes_list` |
| **FR2.4** | Missing frame tolerance check (±7.5m); mark `status='missing'`; **NO forward-filling** | `src/trajectory/trajectory_builder.py` | `tolerance_seconds=450.0`, `valid_mask=0`, `status='missing'` |
| **FR2.5** | Persist separate missing-frame manifest to inspect coverage before training | `src/trajectory/manifest.py` | `MissingFrameManifest`, `TrajectoryFrameManifestItem`, `to_csv()` |
| **FR3.1** | Retrieve DEM once per unique station location (cached across timesteps) | `src/terrain/dem_pipeline.py` | `DEMPipeline.station_cache`, `extract_station_terrain()` |
| **FR3.2** | Compute slope, aspect, D8 flow direction, and flow accumulation from DEM | `src/terrain/d8_flow.py` | `D8FlowRouter.compute_slope_and_aspect`, `compute_flow_direction`, `compute_flow_accumulation` |
| **FR4.1** | CNN frame encoder + temporal model with explicit missing-frame key-padding mask | `src/model/multimodal_model.py` | `PurePythonSpatialCNN`, `PurePythonTemporalEncoderWithMasking` |
| **FR4.2** | Tabular sequence model over GHCNh weather variables | `src/model/multimodal_model.py` | `PurePythonTabularRNN` |
| **FR4.3** | Static terrain feature branch (elevation, slope, aspect, flow accumulation) | `src/model/multimodal_model.py` | `PurePythonTerrainMLP` |
| **FR4.4** | Fusion head combining all 3 branches into calibrated flash flood risk score | `src/model/multimodal_model.py` | `MultimodalFlashFloodNet.forward`, `_calibrate_weights` |
| **FR4.5** | Missing frames handled via masking (not zero-imputation or forward-filling) | `src/model/multimodal_model.py` | Masked temporal attention energy ($-\infty$ suppression for mask=0) |
| **FR5.1** | Baseline circular-buffer alert (fires within fixed radius $R$ from elevated cell) | `src/alerts/circular_buffer.py` | `CircularBufferAlertEngine.evaluate_and_propagate` |
| **FR5.2** | DEM Flow-routing alert (propagates downhill along D8 path, distance decay & accumulation boost) | `src/alerts/flow_routing_alert.py` | `DEMFlowRoutingAlertEngine.evaluate_and_propagate` |
| **FR5.3** | Alert method attribution logging (`v0_circular_buffer` vs `v1_dem_flow_routing`) | `src/alerts/alert_logger.py` | `AlertLogger`, `AlertEventLog`, `get_summary_by_method` |
| **FR6.1** | Benchmark evaluation against rainfall-threshold-only baseline (>25mm/h) | `src/evaluation/baselines.py` | `BaselineRainfallThreshold.evaluate` |
| **FR6.2** | Benchmark evaluation against NOAA Flash Flood Guidance (FFG) baseline | `src/evaluation/baselines.py` | `NOAAFlashFloodGuidanceBaseline.evaluate` |
| **FR6.3** | Standard skill metrics reporting (CSI, POD, FAR, FBIAS, Heidke Skill Score, F1) | `src/evaluation/metrics.py` | `SkillScoreEvaluator.evaluate`, `BinaryClassificationMetrics` |
| **FR6.4** | 4-stage ablation harness: Tabular -> +Terrain -> +Satellite -> +Flow-Routing | `src/evaluation/ablation.py` | `AblationHarness.run_all_stages`, `export_report_json` |
| **NFR1** | Hard Data Integrity: Missing data never forward-filled or mislabeled | `tests/test_nfr1_missing_frames.py` | Strict invariant tests verifying 0 timestamp/label leakage |

---

## 3. Raw Data Drop-In Instructions (`data/`)

The `data/` folder is intentionally kept empty of external datasets so you can drop your own NOAA GHCN-hourly files directly in.

### Expected File Naming Convention
```
data/GHCNh_<Station_ID>_<Year>.psv
data/GHCNh_<Station_ID>_por.psv
data/GHCNh_<Station_ID>_<Year>.psv.gz
```
*Example:* `data/GHCNh_USW00093037_2023.psv`

### Expected Pipe-Separated Header Schema
```psv
Station_ID|Station_name|Latitude|Longitude|Elevation|Year|Month|Day|Hour|Minute|temperature|dew_point_temperature|relative_humidity|wind_direction|wind_speed|wind_gust|station_level_pressure|sea_level_pressure|precipitation|precipitation_1_hour|precipitation_3_hour|precipitation_6_hour|precipitation_24_hour|present_weather|quality_flag
```

*(See `data/README.md` for complete data dictionary, units, and missing value sentinel documentation).*

---

## 4. How to Run the Pipeline

### Option A: Complete Pipeline Execution (All 6 Stages)
```bash
python3 run_pipeline.py
```
Or specify a custom data directory:
```bash
python3 run_pipeline.py /path/to/my/ghcnh_psv_files/
```

### Option B: Stage-by-Stage Python Execution
```python
from src.pipeline_runner import FlashFloodPipelineRunner

runner = FlashFloodPipelineRunner()

# Stage 1: Ingest NOAA GHCN-hourly records
station_data = runner.run_stage_1_ingestion("data")
stations = [meta for meta, _ in station_data.values()]

# Stage 2: Build satellite trajectories with missing-frame manifest
events = runner.run_stage_2_trajectories(station_data)

# Stage 3: Extract DEM flow directions & terrain features
terrain_features = runner.run_stage_3_terrain(stations)

# Stage 4: Run multimodal prediction with masked attention
dataset, risk_scores = runner.run_stage_4_multimodal_inference(events)

# Stage 5: Downhill alert propagation (Circular buffer vs DEM flow routing)
alert_summary = runner.run_stage_5_alert_propagation(dataset, risk_scores, stations)

# Stage 6: Benchmark against baselines & run 4-stage ablation
ablation_results = runner.run_stage_6_evaluation(dataset, stations)
```

---

## 5. Running the Test Suite (Including NFR1 Verification)

Execute all unit and integration tests with the test runner:
```bash
python3 run_tests.py
```

### What `tests/test_nfr1_missing_frames.py` Verifies:
1. When GEE frames are missing (due to satellite telemetry loss, eclipse, or scan timing beyond ±7.5 min tolerance):
   - The status is recorded strictly as `"missing"` (never `"ok"`).
   - The `valid_mask` entry is set to `0`.
   - The timestamp is `None` (it is **NEVER** copied from the previous frame).
2. Any forward-filling attempt is detected and flagged by `manifest.verify_nfr1_integrity()`.
3. In `MultimodalFlashFloodNet`, missing frames are masked out of the temporal attention calculation ($-\infty$), ensuring missing-frame placeholder pixels do not corrupt the risk prediction.

---

## 6. Configuration Management

No parameters are hardcoded. Modify settings in `configs/`:

- `configs/pipeline_config.json`: Geographic bounding box, target time range, satellite bands, GEE collection, DEM resolution, data directories.
- `configs/model_config.json`: Patch dimensions, CNN layers, temporal encoder parameters, tabular features, MLP dims, loss weights.
- `configs/alert_config.json`: Circular buffer radius ($R=25\text{ km}$), DEM flow decay factor ($\alpha=0.04$), accumulation boost ($\beta=0.12$), hydraulic wave speed ($7.2\text{ km/h}$).

---

## 7. Interactive Web Dashboard

In addition to the Python CLI pipeline, an interactive dashboard runs in the AI Studio preview environment:
- **Terrain & Watershed Map:** Visualizes mountain stations, downhill valley communities, and D8 flow descent paths.
- **Alert Propagation Simulator:** Compare the circular buffer baseline (which causes uphill false alarms) against DEM flow-routing (which targets downstream communities).
- **Multimodal Masking Inspector:** Inspect how missing satellite frames are masked rather than zero-imputed.
- **Ablation Benchmark Table:** View real-time CSI, POD, FAR, and F1 score deltas across all ablation stages.
#   f l a s h _ f l o o d _ w a r n i n g _ s y s t e m  
 