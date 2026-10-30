# HydroAlert codebase guide

This guide describes what the repository actually does today. It deliberately
separates implemented code, synthetic fallbacks, and claims that have not yet
been demonstrated with a trained model.

## 1. The 60-second and 3-minute explanations

### 60 seconds

HydroAlert watches weather stations, satellite images, and the shape of the
ground. It turns each station's hourly weather report into a clean record,
looks at a short sequence of nearby satellite images, and calculates terrain
features such as slope and where water would flow. A model combines those
three kinds of information into a flood-risk score. If a high station triggers,
the alert can travel downhill along the computed drainage direction instead of
warning every nearby place, including places uphill that are not in the same
watershed. The data-integrity rule is important: a missing satellite picture
stays missing and is not copied from the previous picture. The project has
working parsing, trajectory, terrain, alert, and evaluation code, but the
current model has not been trained on the July 2025 labels.

### 3 minutes

The pipeline begins with NOAA GHCN-hourly (GHCNh) pipe-separated observations.
`GHCNhParser` normalizes column names, rejects failed quality-control rows,
converts missing sentinels to `None`, chooses one report per UTC hour, and
computes rolling precipitation totals. A blank precipitation field is
classified as dry only when there is no station precipitation weather code and
the measurement is not a sensor-down code; otherwise it is missing. That
distinction prevents a dry hour from being confused with an unavailable
measurement.

For each hourly station event, `SatelliteTrajectoryBuilder` requests nine
station-centered frames: two hours before through two hours after the event at
30-minute intervals. The `GEEClient` is the real-data integration point, but
its local fallback synthesizes patches. Every frame is recorded in a manifest.
Missing frames have no matched timestamp, a zero validity mask, and are checked
by the NFR1 integrity audit so that they cannot become forward-filled images.

The terrain branch creates or loads a DEM and uses D8 flow routing. D8 means
eight-direction routing: each raster cell sends water to the steepest of its
eight neighbors. The code derives slope, aspect, flow direction, accumulation,
and a downstream path. The current local DEM is synthesized, not downloaded.
The elevated gate now uses station height above the lowest cell in the
station-centered DEM tile; a 40 m relative-relief threshold is configured.
This is a local-relief implementation, not yet a verified 10 km public DEM
calculation.

The model has four branches: a spatial encoder for each satellite patch, a
masked temporal attention encoder, a tabular weather sequence branch, and a
terrain MLP. A fusion head returns a score from 0 to 1. The implementation is
pure Python with deterministic random initialization and no optimizer or
training loop. The configured loss functions are present but unused.

Finally, Stage 5 applies either the circular-buffer baseline or the DEM flow
routing alert method. Stage 6 reports baselines and an ablation table. Separately,
the July 2025 label evaluator uses USGS gauge-derived labels and keeps NWS
warnings as a benchmark. POD (Probability of Detection) is the fraction of
observed events detected; FAR (False Alarm Ratio) is the fraction of alerts
that were false; CSI (Critical Success Index) is hits divided by hits plus
misses plus false alarms. The real Kerrville label evaluation has 27 labeled
samples, but it does not constitute model training.

## 2. Architecture map

```text
NOAA GHCNh PSV / mock generator
        |
        v
src/ingestion/ghcnh_parser.py
(clean, QC, hourly record, rolling rain)
        |
        +------------------------------+
        |                              |
        v                              v
src/trajectory/trajectory_builder.py  src/terrain/dem_pipeline.py
        |                              |
        v                              v
src/trajectory/gee_client.py          src/terrain/d8_flow.py
(real GEE boundary or synthetic      (DEM features and drainage path)
patch fallback)                       |
        |                              |
        v                              |
src/trajectory/manifest.py             |
(frame status and NFR1 audit)          |
        |                              |
        +---------------+--------------+
                        v
              src/model/dataset.py
              (one multimodal sample)
                        |
                        v
              src/model/multimodal_model.py
              (satellite + weather + terrain score)
                        |
                        +-----------------------+
                        |                       |
                        v                       v
              src/alerts/circular_buffer.py   src/alerts/flow_routing_alert.py
              (v0 radius baseline)             (v1 downhill propagation)
                        \                       /
                         v                     v
                         src/alerts/alert_logger.py
                         (audit CSV/JSON)
                                      |
                                      v
                         src/evaluation/baselines.py
                         src/evaluation/ablation.py
                         src/evaluation/metrics.py
                         src/evaluation/label_evaluation.py

Orchestration: src/pipeline_runner.py
CLI: run_pipeline.py
Tests: tests/
Browser demonstration: src/App.tsx + src/data/mockPipelineData.ts
```

## 3. One real example: Kerrville, 2025-07-04 10:00 UTC

This example is intentionally precise about what is and is not available in
the repository.

* **Station and GHCNh:** Kerrville is `USW00012961`. The July event data
  contains a 24.1 mm one-hour precipitation value at the relevant Kerrville
  airport observation. The parser represents it as an
  `GHCNhObservation`, including the UTC timestamp, precipitation fields, and
  meteorological values. The parser implementation is
  [`ghcnh_parser.py`](../src/ingestion/ghcnh_parser.py), especially
  `parse_content` (about line 191) and `_clean_and_aggregate_hourly` (about
  line 390).
* **Satellite frames:** The trajectory contract selects offsets
  `-120, -90, -60, -30, 0, +30, +60, +90, +120` minutes, so the selected
  target times are 08:00 through 12:00 UTC at half-hour cadence. The
  repository does not contain a verified real GOES frame set for this exact
  event. If the local fallback is used, `GEEClient._synthesize_patch` creates
  the patches; that is not evidence that these are observed satellite values.
* **Terrain:** `DEMPipeline.extract_station_terrain` returns elevation, slope,
  aspect, log flow accumulation, downstream distance, and the absolute-elevation
  boolean. The current implementation synthesizes a station DEM in
  `synthesize_station_dem`; it does not provide a real Kerrville DEM-derived
  feature vector. Therefore no honest guide can quote real terrain numbers for
  this hour from this checkout.
* **Model input and score:** `StationAnchoredFloodDataset.create_sample`
  assembles the nine-frame trajectory, a six-step tabular history, and five
  terrain features. `MultimodalFlashFloodNet.predict_sample` consumes them.
  There is no trained checkpoint and no stored real Kerrville model input/score
  for 10:00 UTC. The score would be a deterministic output of the
  randomly initialized pure-Python network (subject to the fallback inputs),
  not a learned forecast.
* **Label:** Gauge onset was 08:00 UTC, so 10:00 UTC is already after onset.
  The hour is therefore excluded from the pre-onset forecast-target set; it is
  not a valid “1 to 3 hours before onset” target. The flood-label CSV is the
  source of the hour-level gauge label and `nws_warning_active` is a separate
  benchmark field. The label evaluator is
  [`label_evaluation.py`](../src/evaluation/label_evaluation.py), around lines
  17–155.

The concrete conclusion is that the 24.1 mm observation is real project data,
while the satellite, terrain, and model values for this exact example are not
proven real-data artifacts in the repository. That limitation is preferable to
inventing values.

## 4. Module walkthroughs

### Pipeline and ingestion

* [`pipeline_runner.py`](../src/pipeline_runner.py) — `FlashFloodPipelineRunner`
  (about line 45) owns configuration and six-stage orchestration.
  `run_stage_1_ingestion` (88) loads station files or mock data;
  `run_stage_2_trajectories` (about 116) builds frames and writes the manifest;
  `run_stage_3_terrain` (166) extracts terrain; `run_stage_4_multimodal_inference`
  (186) builds samples and scores them; `run_stage_5_alert_propagation` (232)
  dispatches v0/v1 alerts; `run_stage_6_evaluation` (304) writes ablation
  results; `run_all` (348) ties everything together.
* [`ghcnh_parser.py`](../src/ingestion/ghcnh_parser.py) — `StationMetadata`
  (44) and `GHCNhObservation` (57) are the data records. `GHCNhParser`
  (92) provides `parse_file` (176), `parse_content` (191), weather-token
  detection (114), hourly aggregation (390), and directory ingestion (474).
  It accepts NOAA PSV variants, cleans sentinels and QC flags, and computes
  rolling rainfall.
* [`mock_ghcnh_generator.py`](../src/ingestion/mock_ghcnh_generator.py) —
  `MockGHCNhGenerator` (74), `generate_station_records` (88), and
  `write_mock_dataset` (187) create deterministic synthetic five-station
  records for tests and empty-data runs.

### Satellite trajectories

* [`gee_client.py`](../src/trajectory/gee_client.py) — `GEEClient` (15)
  defines the Earth Engine boundary. `query_nearest_image` (45) searches for
  a frame within tolerance; `_synthesize_patch` (93) supplies local synthetic
  data when real imagery is unavailable.
* [`trajectory_builder.py`](../src/trajectory/trajectory_builder.py) —
  `SatelliteTrajectorySequence` (24) stores patches, offsets, masks, and
  statuses. `SatelliteTrajectoryBuilder` (57) and `build_trajectory` (89)
  create the nine-frame sequence and mark unavailable frames as missing.
* [`manifest.py`](../src/trajectory/manifest.py) — `TrajectoryFrameManifestItem`
  (19) is one audit row. `MissingFrameManifest` (48) collects rows,
  `compute_coverage_statistics` (57) summarizes coverage,
  `verify_nfr1_integrity` (98) rejects timestamp/data reuse, and `to_csv`/`to_json`
  (138/163) persist the audit.

### Terrain

* [`dem_pipeline.py`](../src/terrain/dem_pipeline.py) — `StationTerrainFeatures`
  (21) is the feature record. `DEMPipeline` (54) caches per-station terrain;
  `synthesize_station_dem` (69) creates the current synthetic raster;
  `extract_station_terrain` (101) calculates/caches features; and
  `get_downstream_path_from_station` (143) exposes a flow path.
  `relative_height_m` and `is_station_elevated` implement the local-relief gate.
* [`d8_flow.py`](../src/terrain/d8_flow.py) — `FlowDirectionD8` (14) defines
  the eight direction codes. `D8FlowRouter` (41) implements
  `compute_slope_and_aspect` (47), `compute_flow_direction` (96),
  `compute_flow_accumulation` (126), and `trace_downstream_path` (179).

### Model and data representation

* [`dataset.py`](../src/model/dataset.py) — `MultimodalEventSample` (19) is
  the model-ready record. `StationAnchoredFloodDataset` (59) stores samples;
  `create_sample` (75) normalizes weather/terrain and carries masks and labels;
  `split_train_val_test` (150) provides a split helper.
* [`multimodal_model.py`](../src/model/multimodal_model.py) — the pure-Python
  implementation. `PurePythonDenseLayer` (20) is a small dense layer;
  `PurePythonSpatialCNN` (52) summarizes a patch;
  `PurePythonTemporalEncoderWithMasking` (92) applies masked temporal
  attention; `PurePythonTabularRNN` (149) processes weather history;
  `PurePythonTerrainMLP` (168) processes five terrain values; and
  `MultimodalFlashFloodNet` (180) fuses branches. `forward` (229) returns a
  score and `predict_sample` (284) is the sample convenience wrapper.
* [`losses.py`](../src/model/losses.py) — `FocalLoss` (9) and
  `WeightedBCELoss` (40) calculate candidate losses. They are configuration
  and utility code, not a connected training system.

### Alerts and logging

* [`circular_buffer.py`](../src/alerts/circular_buffer.py) — `haversine_distance_km`
  (18) measures station distance; `CircularBufferAlertEngine` (33) alerts
  candidates inside the configured radius in `evaluate_and_propagate` (46),
  regardless of drainage direction.
* [`flow_routing_alert.py`](../src/alerts/flow_routing_alert.py) —
  `DEMFlowRoutingAlertEngine` (20) checks downhill eligibility in
  `is_in_downstream_watershed` (43), then applies distance decay and flow
  accumulation boost in `evaluate_and_propagate` (90).
* [`alert_logger.py`](../src/alerts/alert_logger.py) — `AlertEventLog` (20)
  is the audit record. `AlertLogger` (43) records alerts, summarizes each
  method (52), and writes CSV/JSON (82/109).

### Evaluation

* [`baselines.py`](../src/evaluation/baselines.py) — `BaselineRainfallThreshold`
  (16) uses configured 1-, 3-, and 6-hour thresholds; `NOAAFlashFloodGuidanceBaseline`
  (46) is the FFG comparison. Their `predict` and `evaluate` methods produce
  binary alerts and metrics.
* [`metrics.py`](../src/evaluation/metrics.py) — `BinaryClassificationMetrics`
  (20) stores scores; `SkillScoreEvaluator.evaluate` (45) computes POD, FAR,
  CSI, F-bias, HSS, F1, precision, recall, and accuracy.
* [`ablation.py`](../src/evaluation/ablation.py) — `AblationStageResult` (33)
  serializes one stage; `AblationHarness.run_all_stages` (65) compares
  tabular, terrain, satellite, routing, and baselines; `export_report_json`
  (207) writes the report. Stage 4 and circular-buffer rows now preserve the
  local model predictions; propagation differences are recorded separately by
  the Stage 5 alert audit. This remains an untrained-model benchmark.
* [`label_evaluation.py`](../src/evaluation/label_evaluation.py) —
  `FloodLabel` (17) represents a CSV row; `load_flood_labels` (27) reads the
  labels; `target_for_label` (42) and `labeled_sample_target` (49) apply the
  1–3-hour-onset rule; `evaluate_labeled_samples` (86) scores NWS, rainfall,
  and model alerts; `print_label_evaluation_table` (143) prints the table.

### Frontend and repository entry points

* [`run_pipeline.py`](../run_pipeline.py) is the Python command-line wrapper.
* [`run_tests.py`](../run_tests.py) discovers `tests/test_*.py`, runs unittest
  with verbosity 2, and exits nonzero on failures.
* [`App.tsx`](../src/App.tsx) is a React/Tailwind demonstration dashboard. Its
  state, tabs, pipeline log animation, PSV parser, station cards, trajectory
  display, and ablation display are UI behavior; it does not call the Python
  pipeline.
* [`mockPipelineData.ts`](../src/data/mockPipelineData.ts) supplies five
  synthetic stations, nine trajectory frames, three alert examples, and
  illustrative ablation scores.
* [`pipeline.ts`](../src/types/pipeline.ts) defines frontend interfaces
  `Station`, `TrajectoryFrame`, `AlertEvent`, and `AblationScore`.
* [`main.tsx`](../src/main.tsx) mounts `App`; [`index.css`](../src/index.css)
  imports Tailwind and defines the color/font theme; [`vite.config.ts`](../vite.config.ts)
  configures React, Tailwind, aliases, and HMR; [`index.html`](../index.html)
  supplies metadata and the root element.

## 5. Design decisions and why

| Decision | Alternative rejected | Reason |
|---|---|---|
| Keep a nine-frame station-centered movie at 30-minute cadence. | One image or irregular sampling. | A short sequence can represent storm growth/motion while remaining a fixed model input. |
| Missing frames get a placeholder plus `valid_mask=0`. | Forward-fill, duplicate the last image, or silently drop the timestep. | Forward-filling leaks stale information; a fixed sequence and explicit mask preserve timing and auditable coverage. See `trajectory_builder.py` and `multimodal_model.py`. |
| Split blank precipitation into dry versus missing. | Treat every blank as zero, or every blank as unavailable. | A blank with no precipitation weather code and no sensor-down code is evidence of dry conditions; a sensor-down/weather-coded blank is not. |
| Use the last report in an hour, with explicit precipitation/weather reports preferred during aggregation. | Sum all reports or take the maximum. | The source reports are not guaranteed to be independent increments. Last-report selection preserves the hour's final station state; the preference prevents a later blank from erasing a meaningful report. |
| Gauge stage behavior supplies `flood_label`; NWS warnings are a benchmark. | Put warning issuance directly into the target. | Warnings are an operational comparison and can leak the answer being evaluated. Gauge onset is the independent event definition. |
| Match warning coverage by station county/UGC for the existing event artifacts. | Assume one county for every station or use an unverified polygon. | County UGC is explicit and reproducible in `data/texas_station_counties.csv`; point-in-polygon is a future refinement. |
| Configure three rainfall windows: 25/45/60 mm for 1/3/6 hours. | Hard-code one 15 mm/h threshold or tune on July 2025. | Multiple durations represent accumulation timescales; the README records that the event was not used for tuning. |
| Use a 25 km circular baseline and a downhill D8 method. | Broadcast equally to all nearby stations. | The comparison tests whether drainage direction reduces uphill false alarms. |
| Use DEM-derived local relief rather than absolute station elevation for the terrain gate. | Classify every station above a fixed elevation such as 2,000 m. | Relative relief transfers better between regions; the current implementation is limited to the available station-centered DEM tile. |
| Keep Python and frontend demonstrations separate. | Pretend the React screen is the live backend. | The UI is useful for explaining the design, but it uses TypeScript mock data and illustrative logs. |

## 6. Real versus mock, and training status

### Exercised or stored as real project artifacts

* NOAA-shaped GHCNh parsing has been exercised against real-schema content and
  a real-station audit path when the data file is present.
* The Kerrville July 3–5, 2025 label CSV is a real project artifact built from
  USGS gauge readings and NWS/IEM warning data. It records a gauge onset at
  08:00Z, an NWS warning beginning at 06:00Z, 16 Kerrville positive hours, and
  27 valid pre-onset labeled samples used by the evaluator.
* `data/events/candidate_events.csv` is a real generated warning-by-station
  artifact: 8,124 rows from the recorded IEM query process.
* Station metadata and county mappings are stored under `data/`.

### Synthetic, fallback, or illustrative

* `MockGHCNhGenerator` is synthetic.
* `GEEClient._synthesize_patch` is synthetic. `GEEClient` can query a real
  Earth Engine collection when authenticated and configured with a Cloud
  project, but the checked-in Kerrville example does not prove that live path.
* `DEMPipeline.synthesize_station_dem` is synthetic; the configured SRTM
  collection is not the same as a checked-in downloaded raster.
* `src/data/mockPipelineData.ts` and the React pipeline logs are UI mock data.
* The ablation scores shown in the frontend are illustrative; the Python
  ablation harness also contains synthetic/optimistic stage logic.

### Is the model trained?

No. `MultimodalFlashFloodNet` initializes deterministic random weights and runs
forward passes. `FocalLoss` and `WeightedBCELoss` exist, and
`configs/model_config.json` describes epochs, learning rate, and class weight,
but no optimizer, gradient update, checkpoint, or training command is wired
into the repository. The July labels are connected to evaluation, not training.

## 7. Known limitations, bugs, and TODOs

1. **Synthetic DEM:** the current terrain pipeline synthesizes its DEM instead
   of loading a verified public DEM raster. The gate is now relative to the
   lowest cell in that local tile, not absolute station elevation.
2. **Local relief radius:** the current relative gate uses the station-centered
   DEM tile (450 m across by default), not a verified approximately 10 km
   neighborhood.
3. **Synthetic imagery fallback:** a successful local run can use synthetic
   patches, so “satellite coverage” does not automatically mean observed GOES
   coverage.
4. **No training:** the model has no learned parameters or held-out test
   training workflow.
5. **Single-event label evaluation:** the label-aware evaluation covers the
   July 3–5, 2025 Kerrville window only. A second held-out event is not present.
6. **Small and spatially separated gauge:** the airport gauge can miss the
   storm maximum; the July baseline miss is therefore not evidence of model
   superiority.
7. **Hunt missing readings:** many hours rely on the Kerrville gauge because
   the Hunt gauge has gaps; exact onset claims must state that limitation.
8. **Ablation scope:** sample-level Stage 4 and circular-buffer rows now leave
   model predictions unchanged; propagation differences are recorded by the
   Stage 5 alert audit. This is honest, but it is not yet a downstream
   station-level skill study.
9. **Configuration drift:** `pipeline_config.json` describes GOES/SRTM
   collections, while the local implementation may synthesize data.
10. **Frontend/backend split:** the React dashboard does not execute
    `pipeline_runner.py`; its values and logs are mock/illustrative.
11. **Target coverage:** only Kerrville has labels in the current label-aware
    runner; other stations are skipped rather than assigned false negatives.
12. **No external API contract test:** real IEM, USGS, Census, Earth Engine,
    and NWS availability is not validated by the unit suite.
13. **Potential documentation drift:** the top-level README still describes
    an earlier project framing in places; the code and this guide are the
    authoritative implementation description.

The repository does not contain explicit TODO comments in the core modules
that replace these limitations with completed work; the limitations above are
the substantive unfinished items identified by reading the implementation.

## 8. How to run it

Commands below are repository commands, not claims that external services are
available in every environment.

### Python pipeline

```bash
python run_pipeline.py
python run_pipeline.py path/to/ghcnh_psv_directory
```

`run_pipeline.py` invokes the six stages. With no raw GHCNh files, the
documented fallback creates synthetic observations. Outputs are written under
`outputs/`: processed data, trajectory manifests, terrain cache, alert audit
CSV/JSON, and evaluation JSON. Stage 2 prints frame totals and NFR1 status;
Stage 4 prints sample and high-risk counts; Stage 5 prints v0/v1 alert
summaries; Stage 6 prints the baseline/ablation table.

### Tests

```bash
python -m unittest discover -s tests
python run_tests.py
```

The suite covers parser cleaning and precipitation semantics, NFR1 missing
frames, D8 calculations, alert directionality, model forward passes, and the
end-to-end six-stage synthetic run. The custom runner prints totals, failures,
errors, skipped tests, and an all-passed message.

### Frontend

```bash
npm install
npm run dev
npm run build
npm run lint
```

The Vite app is a browser explanation/demo. `npm run dev` serves the React
dashboard; `npm run build` makes the production bundle; `npm run lint` runs
TypeScript's no-emit check. The UI does not replace the Python pipeline.

### Label-aware evaluation

The label path is configured in `configs/pipeline_config.json`. Running the
Python pipeline with the July data causes `pipeline_runner.py` to load labels,
retain valid Kerrville pre-onset hours, and print the NWS, rainfall-baseline,
and model-threshold rows. Target positives mean onset is 1–3 hours ahead;
already-flooding, post-flood, blank-label, and onset-more-than-three-hours-away
hours are handled according to `label_evaluation.py`.

## 9. Likely questions from a reviewer

1. **What is the core contribution?** Station-centered multimodal inputs plus
   downhill alert propagation; the implementation is in
   [`pipeline_runner.py`](../src/pipeline_runner.py) and the model/alert modules.
2. **Is the model trained?** No; `multimodal_model.py` has initialization and
   inference only.
3. **Where are labels defined?** `data/labels/README.md` and the July CSV.
4. **How do you avoid label leakage?** Warning activity is a benchmark, while
   gauge onset defines the target; filtering is in `label_evaluation.py`.
5. **Why are there only 27 labeled samples?** The valid Kerrville
   pre-onset July window is small and excludes blanks, flooding hours, and
   post-event hours.
6. **Why does the rainfall baseline never fire on July 4?** Its configured
   25/45/60 mm thresholds exceed the observed 1/3/6-hour peaks; see
   `configs/pipeline_config.json` and `README.md`.
7. **Does that prove the model wins?** No. The airport gauge is spatially
   separated from the storm maximum and the model is untrained.
8. **What is POD?** Hits divided by observed positives; implemented in
   `evaluation/metrics.py`.
9. **What is CSI?** Hits divided by hits, misses, and false alarms; same file.
10. **What is FAR?** False alarms divided by all alerts; same file.
11. **What if a satellite frame is missing?** It is marked missing, masked, and
    audited; see `trajectory_builder.py` and `manifest.py`.
12. **Is zero placeholder data dangerous?** The placeholder is accompanied by
    mask zero; temporal attention suppresses it instead of treating it as a
    valid image.
13. **Why nine frames?** Fixed offsets from -120 to +120 minutes at 30-minute
    cadence in `trajectory_builder.py`.
14. **Are the GOES frames real?** Not necessarily; the local GEE client can
    synthesize them.
15. **Is the DEM real?** Not in the current local terrain pipeline; it is
    synthesized by `DEMPipeline`.
16. **What does D8 mean?** Eight-neighbor steepest-descent flow routing,
    implemented in `d8_flow.py`.
17. **Does the terrain gate use relative terrain?** No; it still uses absolute
    elevation, a known limitation.
18. **How are stations matched to warnings?** The supplied station county CSV
    provides county FIPS/UGC; warning rows are expanded to stations in covered
    counties for candidate events.
19. **Why not make warnings the label?** They are an operational benchmark and
    can encode prior human assessment; gauges define the independent onset.
20. **How are blank rain reports handled?** No weather code plus no sensor-down
    code becomes dry zero; sensor-down/weather-coded blanks remain missing.
21. **Why not sum every report in an hour?** Reports are not guaranteed
    independent increments; hourly aggregation chooses a representative final
    report with precipitation-aware preference.
22. **What is the alert threshold?** The model trigger is 0.65 in the runner;
    alert thresholds are in `configs/alert_config.json`.
23. **What does v0 do?** Circular distance propagation within 25 km, without
    terrain direction.
24. **What does v1 do?** It requires downhill elevation/gradient, applies
    distance decay and accumulation boost, and estimates arrival time.
25. **How is an alert audited?** `AlertLogger` writes structured CSV and JSON
    records including method, stations, strength, distance, and lead time.
26. **Are ablation scores publishable?** Not yet; the harness no longer
    manufactures hits, but there is still no trained model or downstream
    station-level skill evaluation.
27. **What data is mocked?** Empty-data GHCNh fallback, GEE patches, DEMs, and
    the React dashboard are mocked or synthetic.
28. **What is held out?** No second flood event is implemented yet; evaluation
    is a single July 2025 event window.
29. **What prevents forward-fill corruption?** Manifest status/timestamps and
    `verify_nfr1_integrity` explicitly detect reused satellite times.
30. **What is novel and proven?** The code demonstrates a coherent prototype
    architecture and integrity checks; operational skill, trained-model
    superiority, and generalization are not yet proven.

## 10. Glossary and one-page cheat sheet

### Glossary

* **ABI:** Advanced Baseline Imager, the GOES instrument used by the satellite
  configuration.
* **Accumulation:** The number or weighted amount of upstream raster cells
  draining through a cell.
* **Alert strength:** The propagated risk after distance decay and accumulation
  adjustment.
* **ASOS/AWOS:** Automated airport weather observing systems; relevant to the
  station inventory but not a model algorithm.
* **CSI:** Critical Success Index, hits / (hits + misses + false alarms).
* **D8:** Eight-direction raster flow routing to the steepest lower neighbor.
* **DEM:** Digital Elevation Model, a raster of ground height.
* **FAR:** False Alarm Ratio, false alarms / all alerts.
* **FFG:** Flash Flood Guidance, an operational rainfall/runoff threshold
  benchmark.
* **GHCNh:** NOAA Global Historical Climatology Network hourly observations.
* **GEE:** Google Earth Engine, the intended satellite data access boundary.
* **Mask:** A validity flag telling the model whether an input frame is usable.
* **NFR1:** Non-functional requirement 1: missing imagery must not be silently
  reused or mislabeled.
* **NWS:** National Weather Service.
* **POD:** Probability of Detection, observed events correctly alerted / all
  observed events.
* **PSV:** Pipe-separated values, the GHCNh file format.
* **QC:** Quality control; flags identify accepted, suspect, or failed records.
* **UGC:** Universal Geographic Code, such as county code `TXC265`.
* **UTC/Z:** Coordinated Universal Time, used for all event matching.
* **VTEC:** NWS product encoding used by the IEM warning archive.

### Numbers to remember

| Item | Value |
|---|---:|
| Project stations in the main station list | 5 |
| Satellite frames per event | 9 |
| Trajectory window | ±2 hours |
| Frame cadence | 30 minutes |
| July Kerrville label window | 72 of 72 hours |
| Gauge onset | 08Z, 2025-07-04 |
| First NWS warning hour | 06Z |
| Approximate NWS lead | 2 hours |
| Valid labeled Kerrville samples | 27 |
| Kerrville positive flood-label hours | 16 |
| Rainfall thresholds (1/3/6 h) | 25/45/60 mm |
| July 4 rainfall-baseline alerts | 0 |
| Candidate warning-by-station rows | 8,124 |
| Default model alert threshold | 0.65 |
| Default circular radius | 25 km |
| Default absolute elevated threshold | 2,000 m |

The safe summary for a reviewer is: five stations, nine-frame trajectories,
72-hour July window, gauge onset at 08Z, NWS about two hours earlier, and 27
valid labeled samples. The prototype is wired for evaluation, but it is not
yet a trained or operationally validated flood model.
