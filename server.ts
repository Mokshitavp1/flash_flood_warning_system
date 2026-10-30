import express from 'express';
import { parse } from 'csv-parse/sync';
import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import fsSync from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';

const app = express();
const root = process.cwd();
const outputs = path.join(root, 'outputs');
const port = Number(process.env.PORT ?? 8787);
type JsonRecord = Record<string, unknown>;
type PipelineJob = {
  id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  logs: string[];
  error?: string;
  startedAt: string;
  finishedAt?: string;
};
const jobs = new Map<string, PipelineJob>();

function numberOrNull(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function numberOrZero(value: unknown): number {
  return numberOrNull(value) ?? 0;
}

function parseCsvRows(text: string, fileName: string, expectedColumns?: number): string[][] {
  const rows = parse(text, {
    bom: true,
    skip_empty_lines: true,
    relax_column_count: false,
  }) as string[][];
  if (rows.length === 0) {
    throw new Error(`${fileName}: CSV is empty.`);
  }
  const columnCount = rows[0].length;
  if (expectedColumns !== undefined && columnCount !== expectedColumns) {
    throw new Error(`${fileName}: expected ${expectedColumns} columns in the header.`);
  }
  rows.slice(1).forEach((row, index) => {
    if (row.length !== columnCount) {
      throw new Error(
        `${fileName}: row ${index + 2} has ${row.length} columns; expected ${columnCount}.`,
      );
    }
  });
  return rows;
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function requireRecord(value: unknown, filePath: string): JsonRecord {
  if (!isRecord(value)) throw new Error(`${filePath}: expected a JSON object.`);
  return value;
}

function requiredString(value: unknown, pathName: string): string {
  if (typeof value !== 'string' || value.length === 0) throw new Error(`${pathName}: expected a non-empty string.`);
  return value;
}

export function mapAlertRecord(raw: JsonRecord, stationRows: string[][], index = 0) {
  const prefix = `alert_event_audit_log.json[${index}]`;
  const targetId = raw.target_station_id;
  return {
    id: requiredString(raw.alert_id, `${prefix}.alert_id`),
    method: requiredString(raw.method, `${prefix}.method`) as 'v0_circular_buffer' | 'v1_dem_flow_routing',
    sourceStationId: requiredString(raw.trigger_station_id, `${prefix}.trigger_station_id`),
    targetStationId: typeof targetId === 'string' ? targetId : '',
    targetName: typeof targetId === 'string'
      ? (stationRows.find(([id]) => id === targetId)?.[1] ?? targetId)
      : 'Unknown target',
    distanceKm: requiredNumber(raw.distance_km, `${prefix}.distance_km`),
    isDownhill: requiredBoolean(raw.is_downhill, `${prefix}.is_downhill`),
    alertStrength: requiredNumber(raw.alert_strength, `${prefix}.alert_strength`),
    leadTimeMin: requiredNumber(raw.estimated_arrival_lead_time_minutes, `${prefix}.estimated_arrival_lead_time_minutes`),
    status: raw.is_downhill ? 'critical_warning' : 'false_alarm_uphill',
  };
}

function requiredNumber(value: unknown, pathName: string): number {
  const parsed = numberOrNull(value);
  if (parsed === null) throw new Error(`${pathName}: expected a finite number.`);
  return parsed;
}

function requiredBoolean(value: unknown, pathName: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${pathName}: expected a boolean.`);
  return value;
}

async function readJson(filePath: string): Promise<unknown> {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8')) as unknown;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      throw new Error(`${filePath}: artifact file does not exist.`);
    }
    throw error;
  }
}

async function pipelineData() {
  await fs.access(path.join(outputs, 'training', 'learned_risk_model.json'));
  const stationsText = await fs.readFile(path.join(root, 'data', 'stations.csv'), 'utf8');
  const stationRows = parseCsvRows(stationsText, 'data/stations.csv', 5).slice(1);
  const terrain = requireRecord(
    await readJson(path.join(outputs, 'terrain', 'station_terrain.json')),
    'outputs/terrain/station_terrain.json',
  );
  const observations = new Map<string, {
    precipitation: number | null;
    temperature: number | null;
    pressure: number | null;
  }>();
  for (const [id, row] of await Promise.all(stationRows.map(async ([id]) => {
    try {
      const text = await fs.readFile(path.join(root, 'data', `GHCNh_${id}_2025.psv`), 'utf8');
      const rows = text.trim().split(/\r?\n/);
      const headers = rows[0].split('|').map((header) => header.toLowerCase());
      const latest = rows.slice(1).at(-1)?.split('|');
      if (!latest) return [id, null] as const;
      const value = (names: string[]) => {
        const index = names.map((name) => headers.indexOf(name)).find((index) => index >= 0);
        return index === undefined ? null : numberOrNull(latest[index]);
      };
      return [id, {
        precipitation: value(['precipitation_1_hour', 'precipitation_1h']),
        temperature: value(['temperature']),
        pressure: value(['station_level_pressure']),
      }] as const;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return [id, null] as const;
      throw error;
    }
  }))) {
    if (row) observations.set(id, row);
  }
  const stations = stationRows.map(([id, name, lat, lon, elevation]) => {
    const feature = isRecord(terrain[id]) ? terrain[id] : {};
    const observation = observations.get(id);
    const terrainAvailable = Object.keys(feature).length > 0;
    const precipitationAvailable = observation?.precipitation !== null && observation?.precipitation !== undefined;
    const temperatureAvailable = observation?.temperature !== null && observation?.temperature !== undefined;
    const pressureAvailable = observation?.pressure !== null && observation?.pressure !== undefined;
    return {
      id, name, lat: numberOrNull(lat), lon: numberOrNull(lon),
      elevationM: terrainAvailable
        ? numberOrNull(feature.elevation_m)
        : numberOrNull(elevation),
      isElevated: terrainAvailable ? feature.is_elevated_terrain === true : null,
      slopeDeg: terrainAvailable ? numberOrNull(feature.slope_degrees) : null,
      aspectDeg: terrainAvailable ? numberOrNull(feature.aspect_degrees) : null,
      flowAccumCells: terrainAvailable ? numberOrNull(feature.flow_accumulation_cells) : null,
      currentPrecipMm: observation?.precipitation ?? null,
      currentTempC: observation?.temperature ?? null,
      currentPressureHpa: observation?.pressure ?? null,
      availability: {
        precipitation: precipitationAvailable,
        temperature: temperatureAvailable,
        pressure: pressureAvailable,
        terrain: terrainAvailable,
      },
      status: precipitationAvailable && observation.precipitation! >= 15
        ? 'storm' : terrainAvailable && feature.is_elevated_terrain === true ? 'storm' : 'normal',
    };
  });

  const manifestText = await fs.readFile(path.join(outputs, 'manifests', 'satellite_trajectory_manifest.csv'), 'utf8');
  const manifestRows = parseCsvRows(manifestText, 'outputs/manifests/satellite_trajectory_manifest.csv').slice(1);
  const trajectoryFrames = manifestRows.slice(0, 9).map((row) => ({
    offsetMin: numberOrZero(row[4]),
    targetTime: row[2],
    status: row[7] === 'ok' ? 'ok' : 'missing',
    matchedTime: row[5] || null,
    satelliteId: row[8] || 'NONE',
    irTempK: null,
    cloudFraction: numberOrNull(row[10]),
    simulatedIntensity: 0,
  }));
  const alertsValue = await readJson(path.join(outputs, 'alerts', 'alert_event_audit_log.json'));
  if (!Array.isArray(alertsValue) || !alertsValue.every(isRecord)) {
    throw new Error('outputs/alerts/alert_event_audit_log.json: expected an array of objects.');
  }
  const alerts = alertsValue.map((raw, index) => mapAlertRecord(raw, stationRows, index));
  const evaluation = requireRecord(
    await readJson(path.join(outputs, 'evaluation', 'ablation_benchmark_results.json')),
    'outputs/evaluation/ablation_benchmark_results.json',
  );
  const ablationScores = Object.entries(evaluation).map(([stageId, rawValue]) => {
    const value = requireRecord(rawValue, `ablation_benchmark_results.json.${stageId}`);
    const metrics = isRecord(value.metrics) ? value.metrics : {};
    return {
    stageId,
    name: value.stage_name ?? stageId,
    csi: numberOrZero(metrics.csi),
    pod: numberOrZero(metrics.pod),
    far: numberOrZero(metrics.far),
    f1: numberOrZero(metrics.f1_score),
    description: value.description ?? value.notes ?? 'Generated by the Python pipeline.',
    isBaseline: stageId.startsWith('baseline'),
    };
  });
  return {
    generatedAt: new Date().toISOString(),
    source: 'HydroAlert pipeline outputs',
    synthetic: false,
    stations, trajectoryFrames, alerts, ablationScores,
  };
}

app.get('/api/pipeline-data', async (_request, response) => {
  try {
    response.json(await pipelineData());
  } catch (error) {
    response.status(500).json({
      error: 'Pipeline artifacts are invalid or unavailable.',
      details: error instanceof Error ? error.message : 'Unknown artifact error.',
    });
  }
});

app.post('/api/pipeline/run', (_request, response) => {
  const running = [...jobs.values()].find((job) => job.status === 'running' || job.status === 'queued');
  if (running) {
    response.status(409).json({ error: 'A pipeline job is already running.', jobId: running.id });
    return;
  }
  const job: PipelineJob = {
    id: randomUUID(),
    status: 'queued',
    logs: [],
    startedAt: new Date().toISOString(),
  };
  jobs.set(job.id, job);
  const projectPython = process.platform === 'win32'
    ? path.join(root, '.venv', 'Scripts', 'python.exe')
    : path.join(root, '.venv', 'bin', 'python');
  const python = process.env.PIPELINE_PYTHON
    ?? (fsSync.existsSync(projectPython) ? projectPython : 'python');
  const child = spawn(python, ['run_pipeline.py', 'data'], { cwd: root, windowsHide: true });
  job.status = 'running';
  const append = (chunk: Buffer) => {
    job.logs.push(...chunk.toString().split(/\r?\n/).filter(Boolean));
    job.logs = job.logs.slice(-200);
  };
  child.stdout.on('data', append);
  child.stderr.on('data', append);
  child.on('error', (error) => {
    job.status = 'failed';
    job.error = `Could not start pipeline with '${python}': ${error.message}`;
    job.finishedAt = new Date().toISOString();
  });
  child.on('close', (code) => {
    if (job.status === 'failed') return;
    job.status = code === 0 ? 'completed' : 'failed';
    if (code !== 0) job.error = `Pipeline exited with code ${code}.`;
    job.finishedAt = new Date().toISOString();
  });
  response.status(202).json({ jobId: job.id, status: job.status });
});

app.get('/api/pipeline/jobs/:jobId', (request, response) => {
  const job = jobs.get(request.params.jobId);
  if (!job) {
    response.status(404).json({ error: 'Pipeline job not found.' });
    return;
  }
  response.json(job);
});

app.use(express.static(path.join(root, 'dist')));
if (process.env.NODE_ENV !== 'test') {
  app.listen(port, () => console.log(`HydroAlert frontend API listening on http://localhost:${port}`));
}

export { app };
