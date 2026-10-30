import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App, { parsePsvInput } from '../../src/App';
import { mapAlertRecord } from '../../server';

const validPsv = 'USW00093037|ALBUQUERQUE FOOTHILLS|35.1522|-106.4851|2420.0|2023|08|15|16|00|22.4|18.5|78.0|240|12.8|19.5|836.4|1006.2|38.0|38.0|48.5|52.0||TSRA|V';

const pipelinePayload = {
  generatedAt: '2026-01-01T00:00:00Z',
  source: 'test',
  stations: [{
    id: 'station-1', name: 'Test Station', lat: 1, lon: 2, elevationM: 3,
    isElevated: false, slopeDeg: 1, aspectDeg: 2, flowAccumCells: 3,
    currentPrecipMm: null, currentTempC: null, currentPressureHpa: null,
    availability: { precipitation: false, temperature: false, pressure: false, terrain: true },
    status: 'normal',
  }],
  trajectoryFrames: Array.from({ length: 9 }, (_, index) => ({
    offsetMin: index, targetTime: '2026-01-01T00:00:00Z', status: 'missing',
    matchedTime: null, satelliteId: 'NONE', irTempK: null, cloudFraction: null,
    simulatedIntensity: 0,
  })),
  alerts: [], ablationScores: [],
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe('pipeline API and parser contracts', () => {
  it('maps persisted alert fields to the frontend contract', () => {
    expect(mapAlertRecord({
      alert_id: 'a1', method: 'v1_dem_flow_routing', trigger_station_id: 's1',
      target_station_id: 's2', distance_km: 4, is_downhill: true,
      alert_strength: 0.8, estimated_arrival_lead_time_minutes: 12,
    }, [['s2', 'Valley']])).toEqual({
      id: 'a1', method: 'v1_dem_flow_routing', sourceStationId: 's1',
      targetStationId: 's2', targetName: 'Valley', distanceKm: 4,
      isDownhill: true, alertStrength: 0.8, leadTimeMin: 12,
      status: 'critical_warning',
    });
  });

  it('rejects malformed or incomplete PSV records', () => {
    const short = parsePsvInput('too|short');
    const invalidNumber = parsePsvInput(validPsv.replace('35.1522', 'not-a-number'));
    expect('error' in short && short.error).toMatch(/Expected exactly 25/);
    expect('error' in invalidNumber && invalidNumber.error).toMatch(/latitude must be numeric/);
    expect(parsePsvInput(validPsv)).toMatchObject({ stationId: 'USW00093037' });
  });
});

describe('dashboard states and actions', () => {
  it('shows loading and then API data', async () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));
    render(<App />);
    expect(screen.getByText('Loading real pipeline data...')).toBeInTheDocument();
  });

  it('shows API artifact errors', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('artifact missing', { status: 500 })));
    render(<App />);
    expect(await screen.findByText('Real model data is unavailable')).toBeInTheDocument();
    expect(screen.getByText('artifact missing')).toBeInTheDocument();
  });

  it('validates the PSV in the UI', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Response.json(pipelinePayload)));
    render(<App />);
    await waitFor(() => expect(screen.getAllByText('Test Station').length).toBeGreaterThan(0));
    fireEvent.click(screen.getAllByText('NOAA Drop-In')[0]);
    fireEvent.click(screen.getByText('Validate Record'));
    expect(screen.getByText('USW00093037')).toBeInTheDocument();
  });

  it('starts, polls, and refreshes after a successful pipeline job', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(Response.json(pipelinePayload))
      .mockResolvedValueOnce(Response.json({ jobId: 'job-1', status: 'running' }, { status: 202 }))
      .mockResolvedValueOnce(Response.json({ status: 'completed', logs: ['done'] }))
      .mockResolvedValueOnce(Response.json(pipelinePayload));
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);
    await waitFor(() => expect(screen.getAllByText('Run Pipeline').length).toBeGreaterThan(0));
    fireEvent.click(screen.getAllByText('Run Pipeline')[0]);
    await waitFor(() => expect(screen.getAllByText('CLI Execution Logs').length).toBeGreaterThan(0), { timeout: 5000 });
    fireEvent.click(screen.getAllByText('CLI Execution Logs')[0]);
    expect(await screen.findByText('Pipeline completed. Reloading persisted artifacts.')).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith('/api/pipeline/run', { method: 'POST' });
    expect(fetchMock).toHaveBeenCalledWith('/api/pipeline/jobs/job-1');
  });
});
