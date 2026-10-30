export interface Station {
  id: string;
  name: string;
  lat: number | null;
  lon: number | null;
  elevationM: number | null;
  isElevated: boolean | null;
  slopeDeg: number | null;
  aspectDeg: number | null;
  flowAccumCells: number | null;
  currentPrecipMm: number | null;
  currentTempC: number | null;
  currentPressureHpa: number | null;
  availability: {
    precipitation: boolean;
    temperature: boolean;
    pressure: boolean;
    terrain: boolean;
  };
  status: 'normal' | 'storm' | 'alert_downhill' | 'alert_false_alarm';
}

export interface TrajectoryFrame {
  offsetMin: number;
  targetTime: string;
  status: 'ok' | 'missing';
  matchedTime: string | null;
  satelliteId: string;
  irTempK: number | null;
  cloudFraction: number | null;
  simulatedIntensity: number; // 0 to 1
}

export interface AlertEvent {
  id: string;
  method: 'v0_circular_buffer' | 'v1_dem_flow_routing';
  sourceStationId: string;
  targetStationId: string;
  targetName: string;
  distanceKm: number;
  isDownhill: boolean;
  alertStrength: number;
  leadTimeMin: number;
  status: 'critical_warning' | 'false_alarm_uphill' | 'informative';
}

export interface ParsedPsvRecord {
  stationId: string;
  stationName: string;
  latitude: number | null;
  longitude: number | null;
  elevationM: number | null;
  date: string;
  temperatureC: number | null;
  relativeHumidity: number | null;
  stationPressureHpa: number | null;
  precipitation1hMm: number | null;
  weatherCode: string;
  qcFlag: string;
}

export type ParsedPsvOutput = ParsedPsvRecord | { error: string };

export type ActiveTab =
  | 'simulation'
  | 'trajectory'
  | 'ingestion'
  | 'model'
  | 'ablation'
  | 'runner'
  | 'overview';

export interface AblationScore {
  stageId: string;
  name: string;
  csi: number;
  pod: number;
  far: number;
  f1: number;
  description: string;
  isBaseline: boolean;
}

export interface PipelineData {
  generatedAt: string;
  stations: Station[];
  trajectoryFrames: TrajectoryFrame[];
  alerts: AlertEvent[];
  ablationScores: AblationScore[];
  source: string;
  synthetic: boolean;
  warning?: string;
}
