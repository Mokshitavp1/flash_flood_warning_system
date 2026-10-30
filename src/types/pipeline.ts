export interface Station {
  id: string;
  name: string;
  lat: number;
  lon: number;
  elevationM: number;
  isElevated: boolean;
  slopeDeg: number;
  aspectDeg: number;
  flowAccumCells: number;
  currentPrecipMm: number;
  currentTempC: number;
  currentPressureHpa: number;
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
}
