import {
  FullConfig,
  TrainingStatus,
  TrainingStepMetric,
  CheckpointMetadata,
  ExperimentRecord,
  DatasetStatistics,
  SystemInfo,
  AttentionSnapshot,
  GeneratedToken,
  GradientCheckResult,
} from '../../shared/types';
import { ParameterDocumentation } from '../../shared/config';

/**
 * ============================================================================
 * MiniGPT Lab - Typed API Client
 * ============================================================================
 *
 * Every network call the UI makes goes through `request`, which exists for one
 * reason: the server reports failures as `{ error, issues }` with a meaningful
 * HTTP status, and that information must reach the user.
 *
 * The previous frontend called `fetch(...).then(r => r.json())` and discarded
 * the status entirely. A validation failure, a 409 "training already running"
 * and a 500 all became either silence or an unexplained blank panel. Here a
 * non-2xx response becomes an `ApiError` carrying the server's own message and
 * per-field issues, which the views surface as a toast.
 */

/** An error carrying the server's explanation, not just a status code. */
export class ApiError extends Error {
  public readonly status: number;
  public readonly issues: { field: string; message: string }[];

  constructor(status: number, message: string, issues: { field: string; message: string }[] = []) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.issues = issues;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
    });
  } catch (cause) {
    // A network-level failure almost always means the backend is not running.
    // Saying so is far more useful than "Failed to fetch".
    throw new ApiError(
      0,
      'Cannot reach the backend on port 3001. Is it running? Start it with `npm run dev`.'
    );
  }

  const text = await response.text();
  let body: unknown = null;
  if (text.length > 0) {
    try {
      body = JSON.parse(text);
    } catch {
      // A non-JSON body from an API route means an unhandled server error that
      // produced an HTML page. Surface a fragment rather than a parse error.
      if (!response.ok) {
        throw new ApiError(response.status, text.slice(0, 300));
      }
    }
  }

  if (!response.ok) {
    const payload = body as { error?: string; issues?: { field: string; message: string }[] };
    throw new ApiError(
      response.status,
      payload?.error ?? `Request failed with status ${response.status}.`,
      payload?.issues ?? []
    );
  }

  return body as T;
}

const post = <T>(path: string, payload?: unknown): Promise<T> =>
  request<T>(path, { method: 'POST', body: payload === undefined ? undefined : JSON.stringify(payload) });

const get = <T>(path: string): Promise<T> => request<T>(path, { method: 'GET' });

const del = <T>(path: string): Promise<T> => request<T>(path, { method: 'DELETE' });

// ---------------------------------------------------------------------------

export interface TrainingStatusResponse {
  status: TrainingStatus;
  step: number;
  history: TrainingStepMetric[];
  bestValLoss: number | null;
  bestValStep: number;
  parameters: number | null;
  datasetReady: boolean;
}

export interface GenerateResponse {
  prompt: string;
  output: string;
  completion: string;
  stoppedEarly: boolean;
  tokens?: GeneratedToken[];
}

export interface ParameterBreakdownResponse {
  totalParameters: number;
  breakdown: { component: string; parameters: number }[];
  tieWeights: boolean;
}

export interface SchedulePreview {
  points: { step: number; learningRate: number }[];
  peak: number;
}

export interface PredefinedDataset {
  name: string;
  url: string;
  description: string;
  sizeMb: string;
}

export type DatasetStatsResponse = DatasetStatistics & { tokenizerType: 'char' | 'word' };

export const api = {
  // --- Runtime ------------------------------------------------------------
  system: () => get<SystemInfo>('/api/system'),
  memory: () =>
    get<{
      numTensors: number;
      numDataBuffers: number;
      numBytes: number;
      heapUsedMb: number;
      rssMb: number;
    }>('/api/system/memory'),

  // --- Configuration ------------------------------------------------------
  getConfig: () =>
    get<{
      config: FullConfig;
      docs: Record<string, ParameterDocumentation>;
      defaults: FullConfig;
    }>('/api/config'),
  setConfig: (config: FullConfig) =>
    post<{ success: boolean; config: FullConfig }>('/api/config', config),
  schedulePreview: () => get<SchedulePreview>('/api/config/schedule'),

  // --- Dataset ------------------------------------------------------------
  datasetStats: () => get<DatasetStatsResponse>('/api/dataset/stats'),
  predefinedDatasets: () => get<PredefinedDataset[]>('/api/dataset/predefined'),
  loadDatasetUrl: (url: string, name?: string, tokenizerType?: 'char' | 'word') =>
    post<{ success: boolean; stats: DatasetStatistics }>('/api/dataset/url', {
      url,
      name,
      tokenizerType,
    }),
  uploadDataset: (name: string, content: string, tokenizerType?: 'char' | 'word') =>
    post<{ success: boolean; stats: DatasetStatistics }>('/api/dataset/upload', {
      name,
      content,
      tokenizerType,
    }),
  setTokenizer: (tokenizerType: 'char' | 'word') =>
    post<{ success: boolean; stats: DatasetStatistics }>('/api/dataset/tokenizer', {
      tokenizerType,
    }),

  // --- Training -----------------------------------------------------------
  trainingStatus: () => get<TrainingStatusResponse>('/api/training/status'),
  startTraining: (config?: FullConfig) =>
    post<{ status: string }>('/api/training/start', config ? { config } : {}),
  pauseTraining: () => post<{ status: string }>('/api/training/pause'),
  stopTraining: () => post<{ status: string }>('/api/training/stop'),
  resetTraining: () => post<{ status: string }>('/api/training/reset'),

  // --- Model inspection ---------------------------------------------------
  parameterBreakdown: () => get<ParameterBreakdownResponse>('/api/model/breakdown'),
  gradientCheck: () => post<GradientCheckResult>('/api/model/gradient-check'),
  attention: (prompt: string) => post<AttentionSnapshot>('/api/model/attention', { prompt }),

  // --- Generation ---------------------------------------------------------
  generate: (payload: {
    prompt: string;
    temperature?: number;
    topK?: number;
    topP?: number;
    maxTokens?: number;
    seed?: number;
    repetitionPenalty?: number;
    includeProbabilities?: boolean;
  }) => post<GenerateResponse>('/api/generate', payload),

  // --- Checkpoints --------------------------------------------------------
  checkpoints: () => get<CheckpointMetadata[]>('/api/checkpoints'),
  loadCheckpoint: (fileName: string) =>
    post<{ success: boolean; message: string; step: number; config: FullConfig }>(
      '/api/checkpoints/load',
      { fileName }
    ),
  deleteCheckpoint: (fileName: string) =>
    del<{ success: boolean }>(`/api/checkpoints/${encodeURIComponent(fileName)}`),
  checkpointDownloadUrl: (fileName: string) =>
    `/api/checkpoints/${encodeURIComponent(fileName)}/download`,

  // --- Experiments --------------------------------------------------------
  experiments: () => get<ExperimentRecord[]>('/api/experiments'),
  saveExperiment: (name: string, description?: string) =>
    post<{ success: boolean; id: string; experiment: ExperimentRecord }>('/api/experiments', {
      name,
      description,
    }),
  deleteExperiment: (id: string) =>
    del<{ success: boolean }>(`/api/experiments/${encodeURIComponent(id)}`),
};
