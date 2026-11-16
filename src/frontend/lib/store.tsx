import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  FullConfig,
  TrainingStatus,
  TrainingStepMetric,
  CheckpointMetadata,
  SystemInfo,
  GeneratedToken,
  WebSocketMessage,
} from '../../shared/types';
import { ParameterDocumentation } from '../../shared/config';
import { api, ApiError, DatasetStatsResponse } from './api';

/**
 * ============================================================================
 * MiniGPT Lab - Application Store
 * ============================================================================
 *
 * Holds everything that is live: training status, the metric history, logs,
 * checkpoints, the runtime report and the WebSocket connection.
 *
 * WHY A SINGLE STORE RATHER THAN PER-COMPONENT STATE
 * Several independent views need the same stream. The run dashboard, the status
 * bar, the inference panel's "is a model available" check and the checkpoint
 * list all react to training events. With state local to one component, each
 * additional consumer meant either another WebSocket connection or another
 * polling loop. One connection feeding one store keeps them consistent by
 * construction - the status bar can never disagree with the chart.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LogEntry {
  id: number;
  level: 'info' | 'warn' | 'error';
  message: string;
  timestamp: number;
}

export interface Toast {
  id: number;
  tone: 'info' | 'success' | 'warning' | 'error';
  title: string;
  detail?: string;
}

export type ConnectionState = 'connecting' | 'open' | 'closed';

/** An in-flight or completed streaming generation. */
export interface StreamState {
  requestId: string | null;
  active: boolean;
  text: string;
  tokens: GeneratedToken[];
  total: number;
  error: string | null;
}

interface StoreValue {
  // Configuration
  config: FullConfig | null;
  docs: Record<string, ParameterDocumentation>;
  defaults: FullConfig | null;
  /** The config as edited in the UI but not yet sent to the server. */
  draftConfig: FullConfig | null;
  setDraftConfig: (next: FullConfig) => void;
  /** True when the draft differs from what the server holds. */
  isDirty: boolean;
  applyConfig: () => Promise<void>;
  revertConfig: () => void;

  // Training
  status: TrainingStatus;
  step: number;
  history: TrainingStepMetric[];
  latest: TrainingStepMetric | null;
  bestValLoss: number | null;
  bestValStep: number;
  parameters: number | null;
  startTraining: () => Promise<void>;
  pauseTraining: () => Promise<void>;
  stopTraining: () => Promise<void>;
  resetTraining: () => Promise<void>;

  // Data
  dataset: DatasetStatsResponse | null;
  refreshDataset: () => Promise<void>;

  // Checkpoints
  checkpoints: CheckpointMetadata[];
  refreshCheckpoints: () => Promise<void>;

  // Runtime
  system: SystemInfo | null;
  connection: ConnectionState;

  // Logs and notifications
  logs: LogEntry[];
  clearLogs: () => void;
  toasts: Toast[];
  pushToast: (tone: Toast['tone'], title: string, detail?: string) => void;
  dismissToast: (id: number) => void;
  /** Reports a caught error as a toast, extracting field detail from ApiError. */
  reportError: (error: unknown, context?: string) => void;

  // Streaming generation
  stream: StreamState;
  startStream: (prompt: string, config: Record<string, unknown>) => void;
  cancelStream: () => void;

  // Theme
  theme: 'dark' | 'light';
  toggleTheme: () => void;
}

const StoreContext = createContext<StoreValue | null>(null);

/** How many log lines to retain. Beyond this the oldest are dropped. */
const MAX_LOGS = 500;

const EMPTY_STREAM: StreamState = {
  requestId: null,
  active: false,
  text: '',
  tokens: [],
  total: 0,
  error: null,
};

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export const StoreProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [config, setConfig] = useState<FullConfig | null>(null);
  const [draftConfig, setDraftConfig] = useState<FullConfig | null>(null);
  const [defaults, setDefaults] = useState<FullConfig | null>(null);
  const [docs, setDocs] = useState<Record<string, ParameterDocumentation>>({});

  const [status, setStatus] = useState<TrainingStatus>('idle');
  const [step, setStep] = useState(0);
  const [history, setHistory] = useState<TrainingStepMetric[]>([]);
  const [bestValLoss, setBestValLoss] = useState<number | null>(null);
  const [bestValStep, setBestValStep] = useState(0);
  const [parameters, setParameters] = useState<number | null>(null);

  const [dataset, setDataset] = useState<DatasetStatsResponse | null>(null);
  const [checkpoints, setCheckpoints] = useState<CheckpointMetadata[]>([]);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [connection, setConnection] = useState<ConnectionState>('connecting');

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [stream, setStream] = useState<StreamState>(EMPTY_STREAM);

  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    // localStorage can throw in a private window with site data blocked, so
    // every access is guarded and falls back to the default.
    try {
      const saved = localStorage.getItem('minigpt-theme');
      if (saved === 'light' || saved === 'dark') return saved;
    } catch {
      /* ignore */
    }
    return 'dark';
  });

  const socketRef = useRef<WebSocket | null>(null);
  const idCounter = useRef(0);
  const nextId = () => ++idCounter.current;

  // --- Theme -------------------------------------------------------------
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('minigpt-theme', theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((previous) => (previous === 'dark' ? 'light' : 'dark'));
  }, []);

  // --- Toasts ------------------------------------------------------------
  const dismissToast = useCallback((id: number) => {
    setToasts((previous) => previous.filter((t) => t.id !== id));
  }, []);

  const pushToast = useCallback(
    (tone: Toast['tone'], title: string, detail?: string) => {
      const id = nextId();
      setToasts((previous) => [...previous.slice(-4), { id, tone, title, detail }]);
      // Errors persist until dismissed; everything else auto-clears. An error
      // that vanishes before it is read is the same as no error at all.
      if (tone !== 'error') {
        window.setTimeout(() => dismissToast(id), tone === 'warning' ? 8000 : 4500);
      }
    },
    [dismissToast]
  );

  const reportError = useCallback(
    (error: unknown, context?: string) => {
      if (error instanceof ApiError) {
        // Field-level validation detail is the most actionable thing the
        // server can return, so it is shown rather than collapsed.
        const detail =
          error.issues.length > 0
            ? error.issues.map((i) => `${i.field}: ${i.message}`).join('\n')
            : undefined;
        pushToast('error', context ? `${context}: ${error.message}` : error.message, detail);
      } else if (error instanceof Error) {
        pushToast('error', context ? `${context}: ${error.message}` : error.message);
      } else {
        pushToast('error', context ?? 'An unexpected error occurred.');
      }
    },
    [pushToast]
  );

  // --- Loaders -----------------------------------------------------------
  const refreshDataset = useCallback(async () => {
    try {
      setDataset(await api.datasetStats());
    } catch (error) {
      reportError(error, 'Could not load dataset statistics');
    }
  }, [reportError]);

  const refreshCheckpoints = useCallback(async () => {
    try {
      setCheckpoints(await api.checkpoints());
    } catch (error) {
      reportError(error, 'Could not list checkpoints');
    }
  }, [reportError]);

  const applyStatusResponse = useCallback(
    (response: Awaited<ReturnType<typeof api.trainingStatus>>) => {
      setStatus(response.status);
      setStep(response.step);
      setHistory(response.history ?? []);
      setBestValLoss(response.bestValLoss);
      setBestValStep(response.bestValStep);
      setParameters(response.parameters);
    },
    []
  );

  // --- Initial load ------------------------------------------------------
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const [configResponse, statusResponse] = await Promise.all([
          api.getConfig(),
          api.trainingStatus(),
        ]);
        if (cancelled) return;

        setConfig(configResponse.config);
        setDraftConfig(structuredClone(configResponse.config));
        setDefaults(configResponse.defaults);
        setDocs(configResponse.docs);
        applyStatusResponse(statusResponse);
      } catch (error) {
        if (!cancelled) reportError(error, 'Startup failed');
      }

      if (!cancelled) {
        await Promise.all([refreshDataset(), refreshCheckpoints()]);
      }
      try {
        const info = await api.system();
        if (!cancelled) setSystem(info);
      } catch {
        /* the WebSocket also delivers this */
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [applyStatusResponse, refreshDataset, refreshCheckpoints, reportError]);

  /**
   * The dataset is downloaded asynchronously at server startup, so the first
   * stats request can legitimately return `name: 'loading'`. Poll until it is
   * ready, then stop. The previous implementation recursed with setTimeout and
   * never cleared the timer, so a re-mount left orphaned polling loops running
   * for the lifetime of the page.
   */
  useEffect(() => {
    if (dataset && dataset.name !== 'loading' && dataset.totalCharacters > 0) return;
    const timer = window.setTimeout(() => {
      void refreshDataset();
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [dataset, refreshDataset]);

  // --- WebSocket ---------------------------------------------------------
  useEffect(() => {
    let disposed = false;
    let reconnectTimer = 0;
    let attempt = 0;

    const connect = () => {
      if (disposed) return;
      setConnection('connecting');

      // Connect to the backend port directly. In dev the page is served by
      // Vite on 5173 while the WebSocket lives on 3001, so the page's own
      // origin is the wrong target; in production both are the same and this
      // still resolves correctly.
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      // The backend port is injected at build time from the PORT environment
      // variable, so dev and production agree without a hardcoded constant.
      const backendPort = (import.meta.env.VITE_BACKEND_PORT as string) || '3001';
      // In dev the page is served by Vite on another port, so the page's own
      // origin is the wrong WebSocket target; in production they match.
      const port = window.location.port === backendPort ? window.location.port : backendPort;
      const socket = new WebSocket(`${protocol}//${window.location.hostname}:${port}`);
      socketRef.current = socket;

      socket.onopen = () => {
        if (disposed) return;
        attempt = 0;
        setConnection('open');
      };

      socket.onmessage = (event) => {
        let message: WebSocketMessage;
        try {
          message = JSON.parse(event.data);
        } catch {
          return;
        }

        switch (message.type) {
          case 'TRAINING_STATUS':
            setStatus(message.status);
            if (message.status === 'error' && message.message) {
              pushToast('error', 'Training stopped', message.message);
            }
            if (message.status === 'completed') {
              pushToast('success', 'Training complete');
              void refreshCheckpoints();
            }
            break;

          case 'TRAINING_METRIC': {
            const metric = message.metric;
            setStep(metric.step);
            setHistory((previous) => [...previous, metric]);
            if (metric.valLoss !== undefined) {
              setBestValLoss((previous) => {
                if (previous === null || metric.valLoss! < previous) {
                  setBestValStep(metric.step);
                  return metric.valLoss!;
                }
                return previous;
              });
            }
            break;
          }

          case 'TRAINING_LOG':
            setLogs((previous) => [
              ...previous.slice(-(MAX_LOGS - 1)),
              {
                id: nextId(),
                level: message.level,
                message: message.message,
                timestamp: message.timestamp,
              },
            ]);
            break;

          case 'CHECKPOINT_SAVED':
            void refreshCheckpoints();
            break;

          case 'SYSTEM_INFO':
            setSystem(message.info);
            break;

          // --- Streaming generation ---
          case 'GENERATION_TOKEN':
            setStream((previous) => {
              // Ignore tokens from a superseded request, so a cancelled
              // generation cannot interleave into a newer one.
              if (previous.requestId !== message.requestId) return previous;
              return {
                ...previous,
                text: previous.text + message.token.token,
                tokens: [...previous.tokens, message.token],
                total: message.total,
              };
            });
            break;

          case 'GENERATION_DONE':
            setStream((previous) =>
              previous.requestId === message.requestId
                ? { ...previous, active: false, text: message.text }
                : previous
            );
            break;

          case 'GENERATION_ERROR':
            setStream((previous) =>
              previous.requestId === message.requestId
                ? { ...previous, active: false, error: message.error }
                : previous
            );
            pushToast('error', 'Generation failed', message.error);
            break;
        }
      };

      socket.onclose = () => {
        if (disposed) return;
        setConnection('closed');
        setStream((previous) => (previous.active ? { ...previous, active: false } : previous));

        // Exponential backoff, capped. A tight reconnect loop against a
        // stopped backend floods the console and burns CPU; waiting forever
        // means the UI never recovers after a backend restart. Capping at 8s
        // gives both a quick recovery and a quiet failure mode.
        attempt++;
        const delay = Math.min(8000, 500 * 2 ** Math.min(attempt, 4));
        reconnectTimer = window.setTimeout(connect, delay);
      };

      socket.onerror = () => {
        // `onclose` always follows, and handles reconnection.
      };
    };

    connect();

    return () => {
      disposed = true;
      window.clearTimeout(reconnectTimer);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [pushToast, refreshCheckpoints]);

  /**
   * Re-sync from the server whenever the socket (re)opens.
   *
   * Metrics arrive as a stream of increments, so any message missed while
   * disconnected leaves a permanent gap in the chart. Re-fetching the
   * authoritative history on reconnect repairs it.
   */
  useEffect(() => {
    if (connection !== 'open') return;
    (async () => {
      try {
        applyStatusResponse(await api.trainingStatus());
      } catch {
        /* non-fatal */
      }
    })();
  }, [connection, applyStatusResponse]);

  // --- Configuration -----------------------------------------------------
  const isDirty = useMemo(
    () => Boolean(config && draftConfig) && JSON.stringify(config) !== JSON.stringify(draftConfig),
    [config, draftConfig]
  );

  const applyConfig = useCallback(async () => {
    if (!draftConfig) return;
    try {
      const response = await api.setConfig(draftConfig);
      setConfig(response.config);
      setDraftConfig(structuredClone(response.config));
      // Changing configuration rebuilds the model, which discards any trained
      // weights. Saying so prevents the user wondering where their run went.
      pushToast('success', 'Configuration applied', 'The model was rebuilt with fresh weights.');
      await Promise.all([refreshDataset(), api.trainingStatus().then(applyStatusResponse)]);
    } catch (error) {
      reportError(error, 'Could not apply configuration');
    }
  }, [draftConfig, pushToast, refreshDataset, applyStatusResponse, reportError]);

  const revertConfig = useCallback(() => {
    if (config) setDraftConfig(structuredClone(config));
  }, [config]);

  // --- Training controls -------------------------------------------------
  const startTraining = useCallback(async () => {
    try {
      // Send the draft so the Start button always runs what the user sees in
      // the Architecture panel, rather than silently using a stale config.
      await api.startTraining(draftConfig ?? undefined);
      if (draftConfig) {
        setConfig(structuredClone(draftConfig));
      }
      setStatus('running');
    } catch (error) {
      reportError(error, 'Could not start training');
    }
  }, [draftConfig, reportError]);

  const pauseTraining = useCallback(async () => {
    try {
      await api.pauseTraining();
    } catch (error) {
      reportError(error, 'Could not pause training');
    }
  }, [reportError]);

  const stopTraining = useCallback(async () => {
    try {
      await api.stopTraining();
    } catch (error) {
      reportError(error, 'Could not stop training');
    }
  }, [reportError]);

  const resetTraining = useCallback(async () => {
    try {
      await api.resetTraining();
      setStep(0);
      setHistory([]);
      setBestValLoss(null);
      setBestValStep(0);
      setStatus('idle');
      setLogs([]);
      pushToast('info', 'Run reset', 'Weights, optimizer state and metrics were cleared.');
    } catch (error) {
      reportError(error, 'Could not reset');
    }
  }, [pushToast, reportError]);

  // --- Streaming generation ----------------------------------------------
  const startStream = useCallback(
    (prompt: string, generationConfig: Record<string, unknown>) => {
      const socket = socketRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        pushToast('error', 'Not connected', 'The live connection to the backend is down.');
        return;
      }
      const requestId = `gen_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      setStream({
        requestId,
        active: true,
        text: prompt,
        tokens: [],
        total: (generationConfig.maxTokens as number) ?? 0,
        error: null,
      });
      socket.send(
        JSON.stringify({ type: 'GENERATE_STREAM', requestId, prompt, config: generationConfig })
      );
    },
    [pushToast]
  );

  const cancelStream = useCallback(() => {
    const socket = socketRef.current;
    setStream((previous) => {
      if (previous.requestId && socket?.readyState === WebSocket.OPEN) {
        socket.send(
          JSON.stringify({ type: 'CANCEL_GENERATION', requestId: previous.requestId })
        );
      }
      return { ...previous, active: false };
    });
  }, []);

  const clearLogs = useCallback(() => setLogs([]), []);

  const latest = history.length > 0 ? history[history.length - 1] : null;

  const value: StoreValue = {
    config,
    docs,
    defaults,
    draftConfig,
    setDraftConfig,
    isDirty,
    applyConfig,
    revertConfig,

    status,
    step,
    history,
    latest,
    bestValLoss,
    bestValStep,
    parameters,
    startTraining,
    pauseTraining,
    stopTraining,
    resetTraining,

    dataset,
    refreshDataset,

    checkpoints,
    refreshCheckpoints,

    system,
    connection,

    logs,
    clearLogs,
    toasts,
    pushToast,
    dismissToast,
    reportError,

    stream,
    startStream,
    cancelStream,

    theme,
    toggleTheme,
  };

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
};

export function useStore(): StoreValue {
  const store = useContext(StoreContext);
  if (!store) throw new Error('useStore must be used inside a StoreProvider.');
  return store;
}
