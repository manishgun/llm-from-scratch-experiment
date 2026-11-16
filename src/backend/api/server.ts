import '../initNativeTf';
import * as tf from '@tensorflow/tfjs';
import express, { Request, Response, NextFunction } from 'express';
import http from 'http';
import os from 'os';
import { WebSocketServer, WebSocket } from 'ws';
import cors from 'cors';
import * as path from 'path';
import * as fs from 'fs';
import { DatasetManager } from '../dataset/datasetManager';
import { PREDEFINED_DATASETS } from '../dataset/predefinedDatasets';
import { TrainingEngine } from '../training/engine';
import { TextGenerator } from '../model/generation';
import { MiniGPT } from '../model/gpt';
import { CausalSelfAttention } from '../model/attention';
import { materializeSchedule } from '../training/schedule';
import { DEFAULT_CONFIG, PARAMETER_DOCS } from '../../shared/config';
import {
  FullConfig,
  WebSocketMessage,
  ClientMessage,
  ExperimentRecord,
  CheckpointMetadata,
  SystemInfo,
} from '../../shared/types';
import {
  fullConfigSchema,
  generateRequestSchema,
  attentionRequestSchema,
  datasetUploadSchema,
  datasetUrlSchema,
  checkpointRefSchema,
  resolveWithinDirectory,
  sanitizeFileName,
  assertAllowedDatasetUrl,
  formatValidationError,
} from './validation';
import { z } from 'zod';

/**
 * ============================================================================
 * MiniGPT Lab - Backend API & WebSocket Server
 * ============================================================================
 *
 * Provides REST endpoints and a WebSocket channel for:
 * 1. Dataset stats, upload, URL loading and previews.
 * 2. Parameter documentation and default configuration.
 * 3. Starting, pausing, resuming, stopping and resetting training.
 * 4. Real-time metric, log and status streaming.
 * 5. Checkpoint saving, listing, loading and resuming.
 * 6. Text generation - one-shot over HTTP, or streamed token-by-token over WS.
 * 7. Attention map extraction for the Attention Explorer.
 * 8. Experiment recording and comparison.
 * 9. A runtime report, including which TensorFlow backend is actually active.
 *
 * Every handler validates its input (see `validation.ts`) and every file path
 * derived from user input is confined to its intended directory.
 */

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

/**
 * CORS, restricted to loopback origins.
 *
 * The original `cors()` with no options sent `Access-Control-Allow-Origin: *`,
 * which let ANY web page the user had open in another tab read this server's
 * responses - including local dataset contents and checkpoint listings.
 *
 * The fix must not be a hardcoded port list, though. This server's own port is
 * configurable and the Vite dev server's is too, so a fixed list breaks the
 * app's own same-origin requests the moment either moves - which is exactly
 * what happened on the first run at a non-default port.
 *
 * Matching on the loopback HOST instead keeps the protection that matters
 * (https://evil.example cannot read anything) while working at any port. The
 * hostname is compared exactly, never by prefix: `startsWith('localhost')`
 * would also accept `localhost.evil.com`.
 */
const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]', '::1']);

function isLoopbackOrigin(origin: string): boolean {
  try {
    const { hostname, protocol } = new URL(origin);
    if (protocol !== 'http:' && protocol !== 'https:') return false;
    return LOOPBACK_HOSTS.has(hostname);
  } catch {
    return false;
  }
}

app.use(
  cors({
    origin: (origin, callback) => {
      // A missing Origin header means a same-origin or non-browser request
      // (curl, or the production build served from this same server).
      if (!origin || isLoopbackOrigin(origin)) return callback(null, true);
      callback(new CorsRejection(origin));
    },
  })
);

/** Distinguishes a CORS refusal so it becomes a 403 rather than a 500. */
class CorsRejection extends Error {
  constructor(origin: string) {
    super(
      `Origin ${origin} is not permitted. This server only accepts requests from localhost.`
    );
    this.name = 'CorsRejection';
  }
}
app.use(express.json({ limit: '50mb' }));

const datasetManager = new DatasetManager();
let currentConfig: FullConfig = structuredClone(DEFAULT_CONFIG);
let trainingEngine: TrainingEngine | null = null;

// Ensure working directories exist.
const checkpointsDir = path.resolve(process.cwd(), 'checkpoints');
const experimentsDir = path.resolve(process.cwd(), 'experiments');
for (const dir of [checkpointsDir, experimentsDir]) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

// ----------------------------------------------------------------------------
// WEBSOCKET BROADCASTING
// ----------------------------------------------------------------------------

function broadcast(msg: WebSocketMessage) {
  const data = JSON.stringify(msg);
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(data);
    }
  });
}

function sendTo(socket: WebSocket, msg: WebSocketMessage) {
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(msg));
  }
}

// ----------------------------------------------------------------------------
// RUNTIME REPORT
// ----------------------------------------------------------------------------

/**
 * Reports which TensorFlow backend is live, among other environment facts.
 *
 * WHY THIS IS A FIRST-CLASS FEATURE AND NOT A DEBUG LOG
 * `@tensorflow/tfjs-node` registers a native backend built on the C++
 * TensorFlow library, which is roughly an order of magnitude faster than the
 * pure-JavaScript `cpu` backend. If the native binding fails to load - a
 * missing DLL, a Node version it was not built against, a failed postinstall -
 * TF.js prints a warning and silently falls back.
 *
 * Nothing breaks. Training just takes ten times longer, and the user concludes
 * that "training a Transformer on CPU is hopeless" when in fact their install
 * is broken. Surfacing the active backend in the status bar turns an invisible
 * ten-fold performance loss into something obvious.
 */
function getSystemInfo(): SystemInfo {
  const backend = tf.getBackend();
  const cpus = os.cpus();
  return {
    tfBackend: backend,
    tfVersion: tf.version.tfjs ?? 'unknown',
    nodeVersion: process.version,
    platform: `${os.type()} ${os.release()} (${os.arch()})`,
    cpuModel: cpus[0]?.model?.trim() ?? 'unknown',
    cpuCores: cpus.length,
    totalMemoryMb: Math.round(os.totalmem() / 1024 / 1024),
    freeMemoryMb: Math.round(os.freemem() / 1024 / 1024),
    nativeAccelerated: backend === 'tensorflow',
  };
}

// ----------------------------------------------------------------------------
// STARTUP
// ----------------------------------------------------------------------------

(async () => {
  try {
    const info = getSystemInfo();
    console.log(
      `TensorFlow backend: ${info.tfBackend}` +
        (info.nativeAccelerated
          ? ' (native acceleration active)'
          : ' - WARNING: the native backend did not load, so training will be roughly 10x slower than it should be.')
    );

    const dataPath = await datasetManager.ensureTinyShakespeare();
    datasetManager.loadFromFile(
      dataPath,
      currentConfig.training.trainRatio,
      currentConfig.training.valRatio,
      currentConfig.training.testRatio
    );
    const stats = datasetManager.getStatistics();
    console.log(
      `Dataset "${stats.name}" ready: ${stats.totalCharacters.toLocaleString('en-US')} characters, ` +
        `${stats.uniqueCharacters} unique tokens.`
    );

    trainingEngine = createEngine();
  } catch (e) {
    console.error('Failed to initialize dataset on startup:', e);
  }
})();

/** Builds a training engine wired to the broadcast callbacks. */
function createEngine(): TrainingEngine {
  const engine = new TrainingEngine(currentConfig, datasetManager);
  engine.setCallbacks({
    onStatus: (status, message) => broadcast({ type: 'TRAINING_STATUS', status, message }),
    onMetric: (metric) => broadcast({ type: 'TRAINING_METRIC', metric }),
    onLog: (level, message) =>
      broadcast({ type: 'TRAINING_LOG', level, message, timestamp: Date.now() }),
    onCheckpoint: (checkpoint) => broadcast({ type: 'CHECKPOINT_SAVED', checkpoint }),
  });
  return engine;
}

/**
 * Replaces the engine, disposing the old one first.
 *
 * Disposing matters: each engine owns a model whose variables are tensors held
 * by TensorFlow. Dropping the reference without disposing leaked the entire
 * parameter set on every config change, and - because `variableGrads` without
 * an explicit variable list differentiates every registered variable - those
 * leaked variables also got gradients computed for them on every step.
 */
function replaceEngine(): TrainingEngine {
  if (trainingEngine) {
    trainingEngine.dispose();
  }
  trainingEngine = createEngine();
  return trainingEngine;
}

/** Rejects a mutation while a run is in progress. */
function requireIdle(res: Response): boolean {
  if (trainingEngine && trainingEngine.getStatus() === 'running') {
    res.status(409).json({
      error: 'Training is currently running. Pause or stop it before changing this.',
    });
    return false;
  }
  return true;
}

/**
 * Wraps an async handler so a rejected promise reaches the error middleware.
 * Express 4 does not await handlers, so without this an async throw becomes an
 * unhandled rejection and the request hangs until the client times out.
 */
function asyncHandler(
  fn: (req: Request, res: Response) => Promise<unknown>
): (req: Request, res: Response, next: NextFunction) => void {
  return (req, res, next) => {
    fn(req, res).catch(next);
  };
}

/** Parses a body against a schema, replying 400 with field detail on failure. */
function parseBody<T extends z.ZodTypeAny>(
  schema: T,
  req: Request,
  res: Response
): z.infer<T> | null {
  const result = schema.safeParse(req.body);
  if (!result.success) {
    res.status(400).json(formatValidationError(result.error));
    return null;
  }
  return result.data;
}

// ----------------------------------------------------------------------------
// REST API
// ----------------------------------------------------------------------------

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', uptimeSeconds: Math.round(process.uptime()) });
});

app.get('/api/system', (_req, res) => {
  res.json(getSystemInfo());
});

/** TensorFlow memory accounting, for spotting leaks during long runs. */
app.get('/api/system/memory', (_req, res) => {
  const mem = tf.memory();
  res.json({
    numTensors: mem.numTensors,
    numDataBuffers: mem.numDataBuffers,
    numBytes: mem.numBytes,
    heapUsedMb: Number((process.memoryUsage().heapUsed / 1024 / 1024).toFixed(1)),
    rssMb: Number((process.memoryUsage().rss / 1024 / 1024).toFixed(1)),
  });
});

// --- Configuration ----------------------------------------------------------

app.get('/api/config', (_req, res) => {
  res.json({ config: currentConfig, docs: PARAMETER_DOCS, defaults: DEFAULT_CONFIG });
});

app.post('/api/config', (req, res) => {
  const parsed = parseBody(fullConfigSchema, req, res);
  if (!parsed) return;
  if (!requireIdle(res)) return;

  currentConfig = parsed as FullConfig;

  // Re-split the dataset if the ratios changed, so val/test reflect the new
  // configuration rather than the ratios in force when the file was loaded.
  if (datasetManager.isLoaded()) {
    try {
      datasetManager.loadFromString(
        datasetManager.getRawText(),
        datasetManager.getName(),
        currentConfig.training.trainRatio,
        currentConfig.training.valRatio,
        currentConfig.training.testRatio,
        datasetManager.getTokenizerType()
      );
    } catch (err: any) {
      return res.status(400).json({ error: err.message });
    }
  }

  replaceEngine();
  res.json({ success: true, config: currentConfig });
});

/** The materialized learning-rate schedule, so the UI can plot it. */
app.get('/api/config/schedule', (_req, res) => {
  const schedule = materializeSchedule(currentConfig.training);
  // Downsample for transport: the chart cannot render more points than it has
  // pixels, and a 100,000-step run would otherwise send a megabyte of JSON.
  const maxPoints = 400;
  const stride = Math.max(1, Math.floor(schedule.length / maxPoints));
  const points = [];
  for (let i = 0; i < schedule.length; i += stride) {
    points.push({ step: i, learningRate: schedule[i] });
  }
  res.json({ points, peak: Math.max(...schedule) });
});

// --- Dataset ----------------------------------------------------------------

app.get('/api/dataset/predefined', (_req, res) => {
  res.json(PREDEFINED_DATASETS);
});

app.get('/api/dataset/stats', (_req, res) => {
  try {
    res.json({
      ...datasetManager.getStatistics(),
      tokenizerType: datasetManager.getTokenizerType(),
    });
  } catch (err: any) {
    res.status(400).json({ error: err.message });
  }
});

app.post(
  '/api/dataset/url',
  asyncHandler(async (req, res) => {
    const parsed = parseBody(datasetUrlSchema, req, res);
    if (!parsed) return;
    if (!requireIdle(res)) return;

    try {
      // Host allowlist: this endpoint makes the SERVER fetch a client-chosen
      // URL, which is an SSRF primitive if unrestricted. See validation.ts.
      assertAllowedDatasetUrl(parsed.url);
    } catch (err: any) {
      return res.status(400).json({ error: err.message });
    }

    await datasetManager.loadFromUrl(parsed.url, parsed.name, parsed.tokenizerType ?? 'char');
    replaceEngine();
    res.json({ success: true, stats: datasetManager.getStatistics() });
  })
);

app.post('/api/dataset/upload', (req, res) => {
  const parsed = parseBody(datasetUploadSchema, req, res);
  if (!parsed) return;
  if (!requireIdle(res)) return;

  try {
    const dataDir = path.resolve(process.cwd(), 'data');
    // Confined to the data directory: `name` comes from the request body, and
    // `path.join` would happily resolve `../../` out of it.
    const filePath = resolveWithinDirectory(dataDir, `${sanitizeFileName(parsed.name)}.txt`);

    fs.writeFileSync(filePath, parsed.content, 'utf-8');
    datasetManager.loadFromFile(
      filePath,
      currentConfig.training.trainRatio,
      currentConfig.training.valRatio,
      currentConfig.training.testRatio,
      parsed.tokenizerType ?? 'char'
    );
    replaceEngine();
    res.json({ success: true, stats: datasetManager.getStatistics() });
  } catch (err: any) {
    res.status(400).json({ error: err.message });
  }
});

/** Switches tokenization without re-downloading, using the loaded text. */
app.post('/api/dataset/tokenizer', (req, res) => {
  const schema = z.object({ tokenizerType: z.enum(['char', 'word']) });
  const parsed = parseBody(schema, req, res);
  if (!parsed) return;
  if (!requireIdle(res)) return;

  if (!datasetManager.isLoaded()) {
    return res.status(400).json({ error: 'No dataset is loaded.' });
  }

  try {
    datasetManager.loadFromString(
      datasetManager.getRawText(),
      datasetManager.getName(),
      currentConfig.training.trainRatio,
      currentConfig.training.valRatio,
      currentConfig.training.testRatio,
      parsed.tokenizerType
    );
    replaceEngine();
    res.json({ success: true, stats: datasetManager.getStatistics() });
  } catch (err: any) {
    res.status(400).json({ error: err.message });
  }
});

// --- Training controls ------------------------------------------------------

app.post('/api/training/start', (req, res) => {
  if (!datasetManager.isLoaded()) {
    return res.status(409).json({ error: 'The dataset is still loading. Try again shortly.' });
  }

  if (!trainingEngine) {
    trainingEngine = createEngine();
  }
  if (trainingEngine.getStatus() === 'running') {
    return res.status(409).json({ status: 'running', error: 'Training is already running.' });
  }

  // An optional config may accompany the start request.
  if (req.body && Object.keys(req.body).length > 0 && req.body.config) {
    const parsed = fullConfigSchema.safeParse(req.body.config);
    if (!parsed.success) {
      return res.status(400).json(formatValidationError(parsed.error));
    }
    // Only rebuild if something actually changed; rebuilding discards the
    // model, so a redundant rebuild would silently reset a paused run.
    if (JSON.stringify(parsed.data) !== JSON.stringify(currentConfig)) {
      currentConfig = parsed.data as FullConfig;
      replaceEngine();
    }
  }

  // Fire and forget: `start()` runs the loop and streams progress over the
  // WebSocket, so the HTTP response must not wait for it.
  trainingEngine!.start().catch((err) => {
    console.error('Training loop failed:', err);
  });
  res.json({ status: 'running' });
});

app.post('/api/training/pause', (_req, res) => {
  if (!trainingEngine) return res.status(400).json({ error: 'No active training engine.' });
  trainingEngine.pause();
  res.json({ status: 'paused' });
});

app.post('/api/training/stop', (_req, res) => {
  if (!trainingEngine) return res.status(400).json({ error: 'No active training engine.' });
  trainingEngine.stop();
  res.json({ status: 'stopped' });
});

app.post('/api/training/reset', (_req, res) => {
  if (!trainingEngine) return res.status(400).json({ error: 'No active training engine.' });
  trainingEngine.reset();
  res.json({ status: 'idle' });
});

app.get('/api/training/status', (_req, res) => {
  if (!trainingEngine) {
    return res.json({ status: 'idle', step: 0, history: [], datasetReady: datasetManager.isLoaded() });
  }
  const best = trainingEngine.getBestValidation();
  res.json({
    status: trainingEngine.getStatus(),
    step: trainingEngine.getCurrentStep(),
    history: trainingEngine.getHistory(),
    bestValLoss: Number.isFinite(best.loss) ? best.loss : null,
    bestValStep: best.step,
    parameters: trainingEngine.model?.countParameters() ?? null,
    datasetReady: datasetManager.isLoaded(),
  });
});

/** The per-component parameter breakdown, for the Architecture view. */
app.get('/api/model/breakdown', (_req, res) => {
  if (!trainingEngine) return res.status(400).json({ error: 'No engine.' });
  if (!trainingEngine.model) trainingEngine.initialize();
  res.json({
    totalParameters: trainingEngine.model!.countParameters(),
    breakdown: trainingEngine.model!.getParameterBreakdown(),
    tieWeights: trainingEngine.model!.tieWeights,
  });
});

/**
 * Runs the finite-difference gradient check and reports the result.
 *
 * Exposing this in the UI is deliberate: it lets a user verify, on their own
 * machine and their own configuration, that backpropagation through this
 * hand-written Transformer is actually correct - rather than taking the
 * README's word for it.
 */
app.post(
  '/api/model/gradient-check',
  asyncHandler(async (_req, res) => {
    if (!trainingEngine) return res.status(400).json({ error: 'No engine.' });
    if (!requireIdle(res)) return;
    const result = await trainingEngine.checkGradients();
    res.json(result);
  })
);

// --- Generation -------------------------------------------------------------

app.post(
  '/api/generate',
  asyncHandler(async (req, res) => {
    const parsed = parseBody(generateRequestSchema, req, res);
    if (!parsed) return;

    if (!trainingEngine || !trainingEngine.model) {
      return res.status(409).json({
        error:
          'No model exists yet. Start training, or load a checkpoint, and then generate. ' +
          '(An untrained model can also be initialized from the Architecture tab.)',
      });
    }

    const generator = new TextGenerator(trainingEngine.model, datasetManager.getTokenizer());
    const result = await generator.generate(parsed.prompt || '', {
      ...currentConfig.generation,
      ...parsed,
    });

    res.json({
      prompt: parsed.prompt ?? '',
      output: result.text,
      completion: result.completion,
      stoppedEarly: result.stoppedEarly,
      // The per-token distributions are large, so they are opt-in.
      tokens: parsed.includeProbabilities ? result.tokens : undefined,
    });
  })
);

/**
 * Attention maps for a prompt.
 *
 * The response is numLayers * numHeads * T * T floats, which is why the prompt
 * length is capped at 128 characters by the schema: at T=128 with 4 layers and
 * 4 heads that is already 262,144 numbers.
 */
app.post(
  '/api/model/attention',
  asyncHandler(async (req, res) => {
    const parsed = parseBody(attentionRequestSchema, req, res);
    if (!parsed) return;

    if (!trainingEngine || !trainingEngine.model) {
      return res.status(409).json({ error: 'No model exists yet.' });
    }

    const generator = new TextGenerator(trainingEngine.model, datasetManager.getTokenizer());
    const snapshot = await generator.attentionSnapshot(parsed.prompt);

    res.json({
      prompt: parsed.prompt,
      tokens: snapshot.tokens,
      numLayers: trainingEngine.model.config.numLayers,
      numHeads: trainingEngine.model.config.numHeads,
      weights: snapshot.weights,
    });
  })
);

// --- Checkpoints ------------------------------------------------------------

app.get(
  '/api/checkpoints',
  asyncHandler(async (_req, res) => {
    const files = fs.readdirSync(checkpointsDir).filter((f) => f.endsWith('.mgpt'));

    const checkpoints: CheckpointMetadata[] = [];
    for (const f of files) {
      const fullPath = path.join(checkpointsDir, f);
      const stat = fs.statSync(fullPath);
      try {
        // Reading only the JSON header means listing is cheap even for large
        // checkpoints - no weights are touched.
        const header = await MiniGPT.readCheckpointHeader(fullPath);
        checkpoints.push({
          id: f,
          name: f.replace(/\.mgpt$/, ''),
          step: header.step,
          createdAt: header.savedAt ?? stat.mtime.toISOString(),
          modelConfig: header.config,
          trainLoss: header.metrics?.trainLoss ?? 0,
          valLoss: header.metrics?.valLoss,
          filePath: fullPath,
          fileSizeBytes: stat.size,
          vocabularySize: header.config.vocabularySize,
          isBest: f === 'best.mgpt',
          hasOptimizerState: Boolean(header.optimizerState),
        });
      } catch {
        // A truncated or foreign file should not break the whole listing.
        continue;
      }
    }

    checkpoints.sort((a, b) => b.step - a.step);
    res.json(checkpoints);
  })
);

app.post(
  '/api/checkpoints/load',
  asyncHandler(async (req, res) => {
    const parsed = parseBody(checkpointRefSchema, req, res);
    if (!parsed) return;
    if (!requireIdle(res)) return;

    let fullPath: string;
    try {
      // Confined to the checkpoints directory - `fileName` is client-supplied.
      fullPath = resolveWithinDirectory(checkpointsDir, parsed.fileName);
    } catch (err: any) {
      return res.status(400).json({ error: err.message });
    }

    if (!fs.existsSync(fullPath)) {
      return res.status(404).json({ error: 'Checkpoint not found.' });
    }

    const header = await MiniGPT.readCheckpointHeader(fullPath);

    // Adopt the checkpoint's own architecture before loading its weights,
    // instead of failing on a shape mismatch against whatever the UI happens
    // to have configured. The vocabulary comes from the checkpoint too, so a
    // Shakespeare checkpoint cannot be silently decoded with a Moby Dick
    // vocabulary - which would produce fluent-looking garbage.
    currentConfig = { ...currentConfig, model: header.config };
    if (header.tokenizer && datasetManager.isLoaded()) {
      const liveVocab = datasetManager.getTokenizer().getVocabulary();
      const sameVocab =
        liveVocab.length === header.tokenizer.vocab.length &&
        liveVocab.every((c, i) => c === header.tokenizer!.vocab[i]);
      if (!sameVocab) {
        return res.status(409).json({
          error:
            `This checkpoint was trained on a different vocabulary ` +
            `(${header.tokenizer.vocab.length} tokens) than the loaded dataset ` +
            `(${liveVocab.length} tokens). Load the matching dataset first - otherwise ` +
            `token IDs mean different characters and the output would be meaningless.`,
        });
      }
    }

    const engine = replaceEngine();
    engine.initialize();
    await engine.resumeFrom(fullPath);

    res.json({
      success: true,
      message: `Loaded ${parsed.fileName} (step ${header.step}).`,
      step: header.step,
      config: currentConfig,
    });
  })
);

app.delete(
  '/api/checkpoints/:fileName',
  asyncHandler(async (req, res) => {
    let fullPath: string;
    try {
      fullPath = resolveWithinDirectory(checkpointsDir, req.params.fileName);
    } catch (err: any) {
      return res.status(400).json({ error: err.message });
    }
    if (!fs.existsSync(fullPath)) {
      return res.status(404).json({ error: 'Checkpoint not found.' });
    }
    fs.unlinkSync(fullPath);
    res.json({ success: true });
  })
);

/** Streams a checkpoint file to the browser for download. */
app.get('/api/checkpoints/:fileName/download', (req, res) => {
  let fullPath: string;
  try {
    fullPath = resolveWithinDirectory(checkpointsDir, req.params.fileName);
  } catch (err: any) {
    return res.status(400).json({ error: err.message });
  }
  if (!fs.existsSync(fullPath)) {
    return res.status(404).json({ error: 'Checkpoint not found.' });
  }
  res.download(fullPath);
});

// --- Experiments ------------------------------------------------------------

app.get('/api/experiments', (_req, res) => {
  const files = fs.readdirSync(experimentsDir).filter((f) => f.endsWith('.json'));
  const experiments: ExperimentRecord[] = [];
  for (const f of files) {
    try {
      experiments.push(JSON.parse(fs.readFileSync(path.join(experimentsDir, f), 'utf-8')));
    } catch {
      continue;
    }
  }
  experiments.sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
  res.json(experiments);
});

/**
 * Saves the current run as a named experiment.
 *
 * The server builds the record from its own authoritative state rather than
 * trusting a client-supplied body, so a saved experiment always reflects the
 * configuration and history that actually ran.
 */
app.post('/api/experiments', (req, res) => {
  const schema = z.object({
    name: z.string().min(1).max(200),
    description: z.string().max(2000).optional(),
  });
  const parsed = parseBody(schema, req, res);
  if (!parsed) return;

  if (!trainingEngine) return res.status(400).json({ error: 'No engine.' });

  const history = trainingEngine.getHistory();
  const last = history[history.length - 1];
  const best = trainingEngine.getBestValidation();
  const id = `exp_${Date.now()}_${sanitizeFileName(parsed.name).slice(0, 40)}`;

  const record: ExperimentRecord = {
    id,
    name: parsed.name,
    description: parsed.description ?? '',
    createdAt: new Date().toISOString(),
    config: currentConfig,
    totalStepsCompleted: trainingEngine.getCurrentStep(),
    finalTrainLoss: last?.trainLoss ?? 0,
    finalValLoss: Number.isFinite(best.loss) ? best.loss : undefined,
    trainingDurationMs: last?.elapsedMs ?? 0,
    totalParameters: trainingEngine.model?.countParameters() ?? 0,
    metricsHistory: history,
    sampleGenerations: [],
  };

  const expPath = resolveWithinDirectory(experimentsDir, `${id}.json`);
  fs.writeFileSync(expPath, JSON.stringify(record, null, 2));
  res.json({ success: true, id, experiment: record });
});

app.delete('/api/experiments/:id', (req, res) => {
  let fullPath: string;
  try {
    fullPath = resolveWithinDirectory(experimentsDir, `${req.params.id}.json`);
  } catch (err: any) {
    return res.status(400).json({ error: err.message });
  }
  if (!fs.existsSync(fullPath)) return res.status(404).json({ error: 'Not found.' });
  fs.unlinkSync(fullPath);
  res.json({ success: true });
});

// ----------------------------------------------------------------------------
// WEBSOCKET: STREAMING GENERATION
// ----------------------------------------------------------------------------

/**
 * Tracks in-flight streaming generations so they can be cancelled.
 *
 * Streaming matters for more than polish. A one-shot HTTP generation of 300
 * tokens on CPU takes several seconds, during which the user sees a spinner and
 * cannot stop it. Streaming shows the text appearing token by token - which is
 * also an accurate depiction of how autoregressive generation actually works -
 * and lets the Stop button take effect immediately.
 */
const activeGenerations = new Map<string, { cancelled: boolean }>();

wss.on('connection', (socket) => {
  // Tell the client about the runtime immediately on connect.
  sendTo(socket, { type: 'SYSTEM_INFO', info: getSystemInfo() });

  socket.on('message', async (raw) => {
    let msg: ClientMessage;
    try {
      msg = JSON.parse(raw.toString());
    } catch {
      return;
    }

    if (msg.type === 'CANCEL_GENERATION') {
      const entry = activeGenerations.get(msg.requestId);
      if (entry) entry.cancelled = true;
      return;
    }

    if (msg.type !== 'GENERATE_STREAM') return;

    const { requestId, prompt } = msg;
    if (typeof requestId !== 'string' || requestId.length > 128) return;

    if (!trainingEngine || !trainingEngine.model) {
      sendTo(socket, {
        type: 'GENERATION_ERROR',
        requestId,
        error: 'No model exists yet. Start training or load a checkpoint first.',
      });
      return;
    }

    const parsedConfig = generateRequestSchema.safeParse(msg.config ?? {});
    if (!parsedConfig.success) {
      sendTo(socket, {
        type: 'GENERATION_ERROR',
        requestId,
        error: formatValidationError(parsedConfig.error).error,
      });
      return;
    }

    const token = { cancelled: false };
    activeGenerations.set(requestId, token);

    try {
      const generator = new TextGenerator(trainingEngine.model, datasetManager.getTokenizer());
      const genConfig = { ...currentConfig.generation, ...parsedConfig.data };
      const total = genConfig.maxTokens ?? 200;

      const result = await generator.generate(
        typeof prompt === 'string' ? prompt.slice(0, 4096) : '',
        genConfig,
        (generated, index) => {
          if (token.cancelled || socket.readyState !== WebSocket.OPEN) {
            // Returning false from the callback cancels the generation loop.
            return false;
          }
          sendTo(socket, {
            type: 'GENERATION_TOKEN',
            requestId,
            token: generated,
            index,
            total,
          });
          return true;
        }
      );

      sendTo(socket, { type: 'GENERATION_DONE', requestId, text: result.text });
    } catch (err: any) {
      sendTo(socket, { type: 'GENERATION_ERROR', requestId, error: err.message });
    } finally {
      activeGenerations.delete(requestId);
    }
  });

  socket.on('close', () => {
    // Cancel anything this socket started, so a closed tab does not leave the
    // server generating tokens nobody will read.
    for (const entry of activeGenerations.values()) {
      entry.cancelled = true;
    }
  });
});

// ----------------------------------------------------------------------------
// STATIC FRONTEND (production build)
// ----------------------------------------------------------------------------

const frontendDist = path.resolve(process.cwd(), 'dist/frontend');
if (fs.existsSync(frontendDist)) {
  app.use(express.static(frontendDist));
  // SPA fallback, but only for non-API routes - otherwise a typo'd API path
  // returns the HTML shell with a 200, which is far more confusing to debug
  // than a 404.
  app.get('*', (req, res, next) => {
    if (req.path.startsWith('/api/')) return next();
    res.sendFile(path.join(frontendDist, 'index.html'));
  });
}

// 404 for unmatched API routes.
app.use('/api', (_req, res) => {
  res.status(404).json({ error: 'No such endpoint.' });
});

/**
 * Central error handler. Without it, an error thrown in a handler produces
 * Express's default HTML stack-trace page, which the frontend's `res.json()`
 * then fails to parse - turning a clear server error into an opaque
 * "Unexpected token < in JSON" in the browser console.
 */
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  if (res.headersSent) return;

  // A refused origin is a client-side policy decision, not a server fault.
  // Returning 500 for it is both wrong and actively misleading when debugging.
  if (err.name === 'CorsRejection') {
    res.status(403).json({ error: err.message });
    return;
  }

  console.error('Unhandled request error:', err);
  res.status(500).json({ error: err.message || 'Internal server error.' });
});

// ----------------------------------------------------------------------------
// LISTEN & GRACEFUL SHUTDOWN
// ----------------------------------------------------------------------------

const PORT = Number(process.env.PORT) || 3001;

/**
 * Report a port conflict in terms the user can act on.
 *
 * Without this handler, a port already in use produces an unhandled
 * 'error' event on the WebSocket server and Node prints a 30-line stack trace
 * ending in `EADDRINUSE`. The actual problem - something else is on this port -
 * is the one thing that trace does not say plainly.
 */
server.on('error', (error: NodeJS.ErrnoException) => {
  if (error.code === 'EADDRINUSE') {
    console.error(
      `\nPort ${PORT} is already in use by another process.\n\n` +
        `  Either stop whatever is using it, or run this server on a different port:\n` +
        `    PORT=3131 npm run dev\n\n` +
        `  The frontend reads the same variable, so both sides stay in agreement.\n`
    );
    process.exit(1);
  }
  if (error.code === 'EACCES') {
    console.error(`\nPermission denied binding port ${PORT}. Ports below 1024 need elevation.\n`);
    process.exit(1);
  }
  throw error;
});

server.listen(PORT, () => {
  console.log(`MiniGPT Lab API listening on http://localhost:${PORT}`);
});

/**
 * Shuts down cleanly on Ctrl+C.
 *
 * Stopping the training loop first matters: it is mid-step on a tensor graph,
 * and killing the process during a checkpoint write leaves a truncated file.
 * Disposing the engine and the mask cache also makes the native TensorFlow
 * backend release its memory rather than relying on process teardown.
 */
let shuttingDown = false;
function shutdown(signal: string) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`\nReceived ${signal}, shutting down...`);

  if (trainingEngine) {
    trainingEngine.stop();
    trainingEngine.dispose();
  }
  CausalSelfAttention.clearMaskCache();

  wss.clients.forEach((client) => client.close());
  server.close(() => {
    console.log('Server closed.');
    process.exit(0);
  });

  // Do not hang forever if a socket refuses to close.
  setTimeout(() => process.exit(0), 3000).unref();
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
