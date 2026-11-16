import { z } from 'zod';
import * as path from 'path';

/**
 * ============================================================================
 * MiniGPT Lab - Request Validation & Path Safety
 * ============================================================================
 *
 * WHY VALIDATE AT THE BOUNDARY
 * -----------------------------
 * The previous server assigned `req.body` straight onto the live configuration
 * object. Two consequences followed:
 *
 * 1. An invalid combination - most commonly an `embeddingDim` that is not
 *    divisible by `numHeads` - was accepted silently, and then threw from deep
 *    inside the attention constructor several seconds later, surfacing in the
 *    UI as an opaque "training error" with no indication of which field was
 *    wrong.
 *
 * 2. Any field could be any type. A string "64" where a number was expected
 *    propagates into tensor shapes and produces failures a long way from the
 *    cause.
 *
 * Validating here means every rejection is a 400 that names the offending
 * field and explains the constraint, and nothing downstream has to defend
 * itself against malformed input.
 */

/** Model architecture. */
export const modelConfigSchema = z
  .object({
    vocabularySize: z.number().int().min(2).max(100_000),
    embeddingDim: z.number().int().min(8).max(1024),
    numLayers: z.number().int().min(1).max(24),
    numHeads: z.number().int().min(1).max(32),
    contextLength: z.number().int().min(8).max(1024),
    feedForwardMultiplier: z.number().int().min(1).max(16),
    activation: z.enum(['gelu', 'relu']),
    dropoutRate: z.number().min(0).max(0.9),
    layerNormEpsilon: z.number().positive().max(1),
    initializerScale: z.number().positive().max(1),
    randomSeed: z.number().int().min(0).optional(),
    tieWeights: z.boolean().optional(),
  })
  // Cross-field constraint. Multi-head attention splits `embeddingDim` into
  // `numHeads` equal slices of size embeddingDim / numHeads, so the division
  // must be exact - there is no meaningful way to split 64 dimensions across
  // 5 heads.
  .refine((c) => c.embeddingDim % c.numHeads === 0, {
    message:
      'embeddingDim must be divisible by numHeads, because each head receives an equal slice of the embedding (headDim = embeddingDim / numHeads).',
    path: ['numHeads'],
  });

/** Training hyperparameters. */
export const trainingConfigSchema = z
  .object({
    batchSize: z.number().int().min(1).max(512),
    learningRate: z.number().positive().max(1),
    optimizerType: z.enum(['adam', 'sgd']),
    adamBeta1: z.number().min(0).max(1),
    adamBeta2: z.number().min(0).max(1),
    adamEpsilon: z.number().positive().max(1),
    trainingSteps: z.number().int().min(1).max(1_000_000),
    evaluationInterval: z.number().int().min(1).max(100_000),
    checkpointInterval: z.number().int().min(0).max(100_000),
    validationBatchCount: z.number().int().min(1).max(100),
    trainRatio: z.number().min(0.1).max(0.98),
    valRatio: z.number().min(0.01).max(0.5),
    testRatio: z.number().min(0).max(0.5),
    gradientClipNorm: z.number().min(0).max(100),
    lrSchedule: z.enum(['constant', 'cosine', 'linear']),
    warmupSteps: z.number().int().min(0).max(100_000),
    minLearningRate: z.number().min(0).max(1),
    weightDecay: z.number().min(0).max(1),
    earlyStoppingPatience: z.number().int().min(0).max(1000),
    dataSeed: z.number().int().min(0),
  })
  .refine((c) => Math.abs(c.trainRatio + c.valRatio + c.testRatio - 1) < 1e-6, {
    message: 'trainRatio + valRatio + testRatio must sum to 1.0.',
    path: ['trainRatio'],
  })
  .refine((c) => c.minLearningRate <= c.learningRate, {
    message: 'minLearningRate cannot exceed learningRate - the schedule decays downward.',
    path: ['minLearningRate'],
  })
  .refine((c) => c.warmupSteps < c.trainingSteps, {
    message:
      'warmupSteps must be less than trainingSteps, otherwise the run ends before the learning rate ever reaches its peak.',
    path: ['warmupSteps'],
  });

/** Sampling parameters. */
export const generationConfigSchema = z.object({
  temperature: z.number().min(0.01).max(5),
  topK: z.number().int().min(0).max(100_000),
  topP: z.number().min(0).max(1).optional(),
  maxTokens: z.number().int().min(1).max(4096),
  seed: z.number().int().min(0).optional(),
  stopSequences: z.array(z.string().max(64)).max(8).optional(),
  repetitionPenalty: z.number().min(0.5).max(3).optional(),
  samplePrompt: z.string().max(512).optional(),
});

/** The complete configuration posted by the UI. */
export const fullConfigSchema = z.object({
  id: z.string().min(1).max(128),
  name: z.string().min(1).max(200),
  description: z.string().max(2000),
  createdAt: z.string(),
  model: modelConfigSchema,
  training: trainingConfigSchema,
  generation: generationConfigSchema,
});

/** A one-off generation request. */
export const generateRequestSchema = z.object({
  prompt: z.string().max(4096).optional(),
  temperature: z.number().min(0.01).max(5).optional(),
  topK: z.number().int().min(0).max(100_000).optional(),
  topP: z.number().min(0).max(1).optional(),
  maxTokens: z.number().int().min(1).max(4096).optional(),
  seed: z.number().int().min(0).optional(),
  repetitionPenalty: z.number().min(0.5).max(3).optional(),
  stopSequences: z.array(z.string().max(64)).max(8).optional(),
  /** Include the per-token probability distributions in the response. */
  includeProbabilities: z.boolean().optional(),
});

/** An attention-map request. Prompts are capped because the payload is O(T^2). */
export const attentionRequestSchema = z.object({
  prompt: z.string().min(1).max(128),
});

/** Pasted / uploaded custom text. */
export const datasetUploadSchema = z.object({
  name: z.string().min(1).max(64),
  content: z.string().min(32).max(20_000_000),
  tokenizerType: z.enum(['char', 'word']).optional(),
});

/** Loading a dataset from a URL. */
export const datasetUrlSchema = z.object({
  url: z.string().url().max(2048),
  name: z.string().max(64).optional(),
  tokenizerType: z.enum(['char', 'word']).optional(),
});

/** Referring to a checkpoint by file name. */
export const checkpointRefSchema = z.object({
  fileName: z.string().min(1).max(256),
});

/**
 * ----------------------------------------------------------------------------
 * PATH SAFETY
 * ----------------------------------------------------------------------------
 *
 * THE VULNERABILITY THIS CLOSES
 * The previous handlers did `path.join(dataDir, name)` with `name` taken
 * straight from the request body. `path.join` resolves `..` segments, so a
 * body of `{"name": "../../../../Users/me/.ssh/authorized_keys"}` wrote
 * attacker-controlled content anywhere the server process could reach, and the
 * checkpoint loader read any file on disk the same way.
 *
 * This matters even for a localhost tool. Any web page the user visits can
 * POST to http://localhost:3001 (a cross-origin POST with a simple content
 * type is not blocked by the browser, and this server sends permissive CORS
 * headers), so "it only listens locally" is not a boundary.
 *
 * THE FIX, IN TWO INDEPENDENT LAYERS
 * 1. Reduce the input to its base name, discarding every directory component.
 * 2. Resolve the final path and verify it still sits inside the intended
 *    directory, rejecting it if not.
 *
 * Either check alone would suffice; both are present because this is the kind
 * of code that gets refactored later by someone who does not know why it was
 * written.
 */

/** Strips directory components and anything not filename-safe. */
export function sanitizeFileName(input: string): string {
  const base = path.basename(input);
  const cleaned = base.replace(/[^a-zA-Z0-9._-]/g, '_');
  // A name consisting only of dots would resolve to a directory.
  if (cleaned.length === 0 || /^\.+$/.test(cleaned)) {
    throw new Error('Invalid file name.');
  }
  return cleaned;
}

/**
 * Resolves `fileName` inside `baseDir` and asserts that the result has not
 * escaped it. Returns the absolute, verified path.
 */
export function resolveWithinDirectory(baseDir: string, fileName: string): string {
  const safeName = sanitizeFileName(fileName);
  const base = path.resolve(baseDir);
  const resolved = path.resolve(base, safeName);

  // `path.relative` returns a path starting with '..' when `resolved` is
  // outside `base`. Comparing prefixes with `startsWith` alone would wrongly
  // accept a sibling directory such as `/data-evil` for a base of `/data`.
  const relative = path.relative(base, resolved);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error('Resolved path escapes the permitted directory.');
  }

  return resolved;
}

/**
 * Hosts permitted for dataset downloads.
 *
 * WHY AN ALLOWLIST
 * `POST /api/dataset/url` makes the SERVER fetch a URL chosen by the client.
 * Unrestricted, that is a server-side request forgery primitive: the caller
 * can reach anything the server can but they cannot, including
 * `http://localhost:*` services and, on a cloud host, the instance metadata
 * endpoint at 169.254.169.254 that hands out credentials.
 *
 * The response body is then written to disk and displayed, so the data comes
 * back too - this is a fully readable SSRF, not a blind one.
 *
 * An allowlist of the few hosts the predefined datasets actually live on costs
 * nothing and removes the primitive entirely.
 */
const ALLOWED_DATASET_HOSTS = new Set([
  'raw.githubusercontent.com',
  'github.com',
  'www.gutenberg.org',
  'gutenberg.org',
  'huggingface.co',
  'cdn-lfs.huggingface.co',
]);

/**
 * Validates a dataset URL against the host allowlist and the HTTPS
 * requirement. Throws with an explanatory message if rejected.
 */
export function assertAllowedDatasetUrl(rawUrl: string): URL {
  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new Error('Not a valid URL.');
  }

  // Plain HTTP would also permit `http://169.254.169.254/...`, and the
  // downloader only speaks HTTPS in any case.
  if (parsed.protocol !== 'https:') {
    throw new Error('Only https:// dataset URLs are permitted.');
  }

  if (!ALLOWED_DATASET_HOSTS.has(parsed.hostname)) {
    throw new Error(
      `Host "${parsed.hostname}" is not in the dataset allowlist. ` +
        `Permitted hosts: ${[...ALLOWED_DATASET_HOSTS].join(', ')}. ` +
        `To use another source, download the file yourself and upload it on the Data tab.`
    );
  }

  return parsed;
}

/**
 * Formats a Zod error into a flat, human-readable list suitable for a 400
 * response body and for display in the UI.
 */
export function formatValidationError(error: z.ZodError): {
  error: string;
  issues: { field: string; message: string }[];
} {
  const issues = error.issues.map((issue) => ({
    field: issue.path.join('.') || '(root)',
    message: issue.message,
  }));
  return {
    error: `Invalid request: ${issues.map((i) => `${i.field} - ${i.message}`).join('; ')}`,
    issues,
  };
}
