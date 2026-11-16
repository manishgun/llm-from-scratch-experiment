/**
 * ============================================================================
 * MiniGPT Lab - Global Types and Definitions
 * ============================================================================
 *
 * This file contains TypeScript interfaces and types for:
 * 1. Model hyperparameters (embedding dimension, layers, heads, etc.)
 * 2. Training configuration (batch size, learning rate, steps, etc.)
 * 3. Text generation options (temperature, top-k, max tokens)
 * 4. Dataset parameters and statistics
 * 5. Training metrics and WebSocket streaming messages
 * 6. Experiment records for side-by-side comparison
 */

/**
 * Dataset statistics returned by the backend for the Library & Text tab.
 */
export interface DatasetStatistics {
  name: string;
  totalCharacters: number;
  uniqueCharacters: number;
  trainCharacters: number;
  valCharacters: number;
  testCharacters: number;
  samplePreview: string;
  charDistribution: { char: string; count: number; percentage: number }[];
}

/**
 * Hyperparameters specifying the neural architecture of the GPT model.
 */
export interface ModelConfig {
  /**
   * The number of unique characters in the vocabulary.
   * E.g. If the text has letters a-z, A-Z, punctuation, and spaces,
   * vocabularySize might be 65.
   */
  vocabularySize: number;

  /**
   * The size of the vector representation for each token and position.
   * Also known as d_model.
   * E.g., 64 means each token is represented by a list of 64 numbers.
   */
  embeddingDim: number;

  /**
   * Number of Transformer blocks stacked on top of each other.
   * More layers allow the model to learn more complex abstract representations.
   */
  numLayers: number;

  /**
   * Number of attention heads in Multi-Head Attention.
   * Each head can focus on different aspects of relationships (e.g. grammar, rhyming).
   * Note: embeddingDim MUST be evenly divisible by numHeads.
   */
  numHeads: number;

  /**
   * Maximum number of consecutive tokens the model can look at at one time.
   * Also called sequence length, window size, or block size.
   */
  contextLength: number;

  /**
   * Multiplier for the hidden layer in the Feed-Forward Network.
   * In standard GPT-2, this is 4, meaning hiddenDim = 4 * embeddingDim.
   */
  feedForwardMultiplier: number;

  /**
   * Activation function used inside the Feed-Forward Network.
   * 'gelu' is standard in GPT; 'relu' is a faster alternative.
   */
  activation: 'gelu' | 'relu';

  /**
   * Dropout rate (between 0.0 and 1.0) applied during training to prevent overfitting.
   * For small educational models, 0.0 or 0.1 is typical.
   */
  dropoutRate: number;

  /**
   * Small number added to the denominator in LayerNorm to prevent division by zero.
   * Standard default is 1e-5.
   */
  layerNormEpsilon: number;

  /**
   * Standard deviation scale for initial Gaussian weights.
   * Smaller scale prevents exploding activations at initialization.
   */
  initializerScale: number;

  /**
   * Random seed used for reproducible weight initialization.
   * Two runs with the same seed and the same config produce bit-identical
   * weights, which is what makes hyperparameter comparisons meaningful.
   */
  randomSeed?: number;

  /**
   * Share one matrix between the input token embedding and the output
   * language-model head (GPT-2 does this).
   *
   * Saves `vocabSize * embeddingDim` parameters and usually improves quality,
   * because the two matrices represent inverse views of the same vocabulary.
   * Default: true.
   */
  tieWeights?: boolean;
}

/**
 * Parameters controlling how the model learns during training.
 */
export interface TrainingConfig {
  /**
   * How many text sequences are processed together in one forward/backward pass.
   * Higher batch sizes yield smoother gradient estimates but use more RAM.
   */
  batchSize: number;

  /**
   * Step size taken by the Adam optimizer in direction of negative gradient.
   * E.g. 0.001 or 0.0005. If too high, training diverges; if too low, learning is slow.
   */
  learningRate: number;

  /**
   * Name of the optimizer. Standard GPT uses Adam.
   */
  optimizerType: 'adam' | 'sgd';

  /**
   * Adam exponential decay rate for the first moment estimates (momentum).
   * Default: 0.9.
   */
  adamBeta1: number;

  /**
   * Adam exponential decay rate for the second moment estimates (uncentered variance).
   * Default: 0.999.
   */
  adamBeta2: number;

  /**
   * Small constant for numerical stability in Adam optimizer.
   * Default: 1e-8.
   */
  adamEpsilon: number;

  /**
   * Total number of training steps (iterations) to run.
   */
  trainingSteps: number;

  /**
   * How often (in steps) to compute validation loss and report progress.
   */
  evaluationInterval: number;

  /**
   * How often (in steps) to save a model checkpoint to disk.
   */
  checkpointInterval: number;

  /**
   * Number of batches to sample from the validation set when measuring val loss.
   */
  validationBatchCount: number;

  /**
   * Dataset split ratios for Train, Validation, and Test.
   * Sum should be 1.0 (e.g. 0.8, 0.1, 0.1).
   */
  trainRatio: number;
  valRatio: number;
  testRatio: number;

  /**
   * Maximum global gradient norm. Before each optimizer step, if the L2 norm
   * of all gradients concatenated together exceeds this value, every gradient
   * is scaled down by the same factor so the norm equals it exactly.
   *
   * This preserves the gradient's DIRECTION while capping its MAGNITUDE, and
   * is the single most effective guard against a loss spike to NaN. Set to 0
   * to disable. Standard value: 1.0.
   */
  gradientClipNorm: number;

  /**
   * Learning-rate schedule shape.
   *
   * - `constant` holds the learning rate fixed for the whole run.
   * - `cosine`   decays it smoothly along a half cosine from the peak down to
   *              `minLearningRate`. This is what GPT-2/GPT-3 and essentially
   *              every modern LLM use: large early steps to find a good basin,
   *              tiny late steps to settle into its floor.
   * - `linear`   decays straight down to `minLearningRate`.
   */
  lrSchedule: 'constant' | 'cosine' | 'linear';

  /**
   * Number of steps spent linearly ramping the learning rate from ~0 up to
   * `learningRate` before the decay phase begins.
   *
   * Warmup exists because Adam's second-moment estimate is unreliable in the
   * first few dozen steps (it has barely any history to average), so early
   * full-size steps can be wildly misdirected and damage the initialization.
   * Ramping in avoids that.
   */
  warmupSteps: number;

  /**
   * The floor the schedule decays toward, usually ~10% of `learningRate`.
   */
  minLearningRate: number;

  /**
   * Decoupled weight decay (AdamW). Pulls every weight matrix slightly toward
   * zero on each step, which penalises unnecessarily large weights and
   * improves generalisation.
   *
   * "Decoupled" means the decay is applied directly to the weights rather than
   * added into the gradient; adding it to the gradient (classic L2 in Adam)
   * lets Adam's per-parameter normalisation cancel most of its effect.
   * Biases and LayerNorm parameters are always exempt. Set 0 to disable.
   */
  weightDecay: number;

  /**
   * Stop training if validation loss has not improved for this many
   * evaluations. 0 disables early stopping.
   */
  earlyStoppingPatience: number;

  /**
   * Seed for batch sampling, kept separate from the weight-init seed so you
   * can vary data order while holding initialization fixed, or vice versa.
   */
  dataSeed: number;
}

/**
 * Parameters controlling how text is sampled from output probabilities.
 */
export interface GenerationConfig {
  /**
   * Softmax temperature (e.g. 0.1 to 2.0).
   * Higher temperature = more creative/random text.
   * Lower temperature = more focused/deterministic text.
   * 1.0 = raw probabilities without modification.
   */
  temperature: number;

  /**
   * Restricts sampling to the top K most probable tokens.
   * If topK is 10, only the 10 highest-probability choices are considered.
   * If topK is 0 or >= vocabSize, all tokens are considered.
   */
  topK: number;

  /**
   * Nucleus sampling: keep the smallest set of tokens whose cumulative probability is >= topP.
   * E.g. 0.9. (Optional extension).
   */
  topP?: number;

  /**
   * Maximum number of new characters/tokens to generate after the prompt.
   */
  maxTokens: number;

  /**
   * Random seed for repeatable text generation.
   */
  seed?: number;

  /**
   * Optional string sequences that immediately stop generation if encountered.
   */
  stopSequences?: string[];

  /**
   * Penalty applied to tokens already present in the generated text.
   *
   * A small model with a low temperature falls into loops ("the the the").
   * This divides the logit of any already-seen token by `repetitionPenalty`
   * (for positive logits) before sampling, making repeats progressively less
   * attractive. 1.0 disables it; 1.1-1.2 is a typical useful range.
   */
  repetitionPenalty?: number;

  /**
   * The prompt used for the automatic sample generated at each evaluation
   * during training. Dataset-dependent: "ROMEO:" is meaningful for Shakespeare
   * and meaningless for Moby Dick, so it belongs in configuration rather than
   * hardcoded in the training loop.
   */
  samplePrompt?: string;
}

/**
 * Full combined project configuration.
 */
export interface FullConfig {
  id: string;
  name: string;
  description: string;
  createdAt: string;
  model: ModelConfig;
  training: TrainingConfig;
  generation: GenerationConfig;
}

/**
 * Metrics logged at a specific training step.
 */
export interface TrainingStepMetric {
  step: number;
  totalSteps: number;
  trainLoss: number;
  valLoss?: number;
  learningRate: number;
  elapsedMs: number;
  estimatedRemainingMs: number;
  stepsPerSecond: number;
  memoryUsageMb: number;
  generatedSample?: string;
  timestamp: number;

  /**
   * The L2 norm of all gradients before clipping.
   *
   * This is the most diagnostic single number in training. A norm that is
   * stable and slowly shrinking means healthy learning. A norm that spikes by
   * orders of magnitude means the learning rate is too high and the run is
   * about to diverge - visible BEFORE the loss itself goes bad.
   */
  gradNorm?: number;

  /**
   * exp(trainLoss): the effective number of tokens the model is still
   * choosing between. More interpretable than raw loss - perplexity 10 means
   * "about as uncertain as a fair 10-sided die".
   */
  perplexity?: number;

  /** exp(valLoss). */
  valPerplexity?: number;

  /**
   * Loss converted to bits per token (loss / ln 2). The standard unit for
   * comparing character-level language models across papers.
   */
  bitsPerToken?: number;

  /** Throughput in tokens/second (batchSize * contextLength * stepsPerSecond). */
  tokensPerSecond?: number;

  /** Live count of tensors held by TensorFlow, for spotting leaks. */
  tensorCount?: number;
}

/**
 * Runtime environment report, shown in the UI's status bar.
 *
 * Surfacing the ACTIVE TENSORFLOW BACKEND matters: the native `tensorflow`
 * backend is roughly an order of magnitude faster than the pure-JavaScript
 * `cpu` fallback. If the native binding fails to load, TF.js silently falls
 * back and training just... takes ten times longer, with no error. Displaying
 * the backend name turns that into something the user can see.
 */
export interface SystemInfo {
  tfBackend: string;
  tfVersion: string;
  nodeVersion: string;
  platform: string;
  cpuModel: string;
  cpuCores: number;
  totalMemoryMb: number;
  freeMemoryMb: number;
  /** True when the fast native backend is active. */
  nativeAccelerated: boolean;
}

/**
 * One token's predicted probability, for the inference probability inspector.
 */
export interface TokenProbability {
  token: string;
  tokenId: number;
  probability: number;
  /** True if this is the token that was actually sampled. */
  chosen: boolean;
}

/**
 * A generated token together with the distribution it was drawn from.
 * Lets the UI show, per character, what the model was considering.
 */
export interface GeneratedToken {
  token: string;
  tokenId: number;
  /** The top candidates at this position, highest probability first. */
  topCandidates: TokenProbability[];
  /** Entropy of the full distribution, in bits. High = the model was unsure. */
  entropyBits: number;
}

/**
 * Attention maps for one prompt, as served to the Attention Explorer.
 * `weights[layer][head][queryPos][keyPos]` is how much the token at
 * `queryPos` attended to the token at `keyPos`.
 */
export interface AttentionSnapshot {
  prompt: string;
  tokens: string[];
  numLayers: number;
  numHeads: number;
  weights: number[][][][];
}

/**
 * Result of the built-in gradient self-check.
 *
 * The check uses DIRECTIONAL derivatives rather than single-element ones.
 * Perturbing one weight at a time cannot work here: an individual gradient in
 * a Transformer is often around 1e-6, so the resulting change in the loss is
 * ~1e-9 - roughly 400 times smaller than the 2.4e-7 absolute resolution of a
 * float32 loss near 3.2. The measurement would be pure rounding noise.
 *
 * Perturbing the whole tensor ALONG its own gradient direction concentrates
 * the entire tensor's signal into one measurement. The expected directional
 * derivative is then the gradient's L2 norm, which is many orders of magnitude
 * above the noise floor, so the comparison is meaningful in float32.
 */
export interface GradientCheckResult {
  passed: boolean;
  checks: {
    variable: string;
    /** Number of elements in this tensor. */
    size: number;
    /** The analytic directional derivative: the L2 norm of this tensor's gradient. */
    analyticDerivative: number;
    /** The same quantity estimated as (L(w + h*v) - L(w - h*v)) / 2h. */
    numericDerivative: number;
    relativeError: number;
    /** True when the gradient was too small to measure in float32 and was skipped. */
    skipped: boolean;
    /**
     * The tolerance applied to THIS tensor. Larger for tensors with small
     * gradients, because the numeric estimate carries a fixed absolute
     * uncertainty that is relatively larger when the true value is small.
     */
    tolerance?: number;
  }[];
  /** The aggregate check across every parameter at once - the strongest signal. */
  global: {
    analyticDerivative: number;
    numericDerivative: number;
    relativeError: number;
    tolerance?: number;
  };
  maxRelativeError: number;
  /** Worst (error / that tensor's own tolerance). At or below 1 means pass. */
  maxErrorRatio?: number;
  tolerance: number;
  epsilon: number;
  /** Absolute uncertainty on each numeric derivative, from float32 rounding. */
  noiseFloor?: number;
}

/**
 * High-level status of the training engine.
 */
export type TrainingStatus = 'idle' | 'running' | 'paused' | 'completed' | 'error';

/**
 * Metadata stored for a saved model checkpoint.
 */
export interface CheckpointMetadata {
  id: string;
  name: string;
  step: number;
  createdAt: string;
  modelConfig: ModelConfig;
  trainLoss: number;
  valLoss?: number;
  filePath: string;
  fileSizeBytes: number;
  /** Total trainable parameters, read from the checkpoint header. */
  parameters?: number;
  /** True when this checkpoint had the lowest validation loss of its run. */
  isBest?: boolean;
  /** Vocabulary size the checkpoint was trained with, for compatibility checks. */
  vocabularySize?: number;
  /** True when Adam moment estimates are stored, so training can truly resume. */
  hasOptimizerState?: boolean;
}

/**
 * Saved experiment for comparative evaluation.
 */
export interface ExperimentRecord {
  id: string;
  name: string;
  description: string;
  createdAt: string;
  config: FullConfig;
  totalStepsCompleted: number;
  finalTrainLoss: number;
  finalValLoss?: number;
  trainingDurationMs: number;
  totalParameters: number;
  metricsHistory: TrainingStepMetric[];
  sampleGenerations: {
    prompt: string;
    output: string;
    temperature: number;
    topK: number;
  }[];
}

/**
 * Real-time event messages sent from the backend to the UI over WebSocket.
 */
export type WebSocketMessage =
  | { type: 'TRAINING_STATUS'; status: TrainingStatus; message?: string }
  | { type: 'TRAINING_METRIC'; metric: TrainingStepMetric }
  | { type: 'TRAINING_LOG'; level: 'info' | 'warn' | 'error'; message: string; timestamp: number }
  | { type: 'CHECKPOINT_SAVED'; checkpoint: CheckpointMetadata }
  | { type: 'SYSTEM_INFO'; info: SystemInfo }
  /** Streaming generation: one token at a time, with its distribution. */
  | {
      type: 'GENERATION_TOKEN';
      requestId: string;
      token: GeneratedToken;
      index: number;
      total: number;
    }
  | { type: 'GENERATION_DONE'; requestId: string; text: string }
  | { type: 'GENERATION_ERROR'; requestId: string; error: string };

/** Messages the browser sends to the server over the WebSocket. */
export type ClientMessage =
  | {
      type: 'GENERATE_STREAM';
      requestId: string;
      prompt: string;
      config: Partial<GenerationConfig>;
    }
  | { type: 'CANCEL_GENERATION'; requestId: string };
