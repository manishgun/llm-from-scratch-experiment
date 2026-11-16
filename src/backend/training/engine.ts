import * as tf from '@tensorflow/tfjs';
import * as path from 'path';
import * as fs from 'fs';
import { MiniGPT } from '../model/gpt';
import { DatasetManager } from '../dataset/datasetManager';
import { TextGenerator } from '../model/generation';
import { Rng } from '../core/rng';
import { learningRateAt } from './schedule';
import {
  clipGradientsByGlobalNorm,
  applyDecoupledWeightDecay,
  decayableVariables,
} from './optimizer';
import {
  FullConfig,
  TrainingStatus,
  TrainingStepMetric,
  CheckpointMetadata,
  GradientCheckResult,
} from '../../shared/types';

/**
 * ============================================================================
 * MiniGPT Lab - Training Engine & Execution Loop
 * ============================================================================
 *
 * This class coordinates the complete model training lifecycle:
 * 1. Model variable initialization from seeded random weights.
 * 2. Adam/AdamW optimizer configuration.
 * 3. The training loop, per step:
 *      - Learning rate looked up from the schedule
 *      - Batch sampling (X, Y) from the seeded sampler
 *      - Forward pass: logits = model(X)
 *      - Loss: loss = CrossEntropy(logits, Y)
 *      - Backward pass: gradients = d(loss) / d(weights)
 *      - Global-norm gradient clipping
 *      - Optimizer update
 *      - Decoupled weight decay
 * 4. Deterministic periodic validation (no gradients, no dropout).
 * 5. Best-checkpoint tracking and optional early stopping.
 * 6. Memory cleanup via tf.tidy() and explicit disposal.
 * 7. Play / Pause / Resume / Stop controls for the GUI.
 * 8. Live metric streaming via callbacks.
 */

/** How many steps of loss history to average for the smoothed train loss. */
const SMOOTHING_WINDOW = 20;

export class TrainingEngine {
  public model: MiniGPT | null = null;
  public datasetManager: DatasetManager;
  public config: FullConfig;

  private optimizer: tf.Optimizer | null = null;
  private status: TrainingStatus = 'idle';
  private shouldPause: boolean = false;
  private shouldStop: boolean = false;
  private currentStep: number = 0;
  private history: TrainingStepMetric[] = [];

  /**
   * Accumulated wall-clock training time, in milliseconds.
   *
   * Tracked as an accumulating total rather than `Date.now() - startTime`,
   * because a paused run would otherwise count the pause as training time and
   * report a nonsense steps/second once resumed.
   */
  private accumulatedMs: number = 0;
  private segmentStartTime: number = 0;

  /** Seeded generator for dropout masks, so training is reproducible. */
  private dropoutRng: Rng;

  /** Best validation loss seen so far, and the step it occurred at. */
  private bestValLoss: number = Infinity;
  private bestValStep: number = 0;
  /** Consecutive evaluations without a validation improvement. */
  private evaluationsSinceImprovement: number = 0;

  /**
   * The loss an untrained model scores: ln(vocabSize), i.e. uniform guessing.
   * Training only ever moves downward from here, which makes it a principled
   * reference point for both logging and divergence detection.
   */
  private uniformLoss: number = Infinity;

  /**
   * The loss above which a run is considered unrecoverable. Set to a generous
   * multiple of `uniformLoss` - a healthy run never exceeds the baseline at
   * all, and even a severe recoverable spike stays within ~2x of it.
   */
  private divergenceThreshold: number = Infinity;

  /**
   * Adam state read from a checkpoint, waiting to be applied.
   *
   * See `resumeFrom` for why it cannot be applied immediately: the optimizer
   * has no slots to write into until it has taken one step.
   */
  private pendingOptimizerState: tf.Tensor[] | null = null;

  // Callbacks for live UI updates
  private onMetricCallback?: (metric: TrainingStepMetric) => void;
  private onStatusCallback?: (status: TrainingStatus, message?: string) => void;
  private onLogCallback?: (level: 'info' | 'warn' | 'error', message: string) => void;
  private onCheckpointCallback?: (checkpoint: CheckpointMetadata) => void;

  /**
   * Where checkpoints are written.
   *
   * Configurable rather than hardcoded to `cwd()/checkpoints`, because the
   * test suite runs real training loops: with a fixed path, running `npm test`
   * silently overwrote the user's own `best.mgpt` with a 16-dimensional toy
   * model. Tests now point this at a temporary directory.
   */
  private readonly checkpointDir: string;

  constructor(
    config: FullConfig,
    datasetManager: DatasetManager,
    options: { checkpointDir?: string } = {}
  ) {
    this.config = config;
    this.datasetManager = datasetManager;
    this.datasetManager.setDataSeed(config.training.dataSeed ?? 1337);
    this.dropoutRng = new Rng((config.model.randomSeed ?? 42) ^ 0x5bf03635);
    this.checkpointDir = options.checkpointDir ?? path.resolve(process.cwd(), 'checkpoints');
  }

  public setCallbacks(callbacks: {
    onMetric?: (metric: TrainingStepMetric) => void;
    onStatus?: (status: TrainingStatus, message?: string) => void;
    onLog?: (level: 'info' | 'warn' | 'error', message: string) => void;
    onCheckpoint?: (checkpoint: CheckpointMetadata) => void;
  }) {
    this.onMetricCallback = callbacks.onMetric;
    this.onStatusCallback = callbacks.onStatus;
    this.onLogCallback = callbacks.onLog;
    this.onCheckpointCallback = callbacks.onCheckpoint;
  }

  public getStatus(): TrainingStatus {
    return this.status;
  }

  public getCurrentStep(): number {
    return this.currentStep;
  }

  public getHistory(): TrainingStepMetric[] {
    return [...this.history];
  }

  public getBestValidation(): { loss: number; step: number } {
    return { loss: this.bestValLoss, step: this.bestValStep };
  }

  private log(level: 'info' | 'warn' | 'error', message: string) {
    const prefix = level === 'error' ? 'ERROR' : level === 'warn' ? 'WARN' : 'INFO';
    console.log(`[${prefix}] ${message}`);
    if (this.onLogCallback) {
      this.onLogCallback(level, message);
    }
  }

  private setStatus(status: TrainingStatus, message?: string) {
    this.status = status;
    if (this.onStatusCallback) this.onStatusCallback(status, message);
  }

  /**
   * Initializes the model and optimizer.
   */
  public initialize(): void {
    const tokenizer = this.datasetManager.getTokenizer();
    const vocabSize = tokenizer.vocabSize;
    this.config.model.vocabularySize = vocabSize;

    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    if (this.optimizer) {
      this.optimizer.dispose();
      this.optimizer = null;
    }

    this.uniformLoss = Math.log(vocabSize);
    this.divergenceThreshold = this.uniformLoss * 5 + 2;

    this.log('info', `Initializing model (vocabulary: ${vocabSize} tokens)...`);
    this.model = new MiniGPT(this.config.model);

    const totalParams = this.model.countParameters();
    this.log(
      'info',
      `Architecture ready: ${this.config.model.numLayers} layers, ` +
        `${this.config.model.numHeads} heads, d_model ${this.config.model.embeddingDim}, ` +
        `${totalParams.toLocaleString('en-US')} trainable parameters` +
        `${this.model.tieWeights ? ' (embedding/head weights tied)' : ''}.`
    );

    const t = this.config.training;
    this.optimizer =
      t.optimizerType === 'sgd'
        ? tf.train.sgd(t.learningRate)
        : tf.train.adam(t.learningRate, t.adamBeta1, t.adamBeta2, t.adamEpsilon);

    // The expected starting loss is ln(vocabSize) - the loss of a model that
    // assigns every token equal probability. Logging it gives the user a
    // reference point: if step 1 is far above this, something is wrong with
    // initialization; if it is far below, there is a data leak.
    this.log(
      'info',
      `Expected initial loss for an untrained model: ${this.uniformLoss.toFixed(4)} ` +
        `(= ln ${vocabSize}, i.e. uniform guessing).`
    );
  }

  /**
   * Evaluates the model on FIXED validation batches, without gradients and
   * without dropout. See `DatasetManager.getEvalBatch` for why the batches are
   * fixed rather than random.
   */
  public async evaluateLoss(
    split: 'val' | 'test',
    numBatches: number
  ): Promise<number> {
    if (!this.model) throw new Error('Model not initialized.');

    let totalLoss = 0;
    const batchSize = this.config.training.batchSize;
    const contextLength = this.config.model.contextLength;

    for (let b = 0; b < numBatches; b++) {
      const { x, y } = this.datasetManager.getEvalBatch(
        split,
        batchSize,
        contextLength,
        b,
        numBatches
      );

      const lossScalar = tf.tidy(() => {
        // training: false disables dropout, so evaluation is deterministic.
        const logits = this.model!.forward(x, { training: false });
        return this.model!.computeLoss(logits, y);
      });

      // `await .data()` rather than `.dataSync()`: dataSync blocks the Node
      // event loop, which stalls WebSocket metric delivery and makes the UI
      // appear frozen during every evaluation.
      const value = (await lossScalar.data())[0];
      lossScalar.dispose();
      x.dispose();
      y.dispose();

      totalLoss += value;
    }

    return totalLoss / numBatches;
  }

  /**
   * Executes exactly one optimization step and returns its diagnostics.
   *
   * Split out from the loop so it can be unit tested, and so the CLI and the
   * server share identical step semantics.
   */
  private async trainStep(step: number): Promise<{ loss: number; gradNorm: number }> {
    const model = this.model!;
    const t = this.config.training;

    // 1. Learning rate from the schedule.
    const lr = learningRateAt(step, t);
    // TF.js optimizers read `learningRate` on every `applyGradients` call, so
    // assigning it here is how a schedule is implemented without rebuilding
    // the optimizer (which would discard Adam's moment estimates every step).
    (this.optimizer as unknown as { learningRate: number }).learningRate = lr;

    // 2. Sample a training batch.
    const { x, y } = this.datasetManager.getBatch('train', t.batchSize, this.config.model.contextLength);

    try {
      // 3-4. Forward + backward, differentiating ONLY this model's variables.
      //
      // This is why `tf.variableGrads` is given an explicit variable list
      // rather than letting it default. Without the list, TF.js differentiates
      // with respect to EVERY trainable variable registered in its global
      // environment - which includes the variables of any other model instance
      // still alive (for example one created by loading a checkpoint). Those
      // orphaned variables would receive gradients and be updated by Adam,
      // wasting work and silently mutating a model the user thought was frozen.
      const varList = model.getTrainableVariables();
      const { value, grads } = tf.variableGrads(() => {
        const logits = model.forward(x, { training: true, rng: this.dropoutRng });
        return model.computeLoss(logits, y);
      }, varList);

      const lossValue = (await value.data())[0];
      value.dispose();

      // 5. Clip by global norm before applying.
      const { globalNorm, clipped, wasClipped } = clipGradientsByGlobalNorm(
        grads,
        t.gradientClipNorm ?? 0
      );

      // Divergence detection.
      //
      // Checking only for NaN/Infinity is not enough. A run can reach a loss
      // of 1e13 - trillions of times the uniform-guessing baseline - while
      // every number remains a perfectly finite float32. Such a run is dead;
      // continuing it wastes the remaining steps and, worse, the final
      // checkpoint overwrites a usable earlier one.
      //
      // The baseline gives a principled threshold rather than a magic number:
      // an untrained model scores ln(vocabSize), and training only ever moves
      // DOWN from there. A loss several times that value cannot be recovered
      // from, so it is treated as divergence.
      if (!Number.isFinite(lossValue) || !Number.isFinite(globalNorm)) {
        for (const g of Object.values(clipped)) g.dispose();
        throw new Error(
          `Training diverged at step ${step + 1}: loss=${lossValue}, gradient norm=${globalNorm}. ` +
            `The learning rate (${lr.toExponential(2)}) is almost certainly too high. ` +
            `Reduce it by 10x, or set a gradient clip norm of 1.0.`
        );
      }

      if (lossValue > this.divergenceThreshold) {
        for (const g of Object.values(clipped)) g.dispose();
        throw new Error(
          `Training diverged at step ${step + 1}: loss reached ${lossValue.toExponential(3)}, ` +
            `far above the ${this.uniformLoss.toFixed(2)} an untrained model scores ` +
            `(= ln ${this.config.model.vocabularySize}). Loss should only ever fall below that value. ` +
            `The learning rate (${lr.toExponential(2)}) is too high` +
            `${(t.gradientClipNorm ?? 0) <= 0 ? ' and gradient clipping is disabled' : ''}. ` +
            `Reduce the learning rate by 10x, or set a gradient clip norm of 1.0.`
        );
      }

      if (wasClipped && step % 50 === 0) {
        this.log(
          'info',
          `Gradient norm ${globalNorm.toFixed(2)} exceeded the clip threshold ` +
            `${t.gradientClipNorm} and was scaled down (this is normal early in training).`
        );
      }

      // 6. Apply the update.
      this.optimizer!.applyGradients(clipped);
      for (const g of Object.values(clipped)) g.dispose();

      // Adam's slots exist only after that first applyGradients, so a
      // checkpoint's moment estimates can finally be written in now.
      if (this.pendingOptimizerState) {
        await this.applyPendingOptimizerState();
      }

      // 7. Decoupled weight decay (AdamW), applied directly to the weights
      //    and scaled by the current learning rate. Biases and LayerNorm
      //    parameters are excluded - see optimizer.ts.
      if ((t.weightDecay ?? 0) > 0) {
        applyDecoupledWeightDecay(
          decayableVariables(varList, model.getNoDecayVariables()),
          t.weightDecay,
          lr
        );
      }

      return { loss: lossValue, gradNorm: globalNorm };
    } finally {
      x.dispose();
      y.dispose();
    }
  }

  /**
   * Start or resume the training loop.
   */
  public async start(): Promise<void> {
    if (this.status === 'running') {
      this.log('warn', 'Training is already running.');
      return;
    }

    if (!this.model) {
      this.initialize();
    }

    this.shouldPause = false;
    this.shouldStop = false;
    this.setStatus('running');
    this.segmentStartTime = Date.now();

    const t = this.config.training;
    this.log(
      'info',
      `Training ${t.trainingSteps} steps | batch ${t.batchSize} x ${this.config.model.contextLength} tokens | ` +
        `lr ${t.learningRate} (${t.lrSchedule}, ${t.warmupSteps} warmup) | ` +
        `clip ${t.gradientClipNorm} | weight decay ${t.weightDecay}`
    );

    try {
      while (this.currentStep < t.trainingSteps) {
        if (this.shouldStop) {
          this.accumulatedMs += Date.now() - this.segmentStartTime;
          this.setStatus('idle');
          this.log('info', `Training stopped at step ${this.currentStep}.`);
          return;
        }

        if (this.shouldPause) {
          this.accumulatedMs += Date.now() - this.segmentStartTime;
          this.setStatus('paused');
          this.log('info', `Training paused at step ${this.currentStep}.`);
          return;
        }

        const { loss: trainLossVal, gradNorm } = await this.trainStep(this.currentStep);
        const learningRate = learningRateAt(this.currentStep, t);
        this.currentStep++;

        // --- Periodic evaluation ------------------------------------------
        const isEvalStep =
          this.currentStep % t.evaluationInterval === 0 ||
          this.currentStep === t.trainingSteps;

        let valLossVal: number | undefined;
        let sampleText: string | undefined;

        if (isEvalStep) {
          valLossVal = await this.evaluateLoss('val', t.validationBatchCount);

          // Track the best validation loss. The best model is almost never the
          // last one: once the model starts overfitting, validation loss rises
          // while training loss keeps falling, so the final checkpoint is
          // strictly worse than the one from the dip.
          if (valLossVal < this.bestValLoss) {
            this.bestValLoss = valLossVal;
            this.bestValStep = this.currentStep;
            this.evaluationsSinceImprovement = 0;
            await this.saveCheckpoint('best', trainLossVal, valLossVal);
          } else {
            this.evaluationsSinceImprovement++;
          }

          // A short sample, to make progress visible as text rather than only
          // as a number.
          const generator = new TextGenerator(this.model!, this.datasetManager.getTokenizer());
          const result = await generator.generate(this.config.generation.samplePrompt || '\n', {
            ...this.config.generation,
            maxTokens: 48,
            seed: 7, // fixed, so successive samples differ only because the MODEL changed
          });
          sampleText = result.text;

          this.log(
            'info',
            `Step ${this.currentStep}/${t.trainingSteps} | train ${trainLossVal.toFixed(4)} ` +
              `| val ${valLossVal.toFixed(4)} | ppl ${Math.exp(valLossVal).toFixed(1)} ` +
              `| lr ${learningRate.toExponential(2)} | grad ${gradNorm.toFixed(2)}`
          );

          // --- Early stopping --------------------------------------------
          if (
            t.earlyStoppingPatience > 0 &&
            this.evaluationsSinceImprovement >= t.earlyStoppingPatience
          ) {
            this.accumulatedMs += Date.now() - this.segmentStartTime;
            this.log(
              'warn',
              `Early stopping: validation loss has not improved for ` +
                `${this.evaluationsSinceImprovement} evaluations. ` +
                `Best was ${this.bestValLoss.toFixed(4)} at step ${this.bestValStep}; ` +
                `that checkpoint is saved as "best".`
            );
            this.setStatus('completed');
            return;
          }
        }

        // --- Periodic checkpointing ---------------------------------------
        if (
          t.checkpointInterval > 0 &&
          (this.currentStep % t.checkpointInterval === 0 ||
            this.currentStep === t.trainingSteps)
        ) {
          await this.saveCheckpoint(`step_${this.currentStep}`, trainLossVal, valLossVal);
        }

        // --- Metrics -------------------------------------------------------
        const elapsedMs = this.accumulatedMs + (Date.now() - this.segmentStartTime);
        const stepsPerSec = this.currentStep / (elapsedMs / 1000);
        const remainingSteps = t.trainingSteps - this.currentStep;
        const estimatedRemainingMs = stepsPerSec > 0 ? (remainingSteps / stepsPerSec) * 1000 : 0;

        const metric: TrainingStepMetric = {
          step: this.currentStep,
          totalSteps: t.trainingSteps,
          trainLoss: trainLossVal,
          valLoss: valLossVal,
          learningRate,
          gradNorm,
          perplexity: Math.exp(trainLossVal),
          valPerplexity: valLossVal !== undefined ? Math.exp(valLossVal) : undefined,
          bitsPerToken: trainLossVal / Math.LN2,
          elapsedMs,
          estimatedRemainingMs,
          stepsPerSecond: Number(stepsPerSec.toFixed(2)),
          tokensPerSecond: Math.round(
            stepsPerSec * t.batchSize * this.config.model.contextLength
          ),
          memoryUsageMb: Number((process.memoryUsage().heapUsed / 1024 / 1024).toFixed(1)),
          tensorCount: tf.memory().numTensors,
          generatedSample: sampleText,
          timestamp: Date.now(),
        };

        this.history.push(metric);
        if (this.onMetricCallback) {
          this.onMetricCallback(metric);
        }

        // Yield so the Node event loop can serve API requests and WebSockets.
        await new Promise((resolve) => setImmediate(resolve));
      }

      this.accumulatedMs += Date.now() - this.segmentStartTime;
      this.setStatus('completed');
      this.log(
        'info',
        `Training complete. Best validation loss ${this.bestValLoss.toFixed(4)} ` +
          `at step ${this.bestValStep} (perplexity ${Math.exp(this.bestValLoss).toFixed(1)}).`
      );
    } catch (err: any) {
      this.accumulatedMs += Date.now() - this.segmentStartTime;
      this.log('error', err.message);
      this.setStatus('error', err.message);
    }
  }

  /**
   * Writes a checkpoint including the tokenizer vocabulary and the Adam
   * optimizer state, so the run can be resumed exactly.
   */
  private async saveCheckpoint(
    label: string,
    trainLoss: number,
    valLoss?: number
  ): Promise<void> {
    if (!this.model) return;

    const checkpointPath = path.resolve(this.checkpointDir, `${label}.mgpt`);
    const tokenizer = this.datasetManager.getTokenizer();

    // Adam's moment estimates. Restoring them is the difference between
    // resuming and restarting - without them the first resumed step has no
    // momentum and no per-parameter scale, producing a visible loss bump.
    //
    // CRITICAL: `getWeights()` hands back the optimizer's LIVE internal
    // tensors, not copies. They must be cloned before use and the clones -
    // never the originals - disposed afterwards. Disposing the originals
    // destroys Adam's state mid-run, and the very next step fails inside
    // TensorFlow with an unrelated-looking error.
    let optimizerState: tf.Tensor[] | undefined;
    try {
      const weights = await this.optimizer!.getWeights();
      optimizerState = weights.map((w) => tf.keep(w.tensor.clone()));
    } catch {
      // SGD has no state; `getWeights` may legitimately be unsupported.
      optimizerState = undefined;
    }

    await this.model.saveCheckpoint(checkpointPath, {
      step: this.currentStep,
      tokenizer: { type: this.datasetManager.getTokenizerType(), vocab: tokenizer.getVocabulary() },
      metrics: { trainLoss, valLoss },
      optimizerState,
    });

    // Dispose the CLONES we made above. The optimizer's own tensors are
    // untouched and remain valid for the next step.
    if (optimizerState) {
      for (const clone of optimizerState) clone.dispose();
    }

    const stat = fs.statSync(checkpointPath);
    const meta: CheckpointMetadata = {
      id: label,
      name: label === 'best' ? `Best (step ${this.currentStep})` : `Step ${this.currentStep}`,
      step: this.currentStep,
      createdAt: new Date().toISOString(),
      modelConfig: this.config.model,
      trainLoss,
      valLoss,
      filePath: checkpointPath,
      fileSizeBytes: stat.size,
      parameters: this.model.countParameters(),
      isBest: label === 'best',
      vocabularySize: this.config.model.vocabularySize,
      hasOptimizerState: optimizerState !== undefined,
    };

    if (this.onCheckpointCallback) {
      this.onCheckpointCallback(meta);
    }
  }

  /**
   * Writes the stashed Adam state into the optimizer, now that its slots are
   * allocated. Runs exactly once per resume.
   */
  private async applyPendingOptimizerState(): Promise<void> {
    const pending = this.pendingOptimizerState;
    this.pendingOptimizerState = null;
    if (!pending || !this.optimizer) return;

    try {
      // TF.js matches optimizer weights positionally against the variables
      // they track, so this only works because `getTrainableVariables` builds
      // its list in a stable order.
      const current = await this.optimizer.getWeights();
      if (current.length !== pending.length) {
        this.log(
          'warn',
          `Optimizer state in the checkpoint has ${pending.length} tensors but this ` +
            `optimizer expects ${current.length}; continuing with fresh moment estimates.`
        );
        for (const tensorItem of pending) tensorItem.dispose();
        return;
      }

      await this.optimizer.setWeights(
        pending.map((tensor, i) => ({ name: current[i].name, tensor }))
      );
      this.log('info', 'Restored Adam moment estimates - the run resumes with momentum intact.');
    } catch (err: any) {
      this.log('warn', `Could not restore optimizer state (${err.message}); using fresh estimates.`);
      for (const tensorItem of pending) tensorItem.dispose();
    }
  }

  /**
   * Restores weights, step counter and optimizer state from a checkpoint,
   * so `start()` continues where the saved run left off.
   */
  public async resumeFrom(checkpointPath: string): Promise<void> {
    if (!this.model) this.initialize();

    const header = await this.model!.loadCheckpoint(checkpointPath);
    this.currentStep = header.step ?? 0;

    /**
     * Adam's moment estimates, deferred until after the first step.
     *
     * Adam allocates its per-variable slots LAZILY, on the first call to
     * `applyGradients`. Immediately after construction `getWeights()` returns
     * an empty array, so `setWeights` here would have nothing to match against
     * and the restore silently degrades to "fresh momentum".
     *
     * The state is therefore stashed and applied at the end of the first
     * training step, once the slots exist. That first step runs without
     * momentum - one step out of a resumed run - and every step after it has
     * the full restored state.
     */
    const optimizerState = await MiniGPT.readOptimizerState(checkpointPath);
    if (this.pendingOptimizerState) {
      for (const tensorItem of this.pendingOptimizerState) tensorItem.dispose();
    }
    this.pendingOptimizerState = optimizerState.length > 0 ? optimizerState : null;

    this.log(
      'info',
      `Resumed from ${path.basename(checkpointPath)} at step ${this.currentStep}.`
    );
  }

  /**
   * ------------------------------------------------------------------------
   * GRADIENT SELF-CHECK (finite differences)
   * ------------------------------------------------------------------------
   *
   * This is the only test that actually PROVES backpropagation is correct.
   *
   * Asserting that "weights changed after a step" passes even if the gradients
   * are transposed, sign-flipped, or scaled by a constant - all of which are
   * real, common bugs in a hand-written Transformer, and all of which produce a
   * model that trains slowly or not at all with no error message.
   *
   * The check compares the analytic gradient from autodiff against a numerical
   * estimate from the definition of a derivative:
   *
   *     df/dx ~= ( f(x + eps) - f(x - eps) ) / (2 * eps)
   *
   * The CENTRAL difference (both directions) is used rather than the forward
   * difference, because its error is O(eps^2) instead of O(eps) - which matters
   * when eps must stay large enough to survive float32 rounding.
   *
   * A relative error below ~1e-2 in float32 indicates the gradient is right.
   * (float64 would allow 1e-7, but TF.js tensors are float32, so the numerical
   * estimate itself carries that much noise.)
   *
   * @param samplesPerTensor How many random weights to probe per tensor
   * @param epsilon          The perturbation size
   */
  public async checkGradients(
    epsilon: number = 2e-3,
    tolerance: number = 5e-2
  ): Promise<GradientCheckResult> {
    if (!this.model) this.initialize();
    const model = this.model!;

    // A small fixed batch: the check is about correctness, not scale.
    const { x, y } = this.datasetManager.getEvalBatch(
      'val',
      2,
      Math.min(16, this.config.model.contextLength),
      0,
      1
    );

    // Dropout must be OFF. A stochastic loss has no well-defined derivative to
    // compare against, and the two perturbed evaluations would otherwise use
    // different masks - the measured difference would be dominated by which
    // neurons happened to be dropped.
    const lossFn = () =>
      tf.tidy(() => {
        const logits = model.forward(x, { training: false });
        return model.computeLoss(logits, y);
      });

    const varList = model.getTrainableVariables();
    const { value, grads } = tf.variableGrads(lossFn, varList);
    value.dispose();

    /**
     * Measures the directional derivative of the loss along `directions`,
     * where `directions[i]` is a unit-norm perturbation applied to
     * `variables[i]`.
     *
     * Returns (L(w + h*v) - L(w - h*v)) / 2h, the CENTRAL difference. Central
     * rather than forward because its truncation error is O(h^2) instead of
     * O(h), which matters when h must stay large enough to survive float32
     * rounding.
     */
    const measureDirectional = async (
      variables: tf.Variable[],
      directions: Float32Array[],
      originals: Float32Array[]
    ): Promise<number> => {
      const shift = (sign: number) => {
        for (let i = 0; i < variables.length; i++) {
          const perturbed = new Float32Array(originals[i].length);
          for (let j = 0; j < perturbed.length; j++) {
            perturbed[j] = originals[i][j] + sign * epsilon * directions[i][j];
          }
          tf.tidy(() => {
            variables[i].assign(tf.tensor(perturbed, variables[i].shape, 'float32'));
          });
        }
      };

      shift(+1);
      const plusTensor = lossFn();
      const lossPlus = (await plusTensor.data())[0];
      plusTensor.dispose();

      shift(-1);
      const minusTensor = lossFn();
      const lossMinus = (await minusTensor.data())[0];
      minusTensor.dispose();

      // Always restore, so one measurement cannot contaminate the next.
      for (let i = 0; i < variables.length; i++) {
        tf.tidy(() => {
          variables[i].assign(tf.tensor(originals[i], variables[i].shape, 'float32'));
        });
      }

      return (lossPlus - lossMinus) / (2 * epsilon);
    };

    // Cache every tensor's current values and its gradient.
    const gradientData = new Map<string, Float32Array>();
    const originalData = new Map<string, Float32Array>();
    for (const variable of varList) {
      const grad = grads[variable.name];
      if (!grad) continue;
      gradientData.set(variable.name, (await grad.data()) as Float32Array);
      originalData.set(variable.name, ((await variable.data()) as Float32Array).slice());
      grad.dispose();
    }

    /**
     * The noise floor on a numeric derivative.
     *
     * float32 holds ~24 bits of mantissa, so a loss near L can only be
     * represented to about L * 2^-24. Dividing that uncertainty by 2h gives
     * the smallest derivative this method can distinguish from rounding.
     * A gradient below a few multiples of it is not wrong - it is simply
     * unmeasurable, so it is reported as skipped rather than failed.
     */
    const probeTensor = lossFn();
    const baseLoss = Math.abs((await probeTensor.data())[0]);
    probeTensor.dispose();
    const lossResolution = Math.max(baseLoss, 1) * Math.pow(2, -24);
    /** Absolute uncertainty on any numeric derivative measured this way. */
    const noiseFloorAbsolute = lossResolution / (2 * epsilon);
    /** Below this, a gradient is unmeasurable rather than wrong. */
    const noiseFloor = noiseFloorAbsolute * 10;

    /**
     * The tolerance that applies to a tensor whose gradient has norm `g`.
     *
     * A flat tolerance is wrong here, and the reason is measurement physics
     * rather than anything about the model. The numeric derivative is a
     * difference of two float32 losses divided by 2h, so it carries an
     * absolute uncertainty of roughly `noiseFloorAbsolute` regardless of how
     * large the true derivative is. Expressed as a RELATIVE error, that fixed
     * uncertainty is therefore large for a small gradient and negligible for a
     * big one:
     *
     *     relative noise  ~=  noiseFloorAbsolute / g
     *
     * Empirically this is exactly what the data shows: tensors with gradient
     * norms around 1.0 agree to 1e-4, while attention query biases with norms
     * around 2e-3 agree only to ~5e-2 - a 500x difference in gradient scale
     * producing a ~500x difference in relative error. Judging both by the same
     * threshold would flag correct gradients as broken.
     *
     * So the tolerance is the base tolerance plus three times the predicted
     * noise. A genuinely wrong gradient (a sign flip, a transpose, a missing
     * term) is wrong by order 100% and is caught regardless.
     */
    const toleranceFor = (gradNorm: number): number =>
      tolerance + 3 * (noiseFloorAbsolute / Math.max(gradNorm, 1e-12));

    const checks: GradientCheckResult['checks'] = [];
    let maxRelativeError = 0;
    /** Worst (relativeError / itsOwnTolerance); <= 1 means everything passed. */
    let maxErrorRatio = 0;

    // --- Per-tensor directional checks -----------------------------------
    // Each tensor is perturbed along its OWN gradient direction. The expected
    // directional derivative is then exactly the gradient's L2 norm, because
    // g . (g/|g|) = |g|.
    for (const variable of varList) {
      const gradValues = gradientData.get(variable.name);
      const original = originalData.get(variable.name);
      if (!gradValues || !original) continue;

      let sumSquares = 0;
      for (let i = 0; i < gradValues.length; i++) sumSquares += gradValues[i] * gradValues[i];
      const gradNorm = Math.sqrt(sumSquares);

      if (gradNorm <= noiseFloor) {
        checks.push({
          variable: variable.name,
          size: gradValues.length,
          analyticDerivative: gradNorm,
          numericDerivative: 0,
          relativeError: 0,
          skipped: true,
        });
        continue;
      }

      const direction = new Float32Array(gradValues.length);
      for (let i = 0; i < gradValues.length; i++) direction[i] = gradValues[i] / gradNorm;

      const numeric = await measureDirectional([variable], [direction], [original]);
      const relativeError =
        Math.abs(gradNorm - numeric) / Math.max(Math.abs(gradNorm), Math.abs(numeric), 1e-12);

      checks.push({
        variable: variable.name,
        size: gradValues.length,
        analyticDerivative: gradNorm,
        numericDerivative: numeric,
        relativeError,
        skipped: false,
        tolerance: toleranceFor(gradNorm),
      });

      // Compare each tensor against ITS OWN tolerance, then track the result
      // as a ratio so a single `maxRelativeError` number stays meaningful
      // across tensors with very different gradient scales.
      maxErrorRatio = Math.max(maxErrorRatio, relativeError / toleranceFor(gradNorm));
      maxRelativeError = Math.max(maxRelativeError, relativeError);
    }

    // --- Global check ------------------------------------------------------
    // Every parameter perturbed simultaneously along the full gradient
    // direction. This is the strongest single assertion available: it verifies
    // the entire backward pass end to end, with the largest possible signal.
    const checkedVars = varList.filter((v) => gradientData.has(v.name));
    let globalSumSquares = 0;
    for (const variable of checkedVars) {
      const g = gradientData.get(variable.name)!;
      for (let i = 0; i < g.length; i++) globalSumSquares += g[i] * g[i];
    }
    const globalNorm = Math.sqrt(globalSumSquares);

    const globalDirections = checkedVars.map((variable) => {
      const g = gradientData.get(variable.name)!;
      const direction = new Float32Array(g.length);
      for (let i = 0; i < g.length; i++) direction[i] = g[i] / globalNorm;
      return direction;
    });
    const globalOriginals = checkedVars.map((v) => originalData.get(v.name)!);

    const globalNumeric = await measureDirectional(
      checkedVars,
      globalDirections,
      globalOriginals
    );
    const globalRelativeError =
      Math.abs(globalNorm - globalNumeric) /
      Math.max(Math.abs(globalNorm), Math.abs(globalNumeric), 1e-12);
    maxRelativeError = Math.max(maxRelativeError, globalRelativeError);
    maxErrorRatio = Math.max(maxErrorRatio, globalRelativeError / toleranceFor(globalNorm));

    x.dispose();
    y.dispose();

    return {
      // Every tensor judged against its own noise-aware tolerance.
      passed: maxErrorRatio <= 1,
      checks,
      global: {
        analyticDerivative: globalNorm,
        numericDerivative: globalNumeric,
        relativeError: globalRelativeError,
        tolerance: toleranceFor(globalNorm),
      },
      maxRelativeError,
      maxErrorRatio,
      tolerance,
      epsilon,
      noiseFloor: noiseFloorAbsolute,
    };
  }

  /** The smoothed training loss, for a readable chart line. */
  public getSmoothedLoss(): number | undefined {
    if (this.history.length === 0) return undefined;
    const window = this.history.slice(-SMOOTHING_WINDOW);
    return window.reduce((acc, m) => acc + m.trainLoss, 0) / window.length;
  }

  public pause(): void {
    if (this.status === 'running') {
      this.shouldPause = true;
    }
  }

  public stop(): void {
    this.shouldStop = true;
  }

  /**
   * Resets to a pristine state: step 0, fresh weights, fresh optimizer.
   *
   * Re-initializing the MODEL as well as the counters matters. Previously
   * `reset` cleared the step count but kept the trained weights, so "reset
   * then train" silently continued from a partially trained model while
   * reporting step 1 - making every comparison against a fresh run invalid.
   */
  public reset(): void {
    this.stop();
    this.currentStep = 0;
    this.history = [];
    this.accumulatedMs = 0;
    this.bestValLoss = Infinity;
    this.bestValStep = 0;
    this.evaluationsSinceImprovement = 0;
    this.dropoutRng = new Rng((this.config.model.randomSeed ?? 42) ^ 0x5bf03635);
    this.datasetManager.setDataSeed(this.config.training.dataSeed ?? 1337);
    if (this.pendingOptimizerState) {
      for (const tensorItem of this.pendingOptimizerState) tensorItem.dispose();
      this.pendingOptimizerState = null;
    }

    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    if (this.optimizer) {
      this.optimizer.dispose();
      this.optimizer = null;
    }

    this.setStatus('idle');
  }

  /** Releases all tensors held by this engine. */
  public dispose(): void {
    if (this.pendingOptimizerState) {
      for (const tensorItem of this.pendingOptimizerState) tensorItem.dispose();
      this.pendingOptimizerState = null;
    }
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    if (this.optimizer) {
      this.optimizer.dispose();
      this.optimizer = null;
    }
  }
}
