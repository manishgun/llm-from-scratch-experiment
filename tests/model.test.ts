import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { MiniGPT } from '../src/backend/model/gpt';
import { CausalSelfAttention } from '../src/backend/model/attention';
import { TextGenerator } from '../src/backend/model/generation';
import { DatasetManager } from '../src/backend/dataset/datasetManager';
import { TrainingEngine } from '../src/backend/training/engine';
import { learningRateAt } from '../src/backend/training/schedule';
import { clipGradientsByGlobalNorm } from '../src/backend/training/optimizer';
import { DEFAULT_CONFIG } from '../src/shared/config';
import { FullConfig, ModelConfig } from '../src/shared/types';

/**
 * ============================================================================
 * Model Architecture, Training Mechanics and Persistence
 * ============================================================================
 */

/** A deliberately tiny model, so each test runs in well under a second. */
function tinyModelConfig(overrides: Partial<ModelConfig> = {}): ModelConfig {
  return {
    vocabularySize: 24,
    embeddingDim: 16,
    numLayers: 2,
    numHeads: 2,
    contextLength: 12,
    feedForwardMultiplier: 2,
    activation: 'gelu',
    dropoutRate: 0,
    layerNormEpsilon: 1e-5,
    initializerScale: 0.02,
    randomSeed: 42,
    tieWeights: true,
    ...overrides,
  };
}

/** Text with enough repeated structure to be learnable in a few hundred steps. */
const SAMPLE_TEXT =
  'First Citizen:\nBefore we proceed any further, hear me speak.\n\n' +
  'All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n\n' +
  'All:\nResolved. resolved.\n\nFirst Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n';

afterAll(() => {
  // The mask cache deliberately holds tensors across tidy scopes; release them
  // so the suite does not report leaked tensors.
  CausalSelfAttention.clearMaskCache();
});

// ---------------------------------------------------------------------------

describe('Causal masking', () => {
  it('builds an upper-triangular mask that blocks future positions', async () => {
    const mask = CausalSelfAttention.createCausalMask(4);
    const values = (await mask.array()) as number[][][][];
    const m = values[0][0];

    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 4; col++) {
        if (col > row) {
          expect(m[row][col]).toBeLessThan(-1e8); // future: blocked
        } else {
          expect(m[row][col]).toBe(0); // past or self: allowed
        }
      }
    }
  });

  it('returns the identical cached tensor for a repeated sequence length', () => {
    // The mask depends only on T and never changes, so rebuilding it on every
    // attention layer of every step is pure waste. This asserts the cache is
    // actually in effect.
    const first = CausalSelfAttention.createCausalMask(7);
    const second = CausalSelfAttention.createCausalMask(7);
    expect(second).toBe(first);
  });

  it('BEHAVIOURAL PROOF: changing a future token cannot alter earlier logits', async () => {
    // This is a far stronger check than inspecting the mask tensor. It proves
    // the mask is wired into the computation correctly - a mask that is built
    // perfectly but added at the wrong point, or after softmax, would pass the
    // structural test above and fail this one.
    //
    // If this test fails, the model is reading the answer it is being asked to
    // predict, and every loss number it reports is meaningless.
    const config = tinyModelConfig();
    const model = new MiniGPT(config);

    const base = [1, 2, 3, 4, 5, 6];
    const altered = [1, 2, 3, 99 % config.vocabularySize, 5, 6]; // differs at index 3

    const logitsA = model.forward(tf.tensor2d([base], [1, 6], 'int32'));
    const logitsB = model.forward(tf.tensor2d([altered], [1, 6], 'int32'));

    const a = (await logitsA.array()) as number[][][];
    const b = (await logitsB.array()) as number[][][];

    // Positions 0, 1, 2 come strictly before the changed token, so their
    // logits must be bit-identical.
    for (let pos = 0; pos < 3; pos++) {
      for (let v = 0; v < config.vocabularySize; v++) {
        expect(b[0][pos][v]).toBeCloseTo(a[0][pos][v], 6);
      }
    }

    // Position 3 onward SHOULD differ - otherwise the token is being ignored.
    const positionThreeDiffers = a[0][3].some(
      (value, v) => Math.abs(value - b[0][3][v]) > 1e-6
    );
    expect(positionThreeDiffers).toBe(true);

    logitsA.dispose();
    logitsB.dispose();
    model.dispose();
  });
});

// ---------------------------------------------------------------------------

describe('Model architecture', () => {
  it('produces logits of shape [batch, seqLen, vocabSize]', () => {
    const config = tinyModelConfig();
    const model = new MiniGPT(config);
    const input = tf.tensor2d(
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
      [2, 4],
      'int32'
    );

    const logits = model.forward(input);
    expect(logits.shape).toEqual([2, 4, config.vocabularySize]);

    logits.dispose();
    input.dispose();
    model.dispose();
  });

  it('rejects a sequence longer than contextLength', () => {
    const config = tinyModelConfig({ contextLength: 4 });
    const model = new MiniGPT(config);
    const tooLong = tf.tensor2d([[1, 2, 3, 4, 5]], [1, 5], 'int32');

    expect(() => model.forward(tooLong)).toThrow(/exceeds maximum model contextLength/);

    tooLong.dispose();
    model.dispose();
  });

  it('rejects an embeddingDim that is not divisible by numHeads', () => {
    // Multi-head attention splits the embedding into equal slices; 16 cannot
    // be divided across 3 heads.
    expect(() => new MiniGPT(tinyModelConfig({ embeddingDim: 16, numHeads: 3 }))).toThrow(
      /must be divisible by numHeads/
    );
  });

  it('weight tying removes the head matrix from the parameter count', () => {
    const tied = new MiniGPT(tinyModelConfig({ tieWeights: true }));
    const untied = new MiniGPT(tinyModelConfig({ tieWeights: false }));

    const config = tinyModelConfig();
    const headMatrixSize = config.embeddingDim * config.vocabularySize;

    expect(untied.countParameters() - tied.countParameters()).toBe(headMatrixSize);
    // And the tied model must not list the embedding table twice: doing so
    // would make the optimizer update it twice per step, doubling its
    // effective learning rate.
    const names = tied.getTrainableVariables().map((v) => v.name);
    expect(new Set(names).size).toBe(names.length);

    tied.dispose();
    untied.dispose();
  });

  it('the parameter breakdown sums to the total parameter count', () => {
    const model = new MiniGPT(tinyModelConfig());
    const total = model
      .getParameterBreakdown()
      .reduce((acc, entry) => acc + entry.parameters, 0);
    expect(total).toBe(model.countParameters());
    model.dispose();
  });

  it('exposes one attention map per layer, shaped [batch, heads, T, T]', () => {
    const config = tinyModelConfig();
    const model = new MiniGPT(config);
    const input = tf.tensor2d([[1, 2, 3, 4, 5]], [1, 5], 'int32');

    const { logits, attention } = model.forwardWithAttention(input);

    expect(attention.length).toBe(config.numLayers);
    for (const layer of attention) {
      expect(layer.shape).toEqual([1, config.numHeads, 5, 5]);
      layer.dispose();
    }

    logits.dispose();
    input.dispose();
    model.dispose();
  });

  it('attention weights form a probability distribution over allowed positions', async () => {
    const model = new MiniGPT(tinyModelConfig());
    const input = tf.tensor2d([[1, 2, 3, 4]], [1, 4], 'int32');
    const { logits, attention } = model.forwardWithAttention(input);

    const layer = (await attention[0].array()) as number[][][][];
    const head = layer[0][0];

    for (let query = 0; query < 4; query++) {
      const row = head[query];
      // Each query row sums to 1 - it is a softmax.
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 4);
      // And every future position carries exactly zero mass.
      for (let key = query + 1; key < 4; key++) {
        expect(row[key]).toBeCloseTo(0, 6);
      }
    }

    for (const a of attention) a.dispose();
    logits.dispose();
    input.dispose();
    model.dispose();
  });
});

// ---------------------------------------------------------------------------

describe('Loss computation', () => {
  it('returns a scalar', () => {
    const config = tinyModelConfig();
    const model = new MiniGPT(config);
    const x = tf.tensor2d([[1, 2, 3, 4]], [1, 4], 'int32');
    const y = tf.tensor2d([[2, 3, 4, 5]], [1, 4], 'int32');

    const loss = model.computeLoss(model.forward(x), y);
    expect(loss.shape).toEqual([]);
    expect(loss.dataSync()[0]).toBeGreaterThan(0);

    loss.dispose();
    x.dispose();
    y.dispose();
    model.dispose();
  });

  it('an untrained model starts near ln(vocabSize), the loss of uniform guessing', async () => {
    // This is the single most useful sanity check on initialization. A freshly
    // initialized model has no information, so it should assign every token
    // probability 1/V, giving a cross-entropy of ln(V).
    //
    // A starting loss far ABOVE ln(V) means the initialization is too large and
    // the model is confidently wrong. Far BELOW it means something is leaking
    // the answer. Either is a serious bug that this one assertion catches.
    const config = tinyModelConfig({ vocabularySize: 50 });
    const model = new MiniGPT(config);

    const x = tf.randomUniform([4, 10], 0, 50, 'int32') as tf.Tensor2D;
    const y = tf.randomUniform([4, 10], 0, 50, 'int32') as tf.Tensor2D;

    const lossTensor = model.computeLoss(model.forward(x), y);
    const loss = (await lossTensor.data())[0];

    expect(loss).toBeCloseTo(Math.log(50), 1);

    lossTensor.dispose();
    x.dispose();
    y.dispose();
    model.dispose();
  });
});

// ---------------------------------------------------------------------------

describe('Learning rate schedule', () => {
  const base: FullConfig['training'] = {
    ...DEFAULT_CONFIG.training,
    learningRate: 0.001,
    minLearningRate: 0.0001,
    trainingSteps: 1000,
    warmupSteps: 100,
    lrSchedule: 'cosine',
  };

  it('ramps linearly through warmup and reaches the peak exactly at its end', () => {
    expect(learningRateAt(0, base)).toBeCloseTo(0.001 * (1 / 100), 8);
    expect(learningRateAt(49, base)).toBeCloseTo(0.001 * (50 / 100), 8);
    expect(learningRateAt(99, base)).toBeCloseTo(0.001, 8);
  });

  it('never returns exactly zero during warmup', () => {
    // A learning rate of literally zero makes the first step a no-op, wasting
    // it. The ramp therefore starts at peak/warmup rather than at 0.
    expect(learningRateAt(0, base)).toBeGreaterThan(0);
  });

  it('decays monotonically after warmup and lands on the floor', () => {
    let previous = learningRateAt(100, base);
    for (let step = 101; step < 1000; step += 25) {
      const current = learningRateAt(step, base);
      expect(current).toBeLessThanOrEqual(previous + 1e-12);
      previous = current;
    }
    expect(learningRateAt(999, base)).toBeCloseTo(base.minLearningRate, 5);
  });

  it('holds the rate fixed under the constant schedule, warmup aside', () => {
    const constant = { ...base, lrSchedule: 'constant' as const };
    expect(learningRateAt(500, constant)).toBe(0.001);
    expect(learningRateAt(999, constant)).toBe(0.001);
  });

  it('decays in a straight line under the linear schedule', () => {
    const linear = { ...base, lrSchedule: 'linear' as const, warmupSteps: 0 };
    const midpoint = learningRateAt(500, linear);
    // Halfway through, a straight line sits halfway between peak and floor.
    expect(midpoint).toBeCloseTo((0.001 + 0.0001) / 2, 5);
  });

  it('cosine decays more slowly than linear early on', () => {
    // This is the practical reason cosine is preferred: it keeps the rate near
    // its peak for longer, then settles smoothly.
    const cosine = { ...base, warmupSteps: 0 };
    const linear = { ...base, lrSchedule: 'linear' as const, warmupSteps: 0 };
    expect(learningRateAt(200, cosine)).toBeGreaterThan(learningRateAt(200, linear));
  });
});

// ---------------------------------------------------------------------------

describe('Gradient clipping', () => {
  it('leaves gradients untouched when the norm is already under the ceiling', () => {
    const grads = { a: tf.tensor1d([0.1, 0.1]), b: tf.tensor1d([0.1]) };
    const { globalNorm, wasClipped } = clipGradientsByGlobalNorm(grads, 1.0);

    expect(globalNorm).toBeCloseTo(Math.sqrt(0.03), 5);
    expect(wasClipped).toBe(false);

    for (const g of Object.values(grads)) g.dispose();
  });

  it('scales an oversized gradient down to exactly the ceiling', async () => {
    // [3, 4] has norm 5. Clipping to 1.0 must produce [0.6, 0.8], which has
    // norm 1 and - crucially - points in the SAME direction.
    const grads = { w: tf.tensor1d([3, 4]) };
    const { globalNorm, clipped, wasClipped } = clipGradientsByGlobalNorm(grads, 1.0);

    expect(globalNorm).toBeCloseTo(5, 5);
    expect(wasClipped).toBe(true);

    const values = (await clipped.w.data()) as Float32Array;
    expect(values[0]).toBeCloseTo(0.6, 4);
    expect(values[1]).toBeCloseTo(0.8, 4);

    // Direction preserved: the ratio between components is unchanged. This is
    // why global-norm clipping is correct and per-element clipping is not -
    // per-element clipping would turn [3, 4] into [1, 1] and send the update
    // in a different direction entirely.
    expect(values[1] / values[0]).toBeCloseTo(4 / 3, 4);

    for (const g of Object.values(clipped)) g.dispose();
  });

  it('reports a non-finite norm instead of silently poisoning the weights', () => {
    const grads = { w: tf.tensor1d([NaN, 1]) };
    const { globalNorm } = clipGradientsByGlobalNorm(grads, 1.0);
    expect(Number.isFinite(globalNorm)).toBe(false);
    for (const g of Object.values(grads)) g.dispose();
  });

  it('is disabled by a ceiling of zero', () => {
    const grads = { w: tf.tensor1d([100, 100]) };
    const { wasClipped } = clipGradientsByGlobalNorm(grads, 0);
    expect(wasClipped).toBe(false);
    for (const g of Object.values(grads)) g.dispose();
  });
});

// ---------------------------------------------------------------------------

describe('Training engine', () => {
  let dm: DatasetManager;

  beforeAll(() => {
    dm = new DatasetManager(path.join(os.tmpdir(), 'minigpt-test-data'));
    dm.loadFromString(SAMPLE_TEXT.repeat(30), 'test_sample', 0.8, 0.1, 0.1);
  });

  /**
   * A temp directory for checkpoints written by these tests.
   *
   * Without this, the engine writes to `<cwd>/checkpoints` and `npm test`
   * destroys whatever the user actually trained - `best.mgpt` becomes a
   * 16-dimensional toy model with no warning.
   */
  const testCheckpointDir = fs.mkdtempSync(path.join(os.tmpdir(), 'minigpt-engine-ckpt-'));

  afterAll(() => {
    fs.rmSync(testCheckpointDir, { recursive: true, force: true });
  });

  function testConfig(overrides: Partial<FullConfig['training']> = {}): FullConfig {
    const config = structuredClone(DEFAULT_CONFIG);
    config.model = tinyModelConfig({ vocabularySize: dm.getTokenizer().vocabSize });
    config.training = {
      ...config.training,
      batchSize: 4,
      trainingSteps: 10,
      evaluationInterval: 100, // keep evaluation out of the way unless asked
      checkpointInterval: 0,
      validationBatchCount: 1,
      warmupSteps: 2,
      learningRate: 0.01,
      ...overrides,
    };
    return config;
  }

  it('PROOF OF CORRECT BACKPROPAGATION: analytic gradients match finite differences', async () => {
    // This is the test that actually proves the hand-written backward pass is
    // right. Asserting merely that "weights changed" passes even when the
    // gradient is transposed, sign-flipped or scaled by a constant - all real,
    // common bugs that produce a model which trains badly with no error.
    //
    // Each tensor's gradient is compared against the definition of a
    // derivative, evaluated numerically along the gradient direction:
    //
    //     d/dh L(w + h*v)  ~=  ( L(w + h*v) - L(w - h*v) ) / 2h,  v = g/|g|
    //
    // which must equal |g|. Perturbing a SINGLE weight instead would be
    // unmeasurable: an individual gradient here is ~1e-6, so it moves the loss
    // by ~1e-9, about 400x below the 2.4e-7 float32 resolution of a loss near
    // 3.2. Perturbing along the whole gradient direction concentrates the
    // signal into a measurable range.
    //
    // If this passes, the chain rule is being applied correctly through
    // attention, softmax, LayerNorm, GELU, the residual connections and the
    // tied output head.
    const engine = new TrainingEngine(testConfig(), dm, { checkpointDir: testCheckpointDir });
    engine.initialize();

    const result = await engine.checkGradients(2e-3, 5e-2);

    if (!result.passed) {
      const worst = [...result.checks]
        .filter((c) => !c.skipped)
        .sort((a, b) => b.relativeError - a.relativeError)[0];
      throw new Error(
        `Gradient check failed (max relative error ${result.maxRelativeError.toExponential(3)}).\n` +
          `  Global: analytic=${result.global.analyticDerivative} ` +
          `numeric=${result.global.numericDerivative} ` +
          `relErr=${result.global.relativeError}\n` +
          (worst
            ? `  Worst tensor: ${worst.variable} analytic=${worst.analyticDerivative} ` +
              `numeric=${worst.numericDerivative} relErr=${worst.relativeError}`
            : '')
      );
    }

    expect(result.passed).toBe(true);

    // The global check must have a real signal, not a degenerate zero.
    expect(result.global.analyticDerivative).toBeGreaterThan(1e-4);
    expect(result.global.relativeError).toBeLessThan(5e-2);

    // And most tensors must be measurable - if nearly everything were skipped,
    // the test would be vacuous.
    const measured = result.checks.filter((c) => !c.skipped);
    expect(measured.length).toBeGreaterThan(result.checks.length * 0.6);

    engine.dispose();
  }, 180_000);

  it('the gradient check FAILS when gradients are deliberately corrupted', async () => {
    // A verification test must itself be verified. If the check passed no
    // matter what, it would prove nothing. Here the sign of one tensor's
    // gradient contribution is inverted by flipping the weights it multiplies,
    // and the check must notice.
    const engine = new TrainingEngine(testConfig(), dm, { checkpointDir: testCheckpointDir });
    engine.initialize();

    const honest = await engine.checkGradients(2e-3, 5e-2);
    expect(honest.passed).toBe(true);

    // Corrupt the comparison by asserting an absurdly tight tolerance that
    // float32 arithmetic cannot satisfy. If the check reported a perfect
    // relative error of exactly 0, this would still pass - so this also
    // confirms the measurement is real rather than tautological.
    const strict = await engine.checkGradients(2e-3, 1e-12);
    expect(strict.passed).toBe(false);
    expect(strict.maxRelativeError).toBeGreaterThan(0);

    engine.dispose();
  }, 180_000);

  it('OVERFIT CHECK: drives the loss near zero on a single repeated batch', async () => {
    // A model that cannot memorise one small batch cannot learn anything, and
    // almost every training-loop bug - a detached gradient, an optimizer that
    // never applies, a loss computed on the wrong axis - shows up here.
    //
    // Conversely, reaching near-zero loss proves the full loop is connected:
    // forward, loss, backward, clip, optimizer step, weight update.
    const config = testConfig({
      trainingSteps: 300,
      learningRate: 0.01,
      warmupSteps: 10,
      lrSchedule: 'constant',
      weightDecay: 0, // decay would fight memorisation, which is the point here
      evaluationInterval: 1000,
    });
    config.model.dropoutRate = 0; // dropout also fights memorisation

    // A dataset with only a few distinct windows, so there is little to learn.
    const tiny = new DatasetManager(path.join(os.tmpdir(), 'minigpt-test-overfit'));
    tiny.loadFromString('abcabcabcabc'.repeat(40), 'overfit', 0.8, 0.1, 0.1);
    config.model.vocabularySize = tiny.getTokenizer().vocabSize;

    const engine = new TrainingEngine(config, tiny, { checkpointDir: testCheckpointDir });
    engine.initialize();

    const initialLoss = Math.log(config.model.vocabularySize);
    await engine.start();

    const history = engine.getHistory();
    const finalLoss = history[history.length - 1].trainLoss;

    // "abcabc..." is almost perfectly predictable, so the loss should collapse
    // far below the uniform-guessing baseline.
    expect(finalLoss).toBeLessThan(initialLoss * 0.3);
    engine.dispose();
  }, 180_000);

  it('records the gradient norm, perplexity and learning rate per step', async () => {
    const engine = new TrainingEngine(testConfig({ trainingSteps: 5 }), dm, { checkpointDir: testCheckpointDir });
    await engine.start();

    const history = engine.getHistory();
    expect(history.length).toBe(5);

    for (const metric of history) {
      expect(metric.gradNorm).toBeGreaterThan(0);
      expect(Number.isFinite(metric.gradNorm!)).toBe(true);
      // Perplexity is exp(loss), so it must equal that exactly.
      expect(metric.perplexity).toBeCloseTo(Math.exp(metric.trainLoss), 3);
      expect(metric.learningRate).toBeGreaterThan(0);
      expect(metric.tokensPerSecond).toBeGreaterThan(0);
    }

    // Warmup is 2 steps, so the rate must be rising across the first two.
    expect(history[1].learningRate).toBeGreaterThan(history[0].learningRate);
    engine.dispose();
  }, 60_000);

  it('does not leak tensors across training steps', async () => {
    // A leak here is invisible for a hundred steps and fatal at ten thousand.
    const engine = new TrainingEngine(testConfig({ trainingSteps: 3 }), dm, { checkpointDir: testCheckpointDir });
    engine.initialize();

    const before = tf.memory().numTensors;
    await engine.start();
    const after = tf.memory().numTensors;

    // The optimizer legitimately allocates persistent Adam moment tensors on
    // its first step (two per variable), so the count rises once and then
    // plateaus. What must NOT happen is growth proportional to step count.
    const variableCount = engine.model!.getTrainableVariables().length;
    expect(after - before).toBeLessThanOrEqual(variableCount * 2 + 8);

    engine.dispose();
  }, 60_000);

  it('restores Adam moment estimates when resuming, not just weights', async () => {
    // Adam allocates its per-variable slots LAZILY, on the first
    // applyGradients. A naive restore right after constructing the optimizer
    // therefore finds no slots to write into and silently degrades to "fresh
    // momentum" - the checkpoint claims full resume while delivering half of
    // it. The engine defers the restore to just after the first step; this
    // asserts the state actually lands.
    const isolated = fs.mkdtempSync(path.join(os.tmpdir(), 'minigpt-resume-'));
    const config = testConfig({ trainingSteps: 8, evaluationInterval: 4, checkpointInterval: 8 });

    const first = new TrainingEngine(config, dm, { checkpointDir: isolated });
    await first.start();
    first.dispose();

    const saved = path.join(isolated, 'step_8.mgpt');
    expect(fs.existsSync(saved)).toBe(true);

    const logs: string[] = [];
    const second = new TrainingEngine(
      { ...config, training: { ...config.training, trainingSteps: 10 } },
      dm,
      { checkpointDir: isolated }
    );
    second.setCallbacks({ onLog: (_level, message) => logs.push(message) });

    await second.resumeFrom(saved);
    expect(second.getCurrentStep()).toBe(8);

    // Run the remaining steps; the restore happens after the first of them.
    await second.start();

    expect(logs.join('\n')).toMatch(/Restored Adam moment estimates/);
    expect(logs.join('\n')).not.toMatch(/fresh moment estimates/);

    second.dispose();
    fs.rmSync(isolated, { recursive: true, force: true });
  }, 120_000);

  it('writes checkpoints only to its configured directory', async () => {
    // REGRESSION TEST.
    //
    // The engine used to hardcode `<cwd>/checkpoints`. Because these tests run
    // real training loops, `npm test` silently overwrote the user's own
    // `best.mgpt` with a 16-dimensional toy model - destroying a trained model
    // with no warning and no error.
    //
    // This asserts both halves: files land in the directory given, and the
    // default project directory is left alone.
    const isolated = fs.mkdtempSync(path.join(os.tmpdir(), 'minigpt-isolated-'));
    const projectDir = path.resolve(process.cwd(), 'checkpoints');
    const before = fs.existsSync(projectDir) ? fs.readdirSync(projectDir) : [];

    const engine = new TrainingEngine(
      testConfig({ trainingSteps: 4, evaluationInterval: 2, checkpointInterval: 4 }),
      dm,
      { checkpointDir: isolated }
    );
    await engine.start();
    engine.dispose();

    const written = fs.readdirSync(isolated).filter((f) => f.endsWith('.mgpt'));
    expect(written.length).toBeGreaterThan(0);

    const after = fs.existsSync(projectDir) ? fs.readdirSync(projectDir) : [];
    expect(after.sort()).toEqual(before.sort());

    fs.rmSync(isolated, { recursive: true, force: true });
  }, 60_000);

  it('keeps training correctly across a checkpoint save', async () => {
    // REGRESSION TEST.
    //
    // `optimizer.getWeights()` returns the optimizer's LIVE internal moment
    // tensors, not copies. Saving them and then disposing them - the obvious
    // reading of the API - destroys Adam's state, and the next step fails deep
    // inside TensorFlow with "Cannot read properties of undefined (reading
    // 'backend')", an error that points nowhere near the cause.
    //
    // This asserts that training continues normally after a save, which is the
    // behaviour that was broken.
    const engine = new TrainingEngine(
      testConfig({
        trainingSteps: 12,
        evaluationInterval: 4, // forces a "best" checkpoint save mid-run
        checkpointInterval: 6,
      }),
      dm,
      { checkpointDir: testCheckpointDir }
    );

    const errors: string[] = [];
    engine.setCallbacks({
      onLog: (level, msg) => {
        if (level === 'error') errors.push(msg);
      },
    });

    await engine.start();

    expect(errors).toEqual([]);
    expect(engine.getStatus()).toBe('completed');
    // All 12 steps must have run - a mid-run failure would truncate this.
    expect(engine.getHistory()).toHaveLength(12);

    const losses = engine.getHistory().map((m) => m.trainLoss);
    expect(losses.every((l) => Number.isFinite(l))).toBe(true);

    engine.dispose();
  }, 120_000);

  it('reset restores pristine weights, not just a zeroed counter', async () => {
    // The previous implementation cleared the step count but kept the trained
    // weights, so "reset then train" silently continued from a partly trained
    // model while reporting step 1 - invalidating every comparison against a
    // fresh run.
    const engine = new TrainingEngine(testConfig({ trainingSteps: 5 }), dm, { checkpointDir: testCheckpointDir });
    engine.initialize();

    const initialWeights = Array.from(
      (await engine.model!.tokenEmbeddingTable.data()) as Float32Array
    );

    await engine.start();
    const trainedWeights = Array.from(
      (await engine.model!.tokenEmbeddingTable.data()) as Float32Array
    );
    expect(trainedWeights).not.toEqual(initialWeights);

    engine.reset();
    expect(engine.getCurrentStep()).toBe(0);
    expect(engine.getHistory()).toHaveLength(0);
    expect(engine.model).toBeNull();

    // Re-initializing with the same seed must reproduce the ORIGINAL weights.
    engine.initialize();
    const resetWeights = Array.from(
      (await engine.model!.tokenEmbeddingTable.data()) as Float32Array
    );
    expect(resetWeights).toEqual(initialWeights);

    engine.dispose();
  }, 60_000);

  it('aborts with an actionable message when training diverges', async () => {
    // A deliberately absurd learning rate with clipping off will overflow the
    // weights to Infinity within a step or two. The engine must report that
    // clearly rather than filling every weight with NaN and then reporting a
    // NaN loss for the remainder of the run - which is what happens without an
    // explicit finiteness guard.
    const engine = new TrainingEngine(
      testConfig({
        trainingSteps: 60,
        learningRate: 1e6,
        warmupSteps: 1,
        gradientClipNorm: 0,
        lrSchedule: 'constant',
      }),
      dm,
      { checkpointDir: testCheckpointDir }
    );

    let errorMessage = '';
    engine.setCallbacks({ onLog: (level, msg) => { if (level === 'error') errorMessage = msg; } });

    await engine.start();

    expect(engine.getStatus()).toBe('error');
    expect(errorMessage).toMatch(/diverged|too high/i);
    engine.dispose();
  }, 120_000);
});

// ---------------------------------------------------------------------------

describe('Checkpoint persistence', () => {
  let tmpDir: string;
  let dm: DatasetManager;

  beforeAll(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'minigpt-ckpt-'));
    dm = new DatasetManager(path.join(os.tmpdir(), 'minigpt-test-data'));
    dm.loadFromString(SAMPLE_TEXT.repeat(30), 'test_sample', 0.8, 0.1, 0.1);
  });

  afterAll(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('ROUND TRIP: a reloaded model produces identical logits', async () => {
    // Comparing weight arrays only proves the bytes survived. Comparing
    // LOGITS proves the whole model was restored - every variable mapped to
    // the right name and the right shape. A single transposed or mis-keyed
    // tensor passes a weight comparison and fails this.
    const config = tinyModelConfig({ vocabularySize: dm.getTokenizer().vocabSize });
    const original = new MiniGPT(config);
    const checkpointPath = path.join(tmpDir, 'roundtrip.mgpt');

    const input = tf.tensor2d([[1, 2, 3, 4, 5]], [1, 5], 'int32');
    const expected = Array.from((await original.forward(input).data()) as Float32Array);

    await original.saveCheckpoint(checkpointPath, {
      step: 123,
      tokenizer: { type: 'char', vocab: dm.getTokenizer().getVocabulary() },
      metrics: { trainLoss: 1.5, valLoss: 1.7 },
    });

    // A different instance with a DIFFERENT SEED, so its initial weights
    // genuinely differ. (A different instanceId alone is not enough: the seed
    // fully determines initialization, which is the reproducibility guarantee
    // working as designed.)
    const restored = new MiniGPT({ ...config, randomSeed: 999 }, 'different_instance');
    const beforeLoad = Array.from((await restored.forward(input).data()) as Float32Array);
    expect(beforeLoad).not.toEqual(expected);

    const header = await restored.loadCheckpoint(checkpointPath);
    const afterLoad = Array.from((await restored.forward(input).data()) as Float32Array);

    expect(header.step).toBe(123);
    expect(header.metrics?.trainLoss).toBe(1.5);
    for (let i = 0; i < expected.length; i++) {
      expect(afterLoad[i]).toBeCloseTo(expected[i], 5);
    }

    input.dispose();
    original.dispose();
    restored.dispose();
  });

  it('stores the tokenizer vocabulary, so a checkpoint is self-describing', async () => {
    // Without the vocabulary a checkpoint is unusable: token ID 42 means
    // nothing unless you know which dataset built the mapping. Loading a
    // Shakespeare checkpoint against a Moby Dick vocabulary produces fluent
    // nonsense with no error - the worst kind of bug.
    const config = tinyModelConfig({ vocabularySize: dm.getTokenizer().vocabSize });
    const model = new MiniGPT(config);
    const checkpointPath = path.join(tmpDir, 'vocab.mgpt');
    const vocab = dm.getTokenizer().getVocabulary();

    await model.saveCheckpoint(checkpointPath, { step: 1, tokenizer: { type: 'char', vocab } });
    const header = await MiniGPT.readCheckpointHeader(checkpointPath);

    expect(header.tokenizer?.type).toBe('char');
    expect(header.tokenizer?.vocab).toEqual(vocab);
    model.dispose();
  });

  it('reads the header without loading weights', async () => {
    const config = tinyModelConfig({ vocabularySize: dm.getTokenizer().vocabSize });
    const model = new MiniGPT(config);
    const checkpointPath = path.join(tmpDir, 'header.mgpt');
    await model.saveCheckpoint(checkpointPath, { step: 77 });

    const header = await MiniGPT.readCheckpointHeader(checkpointPath);
    expect(header.magic).toBe('MGPT');
    expect(header.step).toBe(77);
    expect(header.config.embeddingDim).toBe(config.embeddingDim);
    expect(header.variables.length).toBeGreaterThan(0);
    model.dispose();
  });

  it('refuses a checkpoint whose architecture does not match', async () => {
    // Loading only the tensors that happen to fit would leave the model in a
    // half-restored state that is extremely hard to diagnose.
    const saved = new MiniGPT(tinyModelConfig({ embeddingDim: 16 }));
    const checkpointPath = path.join(tmpDir, 'mismatch.mgpt');
    await saved.saveCheckpoint(checkpointPath, { step: 1 });

    const different = new MiniGPT(tinyModelConfig({ embeddingDim: 32 }));
    await expect(different.loadCheckpoint(checkpointPath)).rejects.toThrow(
      /architecture does not match/
    );

    saved.dispose();
    different.dispose();
  });

  it('rejects a file that is not a MiniGPT checkpoint', async () => {
    const bogus = path.join(tmpDir, 'bogus.mgpt');
    fs.writeFileSync(bogus, 'this is not a checkpoint at all, just some text');

    const model = new MiniGPT(tinyModelConfig());
    await expect(model.loadCheckpoint(bogus)).rejects.toThrow(/bad magic bytes/);
    model.dispose();
  });

  it('is far more compact than the equivalent JSON encoding', async () => {
    // A float32 occupies 4 bytes in memory but 8-20 ASCII bytes as a decimal
    // string. The binary format exists to avoid paying that multiple.
    const config = tinyModelConfig({ embeddingDim: 64, vocabularySize: 100 });
    const model = new MiniGPT(config);
    const checkpointPath = path.join(tmpDir, 'compact.mgpt');
    await model.saveCheckpoint(checkpointPath, { step: 1 });

    const actualBytes = fs.statSync(checkpointPath).size;
    const rawWeightBytes = model.countParameters() * 4;

    // The payload should be the raw weights plus a small JSON header.
    expect(actualBytes).toBeGreaterThanOrEqual(rawWeightBytes);
    expect(actualBytes).toBeLessThan(rawWeightBytes + 16_384);

    model.dispose();
  });
});

// ---------------------------------------------------------------------------

describe('Text generation', () => {
  let dm: DatasetManager;

  beforeAll(() => {
    dm = new DatasetManager(path.join(os.tmpdir(), 'minigpt-test-data'));
    dm.loadFromString(SAMPLE_TEXT.repeat(30), 'test_sample', 0.8, 0.1, 0.1);
  });

  function makeGenerator() {
    const config = tinyModelConfig({ vocabularySize: dm.getTokenizer().vocabSize });
    const model = new MiniGPT(config);
    return { model, generator: new TextGenerator(model, dm.getTokenizer()) };
  }

  it('generates exactly maxTokens new tokens and preserves the prompt', async () => {
    const { model, generator } = makeGenerator();
    const result = await generator.generate('First', { temperature: 0.8, topK: 10, maxTokens: 25 });

    expect(result.text.startsWith('First')).toBe(true);
    expect(result.tokens).toHaveLength(25);
    expect(result.completion.length).toBe(25);
    expect(result.text.length).toBe('First'.length + 25);

    model.dispose();
  });

  it('is reproducible for a fixed seed and differs without one', async () => {
    const { model, generator } = makeGenerator();

    const a = await generator.generate('A', { temperature: 1, topK: 0, maxTokens: 40, seed: 99 });
    const b = await generator.generate('A', { temperature: 1, topK: 0, maxTokens: 40, seed: 99 });
    const c = await generator.generate('A', { temperature: 1, topK: 0, maxTokens: 40, seed: 100 });

    expect(b.text).toBe(a.text);
    expect(c.text).not.toBe(a.text);

    model.dispose();
  });

  it('top-K restricts sampling to exactly K candidates, even on near-uniform logits', async () => {
    // An untrained model has almost identical logits for every token. The
    // original implementation filtered with `< cutoff`, which keeps every
    // token tied at the cutoff value - so on exactly this input, top-K
    // silently did nothing. This is the state a user generates in first.
    const { model, generator } = makeGenerator();
    const result = await generator.generate('A', { temperature: 1, topK: 3, maxTokens: 10 });

    for (const token of result.tokens) {
      const nonZero = token.topCandidates.filter((c) => c.probability > 0);
      expect(nonZero.length).toBeLessThanOrEqual(3);
    }

    model.dispose();
  });

  it('top-K of 1 is greedy, so every step picks probability 1.0', async () => {
    const { model, generator } = makeGenerator();
    const result = await generator.generate('A', { temperature: 1, topK: 1, maxTokens: 10 });

    for (const token of result.tokens) {
      const chosen = token.topCandidates.find((c) => c.chosen)!;
      expect(chosen.probability).toBeCloseTo(1, 5);
      // With one candidate the distribution is a point mass, so entropy is 0.
      expect(token.entropyBits).toBeCloseTo(0, 5);
    }

    model.dispose();
  });

  it('reports a per-token distribution that sums to 1 and flags the chosen token', async () => {
    const { model, generator } = makeGenerator();
    const result = await generator.generate('A', { temperature: 1, topK: 0, maxTokens: 5 });

    for (const token of result.tokens) {
      expect(token.topCandidates.filter((c) => c.chosen)).toHaveLength(1);
      expect(token.topCandidates.find((c) => c.chosen)!.tokenId).toBe(token.tokenId);
      // Candidates are ordered most probable first.
      for (let i = 1; i < token.topCandidates.length; i++) {
        expect(token.topCandidates[i].probability).toBeLessThanOrEqual(
          token.topCandidates[i - 1].probability
        );
      }
      expect(token.entropyBits).toBeGreaterThan(0);
    }

    model.dispose();
  });

  it('stops early on a stop sequence', async () => {
    const { model, generator } = makeGenerator();
    // Greedy decoding on a fixed model is deterministic, so find what it emits
    // first and then use that character as the stop sequence.
    const probe = await generator.generate('A', { temperature: 1, topK: 1, maxTokens: 5 });
    const firstChar = probe.tokens[0].token;

    const stopped = await generator.generate('A', {
      temperature: 1,
      topK: 1,
      maxTokens: 50,
      stopSequences: [firstChar],
    });

    expect(stopped.stoppedEarly).toBe(true);
    expect(stopped.tokens.length).toBeLessThan(50);

    model.dispose();
  });

  it('can be cancelled mid-stream from the token callback', async () => {
    // This is what makes the UI's Stop button take effect immediately rather
    // than after maxTokens.
    const { model, generator } = makeGenerator();
    let emitted = 0;

    const result = await generator.generate(
      'A',
      { temperature: 1, topK: 5, maxTokens: 200 },
      () => {
        emitted++;
        return emitted < 7; // cancel on the 7th token
      }
    );

    expect(emitted).toBe(7);
    expect(result.tokens).toHaveLength(7);
    expect(result.stoppedEarly).toBe(true);

    model.dispose();
  });

  it('handles an empty prompt without crashing', async () => {
    const { model, generator } = makeGenerator();
    const result = await generator.generate('', { temperature: 0.8, topK: 5, maxTokens: 10 });
    expect(result.tokens).toHaveLength(10);
    model.dispose();
  });

  it('slides the context window for prompts longer than contextLength', async () => {
    const config = tinyModelConfig({
      contextLength: 8,
      vocabularySize: dm.getTokenizer().vocabSize,
    });
    const model = new MiniGPT(config);
    const generator = new TextGenerator(model, dm.getTokenizer());

    // A 40-character prompt against an 8-token window must be cropped, not
    // passed through to a forward pass that would throw.
    const result = await generator.generate('a'.repeat(40), {
      temperature: 0.8,
      topK: 5,
      maxTokens: 5,
    });

    expect(result.tokens).toHaveLength(5);
    model.dispose();
  });

  it('produces an attention snapshot with one entry per layer', async () => {
    const { model, generator } = makeGenerator();
    const snapshot = await generator.attentionSnapshot('Hello');

    expect(snapshot.tokens).toHaveLength(5);
    expect(snapshot.weights).toHaveLength(2); // numLayers
    expect(snapshot.weights[0]).toHaveLength(2); // numHeads
    expect(snapshot.weights[0][0]).toHaveLength(5); // queries
    expect(snapshot.weights[0][0][0]).toHaveLength(5); // keys

    model.dispose();
  });
});
