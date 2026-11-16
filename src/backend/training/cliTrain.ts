import '../initNativeTf';
import * as tf from '@tensorflow/tfjs';
import { TrainingEngine } from './engine';
import { DatasetManager } from '../dataset/datasetManager';
import { DEFAULT_CONFIG } from '../../shared/config';
import { TextGenerator } from '../model/generation';
import { FullConfig } from '../../shared/types';

/**
 * ============================================================================
 * MiniGPT Lab - Command-Line Training Runner
 * ============================================================================
 *
 * Trains a model without opening a browser, which is what you want for a long
 * run, for a scripted sweep, or over SSH.
 *
 * Usage:
 *   npm run train
 *   npm run train -- --steps 3000 --layers 6 --embed 192 --lr 0.002
 *   npm run train -- --tokenizer word --sample-prompt "The "
 *   npm run train -- --gradient-check
 *
 * Every flag maps onto the same configuration object the UI edits, so a CLI
 * run and a UI run with matching settings are the same run.
 */

/** Minimal `--flag value` parser - no dependency needed for this. */
function parseArgs(argv: string[]): Record<string, string | boolean> {
  const out: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) {
      out[key] = true;
    } else {
      out[key] = next;
      i++;
    }
  }
  return out;
}

function num(value: string | boolean | undefined, fallback: number): number {
  if (typeof value !== 'string') return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

/**
 * Formats an integer with thousands separators, pinned to en-US.
 *
 * Bare `toLocaleString()` follows the host locale, which on an Indian-locale
 * machine renders 892315 as "8,92,315". Console output should not change shape
 * depending on who runs it.
 */
function int(value: number): string {
  return value.toLocaleString('en-US');
}

/** Renders a value as a fixed-width column entry. */
function pad(value: string | number, width: number): string {
  return String(value).padStart(width);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    console.log(`
MiniGPT Lab CLI trainer

  --steps N            Total training steps              (default ${DEFAULT_CONFIG.training.trainingSteps})
  --layers N           Transformer blocks                (default ${DEFAULT_CONFIG.model.numLayers})
  --heads N            Attention heads                   (default ${DEFAULT_CONFIG.model.numHeads})
  --embed N            Embedding dimension (d_model)     (default ${DEFAULT_CONFIG.model.embeddingDim})
  --context N          Context length                    (default ${DEFAULT_CONFIG.model.contextLength})
  --batch N            Batch size                        (default ${DEFAULT_CONFIG.training.batchSize})
  --lr F               Peak learning rate                (default ${DEFAULT_CONFIG.training.learningRate})
  --warmup N           Warmup steps                      (default ${DEFAULT_CONFIG.training.warmupSteps})
  --schedule S         constant | cosine | linear        (default ${DEFAULT_CONFIG.training.lrSchedule})
  --dropout F          Dropout rate                      (default ${DEFAULT_CONFIG.model.dropoutRate})
  --weight-decay F     AdamW weight decay                (default ${DEFAULT_CONFIG.training.weightDecay})
  --clip F             Gradient clip norm, 0 = off       (default ${DEFAULT_CONFIG.training.gradientClipNorm})
  --seed N             Random seed                       (default ${DEFAULT_CONFIG.model.randomSeed})
  --tokenizer S        char | word                       (default char)
  --no-tie             Disable embedding/head weight tying
  --sample-prompt S    Prompt for periodic samples       (default "${DEFAULT_CONFIG.generation.samplePrompt}")
  --gradient-check     Verify backpropagation and exit
  --help               Show this message
`);
    process.exit(0);
  }

  console.log('='.repeat(72));
  console.log('  MiniGPT Lab - Transformer Training CLI');
  console.log('='.repeat(72));

  // Report the active backend first. A silent fallback to the pure-JS backend
  // is a ~10x slowdown with no error, so it needs to be visible up front.
  const backend = tf.getBackend();
  console.log(`\nTensorFlow backend: ${backend}`);
  if (backend !== 'tensorflow') {
    console.log(
      '  WARNING: the native backend did not load. Training will be roughly 10x\n' +
        '  slower than it should be. Check that @tensorflow/tfjs-node installed correctly.'
    );
  }

  // --- Configuration from flags --------------------------------------------
  const config: FullConfig = structuredClone(DEFAULT_CONFIG);

  config.model.numLayers = num(args.layers, config.model.numLayers);
  config.model.numHeads = num(args.heads, config.model.numHeads);
  config.model.embeddingDim = num(args.embed, config.model.embeddingDim);
  config.model.contextLength = num(args.context, config.model.contextLength);
  config.model.dropoutRate = num(args.dropout, config.model.dropoutRate);
  config.model.randomSeed = num(args.seed, config.model.randomSeed ?? 42);
  if (args['no-tie']) config.model.tieWeights = false;

  config.training.trainingSteps = num(args.steps, config.training.trainingSteps);
  config.training.batchSize = num(args.batch, config.training.batchSize);
  config.training.learningRate = num(args.lr, config.training.learningRate);
  config.training.warmupSteps = num(args.warmup, config.training.warmupSteps);
  config.training.weightDecay = num(args['weight-decay'], config.training.weightDecay);
  config.training.gradientClipNorm = num(args.clip, config.training.gradientClipNorm);
  if (typeof args.schedule === 'string' && ['constant', 'cosine', 'linear'].includes(args.schedule)) {
    config.training.lrSchedule = args.schedule as 'constant' | 'cosine' | 'linear';
  }
  if (typeof args['sample-prompt'] === 'string') {
    config.generation.samplePrompt = args['sample-prompt'];
  }

  // Fail fast on an invalid architecture rather than throwing from deep inside
  // the attention constructor several seconds later.
  if (config.model.embeddingDim % config.model.numHeads !== 0) {
    console.error(
      `\nERROR: --embed ${config.model.embeddingDim} is not divisible by --heads ` +
        `${config.model.numHeads}. Each head takes an equal slice of the embedding.`
    );
    process.exit(1);
  }
  if (config.training.warmupSteps >= config.training.trainingSteps) {
    console.error(
      `\nERROR: --warmup ${config.training.warmupSteps} must be less than --steps ` +
        `${config.training.trainingSteps}, otherwise the learning rate never reaches its peak.`
    );
    process.exit(1);
  }

  // --- Dataset --------------------------------------------------------------
  const tokenizerType = args.tokenizer === 'word' ? 'word' : 'char';
  const datasetManager = new DatasetManager(undefined, config.training.dataSeed);

  console.log('\nPreparing dataset...');
  const dataPath = await datasetManager.ensureTinyShakespeare((pct) => {
    if (pct === 100) console.log('  Download complete.');
  });
  datasetManager.loadFromFile(
    dataPath,
    config.training.trainRatio,
    config.training.valRatio,
    config.training.testRatio,
    tokenizerType
  );

  const stats = datasetManager.getStatistics();
  console.log(`  Dataset:    ${stats.name} (${tokenizerType}-level)`);
  console.log(`  Characters: ${int(stats.totalCharacters)}`);
  console.log(`  Vocabulary: ${int(stats.uniqueCharacters)} tokens`);
  console.log(
    `  Splits:     ${int(stats.trainCharacters)} train / ` +
      `${int(stats.valCharacters)} val / ${int(stats.testCharacters)} test`
  );

  // --- Engine ---------------------------------------------------------------
  const engine = new TrainingEngine(config, datasetManager);
  engine.initialize();

  // --- Gradient check mode --------------------------------------------------
  if (args['gradient-check']) {
    console.log('\nVerifying backpropagation by finite differences...');
    console.log(
      '  For each tensor, the loss is perturbed along its own gradient direction and'
    );
    console.log('  ( L(w + h*v) - L(w - h*v) ) / 2h is compared against |g|.\n');

    const result = await engine.checkGradients();

    console.log(
      `  ${'tensor'.padEnd(40)} ${pad('size', 8)} ${pad('analytic', 12)} ${pad('numeric', 12)} ${pad('rel.err', 10)}`
    );
    console.log('  ' + '-'.repeat(86));
    for (const check of result.checks) {
      const name = check.variable.replace(/^gpt_\d+_\d+_/, '');
      if (check.skipped) {
        console.log(
          `  ${name.slice(0, 40).padEnd(40)} ${pad(check.size, 8)} ` +
            `${pad(check.analyticDerivative.toExponential(3), 12)} ${pad('-', 12)} ${pad('skipped', 10)}`
        );
        continue;
      }
      console.log(
        `  ${name.slice(0, 40).padEnd(40)} ${pad(check.size, 8)} ` +
          `${pad(check.analyticDerivative.toExponential(3), 12)} ` +
          `${pad(check.numericDerivative.toExponential(3), 12)} ` +
          `${pad(check.relativeError.toExponential(2), 10)}`
      );
    }

    console.log('  ' + '-'.repeat(86));
    console.log(
      `  ${'ALL PARAMETERS AT ONCE'.padEnd(40)} ${pad('', 8)} ` +
        `${pad(result.global.analyticDerivative.toExponential(3), 12)} ` +
        `${pad(result.global.numericDerivative.toExponential(3), 12)} ` +
        `${pad(result.global.relativeError.toExponential(2), 10)}`
    );

    const skipped = result.checks.filter((c) => c.skipped).length;
    if (skipped > 0) {
      console.log(
        `\n  ${skipped} tensor(s) skipped: their gradient norm was below what a float32\n` +
          `  loss can resolve at h = ${result.epsilon.toExponential(1)}, so the numeric estimate\n` +
          `  would be rounding noise rather than a measurement.`
      );
    }

    console.log(
      `\n  Base tolerance ${result.tolerance.toExponential(1)}, widened per tensor by the`
    );
    console.log(
      `  float32 measurement noise (${(result.noiseFloor ?? 0).toExponential(2)} absolute), which is`
    );
    console.log(`  relatively larger for tensors with small gradients.`);
    console.log(
      `\n  Worst relative error:        ${result.maxRelativeError.toExponential(3)}`
    );
    console.log(
      `  Worst error / its tolerance: ${(result.maxErrorRatio ?? 0).toFixed(3)} ` +
        `(must be <= 1.000)`
    );
    console.log(
      result.passed
        ? '  PASS - backpropagation is computing correct gradients.'
        : '  FAIL - the analytic and numerical gradients disagree.'
    );
    process.exit(result.passed ? 0 : 1);
  }

  // --- Train ----------------------------------------------------------------
  console.log(
    `\nModel:      ${config.model.numLayers}L / ${config.model.numHeads}H / ` +
      `d_model ${config.model.embeddingDim} / context ${config.model.contextLength}`
  );
  console.log(`Parameters: ${int(engine.model!.countParameters())}`);
  console.log(
    `Schedule:   ${config.training.lrSchedule}, peak ${config.training.learningRate}, ` +
      `${config.training.warmupSteps} warmup, floor ${config.training.minLearningRate}`
  );
  console.log(
    `Regulariz.: dropout ${config.model.dropoutRate}, weight decay ` +
      `${config.training.weightDecay}, clip ${config.training.gradientClipNorm}`
  );
  console.log(`\nStarting ${config.training.trainingSteps} steps...\n`);
  console.log(
    `  ${pad('step', 6)} ${pad('train', 8)} ${pad('val', 8)} ${pad('ppl', 8)} ` +
      `${pad('lr', 10)} ${pad('grad', 7)} ${pad('tok/s', 8)} ${pad('eta', 7)}`
  );
  console.log('  ' + '-'.repeat(70));

  engine.setCallbacks({
    onLog: (level, msg) => {
      // The table carries the per-step detail; only surface warnings and errors.
      if (level !== 'info') console.log(`  [${level.toUpperCase()}] ${msg}`);
    },
    onMetric: (m) => {
      if (m.valLoss === undefined && m.step % 25 !== 0) return;
      const etaSec = Math.round(m.estimatedRemainingMs / 1000);
      console.log(
        `  ${pad(m.step, 6)} ${pad(m.trainLoss.toFixed(4), 8)} ` +
          `${pad(m.valLoss !== undefined ? m.valLoss.toFixed(4) : '-', 8)} ` +
          `${pad(m.valPerplexity !== undefined ? m.valPerplexity.toFixed(1) : '-', 8)} ` +
          `${pad(m.learningRate.toExponential(2), 10)} ` +
          `${pad((m.gradNorm ?? 0).toFixed(2), 7)} ` +
          `${pad(int(m.tokensPerSecond ?? 0), 8)} ` +
          `${pad(etaSec > 60 ? `${Math.round(etaSec / 60)}m` : `${etaSec}s`, 7)}`
      );
    },
  });

  await engine.start();

  const best = engine.getBestValidation();
  console.log('\n' + '='.repeat(72));
  console.log(
    `  Best validation loss: ${best.loss.toFixed(4)} at step ${best.step} ` +
      `(perplexity ${Math.exp(best.loss).toFixed(1)})`
  );
  console.log(`  Saved to checkpoints/best.mgpt`);

  // --- Final sample ---------------------------------------------------------
  console.log('\nGenerating a sample from the trained model...\n');
  const generator = new TextGenerator(engine.model!, datasetManager.getTokenizer());
  const result = await generator.generate(config.generation.samplePrompt || '\n', {
    ...config.generation,
    maxTokens: 400,
  });

  console.log('-'.repeat(72));
  console.log(result.text);
  console.log('-'.repeat(72));

  engine.dispose();
  process.exit(0);
}

if (require.main === module) {
  main().catch((err) => {
    console.error('\nTraining failed:', err.message);
    console.error(err.stack);
    process.exit(1);
  });
}
