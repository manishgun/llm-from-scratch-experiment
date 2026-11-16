import { FullConfig, ModelConfig, TrainingConfig, GenerationConfig } from '../types';

/**
 * ============================================================================
 * MiniGPT Lab - Parameter Metadata and Educational Explanations
 * ============================================================================
 *
 * Every single hyperparameter and setting available in MiniGPT Lab is documented
 * here in depth. This powers both the TypeScript documentation and the UI
 * "What happens if I change this?" interactive guide.
 */

export interface ParameterDocumentation {
  name: string;
  category: 'model' | 'training' | 'generation' | 'dataset';
  simpleExplanation: string;
  defaultValue: any;
  minValue?: number;
  maxValue?: number;
  step?: number;
  affects: ('speed' | 'memory' | 'learning' | 'generation')[];
  increasingEffect: string;
  decreasingEffect: string;
  advantages: string;
  disadvantages: string;
  exampleExperiment: string;
}

export const PARAMETER_DOCS: Record<string, ParameterDocumentation> = {
  embeddingDim: {
    name: 'Embedding Dimension (d_model)',
    category: 'model',
    simpleExplanation:
      'The length of the number list (vector) used to represent each token in concept space. Imagine plotting characters in a space with N dimensions; higher dimensions give more room to represent subtle nuances.',
    defaultValue: 64,
    minValue: 16,
    maxValue: 512,
    step: 16,
    affects: ['speed', 'memory', 'learning'],
    increasingEffect:
      'Increases model capacity and expressive power. The model can represent finer relationships between tokens, but calculations become slower and require more RAM.',
    decreasingEffect:
      'Model runs faster and uses minimal memory, but might underfit or fail to capture complex grammar patterns.',
    advantages: 'Rich conceptual representation and lower loss on complex datasets.',
    disadvantages: 'Slower training on CPU and risk of overfitting on small texts.',
    exampleExperiment:
      'Try comparing 32 vs 64 vs 128 embedding dimensions on Shakespeare. Notice how 128 learns character combinations faster, but takes longer per step.',
  },
  numLayers: {
    name: 'Number of Transformer Layers',
    category: 'model',
    simpleExplanation:
      'How many Transformer blocks are stacked sequentially. Each layer performs self-attention and feed-forward transformation, progressively building higher-level abstractions from the previous layer.',
    defaultValue: 4,
    minValue: 1,
    maxValue: 12,
    step: 1,
    affects: ['speed', 'memory', 'learning'],
    increasingEffect:
      'Allows the network to form deep reasoning chains (e.g. layer 1 learns characters, layer 2 learns words, layer 3 learns phrases, layer 4 learns dialogue structure).',
    decreasingEffect:
      'Shallower model; runs much faster per step, but struggles with long-range or abstract dependencies.',
    advantages: 'Significantly greater expressive capability and pattern recognition.',
    disadvantages: 'Computation scales linearly with depth, increasing CPU step time.',
    exampleExperiment:
      'Train a 1-layer model vs a 4-layer model for 300 steps. Observe how the 1-layer model outputs basic letter groupings while the 4-layer model begins forming plausible words.',
  },
  numHeads: {
    name: 'Number of Attention Heads',
    category: 'model',
    simpleExplanation:
      'How many independent attention viewpoints the model runs simultaneously. One head might focus on rhymes, another on quotes, and another on previous words.',
    defaultValue: 4,
    minValue: 1,
    maxValue: 16,
    step: 1,
    affects: ['learning'],
    increasingEffect:
      'Splits the embedding dimension into smaller subspace slices (d_k = d_model / numHeads). Gives the model multiple angles to inspect relationships.',
    decreasingEffect:
      'Fewer viewpoints; head dimension d_k is larger, but attention focuses on fewer simultaneous patterns.',
    advantages: 'Diverse relationship modeling without increasing total parameter count.',
    disadvantages: 'Must divide embedding dimension evenly (embeddingDim % numHeads === 0).',
    exampleExperiment:
      'With embedding dimension 64, test 1 head vs 4 heads. 4 heads allows diverse attention patterns without extra parameters.',
  },
  contextLength: {
    name: 'Context Length (Block Size)',
    category: 'model',
    simpleExplanation:
      'The maximum number of consecutive past characters the model can look back at to predict the next character.',
    defaultValue: 64,
    minValue: 16,
    maxValue: 256,
    step: 16,
    affects: ['speed', 'memory', 'learning'],
    increasingEffect:
      'Model can remember further back in the dialogue or paragraph. However, attention complexity grows quadratically (O(N^2)) with context length!',
    decreasingEffect:
      'Much faster training and low memory, but model has "amnesia" beyond the short window.',
    advantages: 'Preserves long-term coherence across sentences.',
    disadvantages: 'Quadratic attention compute cost causes slower CPU training for large windows.',
    exampleExperiment:
      'Compare context length 16 vs 64. At 16, the model forgets who is speaking within a sentence; at 64, it maintains speaker names like "ROMEO:".',
  },
  feedForwardMultiplier: {
    name: 'Feed-Forward Multiplier',
    category: 'model',
    simpleExplanation:
      'How much wider the hidden layer inside each Transformer block is compared to embeddingDim. In standard GPT-2, this is 4 (so hidden dimension is 4 * embeddingDim).',
    defaultValue: 4,
    minValue: 1,
    maxValue: 8,
    step: 1,
    affects: ['speed', 'memory', 'learning'],
    increasingEffect:
      'More capacity for storing factual knowledge and non-linear feature interactions.',
    decreasingEffect:
      'Smaller parameter count and faster training, but reduced capacity for complex mappings.',
    advantages: 'Standard architecture proven across modern frontier LLMs.',
    disadvantages: 'More weights means slightly more CPU math per layer.',
    exampleExperiment:
      'Test multiplier 2 vs 4. Multiplier 4 often achieves noticeably lower validation loss.',
  },
  learningRate: {
    name: 'Learning Rate',
    category: 'training',
    simpleExplanation:
      'The step size taken by the optimizer when adjusting model weights in response to prediction errors (gradients).',
    defaultValue: 0.001,
    minValue: 0.00001,
    maxValue: 0.05,
    step: 0.0001,
    affects: ['learning'],
    increasingEffect:
      'Weights change faster. If set too high (e.g. 0.05), training can explode into NaN (Not a Number) or fail to converge.',
    decreasingEffect:
      'Training is stable and smooth, but takes many more steps to reach good loss.',
    advantages: 'Proper tuning provides fast, steady loss reduction.',
    disadvantages: 'Too high causes divergence; too low causes agonizingly slow learning.',
    exampleExperiment:
      'Try 0.01 vs 0.001 vs 0.0001. You will see 0.01 jump erratically, 0.001 smoothly descend, and 0.0001 barely move.',
  },
  batchSize: {
    name: 'Batch Size',
    category: 'training',
    simpleExplanation:
      'How many sequences are packed into a single training step. The model computes loss for all of them and averages the gradient update.',
    defaultValue: 32,
    minValue: 4,
    maxValue: 128,
    step: 4,
    affects: ['speed', 'memory', 'learning'],
    increasingEffect:
      'Smoother and more accurate gradient estimates, but consumes more RAM and each step takes longer on CPU.',
    decreasingEffect:
      'Faster individual steps and low RAM, but the gradient is noisy and loss bounces around more.',
    advantages: 'Bigger batches make better use of vectorized tensor operations.',
    disadvantages: 'Large batches can saturate CPU threads and take longer per step.',
    exampleExperiment:
      'Compare batch size 8 vs 32. At 8, watch the loss graph fluctuate noisily; at 32, watch it produce a smoother curve.',
  },
  temperature: {
    name: 'Generation Temperature',
    category: 'generation',
    simpleExplanation:
      'Controls the randomness when selecting the next character from model output logits before softmax. Higher values flatten the probability distribution; lower values sharpen it.',
    defaultValue: 0.7,
    minValue: 0.1,
    maxValue: 2.0,
    step: 0.05,
    affects: ['generation'],
    increasingEffect:
      'Text becomes wild, chaotic, and diverse. Above 1.2, it will likely output typos and gibberish.',
    decreasingEffect:
      'Text becomes conservative, repetitive, and robotic. Close to 0.1, it always picks the single most probable character (greedy search).',
    advantages: 'Finds the "sweet spot" (usually 0.6 - 0.8) between coherence and creativity.',
    disadvantages: 'Does not change model weights, only affects sampling math at inference.',
    exampleExperiment:
      'Generate from prompt "To be, or not to be" at temperature 0.2, 0.7, and 1.5. Observe the shift from repetitive to natural to chaotic.',
  },
  topK: {
    name: 'Top-K Sampling',
    category: 'generation',
    simpleExplanation:
      'Filters the candidate characters to only the K most probable choices, discarding the rest before sampling.',
    defaultValue: 20,
    minValue: 1,
    maxValue: 65,
    step: 1,
    affects: ['generation'],
    increasingEffect:
      'Allows more rare or unexpected characters to be considered.',
    decreasingEffect:
      'Prevents the model from picking improbable letters or punctuation, preventing jarring mistakes.',
    advantages: 'Cuts off the "long tail" of low-probability bad tokens.',
    disadvantages: 'Setting K too small (e.g. 1 or 2) leads to repetitive looping text.',
    exampleExperiment:
      'Try top-K = 1 (greedy deterministic) vs top-K = 10 vs top-K = 50. Notice how top-K = 1 loops repeated phrases.',
  },
  trainingSteps: {
    name: 'Total Training Steps',
    category: 'training',
    simpleExplanation:
      'How many forward-backward optimization iterations to perform on the dataset.',
    defaultValue: 500,
    minValue: 50,
    maxValue: 10000,
    step: 50,
    affects: ['learning', 'speed'],
    increasingEffect:
      'Model gets more practice seeing character patterns and refines its weights further.',
    decreasingEffect:
      'Finishes faster, but model may still output broken words if stopped too early.',
    advantages: 'More steps lead to lower loss and better text quality.',
    disadvantages: 'Takes longer total wall-clock time; if dataset is tiny, can cause overfitting.',
    exampleExperiment:
      'Sample text generation at 50 steps, 200 steps, and 500 steps. Watch English emerge from random noise!',
  },
  gradientClipNorm: {
    name: 'Gradient Clip Norm',
    category: 'training',
    simpleExplanation:
      'A ceiling on how large a single weight update may be. All gradients are treated as one long vector; if its length exceeds this value, every gradient is shrunk by the same factor so the length matches exactly.',
    defaultValue: 1.0,
    minValue: 0,
    maxValue: 10,
    step: 0.1,
    affects: ['learning'],
    increasingEffect:
      'Allows larger, more aggressive updates. Above about 5 the clipping effectively stops protecting you.',
    decreasingEffect:
      'Caps every update tightly. Very small values (0.1) slow learning because even ordinary updates get shrunk.',
    advantages:
      'The single most effective guard against a loss spike to NaN. Because every gradient is scaled by the SAME factor, the update direction is preserved exactly - only its length changes.',
    disadvantages:
      'Masks the symptom of a learning rate that is genuinely too high; watch the gradient-norm chart as well.',
    exampleExperiment:
      'Set the learning rate to 0.05 with clipping OFF (0) and watch the run diverge to NaN. Set clipping to 1.0 and run it again - it survives, though it still learns poorly.',
  },
  lrSchedule: {
    name: 'Learning Rate Schedule',
    category: 'training',
    simpleExplanation:
      'How the learning rate changes over the course of the run: held constant, or decayed from the peak down to a small floor along a cosine or straight line.',
    defaultValue: 'cosine',
    affects: ['learning'],
    increasingEffect:
      'Cosine decay spends a long time near the peak rate and then settles smoothly - this is what GPT-2, GPT-3 and LLaMA use.',
    decreasingEffect:
      'A constant rate keeps taking large steps at the end of training, so the loss bounces around the minimum instead of settling into it.',
    advantages:
      'Large early steps travel quickly to a good region; small late steps settle precisely into it. Usually worth several tenths of a point of final loss.',
    disadvantages:
      'The schedule is tied to the total step count, so stopping a cosine run early leaves the model at a mid-schedule learning rate.',
    exampleExperiment:
      'Run 1000 steps with `constant`, then with `cosine`. The cosine run ends visibly lower and its loss curve is far smoother in the last quarter.',
  },
  warmupSteps: {
    name: 'Warmup Steps',
    category: 'training',
    simpleExplanation:
      'How many steps are spent ramping the learning rate up from near zero to the peak before decay begins.',
    defaultValue: 60,
    minValue: 0,
    maxValue: 1000,
    step: 10,
    affects: ['learning'],
    increasingEffect:
      'A longer, gentler start. Needed for high peak learning rates and deep models.',
    decreasingEffect:
      'Reaches full speed sooner. With warmup at 0 and a high peak rate, the first few steps can damage the initialization beyond recovery.',
    advantages:
      'Adam estimates each parameter’s step size from a running average of past gradients. In the first few steps that average is based on almost no history, so the "adaptive" step can be large and nearly random. Warmup keeps steps tiny until the estimate is trustworthy.',
    disadvantages: 'Steps spent warming up are steps not spent at the productive peak rate.',
    exampleExperiment:
      'Set learning rate 0.01 with warmup 0, then with warmup 100. The first run often spikes or stalls in the first 20 steps; the second climbs smoothly.',
  },
  weightDecay: {
    name: 'Weight Decay (AdamW)',
    category: 'training',
    simpleExplanation:
      'A steady pull of every weight matrix toward zero on each step, so a weight only stays large if the gradient keeps pushing it there.',
    defaultValue: 0.01,
    minValue: 0,
    maxValue: 0.5,
    step: 0.005,
    affects: ['learning'],
    increasingEffect:
      'Stronger regularisation: a simpler, smoother model that generalises better but may underfit.',
    decreasingEffect:
      'The model is free to use large weights, carving sharp boundaries that fit the training text closely - and often memorise it.',
    advantages:
      'Applied "decoupled" (directly to the weights, not folded into the gradient) so that Adam’s per-parameter normalisation cannot cancel it. Biases and LayerNorm parameters are always exempt, because shrinking them does not reduce model complexity and fights normalisation.',
    disadvantages: 'Too much decay prevents the model from committing to anything.',
    exampleExperiment:
      'Train on a very short text (a few KB) with decay 0 and then 0.1. With 0 the train/val gap widens sharply - that gap is overfitting made visible.',
  },
  dropoutRate: {
    name: 'Dropout Rate',
    category: 'model',
    simpleExplanation:
      'During training, randomly zero this fraction of activations and scale up the survivors. Disabled automatically during evaluation and generation.',
    defaultValue: 0.1,
    minValue: 0,
    maxValue: 0.5,
    step: 0.05,
    affects: ['learning'],
    increasingEffect:
      'Stronger regularisation. Above about 0.3 a small model struggles to learn anything at all.',
    decreasingEffect:
      'Faster fitting of the training data, and a wider gap between training and validation loss.',
    advantages:
      'Breaks up fragile chains of co-adapted neurons (neuron A only works because neuron B always fires with it), forcing each one to be independently useful.',
    disadvantages:
      'Makes training loss look WORSE than the model really is, because it is measured with neurons missing. Always judge quality by validation loss.',
    exampleExperiment:
      'Train 1000 steps with dropout 0, then 0.1. Note that training loss is higher with dropout while validation loss is lower - that is regularisation working.',
  },
  tieWeights: {
    name: 'Tie Embedding & Output Weights',
    category: 'model',
    simpleExplanation:
      'Reuse one matrix for both the input token embedding and the output prediction head, instead of learning two separate ones.',
    defaultValue: true,
    affects: ['memory', 'learning'],
    increasingEffect:
      'Enabled: removes vocabSize x embeddingDim parameters and usually improves quality.',
    decreasingEffect:
      'Disabled: the head learns its own matrix, which gives more freedom but more parameters to fit from the same data.',
    advantages:
      'The input embedding maps id -> vector and the output head maps vector -> score per id. They are inverse views of the same vocabulary, so sharing them makes that relationship structural instead of something the model must learn twice.',
    disadvantages:
      'At word level with a large vocabulary, the shared matrix has to serve two roles at once; occasionally a dedicated head wins.',
    exampleExperiment:
      'Switch to the word-level tokenizer and compare the parameter count in the Architecture view with tying on and off - it often removes most of the model.',
  },
  repetitionPenalty: {
    name: 'Repetition Penalty',
    category: 'generation',
    simpleExplanation:
      'Divides the score of any token already present in the text, making the model progressively less willing to repeat itself.',
    defaultValue: 1.05,
    minValue: 1.0,
    maxValue: 2.0,
    step: 0.05,
    affects: ['generation'],
    increasingEffect:
      'Strongly discourages repeats. Above about 1.3 the text becomes unnaturally varied and starts avoiding common words like "the".',
    decreasingEffect: '1.0 disables it entirely.',
    advantages:
      'Fixes the single most visible failure of a small model: falling into a loop ("the the the"). Applied to raw scores before temperature, so its strength does not shift when you change temperature.',
    disadvantages:
      'Penalises legitimately frequent tokens too; at character level, spaces and vowels must repeat.',
    exampleExperiment:
      'Generate with temperature 0.3 and penalty 1.0 - the output will likely loop. Raise the penalty to 1.15 and the loop breaks.',
  },
  topP: {
    name: 'Top-P (Nucleus) Sampling',
    category: 'generation',
    simpleExplanation:
      'Keep only the most probable tokens whose probabilities add up to P (e.g. 0.95), then sample among those.',
    defaultValue: 0.95,
    minValue: 0.1,
    maxValue: 1.0,
    step: 0.01,
    affects: ['generation'],
    increasingEffect: 'Admits more of the tail. At 1.0 it has no effect.',
    decreasingEffect:
      'Restricts sampling to the few most likely tokens, approaching greedy decoding.',
    advantages:
      'Adapts to the shape of each distribution, which a fixed top-K cannot. Where the model is confident, top-K 40 admits 39 bad options; where it is genuinely uncertain, top-K 40 may cut off good ones. Top-P moves the cutoff per position.',
    disadvantages: 'Interacts with top-K - both filters apply, so the stricter one dominates.',
    exampleExperiment:
      'Set top-K to 0 (off) and compare top-P 0.5 against 0.99. The low value is clipped and safe; the high value takes real risks.',
  },
  randomSeed: {
    name: 'Random Seed',
    category: 'model',
    simpleExplanation:
      'The number that determines every "random" choice: initial weights, dropout masks and batch order. The same seed and config always produce exactly the same run.',
    defaultValue: 42,
    minValue: 0,
    maxValue: 999999,
    step: 1,
    affects: ['learning'],
    increasingEffect: 'Any different value gives a different but equally valid random draw.',
    decreasingEffect: 'Same - the value itself carries no meaning, only its reuse does.',
    advantages:
      'This is what makes the tool an instrument rather than a demo. Without a fixed seed, two runs differ for two reasons at once - your change AND the random draw - so no comparison can attribute the difference to anything.',
    disadvantages:
      'A single seed can flatter or punish a configuration by luck. For a conclusion you care about, run 3 seeds.',
    exampleExperiment:
      'Run the same config twice with seed 42 and confirm the loss curves are identical. Then change to 43 and see how much of the difference between two configs is just noise.',
  },
};

/**
 * Default CPU-friendly educational baseline configuration.
 * Designed to train smoothly on a standard Windows Intel i5 laptop in ~1-3 minutes.
 */
export const DEFAULT_CONFIG: FullConfig = {
  id: 'default-baseline',
  name: 'CPU Baseline',
  description:
    'A balanced character-level configuration that reaches a clearly sub-2.0 validation loss ' +
    'on Tiny Shakespeare within a few minutes on a consumer CPU.',
  createdAt: new Date().toISOString(),
  model: {
    vocabularySize: 65, // Auto-updated to match the loaded dataset
    embeddingDim: 128, // d_model
    numLayers: 4, // 4 Transformer blocks
    numHeads: 4, // 128 / 4 = 32 dimensions per head
    contextLength: 64, // 64 tokens of context history
    feedForwardMultiplier: 4, // 128 * 4 = 512 hidden units in each MLP
    activation: 'gelu',
    // A small amount of dropout. Tiny Shakespeare is ~1 MB and this model has
    // several hundred thousand parameters, so overfitting is a genuine risk;
    // 0.1 is the standard GPT-2 value and costs almost nothing.
    dropoutRate: 0.1,
    layerNormEpsilon: 1e-5,
    initializerScale: 0.02,
    randomSeed: 42,
    // Share the embedding matrix with the output head, as GPT-2 does.
    tieWeights: true,
  },
  training: {
    batchSize: 64,
    // Peak learning rate. With warmup and cosine decay a higher peak than the
    // usual "safe" 1e-3 is both usable and faster to converge.
    learningRate: 0.003,
    optimizerType: 'adam',
    adamBeta1: 0.9,
    adamBeta2: 0.99, // 0.99 rather than 0.999: short runs benefit from a shorter memory
    adamEpsilon: 1e-8,
    trainingSteps: 1000,
    evaluationInterval: 25,
    checkpointInterval: 250,
    validationBatchCount: 4,
    trainRatio: 0.8,
    valRatio: 0.1,
    testRatio: 0.1,
    gradientClipNorm: 1.0, // The standard Transformer value
    lrSchedule: 'cosine',
    warmupSteps: 60, // ~6% of the run, the usual proportion
    minLearningRate: 0.0003, // 10% of peak
    weightDecay: 0.01, // AdamW default; excluded from biases and LayerNorm
    earlyStoppingPatience: 0, // Off by default so a run completes as configured
    dataSeed: 1337,
  },
  generation: {
    temperature: 0.8,
    topK: 40,
    topP: 0.95,
    maxTokens: 300,
    repetitionPenalty: 1.05,
    samplePrompt: 'ROMEO:',
  },
};
