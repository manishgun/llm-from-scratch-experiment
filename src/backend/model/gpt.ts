import * as tf from '@tensorflow/tfjs';
import { ModelConfig } from '../../shared/types';
import { TransformerBlock } from './transformerBlock';
import { Rng } from '../core/rng';
import { normalVariable, zerosVariable, onesVariable } from '../core/init';
import { layerNorm, linear3D, linear3DTransposed, dropout } from './layers';

/**
 * ============================================================================
 * MiniGPT Lab - GPT Language Model Architecture
 * ============================================================================
 *
 * This is the complete GPT (Generative Pre-trained Transformer) decoder-only model.
 *
 * Full Forward Pass Journey of a Token Sequence:
 * 1. INPUT: Sequence of token integer IDs: [batchSize, seqLen]
 * 2. TOKEN EMBEDDING: Looks up a vector representation for each token:
 *    [batchSize, seqLen, embeddingDim]
 * 3. POSITIONAL EMBEDDING: Adds a vector for each position (0, 1, 2, ...):
 *    x = tokenEmbedding + posEmbedding
 *    This is necessary because attention itself is permutation-invariant - it
 *    sees a SET of tokens, not a sequence. Without position information,
 *    "dog bites man" and "man bites dog" would be identical to the model.
 * 4. TRANSFORMER BLOCKS: Passes through N stacked Transformer blocks
 *    (Causal Self-Attention + MLP with residual connections & LayerNorm)
 * 5. FINAL LAYERNORM: Stabilizes output representations before projection
 * 6. LOGITS PROJECTION: Linear layer mapping embeddingDim -> vocabularySize:
 *    [batchSize, seqLen, vocabularySize]
 *    Each number (logit) is the raw unnormalized preference for that token.
 */

/** Options accepted by the forward pass. */
export interface ForwardOptions {
  /** Enables dropout. Must be false for evaluation and generation. */
  training?: boolean;
  /** Seeded generator used for dropout masks, so runs are reproducible. */
  rng?: Rng;
  /** Retain per-layer attention probabilities for the UI's Attention Explorer. */
  captureAttention?: boolean;
}

/** Shape of the JSON header embedded at the start of a `.mgpt` checkpoint. */
export interface CheckpointHeader {
  magic: 'MGPT';
  version: number;
  savedAt: string;
  step: number;
  config: ModelConfig;
  tokenizer?: { type: 'char' | 'word'; vocab: string[] };
  metrics?: { trainLoss?: number; valLoss?: number };
  variables: { name: string; shape: number[]; byteOffset: number; byteLength: number }[];
  optimizerState?: { shape: number[]; byteOffset: number; byteLength: number }[];
}

/** First four bytes of every checkpoint file, used to detect the format. */
const CHECKPOINT_MAGIC = 'MGPT';
/** Bumped whenever the on-disk layout changes incompatibly. */
const CHECKPOINT_VERSION = 2;

export class MiniGPT {
  public config: ModelConfig;
  public instanceId: string;

  /** The seeded generator this model was built from. */
  public readonly rng: Rng;

  // Learnable Embeddings
  /** Token Embedding Table: shape [vocabularySize, embeddingDim] */
  public tokenEmbeddingTable: tf.Variable;
  /** Positional Embedding Table: shape [contextLength, embeddingDim] */
  public positionEmbeddingTable: tf.Variable;

  // Stack of Transformer Blocks
  public blocks: TransformerBlock[];

  // Final LayerNorm before output projection
  public finalLnGamma: tf.Variable;
  public finalLnBeta: tf.Variable;

  /**
   * Final linear projection from embeddingDim -> vocabularySize (logits).
   *
   * When `tieWeights` is enabled this is `null`, and the token embedding table
   * is transposed and reused instead. See the note on weight tying below.
   */
  public headWeight: tf.Variable | null;
  public headBias: tf.Variable;

  /**
   * WEIGHT TYING
   * ------------
   * The input embedding maps token_id -> vector. The output head maps
   * vector -> score for each token_id. These are inverse operations over the
   * same vocabulary, and there is strong evidence (Press & Wolf 2017, and
   * GPT-2's own implementation) that forcing them to share one matrix both
   * reduces parameters and IMPROVES quality.
   *
   * The intuition: if token "q" and token "Q" should have similar input
   * embeddings, they should also receive similar output scores in similar
   * contexts. Tying makes that constraint structural rather than something the
   * model must rediscover twice.
   *
   * The saving is real at character level and large at word level: the two
   * matrices are each vocabSize x embeddingDim. With a 12,000-word vocabulary
   * and embeddingDim 128, tying removes 1.5 million parameters - often most of
   * the model.
   */
  public readonly tieWeights: boolean;

  constructor(config: ModelConfig, instanceId?: string) {
    this.config = config;
    this.instanceId = instanceId || `gpt_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
    this.tieWeights = config.tieWeights ?? true;

    const std = config.initializerScale || 0.02;
    const V = config.vocabularySize;
    const D = config.embeddingDim;
    const T = config.contextLength;
    const pfx = `${this.instanceId}_`;

    // One master generator derived from the configured seed. Every tensor pulls
    // from a named sub-stream so the initialization is fully reproducible and
    // stable under architecture changes.
    this.rng = new Rng(config.randomSeed ?? 42);

    this.tokenEmbeddingTable = normalVariable(
      [V, D],
      std,
      `${pfx}token_embedding`,
      this.rng.derive('token_embedding')
    );

    this.positionEmbeddingTable = normalVariable(
      [T, D],
      std,
      `${pfx}pos_embedding`,
      this.rng.derive('pos_embedding')
    );

    // Initialize stacked Transformer blocks.
    this.blocks = [];
    for (let i = 0; i < config.numLayers; i++) {
      this.blocks.push(new TransformerBlock(config, i, this.instanceId, this.rng));
    }

    // Final LayerNorm parameters.
    this.finalLnGamma = onesVariable([D], `${pfx}final_ln_gamma`);
    this.finalLnBeta = zerosVariable([D], `${pfx}final_ln_beta`);

    // Language Model Head. Omitted entirely when weights are tied.
    this.headWeight = this.tieWeights
      ? null
      : normalVariable([D, V], std, `${pfx}lm_head_w`, this.rng.derive('lm_head_w'));
    this.headBias = zerosVariable([V], `${pfx}lm_head_b`);
  }

  /**
   * Returns a flat array of ALL trainable variables across the entire model.
   * Every one of these receives gradients during backpropagation.
   *
   * Note that when weights are tied, `tokenEmbeddingTable` appears exactly
   * once even though it is used in two places. Listing it twice would make the
   * optimizer apply its update twice per step - effectively doubling the
   * learning rate for that one tensor.
   */
  public getTrainableVariables(): tf.Variable[] {
    const vars: tf.Variable[] = [this.tokenEmbeddingTable, this.positionEmbeddingTable];

    for (const block of this.blocks) {
      vars.push(...block.getVariables());
    }

    vars.push(this.finalLnGamma, this.finalLnBeta, this.headBias);
    if (this.headWeight) vars.push(this.headWeight);
    return vars;
  }

  /**
   * Variables that must be excluded from weight decay: all biases and all
   * LayerNorm scale/shift parameters. See `TransformerBlock.getNoDecayVariables`.
   */
  public getNoDecayVariables(): tf.Variable[] {
    const vars: tf.Variable[] = [];
    for (const block of this.blocks) {
      vars.push(...block.getNoDecayVariables());
    }
    vars.push(this.finalLnGamma, this.finalLnBeta, this.headBias);
    return vars;
  }

  /**
   * Calculates the total number of trainable floating-point parameters.
   */
  public countParameters(): number {
    let count = 0;
    for (const v of this.getTrainableVariables()) {
      count += v.size;
    }
    return count;
  }

  /**
   * A per-component parameter breakdown, for the Architecture view in the UI.
   * Seeing that 60% of a word-level model's parameters live in the embedding
   * table is one of the more instructive things this tool can show.
   */
  public getParameterBreakdown(): { component: string; parameters: number }[] {
    const D = this.config.embeddingDim;
    const V = this.config.vocabularySize;
    const T = this.config.contextLength;
    const hidden = D * (this.config.feedForwardMultiplier || 4);
    const perBlockAttn = 4 * (D * D + D); // Q, K, V, output projection + biases
    const perBlockMlp = D * hidden + hidden + hidden * D + D;
    const perBlockLn = 4 * D; // two LayerNorms, gamma + beta each

    const breakdown = [
      { component: 'Token embeddings', parameters: V * D },
      { component: 'Position embeddings', parameters: T * D },
      { component: 'Attention (all layers)', parameters: perBlockAttn * this.config.numLayers },
      { component: 'Feed-forward (all layers)', parameters: perBlockMlp * this.config.numLayers },
      { component: 'LayerNorm (all layers)', parameters: perBlockLn * this.config.numLayers },
      { component: 'Final LayerNorm', parameters: 2 * D },
    ];

    breakdown.push({
      component: this.tieWeights ? 'LM head (tied, bias only)' : 'LM head',
      parameters: this.tieWeights ? V : D * V + V,
    });

    return breakdown;
  }

  /**
   * Forward pass: computes logits for an input sequence of tokens.
   *
   * @param idx Input token IDs: 2D tensor of shape [batchSize, seqLen]
   * @returns Unnormalized logits: 3D tensor of shape [batchSize, seqLen, vocabularySize]
   */
  public forward(idx: tf.Tensor2D, opts: ForwardOptions = {}): tf.Tensor3D {
    const { training = false, rng, captureAttention = false } = opts;
    const dropRate = this.config.dropoutRate ?? 0;

    return tf.tidy(() => {
      const [, T] = idx.shape as [number, number];

      if (T > this.config.contextLength) {
        throw new Error(
          `Sequence length ${T} exceeds maximum model contextLength ${this.config.contextLength}`
        );
      }

      // 1. Token embeddings: look up token vectors -> [B, T, D].
      // tf.gather pulls the rows corresponding to each token ID.
      const tokEmb = tf.gather(this.tokenEmbeddingTable, idx);

      // 2. Positional embeddings: positions [0, 1, ..., T - 1] -> [T, D]
      const posIndices = tf.range(0, T, 1, 'int32');
      const posEmb = tf.gather(this.positionEmbeddingTable, posIndices);

      // 3. Combine. TensorFlow broadcasts posEmb [T, D] across the batch.
      let x: tf.Tensor = tf.add(tokEmb, posEmb);

      // 3b. Embedding dropout, applied to the sum as GPT-2 does.
      x = dropout(x, dropRate, training, rng);

      // 4. Pass through each Transformer block in sequence.
      for (const block of this.blocks) {
        x = block.forward(x, { training, rng, captureAttention });
      }

      // 5. Final layer normalization.
      x = layerNorm(x, this.finalLnGamma, this.finalLnBeta, this.config.layerNormEpsilon);

      // 6. Project from embeddingDim (D) to vocabularySize (V): [B, T, D] -> [B, T, V].
      // With tied weights we reuse the [V, D] embedding table transposed,
      // which is mathematically identical to a dedicated [D, V] matrix.
      const logits = this.headWeight
        ? linear3D(x, this.headWeight, this.headBias)
        : linear3DTransposed(x, this.tokenEmbeddingTable, this.headBias);

      return logits as tf.Tensor3D;
    });
  }

  /**
   * Runs a forward pass and returns the per-layer attention probability maps
   * alongside the logits. Used by the Attention Explorer.
   *
   * @returns `attention[layer]` has shape [batch, numHeads, seqLen, seqLen]
   */
  public forwardWithAttention(idx: tf.Tensor2D): {
    logits: tf.Tensor3D;
    attention: tf.Tensor4D[];
  } {
    const logits = this.forward(idx, { training: false, captureAttention: true });

    const attention: tf.Tensor4D[] = [];
    for (const block of this.blocks) {
      const weights = block.attention.lastAttentionWeights;
      if (weights && !weights.isDisposed) {
        attention.push(tf.keep(weights.clone()) as tf.Tensor4D);
      }
    }

    // Release the in-model captures now that we have independent copies.
    for (const block of this.blocks) {
      block.attention.disposeCaptured();
    }

    return { logits, attention };
  }

  /**
   * Computes Cross-Entropy Loss between predictions (logits) and target tokens.
   *
   * Formula:
   *   Loss = - (1 / N) * sum( log( P(target_token) ) )
   *
   * INTERPRETING THE NUMBER
   * A model that has learned nothing assigns uniform probability 1/V to every
   * token, giving a loss of ln(V). For a 65-character vocabulary that is
   * ln(65) = 4.17 - which is exactly where training starts. Any value below
   * that means the model has learned something real. `exp(loss)` is the
   * perplexity: the effective number of tokens the model is still choosing
   * between.
   *
   * @param logits Shape [batchSize, seqLen, vocabularySize]
   * @param targets Shape [batchSize, seqLen] containing correct next-token IDs
   * @returns Scalar loss tensor
   */
  public computeLoss(logits: tf.Tensor3D, targets: tf.Tensor2D): tf.Scalar {
    return tf.tidy(() => {
      const [B, T, V] = logits.shape;

      // Reshape logits to 2D [B * T, V] and targets to 1D [B * T].
      const flatLogits = tf.reshape(logits, [B * T, V]);
      const flatTargets = tf.reshape(targets, [B * T]);

      // One-hot encode targets: [B * T, V]
      const oneHotTargets = tf.oneHot(flatTargets, V);

      // softmaxCrossEntropy fuses softmax and the log, handling log-sum-exp
      // stably - computing them separately would overflow for large logits.
      const sampleLosses = tf.losses.softmaxCrossEntropy(oneHotTargets, flatLogits);

      return tf.mean(sampleLosses) as tf.Scalar;
    });
  }

  /**
   * ------------------------------------------------------------------------
   * CHECKPOINT PERSISTENCE (binary `.mgpt` format)
   * ------------------------------------------------------------------------
   *
   * WHY NOT JSON
   * The previous format stored every weight as a decimal string inside a JSON
   * array. A float32 occupies 4 bytes in memory but 8-20 ASCII bytes as text,
   * so checkpoints were 3-5x larger than the data they held, and `JSON.parse`
   * on a multi-million-element number array is slow and allocates a boxed
   * JS number per weight.
   *
   * THE LAYOUT
   *   bytes 0-3    magic "MGPT"
   *   bytes 4-7    header length, uint32 little-endian
   *   bytes 8-N    UTF-8 JSON header (config, tokenizer, variable directory)
   *   bytes N-end  tightly packed float32 payload, little-endian
   *
   * This is the same shape as safetensors or TFJS's own weight manifest: a
   * self-describing header plus one contiguous binary blob. It means a loader
   * can read the header alone to inspect a checkpoint's architecture without
   * touching the weights.
   *
   * WHAT ELSE IS STORED
   * A checkpoint also carries the TOKENIZER VOCABULARY and the STEP COUNT.
   * Without the vocabulary, a checkpoint is unusable: token ID 42 means
   * nothing unless you know which dataset built the mapping. Loading a
   * Shakespeare checkpoint against a Moby Dick vocabulary silently produces
   * garbage, and that is precisely the kind of bug that wastes an afternoon.
   */
  public async saveCheckpoint(
    filePath: string,
    extras: {
      step?: number;
      tokenizer?: { type: 'char' | 'word'; vocab: string[] };
      metrics?: { trainLoss?: number; valLoss?: number };
      optimizerState?: tf.Tensor[];
    } = {}
  ): Promise<void> {
    const fs = await import('fs');
    const path = await import('path');

    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    const prefix = `${this.instanceId}_`;
    const chunks: Float32Array[] = [];
    const variables: CheckpointHeader['variables'] = [];
    let offset = 0;

    for (const variable of this.getTrainableVariables()) {
      const values = (await variable.data()) as Float32Array;
      // Strip the instanceId prefix so a checkpoint loads into any instance.
      const cleanName = variable.name.startsWith(prefix)
        ? variable.name.substring(prefix.length)
        : variable.name;

      const byteLength = values.length * 4;
      variables.push({ name: cleanName, shape: [...variable.shape], byteOffset: offset, byteLength });
      chunks.push(values);
      offset += byteLength;
    }

    // Adam's first and second moment estimates. Saving them is what makes
    // "resume training" genuinely resume: restoring only the weights restarts
    // the optimizer's momentum from zero, which produces a visible loss spike
    // and discards the adaptive per-parameter step sizes it had learned.
    const optimizerState: CheckpointHeader['optimizerState'] = [];
    if (extras.optimizerState) {
      for (const t of extras.optimizerState) {
        const values = (await t.data()) as Float32Array;
        const byteLength = values.length * 4;
        optimizerState.push({ shape: [...t.shape], byteOffset: offset, byteLength });
        chunks.push(values);
        offset += byteLength;
      }
    }

    const header: CheckpointHeader = {
      magic: CHECKPOINT_MAGIC,
      version: CHECKPOINT_VERSION,
      savedAt: new Date().toISOString(),
      step: extras.step ?? 0,
      config: this.config,
      tokenizer: extras.tokenizer,
      metrics: extras.metrics,
      variables,
      optimizerState: optimizerState.length > 0 ? optimizerState : undefined,
    };

    const headerJson = Buffer.from(JSON.stringify(header), 'utf-8');
    const prelude = Buffer.alloc(8);
    prelude.write(CHECKPOINT_MAGIC, 0, 'ascii');
    prelude.writeUInt32LE(headerJson.length, 4);

    // Concatenate every weight buffer into one contiguous payload.
    const payload = Buffer.alloc(offset);
    let writeAt = 0;
    for (const chunk of chunks) {
      Buffer.from(chunk.buffer, chunk.byteOffset, chunk.byteLength).copy(payload, writeAt);
      writeAt += chunk.byteLength;
    }

    fs.writeFileSync(filePath, Buffer.concat([prelude, headerJson, payload]));
  }

  /**
   * Reads only the JSON header of a checkpoint, without loading any weights.
   * Lets the UI list a checkpoint's architecture, step and loss cheaply.
   */
  public static async readCheckpointHeader(filePath: string): Promise<CheckpointHeader> {
    const fs = await import('fs');
    const handle = await fs.promises.open(filePath, 'r');
    try {
      const prelude = Buffer.alloc(8);
      await handle.read(prelude, 0, 8, 0);

      if (prelude.toString('ascii', 0, 4) !== CHECKPOINT_MAGIC) {
        throw new Error(
          `Not a MiniGPT checkpoint (bad magic bytes). Legacy JSON checkpoints are no longer supported; retrain or convert.`
        );
      }

      const headerLength = prelude.readUInt32LE(4);
      const headerBuf = Buffer.alloc(headerLength);
      await handle.read(headerBuf, 0, headerLength, 8);
      return JSON.parse(headerBuf.toString('utf-8')) as CheckpointHeader;
    } finally {
      await handle.close();
    }
  }

  /**
   * Load model weights from a `.mgpt` checkpoint file on disk.
   *
   * Validates that the checkpoint's architecture matches this model before
   * assigning anything. A silent shape mismatch would otherwise leave the
   * model in a half-loaded state that is very hard to diagnose.
   */
  public async loadCheckpoint(filePath: string): Promise<CheckpointHeader> {
    const fs = await import('fs');

    if (!fs.existsSync(filePath)) {
      throw new Error(`Checkpoint file does not exist: ${filePath}`);
    }

    const file = fs.readFileSync(filePath);

    if (file.toString('ascii', 0, 4) !== CHECKPOINT_MAGIC) {
      throw new Error('Not a MiniGPT checkpoint (bad magic bytes).');
    }

    const headerLength = file.readUInt32LE(4);
    const header = JSON.parse(
      file.toString('utf-8', 8, 8 + headerLength)
    ) as CheckpointHeader;
    const payloadStart = 8 + headerLength;

    // Fail loudly on an architecture mismatch rather than loading what fits.
    const mismatches: string[] = [];
    const checks: (keyof ModelConfig)[] = [
      'vocabularySize',
      'embeddingDim',
      'numLayers',
      'numHeads',
      'contextLength',
      'feedForwardMultiplier',
    ];
    for (const key of checks) {
      if (header.config[key] !== this.config[key]) {
        mismatches.push(`${key}: checkpoint=${header.config[key]} model=${this.config[key]}`);
      }
    }
    if (mismatches.length > 0) {
      throw new Error(
        `Checkpoint architecture does not match the current model.\n  ${mismatches.join('\n  ')}\n` +
          `Load this checkpoint's configuration first, then load its weights.`
      );
    }

    const prefix = `${this.instanceId}_`;
    const varMap = new Map<string, tf.Variable>();
    for (const v of this.getTrainableVariables()) {
      const cleanName = v.name.startsWith(prefix) ? v.name.substring(prefix.length) : v.name;
      varMap.set(cleanName, v);
    }

    let assigned = 0;
    for (const entry of header.variables) {
      const target = varMap.get(entry.name);
      if (!target) continue;

      const start = payloadStart + entry.byteOffset;
      // Copy into a fresh buffer: the file Buffer's byteOffset is not
      // guaranteed to be 4-byte aligned, which Float32Array requires.
      const bytes = new Uint8Array(entry.byteLength);
      file.copy(bytes, 0, start, start + entry.byteLength);
      const values = new Float32Array(bytes.buffer);

      tf.tidy(() => {
        target.assign(tf.tensor(values, entry.shape, 'float32'));
      });
      assigned++;
    }

    if (assigned === 0) {
      throw new Error('Checkpoint contained no variables matching this model.');
    }

    return header;
  }

  /**
   * Reads the Adam optimizer state tensors out of a checkpoint, if present.
   * Returned tensors are owned by the caller.
   */
  public static async readOptimizerState(filePath: string): Promise<tf.Tensor[]> {
    const fs = await import('fs');
    const file = fs.readFileSync(filePath);
    const headerLength = file.readUInt32LE(4);
    const header = JSON.parse(file.toString('utf-8', 8, 8 + headerLength)) as CheckpointHeader;
    const payloadStart = 8 + headerLength;

    if (!header.optimizerState) return [];

    return header.optimizerState.map((entry) => {
      const start = payloadStart + entry.byteOffset;
      const bytes = new Uint8Array(entry.byteLength);
      file.copy(bytes, 0, start, start + entry.byteLength);
      return tf.tensor(new Float32Array(bytes.buffer), entry.shape, 'float32');
    });
  }

  /**
   * Dispose all tensor variables from memory.
   */
  public dispose(): void {
    for (const v of this.getTrainableVariables()) {
      if (!v.isDisposed) v.dispose();
    }
    for (const block of this.blocks) {
      block.attention.disposeCaptured();
    }
  }
}
