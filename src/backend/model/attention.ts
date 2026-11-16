import * as tf from '@tensorflow/tfjs';
import { ModelConfig } from '../../shared/types';
import { Rng } from '../core/rng';
import { normalVariable, zerosVariable, residualStd } from '../core/init';
import { stableSoftmax, linear3D, dropout } from './layers';

/**
 * ============================================================================
 * MiniGPT Lab - Causal Multi-Head Self-Attention
 * ============================================================================
 *
 * What is Self-Attention?
 * Self-attention is the core engine of the Transformer.
 * In a sentence, the meaning of a word depends on other words around it.
 * E.g., in "The bank of the river", "bank" means land by water.
 * In "The bank approved the loan", "bank" means a financial institution.
 * Self-attention lets each token look at every earlier token in the sequence,
 * calculate an "attention score" (relevance), and pull information from them.
 *
 * The Three Projections:
 * 1. Query (Q): "What am I looking for?"
 * 2. Key (K):   "What kind of information do I contain?"
 * 3. Value (V): "Here is my actual content/meaning to share."
 *
 * Scaled Dot-Product Attention Formula:
 * Attention(Q, K, V) = Softmax( (Q * K^T) / sqrt(d_k) + M ) * V
 *
 * What is the CAUSAL MASK (M)?
 * In language modeling, when predicting word #5, we must NOT look at words #6, #7, etc.!
 * The causal mask puts a large negative number in the upper triangle of the
 * attention matrix. When passed through Softmax, exp(-1e9) becomes 0.0,
 * strictly blocking future information.
 *
 * Multi-Head:
 * Instead of computing attention once with size D, we split D into `numHeads`
 * smaller heads of size `d_k = D / numHeads`. Each head learns a different kind
 * of relationship (e.g. head 1 tracks grammar, head 2 tracks character pairs,
 * head 3 tracks punctuation/quotes).
 */

/**
 * CAUSAL MASK CACHE
 * -----------------
 * The mask depends on exactly one thing: the sequence length T. It contains no
 * learned parameters and never changes during training.
 *
 * Building it naively means allocating a T-by-T Float32Array in a double JS
 * loop, then uploading it to a tensor - once per attention layer, per training
 * step. For a 6-layer model at T=128 running 1000 steps, that is 6,000
 * allocations of 16,384 floats (~98 million wasted writes) for a value that is
 * byte-identical every time.
 *
 * Caching by T reduces that to one allocation per distinct sequence length for
 * the whole process lifetime. The cache is bounded because T is bounded by
 * `contextLength`, and generation only ever uses T <= contextLength.
 */
const causalMaskCache = new Map<number, tf.Tensor4D>();

/** The additive penalty applied to disallowed (future) positions. */
const MASK_NEG_INFINITY = -1e9;

export class CausalSelfAttention {
  private config: ModelConfig;
  private headDim: number; // Size of each head: embeddingDim / numHeads

  // Trainable weight variables
  public wQuery: tf.Variable;
  public bQuery: tf.Variable;
  public wKey: tf.Variable;
  public bKey: tf.Variable;
  public wValue: tf.Variable;
  public bValue: tf.Variable;
  public wProj: tf.Variable;
  public bProj: tf.Variable;

  /**
   * The most recent attention probability tensor, captured only when the
   * caller asks for it via `forward(x, { captureAttention: true })`.
   *
   * This powers the Attention Explorer in the UI. It is deliberately opt-in:
   * holding a [B, numHeads, T, T] tensor alive on every training step would be
   * a significant and pointless memory cost.
   */
  public lastAttentionWeights: tf.Tensor4D | null = null;

  constructor(config: ModelConfig, layerIndex: number, instanceId: string = '', rng?: Rng) {
    this.config = config;

    if (config.embeddingDim % config.numHeads !== 0) {
      throw new Error(
        `embeddingDim (${config.embeddingDim}) must be divisible by numHeads (${config.numHeads})`
      );
    }

    this.headDim = config.embeddingDim / config.numHeads;
    const std = config.initializerScale || 0.02;
    const d = config.embeddingDim;
    const pfx = instanceId ? `${instanceId}_` : '';
    const base = `${pfx}layer_${layerIndex}_attn`;

    // Each tensor draws from its own derived stream, so that adding a layer
    // does not shift the random numbers used by the layers around it. See
    // `Rng.derive` for why that makes depth experiments cleaner.
    const seedFor = (tag: string): Rng =>
      (rng ?? new Rng(config.randomSeed ?? 42)).derive(`L${layerIndex}_attn_${tag}`);

    // Q, K, V projections: standard init scale.
    this.wQuery = normalVariable([d, d], std, `${base}_w_q`, seedFor('q'));
    this.bQuery = zerosVariable([d], `${base}_b_q`);

    this.wKey = normalVariable([d, d], std, `${base}_w_k`, seedFor('k'));
    this.bKey = zerosVariable([d], `${base}_b_k`);

    this.wValue = normalVariable([d, d], std, `${base}_w_v`, seedFor('v'));
    this.bValue = zerosVariable([d], `${base}_b_v`);

    // OUTPUT PROJECTION: this layer writes into the residual stream, so it gets
    // the GPT-2 depth-scaled initialization (std / sqrt(2 * numLayers)).
    // Without this, activation variance grows linearly with depth and deep
    // configurations become unstable. See `core/init.ts`.
    this.wProj = normalVariable(
      [d, d],
      residualStd(std, config.numLayers),
      `${base}_w_out`,
      seedFor('out')
    );
    this.bProj = zerosVariable([d], `${base}_b_out`);
  }

  /**
   * Return all trainable parameters in this attention layer.
   */
  public getVariables(): tf.Variable[] {
    return [
      this.wQuery,
      this.bQuery,
      this.wKey,
      this.bKey,
      this.wValue,
      this.bValue,
      this.wProj,
      this.bProj,
    ];
  }

  /**
   * Names of variables that should NOT receive weight decay.
   * Biases have no capacity to overfit on their own (they shift, they do not
   * scale), and decaying them measurably hurts. Standard practice is to decay
   * matrices only.
   */
  public getNoDecayVariables(): tf.Variable[] {
    return [this.bQuery, this.bKey, this.bValue, this.bProj];
  }

  /**
   * Builds (or returns a cached) upper-triangular causal attention mask of
   * shape [1, 1, seqLen, seqLen]. Positions where col > row hold -1e9;
   * allowed positions hold 0.0.
   *
   * The returned tensor is OWNED BY THE CACHE. Callers must not dispose it,
   * and must not use it inside a `tf.tidy` that would reclaim it - which is
   * why `forward` clones it before use.
   */
  public static createCausalMask(seqLen: number): tf.Tensor4D {
    const cached = causalMaskCache.get(seqLen);
    if (cached && !cached.isDisposed) {
      return cached;
    }

    const buffer = new Float32Array(seqLen * seqLen);
    for (let r = 0; r < seqLen; r++) {
      for (let c = 0; c < seqLen; c++) {
        // If column > row, this is a future token: MASK IT OUT.
        buffer[r * seqLen + c] = c > r ? MASK_NEG_INFINITY : 0.0;
      }
    }

    // `tf.keep` protects the tensor from enclosing tidy() scopes, so the cache
    // entry survives the training step that created it.
    const mask = tf.keep(tf.tensor4d(buffer, [1, 1, seqLen, seqLen]));
    causalMaskCache.set(seqLen, mask);
    return mask;
  }

  /** Releases every cached mask. Called on process teardown and in tests. */
  public static clearMaskCache(): void {
    for (const mask of causalMaskCache.values()) {
      if (!mask.isDisposed) mask.dispose();
    }
    causalMaskCache.clear();
  }

  /**
   * Forward pass of Causal Multi-Head Self-Attention.
   *
   * @param x Input tensor of shape [batchSize, seqLen, embeddingDim]
   * @param opts.training          Enables attention dropout
   * @param opts.rng               Seeded generator for the dropout mask
   * @param opts.captureAttention  Retain the attention probabilities for the UI
   * @returns Output tensor of shape [batchSize, seqLen, embeddingDim]
   */
  public forward(
    x: tf.Tensor,
    opts: { training?: boolean; rng?: Rng; captureAttention?: boolean } = {}
  ): tf.Tensor {
    const { training = false, rng, captureAttention = false } = opts;

    // Release any previously captured map before overwriting the field, so
    // repeated inspection calls do not leak tensors.
    if (this.lastAttentionWeights && !this.lastAttentionWeights.isDisposed) {
      this.lastAttentionWeights.dispose();
      this.lastAttentionWeights = null;
    }

    return tf.tidy(() => {
      const [B, T, C] = x.shape as [number, number, number];
      const nh = this.config.numHeads;
      const hs = this.headDim; // hs * nh = C
      const dropRate = this.config.dropoutRate ?? 0;

      // 1. Linearly project input x into Queries, Keys, and Values.
      // Using linear3D for robust tensor backpropagation in TensorFlow.js.
      const q = linear3D(x, this.wQuery, this.bQuery);
      const k = linear3D(x, this.wKey, this.bKey);
      const v = linear3D(x, this.wValue, this.bValue);

      // 2. Reshape and transpose to separate heads:
      // [B, T, C] -> [B, T, nh, hs] -> transpose -> [B, nh, T, hs]
      const qReshaped = tf.transpose(tf.reshape(q, [B, T, nh, hs]), [0, 2, 1, 3]);
      const kReshaped = tf.transpose(tf.reshape(k, [B, T, nh, hs]), [0, 2, 1, 3]);
      const vReshaped = tf.transpose(tf.reshape(v, [B, T, nh, hs]), [0, 2, 1, 3]);

      // 3. Compute raw dot-product attention scores:
      // (Q * K^T) -> shape [B, nh, T, T]
      const kTranspose = tf.transpose(kReshaped, [0, 1, 3, 2]);
      const scores = tf.matMul(qReshaped, kTranspose);

      // 4. Scale by sqrt(headDim).
      // Without this, the dot product of two d_k-dimensional vectors has
      // variance proportional to d_k, so the scores grow with head size and
      // push softmax into a regime where one entry is ~1 and the rest ~0. The
      // gradient there is nearly zero, and attention stops learning.
      const scaleFactor = tf.scalar(1.0 / Math.sqrt(hs));
      const scaledScores = tf.mul(scores, scaleFactor);

      // 5. Apply the causal mask. The cached tensor is cloned so this tidy
      // scope does not reclaim the shared cache entry.
      const mask = CausalSelfAttention.createCausalMask(T).clone();
      const maskedScores = tf.add(scaledScores, mask);

      // 6. Convert scores to attention probabilities: [B, nh, T, T]
      const attnProbs = stableSoftmax(maskedScores, -1);

      // Hand a copy to the UI if asked. `tf.keep` lifts it out of this tidy
      // scope; the next forward() call or dispose() frees it.
      if (captureAttention) {
        this.lastAttentionWeights = tf.keep(attnProbs.clone()) as tf.Tensor4D;
      }

      // 6b. ATTENTION DROPOUT: randomly drop entire attention edges, so the
      // model cannot rely on one specific token-to-token link always existing.
      const attnProbsDropped = dropout(attnProbs, dropRate, training, rng);

      // 7. Multiply attention weights by Values (V):
      // [B, nh, T, T] * [B, nh, T, hs] -> [B, nh, T, hs]
      const weightedValues = tf.matMul(attnProbsDropped, vReshaped);

      // 8. Recombine heads:
      // [B, nh, T, hs] -> transpose -> [B, T, nh, hs] -> reshape -> [B, T, C]
      const recombined = tf.reshape(tf.transpose(weightedValues, [0, 2, 1, 3]), [B, T, C]);

      // 9. Final linear output projection, then residual dropout.
      const projected = linear3D(recombined, this.wProj, this.bProj);
      return dropout(projected, dropRate, training, rng);
    });
  }

  /** Frees the captured attention map, if any. */
  public disposeCaptured(): void {
    if (this.lastAttentionWeights && !this.lastAttentionWeights.isDisposed) {
      this.lastAttentionWeights.dispose();
    }
    this.lastAttentionWeights = null;
  }
}
