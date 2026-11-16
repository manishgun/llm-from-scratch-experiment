import * as tf from '@tensorflow/tfjs';
import { ModelConfig } from '../../shared/types';
import { CausalSelfAttention } from './attention';
import { Rng } from '../core/rng';
import { normalVariable, zerosVariable, onesVariable, residualStd } from '../core/init';
import { layerNorm, gelu, linear3D, dropout } from './layers';

/**
 * ============================================================================
 * MiniGPT Lab - Transformer Block
 * ============================================================================
 *
 * What is a Transformer Block?
 * A Transformer Block is the fundamental building brick of the GPT model.
 * Each block has two main sub-layers:
 * 1. Multi-Head Causal Self-Attention: lets tokens communicate with each other.
 * 2. Multi-Layer Perceptron (Feed-Forward Network): lets each token "think"
 *    and process the communicated information independently.
 *
 * A useful mental model: attention is where tokens EXCHANGE information, and
 * the MLP is where each token PROCESSES what it received, privately.
 *
 * Two Critical Modern GPT Architectural Principles:
 *
 * A. RESIDUAL CONNECTIONS (Skip Connections):
 *    output = input + sublayer(input)
 *    Instead of replacing the input entirely, each sub-layer only calculates a
 *    "delta" or adjustment to add to the existing representation.
 *    This allows gradients to flow smoothly through 50+ layers during
 *    backpropagation without vanishing: the derivative of `x + f(x)` with
 *    respect to `x` is `1 + f'(x)`, so there is always a path of gradient 1
 *    straight back to the input.
 *
 * B. PRE-LAYER NORMALIZATION (Pre-LN):
 *    In modern GPT (GPT-2, GPT-3, LLaMA), LayerNorm is applied BEFORE the
 *    sub-layer:
 *      x = x + Attention(LayerNorm(x))
 *      x = x + MLP(LayerNorm(x))
 *    This keeps the residual path itself completely un-normalized, which is
 *    what preserves the clean gradient highway described in (A). The original
 *    2017 Transformer used Post-LN and needed a learning-rate warmup to train
 *    at all; Pre-LN is markedly more forgiving.
 */
export class TransformerBlock {
  private config: ModelConfig;
  public attention: CausalSelfAttention;

  // LayerNorm 1 (before Attention)
  public ln1Gamma: tf.Variable;
  public ln1Beta: tf.Variable;

  // Feed-Forward Network (MLP) weights
  // First linear projection: expands D -> multiplier * D
  public mlpW1: tf.Variable;
  public mlpB1: tf.Variable;
  // Second linear projection: compresses multiplier * D -> D
  public mlpW2: tf.Variable;
  public mlpB2: tf.Variable;

  // LayerNorm 2 (before MLP)
  public ln2Gamma: tf.Variable;
  public ln2Beta: tf.Variable;

  constructor(config: ModelConfig, layerIndex: number, instanceId: string = '', rng?: Rng) {
    this.config = config;
    this.attention = new CausalSelfAttention(config, layerIndex, instanceId, rng);

    const d = config.embeddingDim;
    const hiddenDim = d * (config.feedForwardMultiplier || 4);
    const std = config.initializerScale || 0.02;
    const pfx = instanceId ? `${instanceId}_` : '';
    const base = `${pfx}layer_${layerIndex}`;

    const seedFor = (tag: string): Rng =>
      (rng ?? new Rng(config.randomSeed ?? 42)).derive(`L${layerIndex}_${tag}`);

    // LayerNorm 1 parameters (gamma initialized to 1.0, beta to 0.0), so the
    // layer starts as pure standardization with no learned distortion.
    this.ln1Gamma = onesVariable([d], `${base}_ln1_gamma`);
    this.ln1Beta = zerosVariable([d], `${base}_ln1_beta`);

    // MLP First Linear Layer: [d, hiddenDim]. Standard init scale.
    this.mlpW1 = normalVariable([d, hiddenDim], std, `${base}_mlp_w1`, seedFor('mlp_w1'));
    this.mlpB1 = zerosVariable([hiddenDim], `${base}_mlp_b1`);

    // MLP Second Linear Layer: [hiddenDim, d].
    // This writes into the residual stream, so it gets the GPT-2 depth-scaled
    // initialization, exactly like the attention output projection.
    this.mlpW2 = normalVariable(
      [hiddenDim, d],
      residualStd(std, config.numLayers),
      `${base}_mlp_w2`,
      seedFor('mlp_w2')
    );
    this.mlpB2 = zerosVariable([d], `${base}_mlp_b2`);

    // LayerNorm 2 parameters
    this.ln2Gamma = onesVariable([d], `${base}_ln2_gamma`);
    this.ln2Beta = zerosVariable([d], `${base}_ln2_beta`);
  }

  /**
   * Return all trainable parameters inside this block.
   */
  public getVariables(): tf.Variable[] {
    return [
      ...this.attention.getVariables(),
      this.ln1Gamma,
      this.ln1Beta,
      this.mlpW1,
      this.mlpB1,
      this.mlpW2,
      this.mlpB2,
      this.ln2Gamma,
      this.ln2Beta,
    ];
  }

  /**
   * Variables excluded from weight decay: biases and LayerNorm parameters.
   *
   * Decaying a LayerNorm gamma toward zero actively fights what LayerNorm is
   * for - it would shrink the normalized signal on every step. Decaying biases
   * has no regularizing benefit. Every serious Transformer implementation
   * excludes both.
   */
  public getNoDecayVariables(): tf.Variable[] {
    return [
      ...this.attention.getNoDecayVariables(),
      this.ln1Gamma,
      this.ln1Beta,
      this.ln2Gamma,
      this.ln2Beta,
      this.mlpB1,
      this.mlpB2,
    ];
  }

  /**
   * Forward pass through the Transformer Block.
   *
   * @param x Input tensor: [batchSize, seqLen, embeddingDim]
   * @returns Transformed tensor: [batchSize, seqLen, embeddingDim]
   */
  public forward(
    x: tf.Tensor,
    opts: { training?: boolean; rng?: Rng; captureAttention?: boolean } = {}
  ): tf.Tensor {
    const { training = false, rng, captureAttention = false } = opts;
    const dropRate = this.config.dropoutRate ?? 0;

    return tf.tidy(() => {
      // 1. Attention sub-layer with Pre-LN and a residual connection.
      const norm1 = layerNorm(x, this.ln1Gamma, this.ln1Beta, this.config.layerNormEpsilon);
      const attnOut = this.attention.forward(norm1, { training, rng, captureAttention });
      const xAfterAttn = tf.add(x, attnOut);

      // 2. MLP sub-layer with Pre-LN and a residual connection.
      const norm2 = layerNorm(
        xAfterAttn,
        this.ln2Gamma,
        this.ln2Beta,
        this.config.layerNormEpsilon
      );

      // MLP expansion: [B, T, D] -> [B, T, hiddenDim]
      const h1 = linear3D(norm2, this.mlpW1, this.mlpB1);

      // Non-linear activation (GELU or ReLU)
      const h1Act = this.config.activation === 'relu' ? tf.relu(h1) : gelu(h1);

      // MLP projection back: [B, T, hiddenDim] -> [B, T, D]
      const mlpOut = linear3D(h1Act, this.mlpW2, this.mlpB2);

      // Residual dropout before the addition, then the residual addition.
      const mlpDropped = dropout(mlpOut, dropRate, training, rng);
      return tf.add(xAfterAttn, mlpDropped);
    });
  }
}
