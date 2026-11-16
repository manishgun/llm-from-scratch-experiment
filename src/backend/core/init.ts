import * as tf from '@tensorflow/tfjs';
import { Rng } from './rng';

/**
 * ============================================================================
 * MiniGPT Lab - Weight Initialization Strategies
 * ============================================================================
 *
 * WHY INITIALIZATION MATTERS
 * --------------------------
 * A deep network is a long chain of matrix multiplications. If each weight
 * matrix slightly *amplifies* the signal passing through it, then after N
 * layers the activations are amplified N times over — they explode. If each
 * one slightly shrinks the signal, the activations vanish to zero and no
 * gradient reaches the early layers.
 *
 * Good initialization picks a variance that keeps the signal's magnitude
 * roughly constant as it flows forward, and keeps the gradient's magnitude
 * roughly constant as it flows backward.
 *
 * THE GPT-2 RESIDUAL SCALING TRICK
 * --------------------------------
 * A Transformer is special because of residual connections:
 *
 *     x = x + attention(ln(x))
 *     x = x + mlp(ln(x))
 *
 * Every block ADDS to a running sum. With `numLayers` blocks, each
 * contributing a unit-variance term, the variance of `x` at the output is
 * roughly `2 * numLayers` times the input variance (two residual branches per
 * block). The deeper the model, the larger the activations at the top — which
 * is exactly the instability that makes deep Transformers hard to train.
 *
 * GPT-2's fix (Radford et al. 2019, section 2.3) is to scale down the weights
 * of the layers that WRITE INTO the residual stream — the attention output
 * projection and the MLP's second linear layer — by `1 / sqrt(2 * numLayers)`.
 * Each block then contributes a correspondingly smaller amount, and the total
 * variance stays near 1 regardless of depth.
 *
 * This is the single highest-value change for anyone experimenting with
 * `numLayers`: without it, a 12-layer model is markedly less stable than a
 * 2-layer one, and the user wrongly concludes "deep models just don't work
 * here".
 */

/**
 * Creates a trainable variable filled with N(0, stdDev) values, drawn from a
 * seeded generator so the result is bit-for-bit reproducible.
 *
 * @param shape    Tensor shape, e.g. [inDim, outDim]
 * @param stdDev   Standard deviation of the Gaussian
 * @param name     Variable name (used as the checkpoint key)
 * @param rng      Seeded generator providing the random numbers
 */
export function normalVariable(
  shape: number[],
  stdDev: number,
  name: string,
  rng: Rng
): tf.Variable {
  const size = shape.reduce((a, b) => a * b, 1);
  const values = rng.normalArray(size, 0, stdDev);
  // `tf.tensor` takes ownership of the buffer; tf.variable then copies it into
  // a trainable variable. Wrapped in tidy so the intermediate tensor is freed.
  return tf.tidy(() => tf.variable(tf.tensor(values, shape, 'float32'), true, name));
}

/**
 * Creates a trainable variable filled entirely with zeros.
 * Standard for biases: there is no symmetry to break in a bias term, and a
 * zero bias means the layer starts as a pure linear map of its input.
 */
export function zerosVariable(shape: number[], name: string): tf.Variable {
  return tf.tidy(() => tf.variable(tf.zeros(shape), true, name));
}

/**
 * Creates a trainable variable filled entirely with ones.
 * Standard for LayerNorm's `gamma` (scale): starting at 1.0 means LayerNorm
 * initially performs pure standardization with no learned rescaling.
 */
export function onesVariable(shape: number[], name: string): tf.Variable {
  return tf.tidy(() => tf.variable(tf.ones(shape), true, name));
}

/**
 * The standard deviation to use for a layer that writes into the residual
 * stream, per the GPT-2 scaling rule described at the top of this file.
 *
 * @param baseStd   The model's nominal init scale (usually 0.02)
 * @param numLayers How many Transformer blocks the model has
 */
export function residualStd(baseStd: number, numLayers: number): number {
  // Two residual branches per block (attention + MLP), hence the factor of 2.
  return baseStd / Math.sqrt(2 * Math.max(1, numLayers));
}
