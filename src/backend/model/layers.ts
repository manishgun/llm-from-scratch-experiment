import * as tf from '@tensorflow/tfjs';
import { Rng } from '../core/rng';

/**
 * ============================================================================
 * MiniGPT Lab - Mathematical Activation Functions & Linear Layers
 * ============================================================================
 *
 * Why do neural networks need Activation Functions?
 * Without activation functions, stacking 100 linear matrix multiplications
 * W2 * (W1 * x) is still just ONE big linear transformation (W_combined * x).
 * Non-linear activation functions bend the space and allow the network to learn
 * complex curved boundaries, reasoning, logic gates, and language syntax!
 */

/**
 * Linear projection for 3D tensors: [batchSize, seqLen, inDim] -> [batchSize, seqLen, outDim].
 *
 * In TensorFlow.js, performing 3D matmul with a 2D weight matrix directly can lead to
 * gradient shape mismatches during backpropagation. This helper cleanly flattens [B, T, D]
 * into [B * T, D], executes the 2D matrix multiplication with W, adds bias B, and reshapes
 * back to [B, T, outDim].
 *
 * @param x Input tensor: [batchSize, seqLen, inDim]
 * @param weight Weight matrix: [inDim, outDim]
 * @param bias Bias vector: [outDim]
 */
export function linear3D(x: tf.Tensor, weight: tf.Tensor, bias?: tf.Tensor): tf.Tensor {
  return tf.tidy(() => {
    const [B, T, inDim] = x.shape as [number, number, number];
    const outDim = weight.shape[1] as number;

    // Flatten batch and sequence dimensions: [B * T, inDim]
    const flatX = tf.reshape(x, [B * T, inDim]);

    // Matrix multiplication: [B * T, inDim] x [inDim, outDim] -> [B * T, outDim]
    let out = tf.matMul(flatX, weight);

    // Add bias if provided
    if (bias) {
      out = tf.add(out, bias);
    }

    // Reshape back to 3D: [B, T, outDim]
    return tf.reshape(out, [B, T, outDim]);
  });
}

/**
 * Linear projection using a TRANSPOSED weight matrix:
 * output [B, T, outDim] computed from a weight of shape [outDim, inDim].
 *
 * This exists to support WEIGHT TYING. The language-model head needs a matrix
 * of shape [embeddingDim, vocabSize], while the token embedding table has
 * shape [vocabSize, embeddingDim] - the transpose of one another. Tying them
 * means reusing the embedding table as the output projection, which is what
 * GPT-2 does. See `gpt.ts` for why that helps.
 */
export function linear3DTransposed(
  x: tf.Tensor,
  weightTransposed: tf.Tensor,
  bias?: tf.Tensor
): tf.Tensor {
  return tf.tidy(() => {
    const [B, T, inDim] = x.shape as [number, number, number];
    const outDim = weightTransposed.shape[0] as number;

    const flatX = tf.reshape(x, [B * T, inDim]);

    // matMul with transposeB: [B*T, inDim] x [outDim, inDim]^T -> [B*T, outDim]
    let out = tf.matMul(flatX, weightTransposed, false, true);

    if (bias) {
      out = tf.add(out, bias);
    }

    return tf.reshape(out, [B, T, outDim]);
  });
}

/**
 * GELU (Gaussian Error Linear Unit)
 * Used in modern GPT models (GPT-2, GPT-3, BERT).
 *
 * Formula approximation:
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
 *
 * Why GELU over ReLU?
 * Unlike ReLU (which hard-clips all negative numbers to 0), GELU smoothly curves
 * negative numbers, allowing small gradient information to still pass through
 * (preventing "dead neurons").
 *
 * PERFORMANCE NOTE: the cube is computed as `x * x * x` rather than
 * `tf.pow(x, 3)`. `pow` dispatches to a generic exponentiation kernel that
 * evaluates exp(3 * log(x)) per element and needs a special case for negative
 * inputs; two multiplies are substantially faster on the CPU backend and are
 * also exactly representable. GELU runs on the widest tensor in the model
 * (the MLP hidden layer, 4x embeddingDim), so this is a measurable win.
 */
export function gelu(x: tf.Tensor): tf.Tensor {
  return tf.tidy(() => {
    // sqrt(2 / pi) ~= 0.7978845608
    const sqrt2OverPi = tf.scalar(0.7978845608028654);
    const coeff = tf.scalar(0.044715);
    const half = tf.scalar(0.5);
    const one = tf.scalar(1.0);

    const xCubed = tf.mul(tf.mul(x, x), x);
    const inner = tf.add(x, tf.mul(coeff, xCubed));
    const scaled = tf.mul(sqrt2OverPi, inner);
    const tanhVal = tf.tanh(scaled);
    const term = tf.add(one, tanhVal);

    return tf.mul(half, tf.mul(x, term));
  });
}

/**
 * Layer Normalization
 * Formula: LN(x) = ((x - mean) / sqrt(variance + epsilon)) * gamma + beta
 *
 * Why is LayerNorm critical in Transformers?
 * As numbers pass through multiple layers of matrix multiplications, they can
 * blow up (exploding activations) or shrink to near zero (vanishing activations).
 * LayerNorm rescales the vector at each token position so its mean is 0 and
 * standard deviation is 1, then scales and shifts it using learned parameters
 * gamma (scale) and beta (shift).
 *
 * @param x Input tensor: [batchSize, seqLen, embeddingDim]
 * @param gamma Learnable scale parameter: [embeddingDim]
 * @param beta Learnable shift parameter: [embeddingDim]
 * @param eps Small constant to avoid dividing by zero: default 1e-5
 */
export function layerNorm(
  x: tf.Tensor,
  gamma: tf.Tensor,
  beta: tf.Tensor,
  eps: number = 1e-5
): tf.Tensor {
  return tf.tidy(() => {
    // Compute mean across the last dimension (embeddingDim).
    // keepDims = true preserves the rank so shape remains [batchSize, seqLen, 1]
    const mean = tf.mean(x, -1, true);

    // Variance = mean of squared differences: E[(x - mean)^2]
    const diff = tf.sub(x, mean);
    const variance = tf.mean(tf.square(diff), -1, true);

    // Standard deviation = sqrt(variance + epsilon)
    const std = tf.sqrt(tf.add(variance, tf.scalar(eps)));

    // Standardize, then rescale and shift: ((x - mean) / std) * gamma + beta
    return tf.add(tf.mul(tf.div(diff, std), gamma), beta);
  });
}

/**
 * Softmax with Numerical Stability
 * Formula: Softmax(z_i) = exp(z_i - max(z)) / sum_j(exp(z_j - max(z)))
 *
 * Converts arbitrary numbers (logits) into a probability distribution
 * where all numbers are between 0.0 and 1.0, and they sum to 1.0.
 *
 * Subtracting the row maximum before exponentiating is mathematically a no-op
 * (it cancels between numerator and denominator) but numerically essential:
 * exp(1000) overflows float32 to Infinity, while exp(1000 - 1000) = 1.
 *
 * @param logits Input tensor
 * @param axis Dimension along which to compute softmax (usually -1)
 */
export function stableSoftmax(logits: tf.Tensor, axis: number = -1): tf.Tensor {
  return tf.tidy(() => {
    const maxVal = tf.max(logits, axis, true);
    const exp = tf.exp(tf.sub(logits, maxVal));
    const sum = tf.sum(exp, axis, true);
    return tf.div(exp, sum);
  });
}

/**
 * ----------------------------------------------------------------------------
 * DROPOUT
 * ----------------------------------------------------------------------------
 *
 * WHAT IT DOES
 * During training, randomly set a fraction `rate` of the activations to zero,
 * and scale the survivors up by `1 / (1 - rate)`.
 *
 * WHY IT HELPS
 * A network with redundant capacity can "memorise" its training data by
 * building fragile chains of co-adapted neurons - neuron A only works because
 * neuron B always fires alongside it. Randomly deleting neurons breaks those
 * chains, forcing every neuron to be independently useful. The result is a
 * model that generalises instead of memorising. On a small dataset like Tiny
 * Shakespeare with a few hundred thousand parameters, overfitting is the
 * dominant failure mode, so this matters.
 *
 * WHY THE SCALING ("inverted dropout")
 * If we zero 10% of activations, the expected sum arriving at the next layer
 * drops by 10%. Scaling survivors by 1/(1 - 0.1) = 1.111 restores the expected
 * magnitude. Doing the correction HERE, during training, means inference needs
 * no correction at all - we simply skip dropout entirely. That is why the
 * `training` flag exists and why evaluation is deterministic.
 *
 * @param x        Input tensor
 * @param rate     Fraction of units to drop, in [0, 1)
 * @param training Whether we are training. When false, this is the identity.
 * @param rng      Seeded generator, so the dropout mask is reproducible
 */
export function dropout(x: tf.Tensor, rate: number, training: boolean, rng?: Rng): tf.Tensor {
  // At inference time, or with dropout disabled, pass the tensor through
  // untouched. `x.clone()` keeps ownership semantics consistent for callers
  // that dispose the result.
  if (!training || rate <= 0) {
    return x.clone();
  }
  if (rate >= 1) {
    throw new Error(`dropoutRate must be < 1.0, received ${rate}`);
  }

  return tf.tidy(() => {
    const size = x.size;
    const keepProb = 1 - rate;
    const mask = new Float32Array(size);

    // Build the Bernoulli keep-mask from our seeded generator rather than
    // `tf.randomUniform`, which cannot be seeded per-call.
    if (rng) {
      for (let i = 0; i < size; i++) {
        mask[i] = rng.next() < keepProb ? 1 / keepProb : 0;
      }
    } else {
      for (let i = 0; i < size; i++) {
        mask[i] = Math.random() < keepProb ? 1 / keepProb : 0;
      }
    }

    // Multiplying by a constant mask is differentiable: the gradient flows
    // through the surviving units and is blocked at the dropped ones, which is
    // exactly the intended behaviour.
    return tf.mul(x, tf.tensor(mask, x.shape, 'float32'));
  });
}
