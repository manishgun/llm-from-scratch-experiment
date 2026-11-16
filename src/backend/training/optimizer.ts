import * as tf from '@tensorflow/tfjs';

/**
 * ============================================================================
 * MiniGPT Lab - Gradient Clipping & Decoupled Weight Decay (AdamW)
 * ============================================================================
 *
 * TensorFlow.js ships `tf.train.adam`, which handles the adaptive moment
 * estimation itself. What it does NOT provide, and what every production
 * Transformer training loop needs, are the two mechanisms in this file.
 */

/**
 * ----------------------------------------------------------------------------
 * GLOBAL GRADIENT NORM CLIPPING
 * ----------------------------------------------------------------------------
 *
 * THE PROBLEM IT SOLVES
 * Loss landscapes have cliffs. Occasionally a batch contains an unusual
 * sequence, the model is confidently wrong about it, and the resulting gradient
 * is one or two orders of magnitude larger than normal. That single step throws
 * the weights far outside the region where the model made sense. The loss jumps
 * to a huge value, or straight to NaN, and the run is dead - often 80% of the
 * way through.
 *
 * THE FIX
 * Treat all gradients as ONE long vector. Compute its L2 norm. If the norm
 * exceeds `maxNorm`, multiply every gradient by `maxNorm / norm`.
 *
 * WHY GLOBAL, NOT PER-TENSOR
 * Clipping each tensor separately (`clipByValue`) distorts the gradient's
 * direction: it changes the relative magnitudes between layers, so the update
 * no longer points downhill. Scaling everything by one shared factor preserves
 * the direction exactly and only shortens the step. That is why this is the
 * standard approach (Pascanu et al. 2013) and why the value is almost always
 * 1.0 in Transformer training recipes.
 *
 * @param gradients Map from variable name to its gradient tensor
 * @param maxNorm   The norm ceiling. Pass 0 or less to disable clipping.
 * @returns The PRE-clipping global norm (worth logging - a spiking norm is the
 *          earliest warning that the learning rate is too high) and the
 *          possibly-rescaled gradients.
 */
export function clipGradientsByGlobalNorm(
  gradients: tf.NamedTensorMap,
  maxNorm: number
): { globalNorm: number; clipped: tf.NamedTensorMap; wasClipped: boolean } {
  // Compute sum of squares across every gradient tensor, then square-root it.
  const globalNorm = tf.tidy(() => {
    const squaredSums = Object.values(gradients).map((g) => tf.sum(tf.square(g)));
    const total = squaredSums.reduce(
      (acc: tf.Tensor, cur: tf.Tensor) => tf.add(acc, cur),
      tf.scalar(0)
    );
    return tf.sqrt(total).dataSync()[0];
  });

  // A NaN or Infinity norm means the forward pass already diverged. Report it
  // so the caller can abort rather than quietly poisoning every weight.
  if (!Number.isFinite(globalNorm)) {
    return { globalNorm, clipped: gradients, wasClipped: false };
  }

  if (maxNorm <= 0 || globalNorm <= maxNorm) {
    return { globalNorm, clipped: gradients, wasClipped: false };
  }

  // Rescale every gradient by the same factor, preserving direction.
  const scale = maxNorm / (globalNorm + 1e-6);
  const clipped: tf.NamedTensorMap = {};
  for (const [name, grad] of Object.entries(gradients)) {
    clipped[name] = tf.tidy(() => tf.mul(grad, scale));
    grad.dispose();
  }

  return { globalNorm, clipped, wasClipped: true };
}

/**
 * ----------------------------------------------------------------------------
 * DECOUPLED WEIGHT DECAY (the "W" in AdamW)
 * ----------------------------------------------------------------------------
 *
 * WHAT WEIGHT DECAY IS FOR
 * Large weights let a model carve very sharp, specific decision boundaries -
 * ideal for memorising individual training examples, bad for generalising.
 * Weight decay adds a steady pull toward zero, so a weight only stays large if
 * the gradient keeps pushing it there. The result is a simpler model that
 * generalises better.
 *
 * WHY "DECOUPLED" IS NOT A DETAIL
 * The classic way to implement this is L2 regularisation: add `wd * w` to the
 * gradient. Inside plain SGD that is exactly equivalent to decay. Inside ADAM
 * it is not, and the difference is severe.
 *
 * Adam divides each gradient by the running RMS of that parameter's past
 * gradients. A weight with consistently large gradients gets divided by a large
 * number - including the `wd * w` term folded into it. So weights that need
 * decay most receive the least, and the effective decay rate differs per
 * parameter in a way nobody intended.
 *
 * Loshchilov & Hutter (2017) showed the fix: apply the decay DIRECTLY to the
 * weights, after and separate from the adaptive update:
 *
 *     w <- w - lr * adam_update(grad)       (handled by tf.train.adam)
 *     w <- w * (1 - lr * weightDecay)       (this function)
 *
 * Now every weight decays at the same true rate, and `weightDecay` means the
 * same thing regardless of gradient history. This is what every modern
 * Transformer uses.
 *
 * WHAT IS EXEMPT
 * Biases and LayerNorm parameters. A bias shifts a distribution rather than
 * scaling it, so shrinking it does not reduce model complexity. And decaying a
 * LayerNorm `gamma` toward zero directly fights normalization - it would shrink
 * the normalized signal a little on every single step.
 *
 * @param variables   The variables eligible for decay (decay-exempt ones already removed)
 * @param weightDecay The decay coefficient (e.g. 0.01). No-op when <= 0.
 * @param learningRate The CURRENT learning rate, so decay follows the schedule
 */
export function applyDecoupledWeightDecay(
  variables: tf.Variable[],
  weightDecay: number,
  learningRate: number
): void {
  if (weightDecay <= 0) return;

  const factor = 1 - learningRate * weightDecay;
  // A non-positive factor would flip or zero every weight - that only happens
  // with an absurd configuration, but silently destroying the model is worse
  // than skipping the step.
  if (factor <= 0 || !Number.isFinite(factor)) return;

  for (const variable of variables) {
    if (variable.isDisposed) continue;
    tf.tidy(() => {
      variable.assign(tf.mul(variable, factor));
    });
  }
}

/**
 * Builds the set of variables that SHOULD receive weight decay, by removing
 * the exempt ones. Identity is compared by reference, which is what we want -
 * two variables can share a name across model instances.
 */
export function decayableVariables(
  all: tf.Variable[],
  exempt: tf.Variable[]
): tf.Variable[] {
  const exemptSet = new Set(exempt);
  return all.filter((v) => !exemptSet.has(v));
}
