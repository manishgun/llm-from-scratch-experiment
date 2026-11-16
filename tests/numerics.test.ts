import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { gelu, layerNorm, stableSoftmax, linear3D, linear3DTransposed, dropout } from '../src/backend/model/layers';
import { Rng } from '../src/backend/core/rng';

/**
 * ============================================================================
 * Numerical Correctness of the Layer Primitives
 * ============================================================================
 *
 * These tests check the layer functions against values computed independently
 * from their mathematical definitions, rather than against whatever the code
 * currently produces.
 *
 * This is the distinction that matters: a test asserting `gelu(x)` returns the
 * same thing it returned yesterday catches regressions but not a formula that
 * was wrong from the start. Comparing to hand-derived values catches both.
 */

describe('GELU activation', () => {
  it('matches hand-computed values from the tanh approximation', async () => {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    const reference = (x: number) =>
      0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));

    const inputs = [-3, -1, -0.5, 0, 0.5, 1, 2, 3];
    const result = await gelu(tf.tensor1d(inputs)).data();

    for (let i = 0; i < inputs.length; i++) {
      expect(result[i]).toBeCloseTo(reference(inputs[i]), 5);
    }
  });

  it('maps 0 to exactly 0 and is monotonically increasing above 0', async () => {
    const values = await gelu(tf.tensor1d([0, 0.1, 0.5, 1, 2, 4])).data();
    expect(values[0]).toBeCloseTo(0, 6);
    for (let i = 1; i < values.length; i++) {
      expect(values[i]).toBeGreaterThan(values[i - 1]);
    }
  });

  it('lets a small negative gradient through, unlike ReLU', async () => {
    // This is the whole point of GELU over ReLU: negative inputs produce
    // small negative outputs rather than a hard zero, so the unit is never
    // completely dead.
    const values = await gelu(tf.tensor1d([-0.5, -1])).data();
    expect(values[0]).toBeLessThan(0);
    expect(values[0]).toBeGreaterThan(-0.2);
  });
});

describe('Layer normalization', () => {
  it('produces zero mean and unit variance per position', async () => {
    const d = 8;
    // Deliberately off-centre and badly scaled input.
    const x = tf.tensor3d([[[10, 12, 14, 16, 18, 20, 22, 24]]], [1, 1, d]);
    const gamma = tf.ones([d]);
    const beta = tf.zeros([d]);

    const out = (await layerNorm(x, gamma, beta, 1e-5).data()) as Float32Array;

    const mean = Array.from(out).reduce((a, b) => a + b, 0) / d;
    const variance = Array.from(out).reduce((a, b) => a + (b - mean) ** 2, 0) / d;

    expect(mean).toBeCloseTo(0, 5);
    // Variance approaches 1 from below because of the epsilon in the
    // denominator - it can never be exactly 1.
    expect(variance).toBeCloseTo(1, 3);
  });

  it('applies gamma and beta after standardizing', async () => {
    const d = 4;
    const x = tf.tensor3d([[[1, 2, 3, 4]]], [1, 1, d]);
    const gamma = tf.tensor1d([2, 2, 2, 2]);
    const beta = tf.tensor1d([5, 5, 5, 5]);

    const out = (await layerNorm(x, gamma, beta, 1e-5).data()) as Float32Array;

    // Standardized values for [1,2,3,4]: mean 2.5, population std sqrt(1.25).
    const std = Math.sqrt(1.25);
    for (let i = 0; i < d; i++) {
      const expected = ((i + 1 - 2.5) / std) * 2 + 5;
      expect(out[i]).toBeCloseTo(expected, 3);
    }
  });

  it('normalizes each token position independently', async () => {
    // Two positions with wildly different scales must both come out
    // standardized - LayerNorm operates along the feature axis only.
    const x = tf.tensor3d([[[1, 2, 3, 4], [1000, 2000, 3000, 4000]]], [1, 2, 4]);
    const out = (await layerNorm(x, tf.ones([4]), tf.zeros([4]), 1e-5).array()) as number[][][];

    for (const position of out[0]) {
      const mean = position.reduce((a, b) => a + b, 0) / position.length;
      expect(mean).toBeCloseTo(0, 4);
    }
  });
});

describe('Numerically stable softmax', () => {
  it('produces a distribution that sums to 1', async () => {
    const out = (await stableSoftmax(tf.tensor1d([1, 2, 3, 4]), -1).data()) as Float32Array;
    const sum = Array.from(out).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
  });

  it('matches hand-computed softmax values', async () => {
    const logits = [1, 2, 3];
    const expSum = logits.reduce((a, v) => a + Math.exp(v), 0);
    const out = (await stableSoftmax(tf.tensor1d(logits), -1).data()) as Float32Array;

    for (let i = 0; i < logits.length; i++) {
      expect(out[i]).toBeCloseTo(Math.exp(logits[i]) / expSum, 6);
    }
  });

  it('does not overflow on extreme logits', async () => {
    // exp(1000) is Infinity in float32. A naive implementation returns NaN
    // here; subtracting the row max keeps every exponent at or below 0.
    const out = (await stableSoftmax(tf.tensor1d([1000, 1001, 1002]), -1).data()) as Float32Array;

    expect(Array.from(out).every((v) => Number.isFinite(v))).toBe(true);
    expect(Array.from(out).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    // The largest logit must still win.
    expect(out[2]).toBeGreaterThan(out[1]);
  });

  it('is shift invariant, as the mathematics requires', async () => {
    const a = (await stableSoftmax(tf.tensor1d([1, 2, 3]), -1).data()) as Float32Array;
    const b = (await stableSoftmax(tf.tensor1d([101, 102, 103]), -1).data()) as Float32Array;
    for (let i = 0; i < 3; i++) {
      expect(a[i]).toBeCloseTo(b[i], 5);
    }
  });
});

describe('Linear projections', () => {
  it('linear3D computes x @ W + b with correct shapes', async () => {
    const x = tf.tensor3d([[[1, 2], [3, 4]]], [1, 2, 2]);
    const w = tf.tensor2d([[1, 0, 2], [0, 1, 3]], [2, 3]);
    const b = tf.tensor1d([10, 20, 30]);

    const out = linear3D(x, w, b);
    expect(out.shape).toEqual([1, 2, 3]);

    const values = (await out.array()) as number[][][];
    // [1,2] @ W = [1, 2, 1*2 + 2*3] = [1, 2, 8], plus bias.
    expect(values[0][0]).toEqual([11, 22, 38]);
    // [3,4] @ W = [3, 4, 3*2 + 4*3] = [3, 4, 18], plus bias.
    expect(values[0][1]).toEqual([13, 24, 48]);
  });

  it('linear3DTransposed equals linear3D with a transposed weight', async () => {
    // This equivalence is what makes weight tying valid: reusing the [V, D]
    // embedding table transposed must be identical to a dedicated [D, V] head.
    const x = tf.randomNormal([2, 3, 4]) as tf.Tensor3D;
    const w = tf.randomNormal([4, 7]) as tf.Tensor2D; // [inDim, outDim]
    const wT = tf.transpose(w) as tf.Tensor2D; // [outDim, inDim]

    const direct = (await linear3D(x, w).data()) as Float32Array;
    const transposed = (await linear3DTransposed(x, wT).data()) as Float32Array;

    expect(direct.length).toBe(transposed.length);
    for (let i = 0; i < direct.length; i++) {
      expect(transposed[i]).toBeCloseTo(direct[i], 4);
    }
  });
});

describe('Dropout', () => {
  it('is the identity when not training', async () => {
    const x = tf.tensor1d([1, 2, 3, 4, 5]);
    const out = (await dropout(x, 0.5, false).data()) as Float32Array;
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5]);
  });

  it('is the identity when the rate is zero', async () => {
    const x = tf.tensor1d([1, 2, 3]);
    const out = (await dropout(x, 0, true).data()) as Float32Array;
    expect(Array.from(out)).toEqual([1, 2, 3]);
  });

  it('zeroes roughly the configured fraction and rescales the survivors', async () => {
    const size = 20000;
    const rate = 0.25;
    const x = tf.ones([size]);
    const out = (await dropout(x, rate, true, new Rng(1)).data()) as Float32Array;

    const zeros = Array.from(out).filter((v) => v === 0).length;
    expect(zeros / size).toBeCloseTo(rate, 1);

    // Inverted dropout: survivors are scaled by 1/(1-rate) so the EXPECTED
    // sum is preserved. This is why inference needs no correction at all.
    const survivorValue = Array.from(out).find((v) => v !== 0)!;
    expect(survivorValue).toBeCloseTo(1 / (1 - rate), 4);

    const mean = Array.from(out).reduce((a, b) => a + b, 0) / size;
    expect(mean).toBeCloseTo(1, 1);
  });

  it('produces an identical mask for an identical seed', async () => {
    const x = tf.ones([500]);
    const a = (await dropout(x, 0.3, true, new Rng(42)).data()) as Float32Array;
    const b = (await dropout(x, 0.3, true, new Rng(42)).data()) as Float32Array;
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it('rejects a rate of 1.0, which would zero everything', () => {
    expect(() => dropout(tf.ones([4]), 1.0, true)).toThrow(/must be < 1/);
  });
});
