/**
 * ============================================================================
 * MiniGPT Lab - Deterministic Random Number Generation
 * ============================================================================
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * An experimentation tool is only useful if runs are COMPARABLE. If you train
 * with `embeddingDim: 64`, then again with `embeddingDim: 128`, and the second
 * run reaches a lower loss, you want to be able to say "the extra capacity
 * caused that". You can only say it if everything *else* was identical —
 * including the random initial weights and the random batch order.
 *
 * `Math.random()` and `tf.randomNormal()` (without a seed) are reseeded from
 * entropy on every process start, so two runs of the same config produce
 * different numbers. That makes every comparison unfalsifiable.
 *
 * This module provides a small, fast, fully deterministic PRNG so that a given
 * `randomSeed` always reproduces: the same initial weights, the same training
 * batches, the same validation batches, and the same sampled text.
 *
 * ALGORITHM: mulberry32
 * ---------------------
 * A 32-bit state generator. It is NOT cryptographically secure (do not use it
 * for keys or tokens) but it has excellent statistical properties for
 * simulation, passes the gjrand test suite, is ~2 lines of arithmetic, and —
 * crucially — produces identical output on every platform and Node version.
 */

/**
 * A deterministic pseudo-random number generator.
 *
 * Every instance carries its own state, so separate concerns can each hold an
 * independent stream without interfering. That matters: we want the *batch*
 * sampler and the *text sampler* to be independent, so that generating a
 * sample mid-training does not shift which batches the next step trains on.
 */
export class Rng {
  private state: number;

  /** The seed this generator was created with (kept for logging / checkpoints). */
  public readonly seed: number;

  constructor(seed: number = 42) {
    // Force into an unsigned 32-bit integer. `>>> 0` is the idiomatic cast.
    this.seed = seed >>> 0;
    this.state = this.seed;
  }

  /**
   * Restore the generator to its initial state.
   * Used at the start of each epoch/eval pass so the same batches recur.
   */
  public reset(): void {
    this.state = this.seed;
  }

  /**
   * Core mulberry32 step: returns a float uniformly distributed in [0, 1).
   *
   * The constants are the published mulberry32 magic numbers; they are chosen
   * so the state sequence has a full 2^32 period and the output bits are well
   * mixed. `Math.imul` performs a true 32-bit integer multiply (plain `*`
   * would lose precision above 2^53).
   */
  public next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    // `>>> 0` makes it unsigned; dividing by 2^32 maps it into [0, 1).
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /**
   * A uniformly distributed integer in [0, maxExclusive).
   */
  public nextInt(maxExclusive: number): number {
    return Math.floor(this.next() * maxExclusive);
  }

  /**
   * A sample from the standard normal distribution N(0, 1), via the
   * Box-Muller transform.
   *
   * Box-Muller turns two uniform samples (u1, u2) into two independent normal
   * samples. We generate both and cache the spare one, so the amortised cost
   * is one `next()` call per normal sample.
   *
   *   z0 = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
   *   z1 = sqrt(-2 * ln(u1)) * sin(2 * pi * u2)
   */
  private spareNormal: number | null = null;

  public nextNormal(mean: number = 0, stdDev: number = 1): number {
    if (this.spareNormal !== null) {
      const spare = this.spareNormal;
      this.spareNormal = null;
      return mean + stdDev * spare;
    }

    // Guard against u1 === 0, which would make ln(u1) = -Infinity.
    let u1 = this.next();
    while (u1 === 0) u1 = this.next();
    const u2 = this.next();

    const radius = Math.sqrt(-2.0 * Math.log(u1));
    const theta = 2.0 * Math.PI * u2;

    this.spareNormal = radius * Math.sin(theta);
    return mean + stdDev * (radius * Math.cos(theta));
  }

  /**
   * Fill a Float32Array with N(mean, stdDev) samples.
   * This is how we build every weight tensor: we generate the numbers here
   * deterministically, then hand the buffer to TensorFlow, rather than calling
   * `tf.randomNormal` (which we cannot seed per-tensor).
   */
  public normalArray(length: number, mean: number = 0, stdDev: number = 1): Float32Array {
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      out[i] = this.nextNormal(mean, stdDev);
    }
    return out;
  }

  /**
   * Derive a new, independent generator from this one.
   *
   * Used to give each weight tensor its own stream in a way that still depends
   * deterministically on the master seed. Without this, adding a layer would
   * shift every subsequent tensor's random numbers, so a 3-layer and a 4-layer
   * model would not share *any* initial weights — making depth comparisons
   * noisier than they need to be.
   */
  public derive(label: string): Rng {
    // FNV-1a hash of the label, mixed with this generator's seed.
    let hash = 0x811c9dc5 ^ this.seed;
    for (let i = 0; i < label.length; i++) {
      hash ^= label.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193);
    }
    return new Rng(hash >>> 0);
  }
}

/**
 * A shared, module-level generator for callers that do not thread an `Rng`
 * through. Explicitly seeded so even the "lazy" path is reproducible.
 */
export const globalRng = new Rng(42);
