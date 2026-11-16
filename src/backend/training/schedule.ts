import { TrainingConfig } from '../../shared/types';

/**
 * ============================================================================
 * MiniGPT Lab - Learning Rate Schedules
 * ============================================================================
 *
 * WHY A SCHEDULE AT ALL
 * ---------------------
 * A single fixed learning rate has to do two incompatible jobs:
 *
 *   - Early in training, the weights are random and far from anything useful.
 *     You want LARGE steps to travel quickly to a good region of the loss
 *     landscape.
 *   - Late in training, the weights are nearly right. Large steps now just
 *     bounce around the minimum without ever settling into it - the loss
 *     plateaus at a value well above what the model is capable of.
 *
 * A schedule resolves this by using large steps early and small steps late.
 * In practice this is the difference between a character-level Shakespeare
 * model plateauing around loss 2.5 and the same model reaching ~1.5.
 *
 * THE THREE PHASES
 * ----------------
 *
 *   lr
 *   |      ,-~~~-.
 *   |     /       `-.
 *   |    /            `-._
 *   |   /                  `--.___
 *   |  /                           `----
 *   | /
 *   |/
 *   +----------------------------------------> step
 *    warmup |         decay
 *
 * 1. WARMUP (linear ramp from ~0 to the peak rate)
 *    Adam adapts its per-parameter step size using a running average of
 *    squared gradients. At step 1 that average is based on a single sample, so
 *    it is a terrible estimate, and Adam's "adaptive" step can be enormous and
 *    pointed in a near-random direction. A few hundred such steps can damage
 *    the careful initialization beyond recovery. Warmup keeps steps tiny until
 *    the moment estimates have enough history to be trustworthy.
 *
 * 2. DECAY (cosine or linear, down to a small floor)
 *    Cosine decay is the standard choice (GPT-2, GPT-3, LLaMA, Chinchilla). It
 *    spends a relatively long time near the peak rate, then decays smoothly
 *    with no sudden drops - a sudden drop causes a visible loss discontinuity
 *    and wastes the steps immediately after it.
 *
 * 3. FLOOR (minLearningRate)
 *    Decaying all the way to exactly zero means the final steps accomplish
 *    nothing at all. A floor of roughly 10% of the peak keeps the last stretch
 *    productive.
 */

/**
 * Computes the learning rate for a given step.
 *
 * @param step   Zero-based step index about to be executed
 * @param config The training configuration (peak rate, schedule, warmup, floor)
 */
export function learningRateAt(step: number, config: TrainingConfig): number {
  const peak = config.learningRate;
  const warmup = Math.max(0, config.warmupSteps ?? 0);
  const total = Math.max(1, config.trainingSteps);
  const floor = Math.max(0, config.minLearningRate ?? 0);
  const schedule = config.lrSchedule ?? 'constant';

  // --- Phase 1: linear warmup -------------------------------------------
  // At step 0 we use peak * (1 / warmup) rather than exactly 0, because a
  // learning rate of literally zero makes the first step a no-op and wastes it.
  if (warmup > 0 && step < warmup) {
    return peak * ((step + 1) / warmup);
  }

  if (schedule === 'constant') {
    return peak;
  }

  // --- Phase 2: decay ----------------------------------------------------
  // `progress` runs 0 -> 1 across the post-warmup portion of the run.
  const decaySteps = Math.max(1, total - warmup);
  const progress = Math.min(1, Math.max(0, (step - warmup) / decaySteps));

  if (schedule === 'linear') {
    return floor + (peak - floor) * (1 - progress);
  }

  // Cosine: 0.5 * (1 + cos(pi * progress)) falls smoothly from 1 to 0 with
  // zero derivative at both ends - no abrupt change in step size anywhere.
  const cosineFactor = 0.5 * (1 + Math.cos(Math.PI * progress));
  return floor + (peak - floor) * cosineFactor;
}

/**
 * Precomputes the whole schedule, so the UI can draw the curve before the run
 * starts. Being able to SEE the schedule you configured - and notice that your
 * warmup is longer than your entire run - is worth the few hundred bytes.
 */
export function materializeSchedule(config: TrainingConfig): number[] {
  const out: number[] = new Array(config.trainingSteps);
  for (let i = 0; i < config.trainingSteps; i++) {
    out[i] = learningRateAt(i, config);
  }
  return out;
}
