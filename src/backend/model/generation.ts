import * as tf from '@tensorflow/tfjs';
import { MiniGPT } from './gpt';
import { Tokenizer } from '../dataset/types';
import { GenerationConfig, GeneratedToken, TokenProbability } from '../../shared/types';
import { Rng } from '../core/rng';

/**
 * ============================================================================
 * MiniGPT Lab - Autoregressive Text Generator & Sampler
 * ============================================================================
 *
 * What is Autoregressive Text Generation?
 * "Autoregressive" means the model generates text one token at a time,
 * feeding its own newly predicted token back into the input sequence
 * to predict the subsequent token.
 *
 * Step-by-step Loop:
 * 1. Take prompt: e.g. "To be"
 * 2. Tokenize prompt into IDs: [35, 60, 1, 47, 49]
 * 3. Crop sequence if it exceeds the model's contextLength (sliding window)
 * 4. Run forward pass: logits = model(tokens)
 * 5. Extract logits for the LAST token position only
 * 6. Apply repetition penalty to tokens already produced
 * 7. Apply temperature scaling: logits = logits / temperature
 * 8. Apply Top-K and/or Top-P (nucleus) filtering
 * 9. Convert to probabilities via softmax
 * 10. Sample a token ID from this distribution
 * 11. Append sampled token to the sequence and repeat
 *
 * THE ORDER OF OPERATIONS IS NOT ARBITRARY
 * Repetition penalty acts on raw logits (before temperature), so its strength
 * does not depend on the temperature setting. Temperature is applied before
 * filtering, because filtering decides WHICH tokens are eligible and
 * temperature decides how sharply to discriminate among them - doing it the
 * other way round would make top-k's cutoff shift as temperature changes.
 */

/** Everything the caller may want back from a generation run. */
export interface GenerationResult {
  /** The prompt plus the continuation. */
  text: string;
  /** Just the newly generated portion. */
  completion: string;
  /** Per-token detail: what was chosen and what else was considered. */
  tokens: GeneratedToken[];
  /** True when generation ended early on a stop sequence. */
  stoppedEarly: boolean;
}

/** How many alternatives to report per position for the UI inspector. */
const TOP_CANDIDATES_TO_REPORT = 8;

export class TextGenerator {
  private model: MiniGPT;
  private tokenizer: Tokenizer;

  constructor(model: MiniGPT, tokenizer: Tokenizer) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  /**
   * Samples an index from a normalized probability distribution.
   *
   * Uses the "inverse CDF" method: walk the cumulative sum until it passes a
   * uniform random draw. The accumulated floating-point error over a long
   * vocabulary means the cumulative total may land slightly under 1.0, so the
   * final index is returned as a fallback rather than looping off the end.
   */
  private sampleMultinomial(probabilities: Float32Array, rng: Rng): number {
    const r = rng.next();
    let cumulative = 0;
    for (let i = 0; i < probabilities.length; i++) {
      cumulative += probabilities[i];
      if (r <= cumulative) return i;
    }
    // Fall back to the last token with non-zero probability.
    for (let i = probabilities.length - 1; i >= 0; i--) {
      if (probabilities[i] > 0) return i;
    }
    return probabilities.length - 1;
  }

  /**
   * Shannon entropy of a distribution, in bits.
   *
   * This is a direct readout of the model's CONFIDENCE at a position. Zero bits
   * means it is certain; log2(vocabSize) bits means it has no idea. Watching it
   * fall during training, and watching which characters it stays high at
   * (the first letter of a word) is genuinely informative.
   */
  private entropyBits(probabilities: Float32Array): number {
    let h = 0;
    for (let i = 0; i < probabilities.length; i++) {
      const p = probabilities[i];
      if (p > 0) h -= p * Math.log2(p);
    }
    return h;
  }

  /**
   * Converts raw logits into a sampled token plus a reportable distribution.
   * Implemented as one pure function over Float32Arrays so it can be unit
   * tested without a model.
   */
  private sampleFromLogits(
    rawLogits: Float32Array,
    config: Required<Pick<GenerationConfig, 'temperature' | 'topK'>> & {
      topP?: number;
      repetitionPenalty?: number;
    },
    seenTokenIds: Set<number>,
    rng: Rng
  ): { tokenId: number; probabilities: Float32Array } {
    const vocabSize = rawLogits.length;
    const logits = new Float32Array(rawLogits);

    // --- 1. Repetition penalty, on raw logits -----------------------------
    // For a positive logit we divide (reducing it); for a negative logit we
    // multiply (making it more negative). Dividing a negative number would
    // move it UP toward zero and thereby reward repetition - the asymmetry is
    // required, and is the standard formulation from the CTRL paper.
    const penalty = config.repetitionPenalty ?? 1;
    if (penalty !== 1 && penalty > 0) {
      for (const id of seenTokenIds) {
        if (id < 0 || id >= vocabSize) continue;
        logits[id] = logits[id] > 0 ? logits[id] / penalty : logits[id] * penalty;
      }
    }

    // --- 2. Temperature ---------------------------------------------------
    // Dividing logits by T < 1 spreads them further apart, so softmax becomes
    // sharper (more deterministic). T > 1 compresses them, flattening the
    // distribution toward uniform (more random).
    const temperature = Math.max(0.01, config.temperature);
    for (let i = 0; i < vocabSize; i++) {
      logits[i] /= temperature;
    }

    // --- 3. Top-K filtering ------------------------------------------------
    // Keep only the K highest logits. The comparison must be `>= cutoff`, and
    // then the surplus from ties must be trimmed; using a bare `< cutoff`
    // test keeps every token tied at the cutoff value, so an untrained model
    // whose logits are nearly identical would effectively bypass top-k
    // entirely - exactly the state a user generates in first.
    const topK = config.topK ?? 0;
    const eligible = new Uint8Array(vocabSize).fill(1);

    if (topK > 0 && topK < vocabSize) {
      const order = Array.from({ length: vocabSize }, (_, i) => i).sort(
        (a, b) => logits[b] - logits[a]
      );
      eligible.fill(0);
      for (let i = 0; i < topK; i++) {
        eligible[order[i]] = 1;
      }
    }

    // --- 4. Softmax over the eligible set ---------------------------------
    let maxLogit = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      if (eligible[i] && logits[i] > maxLogit) maxLogit = logits[i];
    }

    const probabilities = new Float32Array(vocabSize);
    let sum = 0;
    for (let i = 0; i < vocabSize; i++) {
      if (!eligible[i]) continue;
      const e = Math.exp(logits[i] - maxLogit);
      probabilities[i] = e;
      sum += e;
    }
    for (let i = 0; i < vocabSize; i++) {
      probabilities[i] /= sum;
    }

    // --- 5. Top-P (nucleus) filtering -------------------------------------
    // Keep the smallest set of tokens whose probabilities sum to at least P,
    // then renormalize.
    //
    // Why nucleus sampling exists alongside top-k: a FIXED k is wrong in both
    // directions. Where the model is confident (mid-word, after "Rome" comes
    // "o"), k=20 admits 19 bad options. Where it is genuinely uncertain (the
    // start of a new line), k=20 may cut off reasonable ones. Top-P adapts the
    // cutoff to the actual shape of the distribution at each position.
    const topP = config.topP;
    if (topP !== undefined && topP > 0 && topP < 1) {
      const order = Array.from({ length: vocabSize }, (_, i) => i).sort(
        (a, b) => probabilities[b] - probabilities[a]
      );
      let cumulative = 0;
      let cutoffIndex = order.length;
      for (let i = 0; i < order.length; i++) {
        cumulative += probabilities[order[i]];
        if (cumulative >= topP) {
          cutoffIndex = i + 1; // inclusive of the token that crossed the threshold
          break;
        }
      }
      let keptMass = 0;
      for (let i = 0; i < order.length; i++) {
        if (i >= cutoffIndex) probabilities[order[i]] = 0;
        else keptMass += probabilities[order[i]];
      }
      if (keptMass > 0) {
        for (let i = 0; i < vocabSize; i++) probabilities[i] /= keptMass;
      }
    }

    const tokenId = this.sampleMultinomial(probabilities, rng);
    return { tokenId, probabilities };
  }

  /**
   * Builds the top-N candidate report for the UI.
   *
   * The CHOSEN token is always included, even when it falls outside the top N.
   * That is not a rare edge case: with a flat distribution - an untrained
   * model, or a high temperature with no top-k filter - the sampled token
   * frequently is not among the most probable few. An inspector that omits
   * what was actually picked is worse than useless, because the user sees a
   * list of candidates with none of them marked.
   */
  private topCandidates(
    probabilities: Float32Array,
    chosenId: number,
    limit: number = TOP_CANDIDATES_TO_REPORT
  ): TokenProbability[] {
    const ranked = Array.from({ length: probabilities.length }, (_, i) => i)
      .filter((i) => probabilities[i] > 0)
      .sort((a, b) => probabilities[b] - probabilities[a]);

    const selected = ranked.slice(0, limit);
    if (!selected.includes(chosenId)) {
      // Drop the weakest of the top N to make room, so the list length is
      // stable and the chosen token is always visible.
      if (selected.length >= limit) selected.pop();
      selected.push(chosenId);
      selected.sort((a, b) => probabilities[b] - probabilities[a]);
    }

    return selected.map((id) => ({
      token: this.tokenizer.decode([id]),
      tokenId: id,
      probability: probabilities[id],
      chosen: id === chosenId,
    }));
  }

  /**
   * Generates a text continuation from a prompt.
   *
   * @param prompt  Starting string (e.g. "ROMEO:")
   * @param config  Sampling hyperparameters
   * @param onToken Optional callback fired per generated token. Returning
   *                `false` from it cancels generation - which is how the UI's
   *                Stop button works mid-stream.
   */
  public async generate(
    prompt: string,
    config: GenerationConfig,
    onToken?: (token: GeneratedToken, index: number) => boolean | void
  ): Promise<GenerationResult> {
    const initialText = prompt.length > 0 ? prompt : '\n';
    const promptTokens = this.tokenizer.encode(initialText);
    const currentTokens = [...promptTokens];

    const maxTokens = Math.max(1, config.maxTokens || 200);
    const contextLength = this.model.config.contextLength;
    const vocabSize = this.model.config.vocabularySize;
    const stopSequences = (config.stopSequences ?? []).filter((s) => s.length > 0);

    // Seeded so that the same prompt, config and seed reproduce the same text.
    // Omitting the seed gives a fresh stream each call, which is what a user
    // clicking "Generate" repeatedly expects.
    const rng = new Rng(config.seed ?? (Math.random() * 0xffffffff) >>> 0);

    const generatedTokens: GeneratedToken[] = [];
    // Tracks which tokens the repetition penalty applies to. Seeded with the
    // prompt so the model is discouraged from parroting it straight back.
    const seenTokenIds = new Set<number>(promptTokens);
    let stoppedEarly = false;

    for (let step = 0; step < maxTokens; step++) {
      // 1. Crop to the last `contextLength` tokens (sliding window).
      const contextTokens =
        currentTokens.length > contextLength
          ? currentTokens.slice(currentTokens.length - contextLength)
          : currentTokens;

      // 2-4. Forward pass, then pull out the final position's logits.
      // Everything is wrapped so the intermediate 3D logits tensor is freed
      // even if `data()` throws.
      const lastTokenLogits = tf.tidy(() => {
        const inputTensor = tf.tensor2d(
          [contextTokens],
          [1, contextTokens.length],
          'int32'
        );
        // training: false - no dropout at inference, so output is deterministic
        // given the sampling seed.
        const logitsTensor = this.model.forward(inputTensor, { training: false });
        const sliced = tf.slice(
          logitsTensor,
          [0, contextTokens.length - 1, 0],
          [1, 1, vocabSize]
        );
        return tf.reshape(sliced, [vocabSize]);
      });

      const rawLogits = (await lastTokenLogits.data()) as Float32Array;
      lastTokenLogits.dispose();

      // 5-9. Penalty, temperature, filtering, softmax, sample.
      const { tokenId, probabilities } = this.sampleFromLogits(
        rawLogits,
        {
          temperature: config.temperature ?? 0.7,
          topK: config.topK ?? 0,
          topP: config.topP,
          repetitionPenalty: config.repetitionPenalty,
        },
        seenTokenIds,
        rng
      );

      // 10. Append and record.
      currentTokens.push(tokenId);
      seenTokenIds.add(tokenId);

      const token: GeneratedToken = {
        token: this.tokenizer.decode([tokenId]),
        tokenId,
        topCandidates: this.topCandidates(probabilities, tokenId),
        entropyBits: this.entropyBits(probabilities),
      };
      generatedTokens.push(token);

      // The callback may cancel: this is how the UI stops a long generation
      // without waiting for maxTokens.
      if (onToken) {
        const keepGoing = onToken(token, step);
        if (keepGoing === false) {
          stoppedEarly = true;
          break;
        }
      }

      // 11. Stop sequences, checked against the text generated so far.
      if (stopSequences.length > 0) {
        const completionSoFar = this.tokenizer.decode(
          currentTokens.slice(promptTokens.length)
        );
        if (stopSequences.some((s) => completionSoFar.includes(s))) {
          stoppedEarly = true;
          break;
        }
      }

      // Yield to the event loop every few tokens so the HTTP server and
      // WebSocket stay responsive during a long generation. Without this, a
      // 500-token request blocks every other request for its full duration.
      if (step % 8 === 7) {
        await new Promise((resolve) => setImmediate(resolve));
      }
    }

    const fullText = this.tokenizer.decode(currentTokens);
    const completion = this.tokenizer.decode(currentTokens.slice(promptTokens.length));

    return { text: fullText, completion, tokens: generatedTokens, stoppedEarly };
  }

  /**
   * Computes the attention maps for a prompt, for the Attention Explorer.
   *
   * Returns plain nested arrays rather than tensors, because this crosses the
   * HTTP boundary. The payload is numLayers * numHeads * T * T floats, which is
   * why the caller should keep the prompt short.
   */
  public async attentionSnapshot(
    prompt: string
  ): Promise<{ tokens: string[]; weights: number[][][][] }> {
    const contextLength = this.model.config.contextLength;
    let tokenIds = this.tokenizer.encode(prompt.length > 0 ? prompt : '\n');

    if (tokenIds.length > contextLength) {
      tokenIds = tokenIds.slice(tokenIds.length - contextLength);
    }

    const input = tf.tensor2d([tokenIds], [1, tokenIds.length], 'int32');
    const { logits, attention } = this.model.forwardWithAttention(input);

    const weights: number[][][][] = [];
    for (const layerAttn of attention) {
      // Shape [1, numHeads, T, T] -> drop the batch dimension.
      const arr = (await layerAttn.array()) as number[][][][];
      weights.push(arr[0] as unknown as number[][][]);
      layerAttn.dispose();
    }

    input.dispose();
    logits.dispose();

    return {
      tokens: tokenIds.map((id) => this.tokenizer.decode([id])),
      weights,
    };
  }
}
