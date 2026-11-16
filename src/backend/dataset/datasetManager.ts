import * as tf from '@tensorflow/tfjs';
import * as https from 'https';
import * as fs from 'fs';
import * as path from 'path';
import { DatasetStatistics } from '../../shared/types';
import { CharTokenizer } from './tokenizer';
import { WordTokenizer } from './wordTokenizer';
import { Tokenizer, TokenizerType } from './types';
import { Rng } from '../core/rng';

/**
 * ============================================================================
 * MiniGPT Lab - Dataset Management & Batch Sampling
 * ============================================================================
 *
 * This module handles:
 * 1. Downloading text datasets (e.g. Tiny Shakespeare by default).
 * 2. Creating the vocabulary and tokenizer (character- or word-level).
 * 3. Strict train / validation / test splits (NO data leakage).
 * 4. Random autoregressive batch generation:
 *      Input sequences:  X = tokens[i ... i + contextLength - 1]
 *      Target sequences: Y = tokens[i + 1 ... i + contextLength]
 *
 * WHY THE TARGETS ARE JUST THE INPUTS SHIFTED BY ONE
 * This is the whole trick behind language-model pretraining. A sequence of
 * length T gives you T separate prediction problems for the price of one
 * forward pass: position 0 predicts token 1, position 1 predicts token 2, and
 * so on. The causal mask is what makes this sound - it guarantees position i
 * cannot see token i+1, so the model cannot cheat by reading the answer.
 *
 * WHY THE SPLITS ARE CONTIGUOUS, NOT SHUFFLED
 * The split is a straight cut at 80% / 90% of the text, not a random
 * assignment of windows. Random windows would overlap: a training window
 * ending at character 1000 and a validation window starting at character 990
 * share ten characters, so validation loss would be measured partly on text
 * the model has already memorised. A contiguous cut makes leakage impossible.
 */

export interface DatasetSplit {
  train: Int32Array;
  val: Int32Array;
  test: Int32Array;
}

export class DatasetManager {
  private rawText: string = '';
  private tokenizer: Tokenizer | null = null;
  private tokenizerType: TokenizerType = 'char';
  private splits: DatasetSplit | null = null;
  private datasetName: string = 'tinyshakespeare';
  private dataDir: string;

  /**
   * Seeded generator for training-batch sampling. Reproducible batch order is
   * what lets two runs of the same config be compared.
   */
  private batchRng: Rng;

  /**
   * FIXED validation batch offsets, chosen once when the dataset loads.
   *
   * WHY THIS MATTERS A LOT
   * If validation loss is measured on freshly random batches each time, then
   * consecutive measurements differ both because the model changed AND because
   * the data changed. The val curve becomes visibly noisier than the train
   * curve, and small genuine improvements are invisible under the noise. Worse,
   * comparing the final val loss of two runs partly compares which batches
   * each happened to draw.
   *
   * Evaluating on a FIXED set of windows removes that second source of
   * variation entirely: every measurement, in every run, scores the same text.
   */
  private evalOffsets: number[] = [];

  constructor(customDataDir?: string, dataSeed: number = 1337) {
    this.dataDir = customDataDir || path.resolve(process.cwd(), 'data');
    if (!fs.existsSync(this.dataDir)) {
      fs.mkdirSync(this.dataDir, { recursive: true });
    }
    this.batchRng = new Rng(dataSeed);
  }

  /** Re-seeds the batch sampler, e.g. when the user changes `dataSeed`. */
  public setDataSeed(seed: number): void {
    this.batchRng = new Rng(seed);
  }

  /**
   * Downloads the Tiny Shakespeare dataset if it does not already exist locally.
   */
  public async ensureTinyShakespeare(onProgress?: (pct: number) => void): Promise<string> {
    const targetFile = path.join(this.dataDir, 'tinyshakespeare.txt');

    if (fs.existsSync(targetFile)) {
      if (onProgress) onProgress(100);
      return targetFile;
    }

    const url =
      'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt';

    return this.downloadTo(url, targetFile, onProgress);
  }

  /**
   * Streams a URL to a local file, following redirects.
   *
   * Downloads to a `.part` file and renames on success, so an interrupted
   * download can never leave a truncated file that later looks valid. The
   * previous implementation wrote in place, so a dropped connection produced a
   * half-written dataset that loaded without error and trained on garbage.
   */
  private downloadTo(
    url: string,
    targetFile: string,
    onProgress?: (pct: number) => void,
    redirectsLeft: number = 5
  ): Promise<string> {
    const tempFile = `${targetFile}.part`;

    return new Promise((resolve, reject) => {
      const cleanup = () => {
        fs.promises.unlink(tempFile).catch(() => undefined);
      };

      const fileStream = fs.createWriteStream(tempFile);

      const request = https.get(url, (response) => {
        const status = response.statusCode ?? 0;

        // Gutenberg and GitHub both redirect; follow up to a bounded depth.
        if (status >= 300 && status < 400 && response.headers.location) {
          response.resume();
          fileStream.destroy();
          cleanup();
          if (redirectsLeft <= 0) {
            reject(new Error('Too many redirects while downloading dataset.'));
            return;
          }
          const next = new URL(response.headers.location, url).toString();
          this.downloadTo(next, targetFile, onProgress, redirectsLeft - 1).then(resolve, reject);
          return;
        }

        if (status !== 200) {
          response.resume();
          fileStream.destroy();
          cleanup();
          reject(new Error(`Failed to download dataset: HTTP status ${status}`));
          return;
        }

        const totalBytes = parseInt(response.headers['content-length'] || '0', 10);
        let downloadedBytes = 0;

        response.on('data', (chunk: Buffer) => {
          downloadedBytes += chunk.length;
          if (totalBytes > 0 && onProgress) {
            onProgress(Math.round((downloadedBytes / totalBytes) * 100));
          }
        });

        response.pipe(fileStream);

        fileStream.on('finish', () => {
          fileStream.close(() => {
            try {
              fs.renameSync(tempFile, targetFile);
              if (onProgress) onProgress(100);
              resolve(targetFile);
            } catch (err) {
              cleanup();
              reject(err);
            }
          });
        });

        response.on('error', (err) => {
          fileStream.destroy();
          cleanup();
          reject(err);
        });
      });

      request.on('error', (err) => {
        fileStream.destroy();
        cleanup();
        reject(err);
      });

      // Without a timeout, a stalled connection hangs the server's startup
      // path forever with no diagnostic.
      request.setTimeout(60_000, () => {
        request.destroy(new Error('Dataset download timed out after 60s.'));
      });
    });
  }

  /**
   * Download a dataset from a URL and load it.
   */
  public async loadFromUrl(
    url: string,
    name?: string,
    tokenizerType: TokenizerType = 'char'
  ): Promise<void> {
    const derivedName = name || path.basename(new URL(url).pathname, path.extname(url));
    // `path.basename` strips any directory component, so a crafted name cannot
    // escape the data directory.
    const safeName = path.basename(derivedName).replace(/[^a-zA-Z0-9._-]/g, '_');
    const targetPath = path.join(this.dataDir, `${safeName}.txt`);

    if (!fs.existsSync(targetPath)) {
      await this.downloadTo(url, targetPath);
    }

    this.loadFromFile(targetPath, 0.8, 0.1, 0.1, tokenizerType);
  }

  /**
   * Loads a text file from disk, builds the tokenizer, and divides the data
   * strictly into train / val / test splits.
   */
  public loadFromFile(
    filePath: string,
    trainRatio: number = 0.8,
    valRatio: number = 0.1,
    testRatio: number = 0.1,
    tokenizerType: TokenizerType = 'char'
  ): void {
    if (!fs.existsSync(filePath)) {
      throw new Error(`Dataset file not found at: ${filePath}`);
    }

    const text = fs.readFileSync(filePath, 'utf-8');
    this.loadFromString(
      text,
      path.basename(filePath, path.extname(filePath)),
      trainRatio,
      valRatio,
      testRatio,
      tokenizerType
    );
  }

  /**
   * Load text from a string in memory (custom user input, or tests).
   */
  public loadFromString(
    text: string,
    name: string = 'custom_text',
    trainRatio: number = 0.8,
    valRatio: number = 0.1,
    testRatio: number = 0.1,
    tokenizerType: TokenizerType = 'char'
  ): void {
    const ratioSum = trainRatio + valRatio + testRatio;
    if (Math.abs(ratioSum - 1) > 1e-6) {
      throw new Error(
        `Split ratios must sum to 1.0 (received ${trainRatio} + ${valRatio} + ${testRatio} = ${ratioSum}).`
      );
    }
    if (text.length === 0) {
      throw new Error('Dataset text is empty.');
    }

    // Normalize line endings. A Windows-authored file contains \r\n, which a
    // character tokenizer would treat as TWO vocabulary entries, inflating the
    // vocab and teaching the model to emit a stray \r before every newline.
    this.rawText = text.replace(/\r\n/g, '\n');
    this.datasetName = name;
    this.tokenizerType = tokenizerType;

    this.tokenizer =
      tokenizerType === 'word'
        ? new WordTokenizer(this.rawText)
        : new CharTokenizer(this.rawText);

    const allTokens = this.tokenizer.encode(this.rawText);
    const totalTokens = allTokens.length;

    const trainEnd = Math.floor(totalTokens * trainRatio);
    const valEnd = Math.floor(totalTokens * (trainRatio + valRatio));

    this.splits = {
      train: new Int32Array(allTokens.slice(0, trainEnd)),
      val: new Int32Array(allTokens.slice(trainEnd, valEnd)),
      test: new Int32Array(allTokens.slice(valEnd)),
    };

    // Fixed offsets are invalidated by a new dataset; they are rebuilt lazily
    // on the next call to `getEvalBatch`.
    this.evalOffsets = [];
  }

  public getTokenizerType(): TokenizerType {
    return this.tokenizerType;
  }

  /**
   * Returns the statistical breakdown shown in the Data view.
   */
  public getStatistics(): DatasetStatistics {
    if (!this.tokenizer || !this.splits) {
      return {
        name: 'loading',
        totalCharacters: 0,
        uniqueCharacters: 0,
        trainCharacters: 0,
        valCharacters: 0,
        testCharacters: 0,
        samplePreview: '',
        charDistribution: [],
      };
    }

    const charCounts = new Map<string, number>();
    for (let i = 0; i < this.rawText.length; i++) {
      const c = this.rawText[i];
      charCounts.set(c, (charCounts.get(c) || 0) + 1);
    }

    const total = this.rawText.length;
    const sortedDist = Array.from(charCounts.entries())
      .map(([char, count]) => ({
        char: char === '\n' ? '\\n' : char === ' ' ? '␣' : char === '\t' ? '\\t' : char,
        count,
        percentage: Number(((count / total) * 100).toFixed(2)),
      }))
      .sort((a, b) => b.count - a.count);

    return {
      name: this.datasetName,
      totalCharacters: total,
      uniqueCharacters: this.tokenizer.vocabSize,
      trainCharacters: this.splits.train.length,
      valCharacters: this.splits.val.length,
      testCharacters: this.splits.test.length,
      samplePreview: this.rawText.slice(0, 1200),
      charDistribution: sortedDist.slice(0, 40),
    };
  }

  /**
   * Samples a RANDOM mini-batch of (X, Y) sequences, for training.
   * Draws from the seeded generator, so batch order is reproducible.
   */
  public getBatch(
    splitType: 'train' | 'val' | 'test',
    batchSize: number,
    contextLength: number
  ): { x: tf.Tensor2D; y: tf.Tensor2D } {
    const data = this.requireSplit(splitType, contextLength);
    const maxStart = data.length - contextLength - 1;

    const offsets: number[] = new Array(batchSize);
    for (let b = 0; b < batchSize; b++) {
      offsets[b] = this.batchRng.nextInt(maxStart);
    }

    return this.buildBatch(data, offsets, contextLength);
  }

  /**
   * Returns a DETERMINISTIC validation/test batch.
   *
   * The offsets are computed once, spread evenly across the split, and reused
   * for every evaluation. `batchIndex` selects which slice of those fixed
   * offsets to use, so `validationBatchCount` consecutive calls cover a fixed,
   * repeatable sample of the split. See the note on `evalOffsets` above for why
   * this is important.
   */
  public getEvalBatch(
    splitType: 'val' | 'test',
    batchSize: number,
    contextLength: number,
    batchIndex: number,
    totalBatches: number
  ): { x: tf.Tensor2D; y: tf.Tensor2D } {
    const data = this.requireSplit(splitType, contextLength);
    const maxStart = data.length - contextLength - 1;
    const needed = batchSize * totalBatches;

    // Build the fixed offset list on first use: evenly spaced windows covering
    // the whole split, so the evaluation sample is representative rather than
    // concentrated at the start.
    if (this.evalOffsets.length < needed) {
      const stride = Math.max(1, Math.floor(maxStart / needed));
      this.evalOffsets = [];
      for (let i = 0; i < needed; i++) {
        this.evalOffsets.push((i * stride) % maxStart);
      }
    }

    const start = batchIndex * batchSize;
    const offsets = this.evalOffsets.slice(start, start + batchSize);
    return this.buildBatch(data, offsets, contextLength);
  }

  /** Validates that a split exists and is long enough for the context window. */
  private requireSplit(
    splitType: 'train' | 'val' | 'test',
    contextLength: number
  ): Int32Array {
    if (!this.splits) {
      throw new Error('Dataset is not loaded.');
    }
    const data = this.splits[splitType];
    if (data.length - contextLength - 1 <= 0) {
      throw new Error(
        `Data split '${splitType}' holds ${data.length} tokens, which is too few for ` +
          `contextLength ${contextLength}. Use a larger dataset or a shorter context.`
      );
    }
    return data;
  }

  /** Materializes (X, Y) tensors for a given list of window start offsets. */
  private buildBatch(
    data: Int32Array,
    offsets: number[],
    contextLength: number
  ): { x: tf.Tensor2D; y: tf.Tensor2D } {
    const batchSize = offsets.length;
    const xBuffer = new Int32Array(batchSize * contextLength);
    const yBuffer = new Int32Array(batchSize * contextLength);

    for (let b = 0; b < batchSize; b++) {
      const startIdx = offsets[b];
      for (let i = 0; i < contextLength; i++) {
        const destIdx = b * contextLength + i;
        xBuffer[destIdx] = data[startIdx + i];
        // The target is simply the next token - this is the whole supervision
        // signal in autoregressive language modelling.
        yBuffer[destIdx] = data[startIdx + i + 1];
      }
    }

    return {
      x: tf.tensor2d(xBuffer, [batchSize, contextLength], 'int32'),
      y: tf.tensor2d(yBuffer, [batchSize, contextLength], 'int32'),
    };
  }

  public getTokenizer(): Tokenizer {
    if (!this.tokenizer) {
      throw new Error('Tokenizer not initialized. Load a dataset first.');
    }
    return this.tokenizer;
  }

  public isLoaded(): boolean {
    return this.tokenizer !== null && this.splits !== null;
  }

  public getSplits(): DatasetSplit {
    if (!this.splits) {
      throw new Error('Dataset splits not initialized.');
    }
    return this.splits;
  }

  public getRawText(): string {
    return this.rawText;
  }

  public getName(): string {
    return this.datasetName;
  }
}
