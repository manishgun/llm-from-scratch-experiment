import { describe, it, expect, beforeAll } from 'vitest';
import * as os from 'os';
import * as path from 'path';
import { CharTokenizer } from '../src/backend/dataset/tokenizer';
import { WordTokenizer } from '../src/backend/dataset/wordTokenizer';
import { DatasetManager } from '../src/backend/dataset/datasetManager';
import { Rng } from '../src/backend/core/rng';
import { normalVariable, residualStd } from '../src/backend/core/init';

/**
 * ============================================================================
 * Tokenization, Dataset Splitting, Batching and Reproducibility
 * ============================================================================
 */

const SAMPLE_TEXT =
  'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n';

const TEST_DATA_DIR = path.join(os.tmpdir(), 'minigpt-test-data');

describe('Character tokenizer', () => {
  it('round-trips text exactly', () => {
    const tokenizer = new CharTokenizer(SAMPLE_TEXT);
    const encoded = tokenizer.encode('Before we proceed');

    expect(encoded).toHaveLength('Before we proceed'.length);
    expect(tokenizer.decode(encoded)).toBe('Before we proceed');
  });

  it('assigns IDs deterministically by sorted character order', () => {
    // Deterministic IDs are what make a checkpoint's vocabulary portable: two
    // runs over the same text must agree on what token 7 means.
    const a = new CharTokenizer('cba');
    const b = new CharTokenizer('abc');
    expect(a.getVocabulary()).toEqual(b.getVocabulary());
    expect(a.encode('abc')).toEqual(b.encode('abc'));
  });

  it('maps unknown characters to a fallback without changing length', () => {
    const tokenizer = new CharTokenizer('abc');
    const encoded = tokenizer.encode('abcd');
    expect(encoded).toHaveLength(4);
    expect(tokenizer.decode(encoded)).toHaveLength(4);
  });

  it('survives a serialization round trip', () => {
    const original = new CharTokenizer(SAMPLE_TEXT);
    const restored = CharTokenizer.fromJSON(original.toJSON());

    expect(restored.vocabSize).toBe(original.vocabSize);
    expect(restored.encode('Speak')).toEqual(original.encode('Speak'));
  });

  it('builds a vocabulary of exactly the distinct characters present', () => {
    const tokenizer = new CharTokenizer('aabbcc');
    expect(tokenizer.vocabSize).toBe(3);
  });
});

describe('Word tokenizer', () => {
  it('reserves token 0 for the unknown marker', () => {
    const tokenizer = new WordTokenizer('alpha beta gamma');
    expect(tokenizer.getVocabulary()[0]).toBe('<UNK>');
    expect(tokenizer.encode('delta')).toEqual([0]);
  });

  it('produces far fewer tokens per sentence than the character tokenizer', () => {
    // This is the central tokenization tradeoff: a word vocabulary is much
    // larger but each sequence is much shorter, so a fixed context window
    // covers far more text.
    const text = 'the quick brown fox jumps over the lazy dog';
    const charCount = new CharTokenizer(text).encode(text).length;
    const wordCount = new WordTokenizer(text).encode(text).length;

    expect(wordCount).toBe(9);
    expect(charCount).toBe(text.length);
    expect(wordCount).toBeLessThan(charCount / 4);
  });

  it('grows its vocabulary with corpus size, where the character tokenizer saturates', () => {
    // This is the other half of the tradeoff, and the reason character-level
    // is the right default for a CPU tool. The character vocabulary is bounded
    // by the alphabet, so it stops growing almost immediately. The word
    // vocabulary keeps growing with every new corpus, and vocabulary size
    // drives the two largest tensors in the model (the embedding table and the
    // output head) as well as the cost of the final softmax.
    const short = 'the quick brown fox jumps over the lazy dog';
    const long = short + ' ' + 'alpha beta gamma delta epsilon zeta eta theta iota kappa';

    const charShort = new CharTokenizer(short).vocabSize;
    const charLong = new CharTokenizer(long).vocabSize;
    const wordShort = new WordTokenizer(short).vocabSize;
    const wordLong = new WordTokenizer(long).vocabSize;

    // Adding ten new words adds ten vocabulary entries.
    expect(wordLong - wordShort).toBe(10);
    // Adding the same text adds almost no new characters - the alphabet was
    // already covered.
    expect(charLong - charShort).toBeLessThan(3);
  });
});

describe('Dataset splitting', () => {
  let dm: DatasetManager;

  beforeAll(() => {
    dm = new DatasetManager(TEST_DATA_DIR);
    dm.loadFromString(SAMPLE_TEXT.repeat(20), 'test_sample', 0.8, 0.1, 0.1);
  });

  it('partitions every token into exactly one split', () => {
    const splits = dm.getSplits();
    const total = splits.train.length + splits.val.length + splits.test.length;
    expect(total).toBe(dm.getStatistics().totalCharacters);
  });

  it('cuts contiguously, so no window can appear in two splits', () => {
    // A random per-window assignment would let a training window ending at
    // character 1000 and a validation window starting at 990 share ten
    // characters - so validation loss would be measured partly on memorised
    // text. A straight cut makes that impossible.
    const splits = dm.getSplits();
    const raw = dm.getRawText();
    const tokenizer = dm.getTokenizer();
    const allTokens = tokenizer.encode(raw);

    for (let i = 0; i < splits.train.length; i++) {
      expect(splits.train[i]).toBe(allTokens[i]);
    }
    const valStart = splits.train.length;
    for (let i = 0; i < splits.val.length; i++) {
      expect(splits.val[i]).toBe(allTokens[valStart + i]);
    }
  });

  it('rejects split ratios that do not sum to 1', () => {
    const bad = new DatasetManager(TEST_DATA_DIR);
    expect(() => bad.loadFromString(SAMPLE_TEXT, 'bad', 0.5, 0.2, 0.1)).toThrow(/sum to 1\.0/);
  });

  it('rejects empty text', () => {
    const bad = new DatasetManager(TEST_DATA_DIR);
    expect(() => bad.loadFromString('', 'empty')).toThrow(/empty/);
  });

  it('normalizes CRLF so Windows files do not inflate the vocabulary', () => {
    // Left as-is, a Windows-authored file makes the character tokenizer treat
    // \r as its own token, and the model learns to emit a stray \r before
    // every newline.
    const dmCrlf = new DatasetManager(TEST_DATA_DIR);
    dmCrlf.loadFromString('line one\r\nline two\r\nline three\r\n'.repeat(10), 'crlf');
    expect(dmCrlf.getRawText()).not.toContain('\r');
    expect(dmCrlf.getTokenizer().getVocabulary()).not.toContain('\r');
  });

  it('reports a character distribution that sums to about 100%', () => {
    const stats = dm.getStatistics();
    const totalPercent = stats.charDistribution.reduce((acc, e) => acc + e.percentage, 0);
    expect(totalPercent).toBeGreaterThan(95);
    expect(totalPercent).toBeLessThanOrEqual(100.5);
  });
});

describe('Batch sampling', () => {
  let dm: DatasetManager;

  beforeAll(() => {
    dm = new DatasetManager(TEST_DATA_DIR, 1337);
    dm.loadFromString(SAMPLE_TEXT.repeat(50), 'test_sample', 0.8, 0.1, 0.1);
  });

  it('produces X and Y of shape [batchSize, contextLength]', () => {
    const { x, y } = dm.getBatch('train', 4, 8);
    expect(x.shape).toEqual([4, 8]);
    expect(y.shape).toEqual([4, 8]);
    x.dispose();
    y.dispose();
  });

  it('shifts Y one position ahead of X - the whole supervision signal', async () => {
    // X[i] predicts Y[i], and Y[i] is simply the token after X[i]. This is why
    // one forward pass over a length-T window yields T training examples.
    const { x, y } = dm.getBatch('train', 2, 8);
    const xs = (await x.array()) as number[][];
    const ys = (await y.array()) as number[][];

    for (let b = 0; b < 2; b++) {
      for (let i = 0; i < 7; i++) {
        expect(ys[b][i]).toBe(xs[b][i + 1]);
      }
    }

    x.dispose();
    y.dispose();
  });

  it('is reproducible for a fixed data seed', async () => {
    // Without this, two runs of the same config differ for two reasons at once
    // - your change AND the batch order - so no comparison can attribute the
    // difference to anything.
    const a = new DatasetManager(TEST_DATA_DIR, 777);
    a.loadFromString(SAMPLE_TEXT.repeat(50), 's', 0.8, 0.1, 0.1);
    const b = new DatasetManager(TEST_DATA_DIR, 777);
    b.loadFromString(SAMPLE_TEXT.repeat(50), 's', 0.8, 0.1, 0.1);

    const batchA = a.getBatch('train', 4, 8);
    const batchB = b.getBatch('train', 4, 8);

    expect(await batchA.x.array()).toEqual(await batchB.x.array());

    const c = new DatasetManager(TEST_DATA_DIR, 778);
    c.loadFromString(SAMPLE_TEXT.repeat(50), 's', 0.8, 0.1, 0.1);
    const batchC = c.getBatch('train', 4, 8);
    expect(await batchC.x.array()).not.toEqual(await batchA.x.array());

    for (const batch of [batchA, batchB, batchC]) {
      batch.x.dispose();
      batch.y.dispose();
    }
  });

  it('returns the SAME validation batch every time it is asked', async () => {
    // Evaluation batches are fixed, not random. Otherwise consecutive
    // validation measurements differ both because the model changed and
    // because the data changed, and small genuine improvements vanish into the
    // noise.
    const first = dm.getEvalBatch('val', 2, 8, 0, 2);
    const firstValues = await first.x.array();
    first.x.dispose();
    first.y.dispose();

    // Draw many random training batches in between, to advance the sampler.
    for (let i = 0; i < 5; i++) {
      const noise = dm.getBatch('train', 4, 8);
      noise.x.dispose();
      noise.y.dispose();
    }

    const second = dm.getEvalBatch('val', 2, 8, 0, 2);
    expect(await second.x.array()).toEqual(firstValues);
    second.x.dispose();
    second.y.dispose();
  });

  it('gives different eval batches for different batch indices', async () => {
    const zero = dm.getEvalBatch('val', 2, 8, 0, 2);
    const one = dm.getEvalBatch('val', 2, 8, 1, 2);
    expect(await zero.x.array()).not.toEqual(await one.x.array());

    for (const batch of [zero, one]) {
      batch.x.dispose();
      batch.y.dispose();
    }
  });

  it('explains clearly when a split is too short for the context length', () => {
    const small = new DatasetManager(TEST_DATA_DIR);
    small.loadFromString('abcdefghij'.repeat(4), 'small', 0.8, 0.1, 0.1);
    expect(() => small.getBatch('val', 2, 512)).toThrow(/too few for contextLength/);
  });

  it('never reads past the end of a split', async () => {
    // An off-by-one here would silently read undefined (becoming 0) at the
    // boundary, teaching the model that every split ends with token 0.
    const { x, y } = dm.getBatch('val', 8, 16);
    const xs = (await x.array()) as number[][];
    const ys = (await y.array()) as number[][];
    const vocabSize = dm.getTokenizer().vocabSize;

    for (const row of [...xs, ...ys]) {
      for (const token of row) {
        expect(Number.isInteger(token)).toBe(true);
        expect(token).toBeGreaterThanOrEqual(0);
        expect(token).toBeLessThan(vocabSize);
      }
    }

    x.dispose();
    y.dispose();
  });
});

describe('Seeded random number generation', () => {
  it('reproduces an identical sequence for the same seed', () => {
    const a = new Rng(42);
    const b = new Rng(42);
    for (let i = 0; i < 100; i++) {
      expect(a.next()).toBe(b.next());
    }
  });

  it('produces different sequences for different seeds', () => {
    const a = new Rng(1);
    const b = new Rng(2);
    const sequenceA = Array.from({ length: 20 }, () => a.next());
    const sequenceB = Array.from({ length: 20 }, () => b.next());
    expect(sequenceA).not.toEqual(sequenceB);
  });

  it('returns values strictly within [0, 1)', () => {
    const rng = new Rng(7);
    for (let i = 0; i < 10_000; i++) {
      const value = rng.next();
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it('is approximately uniform', () => {
    // Ten buckets over 50,000 draws should each hold close to 10%.
    const rng = new Rng(11);
    const buckets = new Array(10).fill(0);
    const draws = 50_000;
    for (let i = 0; i < draws; i++) {
      buckets[Math.floor(rng.next() * 10)]++;
    }
    for (const count of buckets) {
      expect(count / draws).toBeCloseTo(0.1, 1);
    }
  });

  it('generates normal samples with the requested mean and standard deviation', () => {
    const rng = new Rng(3);
    const samples = rng.normalArray(50_000, 2, 0.5);

    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const variance =
      samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;

    expect(mean).toBeCloseTo(2, 1);
    expect(Math.sqrt(variance)).toBeCloseTo(0.5, 1);
  });

  it('reset rewinds to the start of the sequence', () => {
    const rng = new Rng(5);
    const first = [rng.next(), rng.next(), rng.next()];
    rng.reset();
    expect([rng.next(), rng.next(), rng.next()]).toEqual(first);
  });

  it('derive produces a stable, label-dependent independent stream', () => {
    // Derived streams are why adding a layer does not shift the random numbers
    // used by the layers around it - so a 3-layer and a 4-layer model share
    // their common tensors' initial weights, making depth comparisons cleaner.
    const master = new Rng(42);
    const a1 = master.derive('layer_0_w_q').next();
    const a2 = master.derive('layer_0_w_q').next();
    const b = master.derive('layer_1_w_q').next();

    expect(a1).toBe(a2);
    expect(a1).not.toBe(b);
  });
});

describe('Weight initialization', () => {
  it('produces the configured standard deviation', async () => {
    const variable = normalVariable([200, 200], 0.02, 'test_w', new Rng(1));
    const values = (await variable.data()) as Float32Array;

    const mean = Array.from(values).reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(
      Array.from(values).reduce((a, b) => a + (b - mean) ** 2, 0) / values.length
    );

    expect(mean).toBeCloseTo(0, 2);
    expect(std).toBeCloseTo(0.02, 3);
    variable.dispose();
  });

  it('is bit-identical for the same seed', async () => {
    const a = normalVariable([50, 50], 0.02, 'a', new Rng(9));
    const b = normalVariable([50, 50], 0.02, 'b', new Rng(9));

    expect(Array.from((await a.data()) as Float32Array)).toEqual(
      Array.from((await b.data()) as Float32Array)
    );

    a.dispose();
    b.dispose();
  });

  it('scales residual-path initialization down as depth grows', () => {
    // GPT-2's depth scaling: layers that WRITE INTO the residual stream are
    // initialized at std / sqrt(2 * numLayers), because every block adds to a
    // running sum. Without it, activation variance grows with depth and deep
    // configurations become unstable - so a user would wrongly conclude that
    // deep models simply do not work here.
    expect(residualStd(0.02, 1)).toBeCloseTo(0.02 / Math.sqrt(2), 6);
    expect(residualStd(0.02, 8)).toBeCloseTo(0.02 / Math.sqrt(16), 6);
    expect(residualStd(0.02, 8)).toBeLessThan(residualStd(0.02, 2));
  });
});
