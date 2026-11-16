// src/backend/dataset/wordTokenizer.ts

/**
 * ============================================================================
 * MiniGPT Lab - Word-Level Tokenizer
 * ============================================================================
 *
 * What is a Word Tokenizer?
 * --------------------------
 * Instead of treating every single character as a token ("a", "b", "c"...),
 * a word tokenizer groups characters into meaningful lexical units – words.
 *
 * Why switch to word-level tokens?
 * ------------------------------------------------
 * - **Faster training**: Fewer tokens per sequence → shorter context length.
 * - **More natural output**: The model directly predicts whole words, so generated
 *   text reads like normal sentences without having to concatenate characters.
 * - **Educational value**: Demonstrates how the tokenization choice influences
 *   model size, memory usage, and the quality of the learned language.
 *
 * How it works internally
 * -----------------------
 * 1. **Build Vocabulary** – Scan the entire corpus, split on whitespace, and
 *    collect the unique words. The vocabulary is stored as an ordered array so
 *    that each word receives a deterministic integer ID.
 * 2. **Encode** – Convert a string into an array of IDs by looking up each word
 *    in the `wordToId` map. Unknown words are mapped to a special `<UNK>` token.
 * 3. **Decode** – Reverse the process: turn an array of IDs back into a string
 *    of words separated by a single space.
 *
 * The implementation mirrors the `CharTokenizer` API so that the rest of the
 * codebase (dataset manager, training loop, tests) can stay unchanged.
 */

import { Tokenizer } from './types';

export class WordTokenizer implements Tokenizer {
  /** Ordered list of all unique words known to the tokenizer. */
  private readonly vocab: string[];

  /** Fast lookup: word → integer token ID. */
  private readonly wordToId: Map<string, number>;

  /** Fast reverse lookup: integer token ID → word. */
  private readonly idToWord: string[];

  /** Placeholder token for any word not present in the vocabulary. */
  private readonly unkToken: string = '<UNK>';

  /**
   * Create a tokenizer from either a raw text corpus or an explicit word list.
   *
   * @param sourceTextOrVocab Either the complete dataset text (string) or an
   *                          array of pre‑computed unique words.
   */
  constructor(sourceTextOrVocab: string | string[]) {
    if (Array.isArray(sourceTextOrVocab)) {
      // User supplied a ready‑made vocabulary.
      this.vocab = [...sourceTextOrVocab];
    } else {
      // Derive vocabulary from raw text.
      // 1️⃣ Split on whitespace (spaces, newlines, tabs).
      //    Using a simple RegExp keeps the implementation lightweight.
      const rawWords = sourceTextOrVocab.split(/\s+/).filter(Boolean);
      // 2️⃣ Gather unique words and sort them alphabetically for deterministic IDs.
      const uniqueWords = Array.from(new Set(rawWords)).sort();
      // 3️⃣ Ensure the unknown token is always present at the beginning of the vocab.
      //    This guarantees ID 0 → <UNK> (a common convention).
      this.vocab = [this.unkToken, ...uniqueWords];
    }

    // Build lookup tables.
    this.wordToId = new Map<string, number>();
    this.idToWord = [];
    this.vocab.forEach((word, idx) => {
      this.wordToId.set(word, idx);
      this.idToWord[idx] = word;
    });
  }

  /** Number of distinct words the tokenizer knows. */
  public get vocabSize(): number {
    return this.vocab.length;
  }

  /** Return a copy of the vocabulary array. */
  public getVocabulary(): string[] {
    return [...this.vocab];
  }

  /** Encode a plain‑text string into an array of integer IDs. */
  public encode(text: string): number[] {
    const words = text.split(/\s+/).filter(Boolean);
    const tokens: number[] = new Array(words.length);
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      let id = this.wordToId.get(w);
      if (id === undefined) {
        // Unknown word → map to the special <UNK> token (ID 0).
        id = this.wordToId.get(this.unkToken) ?? 0;
      }
      tokens[i] = id;
    }
    return tokens;
  }

  /** Decode an array of token IDs back into a human‑readable string. */
  public decode(tokens: number[]): string {
    const words: string[] = [];
    for (let i = 0; i < tokens.length; i++) {
      const id = tokens[i];
      const w = this.idToWord[id];
      words.push(w !== undefined ? w : this.unkToken);
    }
    return words.join(' ');
  }

  /** Serialize the vocabulary so it can be saved alongside model checkpoints. */
  public toJSON(): { vocab: string[] } {
    return { vocab: this.vocab };
  }

  /** Recreate a `WordTokenizer` from a previously saved JSON payload. */
  public static fromJSON(data: { vocab: string[] }): WordTokenizer {
    return new WordTokenizer(data.vocab);
  }
}
