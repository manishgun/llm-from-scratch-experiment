// src/backend/dataset/tokenizer.ts

/**
 * ============================================================================
 * MiniGPT Lab - Character-Level Tokenizer (fallback implementation)
 * ============================================================================
 *
 * This tokenizer provides a minimal character‑level tokenization that's used as a
 * fallback for legacy code paths (e.g., `DatasetManager.loadFromString`).  The
 * primary model now uses `WordTokenizer`, but keeping a simple character tokenizer
 * ensures existing imports continue to work without breaking the UI.
 */

import { Tokenizer } from './types';

export class CharTokenizer implements Tokenizer {
  /** List of unique characters (vocabulary) */
  private readonly vocab: string[];
  /** Mapping: character → integer ID */
  private readonly charToId: Map<string, number>;
  /** Reverse mapping: ID → character */
  private readonly idToChar: string[];
  /** Token used for unknown characters */
  private readonly unkChar: string = '?';

  /**
   * Construct the tokenizer from raw text or an explicit array of characters.
   * @param sourceTextOrVocab raw corpus string or predefined character list
   */
  constructor(sourceTextOrVocab: string | string[]) {
    if (Array.isArray(sourceTextOrVocab)) {
      this.vocab = [...sourceTextOrVocab];
    } else {
      // Extract unique characters from the text and sort for deterministic IDs
      const uniqueChars = Array.from(new Set(sourceTextOrVocab.split(''))).sort();
      this.vocab = uniqueChars;
    }
    // Build lookup tables
    this.charToId = new Map<string, number>();
    this.idToChar = [];
    this.vocab.forEach((c, i) => {
      this.charToId.set(c, i);
      this.idToChar[i] = c;
    });
  }

  /** Number of distinct characters */
  public get vocabSize(): number {
    return this.vocab.length;
  }

  /** Return a copy of the vocabulary */
  public getVocabulary(): string[] {
    return [...this.vocab];
  }

  /** Encode a string into an array of character IDs */
  public encode(text: string): number[] {
    const tokens: number[] = new Array(text.length);
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      let id = this.charToId.get(ch);
      if (id === undefined) {
        id = this.charToId.get(this.unkChar) ?? 0;
      }
      tokens[i] = id;
    }
    return tokens;
  }

  /** Decode an array of IDs back into a string */
  public decode(tokens: number[]): string {
    let result = '';
    for (let i = 0; i < tokens.length; i++) {
      const id = tokens[i];
      const ch = this.idToChar[id];
      result += ch !== undefined ? ch : this.unkChar;
    }
    return result;
  }

  /** Serialize the vocabulary for checkpoint saving */
  public toJSON(): { vocab: string[] } {
    return { vocab: this.vocab };
  }

  /** Recreate a tokenizer from saved JSON */
  public static fromJSON(data: { vocab: string[] }): CharTokenizer {
    return new CharTokenizer(data.vocab);
  }
}
