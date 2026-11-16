/**
 * ============================================================================
 * MiniGPT Lab - Tokenizer Contract
 * ============================================================================
 *
 * Both the character-level and word-level tokenizers implement this interface,
 * so the dataset manager, training engine and generator never need to know
 * which one is in use. Swapping tokenization is then a configuration choice
 * rather than a code change - which is the point, because the vocabulary-size
 * versus sequence-length tradeoff is one of the most instructive experiments
 * this tool can offer.
 */
export interface Tokenizer {
  /** Number of distinct tokens. */
  readonly vocabSize: number;

  /** A copy of the ordered vocabulary; index === token ID. */
  getVocabulary(): string[];

  /** Text -> token IDs. */
  encode(text: string): number[];

  /** Token IDs -> text. */
  decode(tokens: number[]): string;

  /** Serializable form, embedded in checkpoints. */
  toJSON(): { vocab: string[] };
}

/** Which tokenization strategy a dataset was loaded with. */
export type TokenizerType = 'char' | 'word';
