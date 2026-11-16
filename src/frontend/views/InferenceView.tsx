import React, { useMemo, useState } from 'react';
import { useStore } from '../lib/store';
import {
  Panel,
  Button,
  Field,
  SliderInput,
  TextArea,
  TextInput,
  Badge,
  EmptyState,
  Note,
  Toggle,
  Stat,
} from '../components/ui';
import { BarList } from '../components/charts';
import * as fmt from '../lib/format';
import { GeneratedToken } from '../../shared/types';

/**
 * ============================================================================
 * Inference - streaming generation with a probability inspector
 * ============================================================================
 *
 * Two things make this more than a text box.
 *
 * STREAMING. Tokens appear one at a time, as the model produces them. This is
 * not only nicer to wait through - it is an accurate depiction of how
 * autoregressive generation works, and it makes the Stop button meaningful
 * mid-generation rather than after `maxTokens`.
 *
 * THE INSPECTOR. Clicking any generated character shows the full distribution
 * the model was sampling from at that position: the top candidates with their
 * probabilities, and the entropy of the choice. This is the single most
 * instructive view in the application, because it turns temperature, top-k and
 * top-p from abstract numbers into visible changes in a list - and it shows
 * where the model was confident versus guessing.
 */

export const InferenceView: React.FC = () => {
  const { config, draftConfig, setDraftConfig, stream, startStream, cancelStream, status, step, parameters } =
    useStore();

  const [prompt, setPrompt] = useState(config?.generation.samplePrompt ?? 'ROMEO:');
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [useSeed, setUseSeed] = useState(false);
  const [seed, setSeed] = useState(1234);

  const generation = draftConfig?.generation;

  const updateGeneration = (patch: Partial<NonNullable<typeof generation>>) => {
    if (!draftConfig) return;
    setDraftConfig({ ...draftConfig, generation: { ...draftConfig.generation, ...patch } });
  };

  const hasModel = parameters !== null && parameters > 0;
  const isUntrained = step === 0;

  const handleGenerate = () => {
    if (!generation) return;
    setSelectedIndex(null);
    startStream(prompt, {
      temperature: generation.temperature,
      topK: generation.topK,
      topP: generation.topP,
      maxTokens: generation.maxTokens,
      repetitionPenalty: generation.repetitionPenalty,
      ...(useSeed ? { seed } : {}),
    });
  };

  const selectedToken: GeneratedToken | null =
    selectedIndex !== null && stream.tokens[selectedIndex] ? stream.tokens[selectedIndex] : null;

  // Entropy over the whole completion, as a single confidence readout.
  const meanEntropy = useMemo(() => {
    if (stream.tokens.length === 0) return null;
    return stream.tokens.reduce((acc, t) => acc + t.entropyBits, 0) / stream.tokens.length;
  }, [stream.tokens]);

  const promptLength = prompt.length;

  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,320px)]">
      {/* ------------------------------------------------------------------ */}
      {/* Left: prompt, output, inspector                                    */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex min-w-0 flex-col gap-3">
        {!hasModel && (
          <Note tone="warning" title="No model in memory">
            Start a training run, or load a checkpoint from the Checkpoints tab, then generate.
          </Note>
        )}
        {hasModel && isUntrained && (
          <Note tone="info" title="This model is untrained">
            Its weights are random, so the output will be noise with roughly uniform character
            frequencies. That is the correct starting point - generate now, train, then generate
            again to see the difference.
          </Note>
        )}

        <Panel
          title="Prompt"
          actions={
            <span className="num text-2xs text-content-tertiary">{promptLength} chars</span>
          }
        >
          <TextArea
            rows={2}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Enter a prompt, or leave empty to start from a newline"
            onKeyDown={(event) => {
              // Cmd/Ctrl+Enter generates, which is the convention for a
              // multi-line field whose Enter key must still insert a newline.
              if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                event.preventDefault();
                if (!stream.active) handleGenerate();
              }
            }}
            className="font-mono"
          />
          <div className="mt-2.5 flex items-center gap-2">
            {stream.active ? (
              <Button variant="secondary" onClick={cancelStream} icon={<StopIcon />}>
                Stop
              </Button>
            ) : (
              <Button
                variant="primary"
                onClick={handleGenerate}
                disabled={!hasModel}
                icon={<SparkIcon />}
              >
                Generate
              </Button>
            )}
            <span className="text-2xs text-content-tertiary">
              or press <kbd className="num">Ctrl</kbd>+<kbd className="num">Enter</kbd>
            </span>
            {stream.active && (
              <span className="num ml-auto text-2xs text-content-tertiary">
                {stream.tokens.length} / {stream.total} tokens
              </span>
            )}
          </div>
        </Panel>

        <Panel
          title="Output"
          description={
            stream.tokens.length > 0
              ? 'Click any generated character to see the distribution it was drawn from.'
              : undefined
          }
          actions={
            stream.tokens.length > 0 ? (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => navigator.clipboard?.writeText(stream.text)}
              >
                Copy
              </Button>
            ) : undefined
          }
        >
          {stream.text.length === 0 && !stream.active ? (
            <EmptyState
              compact
              title="Nothing generated yet"
              description="The prompt is echoed first, then the model's continuation streams in."
            />
          ) : (
            <TokenStream
              prompt={prompt}
              tokens={stream.tokens}
              active={stream.active}
              selectedIndex={selectedIndex}
              onSelect={setSelectedIndex}
            />
          )}

          {stream.error && (
            <Note tone="danger" className="mt-2.5">
              {stream.error}
            </Note>
          )}
        </Panel>

        {/* The inspector: the distribution behind one chosen token. */}
        {selectedToken && (
          <Panel
            title={`Position ${selectedIndex! + 1}: the model chose "${fmt.visibleToken(selectedToken.token)}"`}
            description="The candidates it was sampling from, after temperature, top-k and top-p were applied."
            actions={
              <Button size="sm" variant="ghost" onClick={() => setSelectedIndex(null)}>
                Close
              </Button>
            }
          >
            <div className="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
              <Stat
                label="Chosen probability"
                value={fmt.percent(
                  selectedToken.topCandidates.find((c) => c.chosen)?.probability ?? 0
                )}
              />
              <Stat
                label="Entropy"
                value={selectedToken.entropyBits.toFixed(2)}
                unit="bits"
                title="How uncertain the model was. 0 bits means completely certain; log2(candidates) means it had no preference at all."
              />
              <Stat
                label="Candidates"
                value={selectedToken.topCandidates.length}
                hint="after filtering"
              />
            </div>
            <BarList
              items={selectedToken.topCandidates.map((candidate) => ({
                label: fmt.visibleToken(candidate.token),
                value: candidate.probability,
                display: fmt.percent(candidate.probability),
                color: candidate.chosen ? 'var(--accent)' : 'var(--series-muted)',
                mono: true,
              }))}
            />
            <p className="mt-2.5 text-2xs leading-relaxed text-content-tertiary">
              The highlighted bar is the token that was actually sampled. It is not always the most
              probable one - that is what sampling means, and it is why the same prompt gives
              different text each run unless you fix the seed.
            </p>
          </Panel>
        )}
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Right: sampling controls                                           */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex min-w-0 flex-col gap-3">
        <Panel
          title="Sampling"
          description="These affect inference only. None of them change the model's weights."
        >
          {generation && (
            <div className="flex flex-col gap-3.5">
              <Field
                label="Temperature"
                valueDisplay={generation.temperature.toFixed(2)}
                help="Divides the scores before softmax. Below 1 sharpens the distribution toward the model's favourite; above 1 flattens it toward uniform."
              >
                {(id) => (
                  <SliderInput
                    id={id}
                    value={generation.temperature}
                    min={0.05}
                    max={2}
                    step={0.05}
                    onChange={(value) => updateGeneration({ temperature: value })}
                  />
                )}
              </Field>

              <Field
                label="Top-K"
                valueDisplay={generation.topK === 0 ? 'off' : generation.topK}
                help="Keep only the K highest-scoring tokens. 0 disables it. 1 is greedy decoding, which is deterministic and tends to loop."
              >
                {(id) => (
                  <SliderInput
                    id={id}
                    value={generation.topK}
                    min={0}
                    max={Math.max(10, config?.model.vocabularySize ?? 65)}
                    step={1}
                    onChange={(value) => updateGeneration({ topK: value })}
                  />
                )}
              </Field>

              <Field
                label="Top-P (nucleus)"
                valueDisplay={
                  generation.topP === undefined || generation.topP >= 1
                    ? 'off'
                    : generation.topP.toFixed(2)
                }
                help="Keep the smallest set of tokens whose probabilities sum to P. Adapts to each position, unlike a fixed K."
              >
                {(id) => (
                  <SliderInput
                    id={id}
                    value={generation.topP ?? 1}
                    min={0.1}
                    max={1}
                    step={0.01}
                    onChange={(value) => updateGeneration({ topP: value })}
                  />
                )}
              </Field>

              <Field
                label="Repetition penalty"
                valueDisplay={(generation.repetitionPenalty ?? 1).toFixed(2)}
                help="Divides the score of tokens already present in the text. 1.0 is off. Fixes the looping a small model falls into at low temperature."
              >
                {(id) => (
                  <SliderInput
                    id={id}
                    value={generation.repetitionPenalty ?? 1}
                    min={1}
                    max={2}
                    step={0.05}
                    onChange={(value) => updateGeneration({ repetitionPenalty: value })}
                  />
                )}
              </Field>

              <Field
                label="Max tokens"
                valueDisplay={generation.maxTokens}
                help="How many characters to generate after the prompt."
              >
                {(id) => (
                  <SliderInput
                    id={id}
                    value={generation.maxTokens}
                    min={10}
                    max={1000}
                    step={10}
                    onChange={(value) => updateGeneration({ maxTokens: value })}
                  />
                )}
              </Field>

              <div className="border-t border-edge-subtle pt-3">
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <label className="text-xs font-medium text-content-secondary">
                      Fixed seed
                    </label>
                    <p className="mt-0.5 text-2xs leading-snug text-content-tertiary">
                      Makes generation reproducible, so changing a slider is the only thing that
                      changes the output.
                    </p>
                  </div>
                  <Toggle checked={useSeed} onChange={setUseSeed} label="Use a fixed seed" />
                </div>
                {useSeed && (
                  <TextInput
                    type="number"
                    value={seed}
                    onChange={(event) => setSeed(Number(event.target.value))}
                    className="num mt-2"
                  />
                )}
              </div>
            </div>
          )}
        </Panel>

        {stream.tokens.length > 0 && (
          <Panel title="This completion">
            <dl className="flex flex-col gap-1.5 text-xs">
              <div className="flex justify-between gap-2">
                <dt className="text-content-tertiary">Tokens</dt>
                <dd className="num text-content-primary">{stream.tokens.length}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-content-tertiary">Mean entropy</dt>
                <dd className="num text-content-primary">
                  {meanEntropy !== null ? `${meanEntropy.toFixed(2)} bits` : '-'}
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-content-tertiary">Model step</dt>
                <dd className="num text-content-primary">{fmt.int(step)}</dd>
              </div>
            </dl>
            <p className="mt-2 text-2xs leading-relaxed text-content-tertiary">
              Mean entropy falls as the model learns. An untrained character model sits near
              log2(vocab) bits; a trained one is typically well under 2.
            </p>
          </Panel>
        )}
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Token stream renderer
// ---------------------------------------------------------------------------

/**
 * Renders the prompt plus each generated token as an individually selectable
 * span, tinted by the model's confidence.
 *
 * The confidence tint is the point: it turns the output into a map of where the
 * model was sure and where it was guessing. Low-confidence characters are
 * exactly where the text tends to go wrong, and seeing them highlighted
 * explains the failure far better than reading the text alone.
 */
const TokenStream: React.FC<{
  prompt: string;
  tokens: GeneratedToken[];
  active: boolean;
  selectedIndex: number | null;
  onSelect: (index: number | null) => void;
}> = ({ prompt, tokens, active, selectedIndex, onSelect }) => {
  const [showConfidence, setShowConfidence] = useState(true);

  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-2">
        {/*
          The toggle is a <button role="switch">, so it must NOT be wrapped in
          a <label>: clicking the text would forward a click to the button AND
          fire the label's own activation, toggling twice. A sibling span with
          its own click handler gives the same click target without the
          double-fire.
        */}
        <div className="flex items-center gap-2.5">
          <Toggle
            checked={showConfidence}
            onChange={setShowConfidence}
            label="Shade by confidence"
          />
          <span
            className="cursor-pointer select-none text-2xs text-content-tertiary"
            onClick={() => setShowConfidence((previous) => !previous)}
          >
            Shade by confidence
          </span>
        </div>
        {showConfidence && (
          <div className="flex items-center gap-1.5 text-2xs text-content-tertiary">
            <span>unsure</span>
            <span className="flex h-2.5 overflow-hidden rounded-sm">
              {[0.1, 0.3, 0.5, 0.7, 0.9].map((p) => (
                <span
                  key={p}
                  className="w-3"
                  style={{ background: confidenceColor(p) }}
                  aria-hidden="true"
                />
              ))}
            </span>
            <span>certain</span>
          </div>
        )}
      </div>

      <div
        className={`max-h-[320px] overflow-auto whitespace-pre-wrap break-words rounded-md border border-edge-subtle bg-surface-input p-3 font-mono text-sm leading-relaxed ${
          active ? 'caret' : ''
        }`}
      >
        <span className="text-content-tertiary">{prompt}</span>
        {tokens.map((token, index) => {
          const chosen = token.topCandidates.find((c) => c.chosen);
          const probability = chosen?.probability ?? 0;
          const isSelected = selectedIndex === index;
          return (
            <span
              key={index}
              role="button"
              tabIndex={0}
              onClick={() => onSelect(isSelected ? null : index)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  onSelect(isSelected ? null : index);
                }
              }}
              title={`${fmt.percent(probability)} · ${token.entropyBits.toFixed(2)} bits`}
              className={`cursor-pointer rounded-sm transition-colors hover:bg-accent-subtle ${
                isSelected ? 'ring-1 ring-accent' : ''
              }`}
              style={
                showConfidence
                  ? { backgroundColor: confidenceColor(probability), color: 'var(--text-primary)' }
                  : { color: 'var(--text-primary)' }
              }
            >
              {/* Whitespace is shown as a faint glyph so it remains clickable
                  and visible - at character level, space is the single most
                  common prediction the model makes. */}
              {fmt.isWhitespaceToken(token.token) ? (
                <span className="text-content-disabled">{fmt.visibleToken(token.token)}</span>
              ) : (
                token.token
              )}
              {token.token === '\n' && '\n'}
            </span>
          );
        })}
      </div>
    </div>
  );
};

/** A translucent green-to-amber tint keyed to the chosen token's probability. */
function confidenceColor(probability: number): string {
  const clamped = Math.max(0, Math.min(1, probability));
  // Low probability -> amber (the model was guessing); high -> green.
  if (clamped >= 0.5) {
    const strength = (clamped - 0.5) / 0.5;
    return `color-mix(in srgb, var(--success) ${8 + strength * 16}%, transparent)`;
  }
  const strength = 1 - clamped / 0.5;
  return `color-mix(in srgb, var(--warning) ${4 + strength * 20}%, transparent)`;
}

const SparkIcon = () => (
  <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
    <path
      d="M6 1l1.3 3.2L10.5 5.5 7.3 6.8 6 10 4.7 6.8 1.5 5.5l3.2-1.3z"
      fill="currentColor"
    />
  </svg>
);

const StopIcon = () => (
  <svg width="11" height="11" viewBox="0 0 11 11" aria-hidden="true">
    <rect x="2" y="2" width="7" height="7" rx="1" fill="currentColor" />
  </svg>
);
