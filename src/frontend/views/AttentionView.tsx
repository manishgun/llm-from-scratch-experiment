import React, { useMemo, useState } from 'react';
import { useStore } from '../lib/store';
import { api } from '../lib/api';
import {
  Panel,
  Button,
  TextInput,
  Field,
  Note,
  Badge,
  EmptyState,
  Segmented,
  Stat,
} from '../components/ui';
import { Heatmap, HeatmapLegend, BarList } from '../components/charts';
import * as fmt from '../lib/format';
import { AttentionSnapshot } from '../../shared/types';

/**
 * ============================================================================
 * Attention - what each token is actually looking at
 * ============================================================================
 *
 * This is the view that makes self-attention click, and it is the thing the
 * previous version of this project was missing despite computing the data on
 * every forward pass.
 *
 * Each cell of the heatmap is one attention weight: how much the token on the
 * row (the QUERY, the one doing the predicting) drew information from the token
 * on the column (the KEY). Reading it reveals several things that are hard to
 * convey any other way:
 *
 * - The upper triangle is always empty. That is the causal mask - position 3
 *   cannot see position 4 - and seeing the blank triangle is the clearest
 *   possible demonstration that the model is not reading the answer.
 * - Row sums are 1. Attention distributes a fixed budget of attention.
 * - Different heads in the same layer learn visibly different patterns. One
 *   often attends to the immediately previous token, another to the start of
 *   the line, another to matching punctuation.
 * - An untrained model's map is nearly uniform within the allowed triangle.
 *   Training is what carves structure into it, and comparing before and after
 *   is strikingly direct.
 */

export const AttentionView: React.FC = () => {
  const { parameters, step, reportError, config } = useStore();

  const [prompt, setPrompt] = useState('ROMEO: But soft');
  const [snapshot, setSnapshot] = useState<AttentionSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [mode, setMode] = useState<'single' | 'allHeads'>('single');
  const [hovered, setHovered] = useState<{ query: number; key: number; value: number } | null>(
    null
  );

  const hasModel = parameters !== null && parameters > 0;

  const fetchAttention = async () => {
    setLoading(true);
    try {
      const result = await api.attention(prompt);
      setSnapshot(result);
      setLayer((previous) => Math.min(previous, result.numLayers - 1));
      setHead((previous) => Math.min(previous, result.numHeads - 1));
    } catch (error) {
      reportError(error, 'Could not compute attention');
    } finally {
      setLoading(false);
    }
  };

  const tokens = useMemo(
    () => snapshot?.tokens.map((token) => fmt.visibleToken(token)) ?? [],
    [snapshot]
  );

  const matrix = snapshot?.weights[layer]?.[head] ?? null;

  /**
   * For the selected head, how much attention each KEY position receives in
   * total, summed across all queries. A key that every query attends to is
   * acting as a global anchor - at character level this is usually the start of
   * the sequence or a line break.
   */
  const keyTotals = useMemo(() => {
    if (!matrix) return [];
    const size = matrix.length;
    const totals = new Array(size).fill(0);
    for (const row of matrix) {
      for (let key = 0; key < size; key++) totals[key] += row[key];
    }
    return totals.map((total, index) => ({
      label: tokens[index] ?? '?',
      value: total,
      display: total.toFixed(2),
      mono: true,
    }));
  }, [matrix, tokens]);

  /**
   * Mean distance each query looks back, in tokens. A head with a mean
   * distance near 1 is a "previous token" head; a large value indicates a head
   * gathering long-range context.
   */
  const meanLookback = useMemo(() => {
    if (!matrix) return null;
    let weightedSum = 0;
    let weightTotal = 0;
    for (let query = 0; query < matrix.length; query++) {
      for (let key = 0; key <= query; key++) {
        weightedSum += matrix[query][key] * (query - key);
        weightTotal += matrix[query][key];
      }
    }
    return weightTotal > 0 ? weightedSum / weightTotal : null;
  }, [matrix]);

  const cellSize = snapshot ? Math.max(12, Math.min(26, Math.floor(640 / Math.max(1, snapshot.tokens.length)))) : 20;

  return (
    <div className="flex flex-col gap-3">
      {!hasModel && (
        <Note tone="warning" title="No model in memory">
          Start a run or load a checkpoint first. You can inspect an untrained model too - its
          attention is nearly uniform, which is a useful thing to see before training.
        </Note>
      )}

      <Panel
        title="Prompt"
        description="Attention is computed over this exact text. Keep it short - the map is one cell per token pair, so cost grows as the square of the length."
      >
        <div className="flex flex-wrap items-end gap-2">
          <Field label="Text" className="min-w-[260px] flex-1">
            {(id) => (
              <TextInput
                id={id}
                value={prompt}
                maxLength={128}
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && hasModel && !loading) void fetchAttention();
                }}
                className="font-mono"
              />
            )}
          </Field>
          <Button
            variant="primary"
            onClick={fetchAttention}
            loading={loading}
            disabled={!hasModel || prompt.length === 0}
          >
            Compute
          </Button>
        </div>
        <p className="mt-1.5 text-2xs text-content-tertiary">
          {prompt.length} / 128 characters · model at step {fmt.int(step)}
          {step === 0 && ' (untrained)'}
        </p>
      </Panel>

      {!snapshot ? (
        <Panel title="Attention map">
          <EmptyState
            title="Nothing computed yet"
            description="Enter a prompt and press Compute. Each row shows where that token drew information from."
            action={
              <Button variant="primary" onClick={fetchAttention} disabled={!hasModel} loading={loading}>
                Compute attention
              </Button>
            }
          />
        </Panel>
      ) : (
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,300px)]">
          <Panel
            title={mode === 'single' ? `Layer ${layer} · head ${head}` : `Layer ${layer} · all heads`}
            description="Rows are queries (the predicting token); columns are keys (the token being read)."
            actions={
              <Segmented
                size="sm"
                value={mode}
                onChange={setMode}
                ariaLabel="View mode"
                options={[
                  { value: 'single', label: 'One head' },
                  { value: 'allHeads', label: 'Compare heads' },
                ]}
              />
            }
          >
            {/* Layer and head selectors */}
            <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2">
              <div className="flex items-center gap-2">
                <span className="label">Layer</span>
                <div className="flex gap-1">
                  {Array.from({ length: snapshot.numLayers }, (_, index) => (
                    <button
                      key={index}
                      type="button"
                      onClick={() => setLayer(index)}
                      className={`num h-6 w-6 rounded border text-2xs transition-colors ${
                        layer === index
                          ? 'border-accent bg-accent text-white'
                          : 'border-edge bg-surface-input text-content-secondary hover:bg-surface-hover'
                      }`}
                    >
                      {index}
                    </button>
                  ))}
                </div>
              </div>

              {mode === 'single' && (
                <div className="flex items-center gap-2">
                  <span className="label">Head</span>
                  <div className="flex gap-1">
                    {Array.from({ length: snapshot.numHeads }, (_, index) => (
                      <button
                        key={index}
                        type="button"
                        onClick={() => setHead(index)}
                        className={`num h-6 w-6 rounded border text-2xs transition-colors ${
                          head === index
                            ? 'border-accent bg-accent text-white'
                            : 'border-edge bg-surface-input text-content-secondary hover:bg-surface-hover'
                        }`}
                      >
                        {index}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="ml-auto">
                <HeatmapLegend />
              </div>
            </div>

            {mode === 'single' && matrix ? (
              <>
                <Heatmap
                  matrix={matrix}
                  tokens={tokens}
                  cellSize={cellSize}
                  onHover={setHovered}
                />
                <div className="mt-2 h-4 text-2xs text-content-secondary">
                  {hovered ? (
                    <span className="num">
                      "{tokens[hovered.query]}" attended {fmt.percent(hovered.value)} to "
                      {tokens[hovered.key]}" ({hovered.query - hovered.key} token
                      {hovered.query - hovered.key === 1 ? '' : 's'} back)
                    </span>
                  ) : (
                    <span className="text-content-tertiary">
                      Hover a cell for the exact weight.
                    </span>
                  )}
                </div>
              </>
            ) : (
              // All heads of the layer, side by side. Seeing them together is
              // what demonstrates that heads specialise rather than duplicating
              // each other.
              <div className="flex flex-wrap gap-4">
                {Array.from({ length: snapshot.numHeads }, (_, headIndex) => {
                  const headMatrix = snapshot.weights[layer]?.[headIndex];
                  if (!headMatrix) return null;
                  return (
                    <div key={headIndex} className="flex flex-col gap-1">
                      <div className="flex items-center gap-1.5">
                        <span className="label">Head {headIndex}</span>
                        <button
                          type="button"
                          className="text-2xs text-accent-text hover:underline"
                          onClick={() => {
                            setHead(headIndex);
                            setMode('single');
                          }}
                        >
                          expand
                        </button>
                      </div>
                      <Heatmap
                        matrix={headMatrix}
                        tokens={tokens}
                        cellSize={Math.max(7, Math.min(14, Math.floor(300 / Math.max(1, tokens.length))))}
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </Panel>

          <div className="flex min-w-0 flex-col gap-3">
            <Panel title="This head">
              <div className="grid grid-cols-2 gap-2">
                <Stat
                  label="Mean lookback"
                  value={meanLookback !== null ? meanLookback.toFixed(2) : '-'}
                  unit="tok"
                  title="On average, how many positions back this head draws from. Near 1 means it is a 'previous token' head; larger values mean it gathers longer-range context."
                />
                <Stat label="Tokens" value={snapshot.tokens.length} hint="sequence length" />
              </div>
              <p className="mt-2.5 text-2xs leading-relaxed text-content-tertiary">
                The blank upper-right triangle is the causal mask. Position {Math.min(3, snapshot.tokens.length - 1)}{' '}
                cannot attend to position {Math.min(4, snapshot.tokens.length)} or beyond, because
                during training those are the tokens it is being asked to predict.
              </p>
            </Panel>

            <Panel
              title="Most-attended keys"
              description="Total attention each position receives, summed over all queries in this head."
            >
              {keyTotals.length > 0 ? (
                <BarList items={[...keyTotals].sort((a, b) => b.value - a.value)} maxItems={12} />
              ) : (
                <EmptyState compact title="No data" />
              )}
              <p className="mt-2.5 text-2xs leading-relaxed text-content-tertiary">
                A position that nearly every query attends to is acting as an anchor. Trained
                character models frequently park attention on the first token or a line break when
                a head has nothing specific to contribute - the attention budget has to go
                somewhere, since each row must sum to 1.
              </p>
            </Panel>

            {step === 0 && (
              <Note tone="info" title="This model is untrained">
                The map should look close to uniform across the allowed triangle. Train for a few
                hundred steps and recompute - structure appears, and different heads diverge.
              </Note>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
