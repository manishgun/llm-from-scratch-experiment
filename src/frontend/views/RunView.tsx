import React, { useMemo, useState } from 'react';
import { useStore } from '../lib/store';
import { Panel, Stat, Button, Segmented, Badge, EmptyState, Note, ProgressBar } from '../components/ui';
import { LineChart, Legend, Series, smoothSeries } from '../components/charts';
import * as fmt from '../lib/format';

/**
 * ============================================================================
 * Run - the training dashboard
 * ============================================================================
 *
 * The screen a user watches while a model trains. Its job is to answer four
 * questions at a glance, in descending order of urgency:
 *
 *   1. Is it working?        -> the loss curve, against the baseline
 *   2. Is it about to break? -> the gradient-norm chart
 *   3. How long?             -> progress and throughput
 *   4. What is it producing? -> the live text sample
 *
 * The layout follows that order top to bottom, because that is the order
 * someone checking in on a run actually reads.
 */

/** The charts the user can switch between beneath the loss plot. */
type DiagnosticChart = 'grad' | 'lr' | 'throughput';

export const RunView: React.FC = () => {
  const {
    status,
    step,
    history,
    latest,
    bestValLoss,
    bestValStep,
    parameters,
    config,
    dataset,
    logs,
    clearLogs,
    startTraining,
    pauseTraining,
    stopTraining,
    resetTraining,
  } = useStore();

  const [logScale, setLogScale] = useState(false);
  const [smoothing, setSmoothing] = useState(0.6);
  const [diagnostic, setDiagnostic] = useState<DiagnosticChart>('grad');

  const totalSteps = config?.training.trainingSteps ?? 0;
  const vocabSize = dataset?.uniqueCharacters ?? config?.model.vocabularySize ?? 0;
  /** The loss of a model that guesses uniformly: the reference for "learning". */
  const uniformLoss = vocabSize > 1 ? Math.log(vocabSize) : null;

  // --- Series --------------------------------------------------------------
  const lossSeries = useMemo<Series[]>(() => {
    const trainPoints = history.map((m) => ({ x: m.step, y: m.trainLoss }));
    // Validation is recorded only on evaluation steps, so it is genuinely
    // sparse rather than missing - it gets its own markers.
    const valPoints = history
      .filter((m) => m.valLoss !== undefined)
      .map((m) => ({ x: m.step, y: m.valLoss as number }));

    const series: Series[] = [];

    if (smoothing > 0 && trainPoints.length > 4) {
      // Raw behind, smoothed in front: the raw line carries the instability
      // information, the smoothed line carries the trend. Showing only one
      // loses half of what the chart is for.
      series.push({
        key: 'train-raw',
        label: 'train (raw)',
        color: 'var(--series-train)',
        points: trainPoints,
        width: 1,
      });
      series.push({
        key: 'train',
        label: 'train',
        color: 'var(--series-train)',
        points: smoothSeries(trainPoints, smoothing),
        width: 2,
      });
    } else {
      series.push({
        key: 'train',
        label: 'train',
        color: 'var(--series-train)',
        points: trainPoints,
        width: 1.5,
      });
    }

    if (valPoints.length > 0) {
      series.push({
        key: 'val',
        label: 'validation',
        color: 'var(--series-val)',
        points: valPoints,
        width: 2,
        showDots: true,
      });
    }

    return series;
  }, [history, smoothing]);

  const diagnosticSeries = useMemo<Series[]>(() => {
    if (diagnostic === 'grad') {
      return [
        {
          key: 'grad',
          label: 'gradient norm',
          color: 'var(--series-grad)',
          points: history
            .filter((m) => m.gradNorm !== undefined)
            .map((m) => ({ x: m.step, y: m.gradNorm as number })),
          width: 1.25,
        },
      ];
    }
    if (diagnostic === 'lr') {
      return [
        {
          key: 'lr',
          label: 'learning rate',
          color: 'var(--series-lr)',
          points: history.map((m) => ({ x: m.step, y: m.learningRate })),
          width: 1.75,
        },
      ];
    }
    return [
      {
        key: 'tps',
        label: 'tokens/sec',
        color: 'var(--series-throughput)',
        points: history
          .filter((m) => m.tokensPerSecond !== undefined)
          .map((m) => ({ x: m.step, y: m.tokensPerSecond as number })),
        width: 1.25,
      },
    ];
  }, [history, diagnostic]);

  const isRunning = status === 'running';
  const canStart = status === 'idle' || status === 'paused' || status === 'completed';
  const hasRun = history.length > 0;

  // The most recent sample emitted during evaluation.
  const liveSample = useMemo(() => {
    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].generatedSample) {
        return { text: history[i].generatedSample as string, step: history[i].step };
      }
    }
    return null;
  }, [history]);

  const clipThreshold = config?.training.gradientClipNorm ?? 0;

  return (
    <div className="flex flex-col gap-3">
      {/* ------------------------------------------------------------------ */}
      {/* Control bar                                                        */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-edge-subtle bg-surface-raised px-3.5 py-2.5">
        <div className="flex items-center gap-2">
          {canStart ? (
            <Button variant="primary" onClick={startTraining} icon={<PlayIcon />}>
              {status === 'paused' ? 'Resume' : hasRun ? 'Continue' : 'Start training'}
            </Button>
          ) : (
            <Button variant="secondary" onClick={pauseTraining} icon={<PauseIcon />}>
              Pause
            </Button>
          )}
          <Button
            variant="secondary"
            size="md"
            onClick={stopTraining}
            disabled={!isRunning}
            icon={<StopIcon />}
          >
            Stop
          </Button>
          <Button
            variant="danger"
            size="md"
            onClick={resetTraining}
            disabled={isRunning || !hasRun}
            title="Discard weights, optimizer state and metrics"
          >
            Reset
          </Button>
        </div>

        <div className="flex min-w-[220px] flex-1 flex-col gap-1">
          <div className="flex items-baseline justify-between text-2xs text-content-tertiary">
            <span className="num">
              step <span className="text-content-primary">{fmt.int(step)}</span> /{' '}
              {fmt.int(totalSteps)}
            </span>
            {isRunning && latest && (
              <span className="num">{fmt.duration(latest.estimatedRemainingMs)} remaining</span>
            )}
          </div>
          <ProgressBar
            value={step}
            max={totalSteps}
            tone={status === 'error' ? 'var(--danger)' : 'var(--accent)'}
          />
        </div>
      </div>

      {status === 'error' && (
        <Note tone="danger" title="The run stopped with an error">
          {logs.filter((l) => l.level === 'error').slice(-1)[0]?.message ??
            'See the engine log below for details.'}
        </Note>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Metric tiles                                                       */}
      {/* ------------------------------------------------------------------ */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        <Stat
          label="Train loss"
          value={fmt.loss(latest?.trainLoss)}
          seriesColor="var(--series-train)"
          hint={
            latest?.perplexity !== undefined ? `ppl ${fmt.perplexity(latest.perplexity)}` : undefined
          }
          title="Cross-entropy on the current training batch. Measured WITH dropout active, so it reads slightly worse than the model really is."
        />
        <Stat
          label="Val loss"
          value={fmt.loss(latest?.valLoss ?? bestValLoss ?? undefined)}
          seriesColor="var(--series-val)"
          hint={
            latest?.valPerplexity !== undefined
              ? `ppl ${fmt.perplexity(latest.valPerplexity)}`
              : 'on held-out text'
          }
          title="Cross-entropy on held-out text, with dropout disabled. This is the number that reflects real quality."
        />
        <Stat
          label="Best val"
          value={fmt.loss(bestValLoss ?? undefined)}
          tone={bestValLoss !== null ? 'success' : 'default'}
          hint={bestValStep > 0 ? `step ${fmt.int(bestValStep)}` : 'saved as best.mgpt'}
          title="The lowest validation loss of the run. Its checkpoint is kept separately, because the FINAL model is often worse than the best one."
        />
        <Stat
          label="Grad norm"
          value={fmt.gradNorm(latest?.gradNorm)}
          seriesColor="var(--series-grad)"
          tone={
            latest?.gradNorm !== undefined && clipThreshold > 0 && latest.gradNorm > clipThreshold * 3
              ? 'warning'
              : 'default'
          }
          hint={clipThreshold > 0 ? `clip at ${clipThreshold}` : 'clipping off'}
          title="L2 norm of all gradients before clipping. A sudden spike is the earliest warning that the learning rate is too high - it shows up before the loss does."
        />
        <Stat
          label="Learning rate"
          value={fmt.scientific(latest?.learningRate)}
          seriesColor="var(--series-lr)"
          hint={config ? `${config.training.lrSchedule} schedule` : undefined}
        />
        <Stat
          label="Throughput"
          value={fmt.compactInt(latest?.tokensPerSecond)}
          unit="tok/s"
          seriesColor="var(--series-throughput)"
          hint={latest ? `${latest.stepsPerSecond.toFixed(2)} step/s` : undefined}
        />
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Loss chart                                                         */}
      {/* ------------------------------------------------------------------ */}
      <Panel
        title="Loss"
        description={
          latest && uniformLoss
            ? fmt.describeLoss(latest.valLoss ?? latest.trainLoss, vocabSize)
            : 'Cross-entropy per token. Lower is better.'
        }
        actions={
          <>
            <Segmented
              size="sm"
              ariaLabel="Y axis scale"
              value={logScale ? 'log' : 'linear'}
              onChange={(value) => setLogScale(value === 'log')}
              options={[
                { value: 'linear', label: 'Linear' },
                {
                  value: 'log',
                  label: 'Log',
                  title:
                    'A logarithmic axis. Once loss has fallen from 4.2 to ~1.5, the remaining progress occupies the bottom tenth of a linear axis and is unreadable.',
                },
              ]}
            />
            <label className="flex items-center gap-1.5 pl-1 text-2xs text-content-tertiary">
              smooth
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={smoothing}
                onChange={(event) => setSmoothing(Number(event.target.value))}
                className="w-[70px]"
                aria-label="Smoothing strength"
              />
            </label>
          </>
        }
      >
        {hasRun ? (
          <>
            <LineChart
              series={lossSeries}
              height={262}
              logScale={logScale}
              xLabel="step"
              formatY={(value) => (value >= 1 ? value.toFixed(2) : value.toFixed(3))}
              referenceLines={
                uniformLoss
                  ? [
                      {
                        y: uniformLoss,
                        // Naming the baseline is what makes the loss number
                        // interpretable: it is where an untrained model sits.
                        label: `ln(${vocabSize}) = ${uniformLoss.toFixed(2)} - uniform guessing`,
                      },
                    ]
                  : []
              }
              markers={
                bestValStep > 0
                  ? [{ x: bestValStep, label: 'best', color: 'var(--series-val)' }]
                  : []
              }
            />
            <div className="mt-2 flex items-center justify-between gap-3">
              <Legend
                items={[
                  {
                    label: 'train',
                    color: 'var(--series-train)',
                    value: fmt.loss(latest?.trainLoss),
                  },
                  {
                    label: 'validation',
                    color: 'var(--series-val)',
                    value: fmt.loss(latest?.valLoss ?? bestValLoss ?? undefined),
                  },
                ]}
              />
              <OverfittingHint history={history} />
            </div>
          </>
        ) : (
          <EmptyState
            title="No metrics yet"
            description="Press Start training. The loss should begin at ln(vocabSize) and fall from there."
            action={
              <Button variant="primary" onClick={startTraining} icon={<PlayIcon />}>
                Start training
              </Button>
            }
          />
        )}
      </Panel>

      {/* ------------------------------------------------------------------ */}
      {/* Diagnostics + sample                                               */}
      {/* ------------------------------------------------------------------ */}
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        <Panel
          title="Diagnostics"
          description={DIAGNOSTIC_DESCRIPTIONS[diagnostic]}
          actions={
            <Segmented
              size="sm"
              ariaLabel="Diagnostic chart"
              value={diagnostic}
              onChange={setDiagnostic}
              options={[
                { value: 'grad', label: 'Grad norm' },
                { value: 'lr', label: 'LR' },
                { value: 'throughput', label: 'Speed' },
              ]}
            />
          }
        >
          {hasRun ? (
            <LineChart
              series={diagnosticSeries}
              height={168}
              logScale={diagnostic === 'lr'}
              includeZero={diagnostic === 'throughput'}
              xLabel="step"
              formatY={
                diagnostic === 'lr'
                  ? (value) => value.toExponential(0)
                  : diagnostic === 'throughput'
                    ? (value) => fmt.compactInt(value)
                    : (value) => value.toFixed(2)
              }
              referenceLines={
                diagnostic === 'grad' && clipThreshold > 0
                  ? [{ y: clipThreshold, label: `clip ${clipThreshold}`, color: 'var(--warning)' }]
                  : []
              }
            />
          ) : (
            <EmptyState compact title="Nothing to plot yet" />
          )}
        </Panel>

        <Panel
          title="Live sample"
          description={
            liveSample
              ? `Generated at step ${fmt.int(liveSample.step)} with a fixed sampling seed, so changes reflect the model only.`
              : 'A short completion is generated at every evaluation.'
          }
        >
          {liveSample ? (
            <pre className="max-h-[168px] overflow-auto whitespace-pre-wrap break-words rounded-md border border-edge-subtle bg-surface-input p-2.5 font-mono text-xs leading-relaxed text-content-secondary">
              {liveSample.text}
            </pre>
          ) : (
            <EmptyState
              compact
              title="No sample yet"
              description="The first sample appears at the first evaluation step."
            />
          )}
        </Panel>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Run summary + log                                                  */}
      {/* ------------------------------------------------------------------ */}
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,320px)_minmax(0,1fr)]">
        <Panel title="Run">
          <dl className="flex flex-col gap-1.5 text-xs">
            <SummaryRow label="Parameters" value={fmt.int(parameters)} />
            <SummaryRow
              label="Architecture"
              value={
                config
                  ? `${config.model.numLayers}L · ${config.model.numHeads}H · d${config.model.embeddingDim}`
                  : '-'
              }
            />
            <SummaryRow
              label="Batch"
              value={
                config
                  ? `${config.training.batchSize} × ${config.model.contextLength} tok`
                  : '-'
              }
            />
            <SummaryRow
              label="Tokens seen"
              value={
                config
                  ? fmt.compactInt(step * config.training.batchSize * config.model.contextLength)
                  : '-'
              }
            />
            <SummaryRow label="Elapsed" value={fmt.duration(latest?.elapsedMs)} />
            <SummaryRow
              label="Host memory"
              value={latest ? `${latest.memoryUsageMb.toFixed(0)} MB` : '-'}
            />
            <SummaryRow
              label="Live tensors"
              value={fmt.int(latest?.tensorCount)}
              title="Tensors currently held by TensorFlow. A value that climbs steadily with step count indicates a memory leak."
            />
            <SummaryRow label="Seed" value={fmt.int(config?.model.randomSeed)} />
          </dl>
        </Panel>

        <Panel
          title="Engine log"
          actions={
            <Button size="sm" variant="ghost" onClick={clearLogs} disabled={logs.length === 0}>
              Clear
            </Button>
          }
          flush
        >
          <div className="h-[196px] overflow-y-auto px-3.5 py-2 font-mono text-2xs leading-relaxed">
            {logs.length === 0 ? (
              <p className="pt-2 text-content-tertiary">Idle. Engine messages appear here.</p>
            ) : (
              logs.map((entry) => (
                <div key={entry.id} className="flex gap-2 py-px">
                  <span className="shrink-0 text-content-disabled">
                    {fmt.clockTime(entry.timestamp)}
                  </span>
                  <span
                    className={
                      entry.level === 'error'
                        ? 'text-danger'
                        : entry.level === 'warn'
                          ? 'text-warning'
                          : 'text-content-secondary'
                    }
                  >
                    {entry.message}
                  </span>
                </div>
              ))
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------

const DIAGNOSTIC_DESCRIPTIONS: Record<DiagnosticChart, string> = {
  grad:
    'Gradient L2 norm before clipping. Stable and slowly shrinking is healthy; a spike of several orders of magnitude means the learning rate is too high.',
  lr: 'The learning rate actually applied at each step, including warmup and decay.',
  throughput: 'Tokens processed per second. Flat is normal; a decline suggests memory pressure.',
};

const SummaryRow: React.FC<{ label: string; value: React.ReactNode; title?: string }> = ({
  label,
  value,
  title,
}) => (
  <div className="flex items-baseline justify-between gap-3" title={title}>
    <dt className="text-content-tertiary">{label}</dt>
    <dd className="num truncate text-content-primary">{value}</dd>
  </div>
);

/**
 * Flags the train/validation divergence that indicates overfitting.
 *
 * This is the most common thing a newcomer misreads. Training loss keeps
 * falling, so the run looks like it is improving, while validation loss has
 * turned upward - meaning the model is now memorising rather than learning, and
 * every subsequent step makes it worse on new text. Saying so explicitly, at
 * the moment it happens, is more useful than any amount of documentation.
 */
const OverfittingHint: React.FC<{ history: { step: number; trainLoss: number; valLoss?: number }[] }> = ({
  history,
}) => {
  const verdict = useMemo(() => {
    const evaluated = history.filter((m) => m.valLoss !== undefined);
    if (evaluated.length < 4) return null;

    const recent = evaluated.slice(-3);
    const best = Math.min(...evaluated.map((m) => m.valLoss as number));
    const current = recent[recent.length - 1].valLoss as number;
    const trainNow = recent[recent.length - 1].trainLoss;

    // Validation meaningfully above its own best, while training loss is
    // clearly below it, is the signature of overfitting.
    if (current > best + 0.02 && trainNow < current - 0.1) {
      return {
        tone: 'warning' as const,
        text: `Validation is ${(current - best).toFixed(3)} above its best - overfitting`,
      };
    }
    const gap = current - trainNow;
    if (gap < -0.05) {
      // Train loss above val loss is expected WITH dropout, and worth
      // explaining rather than leaving as a puzzling inversion.
      return { tone: 'neutral' as const, text: 'Train above val (expected with dropout)' };
    }
    return null;
  }, [history]);

  if (!verdict) return null;

  return (
    <Badge tone={verdict.tone === 'warning' ? 'warning' : 'neutral'}>{verdict.text}</Badge>
  );
};

// --- Icons -----------------------------------------------------------------

const PlayIcon = () => (
  <svg width="11" height="11" viewBox="0 0 11 11" aria-hidden="true">
    <path d="M2 1.5v8l7-4z" fill="currentColor" />
  </svg>
);

const PauseIcon = () => (
  <svg width="11" height="11" viewBox="0 0 11 11" aria-hidden="true">
    <path d="M2.5 1.5h2.5v8H2.5zM6 1.5h2.5v8H6z" fill="currentColor" />
  </svg>
);

const StopIcon = () => (
  <svg width="11" height="11" viewBox="0 0 11 11" aria-hidden="true">
    <rect x="2" y="2" width="7" height="7" rx="1" fill="currentColor" />
  </svg>
);
