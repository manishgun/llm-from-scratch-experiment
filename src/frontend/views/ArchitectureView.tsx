import React, { useEffect, useMemo, useState } from 'react';
import { useStore } from '../lib/store';
import { api } from '../lib/api';
import {
  Panel,
  Button,
  Field,
  SliderInput,
  Select,
  Toggle,
  Badge,
  Note,
  Stat,
  Table,
  Th,
  Td,
  Segmented,
  EmptyState,
} from '../components/ui';
import { LineChart, BarList, Series } from '../components/charts';
import { ParameterReference } from '../components/ParameterReference';
import * as fmt from '../lib/format';
import { GradientCheckResult, ModelConfig, TrainingConfig } from '../../shared/types';
import { learningRateAt } from '../../backend/training/schedule';

/**
 * ============================================================================
 * Architecture - configuration, with consequences made visible
 * ============================================================================
 *
 * A form is the wrong model for hyperparameters, because a hyperparameter's
 * value means nothing without its effect. Every control here is therefore
 * paired with something that changes as you move it:
 *
 * - Architecture sliders update the PARAMETER COUNT and its breakdown live, so
 *   it is immediately obvious that embeddings dominate a word-level model and
 *   that `numLayers` scales the attention and MLP blocks linearly.
 * - Schedule controls redraw the LEARNING-RATE CURVE, so a warmup longer than
 *   the run, or a floor above the peak, is visible rather than a later error.
 * - Invalid combinations are caught and explained HERE, not when training
 *   fails several seconds after pressing Start.
 */

export const ArchitectureView: React.FC = () => {
  const {
    draftConfig,
    setDraftConfig,
    config,
    defaults,
    docs,
    isDirty,
    applyConfig,
    revertConfig,
    status,
    dataset,
    reportError,
    pushToast,
  } = useStore();

  const [section, setSection] = useState<'model' | 'training'>('model');
  const [gradientCheck, setGradientCheck] = useState<GradientCheckResult | null>(null);
  const [checking, setChecking] = useState(false);
  const [breakdown, setBreakdown] = useState<
    { component: string; parameters: number }[] | null
  >(null);

  const isRunning = status === 'running';

  const updateModel = (patch: Partial<ModelConfig>) => {
    if (!draftConfig) return;
    setDraftConfig({ ...draftConfig, model: { ...draftConfig.model, ...patch } });
  };

  const updateTraining = (patch: Partial<TrainingConfig>) => {
    if (!draftConfig) return;
    setDraftConfig({ ...draftConfig, training: { ...draftConfig.training, ...patch } });
  };

  // --- Client-side validation ---------------------------------------------
  /**
   * Mirrors the server's cross-field rules so the user sees the problem while
   * editing. The server still validates - this is a convenience layer, never
   * the enforcement layer.
   */
  const errors = useMemo(() => {
    if (!draftConfig) return {};
    const { model, training } = draftConfig;
    const found: Record<string, string> = {};

    if (model.embeddingDim % model.numHeads !== 0) {
      found.numHeads =
        `${model.embeddingDim} is not divisible by ${model.numHeads}. ` +
        `Each head takes an equal slice of the embedding (headDim = ${model.embeddingDim} / ${model.numHeads}).`;
    }
    if (training.warmupSteps >= training.trainingSteps) {
      found.warmupSteps =
        'Warmup must be shorter than the run, or the learning rate never reaches its peak.';
    }
    if (training.minLearningRate > training.learningRate) {
      found.minLearningRate = 'The floor cannot be above the peak - the schedule decays downward.';
    }
    const ratioSum = training.trainRatio + training.valRatio + training.testRatio;
    if (Math.abs(ratioSum - 1) > 1e-6) {
      found.trainRatio = `Split ratios must sum to 1.0 (currently ${ratioSum.toFixed(3)}).`;
    }
    // Context length is bounded by the dataset: a window longer than the
    // validation split makes batching impossible.
    if (dataset && dataset.valCharacters > 0 && model.contextLength >= dataset.valCharacters) {
      found.contextLength = `The validation split holds only ${fmt.int(dataset.valCharacters)} tokens, which is too few for this context length.`;
    }
    return found;
  }, [draftConfig, dataset]);

  const hasErrors = Object.keys(errors).length > 0;

  // --- Live parameter estimate --------------------------------------------
  /**
   * Computed in the browser rather than fetched, so it updates as the user
   * drags a slider. The formula mirrors `MiniGPT.getParameterBreakdown`.
   */
  const estimate = useMemo(() => {
    if (!draftConfig) return null;
    const m = draftConfig.model;
    const V = dataset?.uniqueCharacters || m.vocabularySize;
    const D = m.embeddingDim;
    const hidden = D * m.feedForwardMultiplier;
    const perBlockAttn = 4 * (D * D + D);
    const perBlockMlp = D * hidden + hidden + hidden * D + D;
    const perBlockLn = 4 * D;

    const rows = [
      { component: 'Token embeddings', parameters: V * D },
      { component: 'Position embeddings', parameters: m.contextLength * D },
      { component: 'Attention (all layers)', parameters: perBlockAttn * m.numLayers },
      { component: 'Feed-forward (all layers)', parameters: perBlockMlp * m.numLayers },
      { component: 'LayerNorm (all layers)', parameters: perBlockLn * m.numLayers },
      { component: 'Final LayerNorm', parameters: 2 * D },
      {
        component: m.tieWeights === false ? 'LM head' : 'LM head (tied, bias only)',
        parameters: m.tieWeights === false ? D * V + V : V,
      },
    ];
    return { rows, total: rows.reduce((acc, r) => acc + r.parameters, 0), vocabSize: V };
  }, [draftConfig, dataset]);

  // --- Learning-rate schedule preview -------------------------------------
  const scheduleSeries = useMemo<Series[]>(() => {
    if (!draftConfig) return [];
    const t = draftConfig.training;
    const points: { x: number; y: number }[] = [];
    // Cap the sample count: the chart cannot show more points than pixels.
    const stride = Math.max(1, Math.floor(t.trainingSteps / 400));
    for (let i = 0; i < t.trainingSteps; i += stride) {
      points.push({ x: i, y: learningRateAt(i, t) });
    }
    return [
      { key: 'lr', label: 'learning rate', color: 'var(--series-lr)', points, width: 1.75 },
    ];
  }, [draftConfig]);

  useEffect(() => {
    // Only meaningful once a model exists; silently skip otherwise.
    api
      .parameterBreakdown()
      .then((response) => setBreakdown(response.breakdown))
      .catch(() => setBreakdown(null));
  }, [config]);

  const runGradientCheck = async () => {
    setChecking(true);
    setGradientCheck(null);
    try {
      const result = await api.gradientCheck();
      setGradientCheck(result);
      pushToast(
        result.passed ? 'success' : 'error',
        result.passed
          ? 'Backpropagation verified'
          : 'Gradient check failed - the analytic and numeric gradients disagree'
      );
    } catch (error) {
      reportError(error, 'Gradient check failed to run');
    } finally {
      setChecking(false);
    }
  };

  if (!draftConfig || !estimate) {
    return <EmptyState title="Loading configuration..." />;
  }

  const m = draftConfig.model;
  const t = draftConfig.training;

  return (
    <div className="flex flex-col gap-3">
      {/* ------------------------------------------------------------------ */}
      {/* Apply bar                                                          */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-edge-subtle bg-surface-raised px-3.5 py-2.5">
        <div className="flex items-center gap-2.5">
          <Segmented
            value={section}
            onChange={setSection}
            ariaLabel="Configuration section"
            options={[
              { value: 'model', label: 'Model' },
              { value: 'training', label: 'Training' },
            ]}
          />
          {isDirty && <Badge tone="warning">Unapplied changes</Badge>}
          {hasErrors && <Badge tone="danger">{Object.keys(errors).length} problem(s)</Badge>}
        </div>

        <div className="flex items-center gap-2">
          {defaults && (
            <Button
              size="sm"
              variant="ghost"
              disabled={isRunning}
              onClick={() => setDraftConfig(structuredClone(defaults))}
            >
              Load defaults
            </Button>
          )}
          <Button size="sm" variant="ghost" onClick={revertConfig} disabled={!isDirty}>
            Revert
          </Button>
          <Button
            size="sm"
            variant="primary"
            onClick={applyConfig}
            disabled={!isDirty || hasErrors || isRunning}
            title={
              isRunning
                ? 'Pause or stop the run before changing the configuration'
                : 'Rebuild the model with these settings'
            }
          >
            Apply
          </Button>
        </div>
      </div>

      {isRunning && (
        <Note tone="info">
          Training is in progress. Configuration is read-only until you pause or stop it, because
          applying a change rebuilds the model and discards its weights.
        </Note>
      )}

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,340px)]">
        {/* ---------------------------------------------------------------- */}
        {/* Controls                                                         */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex min-w-0 flex-col gap-3">
          {section === 'model' ? (
            <>
              <Panel
                title="Architecture"
                description="The shape of the network. Changing any of these rebuilds the model from scratch."
              >
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <Field
                    label="Embedding dimension (d_model)"
                    valueDisplay={m.embeddingDim}
                    help={docs.embeddingDim?.simpleExplanation}
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={m.embeddingDim}
                        min={16}
                        max={512}
                        step={16}
                        disabled={isRunning}
                        onChange={(value) => updateModel({ embeddingDim: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Attention heads"
                    valueDisplay={`${m.numHeads} × d${m.embeddingDim / m.numHeads || '?'}`}
                    error={errors.numHeads}
                    help="Each head attends over its own slice of the embedding. More heads means more simultaneous relationships at no extra parameter cost."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={m.numHeads}
                        min={1}
                        max={16}
                        step={1}
                        disabled={isRunning}
                        onChange={(value) => updateModel({ numHeads: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Transformer layers"
                    valueDisplay={m.numLayers}
                    help="Depth. Compute scales linearly; residual-path weights are initialized at 1/sqrt(2L) so depth stays stable."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={m.numLayers}
                        min={1}
                        max={16}
                        step={1}
                        disabled={isRunning}
                        onChange={(value) => updateModel({ numLayers: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Context length"
                    valueDisplay={`${m.contextLength} tok`}
                    error={errors.contextLength}
                    help="How far back the model can see. Attention cost grows as the square of this, so it is the most expensive dial here."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={m.contextLength}
                        min={16}
                        max={512}
                        step={16}
                        disabled={isRunning}
                        onChange={(value) => updateModel({ contextLength: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Feed-forward multiplier"
                    valueDisplay={`${m.feedForwardMultiplier}× = ${m.embeddingDim * m.feedForwardMultiplier}`}
                    help="Hidden width of each MLP, as a multiple of d_model. GPT-2 uses 4."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={m.feedForwardMultiplier}
                        min={1}
                        max={8}
                        step={1}
                        disabled={isRunning}
                        onChange={(value) => updateModel({ feedForwardMultiplier: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Activation"
                    help="GELU is standard in GPT; ReLU is slightly faster but hard-zeroes all negative values."
                  >
                    {(id) => (
                      <Select
                        id={id}
                        value={m.activation}
                        disabled={isRunning}
                        onChange={(event) =>
                          updateModel({ activation: event.target.value as 'gelu' | 'relu' })
                        }
                        options={[
                          { value: 'gelu', label: 'GELU' },
                          { value: 'relu', label: 'ReLU' },
                        ]}
                      />
                    )}
                  </Field>
                </div>
              </Panel>

              <Panel
                title="Regularization & initialization"
                description="These control how the model generalizes, and how reproducible a run is."
              >
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <Field
                    label="Dropout rate"
                    valueDisplay={m.dropoutRate.toFixed(2)}
                    help={docs.dropoutRate?.simpleExplanation}
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={m.dropoutRate}
                        min={0}
                        max={0.5}
                        step={0.05}
                        disabled={isRunning}
                        onChange={(value) => updateModel({ dropoutRate: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Random seed"
                    valueDisplay={m.randomSeed ?? 42}
                    help="Determines initial weights and dropout masks. The same seed and config reproduce a run exactly."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={m.randomSeed ?? 42}
                        min={0}
                        max={9999}
                        step={1}
                        disabled={isRunning}
                        onChange={(value) => updateModel({ randomSeed: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Initializer scale"
                    valueDisplay={m.initializerScale.toFixed(3)}
                    help="Standard deviation of the initial weights. GPT-2 uses 0.02."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={m.initializerScale}
                        min={0.005}
                        max={0.1}
                        step={0.005}
                        disabled={isRunning}
                        onChange={(value) => updateModel({ initializerScale: value })}
                      />
                    )}
                  </Field>

                  <div className="flex items-start justify-between gap-3 pt-1">
                    <div className="min-w-0">
                      <label className="text-xs font-medium text-content-secondary">
                        Tie embedding & output weights
                      </label>
                      <p className="mt-1 text-2xs leading-snug text-content-tertiary">
                        {docs.tieWeights?.simpleExplanation ??
                          'Share one matrix between the input embedding and the output head.'}{' '}
                        Saves {fmt.compactInt(estimate.vocabSize * m.embeddingDim)} parameters here.
                      </p>
                    </div>
                    <Toggle
                      checked={m.tieWeights !== false}
                      disabled={isRunning}
                      onChange={(checked) => updateModel({ tieWeights: checked })}
                      label="Tie weights"
                    />
                  </div>
                </div>
              </Panel>
            </>
          ) : (
            <>
              <Panel title="Optimization">
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <Field
                    label="Peak learning rate"
                    valueDisplay={fmt.scientific(t.learningRate)}
                    help="The maximum step size, reached at the end of warmup."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        scientific
                        value={t.learningRate}
                        min={0.00001}
                        max={0.02}
                        step={0.00001}
                        disabled={isRunning}
                        onChange={(value) => updateTraining({ learningRate: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Batch size"
                    valueDisplay={`${t.batchSize} × ${m.contextLength} = ${fmt.compactInt(t.batchSize * m.contextLength)} tok`}
                    help="Sequences per step. Larger gives a smoother gradient estimate and better CPU utilisation."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={t.batchSize}
                        min={4}
                        max={256}
                        step={4}
                        disabled={isRunning}
                        onChange={(value) => updateTraining({ batchSize: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Training steps"
                    valueDisplay={fmt.int(t.trainingSteps)}
                    help="Total optimizer steps. The cosine schedule is stretched to fit this, so changing it changes the whole decay curve."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={t.trainingSteps}
                        min={50}
                        max={20000}
                        step={50}
                        disabled={isRunning}
                        onChange={(value) => updateTraining({ trainingSteps: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Gradient clip norm"
                    valueDisplay={t.gradientClipNorm === 0 ? 'off' : t.gradientClipNorm.toFixed(1)}
                    help={docs.gradientClipNorm?.simpleExplanation}
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={t.gradientClipNorm}
                        min={0}
                        max={5}
                        step={0.1}
                        disabled={isRunning}
                        onChange={(value) => updateTraining({ gradientClipNorm: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Weight decay (AdamW)"
                    valueDisplay={t.weightDecay.toFixed(3)}
                    help="Applied directly to weight matrices, never to biases or LayerNorm."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={t.weightDecay}
                        min={0}
                        max={0.2}
                        step={0.005}
                        disabled={isRunning}
                        onChange={(value) => updateTraining({ weightDecay: value })}
                      />
                    )}
                  </Field>

                  <Field label="Optimizer" help="Adam adapts a step size per parameter; plain SGD does not.">
                    {(id) => (
                      <Select
                        id={id}
                        value={t.optimizerType}
                        disabled={isRunning}
                        onChange={(event) =>
                          updateTraining({ optimizerType: event.target.value as 'adam' | 'sgd' })
                        }
                        options={[
                          { value: 'adam', label: 'Adam / AdamW' },
                          { value: 'sgd', label: 'SGD' },
                        ]}
                      />
                    )}
                  </Field>
                </div>
              </Panel>

              <Panel
                title="Learning rate schedule"
                description="Large steps early to find a good region, small steps late to settle into it."
              >
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                  <Field label="Shape" help="Cosine is what GPT-2, GPT-3 and LLaMA use.">
                    {(id) => (
                      <Select
                        id={id}
                        value={t.lrSchedule}
                        disabled={isRunning}
                        onChange={(event) =>
                          updateTraining({
                            lrSchedule: event.target.value as 'constant' | 'cosine' | 'linear',
                          })
                        }
                        options={[
                          { value: 'cosine', label: 'Cosine' },
                          { value: 'linear', label: 'Linear' },
                          { value: 'constant', label: 'Constant' },
                        ]}
                      />
                    )}
                  </Field>

                  <Field
                    label="Warmup steps"
                    valueDisplay={`${t.warmupSteps} (${fmt.percent(t.warmupSteps / Math.max(1, t.trainingSteps), 0)})`}
                    error={errors.warmupSteps}
                    help="Adam's step-size estimate is unreliable in the first few dozen steps; warmup keeps them small until it settles."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={t.warmupSteps}
                        min={0}
                        max={Math.max(100, Math.floor(t.trainingSteps / 2))}
                        step={10}
                        disabled={isRunning}
                        onChange={(value) => updateTraining({ warmupSteps: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Minimum rate"
                    valueDisplay={fmt.scientific(t.minLearningRate)}
                    error={errors.minLearningRate}
                    help="The floor the decay approaches. Decaying all the way to zero wastes the final steps."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        scientific
                        value={t.minLearningRate}
                        min={0}
                        max={Math.max(0.0001, t.learningRate)}
                        step={0.00001}
                        disabled={isRunning}
                        onChange={(value) => updateTraining({ minLearningRate: value })}
                      />
                    )}
                  </Field>
                </div>

                <div className="mt-3 border-t border-edge-subtle pt-3">
                  <LineChart
                    series={scheduleSeries}
                    height={150}
                    xLabel="step"
                    formatY={(value) => value.toExponential(0)}
                    markers={
                      t.warmupSteps > 0
                        ? [{ x: t.warmupSteps, label: 'warmup ends', color: 'var(--accent)' }]
                        : []
                    }
                  />
                </div>
              </Panel>

              <Panel title="Evaluation & checkpointing">
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <Field
                    label="Evaluation interval"
                    valueDisplay={`every ${t.evaluationInterval} steps`}
                    help="How often to measure validation loss and emit a sample. Evaluation pauses training briefly."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={t.evaluationInterval}
                        min={1}
                        max={500}
                        step={1}
                        disabled={isRunning}
                        onChange={(value) => updateTraining({ evaluationInterval: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Checkpoint interval"
                    valueDisplay={t.checkpointInterval === 0 ? 'off' : `every ${t.checkpointInterval}`}
                    help="Periodic snapshots. The best-validation checkpoint is always saved regardless of this."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={t.checkpointInterval}
                        min={0}
                        max={2000}
                        step={50}
                        disabled={isRunning}
                        onChange={(value) => updateTraining({ checkpointInterval: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Validation batches"
                    valueDisplay={t.validationBatchCount}
                    help="Batches averaged per evaluation. They are a FIXED sample, so successive measurements differ only because the model changed."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={t.validationBatchCount}
                        min={1}
                        max={32}
                        step={1}
                        disabled={isRunning}
                        onChange={(value) => updateTraining({ validationBatchCount: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Early stopping patience"
                    valueDisplay={t.earlyStoppingPatience === 0 ? 'off' : t.earlyStoppingPatience}
                    help="Stop after this many evaluations with no validation improvement. 0 disables it."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={t.earlyStoppingPatience}
                        min={0}
                        max={50}
                        step={1}
                        disabled={isRunning}
                        onChange={(value) => updateTraining({ earlyStoppingPatience: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Data seed"
                    valueDisplay={t.dataSeed}
                    help="Seeds batch sampling, separately from the weight seed - so you can vary data order while holding initialization fixed."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={t.dataSeed}
                        min={0}
                        max={9999}
                        step={1}
                        disabled={isRunning}
                        onChange={(value) => updateTraining({ dataSeed: value })}
                      />
                    )}
                  </Field>

                  <Field
                    label="Train / val / test split"
                    valueDisplay={`${fmt.percent(t.trainRatio, 0)} / ${fmt.percent(t.valRatio, 0)} / ${fmt.percent(t.testRatio, 0)}`}
                    error={errors.trainRatio}
                    help="Contiguous cuts, not random windows - so no window can appear in two splits."
                  >
                    {(id) => (
                      <SliderInput
                        id={id}
                        value={t.trainRatio}
                        min={0.5}
                        max={0.95}
                        step={0.01}
                        disabled={isRunning}
                        onChange={(value) => {
                          // Keep the three ratios summing to exactly 1 by
                          // splitting the remainder evenly - otherwise every
                          // adjustment produces a validation error.
                          const remainder = 1 - value;
                          updateTraining({
                            trainRatio: Number(value.toFixed(4)),
                            valRatio: Number((remainder / 2).toFixed(4)),
                            testRatio: Number((remainder / 2).toFixed(4)),
                          });
                        }}
                      />
                    )}
                  </Field>
                </div>
              </Panel>
            </>
          )}
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* Right: consequences                                              */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex min-w-0 flex-col gap-3">
          <Panel title="Model size" description="Updates as you edit, before anything is applied.">
            <div className="mb-3 grid grid-cols-2 gap-2">
              <Stat label="Parameters" value={fmt.compactInt(estimate.total)} hint={fmt.int(estimate.total)} />
              <Stat
                label="Weights size"
                value={fmt.bytes(estimate.total * 4)}
                hint="float32 on disk"
              />
            </div>
            <BarList
              labelWidth={120}
              items={estimate.rows
                .filter((row) => row.parameters > 0)
                .sort((a, b) => b.parameters - a.parameters)
                .map((row) => ({
                  label: row.component.replace(' (all layers)', ''),
                  value: row.parameters,
                  display: fmt.compactInt(row.parameters),
                }))}
            />
            <p className="mt-2.5 text-2xs leading-relaxed text-content-tertiary">
              Attention and feed-forward scale with depth; the embedding table scales with
              vocabulary. Switch to the word tokenizer on the Data tab to see embeddings become
              most of the model.
            </p>
          </Panel>

          <Panel
            title="Verify backpropagation"
            description="Checks the analytic gradients against numerical finite differences."
            actions={
              <Button size="sm" variant="secondary" onClick={runGradientCheck} loading={checking} disabled={isRunning}>
                Run check
              </Button>
            }
          >
            {gradientCheck ? (
              <GradientCheckReport result={gradientCheck} />
            ) : (
              <p className="text-xs leading-relaxed text-content-tertiary">
                Perturbs each tensor along its own gradient direction and compares the measured
                change in loss against the gradient's norm. This is the only test that genuinely
                proves the hand-written backward pass is correct - a model whose gradients are
                transposed or sign-flipped still "trains", just badly and silently.
              </p>
            )}
          </Panel>
        </div>
      </div>

      <ParameterReference docs={docs} />
    </div>
  );
};

// ---------------------------------------------------------------------------

const GradientCheckReport: React.FC<{ result: GradientCheckResult }> = ({ result }) => {
  const measured = result.checks.filter((c) => !c.skipped);
  const skipped = result.checks.length - measured.length;
  const worst = [...measured].sort((a, b) => b.relativeError - a.relativeError)[0];

  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex items-center gap-2">
        <Badge tone={result.passed ? 'success' : 'danger'}>
          {result.passed ? 'PASS' : 'FAIL'}
        </Badge>
        <span className="text-xs text-content-secondary">
          {measured.length} tensors verified
          {skipped > 0 && `, ${skipped} below float32 resolution`}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Stat
          label="All parameters at once"
          value={result.global.relativeError.toExponential(2)}
          hint="relative error"
          tone={result.global.relativeError < 1e-3 ? 'success' : 'default'}
          title="Every weight perturbed simultaneously along the full gradient direction. The strongest single assertion available."
        />
        <Stat
          label="Worst tensor"
          value={worst ? worst.relativeError.toExponential(2) : '-'}
          hint={worst ? worst.variable.replace(/^gpt_\d+_\d+_/, '').slice(0, 22) : undefined}
        />
      </div>

      <p className="text-2xs leading-relaxed text-content-tertiary">
        Tolerance widens for tensors with small gradients: the numeric estimate carries a fixed
        absolute uncertainty of about {(result.noiseFloor ?? 0).toExponential(1)}, which is
        relatively larger when the true derivative is small. A genuinely wrong gradient is wrong by
        order 100% and is caught regardless.
      </p>

      {!result.passed && worst && (
        <Note tone="danger" title="Gradients disagree">
          {worst.variable}: analytic {worst.analyticDerivative.toExponential(3)} vs numeric{' '}
          {worst.numericDerivative.toExponential(3)}.
        </Note>
      )}
    </div>
  );
};
