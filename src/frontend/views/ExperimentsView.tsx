import React, { useEffect, useMemo, useState } from 'react';
import { useStore } from '../lib/store';
import { api } from '../lib/api';
import {
  Panel,
  Button,
  Badge,
  Note,
  Table,
  Th,
  Td,
  EmptyState,
  Modal,
  Field,
  TextInput,
  TextArea,
  Segmented,
} from '../components/ui';
import { LineChart, Legend, Series } from '../components/charts';
import * as fmt from '../lib/format';
import { ExperimentRecord } from '../../shared/types';

/**
 * ============================================================================
 * Experiments - comparing runs against each other
 * ============================================================================
 *
 * This is what turns the project from a demo into an instrument.
 *
 * A single loss curve tells you almost nothing on its own: is 1.62 good? The
 * only way to know is to run the same thing with one parameter changed and
 * overlay the two curves. That requires three things working together, all of
 * which are in place:
 *
 *   1. SEEDED EVERYTHING - weights, batch order, dropout masks. Without it, two
 *      runs differ for two reasons at once and no difference can be attributed.
 *   2. FIXED VALIDATION BATCHES, so the val curves are comparable point for
 *      point rather than each scoring different text.
 *   3. SAVED RUNS WITH THEIR FULL CONFIG, so the diff between two curves can be
 *      traced to the exact parameters that differ.
 *
 * The config diff below is the payoff: select two runs and the table shows only
 * the fields that are not identical.
 */

/** Series colours assigned in order, so each overlaid run is distinguishable. */
const COMPARE_COLORS = [
  'var(--series-train)',
  'var(--series-val)',
  'var(--series-grad)',
  'var(--series-lr)',
  'var(--series-throughput)',
  '#ec4899',
];

export const ExperimentsView: React.FC = () => {
  const { history, step, config, parameters, bestValLoss, latest, reportError, pushToast } =
    useStore();

  const [experiments, setExperiments] = useState<ExperimentRecord[] | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [saveOpen, setSaveOpen] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [busy, setBusy] = useState(false);
  const [metric, setMetric] = useState<'val' | 'train'>('val');
  const [includeCurrent, setIncludeCurrent] = useState(true);

  const refresh = async () => {
    try {
      setExperiments(await api.experiments());
    } catch (error) {
      reportError(error, 'Could not load experiments');
      setExperiments([]);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const save = async () => {
    setBusy(true);
    try {
      const result = await api.saveExperiment(name, description);
      await refresh();
      // Select it immediately, so the comparison updates without a second click.
      setSelected((previous) => new Set([...previous, result.id]));
      setSaveOpen(false);
      setName('');
      setDescription('');
      pushToast('success', 'Experiment saved');
    } catch (error) {
      reportError(error, 'Could not save experiment');
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: string) => {
    try {
      await api.deleteExperiment(id);
      setSelected((previous) => {
        const next = new Set(previous);
        next.delete(id);
        return next;
      });
      await refresh();
    } catch (error) {
      reportError(error, 'Could not delete experiment');
    }
  };

  const toggle = (id: string) => {
    setSelected((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const chosen = useMemo(
    () => (experiments ?? []).filter((experiment) => selected.has(experiment.id)),
    [experiments, selected]
  );

  // --- Overlay series ------------------------------------------------------
  const series = useMemo<Series[]>(() => {
    const result: Series[] = [];

    chosen.forEach((experiment, index) => {
      const points = experiment.metricsHistory
        .filter((m) => (metric === 'val' ? m.valLoss !== undefined : true))
        .map((m) => ({
          x: m.step,
          y: metric === 'val' ? (m.valLoss as number) : m.trainLoss,
        }));
      if (points.length === 0) return;
      result.push({
        key: experiment.id,
        label: experiment.name,
        color: COMPARE_COLORS[index % COMPARE_COLORS.length],
        points,
        width: 1.75,
        showDots: metric === 'val',
      });
    });

    if (includeCurrent && history.length > 0) {
      const points = history
        .filter((m) => (metric === 'val' ? m.valLoss !== undefined : true))
        .map((m) => ({
          x: m.step,
          y: metric === 'val' ? (m.valLoss as number) : m.trainLoss,
        }));
      if (points.length > 0) {
        result.push({
          key: '__current',
          label: 'current run',
          color: 'var(--text-primary)',
          points,
          width: 2,
          dashed: true,
          showDots: metric === 'val',
        });
      }
    }

    return result;
  }, [chosen, metric, includeCurrent, history]);

  // --- Config diff ---------------------------------------------------------
  /**
   * Finds the fields that differ across the selected runs.
   *
   * Showing the full configuration of each run side by side would be dozens of
   * identical rows with the two interesting ones buried. Only the differences
   * answer the question the user actually has: what did I change?
   */
  const diff = useMemo(() => {
    const configs = [
      ...chosen.map((experiment) => ({ label: experiment.name, config: experiment.config })),
      ...(includeCurrent && config ? [{ label: 'current run', config }] : []),
    ];
    if (configs.length < 2) return null;

    const rows: { field: string; values: string[] }[] = [];
    const sections: ('model' | 'training' | 'generation')[] = ['model', 'training'];

    for (const section of sections) {
      const keys = new Set<string>();
      for (const entry of configs) {
        Object.keys(entry.config[section] ?? {}).forEach((key) => keys.add(key));
      }
      for (const key of keys) {
        const values = configs.map((entry) => {
          const sectionValues = entry.config[section] as unknown as Record<string, unknown>;
          const value = sectionValues?.[key];
          return value === undefined ? '-' : String(value);
        });
        if (new Set(values).size > 1) {
          rows.push({ field: `${section}.${key}`, values });
        }
      }
    }

    return { labels: configs.map((entry) => entry.label), rows };
  }, [chosen, includeCurrent, config]);

  const canSaveCurrent = history.length > 0;

  return (
    <div className="flex flex-col gap-3">
      <Panel
        title="Comparison"
        description={
          series.length > 1
            ? 'Overlaid loss curves. Because every run is seeded, a difference between them is caused by the configuration - not by luck.'
            : 'Select two or more runs below to overlay their curves.'
        }
        actions={
          <>
            <Segmented
              size="sm"
              value={metric}
              onChange={setMetric}
              ariaLabel="Metric"
              options={[
                { value: 'val', label: 'Validation' },
                { value: 'train', label: 'Train' },
              ]}
            />
            <label className="flex items-center gap-1.5 pl-1 text-2xs text-content-tertiary">
              <input
                type="checkbox"
                checked={includeCurrent}
                onChange={(event) => setIncludeCurrent(event.target.checked)}
                className="accent-[var(--accent)]"
              />
              current run
            </label>
          </>
        }
      >
        {series.length > 0 ? (
          <>
            <LineChart
              series={series}
              height={280}
              xLabel="step"
              formatY={(value) => value.toFixed(3)}
              emptyMessage="No comparable points."
            />
            <div className="mt-2">
              <Legend
                items={series.map((s) => ({
                  label: s.label,
                  color: s.color,
                  dashed: s.dashed,
                }))}
              />
            </div>
          </>
        ) : (
          <EmptyState
            title="Nothing to compare"
            description="Run a model, save it as an experiment, change one parameter, run again - then overlay the two."
          />
        )}
      </Panel>

      {diff && diff.rows.length > 0 && (
        <Panel
          title="What differs"
          description="Only the configuration fields that are not identical across the selected runs."
          flush
        >
          <Table>
            <thead>
              <tr>
                <Th>Field</Th>
                {diff.labels.map((label) => (
                  <Th key={label} align="right">
                    {label}
                  </Th>
                ))}
              </tr>
            </thead>
            <tbody>
              {diff.rows.map((row) => (
                <tr key={row.field} className="hover:bg-surface-hover">
                  <Td mono className="text-content-primary">
                    {row.field}
                  </Td>
                  {row.values.map((value, index) => (
                    <Td key={index} align="right" mono>
                      {value}
                    </Td>
                  ))}
                </tr>
              ))}
            </tbody>
          </Table>
        </Panel>
      )}

      {diff && diff.rows.length === 0 && (
        <Note tone="info" title="These runs used identical configurations">
          Any difference between their curves is therefore not caused by configuration. With the
          same seed they should be identical; if they are not, something outside the configuration
          changed - most likely the dataset.
        </Note>
      )}

      <Panel
        title="Saved experiments"
        description="Each record holds the full configuration and metric history of a run."
        actions={
          <>
            <Button size="sm" variant="ghost" onClick={refresh}>
              Refresh
            </Button>
            <Button
              size="sm"
              variant="primary"
              onClick={() => {
                // Pre-fill a name that identifies the run by its distinguishing
                // parameters, since that is what the user will want to compare.
                if (config) {
                  setName(
                    `${config.model.numLayers}L-d${config.model.embeddingDim}-lr${config.training.learningRate}-seed${config.model.randomSeed}`
                  );
                }
                setSaveOpen(true);
              }}
              disabled={!canSaveCurrent}
              title={canSaveCurrent ? undefined : 'Run some steps first'}
            >
              Save current run
            </Button>
          </>
        }
        flush
      >
        {experiments === null ? (
          <div className="p-3.5 text-xs text-content-tertiary">Loading...</div>
        ) : experiments.length === 0 ? (
          <div className="p-3.5">
            <EmptyState
              title="No saved experiments"
              description="Save the current run to start building a comparison set."
              action={
                <Button
                  variant="primary"
                  disabled={!canSaveCurrent}
                  onClick={() => setSaveOpen(true)}
                >
                  Save current run
                </Button>
              }
            />
          </div>
        ) : (
          <Table>
            <thead>
              <tr>
                <Th />
                <Th>Name</Th>
                <Th align="right">Steps</Th>
                <Th align="right">Final train</Th>
                <Th align="right">Best val</Th>
                <Th align="right">Params</Th>
                <Th align="right">Duration</Th>
                <Th>Saved</Th>
                <Th align="right" />
              </tr>
            </thead>
            <tbody>
              {experiments.map((experiment, index) => {
                const isSelected = selected.has(experiment.id);
                const color = isSelected
                  ? COMPARE_COLORS[
                      [...selected].indexOf(experiment.id) % COMPARE_COLORS.length
                    ]
                  : undefined;
                return (
                  <tr
                    key={experiment.id}
                    className={`cursor-pointer hover:bg-surface-hover ${
                      isSelected ? 'bg-accent-subtle' : ''
                    }`}
                    onClick={() => toggle(experiment.id)}
                  >
                    <Td className="w-8">
                      <span className="flex items-center gap-1.5">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggle(experiment.id)}
                          onClick={(event) => event.stopPropagation()}
                          className="accent-[var(--accent)]"
                          aria-label={`Compare ${experiment.name}`}
                        />
                        {color && (
                          <span
                            aria-hidden="true"
                            className="h-2.5 w-[3px] rounded-full"
                            style={{ background: color }}
                          />
                        )}
                      </span>
                    </Td>
                    <Td className="font-medium text-content-primary">
                      <div>{experiment.name}</div>
                      {experiment.description && (
                        <div className="text-2xs text-content-tertiary">
                          {experiment.description}
                        </div>
                      )}
                    </Td>
                    <Td align="right" mono>
                      {fmt.int(experiment.totalStepsCompleted)}
                    </Td>
                    <Td align="right" mono>
                      {fmt.loss(experiment.finalTrainLoss)}
                    </Td>
                    <Td align="right" mono className="text-content-primary">
                      {fmt.loss(experiment.finalValLoss)}
                    </Td>
                    <Td align="right" mono>
                      {fmt.compactInt(experiment.totalParameters)}
                    </Td>
                    <Td align="right" mono>
                      {fmt.duration(experiment.trainingDurationMs)}
                    </Td>
                    <Td mono>{fmt.dateTime(experiment.createdAt)}</Td>
                    <Td align="right">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-danger hover:bg-danger-subtle"
                        onClick={(event) => {
                          event.stopPropagation();
                          void remove(experiment.id);
                        }}
                      >
                        Delete
                      </Button>
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        )}
      </Panel>

      <Modal
        open={saveOpen}
        title="Save the current run"
        onClose={() => setSaveOpen(false)}
        width={480}
        footer={
          <>
            <Button variant="ghost" onClick={() => setSaveOpen(false)}>
              Cancel
            </Button>
            <Button variant="primary" loading={busy} disabled={name.length === 0} onClick={save}>
              Save
            </Button>
          </>
        }
      >
        <div className="flex flex-col gap-3">
          <Note tone="info">
            The server records the configuration and metric history it actually ran, not whatever
            the browser currently shows - so a saved experiment always matches reality.
          </Note>
          <Field
            label="Name"
            help="Naming it after the parameters you changed makes the comparison table readable later."
          >
            {(id) => (
              <TextInput id={id} value={name} onChange={(event) => setName(event.target.value)} />
            )}
          </Field>
          <Field label="Notes" help="Optional. What you were testing, and what you concluded.">
            {(id) => (
              <TextArea
                id={id}
                rows={3}
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                placeholder="e.g. Doubled d_model to see whether the plateau at 1.8 was a capacity limit."
              />
            )}
          </Field>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 rounded-md border border-edge-subtle bg-surface-input p-3 text-xs">
            {[
              ['Steps completed', fmt.int(step)],
              ['Final train loss', fmt.loss(latest?.trainLoss)],
              ['Best val loss', fmt.loss(bestValLoss ?? undefined)],
              ['Parameters', fmt.compactInt(parameters)],
            ].map(([label, value]) => (
              <div key={label} className="flex justify-between gap-2">
                <dt className="text-content-tertiary">{label}</dt>
                <dd className="num text-content-primary">{value}</dd>
              </div>
            ))}
          </dl>
        </div>
      </Modal>
    </div>
  );
};
