import React, { useMemo, useState } from 'react';
import { Panel, Badge, TextInput, Segmented, EmptyState } from './ui';
import { ParameterDocumentation } from '../../shared/config';

/**
 * ============================================================================
 * Parameter reference
 * ============================================================================
 *
 * The full documentation for every hyperparameter: what it does, what happens
 * when you raise or lower it, its tradeoffs, and a concrete experiment that
 * demonstrates its effect.
 *
 * WHY IT IS SEPARATE FROM THE CONTROLS
 * Each slider already carries a one-line explanation, which is the right
 * amount while you are adjusting something. But the one-line form cannot hold
 * "train 1000 steps with constant, then with cosine, and note the difference
 * in the last quarter" - and that suggested experiment is often the most
 * valuable part. Putting the long form in a searchable reference keeps the
 * controls uncluttered without discarding the depth.
 *
 * Each entry's `exampleExperiment` is deliberately prominent: this tool exists
 * to be experimented with, and a concrete thing to try is more useful than any
 * amount of description.
 */

type CategoryFilter = 'all' | 'model' | 'training' | 'generation';

export const ParameterReference: React.FC<{
  docs: Record<string, ParameterDocumentation>;
}> = ({ docs }) => {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState<CategoryFilter>('all');
  const [expanded, setExpanded] = useState<string | null>(null);

  const entries = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return Object.entries(docs)
      .filter(([key, doc]) => {
        if (category !== 'all' && doc.category !== category) return false;
        if (normalizedQuery.length === 0) return true;
        // Search the explanation text too, not just the name - someone looking
        // for "overfitting" should find dropout and weight decay.
        return (
          key.toLowerCase().includes(normalizedQuery) ||
          doc.name.toLowerCase().includes(normalizedQuery) ||
          doc.simpleExplanation.toLowerCase().includes(normalizedQuery) ||
          doc.advantages.toLowerCase().includes(normalizedQuery) ||
          doc.disadvantages.toLowerCase().includes(normalizedQuery)
        );
      })
      .sort(([, a], [, b]) => a.name.localeCompare(b.name));
  }, [docs, query, category]);

  return (
    <Panel
      title="Parameter reference"
      description="What each hyperparameter does, and an experiment that demonstrates it."
      actions={
        <Segmented
          size="sm"
          value={category}
          onChange={setCategory}
          ariaLabel="Category"
          options={[
            { value: 'all', label: 'All' },
            { value: 'model', label: 'Model' },
            { value: 'training', label: 'Training' },
            { value: 'generation', label: 'Sampling' },
          ]}
        />
      }
    >
      <TextInput
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Search parameters, or a concept like 'overfitting'..."
        className="mb-3"
      />

      {entries.length === 0 ? (
        <EmptyState compact title="No matching parameters" />
      ) : (
        <div className="flex flex-col gap-1.5">
          {entries.map(([key, doc]) => {
            const isOpen = expanded === key;
            return (
              <div
                key={key}
                className="overflow-hidden rounded-md border border-edge-subtle bg-surface-input"
              >
                <button
                  type="button"
                  aria-expanded={isOpen}
                  onClick={() => setExpanded(isOpen ? null : key)}
                  className="flex w-full items-start justify-between gap-3 px-3 py-2 text-left transition-colors hover:bg-surface-hover"
                >
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-xs font-semibold text-content-primary">
                        {doc.name}
                      </span>
                      <span className="num text-2xs text-content-tertiary">{key}</span>
                      <Badge tone="neutral">{doc.category}</Badge>
                    </div>
                    {!isOpen && (
                      <p className="mt-0.5 line-clamp-2 text-2xs leading-snug text-content-tertiary">
                        {doc.simpleExplanation}
                      </p>
                    )}
                  </div>
                  <span className="num shrink-0 text-2xs text-content-tertiary">
                    {String(doc.defaultValue)}
                  </span>
                </button>

                {isOpen && (
                  <div className="border-t border-edge-subtle px-3 py-2.5">
                    <p className="text-xs leading-relaxed text-content-secondary">
                      {doc.simpleExplanation}
                    </p>

                    <div className="mt-2.5 grid grid-cols-1 gap-2 sm:grid-cols-2">
                      <DocBlock label="Raising it" text={doc.increasingEffect} />
                      <DocBlock label="Lowering it" text={doc.decreasingEffect} />
                      <DocBlock label="Why it helps" text={doc.advantages} tone="success" />
                      <DocBlock label="What it costs" text={doc.disadvantages} tone="warning" />
                    </div>

                    <div className="mt-2.5 rounded border border-accent-border bg-accent-subtle px-2.5 py-2">
                      <div className="label mb-0.5 text-accent-text">Try this</div>
                      <p className="text-xs leading-relaxed text-content-secondary">
                        {doc.exampleExperiment}
                      </p>
                    </div>

                    <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-2xs text-content-tertiary">
                      <span>
                        default <span className="num">{String(doc.defaultValue)}</span>
                      </span>
                      {doc.minValue !== undefined && (
                        <span>
                          range{' '}
                          <span className="num">
                            {doc.minValue} – {doc.maxValue}
                          </span>
                        </span>
                      )}
                      <span className="flex items-center gap-1">
                        affects
                        {doc.affects.map((effect) => (
                          <Badge key={effect} tone="neutral">
                            {effect}
                          </Badge>
                        ))}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
};

const DocBlock: React.FC<{
  label: string;
  text: string;
  tone?: 'default' | 'success' | 'warning';
}> = ({ label, text, tone = 'default' }) => (
  <div className="rounded border border-edge-subtle bg-surface-raised px-2.5 py-2">
    <div
      className={`label mb-0.5 ${
        tone === 'success' ? 'text-success' : tone === 'warning' ? 'text-warning' : ''
      }`}
    >
      {label}
    </div>
    <p className="text-2xs leading-relaxed text-content-secondary">{text}</p>
  </div>
);
