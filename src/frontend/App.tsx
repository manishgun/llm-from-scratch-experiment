import React, { useEffect, useState } from 'react';
import { StoreProvider, useStore } from './lib/store';
import { ToastStack, Badge, StatusDot, Button, Note } from './components/ui';
import { RunView } from './views/RunView';
import { InferenceView } from './views/InferenceView';
import { ArchitectureView } from './views/ArchitectureView';
import { DataView } from './views/DataView';
import { AttentionView } from './views/AttentionView';
import { CheckpointsView } from './views/CheckpointsView';
import { ExperimentsView } from './views/ExperimentsView';
import * as fmt from './lib/format';

/**
 * ============================================================================
 * MiniGPT Lab - Application Shell
 * ============================================================================
 *
 * A fixed icon rail, a top bar carrying the live run state, a scrolling view
 * area and a status bar.
 *
 * WHY THE RUN STATE LIVES IN THE SHELL
 * Training continues regardless of which tab is open. Someone adjusting
 * sampling parameters on the Inference tab still needs to know the run is
 * progressing, and whether it has failed. Putting step, loss and status in the
 * top bar means that information is never more than a glance away, and the
 * controls to stop a bad run are reachable from every screen.
 */

type ViewId = 'run' | 'inference' | 'attention' | 'architecture' | 'data' | 'checkpoints' | 'experiments';

interface ViewDefinition {
  id: ViewId;
  label: string;
  shortcut: string;
  icon: React.FC;
  component: React.FC;
  /** Shown beneath the title in the content header. */
  subtitle: string;
}

const VIEWS: ViewDefinition[] = [
  {
    id: 'run',
    label: 'Run',
    shortcut: '1',
    icon: () => (
      <svg width="17" height="17" viewBox="0 0 17 17" fill="none" aria-hidden="true">
        <path
          d="M2 13.5L5.5 8l3 3L14 3.5"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path d="M2 15.5h13" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      </svg>
    ),
    component: RunView,
    subtitle: 'Live training metrics, diagnostics and the engine log',
  },
  {
    id: 'inference',
    label: 'Inference',
    shortcut: '2',
    icon: () => (
      <svg width="17" height="17" viewBox="0 0 17 17" fill="none" aria-hidden="true">
        <path
          d="M8.5 2l1.7 4.3L14.5 8l-4.3 1.7L8.5 14l-1.7-4.3L2.5 8l4.3-1.7z"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
      </svg>
    ),
    component: InferenceView,
    subtitle: 'Generate text and inspect the distribution behind every token',
  },
  {
    id: 'attention',
    label: 'Attention',
    shortcut: '3',
    icon: () => (
      <svg width="17" height="17" viewBox="0 0 17 17" fill="none" aria-hidden="true">
        <rect x="2.5" y="2.5" width="12" height="12" rx="1.5" stroke="currentColor" strokeWidth="1.4" />
        <path d="M2.5 6.5h12M2.5 10.5h12M6.5 2.5v12M10.5 2.5v12" stroke="currentColor" strokeWidth="0.9" opacity="0.5" />
        <rect x="2.5" y="2.5" width="4" height="4" fill="currentColor" opacity="0.5" />
        <rect x="6.5" y="6.5" width="4" height="4" fill="currentColor" opacity="0.3" />
      </svg>
    ),
    component: AttentionView,
    subtitle: 'What each token attends to, per layer and per head',
  },
  {
    id: 'architecture',
    label: 'Architecture',
    shortcut: '4',
    icon: () => (
      <svg width="17" height="17" viewBox="0 0 17 17" fill="none" aria-hidden="true">
        <path
          d="M3 4.5h11M3 8.5h11M3 12.5h11"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinecap="round"
        />
        <circle cx="6" cy="4.5" r="1.6" fill="var(--surface-base)" stroke="currentColor" strokeWidth="1.3" />
        <circle cx="10.5" cy="8.5" r="1.6" fill="var(--surface-base)" stroke="currentColor" strokeWidth="1.3" />
        <circle cx="7" cy="12.5" r="1.6" fill="var(--surface-base)" stroke="currentColor" strokeWidth="1.3" />
      </svg>
    ),
    component: ArchitectureView,
    subtitle: 'Hyperparameters, with their consequences shown live',
  },
  {
    id: 'data',
    label: 'Data',
    shortcut: '5',
    icon: () => (
      <svg width="17" height="17" viewBox="0 0 17 17" fill="none" aria-hidden="true">
        <ellipse cx="8.5" cy="4.5" rx="5.5" ry="2.2" stroke="currentColor" strokeWidth="1.4" />
        <path d="M3 4.5v7.8c0 1.2 2.5 2.2 5.5 2.2s5.5-1 5.5-2.2V4.5" stroke="currentColor" strokeWidth="1.4" />
        <path d="M3 8.4c0 1.2 2.5 2.2 5.5 2.2s5.5-1 5.5-2.2" stroke="currentColor" strokeWidth="1.2" opacity="0.6" />
      </svg>
    ),
    component: DataView,
    subtitle: 'Corpus, tokenization and the train/validation/test splits',
  },
  {
    id: 'checkpoints',
    label: 'Checkpoints',
    shortcut: '6',
    icon: () => (
      <svg width="17" height="17" viewBox="0 0 17 17" fill="none" aria-hidden="true">
        <path
          d="M4 2.5h9v12l-4.5-3.2L4 14.5z"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinejoin="round"
        />
      </svg>
    ),
    component: CheckpointsView,
    subtitle: 'Saved weights, and resuming a run exactly where it stopped',
  },
  {
    id: 'experiments',
    label: 'Experiments',
    shortcut: '7',
    icon: () => (
      <svg width="17" height="17" viewBox="0 0 17 17" fill="none" aria-hidden="true">
        <path
          d="M6.5 2.5h4v3.2l3.2 6.1a1.5 1.5 0 0 1-1.3 2.2H4.6a1.5 1.5 0 0 1-1.3-2.2l3.2-6.1z"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinejoin="round"
        />
        <path d="M5.2 10.5h6.6" stroke="currentColor" strokeWidth="1.2" />
      </svg>
    ),
    component: ExperimentsView,
    subtitle: 'Compare runs, and see exactly which parameters differ',
  },
];

const AppShell: React.FC = () => {
  const [view, setView] = useState<ViewId>('run');
  const {
    status,
    step,
    latest,
    bestValLoss,
    parameters,
    config,
    connection,
    system,
    toasts,
    dismissToast,
    theme,
    toggleTheme,
    pauseTraining,
    startTraining,
  } = useStore();

  // Keyboard navigation: digits switch views. Guarded so it never fires while
  // the user is typing a prompt or editing a number.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable);
      if (typing || event.metaKey || event.ctrlKey || event.altKey) return;

      const match = VIEWS.find((candidate) => candidate.shortcut === event.key);
      if (match) {
        event.preventDefault();
        setView(match.id);
        return;
      }
      // Space toggles the run, the way a media player would.
      if (event.key === ' ') {
        event.preventDefault();
        if (status === 'running') void pauseTraining();
        else void startTraining();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [status, pauseTraining, startTraining]);

  const active = VIEWS.find((candidate) => candidate.id === view) ?? VIEWS[0];
  const ActiveComponent = active.component;

  const statusTone =
    connection === 'closed'
      ? 'offline'
      : status === 'running'
        ? 'running'
        : status === 'paused'
          ? 'paused'
          : status === 'error'
            ? 'error'
            : status === 'completed'
              ? 'completed'
              : 'idle';

  const statusLabel =
    connection === 'closed' ? 'Backend offline' : status.charAt(0).toUpperCase() + status.slice(1);

  return (
    <div className="flex h-full flex-col bg-surface-base">
      <div className="flex min-h-0 flex-1">
        {/* ---------------------------------------------------------------- */}
        {/* Navigation rail                                                  */}
        {/* ---------------------------------------------------------------- */}
        <nav
          aria-label="Main"
          className="flex w-rail shrink-0 flex-col items-center gap-1 border-r border-edge-subtle bg-surface-raised py-2.5"
        >
          <div
            className="mb-2 flex h-7 w-7 items-center justify-center rounded-md bg-accent text-[11px] font-bold text-white"
            title="MiniGPT Lab"
          >
            GP
          </div>

          {VIEWS.map((definition) => {
            const Icon = definition.icon;
            const isActive = definition.id === view;
            return (
              <button
                key={definition.id}
                type="button"
                onClick={() => setView(definition.id)}
                aria-current={isActive ? 'page' : undefined}
                title={`${definition.label}  (${definition.shortcut})`}
                className={`group relative flex h-9 w-9 items-center justify-center rounded-md transition-colors
                  focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent
                  ${
                    isActive
                      ? 'bg-accent-subtle text-accent-text'
                      : 'text-content-tertiary hover:bg-surface-hover hover:text-content-primary'
                  }`}
              >
                {isActive && (
                  <span
                    aria-hidden="true"
                    className="absolute -left-[9px] top-1.5 bottom-1.5 w-[2px] rounded-full bg-accent"
                  />
                )}
                <Icon />
                <span className="sr-only">{definition.label}</span>
              </button>
            );
          })}

          <div className="mt-auto flex flex-col items-center gap-1">
            <button
              type="button"
              onClick={toggleTheme}
              title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
              className="flex h-8 w-8 items-center justify-center rounded-md text-content-tertiary transition-colors hover:bg-surface-hover hover:text-content-primary"
            >
              {theme === 'dark' ? (
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
                  <circle cx="7.5" cy="7.5" r="3" stroke="currentColor" strokeWidth="1.4" />
                  <path
                    d="M7.5 1v1.5M7.5 12.5V14M1 7.5h1.5M12.5 7.5H14M3 3l1 1M11 11l1 1M12 3l-1 1M4 11l-1 1"
                    stroke="currentColor"
                    strokeWidth="1.3"
                    strokeLinecap="round"
                  />
                </svg>
              ) : (
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
                  <path
                    d="M13 9.5A6 6 0 0 1 5.5 2a6 6 0 1 0 7.5 7.5z"
                    stroke="currentColor"
                    strokeWidth="1.4"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
              <span className="sr-only">Toggle theme</span>
            </button>
          </div>
        </nav>

        {/* ---------------------------------------------------------------- */}
        {/* Main column                                                      */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex min-w-0 flex-1 flex-col">
          {/* Top bar */}
          <header className="flex h-topbar shrink-0 items-center justify-between gap-4 border-b border-edge-subtle bg-surface-raised px-4">
            <div className="min-w-0">
              <h1 className="truncate text-sm font-semibold text-content-primary">
                {active.label}
              </h1>
              <p className="truncate text-2xs text-content-tertiary">{active.subtitle}</p>
            </div>

            {/* Live run readout, visible from every view */}
            <div className="flex shrink-0 items-center gap-4">
              {step > 0 && (
                <div className="hidden items-center gap-4 sm:flex">
                  <TopMetric
                    label="step"
                    value={`${fmt.int(step)} / ${fmt.int(config?.training.trainingSteps ?? 0)}`}
                  />
                  <TopMetric
                    label="train"
                    value={fmt.loss(latest?.trainLoss)}
                    color="var(--series-train)"
                  />
                  <TopMetric
                    label="best val"
                    value={fmt.loss(bestValLoss ?? undefined)}
                    color="var(--series-val)"
                  />
                </div>
              )}

              <div className="flex items-center gap-1.5 rounded-md border border-edge-subtle bg-surface-input px-2 py-1">
                <StatusDot tone={statusTone} pulse={status === 'running'} />
                <span className="text-2xs font-medium text-content-secondary">{statusLabel}</span>
              </div>
            </div>
          </header>

          {/* Backend-offline banner. Nothing in the app works without it, so
              this is the one condition that interrupts the whole layout. */}
          {connection === 'closed' && (
            <div className="shrink-0 border-b border-danger/40 bg-danger-subtle px-4 py-2">
              <p className="text-xs text-content-primary">
                <span className="font-semibold">Backend offline.</span> Reconnecting
                automatically. If it does not come back, start it with{' '}
                <code className="num rounded bg-surface-input px-1">npm run dev</code>.
              </p>
            </div>
          )}

          {/* View */}
          <main className="min-h-0 flex-1 overflow-y-auto p-4">
            <ActiveComponent />
          </main>

          {/* Status bar */}
          <footer className="flex h-statusbar shrink-0 items-center justify-between gap-4 border-t border-edge-subtle bg-surface-raised px-3 text-2xs text-content-tertiary">
            <div className="flex items-center gap-3">
              {/*
               * The active TensorFlow backend.
               *
               * This is the single most valuable thing in the status bar. The
               * native backend is roughly 10x faster than the pure-JavaScript
               * fallback, and if the native binding fails to load TF.js falls
               * back SILENTLY. Without this indicator a broken install just
               * looks like "training on CPU is slow", and nobody investigates.
               */}
              {system && (
                <span
                  className="flex items-center gap-1.5"
                  title={
                    system.nativeAccelerated
                      ? 'The native TensorFlow backend is active.'
                      : 'The native backend did not load, so TensorFlow.js fell back to pure JavaScript. Training will be roughly 10x slower than it should be.'
                  }
                >
                  <StatusDot tone={system.nativeAccelerated ? 'running' : 'error'} />
                  <span className="num">{system.tfBackend}</span>
                  {!system.nativeAccelerated && (
                    <span className="text-danger">(not accelerated)</span>
                  )}
                </span>
              )}
              {system && (
                <>
                  <Divider />
                  <span className="num" title={system.cpuModel}>
                    {system.cpuCores} cores
                  </span>
                  <Divider />
                  <span className="num">tfjs {system.tfVersion}</span>
                </>
              )}
            </div>

            <div className="flex items-center gap-3">
              {parameters !== null && (
                <>
                  <span className="num" title={`${fmt.int(parameters)} trainable parameters`}>
                    {fmt.compactInt(parameters)} params
                  </span>
                  <Divider />
                </>
              )}
              {latest?.tensorCount !== undefined && (
                <>
                  <span
                    className="num"
                    title="Tensors held by TensorFlow. Steady growth with step count would indicate a leak."
                  >
                    {fmt.int(latest.tensorCount)} tensors
                  </span>
                  <Divider />
                </>
              )}
              {latest && (
                <>
                  <span className="num">{latest.memoryUsageMb.toFixed(0)} MB heap</span>
                  <Divider />
                </>
              )}
              <span className="hidden sm:inline">
                <span className="num">1</span>-<span className="num">7</span> switch view ·{' '}
                <span className="num">space</span> start/pause
              </span>
            </div>
          </footer>
        </div>
      </div>

      <ToastStack toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
};

const TopMetric: React.FC<{ label: string; value: string; color?: string }> = ({
  label,
  value,
  color,
}) => (
  <div className="flex items-baseline gap-1.5">
    {color && (
      <span
        aria-hidden="true"
        className="h-[2px] w-2.5 shrink-0 rounded-full"
        style={{ background: color }}
      />
    )}
    <span className="label">{label}</span>
    <span className="num text-xs text-content-primary">{value}</span>
  </div>
);

const Divider = () => (
  <span aria-hidden="true" className="h-2.5 w-px bg-edge-subtle" />
);

export const App: React.FC = () => (
  <StoreProvider>
    <AppShell />
  </StoreProvider>
);
