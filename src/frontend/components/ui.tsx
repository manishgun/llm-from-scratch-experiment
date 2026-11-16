import React, { useEffect, useId, useRef, useState } from 'react';

/**
 * ============================================================================
 * MiniGPT Lab - UI Primitives
 * ============================================================================
 *
 * The small set of components every view is built from. They exist so that
 * spacing, type scale, focus behaviour and disabled states are decided once
 * rather than re-improvised per screen - which is what makes an interface feel
 * like a tool rather than a collection of pages.
 *
 * Accessibility is handled here rather than per view: real <button> elements,
 * labels bound to inputs by id, `aria-*` on anything whose meaning is carried
 * by colour or position, and visible focus rings on every interactive element.
 */

// ---------------------------------------------------------------------------
// Button
// ---------------------------------------------------------------------------

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
type ButtonSize = 'sm' | 'md';

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary:
    'bg-accent text-white border-accent hover:bg-accent-hover disabled:bg-accent/40 disabled:border-transparent',
  secondary:
    'bg-surface-overlay text-content-primary border-edge hover:bg-surface-hover hover:border-edge-strong',
  ghost:
    'bg-transparent text-content-secondary border-transparent hover:bg-surface-hover hover:text-content-primary',
  danger:
    'bg-transparent text-danger border-edge hover:bg-danger-subtle hover:border-danger/50',
};

const BUTTON_SIZES: Record<ButtonSize, string> = {
  sm: 'h-7 px-2.5 text-xs gap-1.5',
  md: 'h-8 px-3 text-sm gap-2',
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: React.ReactNode;
  loading?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'secondary',
  size = 'md',
  icon,
  loading = false,
  children,
  className = '',
  disabled,
  ...rest
}) => (
  <button
    type="button"
    // A loading button must be unclickable, not merely styled as busy -
    // otherwise a double click fires the action twice.
    disabled={disabled || loading}
    aria-busy={loading || undefined}
    className={`inline-flex items-center justify-center whitespace-nowrap rounded-md border
      font-medium transition-colors
      focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent
      disabled:cursor-not-allowed disabled:opacity-55
      ${BUTTON_VARIANTS[variant]} ${BUTTON_SIZES[size]} ${className}`}
    {...rest}
  >
    {loading ? <Spinner /> : icon}
    {children}
  </button>
);

export const Spinner: React.FC<{ className?: string }> = ({ className = '' }) => (
  <svg
    className={`spin h-3.5 w-3.5 ${className}`}
    viewBox="0 0 16 16"
    fill="none"
    aria-hidden="true"
  >
    <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" />
    <path
      d="M14.5 8a6.5 6.5 0 0 0-6.5-6.5"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
    />
  </svg>
);

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

export interface PanelProps {
  title?: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  /** Removes the body padding, for tables and charts that bleed to the edge. */
  flush?: boolean;
}

export const Panel: React.FC<PanelProps> = ({
  title,
  description,
  actions,
  children,
  className = '',
  flush = false,
}) => (
  <section
    className={`flex min-w-0 flex-col overflow-hidden rounded-lg border border-edge-subtle bg-surface-raised ${className}`}
  >
    {(title || actions) && (
      <header className="flex shrink-0 items-start justify-between gap-3 border-b border-edge-subtle px-3.5 py-2.5">
        <div className="min-w-0">
          {title && (
            <h2 className="truncate text-xs font-semibold tracking-wide text-content-primary">
              {title}
            </h2>
          )}
          {description && (
            <p className="mt-0.5 text-xs leading-snug text-content-tertiary">{description}</p>
          )}
        </div>
        {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
      </header>
    )}
    <div className={`min-w-0 flex-1 ${flush ? '' : 'p-3.5'}`}>{children}</div>
  </section>
);

// ---------------------------------------------------------------------------
// Stat
// ---------------------------------------------------------------------------

export interface StatProps {
  label: string;
  value: React.ReactNode;
  unit?: string;
  /** Secondary line beneath the value. */
  hint?: React.ReactNode;
  tone?: 'default' | 'accent' | 'success' | 'warning' | 'danger';
  /** A coloured bar on the left edge, to tie the stat to a chart series. */
  seriesColor?: string;
  title?: string;
}

const STAT_TONES: Record<NonNullable<StatProps['tone']>, string> = {
  default: 'text-content-primary',
  accent: 'text-accent-text',
  success: 'text-success',
  warning: 'text-warning',
  danger: 'text-danger',
};

export const Stat: React.FC<StatProps> = ({
  label,
  value,
  unit,
  hint,
  tone = 'default',
  seriesColor,
  title,
}) => (
  <div
    className="relative min-w-0 rounded-md border border-edge-subtle bg-surface-input px-3 py-2"
    title={title}
  >
    {seriesColor && (
      <span
        aria-hidden="true"
        className="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-full"
        style={{ background: seriesColor }}
      />
    )}
    <div className="label truncate">{label}</div>
    <div className={`num mt-0.5 flex items-baseline gap-1 text-lg ${STAT_TONES[tone]}`}>
      <span className="truncate">{value}</span>
      {unit && <span className="text-xs font-normal text-content-tertiary">{unit}</span>}
    </div>
    {hint && <div className="mt-0.5 truncate text-2xs text-content-tertiary">{hint}</div>}
  </div>
);

// ---------------------------------------------------------------------------
// Badge
// ---------------------------------------------------------------------------

export type BadgeTone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger';

const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: 'bg-surface-overlay text-content-secondary border-edge-subtle',
  accent: 'bg-accent-subtle text-accent-text border-accent-border',
  success: 'bg-success-subtle text-success border-success/35',
  warning: 'bg-warning-subtle text-warning border-warning/35',
  danger: 'bg-danger-subtle text-danger border-danger/35',
};

export const Badge: React.FC<{
  tone?: BadgeTone;
  children: React.ReactNode;
  className?: string;
  title?: string;
}> = ({ tone = 'neutral', children, className = '', title }) => (
  <span
    title={title}
    className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-2xs font-semibold ${BADGE_TONES[tone]} ${className}`}
  >
    {children}
  </span>
);

// ---------------------------------------------------------------------------
// Field - the label + control + help wrapper
// ---------------------------------------------------------------------------

export interface FieldProps {
  label: string;
  /** Explanatory text, always visible rather than hidden behind a tooltip. */
  help?: React.ReactNode;
  /** The current value, rendered beside the label. */
  valueDisplay?: React.ReactNode;
  children: (id: string) => React.ReactNode;
  /** A validation message. Presence switches the field to its error state. */
  error?: string;
  className?: string;
}

export const Field: React.FC<FieldProps> = ({
  label,
  help,
  valueDisplay,
  children,
  error,
  className = '',
}) => {
  const id = useId();
  return (
    <div className={`min-w-0 ${className}`}>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-xs font-medium text-content-secondary">
          {label}
        </label>
        {valueDisplay !== undefined && (
          <span className="num shrink-0 text-xs text-content-primary">{valueDisplay}</span>
        )}
      </div>
      {children(id)}
      {error ? (
        <p className="mt-1 text-2xs leading-snug text-danger" role="alert">
          {error}
        </p>
      ) : (
        help && <p className="mt-1 text-2xs leading-snug text-content-tertiary">{help}</p>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

const INPUT_BASE = `w-full rounded-md border bg-surface-input px-2.5 text-content-primary
  placeholder:text-content-disabled transition-colors
  focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent-subtle
  disabled:cursor-not-allowed disabled:opacity-55`;

export const TextInput: React.FC<
  React.InputHTMLAttributes<HTMLInputElement> & { invalid?: boolean }
> = ({ className = '', invalid, ...rest }) => (
  <input
    aria-invalid={invalid || undefined}
    className={`${INPUT_BASE} h-8 text-sm ${
      invalid ? 'border-danger' : 'border-edge'
    } ${className}`}
    {...rest}
  />
);

export const NumberInput: React.FC<
  React.InputHTMLAttributes<HTMLInputElement> & { invalid?: boolean }
> = ({ className = '', invalid, ...rest }) => (
  <input
    type="number"
    aria-invalid={invalid || undefined}
    className={`num ${INPUT_BASE} h-8 text-sm ${
      invalid ? 'border-danger' : 'border-edge'
    } ${className}`}
    {...rest}
  />
);

export const TextArea: React.FC<React.TextareaHTMLAttributes<HTMLTextAreaElement>> = ({
  className = '',
  ...rest
}) => (
  <textarea
    className={`${INPUT_BASE} resize-y border-edge py-1.5 text-sm leading-relaxed ${className}`}
    {...rest}
  />
);

export const Select: React.FC<
  React.SelectHTMLAttributes<HTMLSelectElement> & { options: { value: string; label: string }[] }
> = ({ options, className = '', ...rest }) => (
  <select
    className={`${INPUT_BASE} h-8 cursor-pointer appearance-none border-edge pr-7 text-sm ${className}`}
    style={{
      // An inline SVG chevron, so the control does not depend on the browser's
      // native arrow (which ignores the dark theme on several platforms).
      backgroundImage:
        "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath d='M3 4.5L6 7.5L9 4.5' stroke='%236b7688' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E\")",
      backgroundRepeat: 'no-repeat',
      backgroundPosition: 'right 8px center',
    }}
    {...rest}
  >
    {options.map((option) => (
      <option key={option.value} value={option.value}>
        {option.label}
      </option>
    ))}
  </select>
);

/**
 * A slider paired with a number input.
 *
 * Both are present because they serve different intents: the slider is for
 * exploring a range by feel, the number box is for entering a value you already
 * know. A slider alone makes it impossible to set a learning rate of exactly
 * 0.0003; a number box alone gives no sense of the usable range.
 */
export const SliderInput: React.FC<{
  id?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  /** Renders the numeric box in scientific notation, for learning rates. */
  scientific?: boolean;
}> = ({ id, value, min, max, step, onChange, disabled, scientific = false }) => {
  // The text box keeps its own draft while focused, so a partially typed value
  // like "0.00" is not immediately clamped or reformatted under the cursor.
  const [draft, setDraft] = useState<string | null>(null);

  const commit = (raw: string) => {
    const parsed = Number(raw);
    setDraft(null);
    if (!Number.isFinite(parsed)) return;
    onChange(Math.min(max, Math.max(min, parsed)));
  };

  return (
    <div className="flex items-center gap-2.5">
      <input
        id={id}
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="min-w-0 flex-1"
      />
      <input
        type="text"
        inputMode="decimal"
        disabled={disabled}
        value={draft ?? (scientific ? value.toExponential(2) : String(value))}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={(event) => commit(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            commit((event.target as HTMLInputElement).value);
            (event.target as HTMLInputElement).blur();
          }
          if (event.key === 'Escape') setDraft(null);
        }}
        className={`num h-7 w-[82px] shrink-0 rounded border border-edge bg-surface-input px-1.5
          text-right text-xs text-content-primary
          focus:border-accent focus:outline-none disabled:opacity-55`}
      />
    </div>
  );
};

export const Toggle: React.FC<{
  id?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  label?: string;
}> = ({ id, checked, onChange, disabled, label }) => (
  <button
    id={id}
    type="button"
    role="switch"
    aria-checked={checked}
    aria-label={label}
    disabled={disabled}
    onClick={() => onChange(!checked)}
    className={`relative h-[18px] w-8 shrink-0 rounded-full border transition-colors
      focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent
      disabled:cursor-not-allowed disabled:opacity-55
      ${checked ? 'border-accent bg-accent' : 'border-edge bg-surface-overlay'}`}
  >
    <span
      aria-hidden="true"
      className={`absolute top-[2px] h-[12px] w-[12px] rounded-full bg-white transition-transform
        ${checked ? 'translate-x-[16px]' : 'translate-x-[2px]'}`}
    />
  </button>
);

// ---------------------------------------------------------------------------
// Segmented control
// ---------------------------------------------------------------------------

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  disabled,
  size = 'md',
  ariaLabel,
}: {
  value: T;
  options: { value: T; label: string; title?: string }[];
  onChange: (value: T) => void;
  disabled?: boolean;
  size?: 'sm' | 'md';
  ariaLabel?: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className="inline-flex rounded-md border border-edge-subtle bg-surface-input p-[2px]"
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={option.title}
            disabled={disabled}
            onClick={() => onChange(option.value)}
            className={`rounded-[4px] font-medium transition-colors
              focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent
              disabled:cursor-not-allowed disabled:opacity-55
              ${size === 'sm' ? 'px-2 py-0.5 text-2xs' : 'px-2.5 py-1 text-xs'}
              ${
                active
                  ? 'bg-surface-overlay text-content-primary shadow-sm'
                  : 'text-content-tertiary hover:text-content-secondary'
              }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Status dot
// ---------------------------------------------------------------------------

export const StatusDot: React.FC<{
  tone: 'idle' | 'running' | 'paused' | 'completed' | 'error' | 'offline';
  pulse?: boolean;
}> = ({ tone, pulse = false }) => {
  const colors: Record<typeof tone, string> = {
    idle: 'var(--text-tertiary)',
    running: 'var(--success)',
    paused: 'var(--warning)',
    completed: 'var(--accent)',
    error: 'var(--danger)',
    offline: 'var(--danger)',
  };
  return (
    <span
      aria-hidden="true"
      className={`inline-block h-[7px] w-[7px] shrink-0 rounded-full ${pulse ? 'pulse-dot' : ''}`}
      style={{ background: colors[tone] }}
    />
  );
};

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

/**
 * Shown wherever a panel has nothing to display.
 *
 * `action` is part of the contract rather than optional decoration: an empty
 * state that only says "no data" leaves the user stuck, while one that offers
 * the next step is how a tool teaches itself.
 */
export const EmptyState: React.FC<{
  title: string;
  description?: React.ReactNode;
  action?: React.ReactNode;
  icon?: React.ReactNode;
  compact?: boolean;
}> = ({ title, description, action, icon, compact = false }) => (
  <div
    className={`flex flex-col items-center justify-center text-center ${
      compact ? 'gap-1.5 py-6' : 'gap-2 py-12'
    }`}
  >
    {icon && <div className="text-content-disabled">{icon}</div>}
    <p className="text-sm font-medium text-content-secondary">{title}</p>
    {description && (
      <p className="max-w-sm text-xs leading-relaxed text-content-tertiary">{description}</p>
    )}
    {action && <div className="mt-1.5">{action}</div>}
  </div>
);

// ---------------------------------------------------------------------------
// Toasts
// ---------------------------------------------------------------------------

export const ToastStack: React.FC<{
  toasts: { id: number; tone: 'info' | 'success' | 'warning' | 'error'; title: string; detail?: string }[];
  onDismiss: (id: number) => void;
}> = ({ toasts, onDismiss }) => {
  const tones = {
    info: 'border-edge bg-surface-overlay text-content-primary',
    success: 'border-success/40 bg-surface-overlay text-content-primary',
    warning: 'border-warning/40 bg-surface-overlay text-content-primary',
    error: 'border-danger/50 bg-surface-overlay text-content-primary',
  };
  const accents = {
    info: 'var(--accent)',
    success: 'var(--success)',
    warning: 'var(--warning)',
    error: 'var(--danger)',
  };

  return (
    // aria-live so a screen reader announces failures, which are otherwise
    // conveyed only by a coloured strip appearing in a corner.
    <div
      aria-live="polite"
      className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-[340px] flex-col gap-2"
    >
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`pointer-events-auto relative overflow-hidden rounded-md border px-3 py-2.5 pl-3.5 shadow-lg ${tones[toast.tone]}`}
        >
          <span
            aria-hidden="true"
            className="absolute inset-y-0 left-0 w-[3px]"
            style={{ background: accents[toast.tone] }}
          />
          <div className="flex items-start justify-between gap-2">
            <p className="text-xs font-medium leading-snug">{toast.title}</p>
            <button
              type="button"
              onClick={() => onDismiss(toast.id)}
              aria-label="Dismiss"
              className="-mr-1 -mt-0.5 shrink-0 rounded p-0.5 text-content-tertiary hover:bg-surface-hover hover:text-content-primary"
            >
              <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
                <path
                  d="M3 3l6 6M9 3l-6 6"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
              </svg>
            </button>
          </div>
          {toast.detail && (
            <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-2xs leading-relaxed text-content-tertiary">
              {toast.detail}
            </pre>
          )}
        </div>
      ))}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

export const Modal: React.FC<{
  open: boolean;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  footer?: React.ReactNode;
  width?: number;
}> = ({ open, title, onClose, children, footer, width = 420 }) => {
  const panelRef = useRef<HTMLDivElement>(null);

  // Escape closes, and focus moves into the dialog on open. Without the focus
  // move, keyboard users stay on the element behind the overlay.
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    panelRef.current?.focus();
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/55 p-4"
      onMouseDown={(event) => {
        // Close only on a click that both starts and ends on the backdrop, so
        // a text selection dragged out of the dialog does not dismiss it.
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        style={{ width }}
        className="max-h-[85vh] overflow-hidden rounded-lg border border-edge bg-surface-raised shadow-2xl focus:outline-none"
      >
        <header className="flex items-center justify-between border-b border-edge-subtle px-4 py-3">
          <h2 className="text-sm font-semibold text-content-primary">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded p-1 text-content-tertiary hover:bg-surface-hover hover:text-content-primary"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
              <path
                d="M3.5 3.5l7 7M10.5 3.5l-7 7"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </header>
        <div className="max-h-[60vh] overflow-y-auto p-4">{children}</div>
        {footer && (
          <footer className="flex justify-end gap-2 border-t border-edge-subtle px-4 py-3">
            {footer}
          </footer>
        )}
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

export const Table: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = '',
}) => (
  <div className={`overflow-x-auto ${className}`}>
    <table className="w-full border-collapse text-xs">{children}</table>
  </div>
);

export const Th: React.FC<{
  children?: React.ReactNode;
  align?: 'left' | 'right' | 'center';
  className?: string;
}> = ({ children, align = 'left', className = '' }) => (
  <th
    scope="col"
    className={`label sticky top-0 z-10 border-b border-edge-subtle bg-surface-overlay px-3 py-2 ${
      align === 'right' ? 'text-right' : align === 'center' ? 'text-center' : 'text-left'
    } ${className}`}
  >
    {children}
  </th>
);

export const Td: React.FC<{
  children?: React.ReactNode;
  align?: 'left' | 'right' | 'center';
  className?: string;
  mono?: boolean;
  title?: string;
}> = ({ children, align = 'left', className = '', mono = false, title }) => (
  <td
    title={title}
    className={`border-b border-edge-subtle px-3 py-1.5 text-content-secondary ${
      align === 'right' ? 'text-right' : align === 'center' ? 'text-center' : 'text-left'
    } ${mono ? 'num' : ''} ${className}`}
  >
    {children}
  </td>
);

// ---------------------------------------------------------------------------
// Keyboard hint
// ---------------------------------------------------------------------------

export const Kbd: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <kbd className="num rounded border border-edge bg-surface-input px-1 py-px text-2xs text-content-tertiary">
    {children}
  </kbd>
);

// ---------------------------------------------------------------------------
// Note - an inline explanatory callout
// ---------------------------------------------------------------------------

export const Note: React.FC<{
  tone?: 'info' | 'warning' | 'danger' | 'success';
  title?: string;
  children: React.ReactNode;
  className?: string;
}> = ({ tone = 'info', title, children, className = '' }) => {
  const tones = {
    info: 'border-accent-border bg-accent-subtle',
    warning: 'border-warning/40 bg-warning-subtle',
    danger: 'border-danger/45 bg-danger-subtle',
    success: 'border-success/40 bg-success-subtle',
  };
  return (
    <div className={`rounded-md border px-3 py-2 ${tones[tone]} ${className}`}>
      {title && <p className="mb-0.5 text-xs font-semibold text-content-primary">{title}</p>}
      <div className="text-xs leading-relaxed text-content-secondary">{children}</div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Progress bar
// ---------------------------------------------------------------------------

export const ProgressBar: React.FC<{
  value: number;
  max: number;
  className?: string;
  tone?: string;
}> = ({ value, max, className = '', tone = 'var(--accent)' }) => {
  const fraction = max > 0 ? Math.min(1, Math.max(0, value / max)) : 0;
  return (
    <div
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={max}
      className={`h-1 w-full overflow-hidden rounded-full bg-surface-input ${className}`}
    >
      <div
        className="h-full rounded-full transition-[width] duration-300"
        style={{ width: `${fraction * 100}%`, background: tone }}
      />
    </div>
  );
};
