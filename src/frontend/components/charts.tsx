import React, { useMemo, useRef, useState } from 'react';

/**
 * ============================================================================
 * MiniGPT Lab - Charts
 * ============================================================================
 *
 * Hand-built SVG rather than a charting library, for three reasons specific to
 * this application:
 *
 * 1. LOG SCALE ON LOSS IS NOT OPTIONAL. A run falls from 4.2 to 1.5, but the
 *    interesting part - the last few tenths - occupies the bottom 10% of a
 *    linear axis and is unreadable. Most lightweight chart libraries treat log
 *    scales as an afterthought.
 *
 * 2. TWO SERIES AT DIFFERENT SAMPLING RATES. Training loss is recorded every
 *    step; validation loss only every `evaluationInterval` steps. They must
 *    share one x-axis with the sparse series drawn as a connected line through
 *    its own points, not resampled or gap-filled.
 *
 * 3. A FIXED SERIES IDENTITY. "Train" is always the same blue in every chart,
 *    including the stat tiles, so the mapping is learned once.
 *
 * Everything here is deterministic, dependency-free and themed from the same
 * CSS custom properties as the rest of the app.
 */

export interface Series {
  key: string;
  label: string;
  color: string;
  /** Points may be sparse; x values need not be contiguous. */
  points: { x: number; y: number }[];
  /** Renders as a dashed line, used for smoothed or reference overlays. */
  dashed?: boolean;
  width?: number;
  /** Draws a marker at every point, for sparse series like validation loss. */
  showDots?: boolean;
}

/** `nice` axis step sizes, so gridline labels are round numbers. */
function niceStep(range: number, targetTicks: number): number {
  if (range <= 0) return 1;
  const rough = range / targetTicks;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const normalized = rough / magnitude;
  // 1, 2, 5, 10 - the only multipliers that produce labels humans read easily.
  const multiplier = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return multiplier * magnitude;
}

function formatTick(value: number): string {
  const abs = Math.abs(value);
  if (abs === 0) return '0';
  if (abs >= 1e6) return `${(value / 1e6).toFixed(abs >= 1e7 ? 0 : 1)}M`;
  if (abs >= 1e4) return `${Math.round(value / 1e3)}K`;
  if (abs >= 1000) return value.toLocaleString('en-US');
  if (abs >= 1) return Number(value.toFixed(2)).toString();
  if (abs >= 0.001) return Number(value.toFixed(4)).toString();
  return value.toExponential(0);
}

/**
 * Exponential moving average over a series.
 *
 * Training loss is extremely noisy step to step - each step sees a different
 * random batch, so consecutive values can differ by more than the total
 * improvement over a hundred steps. The raw line is important (it shows
 * instability) but the TREND is what tells you whether the run is working, and
 * the eye cannot extract it from the noise reliably. Drawing both is why the
 * smoothing control exists.
 */
export function smoothSeries(
  points: { x: number; y: number }[],
  factor: number
): { x: number; y: number }[] {
  if (factor <= 0 || points.length === 0) return points;
  // Map a 0..1 control onto a usable alpha range. At factor 1 the line is very
  // heavily smoothed but still tracks the overall shape.
  const alpha = 1 - Math.min(0.97, 0.5 + factor * 0.47);
  const out: { x: number; y: number }[] = [];
  let value = points[0].y;
  for (const point of points) {
    value = alpha * point.y + (1 - alpha) * value;
    out.push({ x: point.x, y: value });
  }
  return out;
}

// ---------------------------------------------------------------------------
// LineChart
// ---------------------------------------------------------------------------

export interface LineChartProps {
  series: Series[];
  height?: number;
  /** Logarithmic y-axis. Values at or below zero are dropped. */
  logScale?: boolean;
  yLabel?: string;
  xLabel?: string;
  /** Forces the y-axis to include zero. */
  includeZero?: boolean;
  /** A labelled horizontal reference line, e.g. the uniform-guessing loss. */
  referenceLines?: { y: number; label: string; color?: string; dashed?: boolean }[];
  /** A labelled vertical marker, e.g. the best-validation step. */
  markers?: { x: number; label: string; color?: string }[];
  formatY?: (value: number) => string;
  formatX?: (value: number) => string;
  emptyMessage?: string;
}

export const LineChart: React.FC<LineChartProps> = ({
  series,
  height = 240,
  logScale = false,
  yLabel,
  xLabel,
  includeZero = false,
  referenceLines = [],
  markers = [],
  formatY = formatTick,
  formatX = formatTick,
  emptyMessage = 'No data yet.',
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  // Width is tracked so the chart is responsive without a resize library.
  const [width, setWidth] = useState(720);
  const containerRef = useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      const next = entries[0]?.contentRect.width;
      if (next && next > 0) setWidth(next);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // Left padding is generous because y labels can be "3.0e-4" wide.
  const padding = { top: 10, right: 12, bottom: xLabel ? 32 : 22, left: 52 };
  const plotWidth = Math.max(10, width - padding.left - padding.right);
  const plotHeight = Math.max(10, height - padding.top - padding.bottom);

  const visible = series.filter((s) => s.points.length > 0);

  const domain = useMemo(() => {
    const allPoints = visible.flatMap((s) => s.points);
    if (allPoints.length === 0) return null;

    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;

    for (const point of allPoints) {
      if (point.x < xMin) xMin = point.x;
      if (point.x > xMax) xMax = point.x;
      // On a log axis, non-positive values have no position at all.
      if (logScale && point.y <= 0) continue;
      if (point.y < yMin) yMin = point.y;
      if (point.y > yMax) yMax = point.y;
    }

    for (const line of referenceLines) {
      if (logScale && line.y <= 0) continue;
      if (line.y < yMin) yMin = line.y;
      if (line.y > yMax) yMax = line.y;
    }

    if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) return null;
    if (includeZero && !logScale) yMin = Math.min(0, yMin);

    // A flat series would otherwise collapse to a zero-height domain.
    if (yMax - yMin < 1e-12) {
      const centre = yMax || 1;
      yMin = centre * 0.9;
      yMax = centre * 1.1;
    } else {
      const headroom = (yMax - yMin) * 0.08;
      yMax += headroom;
      if (!logScale) yMin -= headroom;
      else yMin = Math.max(yMin * 0.9, 1e-12);
    }

    if (xMax - xMin < 1) xMax = xMin + 1;

    return { xMin, xMax, yMin, yMax };
  }, [visible, logScale, includeZero, referenceLines]);

  if (!domain) {
    return (
      <div
        ref={containerRef}
        style={{ height }}
        className="flex items-center justify-center text-xs text-content-tertiary"
      >
        {emptyMessage}
      </div>
    );
  }

  const { xMin, xMax, yMin, yMax } = domain;

  const scaleX = (x: number) => padding.left + ((x - xMin) / (xMax - xMin)) * plotWidth;

  const scaleY = (y: number) => {
    if (logScale) {
      const logMin = Math.log10(Math.max(yMin, 1e-12));
      const logMax = Math.log10(Math.max(yMax, 1e-11));
      const logValue = Math.log10(Math.max(y, 1e-12));
      return padding.top + plotHeight - ((logValue - logMin) / (logMax - logMin)) * plotHeight;
    }
    return padding.top + plotHeight - ((y - yMin) / (yMax - yMin)) * plotHeight;
  };

  // --- Ticks ---------------------------------------------------------------
  const yTicks = useMemo(() => {
    if (logScale) {
      // One tick per power of ten, which is what makes a log axis readable.
      const ticks: number[] = [];
      const start = Math.floor(Math.log10(Math.max(yMin, 1e-12)));
      const end = Math.ceil(Math.log10(Math.max(yMax, 1e-11)));
      for (let exponent = start; exponent <= end; exponent++) {
        const value = 10 ** exponent;
        if (value >= yMin * 0.95 && value <= yMax * 1.05) ticks.push(value);
      }
      // If the whole range sits inside one decade there are no powers of ten
      // to show, so fall back to linear ticks over the actual range.
      if (ticks.length < 2) {
        const step = niceStep(yMax - yMin, 4);
        for (let v = Math.ceil(yMin / step) * step; v <= yMax; v += step) ticks.push(v);
      }
      return ticks;
    }
    const step = niceStep(yMax - yMin, 5);
    const ticks: number[] = [];
    for (let v = Math.ceil(yMin / step) * step; v <= yMax + step * 0.01; v += step) {
      ticks.push(Number(v.toFixed(10)));
    }
    return ticks;
  }, [yMin, yMax, logScale]);

  const xTicks = useMemo(() => {
    const step = niceStep(xMax - xMin, Math.max(2, Math.floor(plotWidth / 90)));
    const ticks: number[] = [];
    for (let v = Math.ceil(xMin / step) * step; v <= xMax; v += step) ticks.push(v);
    // Always show the latest x, so the reader knows where the run is.
    if (ticks.length === 0 || ticks[ticks.length - 1] < xMax * 0.92) ticks.push(xMax);
    return ticks;
  }, [xMin, xMax, plotWidth]);

  // --- Paths ---------------------------------------------------------------
  const buildPath = (points: { x: number; y: number }[]) => {
    const usable = logScale ? points.filter((p) => p.y > 0) : points;
    if (usable.length === 0) return '';
    return usable
      .map((point, index) => `${index === 0 ? 'M' : 'L'}${scaleX(point.x)} ${scaleY(point.y)}`)
      .join(' ');
  };

  // --- Hover ---------------------------------------------------------------
  const hoveredValues = useMemo(() => {
    if (hoverX === null) return null;
    const dataX = xMin + ((hoverX - padding.left) / plotWidth) * (xMax - xMin);
    const entries = visible.map((s) => {
      // Nearest point by x, so the tooltip reads the real sample rather than
      // an interpolated value that never existed.
      let nearest = s.points[0];
      let bestDistance = Infinity;
      for (const point of s.points) {
        const distance = Math.abs(point.x - dataX);
        if (distance < bestDistance) {
          bestDistance = distance;
          nearest = point;
        }
      }
      return { series: s, point: nearest, distance: bestDistance };
    });
    return { dataX, entries };
  }, [hoverX, visible, xMin, xMax, plotWidth, padding.left]);

  const handlePointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = event.clientX - rect.left;
    if (x < padding.left || x > padding.left + plotWidth) {
      setHoverX(null);
      return;
    }
    setHoverX(x);
  };

  const tooltipOnLeft = hoverX !== null && hoverX > padding.left + plotWidth * 0.6;

  return (
    <div ref={containerRef} className="relative w-full">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        role="img"
        aria-label={`${yLabel ?? 'Value'} against ${xLabel ?? 'step'}`}
        onPointerMove={handlePointerMove}
        onPointerLeave={() => setHoverX(null)}
        className="block touch-none"
      >
        {/* Horizontal gridlines and y labels */}
        {yTicks.map((tick) => (
          <g key={`y${tick}`}>
            <line
              x1={padding.left}
              x2={padding.left + plotWidth}
              y1={scaleY(tick)}
              y2={scaleY(tick)}
              stroke="var(--border-subtle)"
              strokeWidth={1}
            />
            <text
              x={padding.left - 7}
              y={scaleY(tick)}
              textAnchor="end"
              dominantBaseline="middle"
              className="num"
              fontSize={9.5}
              fill="var(--text-tertiary)"
            >
              {formatY(tick)}
            </text>
          </g>
        ))}

        {/* X labels */}
        {xTicks.map((tick) => (
          <text
            key={`x${tick}`}
            x={scaleX(tick)}
            y={padding.top + plotHeight + 14}
            textAnchor="middle"
            className="num"
            fontSize={9.5}
            fill="var(--text-tertiary)"
          >
            {formatX(tick)}
          </text>
        ))}

        {/* Axis baseline */}
        <line
          x1={padding.left}
          x2={padding.left + plotWidth}
          y1={padding.top + plotHeight}
          y2={padding.top + plotHeight}
          stroke="var(--border-default)"
          strokeWidth={1}
        />

        {/* Reference lines, e.g. the uniform-guessing loss baseline */}
        {referenceLines.map((line) => {
          const y = scaleY(line.y);
          if (!Number.isFinite(y)) return null;
          return (
            <g key={line.label}>
              <line
                x1={padding.left}
                x2={padding.left + plotWidth}
                y1={y}
                y2={y}
                stroke={line.color ?? 'var(--series-muted)'}
                strokeWidth={1}
                strokeDasharray={line.dashed === false ? undefined : '3 3'}
              />
              {/*
                Anchored to the LEFT edge. Reference lines sit near the top of
                the plot (the uniform-guessing baseline is the highest value on
                a loss chart) and vertical markers label themselves at the top
                too - right-anchoring put the two labels on top of each other
                whenever the best-validation step was late in the run.
              */}
              <text
                x={padding.left + 4}
                y={y - 4}
                textAnchor="start"
                fontSize={9}
                fill={line.color ?? 'var(--series-muted)'}
              >
                {line.label}
              </text>
            </g>
          );
        })}

        {/* Vertical markers, e.g. the best-validation step */}
        {markers.map((marker) => {
          const x = scaleX(marker.x);
          if (x < padding.left || x > padding.left + plotWidth) return null;
          // A marker near the right edge - which is the common case, since the
          // best-validation step is usually recent - would have its label
          // clipped by the viewport. Flip it to the left of the line instead.
          const nearRightEdge = x > padding.left + plotWidth - 40;
          return (
            <g key={`${marker.label}${marker.x}`}>
              <line
                x1={x}
                x2={x}
                y1={padding.top}
                y2={padding.top + plotHeight}
                stroke={marker.color ?? 'var(--series-val)'}
                strokeWidth={1}
                strokeDasharray="2 3"
                opacity={0.8}
              />
              <text
                x={nearRightEdge ? x - 3 : x + 3}
                y={padding.top + 9}
                textAnchor={nearRightEdge ? 'end' : 'start'}
                fontSize={9}
                fill={marker.color ?? 'var(--series-val)'}
              >
                {marker.label}
              </text>
            </g>
          );
        })}

        {/* Series */}
        {visible.map((s) => (
          <g key={s.key}>
            <path
              d={buildPath(s.points)}
              fill="none"
              stroke={s.color}
              strokeWidth={s.width ?? 1.5}
              strokeDasharray={s.dashed ? '4 3' : undefined}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {s.showDots &&
              s.points.length <= 150 &&
              s.points.map((point) => (
                <circle
                  key={`${s.key}${point.x}`}
                  cx={scaleX(point.x)}
                  cy={scaleY(point.y)}
                  r={2}
                  fill={s.color}
                />
              ))}
          </g>
        ))}

        {/* Crosshair and hover markers */}
        {hoverX !== null && hoveredValues && (
          <g>
            <line
              x1={hoverX}
              x2={hoverX}
              y1={padding.top}
              y2={padding.top + plotHeight}
              stroke="var(--border-strong)"
              strokeWidth={1}
            />
            {hoveredValues.entries.map((entry) => {
              const y = scaleY(entry.point.y);
              if (!Number.isFinite(y)) return null;
              return (
                <circle
                  key={entry.series.key}
                  cx={scaleX(entry.point.x)}
                  cy={y}
                  r={3}
                  fill={entry.series.color}
                  stroke="var(--surface-raised)"
                  strokeWidth={1.5}
                />
              );
            })}
          </g>
        )}
      </svg>

      {/* Tooltip, positioned to stay inside the chart */}
      {hoverX !== null && hoveredValues && (
        <div
          className="pointer-events-none absolute top-2 z-10 min-w-[128px] rounded-md border border-edge bg-surface-overlay px-2 py-1.5 shadow-lg"
          style={tooltipOnLeft ? { left: padding.left + 6 } : { right: padding.right + 6 }}
        >
          <div className="num mb-1 text-2xs text-content-tertiary">
            {xLabel ?? 'step'} {formatX(Math.round(hoveredValues.entries[0]?.point.x ?? 0))}
          </div>
          {hoveredValues.entries.map((entry) => (
            <div
              key={entry.series.key}
              className="flex items-center justify-between gap-3 text-2xs"
            >
              <span className="flex items-center gap-1.5 text-content-secondary">
                <span
                  aria-hidden="true"
                  className="h-[2px] w-2.5 rounded-full"
                  style={{ background: entry.series.color }}
                />
                {entry.series.label}
              </span>
              <span className="num text-content-primary">{formatY(entry.point.y)}</span>
            </div>
          ))}
        </div>
      )}

      {xLabel && (
        <div className="label mt-0.5 text-center" style={{ paddingLeft: padding.left }}>
          {xLabel}
        </div>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------

export const Legend: React.FC<{
  items: { label: string; color: string; dashed?: boolean; value?: string }[];
}> = ({ items }) => (
  <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
    {items.map((item) => (
      <span key={item.label} className="flex items-center gap-1.5 text-2xs">
        <svg width="14" height="3" aria-hidden="true" className="shrink-0">
          <line
            x1="0"
            y1="1.5"
            x2="14"
            y2="1.5"
            stroke={item.color}
            strokeWidth="2.5"
            strokeDasharray={item.dashed ? '3 2' : undefined}
            strokeLinecap="round"
          />
        </svg>
        <span className="text-content-secondary">{item.label}</span>
        {item.value && <span className="num text-content-primary">{item.value}</span>}
      </span>
    ))}
  </div>
);

// ---------------------------------------------------------------------------
// Sparkline
// ---------------------------------------------------------------------------

/** A tiny inline trend line, for stat tiles and table rows. */
export const Sparkline: React.FC<{
  values: number[];
  width?: number;
  height?: number;
  color?: string;
}> = ({ values, width = 68, height = 20, color = 'var(--accent)' }) => {
  if (values.length < 2) {
    return <svg width={width} height={height} aria-hidden="true" />;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * (width - 2) + 1;
      const y = height - 1 - ((value - min) / range) * (height - 2);
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(' ');

  return (
    <svg width={width} height={height} aria-hidden="true" className="block">
      <path d={points} fill="none" stroke={color} strokeWidth="1.25" strokeLinejoin="round" />
    </svg>
  );
};

// ---------------------------------------------------------------------------
// Horizontal bar list
// ---------------------------------------------------------------------------

/** Used for vocabulary frequency and the parameter breakdown. */
export const BarList: React.FC<{
  items: { label: string; value: number; display?: string; color?: string; mono?: boolean }[];
  maxItems?: number;
  /**
   * Width of the label column. Defaults to a size that suits single-character
   * vocabulary entries; component names such as "Feed-forward (all layers)"
   * need considerably more before they stop truncating into uselessness.
   */
  labelWidth?: number;
}> = ({ items, maxItems, labelWidth = 72 }) => {
  const shown = maxItems ? items.slice(0, maxItems) : items;
  const max = Math.max(...shown.map((item) => item.value), 1);

  return (
    <div className="flex flex-col gap-1">
      {shown.map((item) => (
        <div key={item.label} className="flex items-center gap-2">
          <span
            style={{ width: labelWidth }}
            className={`shrink-0 truncate text-xs text-content-secondary ${
              item.mono ? 'num' : ''
            }`}
            title={item.label}
          >
            {item.label}
          </span>
          <div className="h-3 min-w-0 flex-1 overflow-hidden rounded-sm bg-surface-input">
            <div
              className="h-full rounded-sm"
              style={{
                width: `${(item.value / max) * 100}%`,
                background: item.color ?? 'var(--accent)',
              }}
            />
          </div>
          <span className="num w-[60px] shrink-0 text-right text-2xs text-content-tertiary">
            {item.display ?? item.value.toLocaleString('en-US')}
          </span>
        </div>
      ))}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Heatmap (attention)
// ---------------------------------------------------------------------------

/** The six-stop attention ramp from the design tokens. */
const HEAT_STOPS = [
  'var(--heat-0)',
  'var(--heat-1)',
  'var(--heat-2)',
  'var(--heat-3)',
  'var(--heat-4)',
  'var(--heat-5)',
];

/**
 * Maps a 0..1 weight to a colour.
 *
 * The square root is deliberate. Attention distributions are extremely
 * peaked - in a trained model one key often takes 0.9 of the mass and the rest
 * share 0.1. On a linear ramp, every cell except the peak renders as the
 * background colour and the map looks empty. Taking the square root lifts the
 * low end so the secondary structure, which is the interesting part, becomes
 * visible.
 */
export function heatColor(weight: number): string {
  const adjusted = Math.sqrt(Math.max(0, Math.min(1, weight)));
  const scaled = adjusted * (HEAT_STOPS.length - 1);
  return HEAT_STOPS[Math.min(HEAT_STOPS.length - 1, Math.round(scaled))];
}

export const Heatmap: React.FC<{
  /** matrix[query][key] in 0..1 */
  matrix: number[][];
  tokens: string[];
  cellSize?: number;
  onHover?: (cell: { query: number; key: number; value: number } | null) => void;
}> = ({ matrix, tokens, cellSize = 20, onHover }) => {
  const [hover, setHover] = useState<{ query: number; key: number } | null>(null);
  const size = matrix.length;
  const labelSpace = 26;

  if (size === 0) return null;

  const width = labelSpace + size * cellSize;
  const height = labelSpace + size * cellSize;

  return (
    <div className="overflow-auto">
      <svg width={width} height={height} className="block">
        {/* Key labels along the top */}
        {tokens.map((token, index) => (
          <text
            key={`k${index}`}
            x={labelSpace + index * cellSize + cellSize / 2}
            y={labelSpace - 8}
            textAnchor="middle"
            fontSize={9}
            className="num"
            fill={hover?.key === index ? 'var(--text-primary)' : 'var(--text-tertiary)'}
          >
            {token}
          </text>
        ))}

        {/* Query labels down the left */}
        {tokens.map((token, index) => (
          <text
            key={`q${index}`}
            x={labelSpace - 7}
            y={labelSpace + index * cellSize + cellSize / 2}
            textAnchor="end"
            dominantBaseline="middle"
            fontSize={9}
            className="num"
            fill={hover?.query === index ? 'var(--text-primary)' : 'var(--text-tertiary)'}
          >
            {token}
          </text>
        ))}

        {/* Cells */}
        {matrix.map((row, query) =>
          row.map((value, key) => {
            const isHovered = hover?.query === query && hover?.key === key;
            return (
              <rect
                key={`${query}-${key}`}
                x={labelSpace + key * cellSize}
                y={labelSpace + query * cellSize}
                width={cellSize - 1}
                height={cellSize - 1}
                fill={heatColor(value)}
                stroke={isHovered ? 'var(--accent)' : 'transparent'}
                strokeWidth={1.5}
                onPointerEnter={() => {
                  setHover({ query, key });
                  onHover?.({ query, key, value });
                }}
                onPointerLeave={() => {
                  setHover(null);
                  onHover?.(null);
                }}
              >
                <title>{`query "${tokens[query]}" -> key "${tokens[key]}": ${(value * 100).toFixed(1)}%`}</title>
              </rect>
            );
          })
        )}
      </svg>
    </div>
  );
};

/** The legend for the heatmap ramp, including its non-linear mapping. */
export const HeatmapLegend: React.FC = () => (
  <div className="flex items-center gap-2">
    <span className="text-2xs text-content-tertiary">0%</span>
    <div className="flex h-2.5 overflow-hidden rounded-sm">
      {Array.from({ length: 24 }, (_, index) => (
        <span
          key={index}
          className="w-2"
          style={{ background: heatColor((index / 23) ** 2) }}
          aria-hidden="true"
        />
      ))}
    </div>
    <span className="text-2xs text-content-tertiary">100%</span>
  </div>
);
