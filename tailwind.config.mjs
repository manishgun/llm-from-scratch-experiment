/** @type {import('tailwindcss').Config} */

/*
 * Tailwind is wired to the CSS custom properties in index.css rather than to
 * literal hex values. One palette definition then serves utility classes,
 * inline styles and SVG attributes alike, and switching theme is a single
 * attribute change on <html> with no JS re-render of colours.
 */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          base: 'var(--surface-base)',
          raised: 'var(--surface-raised)',
          overlay: 'var(--surface-overlay)',
          input: 'var(--surface-input)',
          hover: 'var(--surface-hover)',
        },
        edge: {
          subtle: 'var(--border-subtle)',
          DEFAULT: 'var(--border-default)',
          strong: 'var(--border-strong)',
        },
        content: {
          primary: 'var(--text-primary)',
          secondary: 'var(--text-secondary)',
          tertiary: 'var(--text-tertiary)',
          disabled: 'var(--text-disabled)',
        },
        accent: {
          DEFAULT: 'var(--accent)',
          hover: 'var(--accent-hover)',
          subtle: 'var(--accent-subtle)',
          border: 'var(--accent-border)',
          text: 'var(--accent-text)',
        },
        success: { DEFAULT: 'var(--success)', subtle: 'var(--success-subtle)' },
        warning: { DEFAULT: 'var(--warning)', subtle: 'var(--warning-subtle)' },
        danger: { DEFAULT: 'var(--danger)', subtle: 'var(--danger-subtle)' },
        series: {
          train: 'var(--series-train)',
          val: 'var(--series-val)',
          lr: 'var(--series-lr)',
          grad: 'var(--series-grad)',
          throughput: 'var(--series-throughput)',
          muted: 'var(--series-muted)',
        },
      },
      fontFamily: {
        sans: ['var(--font-sans)'],
        mono: ['var(--font-mono)'],
      },
      fontSize: {
        '2xs': ['10px', { lineHeight: '14px' }],
        xs: ['11px', { lineHeight: '16px' }],
        sm: ['12px', { lineHeight: '18px' }],
        base: ['13px', { lineHeight: '20px' }],
        md: ['14px', { lineHeight: '21px' }],
        lg: ['16px', { lineHeight: '24px' }],
        xl: ['19px', { lineHeight: '27px' }],
        '2xl': ['24px', { lineHeight: '32px' }],
        '3xl': ['30px', { lineHeight: '36px' }],
      },
      borderRadius: {
        sm: 'var(--radius-sm)',
        md: 'var(--radius-md)',
        lg: 'var(--radius-lg)',
      },
      spacing: {
        // The app shell's fixed dimensions, named so views can offset by them.
        rail: '52px',
        topbar: '48px',
        statusbar: '26px',
      },
    },
  },
  plugins: [],
};
