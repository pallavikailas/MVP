/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Syne"', 'sans-serif'],
        body: ['"DM Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        night: {
          DEFAULT: '#0a0a0f',
          50: '#0f0f1a',
          100: '#13131f',
          200: '#1a1a2e',
          300: '#22223b',
        },
        lens: {
          DEFAULT: '#7c3aed',
          light: '#a78bfa',
          glow: '#7c3aed33',
        },
        signal: {
          red: '#ef4444',
          amber: '#f59e0b',
          green: '#10b981',
          blue: '#3b82f6',
        },
        bias: {
          critical: '#ef4444',
          high: '#f97316',
          medium: '#eab308',
          low: '#22c55e',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'scan': 'scan 2s linear infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        }
      }
    }
  },
  plugins: [],
}
