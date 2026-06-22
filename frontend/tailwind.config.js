/** @type {import('tailwindcss').Config} */
import tailwindcssForms from '@tailwindcss/forms';
import tailwindcssTypography from '@tailwindcss/typography';
import tailwindcssAnimate from 'tailwindcss-animate';
import { fontFamily } from 'tailwindcss/defaultTheme';

export default {
  darkMode: ['class'],
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
    './lib/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    container: {
      center: true,
      screens: { '2xl': '1400px' },
    },
    extend: {
      colors: {
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        placeholder: 'hsl(var(--placeholder-foreground))',
        'placeholder-foreground': 'hsl(var(--placeholder-foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
          hover: 'hsl(var(--primary-hover))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
          hover: 'hsl(var(--secondary-hover))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        warning: {
          DEFAULT: 'hsl(var(--warning))',
          foreground: 'hsl(var(--warning-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        'accent-emerald': {
          DEFAULT: 'hsl(var(--accent-emerald))',
          foreground: 'hsl(var(--accent-emerald-foreground))',
        },
        'accent-amber': {
          DEFAULT: 'hsl(var(--accent-amber))',
          foreground: 'hsl(var(--accent-amber-foreground))',
        },
        'accent-red': {
          DEFAULT: 'hsl(var(--accent-red))',
          foreground: 'hsl(var(--accent-red-foreground))',
        },
        'accent-indigo': {
          DEFAULT: 'hsl(var(--accent-indigo))',
          foreground: 'hsl(var(--accent-indigo-foreground))',
        },
        'accent-purple': {
          DEFAULT: 'hsl(var(--accent-purple))',
          foreground: 'hsl(var(--accent-purple-foreground))',
        },
        'accent-orange': {
          DEFAULT: 'hsl(var(--accent-orange))',
          foreground: 'hsl(var(--accent-orange-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        badge: {
          DEFAULT: 'hsl(var(--badge))',
          foreground: 'hsl(var(--badge-foreground))',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      fontFamily: {
        sans: ['var(--font-sans)', ...fontFamily.sans],
        mono: ['var(--font-mono)', ...fontFamily.mono],
      },
      keyframes: {
        overlayShow: {
          from: { opacity: 0 },
          to: { opacity: 1 },
        },
        contentShow: {
          from: {
            opacity: 0,
            transform: 'translate(-50%, -50%) scale(0.95)',
          },
          to: {
            opacity: 1,
            transform: 'translate(-50%, -50%) scale(1)',
          },
        },
      },
      animation: {
        overlayShow: 'overlayShow 200ms cubic-bezier(0.16, 1, 0.3, 1)',
        contentShow: 'contentShow 200ms cubic-bezier(0.16, 1, 0.3, 1)',
      },
    },
  },
  plugins: [tailwindcssAnimate, tailwindcssForms({ strategy: 'class' }), tailwindcssTypography],
};
